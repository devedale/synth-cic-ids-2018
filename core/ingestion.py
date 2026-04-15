#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous ingestion pipeline: S3 download -> CICFlowMeter -> labeling -> cache."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from configs.settings import (
    ATTACK_SCHEDULE_YAML,
    ARCHIVES_DIR,
    CACHE_DIR,
    CICFLOWMETER_ROOT,
    DAY_TO_ARCHIVE,
    FLOW_CSV_DIR,
    INGEST_SAMPLE_SIZE,
    JAVA8_HOME,
    JAVA_XMX,
    PCAP_DIR,
    S3_BUCKET,
    S3_PREFIX,
    S3_REGION,
)
from core.labeling import label_day_csvs


class Ingestion:
    """End-to-end ingestion and day-wise cache builder."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.archives_dir = ARCHIVES_DIR
        self.pcap_dir = PCAP_DIR
        self.flow_csv_dir = FLOW_CSV_DIR
        self.cache_dir = CACHE_DIR
        self.cic_root = CICFLOWMETER_ROOT
        self.schedule_yaml = ATTACK_SCHEDULE_YAML

        for path in [self.archives_dir, self.pcap_dir, self.flow_csv_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def run(self, days: Optional[List[str]], force_rerun: bool = False) -> Dict[str, Any]:
        selected_days = days or []
        if not selected_days:
            raise ValueError("No days provided. Pass --days or set DAYS in configs/settings.py")

        for day in selected_days:
            if force_rerun:
                self._clear_day_cache(day)
            if not self._is_day_cached(day):
                self._process_day(day)

        frames = [self._load_day_cache(day) for day in selected_days if self._is_day_cached(day)]
        frames = [df for df in frames if not df.empty]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return {
            "dataframe": combined,
            "days_processed": selected_days,
            "total_records": len(combined),
        }

    def _day_cache_dir(self, day: str) -> Path:
        return self.cache_dir / day

    def _is_day_cached(self, day: str) -> bool:
        day_dir = self._day_cache_dir(day)
        return (day_dir / "benign_records.csv").exists() and (day_dir / "attack_records.csv").exists()

    def _clear_day_cache(self, day: str) -> None:
        shutil.rmtree(self._day_cache_dir(day), ignore_errors=True)
        shutil.rmtree(self.pcap_dir / day, ignore_errors=True)
        shutil.rmtree(self.flow_csv_dir / day, ignore_errors=True)
        shutil.rmtree(self.archives_dir / day, ignore_errors=True)

    def _s3_client(self):
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        return boto3.client(
            "s3",
            region_name=S3_REGION,
            config=Config(signature_version=UNSIGNED),
        )

    def _download_archive(self, day: str) -> Optional[Path]:
        archive_name = DAY_TO_ARCHIVE.get(day)
        if archive_name is None:
            print(f"[ingestion] No archive mapping found for day '{day}'")
            return None

        s3_key = f"{S3_PREFIX}{day}/{archive_name}"
        day_archive_dir = self.archives_dir / day
        day_archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = day_archive_dir / archive_name

        if archive_path.exists():
            return archive_path

        print(f"[ingestion] Downloading s3://{S3_BUCKET}/{s3_key}")
        s3 = self._s3_client()

        try:
            s3.download_file(S3_BUCKET, s3_key, str(archive_path))
        except Exception as exc:
            print(f"[ingestion] Download failed for {day}: {exc}")
            archive_path.unlink(missing_ok=True)
            return None

        return archive_path

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = archive_path.suffix.lower()

        if suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)
            return

        if suffix == ".rar":
            candidates = [
                ["unrar", "x", "-o+", "-kb", str(archive_path), str(target_dir) + "/"],
                ["7z", "x", str(archive_path), f"-o{target_dir}", "-y"],
                ["unar", "-o", str(target_dir), str(archive_path)],
            ]
            for cmd in candidates:
                if shutil.which(cmd[0]):
                    result = subprocess.run(cmd)
                    if result.returncode == 0 or any(target_dir.rglob("*")):
                        return
            raise RuntimeError(
                "RAR extraction failed. Install one of: unrar, p7zip-full, unar"
            )

        raise ValueError(f"Unsupported archive format: {suffix}")

    def _make_gradle_env(self) -> dict:
        env = os.environ.copy()
        jnet_lib = self.cic_root / "jnetpcap" / "linux" / "jnetpcap-1.4.r1425"

        java_home = JAVA8_HOME if os.path.isdir(JAVA8_HOME) else env.get("JAVA_HOME", "")
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{java_home}/bin:{java_home}/jre/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(
            filter(None, [str(jnet_lib), env.get("LD_LIBRARY_PATH", "")])
        )
        env["JAVA_TOOL_OPTIONS"] = f"-Djava.library.path={jnet_lib} -Xmx{JAVA_XMX}"
        env["GRADLE_OPTS"] = f"-Dorg.gradle.daemon=false -Djava.library.path={jnet_lib}"
        return env

    # Magic-byte signatures that identify valid PCAP / pcap-ng files.
    _PCAP_MAGIC = (
        b"\xd4\xc3\xb2\xa1",  # pcap little-endian
        b"\xa1\xb2\xc3\xd4",  # pcap big-endian
        b"\xa1\xb2\x3c\x4d",  # pcap big-endian nanosecond
        b"\x4d\x3c\xb2\xa1",  # pcap little-endian nanosecond
        b"\x0a\x0d\x0d\x0a",  # pcap-ng
    )

    @staticmethod
    def _is_valid_pcap(path: Path) -> bool:
        """Return True only when *path* starts with a recognised PCAP magic."""
        try:
            with open(path, "rb") as fh:
                header = fh.read(4)
            return header in Ingestion._PCAP_MAGIC
        except OSError:
            return False

    @staticmethod
    def _sanitize_pcap_name(pcap_path: Path) -> Path:
        """
        Return a path whose stem is safe for Gradle / Java:
        - strips leading/trailing whitespace from the stem
        - replaces any character that is not alphanumeric, '-', or '_' with '_'
        - always ensures the extension is '.pcap'
        The file is renamed on disk; the new Path is returned.
        """
        import re as _re
        clean_stem = _re.sub(r"[^A-Za-z0-9\-_]", "_", pcap_path.stem.strip())
        # collapse consecutive underscores introduced by the substitution
        clean_stem = _re.sub(r"_+", "_", clean_stem).strip("_") or "pcap"
        safe_path = pcap_path.parent / f"{clean_stem}.pcap"
        if safe_path != pcap_path:
            # avoid clobbering an existing file with the same safe name
            if safe_path.exists():
                idx = 1
                while safe_path.exists():
                    safe_path = pcap_path.parent / f"{clean_stem}_{idx}.pcap"
                    idx += 1
            pcap_path.rename(safe_path)
        return safe_path

    def _run_cicflowmeter(self, pcap_day_dir: Path, csv_day_dir: Path) -> None:
        gradlew = self.cic_root / "gradlew"
        build_gradle = self.cic_root / "build.gradle"

        if not gradlew.exists():
            raise FileNotFoundError(f"gradlew not found: {gradlew}")
        gradlew.chmod(0o755)

        candidates = sorted(
            p for p in pcap_day_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
        )
        if not candidates:
            raise FileNotFoundError(f"No files found in {pcap_day_dir}")

        original = build_gradle.read_text()
        env = self._make_gradle_env()
        processed = 0
        skipped = 0

        try:
            for pcap_path in candidates:
                # ── 1. Validate: skip files that are not real PCAPs ────────────
                if not self._is_valid_pcap(pcap_path):
                    print(
                        f"[ingestion] skip (not a valid PCAP): {pcap_path.name}"
                    )
                    skipped += 1
                    continue

                # ── 2. Sanitize filename (spaces, dots, special chars) ─────────
                pcap_path = self._sanitize_pcap_name(pcap_path)

                out_dir = csv_day_dir / pcap_path.stem
                out_dir.mkdir(parents=True, exist_ok=True)

                # ── 3. Patch the exeCMD args line ──────────────────────────────
                args_line = f'args = ["{pcap_path}", "{out_dir}"]'
                patched, replacements = re.subn(
                    r"(?m)^\s*//?\s*args\s*=\s*\[[^\]]*\]",
                    f"    {args_line}",
                    original,
                    count=1,
                )
                if replacements != 1:
                    raise RuntimeError("Could not patch exeCMD args in CICFlowMeter build.gradle")
                build_gradle.write_text(patched)

                # ── 4. Run CICFlowMeter ────────────────────────────────────────
                proc = subprocess.run(
                    [str(gradlew), "--no-daemon", "exeCMD"],
                    cwd=str(self.cic_root),
                    env=env,
                )
                if proc.returncode != 0:
                    print(
                        f"[ingestion] CICFlowMeter failed (exit {proc.returncode}) "
                        f"for {pcap_path.name} — skipping"
                    )
                    skipped += 1
                    continue

                processed += 1
        finally:
            build_gradle.write_text(original)

        if processed == 0:
            raise RuntimeError(
                f"CICFlowMeter produced no output for {pcap_day_dir} "
                f"({skipped} file(s) skipped)"
            )

    def _merge_and_label(self, day: str, csv_day_dir: Path) -> None:
        flow_csvs = list(csv_day_dir.rglob("*_Flow.csv"))
        if not flow_csvs:
            raise FileNotFoundError(f"No *_Flow.csv generated for {day}")

        day_cache = self._day_cache_dir(day)
        day_cache.mkdir(parents=True, exist_ok=True)

        # Spark legge tutti i CSV in parallelo (MAP), poi scrive benign/attack (REDUCE)
        label_day_csvs(
            csv_dir=csv_day_dir,
            day=day,
            schedule_yaml=self.schedule_yaml,
            out_dir=day_cache,
        )

    def _load_day_cache(self, day: str) -> pd.DataFrame:
        day_dir = self._day_cache_dir(day)
        chunks = []
        for name in ("benign_records.csv", "attack_records.csv"):
            path = day_dir / name
            if path.exists():
                chunk = pd.read_csv(path, low_memory=False)
                chunk["_source_day"] = day
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def _process_day(self, day: str) -> None:
        print(f"[ingestion] Processing day: {day}")
        archive = self._download_archive(day)
        if archive is None:
            print(f"[ingestion] Skip day {day}: archive unavailable")
            return

        pcap_day_dir = self.pcap_dir / day
        csv_day_dir = self.flow_csv_dir / day

        self._extract_archive(archive, pcap_day_dir)
        archive.unlink(missing_ok=True)  # free disk space immediately

        self._run_cicflowmeter(pcap_day_dir, csv_day_dir)
        shutil.rmtree(pcap_day_dir, ignore_errors=True)  # free disk space

        self._merge_and_label(day, csv_day_dir)
        shutil.rmtree(csv_day_dir, ignore_errors=True)  # keep only final cache
