#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous ingestion pipeline: S3 download -> Threat Intelligence IP Injection -> cache."""

from __future__ import annotations

import shutil
import ipaddress
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request

import pandas as pd
import numpy as np

from configs.settings import (
    CSVS_DIR,
    CACHE_DIR,
    DAY_TO_CSV,
    S3_BUCKET,
    S3_REGION,
    S3_PREFIX,
    THREAT_INTEL_FEEDS,
    BENIGN_INTEL_FEEDS,
    BASE_MALICIOUS_IPS,
    BASE_GOOD_PUBLIC_IPS
)

class Ingestion:
    """Download ML CSVs from S3, inject realistic Threat Intelligence IPs for attacks, and format benign traffic."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.csvs_dir = CSVS_DIR
        self.cache_dir = CACHE_DIR
        
        # Threat Intelligence Feeds mapped from settings.py
        self.malicious_feeds = [
            info["url"] 
            for name, info in THREAT_INTEL_FEEDS.items() 
            if info.get("enabled", False)
        ]
        
        # Benign Feeds mapped from settings.py
        self.benign_feeds = [
            info["url"] 
            for name, info in BENIGN_INTEL_FEEDS.items() 
            if info.get("enabled", False)
        ]
        
        self.good_public_ips = []
        self.malicious_ips = []

        for path in [self.csvs_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def run(self, days: Optional[List[str]], force_rerun: bool = False) -> Dict[str, Any]:
        selected_days = days or []
        if not selected_days:
            raise ValueError("No days provided. Pass --days or set DAYS in configs/settings.py")

        self.malicious_ips = self._fetch_feed_ips(self.malicious_feeds, BASE_MALICIOUS_IPS, "malicious")
        self.good_public_ips = self._fetch_feed_ips(self.benign_feeds, BASE_GOOD_PUBLIC_IPS, "benign")

        for day in selected_days:
            if force_rerun:
                self._clear_day_cache(day)
            if not self._is_day_cached(day):
                self._process_day(day)

        frames = []
        for day in selected_days:
            if self._is_day_cached(day):
                df = self._load_day_cache(day)
                if not df.empty:
                    frames.append(df)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return {
            "dataframe": combined,
            "days_processed": selected_days,
            "total_records": len(combined),
        }
        
    def _fetch_feed_ips(self, feeds: List[str], base_pool: List[str], feed_type: str) -> List[str]:
        print(f"[ingestion] Aggregating {feed_type} IPs from {len(feeds)} enabled feeds...")
        
        all_ips = set(base_pool)
        import re
        ipv4_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        
        for feed in feeds:
            try:
                print(f"[ingestion] Fetching {feed_type}: {feed}")
                req = urllib.request.Request(feed, headers={'User-Agent': 'Mozilla/5.0 (Pipeline)'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    lines = response.read().decode('utf-8').splitlines()
                
                for line in lines:
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    found_ips = ipv4_pattern.findall(line)
                    all_ips.update(found_ips)
                    
            except Exception as e:
                print(f"[ingestion] Failed to fetch feed {feed}: {e}")
                
        if not all_ips:
            print(f"[ingestion] Fallback used because all feeds failed for {feed_type}.")
            if feed_type == "malicious":
                all_ips = {"198.51.100.1"}
            else:
                all_ips = {"8.8.8.8"}
            
        print(f"[ingestion] Loaded {len(all_ips)} unique {feed_type} IPs.")
        return list(all_ips)

    def _day_cache_dir(self, day: str) -> Path:
        return self.cache_dir / day

    def _is_day_cached(self, day: str) -> bool:
        day_dir = self._day_cache_dir(day)
        return (day_dir / "benign_records.csv").exists() and (day_dir / "attack_records.csv").exists()

    def _clear_day_cache(self, day: str) -> None:
        shutil.rmtree(self._day_cache_dir(day), ignore_errors=True)
        csv_path = self.csvs_dir / day / DAY_TO_CSV.get(day, "")
        if csv_path.exists():
            csv_path.unlink()

    def _download_and_extract_dataset(self) -> None:
        """Download and extract the full dataset zip locally if it doesn't already exist.
        
        Analytical insight: Shifting from a day-by-day S3 download to a single bulk zip download 
        (and local extraction) provides PySpark with immediate, zero-latency access to the raw data.
        Since PySpark (in local mode) optimizes disk I/O by streaming chunks into memory directly from
        disk, having all CSV files locally unzipped minimizes HTTP retrieval bottlenecks.
        """
        import zipfile
        import sys
        
        target_dir = self.base_dir / "CSECICIDS2018_improved"
        # Check if the folder is populated (meaning extraction was already manually completed by the user)
        if target_dir.exists() and any(target_dir.iterdir()):
            print("[ingestion] Dataset already extracted. Skipping download.")
            return

        zip_path = self.base_dir / "CSECICIDS2018_improved.zip"
        if not zip_path.exists():
            print("[ingestion] Downloading dataset from distrinet-research (approx 5.3GB)...")
            from configs.settings import DATASET_URL
            def report(count, blockSize, totalSize):
                percent = int(count * blockSize * 100 / totalSize)
                sys.stdout.write(f"\rDownloading... {percent}%")
                sys.stdout.flush()
            urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=report)
            print("\n[ingestion] Download complete.")
            
        print("[ingestion] Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.base_dir)
        print("[ingestion] Extraction complete.")

    def _generate_mixed_pool(self, size: int) -> List[str]:
        pool = list(self.good_public_ips)
        for _ in range(size):
            subnet = np.random.choice([10, 172, 192])
            if subnet == 10:
                pool.append(f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}")
            elif subnet == 172:
                pool.append(f"172.{np.random.randint(16, 32)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}")
            else:
                pool.append(f"192.168.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}")
        return pool

    def _replace_ips(self, df: pd.DataFrame, is_malicious: pd.Series) -> pd.DataFrame:
        """
        Replace logic (completely randomized):
        1. Attack Src IP -> Threat Intel Feed IP
        2. Attack Dst IP -> Random (Private OR Good Public)
        3. Benign Src IP -> Random (Private OR Good Public)
        4. Benign Dst IP -> Random (Private OR Good Public)
        """
        if "Src IP" not in df.columns:
            df["Src IP"] = ""
        if "Dst IP" not in df.columns:
            df["Dst IP"] = ""
            
        print("[ingestion] Completely randomizing IP addresses as requested...")
        
        attack_mask = is_malicious.to_numpy()
        benign_mask = ~attack_mask
        
        n_attack = attack_mask.sum()
        n_benign = benign_mask.sum()
        
        # Creiamo un pool misto di IP privati generati randomicamente e IP pubblici approvati
        mixed_pool = self._generate_mixed_pool(min(10000, len(df)))
        
        # --------- Malicious Replacements ---------
        if n_attack > 0 and self.malicious_ips:
            df.loc[attack_mask, "Src IP"] = np.random.choice(self.malicious_ips, size=n_attack)
            df.loc[attack_mask, "Dst IP"] = np.random.choice(mixed_pool, size=n_attack)
            
        # --------- Benign Replacements ---------
        if n_benign > 0:
            df.loc[benign_mask, "Src IP"] = np.random.choice(mixed_pool, size=n_benign)
            df.loc[benign_mask, "Dst IP"] = np.random.choice(mixed_pool, size=n_benign)
            
        return df

    def _process_day(self, day: str) -> None:
        print(f"[ingestion] Processing day: {day}")
        csv_path = self._download_csv(day)
        if csv_path is None:
            print(f"[ingestion] Skip day {day}: CSV unavailable")
            return

        print(f"[ingestion] Reading CSV: {csv_path.name}")
        df = pd.read_csv(csv_path, low_memory=False)
        
        label_col = "Label" if "Label" in df.columns else None
        if not label_col:
            print("[ingestion] Warning: No 'Label' column found. All records treated as benign.")
            is_malicious = pd.Series(False, index=df.index)
        else:
            is_malicious = df[label_col].str.lower() != "benign"
            
        df = self._replace_ips(df, is_malicious)

        cols = df.columns.tolist()
        if "Src IP" in cols and "Dst IP" in cols:
            cols.remove("Src IP")
            cols.remove("Dst IP")
            cols = ["Src IP", "Dst IP"] + cols
            df = df[cols]

        benign_df = df[~is_malicious]
        attack_df = df[is_malicious]
        
        day_cache = self._day_cache_dir(day)
        day_cache.mkdir(parents=True, exist_ok=True)
        
        benign_df.to_csv(day_cache / "benign_records.csv", index=False)
        attack_df.to_csv(day_cache / "attack_records.csv", index=False)
            
        csv_path.unlink(missing_ok=True)
        print(f"[ingestion] Cleaned downloaded file: {csv_path.name}\n")

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
