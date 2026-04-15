#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow labeling — Apache Spark map-reduce implementation.

Pipeline per giorno:
  1. MAP  : legge tutti i *_Flow.csv del giorno in parallelo, applica una UDF
            row-level che assegna "Label" confrontando timestamp + IP con le
            finestre di attacco lette dall'attack_schedule.yaml (broadcast).
  2. REDUCE: separa benign / attack e scrive i due CSV di output.

Entry point principale: label_day_csvs()
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, time
from pathlib import Path
from typing import List, Optional

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _load_schedule(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_windows(schedule: dict, day: str) -> list:
    """Estrae la lista di finestre di attacco per il giorno richiesto."""
    day_info = schedule.get(day)
    if not day_info:
        return []
    date_str = day_info["date"]          # "2018-03-01"
    windows = []
    for attack in day_info.get("attacks", []):
        windows.append({
            "label":        attack["label"],
            "date":         date_str,
            "start":        attack["start"],   # "HH:MM"
            "end":          attack["end"],
            "attacker_ips": list(attack.get("attacker_ips", [])),
            "victim_ips":   list(attack.get("victim_ips", [])),
        })
    return windows


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_TS_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %I:%M:%S",
    "%d/%m/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
]


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Column name resolution  (CICFlowMeter ha nomi con spazi e varianti)
# ---------------------------------------------------------------------------

def _find_col(columns: list, candidates: list) -> Optional[str]:
    lowered = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def _get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("nnids-labeling")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# UDF factory — chiude solo sul broadcast (serializzabile)
# ---------------------------------------------------------------------------

def _make_label_udf(windows_broadcast):
    """
    Restituisce una Spark UDF StringType.
    La UDF e' la fase MAP: per ogni riga (ts, src_ip, dst_ip) -> Label.
    """
    def _label(ts_raw: str, src_ip: str, dst_ip: str) -> str:
        windows: list = windows_broadcast.value

        dt = _parse_timestamp(ts_raw)
        if dt is None:
            return "Benign"

        flow_time = dt.time()
        flow_date = str(dt.date())  # "YYYY-MM-DD"

        src = str(src_ip).strip() if src_ip else ""
        dst = str(dst_ip).strip() if dst_ip else ""

        for w in windows:
            if flow_date != w["date"]:
                continue

            start_t = datetime.strptime(w["start"], "%H:%M").time()
            end_t   = datetime.strptime(w["end"],   "%H:%M").time()

            if not (start_t <= flow_time <= end_t):
                continue

            attacker = w["attacker_ips"]
            victim   = w["victim_ips"]

            if (src in attacker or dst in victim or
                    src in victim   or dst in attacker):
                return w["label"]

        return "Benign"

    return udf(_label, StringType())


# ---------------------------------------------------------------------------
# Write helper — coalesce(1) -> un solo file CSV finale
# ---------------------------------------------------------------------------

def _write_single_csv(df: DataFrame, dest: Path) -> None:
    """Scrive un DataFrame Spark come singolo file CSV in dest."""
    tmp = Path(tempfile.mkdtemp())
    try:
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp))
        part_files = sorted(tmp.glob("part-*.csv"))
        if part_files:
            shutil.move(str(part_files[0]), str(dest))
        else:
            # nessuna riga: crea file vuoto con header
            dest.write_text("")
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API — MAP-REDUCE principale
# ---------------------------------------------------------------------------

def label_day_csvs(
    csv_dir: Path,
    day: str,
    schedule_yaml,
    out_dir: Path,
    spark: Optional[SparkSession] = None,
) -> None:
    """
    Legge tutti i *_Flow.csv di un giorno, assegna la Label (MAP),
    poi separa benign / attack e scrive i due CSV (REDUCE).

    Parameters
    ----------
    csv_dir       : directory con i *_Flow.csv prodotti da CICFlowMeter
    day           : chiave schedule, es. "Thursday-01-03-2018"
    schedule_yaml : percorso all'attack_schedule.yaml
    out_dir       : directory di output (benign_records.csv, attack_records.csv)
    spark         : SparkSession esistente, o None per crearne una locale
    """
    if schedule_yaml is None:
        from configs.settings import ATTACK_SCHEDULE_YAML
        schedule_yaml = ATTACK_SCHEDULE_YAML

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _spark = spark or _get_spark()

    # --- 1. Carica schedule e broadcast le finestre di attacco ---
    schedule   = _load_schedule(Path(schedule_yaml))
    windows    = _build_windows(schedule, day)
    bc_windows = _spark.sparkContext.broadcast(windows)

    # --- 2. Leggi tutti i Flow CSV del giorno in un unico DataFrame ---
    #        Spark partiziona automaticamente la lettura su tutti i core
    df = (
        _spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("recursiveFileLookup", True)
        .csv(str(csv_dir))
    )

    # Normalizza nomi colonna (rimuove spazi CICFlowMeter)
    df = df.toDF(*[c.strip() for c in df.columns])

    # --- 3. MAP: applica UDF row-level per assegnare "Label" ---
    ts_col  = _find_col(df.columns, ["Timestamp", "timestamp", "Flow Start Time"])
    src_col = _find_col(df.columns, ["Src IP", "Source IP", "src_ip", "SrcIP"])
    dst_col = _find_col(df.columns, ["Dst IP", "Destination IP", "dst_ip", "DstIP"])

    if ts_col is None or src_col is None or dst_col is None:
        df = df.withColumn("Label", udf(lambda *_: "Benign", StringType())())
        print(f"[labeling] WARNING: colonne timestamp/IP non trovate per {day} — tutto Benign")
    else:
        label_udf = _make_label_udf(bc_windows)
        df = df.withColumn("Label", label_udf(col(ts_col), col(src_col), col(dst_col)))

    # Cache in memoria per evitare ricalcolo nel doppio filter
    df.cache()

    # --- 4. REDUCE: separa benign / attack e scrivi i due CSV ---
    benign_df = df.filter(col("Label") == "Benign")
    attack_df = df.filter(col("Label") != "Benign")

    _write_single_csv(benign_df, out_dir / "benign_records.csv")
    _write_single_csv(attack_df, out_dir / "attack_records.csv")

    n_benign = benign_df.count()
    n_attack = attack_df.count()
    print(f"[labeling] {day}: benign={n_benign:,}  attack={n_attack:,}  "
          f"total={n_benign + n_attack:,}")

    df.unpersist()
    bc_windows.unpersist()


# ---------------------------------------------------------------------------
# Pandas shim — compatibilita' backward (usata da test e codice legacy)
# ---------------------------------------------------------------------------

def apply(df, day: Optional[str] = None, schedule_yaml=None):
    """Compatibilita' backward: labeling pandas in-memory."""
    import pandas as pd
    if schedule_yaml is None:
        from configs.settings import ATTACK_SCHEDULE_YAML
        schedule_yaml = ATTACK_SCHEDULE_YAML

    if df.empty:
        return df

    schedule = _load_schedule(Path(schedule_yaml))
    if day:
        windows = _build_windows(schedule, day)
    else:
        windows = [w for d in schedule for w in _build_windows(schedule, d)]

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    ts_col  = _find_col(df.columns, ["Timestamp", "timestamp", "Flow Start Time"])
    src_col = _find_col(df.columns, ["Src IP", "Source IP", "src_ip", "SrcIP"])
    dst_col = _find_col(df.columns, ["Dst IP", "Destination IP", "dst_ip", "DstIP"])

    if ts_col is None or src_col is None or dst_col is None:
        df["Label"] = "Benign"
        return df

    parsed     = df[ts_col].astype(str).apply(_parse_timestamp)
    flow_times = parsed.apply(lambda x: x.time() if x else None)
    flow_dates = parsed.apply(lambda x: str(x.date()) if x else None)

    df["Label"] = "Benign"
    for w in windows:
        start_t = datetime.strptime(w["start"], "%H:%M").time()
        end_t   = datetime.strptime(w["end"],   "%H:%M").time()
        date_mask = flow_dates == w["date"]
        time_mask = (flow_times >= start_t) & (flow_times <= end_t)
        attacker  = set(w["attacker_ips"])
        victim    = set(w["victim_ips"])
        ip_mask   = (
            df[src_col].isin(attacker) | df[dst_col].isin(victim) |
            df[src_col].isin(victim)   | df[dst_col].isin(attacker)
        )
        df.loc[date_mask & time_mask & ip_mask, "Label"] = w["label"]

    return df
