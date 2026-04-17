#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry-point for the minimal autonomous ingestion + preprocessing pipeline."""
import argparse
from pathlib import Path
from configs.settings import (
    DAYS,
    SAMPLE_SIZE,
    CACHE_ENABLED,
    CACHE_DIR,
)
from core.ingestion import Ingestion
from core.preprocessing import preprocess, clean_temp


def run(args):
    ing = Ingestion(base_dir=Path(__file__).resolve().parent)
    raw = ing.run(days=args.days, force_rerun=args.force)
    print("\n[main] Ingestion Pipeline execution completed.")
    print(f"[main] Status: {raw.get('status')}")
    print(f"[main] Days processed: {raw.get('days_processed')}")
    print(f"[main] Cache output path: {Path(CACHE_DIR)}\n")
    print("[main] Note: In-memory Pandas preprocessing (StandardScaler) is currently skipped.")
    print("[main] To preprocess the 40GB Parquet dataset without OOM, the preprocessing.py module should also be migrated to PySpark.\n")
    
    clean_temp(ing.base_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", nargs="+", default=DAYS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    p.add_argument("--cache", action="store_true", default=CACHE_ENABLED)
    p.add_argument("--no-cache", dest="cache", action="store_false")
    run(p.parse_args())
