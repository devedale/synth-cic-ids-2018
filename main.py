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
    raw_df = raw["dataframe"]
    preproc_df = preprocess(
        raw_df,
        sample_size=args.sample,
        cache=args.cache,
        cache_dir=CACHE_DIR,
    )
    clean_temp(ing.base_dir)
    print("Pipeline execution completed.")
    print("Total records extracted:", len(raw_df))
    print("Preprocessed records formulated:", len(preproc_df))
    print("Cache output path:", Path(CACHE_DIR))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", nargs="+", default=DAYS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    p.add_argument("--cache", action="store_true", default=CACHE_ENABLED)
    p.add_argument("--no-cache", dest="cache", action="store_false")
    run(p.parse_args())
