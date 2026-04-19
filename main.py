#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry-point for the minimal autonomous ingestion + preprocessing pipeline."""
import argparse
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from configs.settings import (
    DAYS,
    SAMPLE_SIZE,
    CACHE_ENABLED,
    CACHE_DIR,
)
from core.ingestion import Ingestion


def run(args):
    ing = Ingestion(base_dir=Path(__file__).resolve().parent)
    raw = ing.run(days=args.days, force_rerun=args.force)
    
    print("\n[main] Ingestion Phase execution completed.")
    print(f"[main] Days processed: {raw.get('days_processed')}")
    
    # Phase 2: PySpark Preprocessing
    from pyspark.sql import SparkSession
    spark_tmp = Path(__file__).resolve().parent / "spark_tmp"
    spark_tmp.mkdir(exist_ok=True)
    spark = (
        SparkSession.builder
        .appName("CICIDS2018_Pipeline")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.local.dir", str(spark_tmp))
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    
    csv_paths = [str(Path(CACHE_DIR) / day / "unified_records.parquet") for day in raw.get('days_processed', [])]
    
    # Read the unified unified parquet chunks dynamically
    full_df = spark.read.parquet(*csv_paths)
    
    # -------- Dataset Statistics Report (Cross-Tabulation) --------
    from core.utils import generate_crosstab_report, clean_temp
    from core.visuals import plot_dataset_statistics
    stats_file = ing.base_dir / "data" / "dataset_statistics.csv"
    visuals_dir = ing.base_dir / "data" / "visuals"
    generate_crosstab_report(full_df, stats_file)
    plot_dataset_statistics(stats_file, visuals_dir)
    # -------------------------------------------------------------
    
    from core.dataset_loader import get_dataset
    from configs.settings import ML_CLASS_STRATEGY, TARGET_BENIGN_RATIO
    from core.preprocessing import preprocess_spark
    
    print(f"\n[main] Applying Machine Learning Loader Strategy: {ML_CLASS_STRATEGY}")
    
    final_cache_target = Path(CACHE_DIR) / f"final_preprocessed_{ML_CLASS_STRATEGY}.parquet"
    
    # Lazily evaluate the specialized ML DataFrame request
    ml_df = get_dataset(full_df, strategy=ML_CLASS_STRATEGY, target_benign_ratio=TARGET_BENIGN_RATIO)
    
    preproc_df = preprocess_spark(
        ml_df,
        sample_size=args.sample,
        cache=args.cache,
        cache_dir=final_cache_target,
    )
    
    print("\n[main] End-to-End Dynamic Pipeline execution completed.")
    print(f"[main] Fully scaled and preprocessed dataset saved at: {final_cache_target}")
    
    clean_temp(ing.base_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", nargs="+", default=DAYS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    p.add_argument("--cache", action="store_true", default=CACHE_ENABLED)
    p.add_argument("--no-cache", dest="cache", action="store_false")
    run(p.parse_args())
