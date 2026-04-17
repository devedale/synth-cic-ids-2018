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
    
    print("\n[main] Ingestion Phase execution completed.")
    print(f"[main] Days processed: {raw.get('days_processed')}")
    
    # Phase 2: PySpark Preprocessing
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("CICIDS2018_Pipeline").getOrCreate()
    
    csv_paths = [str(Path(CACHE_DIR) / day / "unified_records.parquet") for day in raw.get('days_processed', [])]
    
    # Read the unified unified parquet chunks dynamically
    full_df = spark.read.parquet(*csv_paths)
    
    # -------- Dataset Statistics Report (Cross-Tabulation) --------
    stats_file = ing.base_dir / "data" / "dataset_statistics.csv"
    if not stats_file.exists() and full_df.count() > 0:
        print("\n[main] Generating Cross-Tabulated Dataset Statistics Report (First run)...")
        # Generate the crosstab using PySpark (fast distributed action)
        crosstab_df = full_df.crosstab("Label", "_source_day")
        
        # Convert to Pandas for computing margins and clean printing (matrix is extremely small, ~15x10)
        crosstab_pd = crosstab_df.toPandas()
        if not crosstab_pd.empty:
            crosstab_pd.set_index("Label__source_day", inplace=True)
            crosstab_pd.index.name = "Label"
            
            # Calculate right margin (Row Totals)
            crosstab_pd["Total"] = crosstab_pd.sum(axis=1)
            
            # Calculate bottom margin (Column Totals)
            crosstab_pd.loc["Total"] = crosstab_pd.sum(axis=0)
            
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            crosstab_pd.to_csv(stats_file)
            print("\n[main] --- Dataset Distribution Statistics ---")
            print(crosstab_pd.to_string())
            print(f"----------------------------------------------\n[main] Saved to {stats_file}\n")
    # -------------------------------------------------------------
    
    from core.dataset_loader import get_dataset
    from configs.settings import ML_CLASS_STRATEGY, TARGET_BENIGN_RATIO
    from core.preprocessing import preprocess_spark
    
    print(f"\n[main] Applying Machine Learning Loader Strategy: {ML_CLASS_STRATEGY}")
    
    final_cache_target = Path(CACHE_DIR) / f"final_preprocessed_{ML_CLASS_STRATEGY}.parquet"
    
    # Lazily evaluate the specialized ML DataFrame request
    ml_df = get_dataset(spark, str(Path(CACHE_DIR)), strategy=ML_CLASS_STRATEGY, target_benign_ratio=TARGET_BENIGN_RATIO)
    
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
