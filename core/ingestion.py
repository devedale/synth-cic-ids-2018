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
    CACHE_DIR,
    THREAT_INTEL_FEEDS,
    BENIGN_INTEL_FEEDS,
    BASE_MALICIOUS_IPS,
    BASE_GOOD_PUBLIC_IPS
)

class Ingestion:
    """Download ML CSVs from S3, inject realistic Threat Intelligence IPs for attacks, and format benign traffic."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
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

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, days: Optional[List[str]], force_rerun: bool = False) -> Dict[str, Any]:
        """Execute the ingestion pipeline."""
        selected_days = days or []
        if not selected_days:
            raise ValueError("No days provided. Pass --days or set DAYS in configs/settings.py")

        # 1. Download and extract local Dataset
        self._download_and_extract_dataset()

        self.malicious_ips = self._fetch_feed_ips(self.malicious_feeds, BASE_MALICIOUS_IPS, "malicious", force_redownload=force_rerun)
        self.good_public_ips = self._fetch_feed_ips(self.benign_feeds, BASE_GOOD_PUBLIC_IPS, "benign", force_redownload=force_rerun)

        for day in selected_days:
            if force_rerun:
                self._clear_day_cache(day)
            if not self._is_day_cached(day):
                self._process_day_pyspark(day)

        return {
            "days_processed": selected_days,
            "status": "Spark partitions written to preprocessed_cache successfully"
        }
        
    def _fetch_feed_ips(self, feeds: List[str], base_pool: List[str], feed_type: str, force_redownload: bool = False) -> List[str]:
        import json
        cache_file = self.base_dir / "data" / "intel_cache" / f"{feed_type}_ips.json"
        
        if not force_redownload and cache_file.exists():
            print(f"[ingestion] Loading {feed_type} IPs from local cache (reproducibility mode)...")
            with open(cache_file, "r") as f:
                return json.load(f)
                
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
                
        if len(all_ips) == len(base_pool):
            print(f"[ingestion] Fallback used because all feeds failed for {feed_type}.")
            if feed_type == "malicious" and not base_pool:
                all_ips = {"198.51.100.1"}
            elif not base_pool:
                all_ips = {"8.8.8.8"}
            
        print(f"[ingestion] Fetched {len(all_ips)} unique {feed_type} IPs. Saving to local cache.")
        
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(list(all_ips), f)
            
        return list(all_ips)

    def _day_cache_dir(self, day: str) -> Path:
        return self.cache_dir / day

    def _is_day_cached(self, day: str) -> bool:
        day_dir = self._day_cache_dir(day)
        return (day_dir / "unified_records.parquet").exists()

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

    def _process_day_pyspark(self, day: str) -> None:
        """Read 4GB CSV using PySpark MapReduce, randomize IP addresses, and write to Parquet cache."""
        import random
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.sql.types import StringType
        
        print(f"[ingestion] Processing day with PySpark: {day}")
        csv_path = self.base_dir / "CSECICIDS2018_improved" / f"{day}.csv"
        
        if not csv_path.exists():
            print(f"[ingestion] Skip day {day}: {csv_path.name} unavailable")
            return

        spark = SparkSession.builder \
            .appName("CICIDS2018_Ingestion") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
            
        print(f"[ingestion] Loaded file in Spark: {csv_path.name}")
        df = spark.read.csv(str(csv_path), header=True, inferSchema=False)
        
        # IP Feeds logic
        mixed_pool = self._generate_mixed_pool(min(10000, df.count() if df.count() > 0 else 1000))
        malicious_pool = self.malicious_ips if self.malicious_ips else ["198.51.100.1"]
        
        def assign_malicious_src(): return random.choice(malicious_pool)
        def assign_random_ip(): return random.choice(mixed_pool)
        
        assign_malicious_src_udf = F.udf(assign_malicious_src, StringType())
        assign_random_ip_udf = F.udf(assign_random_ip, StringType())
        
        # Make sure the IP columns exist
        if "Src IP" not in df.columns:
            df = df.withColumn("Src IP", F.lit(""))
        if "Dst IP" not in df.columns:
            df = df.withColumn("Dst IP", F.lit(""))
            
        label_col = "Label" if "Label" in df.columns else None
        if not label_col:
            df_processed = df.withColumn("Src IP", assign_random_ip_udf())\
                             .withColumn("Dst IP", assign_random_ip_udf())
        else:
            is_attack_cond = (F.lower(F.col(label_col)) != "benign")
            
            df_processed = df.withColumn("Src IP", F.when(is_attack_cond, assign_malicious_src_udf()).otherwise(assign_random_ip_udf()))\
                             .withColumn("Dst IP", assign_random_ip_udf())
        
        # Save output partitions
        day_cache = self._day_cache_dir(day)
        day_cache.mkdir(parents=True, exist_ok=True)
        
        # Tag origin day for ML statistics loader tracking before saving RAW block
        df_processed = df_processed.withColumn("_source_day", F.lit(day))
        
        df_processed.repartition(10).write.mode("overwrite").parquet(str(day_cache / "unified_records.parquet"))
            
        print(f"[ingestion] Saved unified Spark Parquet partition for day {day}\n")

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
