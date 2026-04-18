#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal preprocessing stage for the CIC-IDS-2018 Data Generator."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_spark(
    df,
    sample_size: Optional[int] = None,
    cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
):
    """Apply PySpark out-of-core preprocessing over 40GB parquets and optionally save."""
    from core.ip2vec import compute_ip2vec_embeddings
    from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, PCA
    from configs.settings import NET_ENTITIES, PCA_COMPONENTS, IP2VEC_SENTENCE
    
    if sample_size:
        total_rows = df.count()
        if total_rows > sample_size:
            fraction = min(1.0, sample_size / total_rows)
            df = df.sample(withReplacement=False, fraction=fraction, seed=42)

    # Note: precise single-value dropping is expensive in distributed systems.
    # It will be omitted for speed, or handled dynamically inside ML models.
    df = df.fillna(0)

    if "Label" in df.columns:
        indexer = StringIndexer(inputCol="Label", outputCol="Label_Encoded", handleInvalid="keep")
        df = indexer.fit(df).transform(df).drop("Label").withColumnRenamed("Label_Encoded", "Label")

    # Enforce isolation of Network Entities so they survive for IP2Vec Embeddings
    # Dst Port might be 'Dst Port' in CICIDS datasets, we ensure a clean exclusion list.
    num_cols = [c for c, t in df.dtypes if t in ["int", "double", "float", "bigint"] and c != "Label"]
    num_cols = [c for c in num_cols if c not in NET_ENTITIES]
    
    if len(num_cols) > 0:
        assembler = VectorAssembler(inputCols=num_cols, outputCol="features_raw", handleInvalid="skip")
        df = assembler.transform(df)
        
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df).drop("features_raw")
        
        # Parallel PCA Component extraction for Neural dimensionality tests
        print(f"[preprocessing] Extracting {PCA_COMPONENTS} PCA structural features...")
        pca = PCA(k=PCA_COMPONENTS, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df)
        df = pca_model.transform(df)
        
    # Execute Distributed Skip-gram Embeddings Generation
    # NET_ENTITIES are safely decoupled, we now compute the IP2Vec arrays natively.
    df = compute_ip2vec_embeddings(df, context_columns=IP2VEC_SENTENCE)

    if cache:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "preprocessed_cache" / "final_preprocessed.parquet"
        
        cache_root = Path(cache_dir).parent
        cache_root.mkdir(parents=True, exist_ok=True)

        print("[preprocessing] Writing unified preprocessed parquet to disk...")
        df.write.mode("overwrite").parquet(str(cache_dir))
        print(f"[preprocessing] Unified Parquet saved at: {str(cache_dir)}")

    return df

