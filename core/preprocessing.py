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
    from configs.settings import NET_ENTITIES, PCA_COMPONENTS, IP2VEC_SENTENCE, RANDOM_SEED, USE_IP2VEC, USE_PCA, MODELS_DIR
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if sample_size:
        total_rows = df.count()
        if total_rows > sample_size:
            fraction = min(1.0, sample_size / total_rows)
            df = df.sample(withReplacement=False, fraction=fraction, seed=RANDOM_SEED)

    # Note: precise single-value dropping is expensive in distributed systems.
    # It will be omitted for speed, or handled dynamically inside ML models.
    df = df.fillna(0)

    if "Label" in df.columns:
        indexer = StringIndexer(inputCol="Label", outputCol="Label_Encoded", handleInvalid="keep")
        indexer_model = indexer.fit(df)
        indexer_model.write().overwrite().save(str(MODELS_DIR / "string_indexer_model"))
        df = indexer_model.transform(df).drop("Label").withColumnRenamed("Label_Encoded", "Label")

    # Enforce isolation of Network Entities so they survive for IP2Vec Embeddings
    # Dst Port might be 'Dst Port' in CICIDS datasets, we ensure a clean exclusion list.
    num_cols = [c for c, t in df.dtypes if t in ["int", "double", "float", "bigint"] and c != "Label"]
    num_cols = [c for c in num_cols if c not in NET_ENTITIES]
    
    if len(num_cols) > 0:
        assembler = VectorAssembler(inputCols=num_cols, outputCol="features_raw", handleInvalid="skip")
        df = assembler.transform(df)
        
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        scaler_model = scaler.fit(df)
        scaler_model.write().overwrite().save(str(MODELS_DIR / "scaler_model"))
        df = scaler_model.transform(df).drop("features_raw")
        
        if USE_PCA:
            # Parallel PCA Component extraction for Neural dimensionality tests
            print(f"[preprocessing] Extracting {PCA_COMPONENTS} PCA structural features...")
            pca = PCA(k=PCA_COMPONENTS, inputCol="features", outputCol="pca_features")
            pca_model = pca.fit(df)
            pca_model.write().overwrite().save(str(MODELS_DIR / "pca_model"))
            df = pca_model.transform(df)
            
            # --- Generate PCA Visuals ---
            from core.visuals import plot_pca_variance
            output_dir = MODELS_DIR.parent / "data" / "visuals"
            explained_var_arr = pca_model.explainedVariance.toArray()
            explained_var = explained_var_arr.tolist()
            plot_pca_variance(explained_var, output_dir)
            
            from configs.settings import PCA_FEATURE_SELECTION, PCA_TARGET_FEATURES
            if PCA_FEATURE_SELECTION:
                import numpy as np
                from pyspark.ml.feature import VectorSlicer
                
                num_features = len(num_cols)
                pc_matrix = np.abs(pca_model.pc.toArray())
                
                # Algorithm: Global Feature Importance Scoring
                # Weight the absolute feature loadings by the variance uniquely explained by each PCA component
                weighted_loadings = pc_matrix * explained_var_arr
                
                # Compress into a 1D array representing the definitive global score of each physical feature
                feature_scores = np.sum(weighted_loadings, axis=1)
                
                # Dynamically slice the exact indices of the absolute best features
                top_indices = np.argsort(feature_scores)[::-1][:PCA_TARGET_FEATURES]
                selected_indices = sorted(list(top_indices))

                selected_column_names = [num_cols[i] for i in selected_indices]
                
                # Write log mapping
                with open(MODELS_DIR / "pca_selected_features.txt", "w") as f:
                    f.write("\n".join(selected_column_names))
                    
                print(f"[pca] Feature selection kept {len(selected_column_names)} physical properties.")
                print(f"[pca] Extraction metadata logged to: {MODELS_DIR}/pca_selected_features.txt")
                
                # Prune dataframe computationally
                cols_to_drop = [c for c in num_cols if c not in selected_column_names]
                df = df.drop(*cols_to_drop)
                
                slicer = VectorSlicer(inputCol="features", outputCol="selected_features", indices=selected_indices)
                df = slicer.transform(df)
                df = df.drop("features").withColumnRenamed("selected_features", "features")
                df = df.drop("pca_features")  # Remove abstract representation since we mapped back to physical features
                
        
    if USE_IP2VEC:
        # Execute Distributed Skip-gram Embeddings Generation
        # NET_ENTITIES are safely decoupled, we now compute the IP2Vec arrays natively.
        df = compute_ip2vec_embeddings(df, context_columns=IP2VEC_SENTENCE)

    if cache:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "preprocessed_cache" / "final_preprocessed.parquet"
        
        cache_root = Path(cache_dir).parent
        cache_root.mkdir(parents=True, exist_ok=True)

        print("[preprocessing] Writing unified preprocessed parquet to disk...")
        df = df.persist()
        df.write.mode("overwrite").parquet(str(cache_dir))
        
        # --- TEMPORARY DEBUG ---
        # Dump 5 rows of the final matrix to CSV to inspect the features and embeddings shape
        df.limit(5).toPandas().to_csv(cache_root / "final_preprocessed_head_5.csv", index=False)
        # -----------------------
        
        df.unpersist()

        print(f"[preprocessing] Unified Parquet saved at: {str(cache_dir)}")

    return df

