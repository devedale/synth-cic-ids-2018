#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PySpark out-of-core preprocessing stage for the CIC-IDS-2018 pipeline.

Design principle
----------------
PCA is used exclusively for FEATURE SELECTION — to rank which of the ~75
original flow metrics carry the most variance-weighted signal.  The PCA
projection itself is discarded.  The output Parquet stores the top-N selected
features as individual, *named* double columns, making the dataset
self-documenting and inspection-friendly without any external mapping file.

VectorAssembler is intentionally deferred to training time: it is O(n) and
takes <1 s even on millions of rows, so caching an assembled vector adds
complexity without measurable performance benefit.

Final Parquet schema
---------------------
  <25 named feature columns>  : double  — StandardScaled numeric features
  ip2vec_embeddings           : vector  — 16-dim Skip-gram entity embeddings
  Dst Port                    : double  — categorical token (One-Hot baseline)
  Protocol                    : string  — categorical token (One-Hot baseline)
  Src Region                  : string  — categorical token (One-Hot baseline)
  Label                       : double  — 0.0 = Benign  /  1.0 = Attack
"""

from pathlib import Path
from typing import List, Optional, Union


def preprocess_spark(
    df,
    sample_size: Optional[int] = None,
    cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
):
    """Apply PySpark out-of-core preprocessing over 40 GB parquets and optionally save."""
    from core.ip2vec import compute_ip2vec_embeddings
    from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, PCA
    from configs.settings import (
        NET_ENTITIES, PCA_COMPONENTS, PCA_FEATURE_SELECTION, PCA_TARGET_FEATURES,
        IP2VEC_SENTENCE, RANDOM_SEED, USE_IP2VEC, USE_PCA, MODELS_DIR,
    )
    import pyspark.sql.functions as F
    from pyspark.sql.types import DoubleType
    import numpy as np

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Optional row sampling ──────────────────────────────────────────────────
    if sample_size:
        total_rows = df.count()
        if total_rows > sample_size:
            fraction = min(1.0, sample_size / total_rows)
            df = df.withColumn(
                "_rand_val", 
                F.abs(F.xxhash64(F.lit(RANDOM_SEED), *df.columns)) / F.lit(9223372036854775807)
            )
            df = df.filter(F.col("_rand_val") <= fraction).drop("_rand_val")

    # ── Label Preservation ──────────────────────────────────────────────────────
    # We leave the 'Label' column as strings. This allows the subsequent 
    # training/HPO scripts to perform dynamic multi-class encoding and 
    # generate confusion matrices with specific attack names (e.g. 'DDoS-HOIC').
    print(f"[preprocessing] Preserving string labels for multi-class support.")

    # ── DTYPE RESOLUTION ───────────────────────────────────────────────────────
    # Ingestion writes Parquet without inferSchema for speed, so all columns
    # arrive as strings.  Cast every quantitative measurement to Double.
    KEEP_AS_STRING = {
        "id", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label",
        "Attempted Category", "_source_day", "Protocol",
    }
    for c, t in df.dtypes:
        if t == "string" and c not in KEEP_AS_STRING:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # ── Numeric column selection ───────────────────────────────────────────────
    # NET_ENTITIES are excluded: they feed IP2Vec and must NOT enter the numeric
    # feature vector (they carry identity, not flow behaviour).
    num_cols: List[str] = [
        c for c, t in df.dtypes
        if t in ("int", "double", "float", "bigint")
        and c != "Label"
        and c not in NET_ENTITIES
    ]

    # ── Infinity / NaN sanitization ────────────────────────────────────────────
    # CICIDS2018 has corrupted bytes/s fields that evaluate to ±Infinity when
    # packets arrive at t=0.  PySpark's LAPACK backend aborts on Inf during PCA.
    # A flat select() plan lets Catalyst optimise all replacements in one pass.
    _inf, _ninf = float("inf"), float("-inf")
    df = df.select([
        F.when(
            F.isnan(F.col(c)) | F.col(c).isNull()
            | (F.col(c) == _inf) | (F.col(c) == _ninf),
            0.0,
        ).otherwise(F.col(c)).alias(c)
        if c in num_cols else F.col(c)
        for c in df.columns
    ])

    # ── StandardScaling + PCA (for feature selection only) ────────────────────
    selected_feature_cols: List[str] = num_cols  # default: keep all

    if len(num_cols) > 0 and USE_PCA:
        print(f"[preprocessing] Fitting StandardScaler on {len(num_cols)} numeric columns...")
        assembler = VectorAssembler(inputCols=num_cols, outputCol="_features_raw", handleInvalid="skip")
        df_vec = assembler.transform(df)

        scaler = StandardScaler(inputCol="_features_raw", outputCol="_features_scaled",
                                withStd=True, withMean=True)
        scaler_model = scaler.fit(df_vec)
        scaler_model.write().overwrite().save(str(MODELS_DIR / "scaler_model"))
        df_vec = scaler_model.transform(df_vec)

        print(f"[preprocessing] Fitting PCA ({PCA_COMPONENTS} components) for feature selection...")
        pca = PCA(k=PCA_COMPONENTS, inputCol="_features_scaled", outputCol="_pca_out")
        pca_model = pca.fit(df_vec)
        pca_model.write().overwrite().save(str(MODELS_DIR / "pca_model"))

        # Generate variance plot (uses the fitted model metadata, not the df)
        from core.visuals import plot_pca_variance
        explained_var_arr = pca_model.explainedVariance.toArray()
        plot_pca_variance(explained_var_arr.tolist(), MODELS_DIR.parent / "data" / "visuals")

        if PCA_FEATURE_SELECTION:
            # ── Global Variance-Weighted Importance Scoring ────────────────────
            # Each physical feature's importance = sum over components of
            # (|loading on that component| × variance explained by that component).
            # This yields a single definitive score per original feature.
            pc_matrix = np.abs(pca_model.pc.toArray())          # shape: (n_features, k)
            feature_scores = np.sum(pc_matrix * explained_var_arr, axis=1)  # (n_features,)

            top_indices = np.argsort(feature_scores)[::-1][:PCA_TARGET_FEATURES]
            selected_feature_cols = [num_cols[i] for i in sorted(top_indices)]

            # Persist the manifest so training/benchmark scripts are self-sufficient
            import json
            json_manifest_path = MODELS_DIR / "pca_selected_features.json"
            json_manifest_path.write_text(json.dumps(selected_feature_cols, indent=2))
            
            manifest_path = MODELS_DIR / "pca_selected_features.txt"
            manifest_path.write_text("\n".join(selected_feature_cols))
            print(f"[pca] Selected {len(selected_feature_cols)} features → {json_manifest_path}")

        # Now apply the scaler to the INDIVIDUAL selected columns so each
        # column in the Parquet is the StandardScaled value (mean=0, std=1).
        # We reconstruct this from the scaler's mean/std arrays.
        scaler_mean = scaler_model.mean.toArray()   # one value per num_cols entry
        scaler_std  = scaler_model.std.toArray()

        scale_exprs = []
        col_index = {c: i for i, c in enumerate(num_cols)}
        for c in df.columns:
            if c in num_cols:
                i = col_index[c]
                std = float(scaler_std[i]) if scaler_std[i] != 0 else 1.0
                mean = float(scaler_mean[i])
                scale_exprs.append(((F.col(c) - mean) / std).alias(c))
            else:          # non-numeric: keep as-is
                scale_exprs.append(F.col(c))

        df = df.select(*scale_exprs)

    elif len(num_cols) > 0:
        # PCA disabled: just drop NET_ENTITIES and keep all numeric cols (unscaled)
        metadata_cols = {"id", "Flow ID", "Src IP", "Dst IP", "Timestamp",
                         "Attempted Category", "_source_day"}
        df = df.drop(*[c for c in metadata_cols if c in df.columns])

    # ── IP2Vec ─────────────────────────────────────────────────────────────────
    # Src IP is still present here; ip2vec.py uses it to derive Src Region via
    # the vectorised GeoIP Pandas UDF, then drops all intermediates automatically.
    if USE_IP2VEC:
        df = compute_ip2vec_embeddings(df, context_columns=IP2VEC_SENTENCE)

    # ── Final column pruning ───────────────────────────────────────────────────
    # The Parquet is the training matrix.  Keep ONLY:
    #   • ALL numeric feature columns       (self-documenting, named doubles)
    #   • ip2vec_embeddings                 (16-dim latent vector)
    #   • IP2VEC_SENTENCE categoricals      (raw tokens for baseline One-Hot)
    #   • Label                             (binary target)
    #
    # Everything else (IPs, timestamps, internal metadata) is dropped.
    keep = set(num_cols) | {"ip2vec_embeddings", "Label"} | set(IP2VEC_SENTENCE)
    final_cols = [c for c in df.columns if c in keep]
    dropped = len(df.columns) - len(final_cols)
    if dropped > 0:
        print(f"[preprocessing] Pruned {dropped} metadata columns. Final schema ({len(final_cols)} cols):")
        print(f"  {final_cols}")
    df = df.select(*final_cols)

    # ── Persist to Parquet ─────────────────────────────────────────────────────
    if cache:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "preprocessed_cache" / "final_preprocessed.parquet"

        cache_root = Path(cache_dir).parent
        cache_root.mkdir(parents=True, exist_ok=True)

        print("[preprocessing] Writing preprocessed Parquet to disk...")
        df = df.persist()
        df.write.mode("overwrite").parquet(str(cache_dir))

        # Diagnostic sample: 10 Benign + 10 Attack rows — now fully readable
        # because feature columns are named, not packed into an opaque vector.
        try:
            import pandas as pd
            # Robust filtering: 'Benign' is string-based. Use ILIKE or isin for safety.
            benign_rows = df.filter(F.lower(F.col("Label")).contains("benign")).limit(10).toPandas()
            attack_rows = df.filter(~F.lower(F.col("Label")).contains("benign")).limit(10).toPandas()
            sample = pd.concat([benign_rows, attack_rows]) if not attack_rows.empty else benign_rows
            sample.to_csv(cache_root / "final_preprocessed_head_20.csv", index=False)
        except Exception:
            df.limit(20).toPandas().to_csv(cache_root / "final_preprocessed_head_20.csv", index=False)

        df.unpersist()
        print(f"[preprocessing] Saved → {cache_dir}")

    return df
