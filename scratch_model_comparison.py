#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Baseline (One-Hot Categorical) vs IP2Vec (Skip-gram Embeddings)
===========================================================================
Quantifies the accuracy / F1 / AUC-ROC delta between the two architectural
variants on the preprocessed CICIDS2018 Parquet.

Parquet schema (self-documenting named columns):
  <25 named feature columns>  : double  — StandardScaled numeric features
  ip2vec_embeddings           : vector  — 16-dim Skip-gram entity embeddings
  Dst Port / Protocol / Src Region     — raw categoricals for baseline
  Label                       : double  — 0.0 Benign / 1.0 Attack

Comparison matrix
-----------------
  A. BASELINE  : [numeric features + StandardScaler] + One-Hot categoricals
  B. IP2VEC    : [numeric features + StandardScaler] + ip2vec_embeddings

Usage:
  .venv/bin/python scratch_model_comparison.py
"""

import os
import sys
import time

# ── Must be set before any PySpark import ─────────────────────────────────────
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
# ─────────────────────────────────────────────────────────────────────────────

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

from configs.settings import CACHE_DIR, IP2VEC_SENTENCE, ML_CLASS_STRATEGY
from core.utils import build_numeric_assembler, load_feature_manifest


# ── Config ────────────────────────────────────────────────────────────────────

EMBED_COL   = "ip2vec_embeddings"
NUMERIC_VEC = "_numeric_scaled"      # internal name for the scaled numeric vector
RF_TREES    = 50
RF_MAX_DEPTH = 10
RF_SEED     = 42
SAMPLE_FRACTION = 0.50               # set to 1.0 for a full-dataset run


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Benchmark_Baseline_vs_IP2Vec")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _load_dataset(spark: SparkSession) -> DataFrame:
    path = f"{CACHE_DIR}/final_preprocessed_{ML_CLASS_STRATEGY}.parquet"
    df = spark.read.parquet(path)

    feature_names = load_feature_manifest(df)

    print(f"\n[benchmark] Dataset      : {path}")
    print(f"[benchmark] Total rows   : {df.count():,}")
    print(f"[benchmark] Feature cols : {len(feature_names)}  → {feature_names}")
    print(f"[benchmark] All columns  : {df.columns}")
    df.groupBy("Label").count().orderBy("Label").show()
    return df


def _split(df: DataFrame):
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(fraction=SAMPLE_FRACTION, seed=RF_SEED)
        print(f"[benchmark] Subsampled to {df.count():,} rows ({int(SAMPLE_FRACTION*100)}%).")
    train, test = df.randomSplit([0.8, 0.2], seed=RF_SEED)
    return train, test


def _evaluate(model, test_df: DataFrame, label: str) -> dict:
    preds = model.transform(test_df)
    acc = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="f1"
    ).evaluate(preds)
    auc = BinaryClassificationEvaluator(
        labelCol="Label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(preds)
    return {"label": label, "accuracy": acc, "f1": f1, "auc_roc": auc}


# ── Shared numeric sub-pipeline ───────────────────────────────────────────────
# Both variants share the same numeric preprocessing steps:
#   1. VectorAssembler  — named columns → dense vector
#   2. StandardScaler   — mean=0, std=1  (applied fresh at training time)
# This is idiomatic Spark ML and avoids caching stale scale parameters.

def _numeric_stages(feature_names):
    assembler = build_numeric_assembler(feature_names, output_col="_raw_vec")
    scaler = StandardScaler(
        inputCol="_raw_vec", outputCol=NUMERIC_VEC,
        withStd=True, withMean=True,
    )
    return [assembler, scaler]


# ── Variant A: Baseline ───────────────────────────────────────────────────────

def train_baseline(train_df: DataFrame, test_df: DataFrame, feature_names) -> dict:
    """Numeric features (StandardScaled) + One-Hot encoded categoricals."""
    print("\n" + "─" * 64)
    print("  VARIANT A — BASELINE")
    print("  Numeric (scaled) + One-Hot categorical encoding")
    print(f"  Categorical context: {IP2VEC_SENTENCE}")
    print("─" * 64)

    available_cats = [c for c in IP2VEC_SENTENCE if c in train_df.columns]
    missing = set(IP2VEC_SENTENCE) - set(available_cats)
    if missing:
        print(f"  [WARN] Missing categorical columns skipped: {missing}")

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"_idx_{i}", handleInvalid="keep")
        for i, c in enumerate(available_cats)
    ]
    encoders = [
        OneHotEncoder(inputCol=f"_idx_{i}", outputCol=f"_ohe_{i}", handleInvalid="keep")
        for i in range(len(available_cats))
    ]
    ohe_cols = [f"_ohe_{i}" for i in range(len(available_cats))]

    final_assembler = VectorAssembler(
        inputCols=[NUMERIC_VEC] + ohe_cols,
        outputCol="_baseline_input",
        handleInvalid="skip",
    )
    clf = RandomForestClassifier(
        featuresCol="_baseline_input", labelCol="Label",
        numTrees=RF_TREES, maxDepth=RF_MAX_DEPTH, seed=RF_SEED,
    )

    stages = _numeric_stages(feature_names) + indexers + encoders + [final_assembler, clf]

    t0 = time.time()
    print("  Training...")
    model = Pipeline(stages=stages).fit(train_df)
    result = _evaluate(model, test_df, "Baseline (One-Hot)")
    result["train_time_s"] = time.time() - t0
    return result


# ── Variant B: IP2Vec ─────────────────────────────────────────────────────────

def train_ip2vec(train_df: DataFrame, test_df: DataFrame, feature_names) -> dict:
    """Numeric features (StandardScaled) + pre-computed IP2Vec embeddings."""
    print("\n" + "─" * 64)
    print("  VARIANT B — IP2VEC")
    print("  Numeric (scaled) + 16-dim Skip-gram entity embeddings")
    print(f"  Embedding context sentence: {IP2VEC_SENTENCE}")
    print("─" * 64)

    if EMBED_COL not in train_df.columns:
        print(f"  [ERROR] Column '{EMBED_COL}' not found. Ensure USE_IP2VEC=True in settings.py.")
        return {}

    final_assembler = VectorAssembler(
        inputCols=[NUMERIC_VEC, EMBED_COL],
        outputCol="_ip2vec_input",
        handleInvalid="skip",
    )
    clf = RandomForestClassifier(
        featuresCol="_ip2vec_input", labelCol="Label",
        numTrees=RF_TREES, maxDepth=RF_MAX_DEPTH, seed=RF_SEED,
    )

    stages = _numeric_stages(feature_names) + [final_assembler, clf]

    t0 = time.time()
    print("  Training...")
    model = Pipeline(stages=stages).fit(train_df)
    result = _evaluate(model, test_df, "IP2Vec (Skip-gram)")
    result["train_time_s"] = time.time() - t0
    return result


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(baseline: dict, ip2vec: dict) -> None:
    fmt = "  {:<32} {:>9} {:>9} {:>9} {:>8}"
    print("\n" + "═" * 72)
    print("  BENCHMARK RESULTS")
    print("═" * 72)
    print(fmt.format("Variant", "Accuracy", "F1", "AUC-ROC", "Time(s)"))
    print("  " + "─" * 70)

    def _row(r):
        if not r:
            return
        print(fmt.format(
            r["label"],
            f"{r['accuracy']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auc_roc']:.4f}",
            f"{r['train_time_s']:.1f}",
        ))

    _row(baseline)
    _row(ip2vec)

    if baseline and ip2vec:
        print("  " + "─" * 70)
        d_acc = (ip2vec["accuracy"] - baseline["accuracy"]) * 100
        d_f1  = (ip2vec["f1"]       - baseline["f1"])       * 100
        d_auc = (ip2vec["auc_roc"]  - baseline["auc_roc"])  * 100
        d_t   =  ip2vec["train_time_s"] - baseline["train_time_s"]
        print(fmt.format("Δ IP2Vec vs Baseline", f"{d_acc:+.2f}%", f"{d_f1:+.2f}%",
                          f"{d_auc:+.2f}%", f"{d_t:+.1f}"))

    print("═" * 72 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    spark = _make_spark()
    spark.sparkContext.setLogLevel("ERROR")

    print("\n" + "═" * 72)
    print("  CICIDS2018 — Architecture Comparison Benchmark")
    print(f"  Strategy   : {ML_CLASS_STRATEGY}")
    print(f"  Classifier : RandomForest (trees={RF_TREES}, depth={RF_MAX_DEPTH})")
    print(f"  Sample     : {int(SAMPLE_FRACTION * 100)}% of Parquet")
    print("═" * 72)

    df = _load_dataset(spark)
    feature_names = load_feature_manifest(df)

    train_df, test_df = _split(df)

    baseline_result = train_baseline(train_df, test_df, feature_names)
    ip2vec_result   = train_ip2vec(train_df, test_df, feature_names)

    _print_report(baseline_result, ip2vec_result)
    spark.stop()


if __name__ == "__main__":
    main()
