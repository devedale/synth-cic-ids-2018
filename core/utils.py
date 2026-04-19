#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-wide utility functions for File I/O and Statistics to prevent core modules bloat."""

from pathlib import Path
from typing import List, Optional


def clean_temp(base_dir: Path) -> None:
    """Cleanup temporary extraction and flow-csv folders used by ingestion."""
    import shutil
    for rel in ("spark_tmp",):
        target = base_dir / rel
        if target.exists():
            for path in target.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)


def generate_crosstab_report(full_df, stats_file: Path) -> None:
    """Generate a Cross-Tabulated Dataset Statistics Report explicitly utilizing PySpark actions."""
    if stats_file.exists() or full_df.count() == 0:
        return

    print("\n[utils] Generating Cross-Tabulated Dataset Statistics Report (First run)...")

    crosstab_df = full_df.crosstab("Label", "_source_day")
    crosstab_pd = crosstab_df.toPandas()

    if not crosstab_pd.empty:
        crosstab_pd.set_index("Label__source_day", inplace=True)
        crosstab_pd.index.name = "Label"
        crosstab_pd["Total"] = crosstab_pd.sum(axis=1)
        crosstab_pd.loc["Total"] = crosstab_pd.sum(axis=0)

        stats_file.parent.mkdir(parents=True, exist_ok=True)
        crosstab_pd.to_csv(stats_file)

        print("\n[utils] --- Dataset Distribution Statistics ---")
        print(crosstab_pd.to_string())
        print(f"----------------------------------------------\n[utils] Saved to {stats_file}\n")


def load_feature_manifest(df=None, models_dir: Optional[Path] = None) -> List[str]:
    """Return the ordered list of PCA-selected numeric feature column names.

    Resolution strategy (in priority order):
      1. Read directly from the Parquet DataFrame schema — the source of truth.
         All double/numeric columns that are not Label, ip2vec_embeddings, or
         one of the IP2VEC_SENTENCE categoricals are numeric features.
      2. Fall back to models_cache/pca_selected_features.txt when df is None.

    This means training scripts never need an external mapping file: the column
    names embedded in the Parquet ARE the feature manifest.

    Args:
        df:          Live PySpark DataFrame (preferred). Pass None to read from disk.
        models_dir:  Override the models cache directory (defaults to settings.MODELS_DIR).

    Returns:
        Ordered list of feature column name strings.
    """
    from configs.settings import IP2VEC_SENTENCE, MODELS_DIR as _MODELS_DIR

    NON_FEATURE_COLS = {"Label", "ip2vec_embeddings"} | set(IP2VEC_SENTENCE)

    if df is not None:
        return [
            c for c, t in df.dtypes
            if t in ("double", "float", "int", "bigint")
            and c not in NON_FEATURE_COLS
        ]

    # Fallback: read from the persisted PCA manifest file
    _dir = models_dir or _MODELS_DIR
    manifest = Path(_dir) / "pca_selected_features.txt"
    if manifest.exists():
        return manifest.read_text().splitlines()

    raise FileNotFoundError(
        f"No DataFrame provided and {manifest} not found. "
        "Run the preprocessing pipeline (main.py) at least once first."
    )


def build_numeric_assembler(feature_cols: List[str], output_col: str = "_numeric_features"):
    """Return a configured VectorAssembler for the given named feature columns.

    Typical usage inside a training Pipeline:

        feature_names = load_feature_manifest(df)
        assembler     = build_numeric_assembler(feature_names)
        scaler        = StandardScaler(inputCol="_numeric_features", outputCol="_scaled")
        clf           = RandomForestClassifier(featuresCol="_scaled", labelCol="Label")
        model         = Pipeline(stages=[assembler, scaler, clf]).fit(train_df)

    Args:
        feature_cols: Ordered list of numeric column names (from load_feature_manifest).
        output_col:   Name of the assembled DenseVector column.

    Returns:
        Configured but unfitted VectorAssembler.
    """
    from pyspark.ml.feature import VectorAssembler
    return VectorAssembler(inputCols=feature_cols, outputCol=output_col, handleInvalid="skip")
