#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal preprocessing stage for the CIC-IDS-2018 Data Generator."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Apply lightweight preprocessing and optionally persist day-wise snapshots."""
    if df.empty:
        return df

    out = df.copy()

    if sample_size and len(out) > sample_size:
        out = out.sample(n=sample_size, random_state=42)

    # Drop single-value columns to reduce dimensionality noise.
    nunique = out.nunique(dropna=False)
    to_drop = nunique[nunique <= 1].index.tolist()
    out = out.drop(columns=to_drop, errors="ignore")

    out = out.fillna(0)
    out = out.replace([float("inf"), float("-inf")], 0)

    if "Label" in out.columns:
        le = LabelEncoder()
        out["Label"] = le.fit_transform(out["Label"].astype(str))

    num_cols = out.select_dtypes(include=["number"]).columns.difference(["Label"])
    if len(num_cols) > 0:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])

    if cache:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "preprocessed_cache"
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        #if "_source_day" in out.columns:
        #    for day, day_df in out.groupby("_source_day"):
        #        day_dir = cache_root / str(day)
        #        day_dir.mkdir(parents=True, exist_ok=True)
        #        day_df.to_csv(day_dir / "preprocessed.csv", index=False)
        #else:
        #    out.to_csv(cache_root / "preprocessed.csv", index=False)

    return out


def clean_temp(base_dir: Path) -> None:
    """Cleanup temporary extraction and flow-csv folders used by ingestion."""
    for rel in ("data/s3_csvs",):
        target = base_dir / rel
        if target.exists():
            for path in sorted(target.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
