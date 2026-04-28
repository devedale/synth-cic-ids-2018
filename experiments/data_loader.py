import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from pyspark.sql import SparkSession
from sklearn.preprocessing import LabelEncoder

from experiments.configs import ExperimentConfig
from configs.settings import RANDOM_SEED


def _build_features(pdf: pd.DataFrame, cfg: ExperimentConfig) -> torch.Tensor:
    """Extract feature tensor from a Pandas DataFrame (shared by all loaders)."""
    all_numeric_cols = [
        c for c, dt in pdf.dtypes.items()
        if c not in ["Label", "ip2vec_embeddings", "Dst Port", "Protocol", "Src Region"]
        and pd.api.types.is_numeric_dtype(dt)
    ]

    if cfg.use_pca:
        if not cfg.pca_features_path.exists():
            raise FileNotFoundError(f"PCA features JSON not found: {cfg.pca_features_path}")
        with open(cfg.pca_features_path, "r") as f:
            selected_features = json.load(f)
        feature_names = [c for c in selected_features if c in pdf.columns]
    else:
        feature_names = all_numeric_cols

    X_num = torch.tensor(pdf[feature_names].values.astype(np.float32), dtype=torch.float32)

    X_extra = None
    if cfg.use_ip2vec and "ip2vec_embeddings" in pdf.columns:
        def _parse_vec(v):
            if isinstance(v, (str, bytes)):
                v_str = v.decode() if isinstance(v, bytes) else v
                return np.fromstring(v_str.strip("[]"), sep=",")
            return np.asarray(v.toArray() if hasattr(v, "toArray") else v)

        emb_arr = np.stack(pdf["ip2vec_embeddings"].apply(_parse_vec).values)
        X_extra = torch.tensor(emb_arr.astype(np.float32), dtype=torch.float32)
    elif not cfg.use_ip2vec:
        cols_to_encode = [c for c in ["Dst Port", "Protocol", "Src Region"] if c in pdf.columns]
        if cols_to_encode:
            ohe_df = pd.get_dummies(pdf[cols_to_encode], columns=cols_to_encode, dummy_na=True)
            X_extra = torch.tensor(ohe_df.values.astype(np.float32), dtype=torch.float32)

    return torch.cat([X_num, X_extra], dim=1) if X_extra is not None else X_num


def _load_pandas(cfg: ExperimentConfig, sample_frac: float, seed: int, app_name: str) -> pd.DataFrame:
    """Common Spark → Pandas loader."""
    spark = SparkSession.builder.appName(app_name) \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(str(cfg.parquet_path))

    print(f"[{cfg.name}] Collecting {df.count():,} rows to Pandas...")
    pdf = df.toPandas()
    spark.stop()
    
    # ── Enforce Determinism ──
    # PySpark parquet reading is NOT ordered. We must sort by scalar columns to guarantee 
    # the exact same row order every time before we split/sample in sklearn/PyTorch.
    sort_cols = [c for c in pdf.columns if c not in ["ip2vec_embeddings", "features", "pca_features"]]
    pdf = pdf.sort_values(by=sort_cols).reset_index(drop=True)

    if sample_frac < 1.0:
        pdf = pdf.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        print(f"[{cfg.name}] Sampled to {len(pdf):,} rows")

    return pdf


def load_full_tensors(
    cfg: ExperimentConfig,
    sample_frac: float = 1.0,
    seed: int = RANDOM_SEED,
    k: int = 5,
) -> tuple:
    """
    Load the FULL dataset (no split) ready for k-fold cross validation.

    Classes with fewer than `k` samples are dropped so that StratifiedKFold
    can guarantee at least one sample per class in every fold.

    Returns:
        X       – FloatTensor  [N, F]
        y       – LongTensor   [N]
        class_names – list[str]
    """
    pdf = _load_pandas(cfg, sample_frac, seed, "ExpDataLoader_KFold")

    # Drop classes that are too rare for StratifiedKFold
    label_counts = pdf["Label"].astype(str).value_counts()
    valid_labels = label_counts[label_counts >= k].index
    dropped = set(label_counts.index) - set(valid_labels)
    if dropped:
        print(f"[{cfg.name}] Dropping {len(dropped)} rare class(es) with < {k} samples: {dropped}")
    pdf = pdf[pdf["Label"].astype(str).isin(valid_labels)].reset_index(drop=True)

    le = LabelEncoder()
    pdf["Label"] = le.fit_transform(pdf["Label"].astype(str))
    class_names = le.classes_.tolist()

    print(f"[{cfg.name}] After rare-class filter: {len(pdf):,} rows, {len(class_names)} classes")

    print(f"\n{'='*40}")
    print(f"[{cfg.name}] MODEL INPUT LABEL DISTRIBUTION")
    print(f"{'='*40}")
    # Decode the integers back to strings for the print using the fitted LabelEncoder
    decoded_labels = pd.Series(le.inverse_transform(pdf["Label"]))
    print(decoded_labels.value_counts().to_string())
    print(f"{'='*40}\n")


    X = _build_features(pdf, cfg)
    y = torch.tensor(pdf["Label"].values, dtype=torch.long)

    print(f"[{cfg.name}] X shape: {X.shape}")
    return X, y, class_names


def load_tensors(cfg: ExperimentConfig, sample_frac: float = 1.0, seed: int = RANDOM_SEED) -> tuple:
    """
    Legacy loader that returns a pre-split (train/val/test) tuple.
    Used by the federated runner for compatibility when k-fold is not needed.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test, class_names).
    """
    pdf = _load_pandas(cfg, sample_frac, seed, "ExpDataLoader")

    le = LabelEncoder()
    pdf["Label"] = le.fit_transform(pdf["Label"].astype(str))
    class_names = le.classes_.tolist()
    y = torch.tensor(pdf["Label"].values, dtype=torch.long)

    X_full = _build_features(pdf, cfg)
    print(f"[{cfg.name}] X_full shape: {X_full.shape}")

    n_samples = len(y)
    idx = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    t = int(n_samples * 0.7)
    v = int(n_samples * 0.85)

    train_idx, val_idx, test_idx = idx[:t], idx[t:v], idx[v:]
    return (
        X_full[train_idx], X_full[val_idx], X_full[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
        class_names,
    )


def create_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 512,
    shuffle: bool = True,
    seed: int = RANDOM_SEED,
) -> DataLoader:
    """Create a DataLoader with fully deterministic shuffle when seed is provided."""
    ds = TensorDataset(X, y)

    if shuffle:
        # Fixed Generator → same shuffle order every run
        g = torch.Generator()
        g.manual_seed(seed)

        def worker_init_fn(worker_id: int):
            # Seed every DataLoader worker subprocess independently but reproducibly
            import random
            import numpy as np
            w_seed = seed + worker_id
            random.seed(w_seed)
            np.random.seed(w_seed)
            torch.manual_seed(w_seed)

        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          generator=g, worker_init_fn=worker_init_fn)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
