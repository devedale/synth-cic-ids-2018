import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from pyspark.sql import SparkSession
from sklearn.preprocessing import LabelEncoder
from core.utils import load_feature_manifest

from experiments.configs import ExperimentConfig
from configs.settings import RANDOM_SEED

def load_tensors(cfg: ExperimentConfig, sample_frac: float = 1.0, seed: int = RANDOM_SEED) -> tuple:
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, class_names).
    """
    spark = SparkSession.builder.appName("ExpDataLoader")\
        .config("spark.driver.memory", "6g")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(str(cfg.parquet_path))
    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=seed)

    print(f"[{cfg.name}] Collecting {df.count():,} rows to Pandas...")
    pdf = df.toPandas()
    spark.stop()

    le = LabelEncoder()
    pdf["Label"] = le.fit_transform(pdf["Label"].astype(str))
    class_names = le.classes_.tolist()
    y = torch.tensor(pdf["Label"].values, dtype=torch.long)

    # 1. Base numeric features
    all_numeric_cols = [
        c for c, dt in pdf.dtypes.items() 
        if c not in ["Label", "ip2vec_embeddings", "Dst Port", "Protocol", "Src Region"] 
        and pd.api.types.is_numeric_dtype(dt)
    ]
    
    if cfg.use_pca:
        if not cfg.pca_features_path.exists():
            raise FileNotFoundError(f"PCA features JSON not found: {cfg.pca_features_path}")
        with open(cfg.pca_features_path, 'r') as f:
            selected_features = json.load(f)
            # Ensure they exist in the dataframe
            selected_features = [c for c in selected_features if c in pdf.columns]
        feature_names = selected_features
    else:
        feature_names = all_numeric_cols

    X_num = torch.tensor(pdf[feature_names].values.astype(np.float32), dtype=torch.float32)

    # 2. Add embeddings or OHE
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
        # One-Hot Encoding baseline for [Dst Port, Protocol, Src Region]
        cols_to_encode = [c for c in ["Dst Port", "Protocol", "Src Region"] if c in pdf.columns]
        if cols_to_encode:
            ohe_df = pd.get_dummies(pdf[cols_to_encode], columns=cols_to_encode, dummy_na=True)
            X_extra = torch.tensor(ohe_df.values.astype(np.float32), dtype=torch.float32)
            
    if X_extra is not None:
        X_full = torch.cat([X_num, X_extra], dim=1)
    else:
        X_full = X_num
        
    print(f"[{cfg.name}] X_full shape: {X_full.shape}")

    # Split
    n_samples = len(y)
    idx = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    
    t = int(n_samples * 0.7)
    v = int(n_samples * 0.85)
    
    train_idx, val_idx, test_idx = idx[:t], idx[t:v], idx[v:]
    
    X_train, y_train = X_full[train_idx], y[train_idx]
    X_val, y_val = X_full[val_idx], y[val_idx]
    X_test, y_test = X_full[test_idx], y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

def create_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 512, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
