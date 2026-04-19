#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPO Benchmark — Neural Network Architectures for CICIDS2018
============================================================
Uses Optuna + Hyperband pruner to search over multiple neural network
architectures (MLP, ResNet-Tabular, 1D-CNN, Autoencoder-Classifier)
on two parallel objectives:

  1. SUPERVISED    — binary classification: Benign (0) vs Attack (1)
  2. UNSUPERVISED  — anomaly detection via reconstruction error (AutoencoderClassifier)

  • Binary-anomaly CM   — thresholded anomaly score vs ground truth
  • Multi-class CM      — classifier predictions vs ground truth (multi-label-ready)

All trials are logged to MLflow for comparison and reproducibility.

Usage
-----
  .venv/bin/python scratch_hpo.py [--trials 30] [--sample 0.3]

HPO Parameter Index (Neural search space)
-----------------------------------------
The following parameters can be varied within the `_build_model` and `make_objective` functions:

1. Global Parameters:
   - `use_ip2vec`: [True, False] - Wether to concatenate IP2Vec embeddings to numeric features.
   - `lr`: [1e-5, 1e-2] (log) - Learning rate for the optimizer.
   - `optimizer`: ["AdamW", "SGD"] - Optimization algorithm.
   - `weight_decay`: [1e-7, 1e-3] - Regularization strength.

2. MLP (Multi-Layer Perceptron):
   - `mlp_n_layers`: [2, 6] - Number of hidden layers.
   - `mlp_hidden`: [64, 1024] - Neurons per layer (shared across all layers).
   - `dropout`: [0.0, 0.5] - Probability of dropping neurons.

3. ResNet (Residual Tabular Network):
   - `resnet_hidden`: [64, 512] - Dimension of the residual blocks.
   - `resnet_n_blocks`: [2, 8] - Number of residual skip-connection blocks.

4. CNN1D (1-Dimensional Convolutional):
   - `cnn_filters`: [32, 128] - Number of filters in the convolutional layers.
   - `cnn_kernel`: [3, 5, 7] - Size of the sliding 1D window.

5. Autoencoder (Hybrid Anomaly Detector):
   - `ae_hidden`: [128, 512] - Encoder dimension.
   - `ae_latent`: [16, 64] - Bottleneck dimension (latent space).
   - `RECON_WEIGHT`: [0.3] (fixed) - Balance between classification and anomaly detection loss.
"""

import argparse
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ─── stdlib ───────────────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

# ─── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch

from configs.settings import CACHE_DIR, ML_CLASS_STRATEGY, MODELS_DIR
from core.utils import load_feature_manifest
from core.visuals import plot_confusion_matrix

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 16          # ip2vec vector size (fixed at preprocessing time)
N_EPOCHS  = 20          # max epochs per trial (Hyperband may prune earlier)
BATCH_SIZE = 512
RECON_WEIGHT = 0.3      # weight of reconstruction loss in AutoencoderClassifier

# Anomaly threshold percentile: rows with reconstruction error > this percentile
# of the TRAINING set error are flagged as anomalous.
ANOMALY_THRESHOLD_PCT = 95

MLFLOW_DIR = str(Path(__file__).parent / "mlruns")
VISUALS_DIR = Path(__file__).parent / "data" / "visuals" / "hpo"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_parquet_to_tensors(sample_fraction: float = 1.0):
    """Read the preprocessed Parquet into PyTorch tensors via Pandas.

    Returns
    -------
    Tuple of (X_numeric, X_embed | None, y, feature_names)
    """
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as Fspark

    print("[data] Initialising Spark reader...")
    spark = (
        SparkSession.builder
        .appName("HPO_DataLoader")
        .config("spark.driver.memory", "6g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    path = f"{CACHE_DIR}/final_preprocessed_{ML_CLASS_STRATEGY}.parquet"
    df   = spark.read.parquet(path)

    if sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=42)

    feature_names: List[str] = load_feature_manifest(df)
    has_embed = "ip2vec_embeddings" in df.columns

    # Collect to Pandas (safe: post-undersampling the dataset fits in RAM)
    print(f"[data] Collecting {df.count():,} rows to Pandas...")
    pdf = df.toPandas()
    spark.stop()

    # Dynamic Label Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    pdf["Label"] = le.fit_transform(pdf["Label"].astype(str))
    class_names = le.classes_.tolist()
    
    y = torch.tensor(pdf["Label"].values, dtype=torch.long)

    # Numeric features — named columns, already StandardScaled
    X_num = torch.tensor(
        pdf[feature_names].values.astype(np.float32),
        dtype=torch.float32,
    )

    # IP2Vec embeddings — serialised as string in Pandas; parse back to array
    X_emb = None
    if has_embed:
        def _parse_vec(v):
            if isinstance(v, (str, bytes)):
                v_str = v.decode() if isinstance(v, bytes) else v
                return np.fromstring(v_str.strip("[]"), sep=",")
            return np.asarray(v.toArray() if hasattr(v, "toArray") else v)

        emb_arr = np.stack(pdf["ip2vec_embeddings"].apply(_parse_vec).values)
        X_emb = torch.tensor(emb_arr.astype(np.float32), dtype=torch.float32)

    print(f"[data] X_numeric shape : {X_num.shape}")
    if X_emb is not None:
        print(f"[data] X_embed shape   : {X_emb.shape}")
    print(f"[data] Labels detected : {class_names}")
    print(f"[data] Label counts    : {dict(zip(*np.unique(y.numpy(), return_counts=True)))}")

    return X_num, X_emb, y, feature_names, class_names


def _split(X_num, X_emb, y, train_r=0.7, val_r=0.15):
    n = len(y)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    t = int(n * train_r)
    v = int(n * (train_r + val_r))
    tr, va, te = idx[:t], idx[t:v], idx[v:]

    def _s(x, i): return x[i] if x is not None else None

    splits = {}
    for name, i in [("train", tr), ("val", va), ("test", te)]:
        splits[name] = (X_num[i], _s(X_emb, i), y[i])
    return splits


def _make_loader(X_num, X_emb, y, shuffle=True) -> DataLoader:
    tensors = [X_num, y] if X_emb is None else [X_num, X_emb, y]
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Standard multilayer perceptron with configurable depth and dropout."""
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x), None          # (logits, recon=None)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return F.relu(self.block(x) + x)


class ResNetTabular(nn.Module):
    """Residual network adapted for tabular data (He et al. 2016 variant)."""
    def __init__(self, input_dim: int, hidden_dim: int, n_blocks: int,
                 dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.stem    = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.blocks  = nn.Sequential(*[_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head    = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))), None


class CNN1D(nn.Module):
    """1-D convolutional network — treats feature vector as a 1-D signal.
    Works well when features have local correlation structure (e.g. IAT stats
    grouped together in the PCA-sorted input)."""
    def __init__(self, input_dim: int, n_filters: int, kernel: int,
                 dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters * 2, kernel, padding=pad), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_filters * 2, n_classes))

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.head(z), None


class AutoencoderClassifier(nn.Module):
    """Dual-head network: shared encoder → classifier (supervised) + decoder (unsupervised).

    Loss = α·CrossEntropy  +  (1-α)·MSE_reconstruction

    The reconstruction error is the anomaly score: high error → likely anomaly.
    This allows training on ALL data (including unlabelled) via the reconstruction
    head, while the classification head uses the labelled subset.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes),
        )

    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return logits, recon          # (logits, reconstruction)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error (higher = more anomalous)."""
        z     = self.encoder(x)
        recon = self.decoder(z)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(trial, input_dim: int, n_classes: int) -> nn.Module:
    arch = trial.suggest_categorical("architecture", [
        "mlp", "resnet", "cnn1d", "autoencoder",
    ])

    if arch == "mlp":
        n_layers = trial.suggest_int("mlp_n_layers", 2, 6)
        hidden   = trial.suggest_categorical("mlp_hidden", [64, 128, 256, 512, 1024])
        dropout  = trial.suggest_float("dropout", 0.0, 0.5)
        return MLP(input_dim, [hidden] * n_layers, dropout, n_classes=n_classes)

    elif arch == "resnet":
        hidden  = trial.suggest_categorical("resnet_hidden", [64, 128, 256, 512])
        n_blocks = trial.suggest_int("resnet_n_blocks", 2, 8)
        dropout  = trial.suggest_float("dropout", 0.0, 0.4)
        return ResNetTabular(input_dim, hidden, n_blocks, dropout, n_classes=n_classes)

    elif arch == "cnn1d":
        n_filters = trial.suggest_categorical("cnn_filters", [32, 64, 128])
        kernel    = trial.suggest_categorical("cnn_kernel", [3, 5, 7])
        dropout   = trial.suggest_float("dropout", 0.0, 0.4)
        return CNN1D(input_dim, n_filters, kernel, dropout, n_classes=n_classes)

    elif arch == "autoencoder":
        hidden  = trial.suggest_categorical("ae_hidden", [128, 256, 512])
        latent  = trial.suggest_categorical("ae_latent", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        return AutoencoderClassifier(input_dim, hidden, latent, dropout, n_classes=n_classes)

    raise ValueError(f"Unknown arch: {arch}")


def _train_epoch(model, loader, optimizer, has_embed: bool):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if has_embed:
            x_num, x_emb, y = batch
            x = torch.cat([x_num, x_emb], dim=1).to(DEVICE)
        else:
            x_num, y = batch
            x = x_num.to(DEVICE)
        y = y.to(DEVICE)

        logits, recon = model(x)
        loss = F.cross_entropy(logits, y)
        if recon is not None:
            loss = (1 - RECON_WEIGHT) * loss + RECON_WEIGHT * F.mse_loss(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def _evaluate(model, loader, has_embed: bool) -> Dict:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        if has_embed:
            x_num, x_emb, y = batch
            x = torch.cat([x_num, x_emb], dim=1).to(DEVICE)
        else:
            x_num, y = batch
            x = x_num.to(DEVICE)

        logits, _ = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y)

    logits = torch.cat(all_logits)
    y      = torch.cat(all_y).numpy()
    probs  = F.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(1).numpy()

    # Use multi_class='ovr' for AUC-ROC if multiclass, else simple binary
    n_classes_model = probs.shape[1]
    if n_classes_model > 2:
        # multiclass
        try:
            auc = roc_auc_score(y, probs, multi_class='ovr', labels=range(n_classes_model)) if len(np.unique(y)) > 1 else 0.5
        except:
            auc = 0.5
    else:
        # standard binary AUC uses probability of positive class
        auc = roc_auc_score(y, probs[:, 1]) if len(np.unique(y)) > 1 else 0.5

    return {
        "auc_roc":  auc,
        "accuracy": accuracy_score(y, preds),
        "f1":       f1_score(y, preds, average="weighted", zero_division=0),
        "preds":    preds,
        "probs":    probs,
        "labels":   y,
    }


@torch.no_grad()
def _anomaly_scores(model: AutoencoderClassifier, loader, has_embed: bool) -> np.ndarray:
    if not isinstance(model, AutoencoderClassifier):
        return None
    model.eval()
    scores = []
    for batch in loader:
        if has_embed:
            x_num, x_emb, y = batch
            x = torch.cat([x_num, x_emb], dim=1).to(DEVICE)
        else:
            x_num, y = batch
            x = x_num.to(DEVICE)
        scores.append(model.anomaly_score(x).cpu().numpy())
    return np.concatenate(scores)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

# Internal CM plotting functions removed in favor of core.visuals.plot_confusion_matrix



# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(splits, feature_names, has_embed, n_classes):
    n_numeric = len(feature_names)

    def objective(trial: optuna.Trial) -> float:
        # ── Architecture choices ──────────────────────────────────────────────
        use_ip2vec = trial.suggest_categorical("use_ip2vec", [True, False])
        input_dim  = n_numeric + (EMBED_DIM if (use_ip2vec and has_embed) else 0)

        model = _build_model(trial, input_dim, n_classes).to(DEVICE)

        # ── Optimization ──────────────────────────────────────────────────────
        lr       = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        opt_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        
        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            momentum  = trial.suggest_float("momentum", 0.8, 0.99)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        _embed_flag = use_ip2vec and has_embed

        # Build loaders from splits supporting both with/without embed
        def _loader(split_key, shuffle):
            xn, xe, yy = splits[split_key]
            if _embed_flag and xe is not None:
                ds = TensorDataset(xn, xe, yy)
            else:
                ds = TensorDataset(xn, yy)
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

        tr_loader = _loader("train", True)
        va_loader = _loader("val",   False)

        # ── Training with Hyperband pruning ───────────────────────────────────
        best_auc = 0.0
        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "architecture": trial.params.get("architecture"),
                "use_ip2vec":   use_ip2vec,
                "lr":           lr,
                "input_dim":    input_dim,
            })

            for epoch in range(N_EPOCHS):
                _train_epoch(model, tr_loader, optimizer, _embed_flag)
                scheduler.step()
                val_metrics = _evaluate(model, va_loader, _embed_flag)
                auc = val_metrics["auc_roc"]

                mlflow.log_metrics({
                    "val_auc_roc": auc,
                    "val_f1":      val_metrics["f1"],
                }, step=epoch)

                # Report to Hyperband — prune underperforming trials early
                trial.report(auc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                best_auc = max(best_auc, auc)

            mlflow.log_metric("best_val_auc", best_auc)

        return best_auc

    return objective


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION ON BEST TRIAL
# ══════════════════════════════════════════════════════════════════════════════



def run_final_evaluation(best_params: Dict, splits, feature_names, has_embed, class_names):
    """Retrain best configuration on train+val, evaluate on test, plot CMs."""
    print("\n[final] Retraining best configuration on full train set...")

    n_classes = len(class_names)
    use_ip2vec = best_params["use_ip2vec"]
    _embed_flag = use_ip2vec and has_embed
    input_dim  = len(feature_names) + (EMBED_DIM if _embed_flag else 0)

    # Merge train + val for final retraining
    def _cat(a, b): return torch.cat([a, b]) if a is not None else None
    xn_full = torch.cat([splits["train"][0], splits["val"][0]])
    xe_full = _cat(splits["train"][1], splits["val"][1])
    y_full  = torch.cat([splits["train"][2], splits["val"][2]])

    def _loader(xn, xe, yy, shuffle):
        tensors = [xn, xe, yy] if (_embed_flag and xe is not None) else [xn, yy]
        return DataLoader(TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle)

    tr_loader = _loader(xn_full, xe_full, y_full, True)

    # Rebuild model from best params
    class _MockTrial:
        def __init__(self, params): self._p = params
        def suggest_categorical(self, k, _): return self._p[k]
        def suggest_int(self, k, *_):        return self._p[k]
        def suggest_float(self, k, *_):      return self._p[k]

    model = _build_model(_MockTrial(best_params), input_dim, n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

    for _ in range(N_EPOCHS):
        _train_epoch(model, tr_loader, optimizer, _embed_flag)

    # ── Test evaluation ───────────────────────────────────────────────────────
    te_loader = _loader(*splits["test"], False)
    metrics   = _evaluate(model, te_loader, _embed_flag)

    y_true = metrics["labels"]
    y_pred = metrics["preds"]
    arch   = best_params.get("architecture", "best")

    present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    present_names   = [class_names[i] for i in present_classes if i < len(class_names)]
    print("\n" + classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    ))

    # ── 1. Multi-class confusion matrix (Supervised View) ─────────────────────
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title=f"Supervised Classification CM — {arch}",
        output_path=VISUALS_DIR / f"cm_supervised_{arch}.png",
        cmap="Blues"
    )

    # ── 2. Binary Anomaly confusion matrix (Unsupervised/Binary View) ─────────
    # If the model has an anomaly score (reconstruction), use it.
    # Otherwise, collapse the classifier's predictions into Binary (Benign vs Attack).
    y_binary_true = (y_true > 0).astype(int)
    
    anomaly_scores = _anomaly_scores(model, te_loader, _embed_flag)
    if anomaly_scores is not None:
        # Autoencoder Reconstruction Thresholding
        tr_scores = _anomaly_scores(model, tr_loader, _embed_flag)
        threshold = np.percentile(tr_scores, ANOMALY_THRESHOLD_PCT)
        y_binary_pred = (anomaly_scores > threshold).astype(int)
        cm_title = f"Anomaly Detection CM (Reconstruction) — {arch}"
    else:
        # Standard Classifier Collapsed to Binary
        y_binary_pred = (y_pred > 0).astype(int)
        cm_title = f"Anomaly Detection CM (Binary Collapse) — {arch}"

    plot_confusion_matrix(
        y_binary_true, y_binary_pred, ["Benign", "Attack"],
        title=cm_title,
        output_path=VISUALS_DIR / f"cm_anomaly_{arch}.png",
        cmap="Reds"
    )

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int,   default=20,  help="Number of Optuna trials")
    parser.add_argument("--sample",  type=float, default=0.3, help="Fraction of Parquet to use")
    parser.add_argument("--epochs",  type=int,   default=None)
    args = parser.parse_args()

    global N_EPOCHS
    if args.epochs:
        N_EPOCHS = args.epochs

    print("═" * 70)
    print("  CICIDS2018 — Neural HPO Benchmark")
    print(f"  Device   : {DEVICE}")
    print(f"  Trials   : {args.trials}  (Hyperband pruner, TPE sampler)")
    print(f"  Epochs   : {N_EPOCHS} max per trial")
    print(f"  Sample   : {int(args.sample * 100)}% of Parquet")
    print("═" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_num, X_emb, y, feature_names, class_names = _load_parquet_to_tensors(args.sample)
    has_embed = X_emb is not None
    splits    = _split(X_num, X_emb, y)
    n_classes = len(class_names)

    print(f"\n[data] Feature columns ({len(feature_names)}): {feature_names}")
    print(f"[data] IP2Vec available: {has_embed}")
    print(f"[data] Target classes ({n_classes}): {class_names}")

    # ── MLflow experiment ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_DIR)
    mlflow.set_experiment("CICIDS2018_MultiClass_HPO")

    # ── Optuna study ──────────────────────────────────────────────────────────
    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = HyperbandPruner(
        min_resource=3,           # minimum epochs before pruning
        max_resource=N_EPOCHS,
        reduction_factor=3,       # aggressiveness of pruning
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="multiclass_hpo",
    )

    print(f"\n[optuna] Starting search over {args.trials} trials...")
    with mlflow.start_run(run_name="HPO_Study"):
        study.optimize(
            make_objective(splits, feature_names, has_embed, n_classes),
            n_trials=args.trials,
            n_jobs=1,             # set >1 only with multiple GPUs or cores
            show_progress_bar=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  OPTUNA RESULTS")
    print("═" * 70)
    print(f"  Best AUC-ROC : {study.best_value:.4f}")
    print(f"  Best params  : ")
    for k, v in study.best_params.items():
        print(f"    {k:<25} = {v}")

    # Save best params JSON for reproducibility
    best_json = Path(__file__).parent / "models_cache" / "best_hpo_params.json"
    best_json.parent.mkdir(exist_ok=True)
    best_json.write_text(json.dumps(study.best_params, indent=2))
    print(f"\n  Best params saved → {best_json}")

    # ── Final evaluation with confusion matrices ───────────────────────────────
    final_metrics = run_final_evaluation(study.best_params, splits, feature_names, has_embed, class_names)
    print(f"\n  TEST AUC-ROC : {final_metrics['auc_roc']:.4f}")
    print(f"  TEST F1      : {final_metrics['f1']:.4f}")
    print(f"  TEST Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"\n  Confusion matrices saved to: {VISUALS_DIR}")
    print(f"  MLflow dashboard : mlflow ui --backend-store-uri {MLFLOW_DIR}")
    print("═" * 70)


if __name__ == "__main__":
    main()
