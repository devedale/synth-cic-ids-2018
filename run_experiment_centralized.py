#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized experiment runner with Stratified K-Fold (k=5) cross-validation.

Each fold trains on 80% of the full dataset and tests on the held-out 20%.
Final metrics are averaged across all k folds so they are directly comparable
with the federated runner, which uses the same fold splits.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from configs.settings import RANDOM_SEED, set_global_seed, IP2VEC_SENTENCE
from core.visuals import plot_confusion_matrix, plot_training_curves
from experiments.configs import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.data_loader import create_loader, load_full_tensors
from experiments.metrics import compute_full_metrics
from experiments.model import build_model
from experiments.trainer import eval_epoch, run_training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KF_SPLITS = 5


def run_centralized_for_config(cfg: ExperimentConfig, sample_frac: float, epochs: int) -> dict:
    print(f"\n{'='*60}\nRunning Centralized (Stratified {KF_SPLITS}-Fold CV) — {cfg.name}\n{'='*60}")
    t0 = time.time()

    # ── Read HPO Params (e.g. batch_size) ────────────────────────────────────
    hpo_params = {}
    if cfg.hpo_params_path.exists():
        with open(cfg.hpo_params_path, "r") as f:
            hpo_params = json.load(f)
    batch_sz = hpo_params.get("batch_size", 512)

    # ── Load the full dataset (rare classes already filtered for k-fold) ─────
    X, y, class_names = load_full_tensors(cfg, sample_frac=sample_frac, k=KF_SPLITS)
    y_np = y.numpy()
    n_classes = len(class_names)

    skf = StratifiedKFold(n_splits=KF_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics: list[dict] = []
    last_test_res = None
    last_train_idx = last_test_idx = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_np), start=1):
        print(f"\n  ── Fold {fold}/{KF_SPLITS}  (train={len(train_idx):,}, test={len(test_idx):,}) ──")

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # Use 15% of the training fold as validation (same ratio as before)
        val_size = max(1, int(len(train_idx) * 0.15))
        rng = np.random.default_rng(RANDOM_SEED + fold)
        val_local_idx = rng.choice(len(train_idx), size=val_size, replace=False)
        train_local_idx = np.setdiff1d(np.arange(len(train_idx)), val_local_idx)

        X_val, y_val = X_tr[val_local_idx], y_tr[val_local_idx]
        X_train_f, y_train_f = X_tr[train_local_idx], y_tr[train_local_idx]

        input_dim = X.shape[1]
        model = build_model(input_dim, n_classes, cfg).to(DEVICE)

        train_loader = create_loader(X_train_f, y_train_f, batch_size=batch_sz, shuffle=True, seed=RANDOM_SEED + fold)
        val_loader   = create_loader(X_val, y_val, batch_size=batch_sz, shuffle=False)
        test_loader  = create_loader(X_te, y_te, batch_size=batch_sz, shuffle=False)

        history = run_training(model, train_loader, val_loader, cfg, epochs, DEVICE)

        test_res = eval_epoch(model, test_loader, DEVICE)
        metrics  = compute_full_metrics(test_res["y"], test_res["preds"], test_res["probs"], class_names, t0)
        fold_metrics.append(metrics)

        print(f"  Fold {fold} → Acc={metrics['accuracy']:.4f}  AUC={metrics['auc_roc']:.4f}  "
              f"F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.4f}")

        last_test_res = test_res
        last_train_idx, last_test_idx = train_idx, test_idx

    # ── Average metrics across all folds ─────────────────────────────────────
    avg_metrics: dict = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        avg_metrics[key] = float(np.mean(vals))
        avg_metrics[f"{key}_std"] = float(np.std(vals))

    # exec_time_s counts the total wall-clock for all folds
    avg_metrics["exec_time_s"] = time.time() - t0

    print(f"\n[{cfg.name}] K-Fold Averages → "
          f"Acc={avg_metrics['accuracy']:.4f}±{avg_metrics['accuracy_std']:.4f}  "
          f"AUC={avg_metrics['auc_roc']:.4f}±{avg_metrics['auc_roc_std']:.4f}")

    # ── Persist outputs (using last fold's model/predictions) ─────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    plot_training_curves(history, cfg.output_dir)

    plot_confusion_matrix(
        last_test_res["y"], last_test_res["preds"], class_names,
        title=f"Supervised CM: {cfg.name} (last fold)",
        output_path=cfg.output_dir / "cm_supervised.png",
        cmap="Blues",
    )

    y_bin      = (last_test_res["y"]     > 0).astype(int)
    y_pred_bin = (last_test_res["preds"] > 0).astype(int)
    plot_confusion_matrix(
        y_bin, y_pred_bin, ["Benign", "Attack"],
        title=f"Anomaly CM: {cfg.name} (last fold)",
        output_path=cfg.output_dir / "cm_anomaly.png",
        cmap="Reds",
    )

    # Add IP2VEC_SENTENCE to results if applicable
    sentence_str = " | ".join(IP2VEC_SENTENCE) if cfg.use_ip2vec else "N/A"

    return {"config_name": cfg.name, "ip2vec_sentence": sentence_str, **avg_metrics, **hpo_params}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of dataset to use (1.0 = full dataset)")
    args = parser.parse_args()

    results = []
    Path("results").mkdir(exist_ok=True)
    set_global_seed(RANDOM_SEED)

    for cfg in EXPERIMENT_CONFIGS:
        set_global_seed(RANDOM_SEED)
        cfg.paradigm = "centralized"
        res = run_centralized_for_config(cfg, args.sample, args.epochs)
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("results/centralized_results.csv", index=False)
    print("\nCentralized experiments completed. Results saved to results/centralized_results.csv")
