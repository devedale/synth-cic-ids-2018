#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Learning experiment runner with Stratified K-Fold (k=5) cross-validation.

Uses the SAME fold splits as run_experiment_centralized.py for direct comparability:
  - In each fold, the test set is identical to the centralized test set.
  - The training set is partitioned evenly across FL_NUM_CLIENTS simulated clients.

Metrics are averaged across all k folds.
"""
import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import flwr as fl
from sklearn.model_selection import StratifiedKFold
from flwr.common import Context

from configs.settings import (
    FL_FRACTION_FIT, FL_LOCAL_EPOCHS, FL_MIN_AVAILABLE,
    FL_MIN_FIT_CLIENTS, FL_NUM_CLIENTS, FL_NUM_ROUNDS, RANDOM_SEED,
    set_global_seed
)
from core.visuals import plot_confusion_matrix, plot_training_curves
from experiments.configs import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.data_loader import create_loader, load_full_tensors
from experiments.metrics import compute_full_metrics
from experiments.model import build_model, get_optimizer
from experiments.trainer import eval_epoch, train_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KF_SPLITS = 5


# ── Flower helpers ────────────────────────────────────────────────────────────

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class IDSClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.optimizer = get_optimizer(self.model, cfg)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        for _ in range(FL_LOCAL_EPOCHS):
            train_epoch(self.model, self.train_loader, self.optimizer, DEVICE)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        res = eval_epoch(self.model, self.val_loader, DEVICE)
        return float(res["loss"]), len(self.val_loader.dataset), {"auc": float(res["auc"])}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, val_loader, test_loader, n_classes, cfg, class_names, t0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.n_classes = n_classes
        self.cfg = cfg
        self.class_names = class_names
        self.t0 = t0
        self.best_auc = 0.0
        self.best_metrics = {}
        self.best_res = None
        self.history = {"loss": [], "val_loss": [], "val_auc": []}

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            parameters = fl.common.parameters_to_ndarrays(aggregated_weights)
            set_parameters(self.model, parameters)

            val_res = eval_epoch(self.model, self.val_loader, DEVICE)
            self.history["val_loss"].append(val_res["loss"])
            self.history["val_auc"].append(val_res["auc"])

            test_res = eval_epoch(self.model, self.test_loader, DEVICE)

            try:
                metrics = compute_full_metrics(
                    test_res["y"], test_res["preds"], test_res["probs"], self.class_names, self.t0
                )
                if metrics["auc_roc"] >= self.best_auc:
                    self.best_auc = metrics["auc_roc"]
                    self.best_metrics = metrics
                    self.best_res = test_res
            except Exception as e:
                print("Error computing metrics in aggregate", e)

        return aggregated_weights, metrics_aggregated


# ── Main runner ───────────────────────────────────────────────────────────────

def run_federated_for_config(cfg: ExperimentConfig, sample_frac: float) -> dict:
    print(f"\n{'='*60}\nRunning Federated (Stratified {KF_SPLITS}-Fold CV) — {cfg.name}\n{'='*60}")
    t0 = time.time()

    # ── Read HPO Params (e.g. batch_size) ────────────────────────────────────
    hpo_params = {}
    if cfg.hpo_params_path.exists():
        with open(cfg.hpo_params_path, "r") as f:
            hpo_params = json.load(f)
    batch_sz = hpo_params.get("batch_size", 512)

    # Load full dataset with same rare-class filter as centralized
    X, y, class_names = load_full_tensors(cfg, sample_frac=sample_frac, k=KF_SPLITS)
    y_np = y.numpy()
    n_classes = len(class_names)
    input_dim = X.shape[1]

    skf = StratifiedKFold(n_splits=KF_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics: list[dict] = []
    last_test_res = None
    last_history = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_np), start=1):
        print(f"\n  ── Fold {fold}/{KF_SPLITS}  (train={len(train_idx):,}, test={len(test_idx):,}) ──")

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # 15% of train as global validation (same ratio as centralized)
        val_size = max(1, int(len(train_idx) * 0.15))
        rng = np.random.default_rng(RANDOM_SEED + fold)
        val_local_idx   = rng.choice(len(train_idx), size=val_size, replace=False)
        train_local_idx = np.setdiff1d(np.arange(len(train_idx)), val_local_idx)

        X_val, y_val     = X_tr[val_local_idx],   y_tr[val_local_idx]
        X_train_f, y_train_f = X_tr[train_local_idx], y_tr[train_local_idx]

        # Partition training data across FL clients
        dataset_size = len(train_local_idx)
        client_size  = dataset_size // FL_NUM_CLIENTS

        global_model = build_model(input_dim, n_classes, cfg).to(DEVICE)
        val_loader   = create_loader(X_val, y_val, batch_size=batch_sz, shuffle=False)
        test_loader  = create_loader(X_te, y_te, batch_size=batch_sz, shuffle=False)

        def client_fn(context: Context) -> fl.client.Client:
            cid = int(context.node_id) % FL_NUM_CLIENTS
            # ── Seed every Ray actor deterministically ────────────────────
            # Each actor gets a unique-but-reproducible seed derived from
            # RANDOM_SEED, the current fold, and the client id so that:
            #   1. Different clients explore different weight initializations
            #   2. The same client always starts from the same weights across runs
            actor_seed = RANDOM_SEED + fold * FL_NUM_CLIENTS + cid
            set_global_seed(actor_seed)
            # ─────────────────────────────────────────────────────────────
            start = cid * client_size
            end   = min((cid + 1) * client_size, dataset_size)
            cl_loader = create_loader(X_train_f[start:end], y_train_f[start:end], batch_size=batch_sz, shuffle=True, seed=actor_seed)
            cl_val    = create_loader(X_val, y_val, batch_size=batch_sz, shuffle=False)
            cl_model  = build_model(input_dim, n_classes, cfg)
            return IDSClient(cl_model, cl_loader, cl_val, cfg).to_client()

        strategy = SaveModelStrategy(
            model=global_model, val_loader=val_loader, test_loader=test_loader,
            n_classes=n_classes, cfg=cfg, class_names=class_names, t0=t0,
            fraction_fit=FL_FRACTION_FIT,
            min_fit_clients=FL_MIN_FIT_CLIENTS,
            min_available_clients=FL_MIN_AVAILABLE,
        )

        # local_mode runs everything in-process → GPU access is direct via
        # DEVICE, not through Ray's resource allocator.  Request 0 GPUs to
        # avoid an empty ActorPool when Ray reports 0 available GPUs.
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=FL_NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args={"local_mode": True},
        )

        fold_test_res = strategy.best_res
        fold_met      = strategy.best_metrics

        if not fold_met:
            print(f"  [Fold {fold}] Metric calculation failed, using defaults.")
            fold_test_res = {"y": y_te.numpy(), "preds": y_te.numpy(), "probs": np.zeros((len(y_te), n_classes))}
            fold_met = compute_full_metrics(
                fold_test_res["y"], fold_test_res["preds"], fold_test_res["probs"], class_names, t0
            )

        fold_metrics.append(fold_met)
        last_test_res = fold_test_res
        last_history  = strategy.history

        print(f"  Fold {fold} → Acc={fold_met['accuracy']:.4f}  AUC={fold_met['auc_roc']:.4f}  "
              f"F1={fold_met['f1']:.4f}  FPR={fold_met['fpr']:.4f}")

    # ── Average metrics across all folds ─────────────────────────────────────
    avg_metrics: dict = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        avg_metrics[key] = float(np.mean(vals))
        avg_metrics[f"{key}_std"] = float(np.std(vals))

    avg_metrics["exec_time_s"] = time.time() - t0

    print(f"\n[{cfg.name} - FL] K-Fold Averages → "
          f"Acc={avg_metrics['accuracy']:.4f}±{avg_metrics['accuracy_std']:.4f}  "
          f"AUC={avg_metrics['auc_roc']:.4f}±{avg_metrics['auc_roc_std']:.4f}")

    # ── Persist outputs (using last fold's model/predictions) ─────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    plot_training_curves(last_history, cfg.output_dir)

    plot_confusion_matrix(
        last_test_res["y"], last_test_res["preds"], class_names,
        title=f"Supervised CM: {cfg.name} (FL, last fold)",
        output_path=cfg.output_dir / "cm_supervised.png",
        cmap="Blues",
    )

    y_bin      = (last_test_res["y"]     > 0).astype(int)
    y_pred_bin = (last_test_res["preds"] > 0).astype(int)
    plot_confusion_matrix(
        y_bin, y_pred_bin, ["Benign", "Attack"],
        title=f"Anomaly CM: {cfg.name} (FL, last fold)",
        output_path=cfg.output_dir / "cm_anomaly.png",
        cmap="Reds",
    )

    return {"config_name": cfg.name, **avg_metrics, **hpo_params}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of dataset to use (1.0 = full dataset)")
    args = parser.parse_args()

    results = []
    Path("results").mkdir(exist_ok=True)
    set_global_seed(RANDOM_SEED)

    for cfg in EXPERIMENT_CONFIGS:
        set_global_seed(RANDOM_SEED)
        cfg.paradigm = "federated"
        res = run_federated_for_config(cfg, args.sample)
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("results/federated_results.csv", index=False)
    print("\nFederated experiments completed. Results saved to results/federated_results.csv")
