#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Optimisation (HPO) runner using Optuna.

Search space covers: architecture, layer widths, depth, activation function,
dropout, optimizer, learning rate, weight decay, and batch size.
Best params per config are saved to models_cache/ and consumed by the
centralized and federated experiment runners.
"""
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import optuna
import torch
from pathlib import Path

from configs.settings import (
    RANDOM_SEED, HPO_N_TRIALS, HPO_SAMPLE_FRAC, HPO_EPOCHS,
    BATCH_SIZE, SPARK_DRIVER_MEMORY, set_global_seed
)
from experiments.configs import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.data_loader import load_tensors, create_loader
from experiments.model import MLP, ResNetTabular, CNN1D, AutoencoderClassifier
from experiments.trainer import train_epoch, eval_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Activation factory ────────────────────────────────────────────────────────
ACTIVATION_MAP = {
    "relu":      torch.nn.ReLU,
    "gelu":      torch.nn.GELU,
    "leakyrelu": torch.nn.LeakyReLU,
    "silu":      torch.nn.SiLU,
}


def _get_activation(name: str):
    """Return an activation module class by name."""
    return ACTIVATION_MAP.get(name, torch.nn.ReLU)


# ── Trial model builder ───────────────────────────────────────────────────────

def _build_trial_model(trial, input_dim: int, n_classes: int):
    """
    Suggest hyperparameters for a trial and build the corresponding model.
    Extended search space vs the previous version.
    """
    arch = trial.suggest_categorical(
        "architecture", ["mlp", "resnet", "cnn1d", "autoencoder"]
    )

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    activation = trial.suggest_categorical(
        "activation", ["relu", "gelu", "leakyrelu", "silu"]
    )

    if arch == "mlp":
        n_layers = trial.suggest_int("mlp_n_layers", 2, 6)
        hidden = trial.suggest_categorical(
            "mlp_hidden", [64, 128, 256, 512, 1024]
        )
        # Vary width per layer (wide-then-narrow or uniform)
        taper = trial.suggest_categorical("mlp_taper", [True, False])
        if taper:
            dims = [max(32, hidden // (2 ** i)) for i in range(n_layers)]
        else:
            dims = [hidden] * n_layers
        return MLP(input_dim, dims, dropout, n_classes, activation=activation)

    elif arch == "resnet":
        hidden = trial.suggest_categorical("resnet_hidden", [64, 128, 256, 512])
        n_blocks = trial.suggest_int("resnet_n_blocks", 2, 6)
        return ResNetTabular(input_dim, hidden, n_blocks, dropout, n_classes, activation=activation)

    elif arch == "cnn1d":
        n_filters = trial.suggest_categorical("cnn_filters", [32, 64, 128, 256])
        kernel = trial.suggest_categorical("cnn_kernel", [3, 5, 7])
        return CNN1D(input_dim, n_filters, kernel, dropout, n_classes, activation=activation)

    elif arch == "autoencoder":
        hidden = trial.suggest_categorical("ae_hidden", [128, 256, 512])
        latent = trial.suggest_categorical("ae_latent", [16, 32, 64, 128])
        return AutoencoderClassifier(input_dim, hidden, latent, dropout, n_classes, activation=activation)


# ── Objective factory ─────────────────────────────────────────────────────────

def create_objective(cfg: ExperimentConfig, X_train, y_train, X_val, y_val, n_classes, epochs):
    input_dim = X_train.shape[1]

    def objective(trial):
        model = _build_trial_model(trial, input_dim, n_classes).to(DEVICE)

        # Optimizer hyperparams
        lr        = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
        wd        = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        opt_name  = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
        batch_sz  = trial.suggest_categorical("batch_size", [256, 512, 1024])

        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            momentum  = trial.suggest_float("momentum", 0.7, 0.99)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loader = create_loader(X_train, y_train, batch_size=batch_sz, shuffle=True)
        val_loader   = create_loader(X_val,   y_val,   batch_size=batch_sz, shuffle=False)

        best_auc = 0.0
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, DEVICE)
            scheduler.step()
            val_metrics = eval_epoch(model, val_loader, DEVICE)
            auc = val_metrics["auc"]

            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_auc = max(best_auc, auc)

        return best_auc

    return objective


# ── Runner ────────────────────────────────────────────────────────────────────

def run_hpo_for_config(cfg: ExperimentConfig, trials: int, sample_frac: float, epochs: int):
    print(f"\n{'='*60}\nRunning HPO for {cfg.name}  ({trials} trials, sample={sample_frac}, epochs={epochs})\n{'='*60}")

    X_train, X_val, _, y_train, y_val, _, class_names = load_tensors(cfg, sample_frac)
    n_classes = len(class_names)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs, reduction_factor=3),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"hpo_{cfg.name}",
    )

    objective = create_objective(cfg, X_train, y_train, X_val, y_val, n_classes, epochs)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    print(f"\n[{cfg.name}] Best trial:")
    print(f"  AUC  : {study.best_value:.4f}")
    print(f"  Params: {json.dumps(study.best_params, indent=4)}")

    best_params_path = cfg.hpo_params_path
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"  Saved → {best_params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPO runner for CICIDS-2018 experiments")
    parser.add_argument("--trials",  type=int,   default=HPO_N_TRIALS,    help="Number of Optuna trials")
    parser.add_argument("--sample",  type=float, default=HPO_SAMPLE_FRAC, help="Fraction of dataset for HPO")
    parser.add_argument("--epochs",  type=int,   default=HPO_EPOCHS,      help="Training epochs per trial")
    parser.add_argument("--config",  type=str,   default="ALL",           help="Config name, or ALL")
    args = parser.parse_args()
    
    set_global_seed(RANDOM_SEED)

    for cfg in EXPERIMENT_CONFIGS:
        if args.config == "ALL" or cfg.name == args.config:
            run_hpo_for_config(cfg, args.trials, args.sample, args.epochs)
