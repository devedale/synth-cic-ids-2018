import argparse
import json
import optuna
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from experiments.configs import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.data_loader import load_tensors, create_loader
from experiments.model import MLP, ResNetTabular, CNN1D, AutoencoderClassifier
from experiments.trainer import train_epoch, eval_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_trial_model(trial, input_dim: int, n_classes: int):
    arch = trial.suggest_categorical("architecture", ["mlp", "resnet", "cnn1d", "autoencoder"])
    
    if arch == "mlp":
        n_layers = trial.suggest_int("mlp_n_layers", 2, 4)
        hidden = trial.suggest_categorical("mlp_hidden", [64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        return MLP(input_dim, [hidden] * n_layers, dropout, n_classes)
        
    elif arch == "resnet":
        hidden = trial.suggest_categorical("resnet_hidden", [64, 128, 256])
        n_blocks = trial.suggest_int("resnet_n_blocks", 2, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        return ResNetTabular(input_dim, hidden, n_blocks, dropout, n_classes)
        
    elif arch == "cnn1d":
        n_filters = trial.suggest_categorical("cnn_filters", [32, 64, 128])
        kernel = trial.suggest_categorical("cnn_kernel", [3, 5])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        return CNN1D(input_dim, n_filters, kernel, dropout, n_classes)
        
    elif arch == "autoencoder":
        hidden = trial.suggest_categorical("ae_hidden", [128, 256])
        latent = trial.suggest_categorical("ae_latent", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        return AutoencoderClassifier(input_dim, hidden, latent, dropout, n_classes)

def create_objective(cfg: ExperimentConfig, X_train, y_train, X_val, y_val, n_classes, epochs):
    n_samples = len(X_train)
    input_dim = X_train.shape[1]
    
    def objective(trial):
        model = _build_trial_model(trial, input_dim, n_classes).to(DEVICE)
        
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        opt_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        
        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            momentum = trial.suggest_float("momentum", 0.8, 0.95)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_loader = create_loader(X_train, y_train, batch_size=512, shuffle=True)
        val_loader = create_loader(X_val, y_val, batch_size=512, shuffle=False)
        
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

def run_hpo_for_config(cfg: ExperimentConfig, trials: int, sample_frac: float, epochs: int):
    print(f"\n{'='*60}\nRunning HPO for {cfg.name}\n{'='*60}")
    
    X_train, X_val, _, y_train, y_val, _, class_names = load_tensors(cfg, sample_frac)
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        study_name=f"hpo_{cfg.name}"
    )
    
    objective = create_objective(cfg, X_train, y_train, X_val, y_val, len(class_names), epochs)
    
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    
    print(f"[{cfg.name}] Best AUC: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params_json = cfg.hpo_params_path
    
    print(f"Saving best params to {best_params_json}")
    best_params_json.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_json, 'w') as f:
        json.dump(best_params, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--sample", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config", type=str, default="ALL", help="Name of config, or ALL")
    args = parser.parse_args()
    
    for cfg in EXPERIMENT_CONFIGS:
        if args.config == "ALL" or cfg.name == args.config:
            run_hpo_for_config(cfg, args.trials, args.sample, args.epochs)
