import argparse
import pandas as pd
import torch
import time
import json
from pathlib import Path

from experiments.configs import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.data_loader import load_tensors, create_loader
from experiments.model import build_model
from experiments.trainer import run_training, eval_epoch
from experiments.metrics import compute_full_metrics
from core.visuals import plot_training_curves, plot_confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_centralized_for_config(cfg: ExperimentConfig, sample_frac: float, epochs: int) -> dict:
    print(f"\n{'='*60}\nRunning Centralized for {cfg.name}\n{'='*60}")
    t0 = time.time()
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_tensors(cfg, sample_frac)
    
    input_dim = X_train.shape[1]
    n_classes = len(class_names)
    
    model = build_model(input_dim, n_classes, cfg).to(DEVICE)
    
    train_loader = create_loader(X_train, y_train, shuffle=True)
    val_loader = create_loader(X_val, y_val, shuffle=False)
    test_loader = create_loader(X_test, y_test, shuffle=False)
    
    history = run_training(model, train_loader, val_loader, cfg, epochs, DEVICE)
    
    test_res = eval_epoch(model, test_loader, DEVICE)
    
    metrics = compute_full_metrics(test_res["y"], test_res["preds"], test_res["probs"], class_names, t0)
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    plot_training_curves(history, cfg.output_dir)
    
    plot_confusion_matrix(
        test_res["y"], test_res["preds"], class_names,
        title=f"Supervised CM: {cfg.name}",
        output_path=cfg.output_dir / "cm_supervised.png",
        cmap="Blues"
    )
    
    y_bin = (test_res["y"] > 0).astype(int)
    y_pred_bin = (test_res["preds"] > 0).astype(int)
    plot_confusion_matrix(
        y_bin, y_pred_bin, ["Benign", "Attack"],
        title=f"Anomaly CM: {cfg.name}",
        output_path=cfg.output_dir / "cm_anomaly.png",
        cmap="Reds"
    )
    
    print(f"[{cfg.name}] Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc_roc']:.4f}")
    return {"config_name": cfg.name, **metrics}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sample", type=float, default=0.3)
    args = parser.parse_args()
    
    results = []
    Path("results").mkdir(exist_ok=True)
    for cfg in EXPERIMENT_CONFIGS:
        cfg.paradigm = "centralized"
        res = run_centralized_for_config(cfg, args.sample, args.epochs)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv("results/centralized_results.csv", index=False)
    print("\nCentralized experiments completed. Results saved to results/centralized_results.csv")
