import argparse
import pandas as pd
import numpy as np
import torch
import time
import json
import flwr as fl
from pathlib import Path
from collections import OrderedDict

from experiments.configs import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.data_loader import load_tensors, create_loader
from experiments.model import build_model
from experiments.trainer import train_epoch, eval_epoch, get_optimizer
from experiments.metrics import compute_full_metrics
from configs.settings import FL_NUM_CLIENTS, FL_NUM_ROUNDS, FL_LOCAL_EPOCHS, FL_FRACTION_FIT, FL_MIN_FIT_CLIENTS, FL_MIN_AVAILABLE
from core.visuals import plot_training_curves, plot_confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                metrics = compute_full_metrics(test_res["y"], test_res["preds"], test_res["probs"], self.class_names, self.t0)
                if metrics["auc_roc"] >= self.best_auc:
                    self.best_auc = metrics["auc_roc"]
                    self.best_metrics = metrics
                    self.best_res = test_res
            except Exception as e:
                print("Error computing metrics in aggregate", e)
                
        return aggregated_weights, metrics_aggregated

def run_federated_for_config(cfg: ExperimentConfig, sample_frac: float) -> dict:
    print(f"\n{'='*60}\nRunning Federated for {cfg.name}\n{'='*60}")
    t0 = time.time()
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_tensors(cfg, sample_frac)
    input_dim = X_train.shape[1]
    n_classes = len(class_names)
    
    dataset_size = len(X_train)
    client_size = dataset_size // FL_NUM_CLIENTS
    
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        start_idx = client_id * client_size
        end_idx = min((client_id + 1) * client_size, dataset_size)
        
        train_loader = create_loader(X_train[start_idx:end_idx], y_train[start_idx:end_idx], shuffle=True)
        val_loader = create_loader(X_val, y_val, shuffle=False)
        model = build_model(input_dim, n_classes, cfg)
        return IDSClient(model, train_loader, val_loader, cfg)
        
    global_model = build_model(input_dim, n_classes, cfg).to(DEVICE)
    val_loader = create_loader(X_val, y_val, shuffle=False)
    test_loader = create_loader(X_test, y_test, shuffle=False)
    
    strategy = SaveModelStrategy(
        model=global_model, val_loader=val_loader, test_loader=test_loader, 
        n_classes=n_classes, cfg=cfg, class_names=class_names, t0=t0,
        fraction_fit=FL_FRACTION_FIT,
        min_fit_clients=FL_MIN_FIT_CLIENTS,
        min_available_clients=FL_MIN_AVAILABLE,
    )
    
    client_resources = {"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0.0}
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    metrics = strategy.best_metrics
    test_res = strategy.best_res
    
    if not metrics:
        print("[Federated] Metric calculation failed, using defaults.")
        metrics = compute_full_metrics(y_test.numpy(), y_test.numpy(), np.zeros((len(y_test), n_classes)), class_names, t0)
        test_res = {"y": y_test.numpy(), "preds": y_test.numpy(), "probs": np.zeros((len(y_test), n_classes))}
        
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    plot_training_curves(strategy.history, cfg.output_dir)
    
    plot_confusion_matrix(
        test_res["y"], test_res["preds"], class_names,
        title=f"Supervised CM: {cfg.name} (FL)",
        output_path=cfg.output_dir / "cm_supervised.png",
        cmap="Blues"
    )
    
    y_bin = (test_res["y"] > 0).astype(int)
    y_pred_bin = (test_res["preds"] > 0).astype(int)
    plot_confusion_matrix(
        y_bin, y_pred_bin, ["Benign", "Attack"],
        title=f"Anomaly CM: {cfg.name} (FL)",
        output_path=cfg.output_dir / "cm_anomaly.png",
        cmap="Reds"
    )
    
    print(f"[{cfg.name} - FL] Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc_roc']:.4f}")
    return {"config_name": cfg.name, **metrics}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.3)
    args = parser.parse_args()
    
    results = []
    Path("results").mkdir(exist_ok=True)
    for cfg in EXPERIMENT_CONFIGS:
        cfg.paradigm = "federated"
        res = run_federated_for_config(cfg, args.sample)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv("results/federated_results.csv", index=False)
    print("\nFederated experiments completed. Results saved to results/federated_results.csv")
