import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiments.configs import ExperimentConfig
from experiments.model import get_optimizer

def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        logits = model(X)
        if isinstance(logits, tuple):
            logits = logits[0] # Handle models like Autoencoder returning tuple
            
        loss = F.cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_y = [], []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        logits = model(X)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
        
    logits_cat = torch.cat(all_logits)
    y_cat = torch.cat(all_y)
    
    probs = F.softmax(logits_cat, dim=1).numpy()
    preds = logits_cat.argmax(1).numpy()
    y_np = y_cat.numpy()
    
    # Calculate simple binary AUC for validation tracking
    y_bin = (y_np > 0).astype(int)
    from sklearn.metrics import roc_auc_score
    try:
        if probs.shape[1] > 1:
            auc = roc_auc_score(y_bin, 1.0 - probs[:, 0]) # P(Attack)
        else:
            auc = 0.5
    except:
        auc = 0.5
        
    return {
        "loss": total_loss / len(loader),
        "auc": auc,
        "preds": preds,
        "probs": probs,
        "y": y_np
    }

def run_training(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: ExperimentConfig, epochs: int, device: torch.device) -> dict:
    optimizer = get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {"loss": [], "val_loss": [], "auc": [], "val_auc": []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        val_res = eval_epoch(model, val_loader, device)
        
        history["loss"].append(train_loss)
        history["val_loss"].append(val_res["loss"])
        # We only have val_auc efficiently here
        history["val_auc"].append(val_res["auc"])
        
    return history
