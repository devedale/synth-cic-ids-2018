import json
import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from experiments.configs import ExperimentConfig

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        import torch.nn.functional as F
        return F.relu(self.block(x) + x)

class ResNetTabular(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_blocks: int, dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

class CNN1D(nn.Module):
    def __init__(self, input_dim: int, n_filters: int, kernel: int, dropout: float = 0.2, n_classes: int = 2):
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
        return self.head(z)

class AutoencoderClassifier(nn.Module):
    """Encoder-decoder with a classification head on the latent space."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        logits = self.classifier(latent)
        return logits, reconstructed, x


def build_model(input_dim: int, num_classes: int, cfg: ExperimentConfig) -> nn.Module:
    """Loads best hpo params, rebuilds optimal model."""
    if not cfg.hpo_params_path.exists():
        print(f"[{cfg.name}] Warning: HPO params not found at {cfg.hpo_params_path}. Using fallback MLP.")
        return MLP(input_dim, [256, 128, 64], 0.2, num_classes)
        
    with open(cfg.hpo_params_path, 'r') as f:
        p = json.load(f)
        
    arch = p.get("architecture", "mlp")
    
    if arch == "mlp":
        n_layers = p.get("mlp_n_layers", 3)
        hidden = p.get("mlp_hidden", 128)
        dropout = p.get("dropout", 0.2)
        return MLP(input_dim, [hidden] * n_layers, dropout, num_classes)
        
    elif arch == "resnet":
        hidden = p.get("resnet_hidden", 128)
        n_blocks = p.get("resnet_n_blocks", 2)
        dropout = p.get("dropout", 0.2)
        return ResNetTabular(input_dim, hidden, n_blocks, dropout, num_classes)
        
    elif arch == "cnn1d":
        n_filters = p.get("cnn_filters", 64)
        kernel = p.get("cnn_kernel", 3)
        dropout = p.get("dropout", 0.2)
        return CNN1D(input_dim, n_filters, kernel, dropout, num_classes)
        
    else:
        hidden = p.get("ae_hidden", 256)
        latent = p.get("ae_latent", 32)
        dropout = p.get("dropout", 0.2)
        return AutoencoderClassifier(input_dim, hidden, latent, dropout, num_classes)

def get_optimizer(model: nn.Module, cfg: ExperimentConfig):
    if not cfg.hpo_params_path.exists():
        return torch.optim.AdamW(model.parameters(), lr=1e-3)
        
    with open(cfg.hpo_params_path, 'r') as f:
        p = json.load(f)
        
    lr = p.get("lr", 1e-3)
    wd = p.get("weight_decay", 1e-4)
    opt_name = p.get("optimizer", "AdamW")
    
    if opt_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        momentum = p.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
