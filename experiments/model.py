import json
import torch
import torch.nn as nn
from typing import Optional, List, Type
from pathlib import Path
from experiments.configs import ExperimentConfig


# ── Activation helper ─────────────────────────────────────────────────────────
ACTIVATION_MAP: dict[str, Type[nn.Module]] = {
    "relu":      nn.ReLU,
    "gelu":      nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "silu":      nn.SiLU,
}


def _act(name: str) -> nn.Module:
    """Instantiate an activation by name (case-insensitive)."""
    return ACTIVATION_MAP.get(name.lower(), nn.ReLU)()


# ── Model definitions ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        n_classes: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), _act(activation), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), _act(activation),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self._act_out = _act(activation)

    def forward(self, x):
        return self._act_out(self.block(x) + self.proj(x))


class ResNetTabular(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        n_classes: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), _act(activation))
        
        blocks = []
        in_d = hidden_dims[0]
        for out_d in hidden_dims:
            blocks.append(_ResBlock(in_d, out_d, dropout, activation))
            in_d = out_d
            
        self.blocks = nn.Sequential(*blocks)
        self.head   = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class CNN1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        filters: List[int],
        kernels: List[int],
        dropout: float = 0.2,
        n_classes: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        layers = []
        in_c = 1
        for f, k in zip(filters, kernels):
            pad = k // 2
            layers += [nn.Conv1d(in_c, f, k, padding=pad), _act(activation)]
            in_c = f
            
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(filters[-1], n_classes))

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.head(z)


class AutoencoderClassifier(nn.Module):
    """Encoder-decoder with a classification head on the latent space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2,
        n_classes: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), _act(activation), nn.Dropout(dropout)]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim), nn.BatchNorm1d(latent_dim), _act(activation)]
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder (symmetric)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), _act(activation), nn.Dropout(dropout)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes),
        )

    def forward(self, x):
        latent         = self.encoder(x)
        reconstructed  = self.decoder(latent)
        logits         = self.classifier(latent)
        return logits, reconstructed, x


# ── Model factory (reads saved HPO params) ────────────────────────────────────

def build_model(input_dim: int, num_classes: int, cfg: ExperimentConfig) -> nn.Module:
    """Loads best HPO params and reconstructs the optimal model.

    Reseeds torch before construction so that layer weight initialisation
    is identical regardless of how much global RNG state was consumed
    by prior operations (other configs, data loading, etc.).
    """
    from configs.settings import RANDOM_SEED
    torch.manual_seed(RANDOM_SEED)

    if not cfg.hpo_params_path.exists():
        print(f"[{cfg.name}] Warning: HPO params not found at {cfg.hpo_params_path}. Using fallback MLP.")
        return MLP(input_dim, [256, 128, 64], 0.2, num_classes)

    with open(cfg.hpo_params_path, "r") as f:
        p = json.load(f)

    arch       = p.get("architecture", "mlp")
    dropout    = p.get("dropout", 0.2)
    activation = p.get("activation", "relu")

    if arch == "mlp":
        n_layers = p.get("mlp_n_layers", 3)
        # Reconstruct per-layer dims from individually saved parameters
        dims = [p.get(f"mlp_hidden_{i}", p.get("mlp_hidden", 128)) for i in range(n_layers)]
        return MLP(input_dim, dims, dropout, num_classes, activation)

    elif arch == "resnet":
        n_blks  = p.get("resnet_n_blocks", 2)
        dims = [p.get(f"resnet_hidden_{i}", p.get("resnet_hidden", 128)) for i in range(n_blks)]
        return ResNetTabular(input_dim, dims, dropout, num_classes, activation)

    elif arch == "cnn1d":
        n_layers = p.get("cnn_n_layers", 2)
        filters = [p.get(f"cnn_filters_{i}", p.get("cnn_filters", 64)) for i in range(n_layers)]
        kernels = [p.get(f"cnn_kernel_{i}", p.get("cnn_kernel", 3)) for i in range(n_layers)]
        return CNN1D(input_dim, filters, kernels, dropout, num_classes, activation)

    else:  # autoencoder
        n_layers = p.get("ae_n_layers", 1)
        dims = [p.get(f"ae_hidden_{i}", p.get("ae_hidden", 256)) for i in range(n_layers)]
        latent = p.get("ae_latent", 32)
        return AutoencoderClassifier(input_dim, dims, latent, dropout, num_classes, activation)


# ── Optimizer factory (reads saved HPO params) ────────────────────────────────

def get_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    if not cfg.hpo_params_path.exists():
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    with open(cfg.hpo_params_path, "r") as f:
        p = json.load(f)

    lr       = p.get("lr", 1e-3)
    wd       = p.get("weight_decay", 1e-4)
    opt_name = p.get("optimizer", "AdamW")

    if opt_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        momentum = p.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
