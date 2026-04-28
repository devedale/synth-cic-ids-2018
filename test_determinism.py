import pandas as pd
import numpy as np
from pathlib import Path
from experiments.configs import EXPERIMENT_CONFIGS
from experiments.data_loader import load_full_tensors

if __name__ == "__main__":
    cfg = EXPERIMENT_CONFIGS[0]
    X1, y1, _ = load_full_tensors(cfg, sample_frac=0.1, seed=42)
    X2, y2, _ = load_full_tensors(cfg, sample_frac=0.1, seed=42)
    
    print("X equal:", torch.equal(X1, X2) if hasattr(X1, "equal") else np.array_equal(X1, X2))
    print("y equal:", torch.equal(y1, y2) if hasattr(y1, "equal") else np.array_equal(y1, y2))
