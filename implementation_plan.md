# Experimental Framework: Centralized (A) + Federated (B)

## Goal

Build a systematic, reproducible framework that:
1. **Phase A** — Runs all 4 feature configurations under centralized training, producing full metrics + visuals
2. **Phase B** — Runs the same 4 configurations under Flower federated learning
3. **Report** — Aggregates all 8 results into a single comparative summary CSV + visual charts

The best HPO hyperparameters (from `scratch_hpo.py`) are frozen and reused across all experiments.

---

## Domain Architecture

```
configs/settings.py   →  single source of truth for ALL parameters (pipeline + ML + FL)
core/                 →  Spark pipeline: ingestion, preprocessing, academic visuals
experiments/          →  pure ML orchestration: PyTorch loops + Flower simulation
results/              →  pure output, no logic
```

**Dependency rule**: `experiments/` imports from `configs/` and `core/`, never the reverse.  
`core/` stays independent from `experiments/`.

---

## User Review Required

> [!IMPORTANT]
> **HPO Best Params**: `run_experiment_centralized.py` expects `models_cache/best_hpo_params.json`
> (produced by `scratch_hpo.py`). Falls back to a default 3-layer MLP if missing.
> Confirm whether to run HPO first or proceed with the default.

> [!IMPORTANT]
> **No-PCA Parquet**: "No-PCA" configs require a second Parquet with all original features
> (`_fullfeatures`), produced by running `main.py` with `PCA_FEATURE_SELECTION=False`.
> Without it, "No-PCA" would use the same 25 PCA-selected features, making the experiment
> meaningless. **Confirm whether to produce the full-features Parquet.**

> [!IMPORTANT]
> **Federated partitioning**: IID (random equal splits across clients) or non-IID (by
> attack day/type)? Non-IID is more realistic for federated IDS research.

---

## Proposed Changes

---

### `configs/settings.py` — Extension

#### [MODIFY] `configs/settings.py`

Adds TWO blocks at the **end** of the existing file. The `USE_PCA` and `USE_IP2VEC` flags
remain unchanged (used by `main.py`). Experimental runs override them at runtime via
`ExperimentConfig`, not by editing the file.

```python
# ---------------------------------------------------------
# EXPERIMENT FRAMEWORK — CACHE PATHS
# ---------------------------------------------------------
# Suffix for the full-features Parquet (no PCA reduction).
# Produced by running main.py with PCA_FEATURE_SELECTION=False.
FULLFEATURES_CACHE_SUFFIX = "_fullfeatures"

# ---------------------------------------------------------
# FEDERATED LEARNING SETTINGS (Flower)
# ---------------------------------------------------------
FL_NUM_CLIENTS     = 5      # Simulated clients
FL_NUM_ROUNDS      = 10     # Global aggregation rounds
FL_LOCAL_EPOCHS    = 3      # Local epochs per client per round
FL_FRACTION_FIT    = 1.0    # Fraction of clients selected per round
FL_MIN_FIT_CLIENTS = 3      # Minimum clients for training
FL_MIN_AVAILABLE   = 3      # Minimum clients that must be available
```

---

### `core/` — No Structural Changes

The `core/` domain stays **intact and independent**. No ML files are added here.

#### [MODIFY] `core/visuals.py`

Adds ONE function — `plot_metrics_radar` — because it is a generic academic visual,
not experiment-specific logic.

```python
def plot_metrics_radar(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Radar chart overlay of all 8 configs (4 Centralized + 4 Federated)."""
```

---

### `experiments/` — New Module (ML Only)

Pure ML orchestration only. No Spark logic, no config redefinition.

#### [NEW] `experiments/__init__.py`
Empty.

#### [NEW] `experiments/configs.py`

Single source of truth for the 4×2 matrix. Reads constants from `settings.py`,
does not redefine them. The `parquet_path` property resolves the correct Parquet dynamically.

```python
from dataclasses import dataclass
from pathlib import Path
from configs.settings import (
    CACHE_DIR, ML_CLASS_STRATEGY,
    FULLFEATURES_CACHE_SUFFIX, RANDOM_SEED
)

@dataclass
class ExperimentConfig:
    name: str
    use_ip2vec: bool
    use_pca: bool
    paradigm: str = "centralized"  # "centralized" | "federated"

    @property
    def parquet_path(self) -> Path:
        suffix = "" if self.use_pca else FULLFEATURES_CACHE_SUFFIX
        path = Path(CACHE_DIR) / f"final_preprocessed_{ML_CLASS_STRATEGY}{suffix}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"[ExperimentConfig] Parquet not found: {path}")
        return path

    @property
    def output_dir(self) -> Path:
        return Path("results") / self.name / self.paradigm

EXPERIMENT_CONFIGS = [
    ExperimentConfig("IP2Vec+PCA",     use_ip2vec=True,  use_pca=True),
    ExperimentConfig("IP2Vec+NoPCA",   use_ip2vec=True,  use_pca=False),
    ExperimentConfig("NoIP2Vec+PCA",   use_ip2vec=False, use_pca=True),
    ExperimentConfig("NoIP2Vec+NoPCA", use_ip2vec=False, use_pca=False),
]
```

#### [NEW] `experiments/data_loader.py`

Reads the **already-preprocessed** Parquet (output of `main.py`) and produces PyTorch tensors.
Does NOT replicate `core/dataset_loader.py` logic (which operates on live Spark DataFrames).

```python
def load_tensors(
    cfg: ExperimentConfig,
    sample_frac: float = 1.0,
    seed: int = RANDOM_SEED
) -> tuple:
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, class_names).

    Feature assembly:
    - use_pca=True  → PCA-selected numeric columns already in Parquet
    - use_pca=False → all numeric columns (full-features Parquet)
    - use_ip2vec=True  → appends ip2vec_embeddings (16-dim)
    - use_ip2vec=False → appends OHE of [Dst Port, Protocol, Src Region]
    """
```

#### [NEW] `experiments/model.py`

Single responsibility: build the MLP architecture from `models_cache/best_hpo_params.json`.

```python
def build_model(input_dim: int, num_classes: int) -> nn.Module:
    """Loads best_hpo_params.json → rebuilds optimal model.
    Falls back to 3-layer MLP if file is missing."""

def save_checkpoint(model: nn.Module, cfg: ExperimentConfig) -> None: ...
def load_checkpoint(model: nn.Module, cfg: ExperimentConfig) -> nn.Module: ...
```

#### [NEW] `experiments/metrics.py`

Pure metrics computation, no Spark or `core/` dependencies.

```python
def compute_full_metrics(
    y_true, y_pred, probs, class_names: list, t0: float
) -> dict:
    """Returns: accuracy, precision, recall, f1, fpr, fnr, auc_roc, exec_time_s."""
```

#### [NEW] `experiments/trainer.py`

Isolated PyTorch training loop. Separated from the runner script so `FlowerClient`
can reuse it without duplication.

```python
def train_epoch(model, loader, optimizer, criterion, device) -> float: ...
def eval_epoch(model, loader, criterion, device) -> dict: ...

def run_training(
    model, train_loader, val_loader,
    cfg: ExperimentConfig, epochs: int
) -> dict:
    """Returns history dict: {loss, val_loss, auc, val_auc} per epoch."""
```

---

### Entry Point Scripts — Root Level

Kept at `root/` for ease of use (same level as existing `scratch_hpo.py`).

#### [NEW] `run_experiment_centralized.py`

```bash
python run_experiment_centralized.py [--epochs 20] [--sample 0.5]
```

Iterates over `EXPERIMENT_CONFIGS` with `paradigm="centralized"`. Per config:
1. `experiments.data_loader.load_tensors(cfg)`
2. `experiments.model.build_model(input_dim, num_classes)`
3. `experiments.trainer.run_training(model, ...)`
4. `experiments.metrics.compute_full_metrics(...)`
5. Saves to `cfg.output_dir`: `metrics.json`, `cm_supervised.png`, `cm_anomaly.png`, `training_curves.pdf`
6. Appends row to `results/centralized_results.csv`

#### [NEW] `run_experiment_federated.py`

```bash
python run_experiment_federated.py [--rounds 10] [--clients 5] [--sample 0.5]
```

Flower architecture:
- **Server**: `FedAvg` strategy, parameters read from `configs/settings.FL_*`
- **Client**: `FlowerClient` that reuses `experiments.trainer` for local epochs
- Each config runs `fl.simulation.start_simulation()`

Output structurally identical to centralized.

#### [NEW] `run_report.py`

```bash
python run_report.py
```

Reads both result CSVs, produces:
- `results/full_comparison_report.csv` — 8 rows × all metrics
- `results/visuals/metrics_barchart_<metric>.pdf` — one per metric (accuracy, f1, fpr, fnr, auc_roc)
- `results/visuals/metrics_radar.pdf` — radar overlay of all 8 configs

Delegates rendering to `core/visuals.py`.

---

### `requirements.txt`

#### [MODIFY] `requirements.txt`

```
flwr>=1.8.0
torch>=2.0.0
optuna>=3.5.0
```

> [!NOTE]
> `mlflow` not included: adds infrastructure complexity (server, UI, artifact store)
> unnecessary for this scope. Results tracked as CSV + JSON, simpler to version with
> Git and include in the thesis.

---

### `results/` — Output Directory Structure

```
results/
├── centralized_results.csv
├── federated_results.csv
├── full_comparison_report.csv
├── IP2Vec+PCA/
│   ├── centralized/
│   │   ├── metrics.json
│   │   ├── cm_supervised.png
│   │   ├── cm_anomaly.png
│   │   └── training_curves.pdf
│   └── federated/
│       ├── metrics.json
│       ├── cm_supervised.png
│       ├── cm_anomaly.png
│       └── training_curves_rounds.pdf
├── IP2Vec+NoPCA/  ...
├── NoIP2Vec+PCA/  ...
├── NoIP2Vec+NoPCA/ ...
└── visuals/
    ├── metrics_barchart_accuracy.pdf
    ├── metrics_barchart_f1.pdf
    ├── metrics_barchart_fpr.pdf
    ├── metrics_barchart_fnr.pdf
    ├── metrics_barchart_auc_roc.pdf
    └── metrics_radar.pdf
```

---

## Open Questions

> [!IMPORTANT]
> **No-PCA Parquet**: Should we produce the full-features Parquet by re-running `main.py`
> with `PCA_FEATURE_SELECTION=False`? If not, "No-PCA" configs will use the same 25
> PCA-selected features and the only experimental variable will be IP2Vec vs OHE.

> [!IMPORTANT]
> **Federated partitioning**: IID (random equal splits) or non-IID (by day/attack type)?

---

## Execution Order

```bash
# Step 0: Already done — main.py produced final_preprocessed_undersample_majority.parquet

# Step 1 (optional, for meaningful No-PCA): produce full-features Parquet
# Edit settings.py: PCA_FEATURE_SELECTION=False, then:
python main.py

# Step 2: HPO for best params (if not already done)
python scratch_hpo.py --trials 30 --sample 0.3

# Step 3: Phase A — Centralized
python run_experiment_centralized.py --sample 0.5 --epochs 20

# Step 4: Phase B — Federated
python run_experiment_federated.py --rounds 10 --clients 5 --sample 0.5

# Step 5: Final report
python run_report.py
```

---

## Verification Plan

### Automated
- Each experiment script exits with non-zero code if `metrics.json` is missing
- `run_report.py` asserts both CSVs exist before generating visuals
- `ExperimentConfig.parquet_path` raises `FileNotFoundError` if Parquet is absent

### Manual Verification
- Inspect `results/full_comparison_report.csv` for completeness (8 rows)
- Verify confusion matrices render correctly for both supervised and anomaly views
- Check `results/visuals/metrics_radar.pdf` overlaps all 8 configs
