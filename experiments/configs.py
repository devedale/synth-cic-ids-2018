from dataclasses import dataclass
from pathlib import Path
from configs.settings import CACHE_DIR, ML_CLASS_STRATEGY, MODELS_DIR, RANDOM_SEED

@dataclass
class ExperimentConfig:
    name: str
    use_ip2vec: bool
    use_pca: bool
    paradigm: str = "centralized"  # "centralized" | "federated"

    @property
    def parquet_path(self) -> Path:
        """Single Parquet for all configs — full numeric feature set."""
        path = Path(CACHE_DIR) / f"final_preprocessed_{ML_CLASS_STRATEGY}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"[ExperimentConfig] Parquet not found: {path}")
        return path

    @property
    def pca_features_path(self) -> Path:
        """Sidecar JSON with PCA-selected feature names (used when use_pca=True)."""
        path = Path(MODELS_DIR) / "pca_selected_features.json"
        return path

    @property
    def hpo_params_path(self) -> Path:
        """Per-config best HPO params."""
        return Path(MODELS_DIR) / f"best_params_{self.name}.json"

    @property
    def output_dir(self) -> Path:
        return Path("results") / self.name / self.paradigm

EXPERIMENT_CONFIGS = [
    ExperimentConfig("IP2Vec+PCA",     use_ip2vec=True,  use_pca=True),
    ExperimentConfig("IP2Vec+NoPCA",   use_ip2vec=True,  use_pca=False),
    ExperimentConfig("NoIP2Vec+PCA",   use_ip2vec=False, use_pca=True),
    ExperimentConfig("NoIP2Vec+NoPCA", use_ip2vec=False, use_pca=False),
]
