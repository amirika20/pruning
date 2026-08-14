from src.data.registry import DATASET_REGISTRY, DatasetBundle, build_dataset, register_dataset

# Importing the builder modules populates DATASET_REGISTRY.
from src.data import modular, synthetic, vision  # noqa: F401,E402

__all__ = ["DATASET_REGISTRY", "DatasetBundle", "build_dataset", "register_dataset"]
