from src.models.registry import MODEL_REGISTRY, PrunableModel, build_model, register_model

# Importing the architecture modules populates MODEL_REGISTRY.
from src.models import (  # noqa: F401,E402
    cnn, lenet, mlp, mobilenet, opt, rescnn, resmlp, resnet, transformer, vit,
)

__all__ = ["MODEL_REGISTRY", "PrunableModel", "build_model", "register_model"]
