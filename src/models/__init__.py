from src.models.registry import MODEL_REGISTRY, PrunableModel, build_model, register_model

# Importing the architecture modules populates MODEL_REGISTRY.
from src.models import cnn, mlp, mobilenet, opt, rescnn, resmlp, resnet, transformer, vit  # noqa: F401,E402

__all__ = ["MODEL_REGISTRY", "PrunableModel", "build_model", "register_model"]
