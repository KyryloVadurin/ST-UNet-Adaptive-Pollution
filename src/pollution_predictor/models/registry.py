from typing import Dict, Type, Any
import torch.nn as nn

class ModelRegistry:
    """
    Centralized registry for neural network architectures.
    Decouples architectural dependencies from the training logic.
    """
    # Internal storage for registered model classes
    _models: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator logic for architecture registration."""
        def wrapper(model_cls: Type[nn.Module]):
            cls._models[name.lower()] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """Factory method to instantiate registered architectures."""
        model_name = name.lower()
        
        # Validation of registry keys
        if model_name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available architectures: {available}")

        # Dynamic instantiation
        model_class = cls._models[model_name]
        return model_class(**kwargs)

# Global singleton instance for project-wide access
model_registry = ModelRegistry()