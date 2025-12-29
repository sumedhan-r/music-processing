"""Model registry for managing and creating PyTorch models.

Provides a centralized registry for model architectures, making it easy
to experiment with different models via configuration.
"""

from typing import Any, Callable, Dict, Type
import torch.nn as nn


class ModelRegistry:
    """Registry for managing model architectures.

    Research use: Register and create models by name for easy experimentation.

    Example:
        >>> # Register a model
        >>> @ModelRegistry.register("my_model")
        ... class MyModel(nn.Module):
        ...     def __init__(self, num_classes=10):
        ...         super().__init__()
        ...         self.fc = nn.Linear(256, num_classes)
        ...
        ... def forward(self, x):
        ...     return self.fc(x)

        >>> # Create model by name
        >>> model = ModelRegistry.create("my_model", num_classes=5)

        >>> # List available models
        >>> print(ModelRegistry.list_models())
    """

    _models: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[nn.Module] = None,
    ) -> Callable:
        """Register a model architecture.

        Can be used as a decorator or called directly.

        Args:
            name: Model name for registry
            model_class: Model class to register (if not using as decorator)

        Returns:
            Decorator function or registered class

        Example:
            >>> # As decorator
            >>> @ModelRegistry.register("resnet18")
            ... class ResNet18(nn.Module):
            ...     pass

            >>> # Direct call
            >>> ModelRegistry.register("resnet18", ResNet18)
        """

        def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._models:
                print(f"Warning: Overwriting existing model '{name}' in registry")

            cls._models[name] = model_cls
            return model_cls

        # If model_class provided, register directly
        if model_class is not None:
            return decorator(model_class)

        # Otherwise return decorator
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Create a model instance by name.

        Args:
            name: Model name in registry
            **kwargs: Arguments to pass to model constructor

        Returns:
            Instantiated model

        Raises:
            KeyError: If model name not found in registry

        Example:
            >>> model = ModelRegistry.create("simple_cnn", num_classes=10)
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        model_class = cls._models[name]
        return model_class(**kwargs)

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Get model class by name (without instantiating).

        Args:
            name: Model name in registry

        Returns:
            Model class

        Example:
            >>> ModelClass = ModelRegistry.get("simple_cnn")
            >>> model = ModelClass(num_classes=10)
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names.

        Returns:
            List of registered model names

        Example:
            >>> models = ModelRegistry.list_models()
            >>> print(f"Available models: {models}")
        """
        return list(cls._models.keys())

    @classmethod
    def contains(cls, name: str) -> bool:
        """Check if model name is registered.

        Args:
            name: Model name to check

        Returns:
            True if model is registered

        Example:
            >>> if ModelRegistry.contains("resnet18"):
            ...     model = ModelRegistry.create("resnet18")
        """
        return name in cls._models

    @classmethod
    def remove(cls, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: Model name to remove

        Example:
            >>> ModelRegistry.remove("old_model")
        """
        if name in cls._models:
            del cls._models[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models.

        Example:
            >>> ModelRegistry.clear()
        """
        cls._models.clear()
