"""Small explicit registries; importing this module registers no components."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable names to constructors or factory functions returning T."""

    def __init__(self, name: str = "registry") -> None:
        self.name = name
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registration name must be a nonempty string")

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self.name}: duplicate key {name!r}")
            if not callable(factory):
                raise ValueError(f"{self.name}: {name!r} must be callable")
            self._factories[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(f"{self.name}: unknown key {name!r}; available: {self.list()}") from None

    def build(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._factories)


MODEL_REGISTRY: Registry[Any] = Registry("models")
LOSS_REGISTRY: Registry[Any] = Registry("losses")
POSTPROC_REGISTRY: Registry[Any] = Registry("postprocessors")
