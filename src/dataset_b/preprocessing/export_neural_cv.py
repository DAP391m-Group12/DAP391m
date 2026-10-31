"""Compatibility access to the canonical neural workflow."""

from importlib import import_module
from typing import Any

_CANONICAL = import_module("src.dataset_b.workflows.export_neural_cv")
__all__ = [name for name in dir(_CANONICAL) if not name.startswith("_")]


def __getattr__(name: str) -> Any:
    return getattr(_CANONICAL, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_CANONICAL)))
