"""Compatibility access to the canonical RF training lifecycle module."""

from importlib import import_module
from typing import Any

_CANONICAL = import_module("src.dataset_b.models.classical.rf.training")
__all__ = [name for name in dir(_CANONICAL) if not name.startswith("_")]


def __getattr__(name: str) -> Any:
    return getattr(_CANONICAL, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_CANONICAL)))


if __name__ == "__main__":
    raise SystemExit(_CANONICAL.main())
