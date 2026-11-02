"""Dataset B model namespace with lazy neural-backend exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PlainMLPClassifier",
    "RealMLPClassifier",
    "RealMLPConfigurationError",
    "load_realmlp_v1_config",
    "TabMMiniConfigurationError",
    "TabMMiniModel",
    "compute_pos_weight",
    "expand_member_targets",
    "load_tabmmini_v1_config",
    "mean_member_probabilities",
    "member_wise_weighted_bce",
]

_EXPORT_MODULES = {
    "PlainMLPClassifier": ".neural.plain_mlp.estimator",
    "RealMLPClassifier": ".neural.realmlp.estimator",
    "RealMLPConfigurationError": ".neural.realmlp.estimator",
    "load_realmlp_v1_config": ".neural.realmlp.estimator",
    "TabMMiniConfigurationError": ".neural.tabmmini.estimator",
    "TabMMiniModel": ".neural.tabmmini.estimator",
    "compute_pos_weight": ".neural.tabmmini.estimator",
    "expand_member_targets": ".neural.tabmmini.estimator",
    "load_tabmmini_v1_config": ".neural.tabmmini.estimator",
    "mean_member_probabilities": ".neural.tabmmini.estimator",
    "member_wise_weighted_bce": ".neural.tabmmini.estimator",
}


def __getattr__(name: str) -> Any:
    """Load a model export only when that public name is requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
