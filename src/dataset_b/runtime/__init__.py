"""Compatibility façade for lazily loaded neural runtime exports."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RealMLPCVError": ("src.dataset_b.models.neural.realmlp.cv", "RealMLPCVError"),
    "run_realmlp_production_cv": ("src.dataset_b.models.neural.realmlp.cv", "run_realmlp_production_cv"),
    "validate_realmlp_production_cv": ("src.dataset_b.models.neural.realmlp.cv", "validate_realmlp_production_cv"),
    "RealMLPFullTrainError": ("src.dataset_b.models.neural.realmlp.full_train", "RealMLPFullTrainError"),
    "run_realmlp_full_train": ("src.dataset_b.models.neural.realmlp.full_train", "run_realmlp_full_train"),
    "validate_realmlp_full_train": ("src.dataset_b.models.neural.realmlp.full_train", "validate_realmlp_full_train"),
    "RealMLPRuntimeGateError": ("src.dataset_b.models.neural.realmlp.smoke", "RealMLPRuntimeGateError"),
    "RealMLPSmokeOOM": ("src.dataset_b.models.neural.realmlp.smoke", "RealMLPSmokeOOM"),
    "collect_realmlp_runtime_environment": ("src.dataset_b.models.neural.realmlp.smoke", "collect_realmlp_runtime_environment"),
    "run_realmlp_runtime_gate": ("src.dataset_b.models.neural.realmlp.smoke", "run_realmlp_runtime_gate"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
