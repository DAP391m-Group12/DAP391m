"""Canonical Minimal CUDA/PyTabKit preflight and two-epoch smoke for final RealMLP."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_b.models.neural.realmlp.estimator import PYTABKIT_VERSION, RealMLPClassifier
from src.utils.path import PROJECT_ROOT

from .final_contract import (
    RealMLPFinalPaths,
    base_manifest,
    begin_stage,
    fit_native_preprocessing,
    load_quality_gated_partition,
    publish_stage,
    read_json,
    validate_manifest,
    validate_probabilities,
    write_json,
)
from .final_training import _fit_model, _subset


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"


class RealMLPRuntimeGateError(RuntimeError):
    """Raised when the final RealMLP runtime is not CUDA-ready."""


class RealMLPSmokeOOM(RealMLPRuntimeGateError):
    """Retained for callers that need to distinguish a confirmed CUDA OOM."""


def collect_realmlp_runtime_environment(device_index: int = 0) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "pytabkit"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
        "device_index": device_index,
        "cuda_available": False,
        "cuda_finite_operation": False,
        "gpu_name": None,
        "ready_for_smoke": False,
        "errors": [],
    }
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"] and device_index < torch.cuda.device_count():
            tensor = torch.tensor([1.0], device=f"cuda:{device_index}") * 2.0
            result["cuda_finite_operation"] = bool(torch.isfinite(tensor).all().item())
            result["gpu_name"] = torch.cuda.get_device_name(device_index)
    except Exception as exc:  # pragma: no cover - environment-specific evidence
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    try:
        RealMLPClassifier._backend_class()
    except Exception as exc:  # pragma: no cover - environment-specific evidence
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    if packages["pytabkit"] != PYTABKIT_VERSION:
        result["errors"].append(
            f"Expected pytabkit=={PYTABKIT_VERSION}, found {packages['pytabkit']!r}."
        )
    result["ready_for_smoke"] = bool(
        result["cuda_available"]
        and result["cuda_finite_operation"]
        and not result["errors"]
    )
    return result


def run_realmlp_runtime_gate(
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    device_index: int = 0,
    batch_size: int = 256,
) -> dict[str, Any]:
    environment = collect_realmlp_runtime_environment(device_index)
    if not environment["ready_for_smoke"]:
        raise RealMLPRuntimeGateError(f"RealMLP runtime is not ready: {environment['errors']}")
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    temporary, destination = begin_stage(paths, "runtime")
    gated = load_quality_gated_partition("train")
    labels = gated.y.to_numpy(dtype=np.int8)
    approved = np.flatnonzero(labels == 1)[:512]
    denied = np.flatnonzero(labels == 0)[:512]
    sample = _subset(gated, np.concatenate((approved, denied)))
    preprocessing, prepared = fit_native_preprocessing(sample)
    _model, probabilities, history, elapsed = _fit_model(
        prepared.X,
        prepared.y,
        directory=temporary,
        stage="cuda_smoke",
        epochs=2,
        batch_size=batch_size,
        device_index=device_index,
        X_predict=prepared.X,
        X_val=None,
        y_val=None,
    )
    validate_probabilities(probabilities, len(prepared.X))
    manifest = {
        **base_manifest(run_id, "runtime"),
        "environment": environment,
        "selected_batch_size": batch_size,
        "smoke_rows": len(prepared.X),
        "completed_epochs": len(history),
        "elapsed_seconds": elapsed,
        "execution": {"fit_calls": 1, "predict_proba_calls": 1},
    }
    write_json(temporary / "manifest.json", manifest)
    publish_stage(temporary, destination)
    return validate_realmlp_runtime_gate(paths)


def validate_realmlp_runtime_gate(paths: RealMLPFinalPaths) -> dict[str, Any]:
    payload = validate_manifest(
        read_json(paths.stage("runtime") / "manifest.json"),
        run_id=paths.run_id,
        stage="runtime",
    )
    if payload.get("completed_epochs") != 2 or payload.get("environment", {}).get("ready_for_smoke") is not True:
        raise RealMLPRuntimeGateError("RealMLP two-epoch CUDA smoke is incomplete.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final RealMLP CUDA preflight/smoke.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    if args.preflight_only:
        result = collect_realmlp_runtime_environment(args.device_index)
    else:
        if not args.run_id:
            parser.error("--run-id is required unless --preflight-only is used")
        result = run_realmlp_runtime_gate(
            args.run_id,
            artifact_root=args.artifact_root,
            device_index=args.device_index,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
