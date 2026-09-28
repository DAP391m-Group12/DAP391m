"""GPU compatibility and history-integration gate for Dataset B RealMLP."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.dataset_b.models import RealMLPClassifier
from src.dataset_b.preprocessing import (
    DatasetBSplitLoader,
    make_realmlp_native_preprocessing_pipeline,
)
from src.dataset_b.splits import validate_realmlp_pilot_artifacts
from src.utils.path import PROJECT_ROOT

from .realmlp_history import (
    HISTORY_METHOD,
    HistoryEnabledRealMLPClassifier,
    RealMLPHistoryRecorder,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "runtime"
DEFAULT_PILOT_DIR = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "pilot"
BATCH_CANDIDATES = (256, 128, 64)
REQUIRED_HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "inner_val_loss",
    "learning_rate",
    "epoch_time_seconds",
    "elapsed_time_seconds",
    "gpu_memory_used_mb",
    "best_epoch_so_far",
    "status",
    "timestamp",
)
_RUNTIME_FILENAMES = {
    "runtime_manifest.json",
    "gpu_compatibility.log",
    "smoke_result.json",
    "run.log",
    "training_history.jsonl",
    "training_history.json",
}
_EPOCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+val\s+"
    r"(?P<metric>[A-Za-z0-9_\-]+)\s*=\s*(?P<value>[-+0-9.eE]+)"
)


class RealMLPRuntimeGateError(RuntimeError):
    """Raised when the RealMLP runtime gate cannot safely execute or publish."""


class RealMLPSmokeOOM(RuntimeError):
    """Raised only for a confirmed CUDA out-of-memory smoke attempt."""


@dataclass(frozen=True)
class RealMLPSmokeExecution:
    """Result returned by one fresh RealMLP smoke attempt."""

    probabilities: np.ndarray
    history_records: tuple[dict[str, Any], ...]
    history_method: str
    raw_log: str
    metadata: dict[str, Any]


def collect_realmlp_runtime_environment(device_index: int = 0) -> dict[str, Any]:
    """Collect the locked GPU-COMPAT-01 environment and tensor evidence."""
    packages = {
        name: _package_version(name)
        for name in ("numpy", "pandas", "scikit-learn", "torch", "pytabkit")
    }
    result: dict[str, Any] = {
        "timestamp": _utc_now(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "operating_system": platform.platform(),
        "packages": packages,
        "device_index": device_index,
        "cuda_available": False,
        "cuda_build_version": None,
        "nvidia_driver_version": _nvidia_driver_version(device_index),
        "gpu_name": None,
        "total_vram_bytes": None,
        "finite_cuda_tensor_calculation": False,
        "pytabkit_estimator_import": False,
        "errors": [],
    }
    try:
        import torch
    except ImportError as exc:
        result["errors"].append(f"PyTorch import failed: {exc}")
        result["ready_for_smoke"] = False
        return result

    result["cuda_build_version"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    try:
        RealMLPClassifier._backend_class()
        result["pytabkit_estimator_import"] = True
    except RuntimeError as exc:
        result["errors"].append(str(exc))

    if not result["cuda_available"]:
        result["errors"].append("torch.cuda.is_available() is False.")
    elif device_index < 0 or device_index >= torch.cuda.device_count():
        result["errors"].append(
            f"CUDA device index {device_index} is outside 0..{torch.cuda.device_count() - 1}."
        )
    else:
        try:
            device = torch.device(f"cuda:{device_index}")
            properties = torch.cuda.get_device_properties(device)
            result["gpu_name"] = properties.name
            result["total_vram_bytes"] = int(properties.total_memory)
            values = torch.asarray([1.0, 2.0, 3.0], device=device)
            calculated = (values.square().sum() / values.sum()).detach()
            torch.cuda.synchronize(device)
            result["finite_cuda_tensor_calculation"] = bool(
                torch.isfinite(calculated).item()
            )
            if not result["finite_cuda_tensor_calculation"]:
                result["errors"].append("CUDA tensor calculation was not finite.")
        except Exception as exc:  # runtime-specific CUDA exceptions vary by torch build
            result["errors"].append(f"CUDA tensor probe failed: {type(exc).__name__}: {exc}")

    if packages["pytabkit"] != "1.7.3":
        result["errors"].append(
            f"Expected pytabkit==1.7.3, found {packages['pytabkit']!r}."
        )
    if not result["nvidia_driver_version"]:
        result["errors"].append("NVIDIA driver version could not be read with nvidia-smi.")
    result["ready_for_smoke"] = not result["errors"]
    return result


def run_realmlp_runtime_gate(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    pilot_dir: str | Path = DEFAULT_PILOT_DIR,
    device_index: int = 0,
    smoke_epochs: int = 2,
    _environment_probe: Callable[[int], dict[str, Any]] | None = None,
    _smoke_attempt: Callable[[int, Path], RealMLPSmokeExecution] | None = None,
) -> dict[str, Any]:
    """Run GPU-COMPAT-01 and publish immutable runtime evidence.

    Batch sizes are tried in the locked order 256, 128, 64. A lower batch size
    is attempted only after ``RealMLPSmokeOOM`` from the previous attempt.
    """
    if smoke_epochs != 2:
        raise RealMLPRuntimeGateError(
            "RMLP-04 requires exactly two epochs to prove two history records."
        )
    destination = Path(output_dir)
    if destination.exists():
        raise RealMLPRuntimeGateError(
            f"RealMLP runtime evidence already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _make_staging_directory(destination)

    started_at = _utc_now()
    environment_probe = _environment_probe or collect_realmlp_runtime_environment
    environment = environment_probe(device_index)
    attempts: list[dict[str, Any]] = []
    execution: RealMLPSmokeExecution | None = None
    selected_batch_size: int | None = None
    run_log_path = staging / "run.log"
    run_log_path.write_text("", encoding="utf-8")

    try:
        if environment.get("ready_for_smoke"):
            smoke_attempt = _smoke_attempt
            if smoke_attempt is None:
                executor = _RealSmokeExecutor(
                    pilot_dir=Path(pilot_dir),
                    device_index=device_index,
                    smoke_epochs=smoke_epochs,
                )
                smoke_attempt = executor.run

            for batch_size in BATCH_CANDIDATES:
                attempt_started = time.perf_counter()
                try:
                    execution = smoke_attempt(batch_size, run_log_path)
                    attempts.append(
                        {
                            "batch_size": batch_size,
                            "status": "PASS",
                            "elapsed_seconds": time.perf_counter() - attempt_started,
                            "error_type": None,
                            "error_message": None,
                        }
                    )
                    selected_batch_size = batch_size
                    break
                except RealMLPSmokeOOM as exc:
                    attempts.append(
                        {
                            "batch_size": batch_size,
                            "status": "OOM",
                            "elapsed_seconds": time.perf_counter() - attempt_started,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    continue
                except Exception as exc:
                    attempts.append(
                        {
                            "batch_size": batch_size,
                            "status": "FAILED",
                            "elapsed_seconds": time.perf_counter() - attempt_started,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    break

        probability_summary = _probability_summary(
            None if execution is None else execution.probabilities
        )
        history_records = [] if execution is None else list(execution.history_records)
        history_summary = _validate_history_records(history_records, smoke_epochs)
        history_method = "unavailable" if execution is None else execution.history_method
        smoke_passed = (
            execution is not None
            and probability_summary["valid"]
            and history_summary["complete"]
            and len(attempts) > 0
            and attempts[-1]["status"] == "PASS"
        )
        overall_status = (
            "PASS"
            if environment.get("ready_for_smoke") and smoke_passed
            else "NOT_READY"
        )

        _write_history(staging, history_records, history_method, history_summary)
        smoke_result = {
            "schema_version": 1,
            "contract_id": "GPU-COMPAT-01",
            "history_contract_id": "TRAINING-HISTORY-01",
            "status": "PASS" if smoke_passed else "NOT_READY",
            "smoke_epochs": smoke_epochs,
            "batch_candidates": list(BATCH_CANDIDATES),
            "selected_batch_size": selected_batch_size,
            "attempts": attempts,
            "probabilities": probability_summary,
            "history": {
                "method": history_method,
                **history_summary,
            },
            "execution_metadata": {} if execution is None else execution.metadata,
        }
        _write_json(staging / "smoke_result.json", smoke_result)
        (staging / "gpu_compatibility.log").write_text(
            _environment_log(environment, attempts), encoding="utf-8", newline="\n"
        )
        artifact_hashes = {
            path.name: _sha256_file(path)
            for path in staging.iterdir()
            if path.is_file() and path.name != "runtime_manifest.json"
        }
        runtime_manifest = {
            "schema_version": 1,
            "dataset": "dataset_b",
            "model": "realmlp",
            "artifact_version": "realmlp_v1",
            "contract_id": "GPU-COMPAT-01",
            "status": overall_status,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "environment": environment,
            "selected_batch_size": selected_batch_size,
            "smoke_status": smoke_result["status"],
            "artifact_sha256": artifact_hashes,
        }
        _write_json(staging / "runtime_manifest.json", runtime_manifest)
        _validate_runtime_artifacts(staging)
        os.replace(staging, destination)
        _validate_runtime_artifacts(destination)
        return runtime_manifest
    except OSError as exc:
        raise RealMLPRuntimeGateError(
            f"Cannot publish RealMLP runtime evidence: {destination}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class _RealSmokeExecutor:
    def __init__(self, *, pilot_dir: Path, device_index: int, smoke_epochs: int) -> None:
        self.pilot_dir = pilot_dir
        self.device_index = device_index
        self.smoke_epochs = smoke_epochs
        self.X_fit, self.y_fit, self.X_stop, self.y_stop, self.source_metadata = (
            self._load_native_inputs()
        )

    def _load_native_inputs(self):
        pilot_manifest = validate_realmlp_pilot_artifacts(self.pilot_dir)
        fit_ids = _read_id_csv(self.pilot_dir / "pilot_fit_ids.csv")
        stop_ids = _read_id_csv(self.pilot_dir / "pilot_inner_stop_ids.csv")
        train = DatasetBSplitLoader().load("train")
        positions = {record_id: index for index, record_id in enumerate(train.record_ids)}
        try:
            fit_positions = [positions[record_id] for record_id in fit_ids]
            stop_positions = [positions[record_id] for record_id in stop_ids]
        except KeyError as exc:
            raise RealMLPRuntimeGateError(
                "Pilot smoke IDs are not members of the immutable Train partition."
            ) from exc

        raw_fit = train.X.iloc[fit_positions].reset_index(drop=True)
        raw_stop = train.X.iloc[stop_positions].reset_index(drop=True)
        y_fit = train.y.iloc[fit_positions].to_numpy(dtype=np.int64)
        y_stop = train.y.iloc[stop_positions].to_numpy(dtype=np.int64)
        preprocessing = make_realmlp_native_preprocessing_pipeline()
        X_fit = preprocessing.fit_transform(raw_fit)
        X_stop = preprocessing.transform(raw_stop)
        metadata = {
            "pilot_manifest_sha256": _sha256_file(
                self.pilot_dir / "pilot_split_manifest.json"
            ),
            "pilot_fit_rows": len(fit_ids),
            "pilot_inner_stop_rows": len(stop_ids),
            "pilot_contract_id": pilot_manifest["contract_id"],
            "preprocessing": "dataset_b_neural_preprocessing_v1/realmlp_native",
        }
        return X_fit, y_fit, X_stop, y_stop, metadata

    def run(self, batch_size: int, run_log_path: Path) -> RealMLPSmokeExecution:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_index)
        attempt_history_path = (
            run_log_path.parent / f".training_history.batch_{batch_size}.jsonl"
        )
        history_recorder = RealMLPHistoryRecorder(
            attempt_history_path,
            device_index=self.device_index,
            expected_epochs=self.smoke_epochs,
        )
        estimator = HistoryEnabledRealMLPClassifier.from_yaml().set_params(
            device=f"cuda:{self.device_index}",
            n_epochs=self.smoke_epochs,
            batch_size=batch_size,
        )
        estimator.attach_history_recorder(history_recorder)
        started = time.perf_counter()
        captured = io.StringIO()
        try:
            with run_log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
                stdout_tee = _Tee(sys.stdout, log_handle, captured)
                stderr_tee = _Tee(sys.stderr, log_handle, captured)
                with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                    estimator.fit(
                        self.X_fit,
                        self.y_fit,
                        X_val=self.X_stop,
                        y_val=self.y_stop,
                    )
                    probabilities = estimator.predict_proba(self.X_stop)
        except Exception as exc:
            attempt_history_path.unlink(missing_ok=True)
            if _is_cuda_oom(exc):
                torch.cuda.empty_cache()
                raise RealMLPSmokeOOM(str(exc)) from exc
            raise

        history_path = run_log_path.parent / "training_history.jsonl"
        if history_path.exists():
            raise RealMLPRuntimeGateError(
                f"RealMLP completed history already exists: {history_path}"
            )
        os.replace(attempt_history_path, history_path)

        elapsed = time.perf_counter() - started
        history = list(history_recorder.records)
        peak_memory_mb = max(
            (record["gpu_memory_used_mb"] for record in history),
            default=torch.cuda.max_memory_allocated(self.device_index) / (1024 * 1024),
        )
        metadata = {
            **self.source_metadata,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_mb": peak_memory_mb,
            "requested_device": f"cuda:{self.device_index}",
            "runtime_metadata": estimator.runtime_metadata_,
            "fit_params": getattr(estimator.backend_, "fit_params_", None),
        }
        return RealMLPSmokeExecution(
            probabilities=probabilities,
            history_records=tuple(history),
            history_method=HISTORY_METHOD,
            raw_log=captured.getvalue(),
            metadata=metadata,
        )


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            if stream.closed:
                continue
            try:
                stream.flush()
            except ValueError:
                # Interpreter shutdown may close the underlying log before the tee.
                continue


def _extract_history(
    estimator: RealMLPClassifier, raw_log: str
) -> tuple[list[dict[str, Any]], str]:
    public_history = getattr(estimator.backend_, "history_", None)
    if (
        isinstance(public_history, Sequence)
        and not isinstance(public_history, (str, bytes))
        and all(isinstance(record, dict) for record in public_history)
    ):
        return [dict(record) for record in public_history], "public_backend_history"

    records: list[dict[str, Any]] = []
    for match in _EPOCH_PATTERN.finditer(raw_log):
        records.append(
            {
                "epoch": int(match.group("epoch")),
                "train_loss": None,
                "inner_val_loss": float(match.group("value")),
                "learning_rate": None,
                "epoch_time_seconds": None,
                "elapsed_time_seconds": None,
                "gpu_memory_used_mb": None,
                "best_epoch_so_far": None,
                "status": "INCOMPLETE",
                "timestamp": _utc_now(),
            }
        )
    return records, "terminal_output_parser"


def _validate_history_records(
    records: Sequence[dict[str, Any]],
    epochs: int,
    *,
    inner_validation_required: bool = True,
) -> dict[str, Any]:
    missing_by_epoch: dict[str, list[str]] = {}
    valid_records = 0
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_HISTORY_FIELDS if field not in record]
        for field in REQUIRED_HISTORY_FIELDS:
            if field in {
                "train_loss",
                "learning_rate",
                "epoch_time_seconds",
                "elapsed_time_seconds",
                "gpu_memory_used_mb",
            }:
                value = record.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    if field not in missing:
                        missing.append(field)
        if inner_validation_required:
            value = record.get("inner_val_loss")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                if "inner_val_loss" not in missing:
                    missing.append("inner_val_loss")
        elif (
            record.get("inner_val_loss") is not None
            or record.get("best_epoch_so_far") is not None
            or record.get("inner_val_status") != "NOT_APPLICABLE_FIXED_EPOCH"
            or record.get("best_epoch_status") != "NOT_APPLICABLE_FIXED_EPOCH"
        ):
            missing.append("fixed_epoch_not_applicable_contract")
        if record.get("status") != "COMPLETED" and "status" not in missing:
            missing.append("status")
        if missing:
            missing_by_epoch[str(record.get("epoch", index + 1))] = sorted(missing)
        else:
            valid_records += 1
    complete = len(records) == epochs and valid_records == epochs
    return {
        "complete": complete,
        "required_records": epochs,
        "observed_records": len(records),
        "valid_records": valid_records,
        "missing_fields_by_epoch": missing_by_epoch,
    }


def _probability_summary(probabilities: np.ndarray | None) -> dict[str, Any]:
    if probabilities is None:
        return {
            "valid": False,
            "rows": 0,
            "columns": 0,
            "finite": False,
            "within_unit_interval": False,
            "rows_sum_to_one": False,
            "minimum": None,
            "maximum": None,
        }
    values = np.asarray(probabilities, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    within = bool(finite and (values >= 0).all() and (values <= 1).all())
    sums = bool(
        values.ndim == 2
        and values.shape[1] == 2
        and np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    )
    valid = bool(values.ndim == 2 and values.shape[1] == 2 and finite and within and sums)
    return {
        "valid": valid,
        "rows": int(values.shape[0]) if values.ndim >= 1 else 0,
        "columns": int(values.shape[1]) if values.ndim == 2 else 0,
        "finite": finite,
        "within_unit_interval": within,
        "rows_sum_to_one": sums,
        "minimum": float(values.min()) if values.size and finite else None,
        "maximum": float(values.max()) if values.size and finite else None,
    }


def _write_history(
    directory: Path,
    records: Sequence[dict[str, Any]],
    method: str,
    summary: dict[str, Any],
) -> None:
    jsonl_path = directory / "training_history.jsonl"
    expected_lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        for record in records
    ]
    if jsonl_path.exists():
        observed_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        if observed_lines != expected_lines:
            raise RealMLPRuntimeGateError(
                "Live RealMLP history differs from the completed in-memory history."
            )
    else:
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in expected_lines:
                handle.write(line + "\n")
    _write_json(
        directory / "training_history.json",
        {"schema_version": 1, "method": method, **summary, "records": list(records)},
    )


def _validate_runtime_artifacts(directory: Path) -> None:
    actual = {path.name for path in directory.iterdir()}
    if actual != _RUNTIME_FILENAMES:
        raise RealMLPRuntimeGateError(
            "RealMLP runtime evidence must contain exactly the six required files."
        )
    for filename in ("runtime_manifest.json", "smoke_result.json", "training_history.json"):
        try:
            payload = json.loads((directory / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RealMLPRuntimeGateError(f"Invalid RealMLP runtime artifact: {filename}") from exc
        if not isinstance(payload, dict):
            raise RealMLPRuntimeGateError(f"RealMLP runtime artifact must be an object: {filename}")
    manifest = json.loads((directory / "runtime_manifest.json").read_text(encoding="utf-8"))
    for filename, expected_hash in manifest["artifact_sha256"].items():
        if _sha256_file(directory / filename) != expected_hash:
            raise RealMLPRuntimeGateError(f"RealMLP runtime artifact hash mismatch: {filename}")


def _environment_log(environment: dict[str, Any], attempts: Sequence[dict[str, Any]]) -> str:
    lines = [
        "Dataset B RealMLP GPU-COMPAT-01",
        f"timestamp={environment.get('timestamp')}",
        f"python={environment.get('python_version')}",
        f"python_executable={environment.get('python_executable')}",
        f"operating_system={environment.get('operating_system')}",
        f"packages={json.dumps(environment.get('packages'), sort_keys=True, default=_json_default)}",
        f"cuda_available={environment.get('cuda_available')}",
        f"cuda_build_version={environment.get('cuda_build_version')}",
        f"nvidia_driver_version={environment.get('nvidia_driver_version')}",
        f"gpu_name={environment.get('gpu_name')}",
        f"total_vram_bytes={environment.get('total_vram_bytes')}",
        f"finite_cuda_tensor_calculation={environment.get('finite_cuda_tensor_calculation')}",
        f"pytabkit_estimator_import={environment.get('pytabkit_estimator_import')}",
        f"ready_for_smoke={environment.get('ready_for_smoke')}",
        f"errors={json.dumps(environment.get('errors'), sort_keys=True, default=_json_default)}",
        f"attempts={json.dumps(list(attempts), sort_keys=True, default=_json_default)}",
    ]
    return "\n".join(lines) + "\n"


def _read_id_csv(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise RealMLPRuntimeGateError(f"Cannot read pilot ID artifact: {path}") from exc
    if not rows or rows[0] != ["record_id"] or any(len(row) != 1 for row in rows[1:]):
        raise RealMLPRuntimeGateError(f"Invalid pilot ID artifact: {path}")
    values = tuple(row[0] for row in rows[1:])
    if not values or len(values) != len(set(values)):
        raise RealMLPRuntimeGateError(f"Pilot ID artifact is empty or duplicated: {path}")
    return values


def _is_cuda_oom(exc: Exception) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _nvidia_driver_version(device_index: int) -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().splitlines()
    return value[0].strip() if value else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_staging_directory(destination: Path) -> Path:
    """Create an atomic staging directory without Windows owner-only temp ACLs."""
    for _ in range(10):
        candidate = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            continue
        return candidate
    raise RealMLPRuntimeGateError(
        f"Cannot allocate RealMLP runtime staging directory: {destination.parent}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dataset B RealMLP GPU and history integration smoke gate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Immutable runtime evidence directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=DEFAULT_PILOT_DIR,
        help=f"Validated RMLP-02 pilot artifacts (default: {DEFAULT_PILOT_DIR}).",
    )
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print GPU/package evidence without training or writing artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        environment = collect_realmlp_runtime_environment(args.device_index)
        print(json.dumps(environment, ensure_ascii=False, indent=2, default=_json_default))
        return 0 if environment["ready_for_smoke"] else 1
    try:
        manifest = run_realmlp_runtime_gate(
            output_dir=args.output_dir,
            pilot_dir=args.pilot_dir,
            device_index=args.device_index,
        )
    except RealMLPRuntimeGateError as exc:
        print(f"Dataset B RealMLP runtime gate failed: {exc}", file=sys.stderr)
        return 2
    print(f"Runtime evidence: {args.output_dir}")
    print(f"Status: {manifest['status']}")
    print(f"Selected batch size: {manifest['selected_batch_size']}")
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
