"""Train-only RealMLP pilot execution and post-pilot decision publication."""

from __future__ import annotations

import argparse
import contextlib
import csv
import html
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from src.dataset_b.models import load_realmlp_v1_config
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
from .realmlp_smoke import (
    _Tee,
    _json_default,
    _probability_summary,
    _read_id_csv,
    _sha256_file,
    _validate_history_records,
    _validate_runtime_artifacts,
    _write_history,
    _write_json,
    collect_realmlp_runtime_environment,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "pilot_run"
)
DEFAULT_PILOT_DIR = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "pilot"
)
DEFAULT_RUNTIME_DIR = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "runtime"
)
PILOT_EPOCHS = 256
_WORKING_FILENAMES = {
    "checkpoints/fitted_model.pkl",
    "pilot_result.json",
    "plots/pilot_history.svg",
    "predictions/pilot_inner_stop_predictions.csv",
    "run.log",
    "run_state.json",
    "training_history.json",
    "training_history.jsonl",
}
_FINAL_EXTRA_FILENAMES = {"COMPLETED", "decision.json", "manifest.json"}


class RealMLPPilotRunError(RuntimeError):
    """Raised when RMLP-05 cannot execute or publish safely."""


@dataclass(frozen=True)
class _PilotInputs:
    X_fit: Any
    y_fit: np.ndarray
    X_stop: Any
    y_stop: np.ndarray
    stop_ids: tuple[str, ...]
    preprocessing: Any
    metadata: dict[str, Any]


def run_realmlp_pilot(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    pilot_dir: str | Path = DEFAULT_PILOT_DIR,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    device_index: int = 0,
) -> Path:
    """Run all 256 pilot epochs and leave validated evidence awaiting a decision."""
    destination = Path(output_dir)
    working = _working_directory(destination)
    if destination.exists():
        raise RealMLPPilotRunError(
            f"Published RealMLP pilot run already exists: {destination}"
        )
    if working.exists():
        raise RealMLPPilotRunError(
            f"RealMLP pilot working directory already exists: {working}"
        )

    runtime_manifest = _load_runtime_gate(Path(runtime_dir), device_index)
    config = load_realmlp_v1_config()
    parameters = config["estimator_parameters"]
    if parameters["n_epochs"] != PILOT_EPOCHS:
        raise RealMLPPilotRunError("Frozen RealMLP pilot must use exactly 256 epochs.")
    if parameters["act"] != "selu" or parameters["use_early_stopping"] is not False:
        raise RealMLPPilotRunError(
            "Frozen RealMLP pilot must use SELU with early stopping disabled."
        )
    batch_size = runtime_manifest.get("selected_batch_size")
    if batch_size not in (256, 128, 64):
        raise RealMLPPilotRunError("Runtime gate has no valid frozen batch size.")

    inputs = _load_pilot_inputs(Path(pilot_dir))
    working.mkdir(parents=True)
    (working / "checkpoints").mkdir()
    (working / "predictions").mkdir()
    (working / "plots").mkdir()
    _write_run_state(
        working,
        "STARTED",
        output_dir=destination,
        message="RMLP-05 pilot initialized; Validation and Test are prohibited.",
    )
    (working / "run.log").write_text("", encoding="utf-8")

    try:
        result = _fit_pilot(
            working,
            inputs,
            device_index=device_index,
            batch_size=int(batch_size),
        )
        history_summary = _validate_history_records(result["history"], PILOT_EPOCHS)
        probability_summary = _probability_summary(result["probabilities"])
        if not history_summary["complete"]:
            raise RealMLPPilotRunError(
                "Pilot did not publish 256 complete training-history records."
            )
        if not probability_summary["valid"]:
            raise RealMLPPilotRunError("Pilot inner-stop probabilities are invalid.")

        _write_history(
            working,
            result["history"],
            HISTORY_METHOD,
            history_summary,
        )
        _write_predictions(
            working / "predictions" / "pilot_inner_stop_predictions.csv",
            inputs.stop_ids,
            result["probabilities"],
        )
        _write_history_svg(
            working / "plots" / "pilot_history.svg", result["history"]
        )
        pilot_result = {
            "schema_version": 1,
            "contract_id": "REALMLP-PILOT-EXECUTION-01",
            "status": "PASS",
            "evidence_scope": (
                "Train-only pilot tuning evidence; not CV, Validation, Test, or RQ1 evidence."
            ),
            "configuration": {
                "config_version": config["config_version"],
                "config_sha256": _sha256_file(
                    PROJECT_ROOT / "config" / "dataset_b" / "realmlp_v1.yaml"
                ),
                "activation": parameters["act"],
                "epochs_requested": PILOT_EPOCHS,
                "epochs_completed": len(result["history"]),
                "use_early_stopping": parameters["use_early_stopping"],
                "batch_size": int(batch_size),
                "device": f"cuda:{device_index}",
            },
            "data_boundary": {
                **inputs.metadata,
                "fit_membership": "pilot_fit_only",
                "inner_stop_usage": "validation_loss_and_post_fit_prediction_only",
                "validation_partition_accessed": False,
                "test_partition_accessed": False,
            },
            "history": {"method": HISTORY_METHOD, **history_summary},
            "probabilities": probability_summary,
            "training": result["metadata"],
        }
        _write_json(working / "pilot_result.json", pilot_result)
        _validate_working_artifacts(working)
        _write_run_state(
            working,
            "AWAITING_DECISION",
            output_dir=destination,
            message=(
                "All 256 epochs completed; lock FIXED_EPOCH or "
                "INNER_EARLY_STOPPING before publication."
            ),
        )
        _validate_working_artifacts(working)
        return working
    except Exception as exc:
        _write_run_state(
            working,
            "FAILED",
            output_dir=destination,
            message=f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, RealMLPPilotRunError):
            raise
        raise RealMLPPilotRunError(f"RealMLP pilot execution failed: {exc}") from exc


def finalize_realmlp_pilot_decision(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    decision: str,
    rationale: str,
    fixed_epochs: int | None = None,
    inner_stop_fraction: float | None = None,
    inner_stop_seed: int | None = None,
    additive_patience: int | None = None,
    multiplicative_patience: float | None = None,
) -> dict[str, Any]:
    """Lock one post-pilot policy, validate all evidence, and publish atomically."""
    destination = Path(output_dir)
    working = _working_directory(destination)
    if destination.exists():
        raise RealMLPPilotRunError(
            f"Published RealMLP pilot run already exists: {destination}"
        )
    _validate_working_artifacts(working, expected_state="AWAITING_DECISION")
    if not rationale.strip():
        raise RealMLPPilotRunError("Post-pilot decision requires a rationale.")

    history_payload = _read_json(working / "training_history.json")
    records = history_payload.get("records")
    if not isinstance(records, list) or len(records) != PILOT_EPOCHS:
        raise RealMLPPilotRunError("Post-pilot decision requires 256 history records.")
    diagnostics = _decision_diagnostics(records)
    decision_payload = _decision_payload(
        decision=decision,
        rationale=rationale.strip(),
        fixed_epochs=fixed_epochs,
        inner_stop_fraction=inner_stop_fraction,
        inner_stop_seed=inner_stop_seed,
        additive_patience=additive_patience,
        multiplicative_patience=multiplicative_patience,
        diagnostics=diagnostics,
        history_sha256=_sha256_file(working / "training_history.json"),
    )
    _write_json(working / "decision.json", decision_payload)
    _write_run_state(
        working,
        "COMPLETED",
        output_dir=destination,
        message=f"Post-pilot decision locked: {decision_payload['decision']}.",
    )

    artifact_hashes = {
        path.relative_to(working).as_posix(): _sha256_file(path)
        for path in sorted(working.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED"}
    }
    manifest = {
        "schema_version": 1,
        "dataset": "dataset_b",
        "model": "realmlp",
        "artifact_version": "realmlp_v1",
        "contract_id": "RMLP-05",
        "status": "PASS",
        "decision": decision_payload["decision"],
        "artifact_sha256": artifact_hashes,
    }
    _write_json(working / "manifest.json", manifest)
    (working / "COMPLETED").write_text("RMLP-05 PASS\n", encoding="utf-8")
    _validate_final_artifacts(working)
    os.replace(working, destination)
    _validate_final_artifacts(destination)
    return manifest


def validate_realmlp_pilot_run(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate a published RMLP-05 bundle without changing it."""
    return _validate_final_artifacts(Path(output_dir))


def _load_runtime_gate(runtime_dir: Path, device_index: int) -> dict[str, Any]:
    try:
        _validate_runtime_artifacts(runtime_dir)
        manifest = _read_json(runtime_dir / "runtime_manifest.json")
    except Exception as exc:
        raise RealMLPPilotRunError("RMLP-04 runtime evidence is invalid.") from exc
    if manifest.get("status") != "PASS":
        raise RealMLPPilotRunError("RMLP-04 runtime gate must have status PASS.")

    current = collect_realmlp_runtime_environment(device_index)
    recorded = manifest.get("environment", {})
    if not current.get("ready_for_smoke"):
        raise RealMLPPilotRunError(
            f"Current GPU environment is not ready: {current.get('errors')}"
        )
    identity_fields = (
        "python_executable",
        "packages",
        "device_index",
        "cuda_build_version",
        "gpu_name",
    )
    mismatches = [field for field in identity_fields if current.get(field) != recorded.get(field)]
    if mismatches:
        raise RealMLPPilotRunError(
            "Current environment differs from RMLP-04 evidence: " + ", ".join(mismatches)
        )
    return manifest


def _load_pilot_inputs(pilot_dir: Path) -> _PilotInputs:
    try:
        pilot_manifest = validate_realmlp_pilot_artifacts(pilot_dir)
        fit_ids = _read_id_csv(pilot_dir / "pilot_fit_ids.csv")
        stop_ids = _read_id_csv(pilot_dir / "pilot_inner_stop_ids.csv")
        train = DatasetBSplitLoader().load("train")
    except Exception as exc:
        raise RealMLPPilotRunError("Cannot validate the immutable Train-only pilot.") from exc

    positions = {record_id: index for index, record_id in enumerate(train.record_ids)}
    try:
        fit_positions = [positions[record_id] for record_id in fit_ids]
        stop_positions = [positions[record_id] for record_id in stop_ids]
    except KeyError as exc:
        raise RealMLPPilotRunError("Pilot IDs are not members of immutable Train.") from exc
    if set(fit_ids) & set(stop_ids):
        raise RealMLPPilotRunError("Pilot-fit and inner-stop membership overlaps.")

    raw_fit = train.X.iloc[fit_positions].reset_index(drop=True)
    raw_stop = train.X.iloc[stop_positions].reset_index(drop=True)
    y_fit = train.y.iloc[fit_positions].to_numpy(dtype=np.int64)
    y_stop = train.y.iloc[stop_positions].to_numpy(dtype=np.int64)
    preprocessing = make_realmlp_native_preprocessing_pipeline()
    X_fit = preprocessing.fit_transform(raw_fit)
    X_stop = preprocessing.transform(raw_stop)
    return _PilotInputs(
        X_fit=X_fit,
        y_fit=y_fit,
        X_stop=X_stop,
        y_stop=y_stop,
        stop_ids=stop_ids,
        preprocessing=preprocessing,
        metadata={
            "source_partition": "train",
            "pilot_contract_id": pilot_manifest["contract_id"],
            "pilot_manifest_sha256": _sha256_file(
                pilot_dir / "pilot_split_manifest.json"
            ),
            "pilot_fit_rows": len(fit_ids),
            "pilot_inner_stop_rows": len(stop_ids),
            "preprocessing": "dataset_b_neural_preprocessing_v1/realmlp_native",
        },
    )


def _fit_pilot(
    working: Path,
    inputs: _PilotInputs,
    *,
    device_index: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    _write_run_state(
        working,
        "TRAINING",
        output_dir=Path(str(working)[:-4]),
        message="Training frozen SELU pilot for all 256 epochs.",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    recorder = RealMLPHistoryRecorder(
        working / "training_history.jsonl",
        device_index=device_index,
        expected_epochs=PILOT_EPOCHS,
        stage="pilot",
    )
    estimator = HistoryEnabledRealMLPClassifier.from_yaml().set_params(
        device=f"cuda:{device_index}",
        batch_size=batch_size,
    )
    estimator.attach_history_recorder(recorder)
    started = time.perf_counter()
    with (working / "run.log").open("a", encoding="utf-8", newline="\n") as log_handle:
        stdout_tee = _Tee(sys.stdout, log_handle)
        stderr_tee = _Tee(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            estimator.fit(
                inputs.X_fit,
                inputs.y_fit,
                X_val=inputs.X_stop,
                y_val=inputs.y_stop,
            )
            _write_run_state(
                working,
                "PREDICTING",
                output_dir=Path(str(working)[:-4]),
                message="Generating pilot-inner-stop probabilities once after training.",
            )
            probabilities = estimator.predict_proba(inputs.X_stop)

    _write_run_state(
        working,
        "SAVING",
        output_dir=Path(str(working)[:-4]),
        message="Serializing fitted pilot model and validating round-trip predictions.",
    )
    model_path = working / "checkpoints" / "fitted_model.pkl"
    joblib.dump(
        {
            "estimator": estimator,
            "preprocessing": inputs.preprocessing,
            "scope": "Train-only RealMLP pilot diagnostic model; not a CV or final model.",
        },
        model_path,
        compress=3,
    )
    restored = joblib.load(model_path)
    restored_probabilities = restored["estimator"].predict_proba(inputs.X_stop)
    if not np.allclose(probabilities, restored_probabilities, rtol=0, atol=1e-7):
        raise RealMLPPilotRunError("Serialized pilot model failed prediction round-trip.")

    history = list(recorder.records)
    elapsed = time.perf_counter() - started
    peak_memory = max(record["gpu_memory_used_mb"] for record in history)
    return {
        "history": history,
        "probabilities": probabilities,
        "metadata": {
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_mb": peak_memory,
            "requested_device": f"cuda:{device_index}",
            "runtime_metadata": estimator.runtime_metadata_,
            "fit_params": getattr(estimator.backend_, "fit_params_", None),
            "model_round_trip": "PASS",
        },
    }


def _decision_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    losses = np.asarray([record["inner_val_loss"] for record in records], dtype=float)
    observed_index = int(np.argmin(losses))
    backend_best_epoch = int(records[-1]["best_epoch_so_far"])
    if backend_best_epoch < 1 or backend_best_epoch > len(records):
        raise RealMLPPilotRunError("Backend best epoch is outside completed history.")
    return {
        "completed_epochs": len(records),
        "backend_best_epoch": backend_best_epoch,
        "backend_best_inner_val_loss": float(losses[backend_best_epoch - 1]),
        "minimum_logged_inner_val_epoch": observed_index + 1,
        "minimum_logged_inner_val_loss": float(losses[observed_index]),
        "final_inner_val_loss": float(losses[-1]),
        "last_16_epoch_minimum_inner_val_loss": float(losses[-16:].min()),
        "best_epoch_distance_from_budget": len(records) - backend_best_epoch,
    }


def _decision_payload(
    *,
    decision: str,
    rationale: str,
    fixed_epochs: int | None,
    inner_stop_fraction: float | None,
    inner_stop_seed: int | None,
    additive_patience: int | None,
    multiplicative_patience: float | None,
    diagnostics: dict[str, Any],
    history_sha256: str,
) -> dict[str, Any]:
    normalized = decision.upper()
    common = {
        "schema_version": 1,
        "contract_id": "PILOT-DECISION-01",
        "status": "LOCKED",
        "decision": normalized,
        "rationale": rationale,
        "evidence_scope": (
            "Train-only pilot tuning evidence; not CV, Validation, Test, or RQ1 evidence."
        ),
        "evidence": {"training_history_sha256": history_sha256, **diagnostics},
    }
    if normalized == "FIXED_EPOCH":
        if (
            isinstance(fixed_epochs, bool)
            or not isinstance(fixed_epochs, int)
            or not 1 <= fixed_epochs <= PILOT_EPOCHS
        ):
            raise RealMLPPilotRunError("FIXED_EPOCH requires --fixed-epochs in 1..256.")
        if any(
            value is not None
            for value in (
                inner_stop_fraction,
                inner_stop_seed,
                additive_patience,
                multiplicative_patience,
            )
        ):
            raise RealMLPPilotRunError(
                "FIXED_EPOCH must not include inner early-stopping parameters."
            )
        common["five_fold_cv_policy"] = {
            "fit_membership": "all_outer_fold_train_rows",
            "inner_stop_set": "none",
            "n_epochs": fixed_epochs,
            "use_early_stopping": False,
        }
        common["full_train_refit_policy"] = {
            "fit_membership": "all_immutable_train_rows",
            "inner_stop_set": "none",
            "n_epochs": fixed_epochs,
            "use_early_stopping": False,
            "project_validation_for_epoch_selection": "prohibited",
        }
        return common

    if normalized == "INNER_EARLY_STOPPING":
        if fixed_epochs is not None:
            raise RealMLPPilotRunError(
                "INNER_EARLY_STOPPING must not include --fixed-epochs."
            )
        if (
            inner_stop_fraction is None
            or not 0 < inner_stop_fraction < 1
            or isinstance(inner_stop_seed, bool)
            or not isinstance(inner_stop_seed, int)
            or inner_stop_seed < 0
            or isinstance(additive_patience, bool)
            or not isinstance(additive_patience, int)
            or additive_patience < 0
            or multiplicative_patience is None
            or not math.isfinite(multiplicative_patience)
            or multiplicative_patience < 0
        ):
            raise RealMLPPilotRunError(
                "INNER_EARLY_STOPPING requires a fraction, seed, and non-negative "
                "PyTabKit additive/multiplicative patience values."
            )
        procedure = {
            "split_source": "fit_partition_only",
            "stratified": True,
            "inner_stop_fraction": inner_stop_fraction,
            "random_seed": inner_stop_seed,
            "max_epochs": PILOT_EPOCHS,
            "use_early_stopping": True,
            "early_stopping_additive_patience": additive_patience,
            "early_stopping_multiplicative_patience": multiplicative_patience,
        }
        common["five_fold_cv_policy"] = {
            **procedure,
            "fit_partition": "outer_fold_train_only",
            "outer_fold_validation_for_stopping": "prohibited",
        }
        common["full_train_refit_policy"] = {
            **procedure,
            "fit_partition": "immutable_train_only",
            "project_validation_for_stopping": "prohibited",
        }
        return common

    raise RealMLPPilotRunError(
        "Decision must be exactly FIXED_EPOCH or INNER_EARLY_STOPPING."
    )


def _write_predictions(path: Path, record_ids: Sequence[str], values: np.ndarray) -> None:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (len(record_ids), 2):
        raise RealMLPPilotRunError("Pilot prediction rows do not align with inner-stop IDs.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id", "probability_0", "probability_1"])
        for record_id, row in zip(record_ids, probabilities, strict=True):
            writer.writerow([record_id, repr(float(row[0])), repr(float(row[1]))])


def _write_history_svg(path: Path, records: Sequence[dict[str, Any]]) -> None:
    panels = (
        ("Loss", (("train_loss", "Train", "#2563eb"), ("inner_val_loss", "Inner stop", "#dc2626"))),
        ("Learning rate", (("learning_rate", "LR", "#7c3aed"),)),
        ("Epoch time (s)", (("epoch_time_seconds", "Epoch", "#059669"),)),
        ("Elapsed time (s)", (("elapsed_time_seconds", "Elapsed", "#d97706"),)),
        ("GPU memory (MB)", (("gpu_memory_used_mb", "GPU", "#0891b2"),)),
    )
    width, panel_height, left, right = 1100, 190, 85, 25
    plot_width = width - left - right
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_height * len(panels)}" viewBox="0 0 {width} {panel_height * len(panels)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2}</style>',
    ]
    epochs = [int(record["epoch"]) for record in records]
    for panel_index, (title, series) in enumerate(panels):
        top = panel_index * panel_height + 28
        bottom = panel_index * panel_height + panel_height - 35
        all_values = [float(record[field]) for field, _, _ in series for record in records]
        low, high = min(all_values), max(all_values)
        if math.isclose(low, high):
            high = low + 1.0
        chunks.extend(
            [
                f'<text x="{left}" y="{top - 8}" font-size="15" font-weight="600">{html.escape(title)}</text>',
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{(top+bottom)/2:.1f}" x2="{width-right}" y2="{(top+bottom)/2:.1f}"/>',
                f'<text x="8" y="{top + 5}" font-size="11">{high:.6g}</text>',
                f'<text x="8" y="{bottom + 4}" font-size="11">{low:.6g}</text>',
                f'<text x="{left}" y="{bottom + 20}" font-size="11">1</text>',
                f'<text x="{width-right-24}" y="{bottom + 20}" font-size="11">{epochs[-1]}</text>',
            ]
        )
        for series_index, (field, label, color) in enumerate(series):
            points = []
            for index, record in enumerate(records):
                x = left + (plot_width * index / max(1, len(records) - 1))
                y = bottom - (bottom - top) * (float(record[field]) - low) / (high - low)
                points.append(f"{x:.2f},{y:.2f}")
            chunks.append(
                f'<polyline class="line" stroke="{color}" points="{" ".join(points)}"/>'
            )
            legend_x = width - right - 150 + series_index * 80
            chunks.extend(
                [
                    f'<line x1="{legend_x}" y1="{top-10}" x2="{legend_x+18}" y2="{top-10}" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{legend_x+23}" y="{top-6}" font-size="11">{html.escape(label)}</text>',
                ]
            )
    chunks.append("</svg>\n")
    path.write_text("\n".join(chunks), encoding="utf-8")


def _write_run_state(
    directory: Path,
    state: str,
    *,
    output_dir: Path,
    message: str,
) -> None:
    previous = None
    path = directory / "run_state.json"
    if path.exists():
        try:
            previous = _read_json(path).get("state")
        except RealMLPPilotRunError:
            previous = "UNREADABLE"
    _write_json(
        path,
        {
            "schema_version": 1,
            "contract_id": "RMLP-05",
            "state": state,
            "previous_state": previous,
            "output_dir": str(output_dir.resolve()),
            "message": message,
        },
    )


def _validate_working_artifacts(
    directory: Path,
    *,
    expected_state: str | None = None,
) -> dict[str, Any]:
    if not directory.is_dir():
        raise RealMLPPilotRunError(f"Pilot working directory is missing: {directory}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != _WORKING_FILENAMES:
        raise RealMLPPilotRunError(
            f"Pilot working artifact set differs: {sorted(actual ^ _WORKING_FILENAMES)}"
        )
    state = _read_json(directory / "run_state.json")
    if expected_state is not None and state.get("state") != expected_state:
        raise RealMLPPilotRunError(
            f"Pilot working state must be {expected_state}, found {state.get('state')}."
        )
    result = _read_json(directory / "pilot_result.json")
    history = _read_json(directory / "training_history.json")
    if result.get("status") != "PASS" or not history.get("complete"):
        raise RealMLPPilotRunError("Pilot working evidence does not have PASS history.")
    return result


def _validate_final_artifacts(directory: Path) -> dict[str, Any]:
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    expected = _WORKING_FILENAMES | _FINAL_EXTRA_FILENAMES
    if actual != expected:
        raise RealMLPPilotRunError(
            f"Published pilot artifact set differs: {sorted(actual ^ expected)}"
        )
    manifest = _read_json(directory / "manifest.json")
    decision = _read_json(directory / "decision.json")
    state = _read_json(directory / "run_state.json")
    if (
        manifest.get("status") != "PASS"
        or decision.get("status") != "LOCKED"
        or state.get("state") != "COMPLETED"
        or manifest.get("decision") != decision.get("decision")
        or (directory / "COMPLETED").read_text(encoding="utf-8") != "RMLP-05 PASS\n"
    ):
        raise RealMLPPilotRunError("Published pilot completion state is invalid.")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise RealMLPPilotRunError("Published pilot manifest has no artifact hashes.")
    expected_hashed = expected - {"manifest.json", "COMPLETED"}
    if set(hashes) != expected_hashed:
        raise RealMLPPilotRunError("Published pilot manifest hash coverage is invalid.")
    for relative, expected_hash in hashes.items():
        if _sha256_file(directory / relative) != expected_hash:
            raise RealMLPPilotRunError(f"Published pilot hash mismatch: {relative}")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPPilotRunError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RealMLPPilotRunError(f"JSON artifact must be an object: {path}")
    return payload


def _working_directory(destination: Path) -> Path:
    return destination.with_name(destination.name + ".tmp")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and finalize the Train-only Dataset B RealMLP pilot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Train all 256 pilot epochs.")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    run_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    run_parser.add_argument("--device-index", type=int, default=0)

    decide_parser = subparsers.add_parser(
        "decide", help="Lock the post-pilot policy and publish the bundle."
    )
    decide_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    decide_parser.add_argument(
        "--decision", required=True, choices=("FIXED_EPOCH", "INNER_EARLY_STOPPING")
    )
    decide_parser.add_argument("--rationale", required=True)
    decide_parser.add_argument("--fixed-epochs", type=int)
    decide_parser.add_argument("--inner-stop-fraction", type=float)
    decide_parser.add_argument("--inner-stop-seed", type=int)
    decide_parser.add_argument("--additive-patience", type=int)
    decide_parser.add_argument("--multiplicative-patience", type=float)

    validate_parser = subparsers.add_parser("validate", help="Validate a published bundle.")
    validate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            working = run_realmlp_pilot(
                args.output_dir,
                pilot_dir=args.pilot_dir,
                runtime_dir=args.runtime_dir,
                device_index=args.device_index,
            )
            print(f"Pilot evidence awaiting decision: {working}")
            return 0
        if args.command == "decide":
            manifest = finalize_realmlp_pilot_decision(
                args.output_dir,
                decision=args.decision,
                rationale=args.rationale,
                fixed_epochs=args.fixed_epochs,
                inner_stop_fraction=args.inner_stop_fraction,
                inner_stop_seed=args.inner_stop_seed,
                additive_patience=args.additive_patience,
                multiplicative_patience=args.multiplicative_patience,
            )
            print(f"Published pilot evidence: {args.output_dir}")
            print(f"Decision: {manifest['decision']}")
            return 0
        manifest = validate_realmlp_pilot_run(args.output_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except RealMLPPilotRunError as exc:
        print(f"Dataset B RealMLP pilot failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
