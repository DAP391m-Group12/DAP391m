"""Production Train-only full-fit orchestration for Dataset B RealMLP."""

from __future__ import annotations

import argparse
import contextlib
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

from src.dataset_b.preprocessing import (
    DatasetBSplitLoader,
    make_realmlp_native_preprocessing_pipeline,
)
from src.utils.path import PROJECT_ROOT

from .realmlp_cv import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_CV_ROOT,
    validate_realmlp_production_cv,
)
from .realmlp_cv_smoke import (
    _archive_interrupted_working_directory,
    _atomic_rename_probe,
    _load_fixed_epoch_decision,
    _sha256_ids,
    _write_run_state,
)
from .realmlp_history import (
    HISTORY_METHOD,
    HistoryEnabledRealMLPClassifier,
    RealMLPHistoryRecorder,
)
from .realmlp_pilot_run import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PILOT_RUN_DIR,
    DEFAULT_RUNTIME_DIR,
    _load_runtime_gate,
)
from .realmlp_smoke import (
    _Tee,
    _json_default,
    _sha256_file,
    _validate_history_records,
    _write_history,
    _write_json,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "full_train"
)
EXPECTED_TRAIN_ROWS = 205_310
FIXED_EPOCHS = 51
CHECKPOINT_SCOPE = (
    "RMLP-07 Train-only full-fit state; no Validation or Test access."
)
_BASE_FILES = {
    "checkpoints/fitted_model.pkl",
    "full_train_result.json",
    "plots/full_train_history.svg",
    "run.log",
    "run_state.json",
    "training_history.json",
    "training_history.jsonl",
}
_FINAL_FILES = _BASE_FILES | {"COMPLETED", "manifest.json"}


class RealMLPFullTrainError(RuntimeError):
    """Raised when the RMLP-07A full-Train contract is violated."""


@dataclass(frozen=True)
class _FullTrainInputs:
    X_train: Any
    y_train: np.ndarray
    preprocessing: Any
    metadata: dict[str, Any]


def run_realmlp_full_train(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    pilot_run_dir: str | Path = DEFAULT_PILOT_RUN_DIR,
    cv_root: str | Path = DEFAULT_CV_ROOT,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    device_index: int = 0,
) -> dict[str, Any]:
    """Fit one fresh 51-epoch model on immutable Train and publish atomically."""
    destination = Path(output_dir)
    pilot_path = Path(pilot_run_dir)
    cv_path = Path(cv_root)
    runtime_path = Path(runtime_dir)
    if destination.exists():
        manifest = validate_realmlp_full_train(
            destination,
            pilot_run_dir=pilot_path,
            cv_root=cv_path,
            runtime_dir=runtime_path,
        )
        return {**manifest, "resume_action": "SKIPPED_VALID_COMPLETED"}

    contract = _load_run_contract(
        pilot_path,
        cv_path,
        runtime_path,
        device_index=device_index,
    )
    inputs = _load_full_train_inputs()
    root = destination.parent
    root.mkdir(parents=True, exist_ok=True)
    atomic_probe = _atomic_rename_probe(root)
    working = destination.with_name(destination.name + ".tmp")
    archived_attempt = _archive_interrupted_working_directory(working)
    working.mkdir()
    (working / "checkpoints").mkdir()
    (working / "plots").mkdir()
    (working / "run.log").write_text("", encoding="utf-8")
    _write_run_state(
        working,
        "STARTED",
        message="RMLP-07 Train-only full fit initialized.",
        contract_id="RMLP-07A",
    )

    try:
        execution = _fit_full_train(
            working,
            inputs,
            n_epochs=contract["n_epochs"],
            batch_size=contract["batch_size"],
            device_index=device_index,
        )
        history_summary = _validate_history_records(
            execution["history"],
            contract["n_epochs"],
            inner_validation_required=False,
        )
        if not history_summary["complete"]:
            raise RealMLPFullTrainError("Full-Train history is incomplete.")
        _write_history(
            working,
            execution["history"],
            HISTORY_METHOD,
            history_summary,
        )
        _write_full_train_history_svg(
            working / "plots" / "full_train_history.svg",
            execution["history"],
        )
        result = {
            "schema_version": 1,
            "contract_id": "REALMLP-FULL-TRAIN-01",
            "status": "PASS",
            "evidence_scope": (
                "Train-only fitted state; not Validation, Test, threshold, or RQ1 evidence."
            ),
            "configuration": {
                "decision": "FIXED_EPOCH",
                "n_epochs": contract["n_epochs"],
                "use_early_stopping": False,
                "use_best_epoch": False,
                "batch_size": contract["batch_size"],
                "device": f"cuda:{device_index}",
            },
            "lineage": contract["lineage"],
            "data_boundary": inputs.metadata,
            "history": {
                "method": HISTORY_METHOD,
                "inner_validation": "NOT_APPLICABLE_FIXED_EPOCH",
                **history_summary,
            },
            "execution": execution["metadata"],
            "atomic_rename_probe": atomic_probe,
            "archived_interrupted_attempt": archived_attempt,
        }
        _write_json(working / "full_train_result.json", result)
        _write_run_state(
            working,
            "COMPLETED",
            message="Exactly 51 Train-only epochs completed and fitted state saved.",
            contract_id="RMLP-07A",
        )
        _validate_working_files(working)
        hashes = {
            path.relative_to(working).as_posix(): _sha256_file(path)
            for path in sorted(working.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "COMPLETED"}
        }
        manifest = {
            "schema_version": 1,
            "dataset": "dataset_b",
            "model": "realmlp",
            "artifact_version": "realmlp_v1",
            "contract_id": "RMLP-07A",
            "status": "PASS",
            "train_rows": inputs.metadata["train_rows"],
            "train_record_ids_sha256": inputs.metadata["train_record_ids_sha256"],
            "fixed_epochs": contract["n_epochs"],
            "artifact_sha256": hashes,
        }
        _write_json(working / "manifest.json", manifest)
        (working / "COMPLETED").write_text("RMLP-07A PASS\n", encoding="utf-8")
        validate_realmlp_full_train(
            working,
            pilot_run_dir=pilot_path,
            cv_root=cv_path,
            runtime_dir=runtime_path,
        )
        os.replace(working, destination)
        return validate_realmlp_full_train(
            destination,
            pilot_run_dir=pilot_path,
            cv_root=cv_path,
            runtime_dir=runtime_path,
        )
    except KeyboardInterrupt:
        if working.exists():
            _write_run_state(
                working,
                "INTERRUPTED",
                message="Full-Train fit interrupted before atomic publication.",
                contract_id="RMLP-07A",
            )
        raise
    except Exception as exc:
        if working.exists():
            _write_run_state(
                working,
                "FAILED",
                message=f"{type(exc).__name__}: {exc}",
                contract_id="RMLP-07A",
            )
        if isinstance(exc, RealMLPFullTrainError):
            raise
        raise RealMLPFullTrainError(f"Production RealMLP full-Train failed: {exc}") from exc


def validate_realmlp_full_train(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    pilot_run_dir: str | Path = DEFAULT_PILOT_RUN_DIR,
    cv_root: str | Path = DEFAULT_CV_ROOT,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
) -> dict[str, Any]:
    """Validate an immutable RMLP-07A bundle without running training."""
    directory = Path(output_dir)
    actual = (
        {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if directory.is_dir()
        else set()
    )
    if actual != _FINAL_FILES:
        raise RealMLPFullTrainError(
            f"Full-Train artifact set differs: {sorted(actual ^ _FINAL_FILES)}"
        )
    manifest = _read_json(directory / "manifest.json")
    result = _read_json(directory / "full_train_result.json")
    state = _read_json(directory / "run_state.json")
    marker = (directory / "COMPLETED").read_text(encoding="utf-8")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset") != "dataset_b"
        or manifest.get("model") != "realmlp"
        or manifest.get("artifact_version") != "realmlp_v1"
        or manifest.get("contract_id") != "RMLP-07A"
        or manifest.get("status") != "PASS"
        or result.get("contract_id") != "REALMLP-FULL-TRAIN-01"
        or result.get("status") != "PASS"
        or state.get("contract_id") != "RMLP-07A"
        or state.get("state") != "COMPLETED"
        or marker != "RMLP-07A PASS\n"
    ):
        raise RealMLPFullTrainError("Full-Train completion state or marker is invalid.")

    train = _load_immutable_train()
    train_ids = tuple(train.record_ids.astype(str).tolist())
    train_id_hash = _sha256_ids(train_ids)
    configuration = result.get("configuration", {})
    boundary = result.get("data_boundary", {})
    execution = result.get("execution", {})
    runtime_metadata = execution.get("runtime_metadata", {})
    runtime_manifest = _read_json(Path(runtime_dir) / "runtime_manifest.json")
    expected_batch_size = runtime_manifest.get("selected_batch_size")
    device = configuration.get("device")
    if (
        manifest.get("fixed_epochs") != FIXED_EPOCHS
        or configuration.get("decision") != "FIXED_EPOCH"
        or configuration.get("n_epochs") != FIXED_EPOCHS
        or configuration.get("use_early_stopping") is not False
        or configuration.get("use_best_epoch") is not False
        or not isinstance(expected_batch_size, int)
        or expected_batch_size <= 0
        or configuration.get("batch_size") != expected_batch_size
        or not isinstance(device, str)
        or not device.startswith("cuda:")
        or runtime_metadata.get("validation_source") != "none"
        or runtime_metadata.get("validation_rows") != 0
        or runtime_metadata.get("use_best_epoch") is not False
        or runtime_metadata.get("requested_device") != device
        or runtime_metadata.get("allow_silent_cpu_fallback") is not False
        or runtime_metadata.get("backend_val_fraction") != 0.0
        or execution.get("fit_calls") != 1
        or execution.get("predict_proba_calls") != 0
        or execution.get("fit_validation_arguments") != {"X_val": None, "y_val": None}
        or execution.get("fit_params") != {"stop_epoch": {"cross_entropy": FIXED_EPOCHS}}
        or execution.get("model_deserialization") != "PASS_NO_PREDICTION"
    ):
        raise RealMLPFullTrainError("Full-Train fixed-epoch/no-validation contract is invalid.")
    if (
        len(train_ids) != EXPECTED_TRAIN_ROWS
        or len(set(train_ids)) != EXPECTED_TRAIN_ROWS
        or manifest.get("train_rows") != EXPECTED_TRAIN_ROWS
        or manifest.get("train_record_ids_sha256") != train_id_hash
        or boundary.get("source_partition") != "train"
        or boundary.get("train_rows") != EXPECTED_TRAIN_ROWS
        or boundary.get("unique_train_record_ids") != EXPECTED_TRAIN_ROWS
        or boundary.get("train_record_ids_sha256") != train_id_hash
        or boundary.get("preprocessing_fit_scope") != "all_immutable_train_rows"
        or boundary.get("model_fit_scope") != "all_immutable_train_rows"
        or boundary.get("inner_stop_set") != "none"
        or boundary.get("project_validation_accessed") is not False
        or boundary.get("project_test_accessed") is not False
        or boundary.get("category_support_failures") != 0
        or runtime_metadata.get("train_rows") != EXPECTED_TRAIN_ROWS
    ):
        raise RealMLPFullTrainError("Full-Train cardinality or Train-only boundary is invalid.")

    expected_lineage = _lineage_hashes(
        Path(pilot_run_dir), Path(cv_root), Path(runtime_dir)
    )
    if result.get("lineage") != expected_lineage:
        raise RealMLPFullTrainError("Full-Train upstream lineage hashes are invalid.")
    hashes = manifest.get("artifact_sha256")
    expected_hashed = _FINAL_FILES - {"manifest.json", "COMPLETED"}
    if not isinstance(hashes, dict) or set(hashes) != expected_hashed:
        raise RealMLPFullTrainError("Full-Train manifest hash coverage is invalid.")
    for relative, expected_hash in hashes.items():
        if _sha256_file(directory / relative) != expected_hash:
            raise RealMLPFullTrainError(f"Full-Train artifact hash mismatch: {relative}")

    history = _read_json(directory / "training_history.json")
    summary = _validate_history_records(
        history.get("records", []), FIXED_EPOCHS, inner_validation_required=False
    )
    expected_history = {
        "method": HISTORY_METHOD,
        "inner_validation": "NOT_APPLICABLE_FIXED_EPOCH",
        **summary,
    }
    if (
        not summary["complete"]
        or history.get("schema_version") != 1
        or history.get("method") != HISTORY_METHOD
        or result.get("history") != expected_history
        or any(
            history.get(key) != value for key, value in summary.items()
        )
    ):
        raise RealMLPFullTrainError("Full-Train history is incomplete or inconsistent.")
    jsonl_records = _read_jsonl(directory / "training_history.jsonl")
    if jsonl_records != history.get("records"):
        raise RealMLPFullTrainError("Full-Train JSON and JSONL history differ.")
    svg = (directory / "plots" / "full_train_history.svg").read_text(
        encoding="utf-8"
    )
    if not svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"') or not svg.endswith(
        "</svg>\n"
    ):
        raise RealMLPFullTrainError("Full-Train history SVG is invalid.")
    _validate_checkpoint(directory / "checkpoints" / "fitted_model.pkl")
    return manifest


def _load_run_contract(
    pilot_run_dir: Path,
    cv_root: Path,
    runtime_dir: Path,
    *,
    device_index: int,
) -> dict[str, Any]:
    try:
        decision = _load_fixed_epoch_decision(pilot_run_dir)
        validate_realmlp_production_cv(cv_root)
        runtime = _load_runtime_gate(runtime_dir, device_index)
    except Exception as exc:
        raise RealMLPFullTrainError("RMLP-04/05/06B prerequisite evidence is invalid.") from exc
    policy = decision.get("full_train_refit_policy")
    expected_policy = {
        "fit_membership": "all_immutable_train_rows",
        "inner_stop_set": "none",
        "n_epochs": FIXED_EPOCHS,
        "project_validation_for_epoch_selection": "prohibited",
        "use_early_stopping": False,
    }
    batch_size = runtime.get("selected_batch_size")
    if policy != expected_policy or not isinstance(batch_size, int) or batch_size <= 0:
        raise RealMLPFullTrainError("Locked full-Train policy or batch size is invalid.")
    return {
        "n_epochs": FIXED_EPOCHS,
        "batch_size": batch_size,
        "lineage": _lineage_hashes(pilot_run_dir, cv_root, runtime_dir),
    }


def _load_immutable_train():
    try:
        train = DatasetBSplitLoader().load("train")
    except Exception as exc:
        raise RealMLPFullTrainError("Cannot load the immutable Train partition.") from exc
    ids = tuple(train.record_ids.astype(str).tolist())
    if (
        len(ids) != EXPECTED_TRAIN_ROWS
        or len(set(ids)) != EXPECTED_TRAIN_ROWS
        or len(train.X) != EXPECTED_TRAIN_ROWS
        or len(train.y) != EXPECTED_TRAIN_ROWS
    ):
        raise RealMLPFullTrainError("Immutable Train cardinality differs from RMLP-07A.")
    return train


def _load_full_train_inputs() -> _FullTrainInputs:
    train = _load_immutable_train()
    try:
        preprocessing = make_realmlp_native_preprocessing_pipeline()
        X_train = preprocessing.fit_transform(train.X.reset_index(drop=True))
    except Exception as exc:
        raise RealMLPFullTrainError("Cannot fit fresh full-Train preprocessing.") from exc
    support = preprocessing.named_steps["native"].last_category_support_
    outcomes = [
        category["outcome"]
        for column in support.values()
        for category in column.values()
    ]
    ids = tuple(train.record_ids.astype(str).tolist())
    failures = outcomes.count("fail_absent_train_present_validation")
    if failures:
        raise RealMLPFullTrainError("Full-Train category support validation failed.")
    return _FullTrainInputs(
        X_train=X_train,
        y_train=train.y.to_numpy(dtype=np.int64),
        preprocessing=preprocessing,
        metadata={
            "source_partition": "train",
            "train_rows": len(ids),
            "unique_train_record_ids": len(set(ids)),
            "train_record_ids_sha256": _sha256_ids(ids),
            "preprocessing_fit_scope": "all_immutable_train_rows",
            "model_fit_scope": "all_immutable_train_rows",
            "inner_stop_set": "none",
            "project_validation_accessed": False,
            "project_test_accessed": False,
            "category_support_checked_cells": len(outcomes),
            "category_support_absent_both_warnings": outcomes.count(
                "warning_absent_both"
            ),
            "category_support_failures": failures,
        },
    )


def _fit_full_train(
    directory: Path,
    inputs: _FullTrainInputs,
    *,
    n_epochs: int,
    batch_size: int,
    device_index: int,
) -> dict[str, Any]:
    import torch

    _write_run_state(
        directory,
        "TRAINING",
        message=f"Fitting all immutable Train rows for exactly {n_epochs} epochs.",
        contract_id="RMLP-07A",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    recorder = RealMLPHistoryRecorder(
        directory / "training_history.jsonl",
        device_index=device_index,
        expected_epochs=n_epochs,
        stage="full_train",
    )
    estimator = HistoryEnabledRealMLPClassifier.from_yaml().set_params(
        device=f"cuda:{device_index}",
        n_epochs=n_epochs,
        batch_size=batch_size,
        use_early_stopping=False,
        use_best_epoch=False,
    )
    estimator.attach_history_recorder(recorder)
    started = time.perf_counter()
    with (directory / "run.log").open("a", encoding="utf-8", newline="\n") as log_handle:
        stdout_tee = _Tee(sys.stdout, log_handle)
        stderr_tee = _Tee(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            estimator.fit(
                inputs.X_train,
                inputs.y_train,
                X_val=None,
                y_val=None,
            )
    _write_run_state(
        directory,
        "SAVING",
        message="Saving fitted Train-only estimator and preprocessing state.",
        contract_id="RMLP-07A",
    )
    checkpoint = directory / "checkpoints" / "fitted_model.pkl"
    joblib.dump(
        {
            "estimator": estimator,
            "preprocessing": inputs.preprocessing,
            "scope": CHECKPOINT_SCOPE,
        },
        checkpoint,
        compress=3,
    )
    _validate_checkpoint(checkpoint)
    history = list(recorder.records)
    return {
        "history": history,
        "metadata": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": max(
                record["gpu_memory_used_mb"] for record in history
            ),
            "runtime_metadata": estimator.runtime_metadata_,
            "fit_params": getattr(estimator.backend_, "fit_params_", None),
            "fit_calls": 1,
            "predict_proba_calls": 0,
            "fit_validation_arguments": {"X_val": None, "y_val": None},
            "model_deserialization": "PASS_NO_PREDICTION",
        },
    }


def _write_full_train_history_svg(
    path: Path, records: Sequence[dict[str, Any]]
) -> None:
    panels = (
        ("Training loss", "train_loss", "#2563eb"),
        ("Learning rate", "learning_rate", "#7c3aed"),
        ("Epoch time (s)", "epoch_time_seconds", "#059669"),
        ("Cumulative time (s)", "elapsed_time_seconds", "#d97706"),
        ("GPU memory (MB)", "gpu_memory_used_mb", "#0891b2"),
    )
    if not records:
        raise RealMLPFullTrainError("Cannot render an empty full-Train history.")
    width, panel_height, left, right = 1100, 190, 85, 25
    plot_width = width - left - right
    height = panel_height * len(panels)
    chunks = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}'
            '.axis{stroke:#9ca3af;stroke-width:1}'
            '.grid{stroke:#e5e7eb;stroke-width:1}'
            '.line{fill:none;stroke-width:2}</style>'
        ),
    ]
    for panel_index, (title, field, color) in enumerate(panels):
        top = panel_index * panel_height + 28
        bottom = panel_index * panel_height + panel_height - 35
        values = [float(record[field]) for record in records]
        if not np.isfinite(values).all():
            raise RealMLPFullTrainError(f"History field {field} is not finite.")
        low, high = min(values), max(values)
        if math.isclose(low, high):
            high = low + 1.0
        points = []
        for index, value in enumerate(values):
            x = left + plot_width * index / max(1, len(values) - 1)
            y = bottom - (bottom - top) * (value - low) / (high - low)
            points.append(f"{x:.2f},{y:.2f}")
        chunks.extend(
            [
                (
                    f'<text x="{left}" y="{top - 8}" font-size="15" '
                    f'font-weight="600">{html.escape(title)}</text>'
                ),
                (
                    f'<line class="axis" x1="{left}" y1="{bottom}" '
                    f'x2="{width-right}" y2="{bottom}"/>'
                ),
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                (
                    f'<line class="grid" x1="{left}" '
                    f'y1="{(top+bottom)/2:.1f}" x2="{width-right}" '
                    f'y2="{(top+bottom)/2:.1f}"/>'
                ),
                f'<text x="8" y="{top + 5}" font-size="11">{high:.6g}</text>',
                f'<text x="8" y="{bottom + 4}" font-size="11">{low:.6g}</text>',
                f'<text x="{left}" y="{bottom + 20}" font-size="11">1</text>',
                (
                    f'<text x="{width-right-24}" y="{bottom + 20}" '
                    f'font-size="11">{len(records)}</text>'
                ),
                f'<polyline class="line" stroke="{color}" points="{" ".join(points)}"/>',
            ]
        )
    chunks.append("</svg>\n")
    path.write_text("\n".join(chunks), encoding="utf-8")


def _validate_checkpoint(path: Path) -> None:
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise RealMLPFullTrainError("Full-Train checkpoint cannot be deserialized.") from exc
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"estimator", "preprocessing", "scope"}
        or bundle.get("scope") != CHECKPOINT_SCOPE
        or bundle.get("estimator") is None
        or bundle.get("preprocessing") is None
        or not hasattr(bundle.get("estimator"), "backend_")
        or not hasattr(bundle.get("preprocessing"), "named_steps")
    ):
        raise RealMLPFullTrainError("Full-Train checkpoint bundle is invalid.")


def _lineage_hashes(
    pilot_run_dir: Path, cv_root: Path, runtime_dir: Path
) -> dict[str, str]:
    paths = {
        "rmlp05_decision_sha256": pilot_run_dir / "decision.json",
        "rmlp06b_cv_results_sha256": cv_root / "summary" / "cv_results.json",
        "rmlp06b_manifest_sha256": cv_root / "summary" / "manifest.json",
        "rmlp04_runtime_manifest_sha256": runtime_dir / "runtime_manifest.json",
    }
    try:
        return {name: _sha256_file(path) for name, path in paths.items()}
    except OSError as exc:
        raise RealMLPFullTrainError("Cannot read full-Train lineage evidence.") from exc


def _validate_working_files(directory: Path) -> None:
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != _BASE_FILES:
        raise RealMLPFullTrainError(
            f"Full-Train working artifact set differs: {sorted(actual ^ _BASE_FILES)}"
        )
    state = _read_json(directory / "run_state.json")
    if state.get("state") != "COMPLETED" or state.get("contract_id") != "RMLP-07A":
        raise RealMLPFullTrainError("Full-Train working state is invalid.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPFullTrainError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RealMLPFullTrainError(f"JSON artifact must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPFullTrainError(f"Cannot read JSONL artifact: {path}") from exc
    if any(not isinstance(record, dict) for record in records):
        raise RealMLPFullTrainError("Full-Train JSONL history contains a non-object.")
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run, resume, or validate the Train-only RealMLP full fit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--pilot-run-dir", type=Path, default=DEFAULT_PILOT_RUN_DIR)
    run_parser.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT)
    run_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    run_parser.add_argument("--device-index", type=int, default=0)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate_parser.add_argument(
        "--pilot-run-dir", type=Path, default=DEFAULT_PILOT_RUN_DIR
    )
    validate_parser.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT)
    validate_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_realmlp_full_train(
                args.output_dir,
                pilot_run_dir=args.pilot_run_dir,
                cv_root=args.cv_root,
                runtime_dir=args.runtime_dir,
                device_index=args.device_index,
            )
        else:
            manifest = validate_realmlp_full_train(
                args.output_dir,
                pilot_run_dir=args.pilot_run_dir,
                cv_root=args.cv_root,
                runtime_dir=args.runtime_dir,
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except RealMLPFullTrainError as exc:
        print(f"Dataset B RealMLP full-Train failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
