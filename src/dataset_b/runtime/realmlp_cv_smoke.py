"""Atomic one-fold fixed-epoch integration smoke for Dataset B RealMLP."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
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
from src.dataset_b.splits import DatasetBSplitter
from src.utils.path import PROJECT_ROOT

from .realmlp_history import (
    HISTORY_METHOD,
    HistoryEnabledRealMLPClassifier,
    RealMLPHistoryRecorder,
)
from .realmlp_pilot_run import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PILOT_RUN_DIR,
    DEFAULT_RUNTIME_DIR,
    _load_runtime_gate,
    validate_realmlp_pilot_run,
)
from .realmlp_smoke import (
    _Tee,
    _json_default,
    _probability_summary,
    _sha256_file,
    _validate_history_records,
    _write_history,
    _write_json,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "cv_smoke"
)
SMOKE_FOLD_NUMBER = 1
_BASE_FILES = {
    "checkpoints/fitted_model.pkl",
    "fold_result.json",
    "predictions/fold_validation_predictions.csv",
    "run.log",
    "run_state.json",
    "training_history.json",
    "training_history.jsonl",
}
_FINAL_FILES = _BASE_FILES | {"COMPLETED", "manifest.json"}


class RealMLPCVSmokeError(RuntimeError):
    """Raised when the RMLP-06A fold smoke violates its fixed-epoch contract."""


@dataclass(frozen=True)
class _FoldInputs:
    fold_number: int
    X_fit: Any
    y_fit: np.ndarray
    X_validation: Any
    y_validation: np.ndarray
    validation_ids: tuple[str, ...]
    preprocessing: Any
    metadata: dict[str, Any]


def run_realmlp_fixed_epoch_fold_smoke(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    pilot_run_dir: str | Path = DEFAULT_PILOT_RUN_DIR,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    device_index: int = 0,
    fold_number: int = SMOKE_FOLD_NUMBER,
) -> dict[str, Any]:
    """Run and atomically publish exactly one Train-only outer-fold smoke."""
    if fold_number != SMOKE_FOLD_NUMBER:
        raise RealMLPCVSmokeError("RMLP-06A is restricted to fold 1 only.")
    root = Path(output_root)
    final = root / f"fold_{fold_number:02d}"
    working = root / f"fold_{fold_number:02d}.tmp"
    if final.exists():
        manifest = validate_realmlp_cv_smoke(final)
        return {**manifest, "resume_action": "SKIPPED_VALID_COMPLETED"}

    runtime_manifest = _load_runtime_gate(Path(runtime_dir), device_index)
    decision = _load_fixed_epoch_decision(Path(pilot_run_dir))
    n_epochs = int(decision["five_fold_cv_policy"]["n_epochs"])
    batch_size = int(runtime_manifest["selected_batch_size"])
    root.mkdir(parents=True, exist_ok=True)
    archived_attempt = _archive_interrupted_working_directory(working)
    atomic_probe = _atomic_rename_probe(root)
    inputs = _load_fold_inputs(fold_number)

    working.mkdir()
    (working / "checkpoints").mkdir()
    (working / "predictions").mkdir()
    (working / "run.log").write_text("", encoding="utf-8")
    _write_run_state(
        working,
        "STARTED",
        message="RMLP-06A fold smoke initialized from immutable Train only.",
    )

    try:
        execution = _fit_fixed_epoch_fold(
            working,
            inputs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device_index=device_index,
            stage_prefix="cv_smoke",
            checkpoint_scope=(
                "RMLP-06A one-fold integration smoke; not a production CV model."
            ),
            contract_id="RMLP-06A",
        )
        history_summary = _validate_history_records(
            execution["history"],
            n_epochs,
            inner_validation_required=False,
        )
        probability_summary = _probability_summary(execution["probabilities"])
        if not history_summary["complete"]:
            raise RealMLPCVSmokeError(
                "Fixed-epoch fold history does not contain every completed epoch."
            )
        if not probability_summary["valid"]:
            raise RealMLPCVSmokeError("Outer-fold probabilities violate their contract.")

        _write_history(
            working,
            execution["history"],
            HISTORY_METHOD,
            history_summary,
        )
        prediction_path = (
            working / "predictions" / "fold_validation_predictions.csv"
        )
        _write_predictions(
            prediction_path,
            inputs.validation_ids,
            inputs.y_validation,
            execution["probabilities"],
        )
        _validate_predictions(prediction_path, inputs.validation_ids)

        fold_result = {
            "schema_version": 1,
            "contract_id": "REALMLP-FIXED-EPOCH-FOLD-SMOKE-01",
            "status": "PASS",
            "evidence_scope": (
                "One-fold Train-only integration evidence; not complete five-fold CV, "
                "Validation, Test, or RQ1 evidence."
            ),
            "fold": fold_number,
            "configuration": {
                "decision": decision["decision"],
                "decision_sha256": _sha256_file(Path(pilot_run_dir) / "decision.json"),
                "n_epochs": n_epochs,
                "use_early_stopping": False,
                "use_best_epoch": False,
                "batch_size": batch_size,
                "device": f"cuda:{device_index}",
            },
            "data_boundary": inputs.metadata,
            "history": {
                "method": HISTORY_METHOD,
                "inner_validation": "NOT_APPLICABLE_FIXED_EPOCH",
                **history_summary,
            },
            "probabilities": probability_summary,
            "execution": execution["metadata"],
            "atomic_rename_probe": atomic_probe,
            "archived_interrupted_attempt": archived_attempt,
        }
        _write_json(working / "fold_result.json", fold_result)
        _write_run_state(
            working,
            "COMPLETED",
            message="Training, single outer-fold prediction, model save, and validation passed.",
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
            "contract_id": "RMLP-06A",
            "status": "PASS",
            "fold": fold_number,
            "fixed_epochs": n_epochs,
            "artifact_sha256": hashes,
        }
        _write_json(working / "manifest.json", manifest)
        (working / "COMPLETED").write_text("RMLP-06A PASS\n", encoding="utf-8")
        validate_realmlp_cv_smoke(working)
        os.replace(working, final)
        validate_realmlp_cv_smoke(final)
        return manifest
    except Exception as exc:
        if working.exists():
            _write_run_state(
                working,
                "FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
        if isinstance(exc, RealMLPCVSmokeError):
            raise
        raise RealMLPCVSmokeError(f"RealMLP fixed-epoch fold smoke failed: {exc}") from exc


def validate_realmlp_cv_smoke(
    fold_dir: str | Path = DEFAULT_OUTPUT_ROOT / "fold_01",
) -> dict[str, Any]:
    """Validate a published RMLP-06A fold and all recorded hashes."""
    directory = Path(fold_dir)
    if not directory.is_dir():
        raise RealMLPCVSmokeError(f"Published fold directory is missing: {directory}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != _FINAL_FILES:
        raise RealMLPCVSmokeError(
            f"Published fold artifact set differs: {sorted(actual ^ _FINAL_FILES)}"
        )
    manifest = _read_json(directory / "manifest.json")
    result = _read_json(directory / "fold_result.json")
    state = _read_json(directory / "run_state.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("contract_id") != "RMLP-06A"
        or result.get("status") != "PASS"
        or state.get("state") != "COMPLETED"
        or (directory / "COMPLETED").read_text(encoding="utf-8")
        != "RMLP-06A PASS\n"
    ):
        raise RealMLPCVSmokeError("Published fold completion state is invalid.")
    hashes = manifest.get("artifact_sha256")
    expected_hashed = _FINAL_FILES - {"manifest.json", "COMPLETED"}
    if not isinstance(hashes, dict) or set(hashes) != expected_hashed:
        raise RealMLPCVSmokeError("Published fold hash coverage is invalid.")
    for relative, expected_hash in hashes.items():
        if _sha256_file(directory / relative) != expected_hash:
            raise RealMLPCVSmokeError(f"Published fold hash mismatch: {relative}")
    history = _read_json(directory / "training_history.json")
    summary = _validate_history_records(
        history.get("records", []),
        int(manifest["fixed_epochs"]),
        inner_validation_required=False,
    )
    if not summary["complete"]:
        raise RealMLPCVSmokeError("Published fixed-epoch history is invalid.")
    return manifest


def _load_fixed_epoch_decision(pilot_run_dir: Path) -> dict[str, Any]:
    try:
        validate_realmlp_pilot_run(pilot_run_dir)
        payload = _read_json(pilot_run_dir / "decision.json")
        policy = payload["five_fold_cv_policy"]
    except Exception as exc:
        raise RealMLPCVSmokeError("RMLP-05 decision evidence is invalid.") from exc
    expected = {
        "fit_membership": "all_outer_fold_train_rows",
        "inner_stop_set": "none",
        "n_epochs": 51,
        "use_early_stopping": False,
    }
    if payload.get("status") != "LOCKED" or payload.get("decision") != "FIXED_EPOCH":
        raise RealMLPCVSmokeError("RMLP-06A requires a locked FIXED_EPOCH decision.")
    if policy != expected:
        raise RealMLPCVSmokeError("Five-fold policy differs from the locked 51-epoch contract.")
    return payload


def _load_fold_inputs(fold_number: int) -> _FoldInputs:
    try:
        train = DatasetBSplitLoader().load("train")
        splitter = DatasetBSplitter.from_yaml().cv_splitter()
        selected = None
        for current, positions in enumerate(splitter.split(train.X, train.y), start=1):
            if current == fold_number:
                selected = positions
                break
        if selected is None:
            raise RealMLPCVSmokeError(f"Cannot materialize outer fold {fold_number}.")
        fit_positions, validation_positions = selected
        raw_fit = train.X.iloc[fit_positions].reset_index(drop=True)
        raw_validation = train.X.iloc[validation_positions].reset_index(drop=True)
        y_fit = train.y.iloc[fit_positions].to_numpy(dtype=np.int64)
        y_validation = train.y.iloc[validation_positions].to_numpy(dtype=np.int64)
        validation_ids = tuple(
            train.record_ids.iloc[validation_positions].astype(str).tolist()
        )
        preprocessing = make_realmlp_native_preprocessing_pipeline()
        X_fit = preprocessing.fit_transform(raw_fit)
        X_validation = preprocessing.transform(raw_validation)
    except RealMLPCVSmokeError:
        raise
    except Exception as exc:
        raise RealMLPCVSmokeError(
            f"Cannot prepare immutable Train-only fold {fold_number}."
        ) from exc

    support = preprocessing.named_steps["native"].last_category_support_
    outcomes = [
        category["outcome"]
        for column in support.values()
        for category in column.values()
    ]
    return _FoldInputs(
        fold_number=fold_number,
        X_fit=X_fit,
        y_fit=y_fit,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_ids=validation_ids,
        preprocessing=preprocessing,
        metadata={
            "source_partition": "train",
            "split_contract": "split_v1/StratifiedKFold(n_splits=5,shuffle=True,random_state=42)",
            "fold_train_rows": len(fit_positions),
            "fold_validation_rows": len(validation_positions),
            "fold_train_id_sha256": _sha256_ids(
                train.record_ids.iloc[fit_positions].astype(str).tolist()
            ),
            "fold_validation_id_sha256": _sha256_ids(validation_ids),
            "preprocessing_fit_scope": "outer_fold_train_only",
            "model_fit_scope": "all_outer_fold_train_rows",
            "outer_fold_validation_usage": "single_post_training_predict_proba_only",
            "inner_stop_set": "none",
            "project_validation_accessed": False,
            "project_test_accessed": False,
            "category_support_checked_cells": len(outcomes),
            "category_support_absent_both_warnings": outcomes.count(
                "warning_absent_both"
            ),
            "category_support_failures": outcomes.count(
                "fail_absent_train_present_validation"
            ),
        },
    )


def _fit_fixed_epoch_fold(
    directory: Path,
    inputs: _FoldInputs,
    *,
    n_epochs: int,
    batch_size: int,
    device_index: int,
    stage_prefix: str,
    checkpoint_scope: str,
    contract_id: str,
) -> dict[str, Any]:
    import torch

    _write_run_state(
        directory,
        "TRAINING",
        message=f"Fitting all fold-train rows for exactly {n_epochs} epochs without validation.",
        contract_id=contract_id,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    recorder = RealMLPHistoryRecorder(
        directory / "training_history.jsonl",
        device_index=device_index,
        expected_epochs=n_epochs,
        stage=f"{stage_prefix}_fold_{inputs.fold_number:02d}",
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
            estimator.fit(inputs.X_fit, inputs.y_fit)
            _write_run_state(
                directory,
                "PREDICTING",
                message="Calling predict_proba once on outer-fold validation after training.",
                contract_id=contract_id,
            )
            probabilities = estimator.predict_proba(inputs.X_validation)

    _write_run_state(
        directory,
        "SAVING",
        message="Saving the fitted fold model without a second validation prediction.",
        contract_id=contract_id,
    )
    checkpoint = directory / "checkpoints" / "fitted_model.pkl"
    joblib.dump(
        {
            "estimator": estimator,
            "preprocessing": inputs.preprocessing,
            "scope": checkpoint_scope,
        },
        checkpoint,
        compress=3,
    )
    restored = joblib.load(checkpoint)
    if set(restored) != {"estimator", "preprocessing", "scope"}:
        raise RealMLPCVSmokeError("Serialized fold model bundle is invalid.")
    history = list(recorder.records)
    return {
        "probabilities": probabilities,
        "history": history,
        "metadata": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": max(
                record["gpu_memory_used_mb"] for record in history
            ),
            "runtime_metadata": estimator.runtime_metadata_,
            "fit_params": getattr(estimator.backend_, "fit_params_", None),
            "predict_proba_calls_on_outer_fold_validation": 1,
            "model_deserialization": "PASS_NO_SECOND_PREDICTION",
        },
    }


def _write_predictions(
    path: Path,
    record_ids: Sequence[str],
    labels: np.ndarray,
    values: np.ndarray,
) -> None:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (len(record_ids), 2) or len(labels) != len(record_ids):
        raise RealMLPCVSmokeError("Outer-fold predictions do not align with membership.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id", "y_true", "probability_0", "probability_1"])
        for record_id, label, row in zip(
            record_ids, labels, probabilities, strict=True
        ):
            writer.writerow(
                [record_id, int(label), repr(float(row[0])), repr(float(row[1]))]
            )


def _validate_predictions(path: Path, expected_ids: Sequence[str]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RealMLPCVSmokeError("Cannot read fold-validation predictions.") from exc
    observed_ids = tuple(row["record_id"] for row in rows)
    if observed_ids != tuple(expected_ids):
        raise RealMLPCVSmokeError("Fold-validation prediction order or membership differs.")
    try:
        probabilities = np.asarray(
            [[float(row["probability_0"]), float(row["probability_1"])] for row in rows]
        )
    except (KeyError, ValueError) as exc:
        raise RealMLPCVSmokeError("Fold-validation prediction values are invalid.") from exc
    if not _probability_summary(probabilities)["valid"]:
        raise RealMLPCVSmokeError("Persisted fold-validation probabilities are invalid.")


def _atomic_rename_probe(root: Path) -> dict[str, Any]:
    temporary = root / ".atomic_rename_probe.tmp"
    final = root / ".atomic_rename_probe.final"
    if temporary.exists() or final.exists():
        raise RealMLPCVSmokeError("Atomic rename probe path already exists.")
    temporary.mkdir()
    (temporary / "probe.txt").write_text("same-filesystem\n", encoding="utf-8")
    try:
        os.replace(temporary, final)
        if (final / "probe.txt").read_text(encoding="utf-8") != "same-filesystem\n":
            raise RealMLPCVSmokeError("Atomic rename probe content differs after rename.")
    finally:
        if (final / "probe.txt").exists():
            (final / "probe.txt").unlink()
        if final.exists():
            final.rmdir()
        if temporary.exists():
            temporary.rmdir()
    return {"status": "PASS", "operation": "same_filesystem_os_replace_directory"}


def _archive_interrupted_working_directory(working: Path) -> str | None:
    if not working.exists():
        return None
    for attempt in range(1, 1000):
        archived = working.with_name(
            f"{working.name.removesuffix('.tmp')}.failed_attempt_{attempt:03d}"
        )
        if not archived.exists():
            os.replace(working, archived)
            return archived.name
    raise RealMLPCVSmokeError("Cannot allocate a failed-attempt archive path.")


def _validate_working_files(directory: Path) -> None:
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != _BASE_FILES:
        raise RealMLPCVSmokeError(
            f"Working fold artifact set differs: {sorted(actual ^ _BASE_FILES)}"
        )
    if _read_json(directory / "run_state.json").get("state") != "COMPLETED":
        raise RealMLPCVSmokeError("Working fold is not in COMPLETED state.")


def _write_run_state(
    directory: Path,
    state: str,
    *,
    message: str,
    contract_id: str = "RMLP-06A",
) -> None:
    path = directory / "run_state.json"
    previous = None
    if path.exists():
        previous = _read_json(path).get("state")
    _write_json(
        path,
        {
            "schema_version": 1,
            "contract_id": contract_id,
            "state": state,
            "previous_state": previous,
            "message": message,
        },
    )


def _sha256_ids(record_ids: Sequence[str]) -> str:
    payload = "".join(f"{record_id}\n" for record_id in record_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPCVSmokeError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RealMLPCVSmokeError(f"JSON artifact must be an object: {path}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or validate the RMLP-06A fixed-epoch one-fold smoke."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--pilot-run-dir", type=Path, default=DEFAULT_PILOT_RUN_DIR)
    run_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    run_parser.add_argument("--device-index", type=int, default=0)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--fold-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "fold_01"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_realmlp_fixed_epoch_fold_smoke(
                args.output_root,
                pilot_run_dir=args.pilot_run_dir,
                runtime_dir=args.runtime_dir,
                device_index=args.device_index,
            )
        else:
            manifest = validate_realmlp_cv_smoke(args.fold_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except RealMLPCVSmokeError as exc:
        print(f"Dataset B RealMLP CV smoke failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
