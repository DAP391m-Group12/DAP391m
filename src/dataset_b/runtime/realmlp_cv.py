"""Production Train-only five-fold CV orchestration for Dataset B RealMLP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from src.dataset_b.preprocessing import DatasetBSplitLoader
from src.dataset_b.splits import DatasetBSplitter
from src.utils.path import PROJECT_ROOT

from .realmlp_cv_smoke import (
    RealMLPCVSmokeError,
    _archive_interrupted_working_directory,
    _atomic_rename_probe,
    _fit_fixed_epoch_fold,
    _load_fixed_epoch_decision,
    _load_fold_inputs,
    _validate_predictions,
    _write_predictions,
    _write_run_state,
)
from .realmlp_history import HISTORY_METHOD
from .realmlp_pilot_run import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PILOT_RUN_DIR,
    DEFAULT_RUNTIME_DIR,
    _load_runtime_gate,
)
from .realmlp_smoke import (
    _json_default,
    _probability_summary,
    _sha256_file,
    _validate_history_records,
    _write_history,
    _write_json,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "cv"
FOLD_COUNT = 5
EXPECTED_TRAIN_ROWS = 205_310
_FOLD_BASE_FILES = {
    "checkpoints/fitted_model.pkl",
    "fold_result.json",
    "predictions/fold_validation_predictions.csv",
    "run.log",
    "run_state.json",
    "training_history.json",
    "training_history.jsonl",
}
_FOLD_FINAL_FILES = _FOLD_BASE_FILES | {"COMPLETED", "manifest.json"}
_SUMMARY_FILES = {"COMPLETED", "cv_results.json", "manifest.json", "oof_predictions.csv"}


class RealMLPCVError(RuntimeError):
    """Raised when RMLP-06B violates its production CV/OOF contract."""


def run_realmlp_production_cv(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    pilot_run_dir: str | Path = DEFAULT_PILOT_RUN_DIR,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    device_index: int = 0,
) -> dict[str, Any]:
    """Run or resume five fresh Train-only folds, then publish OOF evidence."""
    root = Path(output_root)
    summary_dir = root / "summary"
    if summary_dir.exists():
        manifest = validate_realmlp_production_cv(root)
        return {**manifest, "resume_action": "SKIPPED_VALID_COMPLETED"}

    try:
        runtime_manifest = _load_runtime_gate(Path(runtime_dir), device_index)
        decision = _load_fixed_epoch_decision(Path(pilot_run_dir))
    except Exception as exc:
        raise RealMLPCVError("RMLP-04/RMLP-05 prerequisite evidence is invalid.") from exc
    n_epochs = int(decision["five_fold_cv_policy"]["n_epochs"])
    batch_size = int(runtime_manifest["selected_batch_size"])
    root.mkdir(parents=True, exist_ok=True)
    atomic_probe = _atomic_rename_probe(root)

    fold_results: list[dict[str, Any]] = []
    for fold_number in range(1, FOLD_COUNT + 1):
        fold_results.append(
            _run_or_resume_fold(
                root,
                fold_number=fold_number,
                pilot_run_dir=Path(pilot_run_dir),
                n_epochs=n_epochs,
                batch_size=batch_size,
                device_index=device_index,
                atomic_probe=atomic_probe,
            )
        )

    return _publish_oof_summary(
        root,
        fold_results=fold_results,
        pilot_run_dir=Path(pilot_run_dir),
        n_epochs=n_epochs,
    )


def validate_realmlp_production_cv(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate five published folds and the immutable OOF summary."""
    root = Path(output_root)
    train, fold_contracts = _train_and_fold_contracts()
    fold_results = []
    for fold_number, (_, validation_positions) in enumerate(fold_contracts, start=1):
        fold_results.append(
            validate_realmlp_production_fold(
                root / f"fold_{fold_number:02d}",
                expected_fold=fold_number,
                expected_ids=tuple(
                    train.record_ids.iloc[validation_positions].astype(str).tolist()
                ),
                expected_labels=train.y.iloc[validation_positions].to_numpy(dtype=np.int64),
            )
        )

    summary = root / "summary"
    actual = {
        path.relative_to(summary).as_posix()
        for path in summary.rglob("*")
        if path.is_file()
    } if summary.is_dir() else set()
    if actual != _SUMMARY_FILES:
        raise RealMLPCVError(
            f"Production CV summary artifact set differs: {sorted(actual ^ _SUMMARY_FILES)}"
        )
    manifest = _read_json(summary / "manifest.json")
    results = _read_json(summary / "cv_results.json")
    if (
        manifest.get("contract_id") != "RMLP-06B"
        or manifest.get("status") != "PASS"
        or results.get("status") != "PASS"
        or (summary / "COMPLETED").read_text(encoding="utf-8")
        != "RMLP-06B PASS\n"
    ):
        raise RealMLPCVError("Production CV summary completion state is invalid.")
    configuration = results.get("configuration", {})
    decision_hash = configuration.get("decision_sha256")
    if (
        configuration.get("decision") != "FIXED_EPOCH"
        or configuration.get("n_epochs") != 51
        or configuration.get("use_early_stopping") is not False
        or configuration.get("use_best_epoch") is not False
        or not isinstance(decision_hash, str)
        or not decision_hash
        or any(
            result.get("configuration", {}).get("decision_sha256") != decision_hash
            for result in fold_results
        )
    ):
        raise RealMLPCVError("Production CV summary decision lineage is invalid.")
    hashes = manifest.get("artifact_sha256")
    expected_hashed = _SUMMARY_FILES - {"manifest.json", "COMPLETED"}
    if not isinstance(hashes, dict) or set(hashes) != expected_hashed:
        raise RealMLPCVError("Production CV summary hash coverage is invalid.")
    for relative, expected_hash in hashes.items():
        if _sha256_file(summary / relative) != expected_hash:
            raise RealMLPCVError(f"Production CV summary hash mismatch: {relative}")

    oof_rows = _read_prediction_rows(summary / "oof_predictions.csv")
    expected_ids = tuple(train.record_ids.astype(str).tolist())
    _validate_published_oof_order(oof_rows, expected_ids)
    ordered = _order_and_validate_oof_rows(
        expected_ids,
        oof_rows,
        expected_labels=dict(
            zip(
                train.record_ids.astype(str).tolist(),
                train.y.astype(int).tolist(),
                strict=True,
            )
        ),
    )
    metrics = _metric_summary(ordered, fold_results)
    _validate_running_metrics(fold_results)
    if results.get("metrics") != metrics:
        raise RealMLPCVError("Stored production CV metrics differ from recomputation.")
    if (
        results.get("rows") != EXPECTED_TRAIN_ROWS
        or results.get("unique_record_ids") != EXPECTED_TRAIN_ROWS
        or results.get("source_order_preserved") is not True
        or results.get("fold_count") != FOLD_COUNT
        or manifest.get("fold_count") != FOLD_COUNT
        or manifest.get("oof_rows") != EXPECTED_TRAIN_ROWS
    ):
        raise RealMLPCVError("Stored production CV row count is invalid.")
    return manifest


def validate_realmlp_production_fold(
    fold_dir: str | Path,
    *,
    expected_fold: int,
    expected_ids: Sequence[str],
    expected_labels: np.ndarray,
) -> dict[str, Any]:
    """Validate one atomic production fold against expected Train membership."""
    directory = Path(fold_dir)
    if not directory.is_dir():
        raise RealMLPCVError(f"Published production fold is missing: {directory}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != _FOLD_FINAL_FILES:
        raise RealMLPCVError(
            f"Production fold artifact set differs: {sorted(actual ^ _FOLD_FINAL_FILES)}"
        )
    manifest = _read_json(directory / "manifest.json")
    result = _read_json(directory / "fold_result.json")
    state = _read_json(directory / "run_state.json")
    if (
        manifest.get("contract_id") != "RMLP-06B-FOLD"
        or manifest.get("status") != "PASS"
        or result.get("status") != "PASS"
        or state.get("state") != "COMPLETED"
        or state.get("contract_id") != "RMLP-06B"
        or (directory / "COMPLETED").read_text(encoding="utf-8")
        != "RMLP-06B FOLD PASS\n"
    ):
        raise RealMLPCVError("Production fold completion state is invalid.")
    fold_number = manifest.get("fold")
    configuration = result.get("configuration", {})
    data_boundary = result.get("data_boundary", {})
    execution = result.get("execution", {})
    runtime_metadata = execution.get("runtime_metadata", {})
    if (
        not isinstance(fold_number, int)
        or fold_number != expected_fold
        or result.get("fold") != fold_number
        or manifest.get("fixed_epochs") != 51
        or configuration.get("n_epochs") != 51
        or configuration.get("decision") != "FIXED_EPOCH"
        or configuration.get("use_early_stopping") is not False
        or configuration.get("use_best_epoch") is not False
        or execution.get("predict_proba_calls_on_outer_fold_validation") != 1
        or runtime_metadata.get("validation_source") != "none"
        or runtime_metadata.get("validation_rows") != 0
        or runtime_metadata.get("use_best_epoch") is not False
        or execution.get("fit_params")
        != {"stop_epoch": {"cross_entropy": 51}}
        or execution.get("model_deserialization")
        != "PASS_NO_SECOND_PREDICTION"
    ):
        raise RealMLPCVError("Production fold fixed-epoch/no-validation contract is invalid.")
    if (
        data_boundary.get("source_partition") != "train"
        or data_boundary.get("preprocessing_fit_scope") != "outer_fold_train_only"
        or data_boundary.get("model_fit_scope") != "all_outer_fold_train_rows"
        or data_boundary.get("outer_fold_validation_usage")
        != "single_post_training_predict_proba_only"
        or data_boundary.get("inner_stop_set") != "none"
        or data_boundary.get("project_validation_accessed") is not False
        or data_boundary.get("project_test_accessed") is not False
        or data_boundary.get("category_support_failures") != 0
        or data_boundary.get("fold_validation_rows") != len(expected_ids)
        or runtime_metadata.get("train_rows")
        != data_boundary.get("fold_train_rows")
    ):
        raise RealMLPCVError("Production fold Train-only data boundary is invalid.")
    hashes = manifest.get("artifact_sha256")
    expected_hashed = _FOLD_FINAL_FILES - {"manifest.json", "COMPLETED"}
    if not isinstance(hashes, dict) or set(hashes) != expected_hashed:
        raise RealMLPCVError("Production fold hash coverage is invalid.")
    for relative, expected_hash in hashes.items():
        if _sha256_file(directory / relative) != expected_hash:
            raise RealMLPCVError(f"Production fold hash mismatch: {relative}")

    history = _read_json(directory / "training_history.json")
    history_summary = _validate_history_records(
        history.get("records", []),
        int(manifest.get("fixed_epochs", 0)),
        inner_validation_required=False,
    )
    if not history_summary["complete"]:
        raise RealMLPCVError("Production fold history is incomplete.")
    prediction_path = directory / "predictions" / "fold_validation_predictions.csv"
    try:
        _validate_predictions(prediction_path, expected_ids)
    except RealMLPCVSmokeError as exc:
        raise RealMLPCVError(str(exc)) from exc
    rows = _read_prediction_rows(prediction_path)
    observed_labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    if not np.array_equal(observed_labels, expected_labels):
        raise RealMLPCVError("Production fold labels differ from immutable Train.")
    probabilities = np.asarray(
        [[float(row["probability_0"]), float(row["probability_1"])] for row in rows]
    )
    if not _probability_summary(probabilities)["valid"]:
        raise RealMLPCVError("Production fold probabilities are invalid.")
    observed_auc = float(roc_auc_score(observed_labels, probabilities[:, 1]))
    stored_auc = result.get("metrics", {}).get("roc_auc")
    if not isinstance(stored_auc, (int, float)) or not math.isclose(
        observed_auc, float(stored_auc), rel_tol=0, abs_tol=1e-12
    ):
        raise RealMLPCVError("Production fold ROC-AUC differs from recomputation.")
    return result


def _run_or_resume_fold(
    root: Path,
    *,
    fold_number: int,
    pilot_run_dir: Path,
    n_epochs: int,
    batch_size: int,
    device_index: int,
    atomic_probe: dict[str, Any],
) -> dict[str, Any]:
    train, contracts = _train_and_fold_contracts()
    _, validation_positions = contracts[fold_number - 1]
    expected_ids = tuple(
        train.record_ids.iloc[validation_positions].astype(str).tolist()
    )
    expected_labels = train.y.iloc[validation_positions].to_numpy(dtype=np.int64)
    final = root / f"fold_{fold_number:02d}"
    if final.exists():
        return validate_realmlp_production_fold(
            final,
            expected_fold=fold_number,
            expected_ids=expected_ids,
            expected_labels=expected_labels,
        )

    working = root / f"fold_{fold_number:02d}.tmp"
    archived_attempt = _archive_interrupted_working_directory(working)
    try:
        inputs = _load_fold_inputs(fold_number)
    except RealMLPCVSmokeError as exc:
        raise RealMLPCVError(str(exc)) from exc
    if inputs.validation_ids != expected_ids:
        raise RealMLPCVError("Prepared fold membership differs from the split contract.")

    working.mkdir()
    (working / "checkpoints").mkdir()
    (working / "predictions").mkdir()
    (working / "run.log").write_text("", encoding="utf-8")
    _write_run_state(
        working,
        "STARTED",
        message=f"RMLP-06B production fold {fold_number} initialized from Train only.",
        contract_id="RMLP-06B",
    )
    try:
        execution = _fit_fixed_epoch_fold(
            working,
            inputs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device_index=device_index,
            stage_prefix="cv",
            checkpoint_scope=(
                f"RMLP-06B production Train-only CV fold {fold_number}; "
                "not a full-Train or final model."
            ),
            contract_id="RMLP-06B",
        )
        history_summary = _validate_history_records(
            execution["history"], n_epochs, inner_validation_required=False
        )
        probability_summary = _probability_summary(execution["probabilities"])
        if not history_summary["complete"] or not probability_summary["valid"]:
            raise RealMLPCVError("Production fold history or probabilities are invalid.")
        _write_history(working, execution["history"], HISTORY_METHOD, history_summary)
        prediction_path = working / "predictions" / "fold_validation_predictions.csv"
        _write_predictions(
            prediction_path,
            inputs.validation_ids,
            inputs.y_validation,
            execution["probabilities"],
        )
        _validate_predictions(prediction_path, expected_ids)
        fold_auc = float(
            roc_auc_score(inputs.y_validation, execution["probabilities"][:, 1])
        )
        prior_aucs = _published_prior_fold_aucs(root, fold_number)
        running_aucs = np.asarray([*prior_aucs, fold_auc], dtype=np.float64)
        result = {
            "schema_version": 1,
            "contract_id": "REALMLP-PRODUCTION-CV-FOLD-01",
            "status": "PASS",
            "evidence_scope": "One production Train-only outer fold; not Validation or Test evidence.",
            "fold": fold_number,
            "configuration": {
                "decision": "FIXED_EPOCH",
                "decision_sha256": _sha256_file(pilot_run_dir / "decision.json"),
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
            "metrics": {
                "roc_auc": fold_auc,
                "running_completed_folds": fold_number,
                "running_mean_roc_auc": float(running_aucs.mean()),
                "running_std_roc_auc": float(running_aucs.std(ddof=0)),
                "running_std_ddof": 0,
            },
            "execution": execution["metadata"],
            "atomic_rename_probe": atomic_probe,
            "archived_interrupted_attempt": archived_attempt,
        }
        _write_json(working / "fold_result.json", result)
        _write_run_state(
            working,
            "COMPLETED",
            message=(
                "Fixed-epoch training and one post-training outer-fold prediction passed."
            ),
            contract_id="RMLP-06B",
        )
        _validate_fold_working_files(working)
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
            "contract_id": "RMLP-06B-FOLD",
            "status": "PASS",
            "fold": fold_number,
            "fixed_epochs": n_epochs,
            "artifact_sha256": hashes,
        }
        _write_json(working / "manifest.json", manifest)
        (working / "COMPLETED").write_text(
            "RMLP-06B FOLD PASS\n", encoding="utf-8"
        )
        validate_realmlp_production_fold(
            working,
            expected_fold=fold_number,
            expected_ids=expected_ids,
            expected_labels=expected_labels,
        )
        os.replace(working, final)
        return validate_realmlp_production_fold(
            final,
            expected_fold=fold_number,
            expected_ids=expected_ids,
            expected_labels=expected_labels,
        )
    except KeyboardInterrupt:
        if working.exists():
            _write_run_state(
                working,
                "INTERRUPTED",
                message="Production fold interrupted before atomic publication.",
                contract_id="RMLP-06B",
            )
        raise
    except Exception as exc:
        if working.exists():
            _write_run_state(
                working,
                "FAILED",
                message=f"{type(exc).__name__}: {exc}",
                contract_id="RMLP-06B",
            )
        if isinstance(exc, RealMLPCVError):
            raise
        raise RealMLPCVError(
            f"Production RealMLP fold {fold_number} failed: {exc}"
        ) from exc


def _publish_oof_summary(
    root: Path,
    *,
    fold_results: Sequence[dict[str, Any]],
    pilot_run_dir: Path,
    n_epochs: int,
) -> dict[str, Any]:
    train, contracts = _train_and_fold_contracts()
    if len(fold_results) != FOLD_COUNT:
        raise RealMLPCVError("OOF aggregation requires five validated fold results.")
    all_rows: list[dict[str, str]] = []
    for fold_number, (_, validation_positions) in enumerate(contracts, start=1):
        fold_dir = root / f"fold_{fold_number:02d}"
        validate_realmlp_production_fold(
            fold_dir,
            expected_fold=fold_number,
            expected_ids=tuple(
                train.record_ids.iloc[validation_positions].astype(str).tolist()
            ),
            expected_labels=train.y.iloc[validation_positions].to_numpy(dtype=np.int64),
        )
        for row in _read_prediction_rows(
            fold_dir / "predictions" / "fold_validation_predictions.csv"
        ):
            all_rows.append({**row, "fold": str(fold_number)})
    expected_ids = tuple(train.record_ids.astype(str).tolist())
    expected_labels = dict(
        zip(expected_ids, train.y.astype(int).tolist(), strict=True)
    )
    ordered = _order_and_validate_oof_rows(
        expected_ids, all_rows, expected_labels=expected_labels
    )
    _validate_running_metrics(fold_results)
    metrics = _metric_summary(ordered, fold_results)

    summary = root / "summary"
    working = root / "summary.tmp"
    if summary.exists():
        return validate_realmlp_production_cv(root)
    if working.exists():
        archived = _archive_interrupted_working_directory(working)
    else:
        archived = None
    working.mkdir()
    _write_oof_predictions(working / "oof_predictions.csv", ordered)
    results = {
        "schema_version": 1,
        "contract_id": "REALMLP-PRODUCTION-CV-01",
        "status": "PASS",
        "evidence_scope": "Train-only five-fold CV and OOF evidence; not Validation or Test evidence.",
        "fold_count": FOLD_COUNT,
        "rows": len(ordered),
        "unique_record_ids": len({row["record_id"] for row in ordered}),
        "source_order_preserved": True,
        "configuration": {
            "decision": "FIXED_EPOCH",
            "decision_sha256": _sha256_file(pilot_run_dir / "decision.json"),
            "n_epochs": n_epochs,
            "use_early_stopping": False,
            "use_best_epoch": False,
        },
        "metrics": metrics,
        "archived_interrupted_summary_attempt": archived,
    }
    _write_json(working / "cv_results.json", results)
    hashes = {
        path.name: _sha256_file(path)
        for path in working.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED"}
    }
    manifest = {
        "schema_version": 1,
        "dataset": "dataset_b",
        "model": "realmlp",
        "artifact_version": "realmlp_v1",
        "contract_id": "RMLP-06B",
        "status": "PASS",
        "fold_count": FOLD_COUNT,
        "oof_rows": len(ordered),
        "artifact_sha256": hashes,
    }
    _write_json(working / "manifest.json", manifest)
    (working / "COMPLETED").write_text("RMLP-06B PASS\n", encoding="utf-8")
    _validate_summary_working_files(working)
    os.replace(working, summary)
    return validate_realmlp_production_cv(root)


def _order_and_validate_oof_rows(
    expected_ids: Sequence[str],
    rows: Iterable[dict[str, str]],
    *,
    expected_labels: dict[str, int],
) -> list[dict[str, str]]:
    expected = tuple(expected_ids)
    if len(expected) != EXPECTED_TRAIN_ROWS or len(set(expected)) != len(expected):
        raise RealMLPCVError("Immutable Train must contain exactly 205,310 unique IDs.")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        record_id = row.get("record_id", "")
        if not record_id or record_id in by_id:
            raise RealMLPCVError("OOF rows contain an empty or duplicate record ID.")
        if record_id not in expected_labels:
            raise RealMLPCVError("OOF row is outside immutable Train.")
        try:
            label = int(row["y_true"])
            probability_0 = float(row["probability_0"])
            probability_1 = float(row["probability_1"])
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RealMLPCVError("OOF row contains invalid values.") from exc
        if label != expected_labels[record_id] or fold not in range(1, FOLD_COUNT + 1):
            raise RealMLPCVError("OOF label or fold differs from the locked contract.")
        if (
            not math.isfinite(probability_0)
            or not math.isfinite(probability_1)
            or not 0 <= probability_0 <= 1
            or not 0 <= probability_1 <= 1
            or not math.isclose(probability_0 + probability_1, 1.0, rel_tol=0, abs_tol=1e-6)
        ):
            raise RealMLPCVError("OOF probabilities violate the binary probability contract.")
        by_id[record_id] = {
            "record_id": record_id,
            "y_true": str(label),
            "probability_0": repr(probability_0),
            "probability_1": repr(probability_1),
            "fold": str(fold),
        }
    if set(by_id) != set(expected):
        raise RealMLPCVError("OOF membership does not cover immutable Train exactly once.")
    return [by_id[record_id] for record_id in expected]


def _metric_summary(
    ordered_rows: Sequence[dict[str, str]],
    fold_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fold_aucs = np.asarray(
        [result["metrics"]["roc_auc"] for result in fold_results], dtype=np.float64
    )
    if len(fold_aucs) != FOLD_COUNT or not np.isfinite(fold_aucs).all():
        raise RealMLPCVError("Five finite fold ROC-AUC values are required.")
    labels = np.asarray([int(row["y_true"]) for row in ordered_rows], dtype=np.int64)
    positive = np.asarray(
        [float(row["probability_1"]) for row in ordered_rows], dtype=np.float64
    )
    return {
        "per_fold_roc_auc": [float(value) for value in fold_aucs],
        "mean_roc_auc": float(fold_aucs.mean()),
        "std_roc_auc": float(fold_aucs.std(ddof=0)),
        "std_ddof": 0,
        "oof_roc_auc": float(roc_auc_score(labels, positive)),
    }


def _validate_running_metrics(fold_results: Sequence[dict[str, Any]]) -> None:
    if len(fold_results) != FOLD_COUNT:
        raise RealMLPCVError("Running metrics require five fold results.")
    observed: list[float] = []
    for fold_number, result in enumerate(fold_results, start=1):
        metrics = result.get("metrics", {})
        try:
            observed.append(float(metrics["roc_auc"]))
            stored_mean = float(metrics["running_mean_roc_auc"])
            stored_std = float(metrics["running_std_roc_auc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RealMLPCVError("Fold running metrics are missing or invalid.") from exc
        values = np.asarray(observed, dtype=np.float64)
        if (
            metrics.get("running_completed_folds") != fold_number
            or metrics.get("running_std_ddof") != 0
            or not math.isclose(stored_mean, float(values.mean()), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(stored_std, float(values.std(ddof=0)), rel_tol=0, abs_tol=1e-12)
        ):
            raise RealMLPCVError("Stored fold running mean/std differs from recomputation.")


def _validate_published_oof_order(
    rows: Sequence[dict[str, str]], expected_ids: Sequence[str]
) -> None:
    if tuple(row.get("record_id", "") for row in rows) != tuple(expected_ids):
        raise RealMLPCVError("Published OOF rows are not in immutable Train source order.")
    try:
        positions = [int(row["source_position"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise RealMLPCVError("Published OOF source positions are invalid.") from exc
    if positions != list(range(len(expected_ids))):
        raise RealMLPCVError("Published OOF source positions are not contiguous and ordered.")


def _published_prior_fold_aucs(root: Path, current_fold: int) -> list[float]:
    values = []
    for fold_number in range(1, current_fold):
        result = _read_json(root / f"fold_{fold_number:02d}" / "fold_result.json")
        values.append(float(result["metrics"]["roc_auc"]))
    if len(values) != current_fold - 1 or not np.isfinite(values).all():
        raise RealMLPCVError("Prior production folds are incomplete or invalid.")
    return values


def _train_and_fold_contracts():
    try:
        train = DatasetBSplitLoader().load("train")
        contracts = list(DatasetBSplitter.from_yaml().cv_splitter().split(train.X, train.y))
    except Exception as exc:
        raise RealMLPCVError("Cannot load immutable Train-only CV contracts.") from exc
    if len(train.record_ids) != EXPECTED_TRAIN_ROWS or len(contracts) != FOLD_COUNT:
        raise RealMLPCVError("Immutable Train/CV cardinality differs from RMLP-06B.")
    return train, contracts


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RealMLPCVError(f"Cannot read prediction artifact: {path}") from exc
    required = {"record_id", "y_true", "probability_0", "probability_1"}
    if any(not required.issubset(row) for row in rows):
        raise RealMLPCVError(f"Prediction artifact schema is invalid: {path}")
    return rows


def _write_oof_predictions(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_position",
                "record_id",
                "y_true",
                "probability_0",
                "probability_1",
                "fold",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for position, row in enumerate(rows):
            writer.writerow({"source_position": position, **row})


def _validate_fold_working_files(directory: Path) -> None:
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != _FOLD_BASE_FILES:
        raise RealMLPCVError(
            f"Production working fold artifact set differs: {sorted(actual ^ _FOLD_BASE_FILES)}"
        )
    state = _read_json(directory / "run_state.json")
    if state.get("state") != "COMPLETED" or state.get("contract_id") != "RMLP-06B":
        raise RealMLPCVError("Production working fold state is invalid.")


def _validate_summary_working_files(directory: Path) -> None:
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != _SUMMARY_FILES:
        raise RealMLPCVError(
            f"Production working summary artifact set differs: {sorted(actual ^ _SUMMARY_FILES)}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPCVError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RealMLPCVError(f"JSON artifact must be an object: {path}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run, resume, or validate production Train-only RealMLP CV."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--pilot-run-dir", type=Path, default=DEFAULT_PILOT_RUN_DIR)
    run_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    run_parser.add_argument("--device-index", type=int, default=0)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_realmlp_production_cv(
                args.output_root,
                pilot_run_dir=args.pilot_run_dir,
                runtime_dir=args.runtime_dir,
                device_index=args.device_index,
            )
        else:
            manifest = validate_realmlp_production_cv(args.output_root)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except RealMLPCVError as exc:
        print(f"Dataset B production RealMLP CV failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
