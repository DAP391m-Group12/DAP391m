"""Canonical Final 15-feature Train-only pilot, CV, and full-Train execution for RealMLP."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.dataset_b.models.neural.realmlp.estimator import load_realmlp_v1_config

from .final_contract import (
    EXPECTED_ROWS,
    RealMLPFinalContractError,
    RealMLPFinalPaths,
    base_manifest,
    begin_stage,
    fit_native_preprocessing,
    load_quality_gated_partition,
    publish_stage,
    read_json,
    transform_native_preprocessing,
    validate_manifest,
    validate_probabilities,
    write_json,
)
from .history import HistoryEnabledRealMLPClassifier, RealMLPHistoryRecorder


FitModel = Callable[..., tuple[Any, np.ndarray | None, list[dict[str, Any]], float]]


class RealMLPFinalTrainingError(RealMLPFinalContractError):
    """Raised when final RealMLP Train-only execution violates its contract."""


def _fit_model(
    X: Any,
    y: np.ndarray,
    *,
    directory: Path,
    stage: str,
    epochs: int,
    batch_size: int,
    device_index: int,
    X_predict: Any | None = None,
    X_val: Any | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[Any, np.ndarray | None, list[dict[str, Any]], float]:
    history_path = directory / "training_history.jsonl"
    recorder = RealMLPHistoryRecorder(
        history_path,
        device_index=device_index,
        expected_epochs=epochs,
        stage=stage,
    )
    estimator = (
        HistoryEnabledRealMLPClassifier.from_yaml()
        .set_params(
            device=f"cuda:{device_index}",
            n_epochs=epochs,
            batch_size=batch_size,
            use_early_stopping=False,
            use_best_epoch=False,
        )
        .attach_history_recorder(recorder)
    )
    started = time.perf_counter()
    estimator.fit(X, y, X_val=X_val, y_val=y_val)
    elapsed = time.perf_counter() - started
    probabilities = None
    if X_predict is not None:
        probabilities = validate_probabilities(estimator.predict_proba(X_predict), len(X_predict))
    write_json(
        directory / "training_history.json",
        {"records": recorder.records, "expected_epochs": epochs, "complete": len(recorder.records) == epochs},
    )
    return estimator, probabilities, recorder.records, elapsed


def run_realmlp_pilot(
    run_id: str,
    *,
    artifact_root: str | Path,
    batch_size: int = 256,
    device_index: int = 0,
    fit_model: FitModel = _fit_model,
) -> dict[str, Any]:
    """Run one Train-only pilot and publish evidence awaiting a fixed-epoch decision."""
    config = load_realmlp_v1_config()
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    temporary, destination = begin_stage(paths, "pilot")
    gated = load_quality_gated_partition("train")
    positions = np.arange(len(gated.X))
    pilot_positions, _ = train_test_split(
        positions,
        train_size=float(config["pilot"]["sample_fraction"]),
        random_state=int(config["pilot"]["random_state"]),
        stratify=gated.y,
    )
    fit_positions, stop_positions = train_test_split(
        pilot_positions,
        test_size=float(config["pilot"]["inner_stop_fraction"]),
        random_state=int(config["pilot"]["random_state"]),
        stratify=gated.y.iloc[pilot_positions],
    )
    fit_gated = _subset(gated, fit_positions)
    stop_gated = _subset(gated, stop_positions)
    preprocessing, fit_data = fit_native_preprocessing(fit_gated)
    stop_data = transform_native_preprocessing(preprocessing, stop_gated)
    epochs = int(config["pilot"]["maximum_epochs"])
    model, probabilities, history, elapsed = fit_model(
        fit_data.X,
        fit_data.y,
        directory=temporary,
        stage="pilot",
        epochs=epochs,
        batch_size=batch_size,
        device_index=device_index,
        X_predict=stop_data.X,
        X_val=stop_data.X,
        y_val=stop_data.y,
    )
    if probabilities is None:
        raise RealMLPFinalTrainingError("Pilot did not produce inner-stop probabilities.")
    joblib.dump({"estimator": model, "preprocessing": preprocessing, "scope": "train_only_pilot"}, temporary / "pilot_model.joblib")
    _write_ids(temporary / "pilot_fit_ids.csv", fit_data.record_ids)
    _write_ids(temporary / "pilot_inner_stop_ids.csv", stop_data.record_ids)
    _write_predictions(temporary / "pilot_predictions.csv", stop_data.record_ids, stop_data.y, probabilities)
    result = {
        **base_manifest(run_id, "pilot", status="AWAITING_DECISION"),
        "status": "AWAITING_DECISION",
        "pilot_fit_rows": len(fit_data.X),
        "pilot_inner_stop_rows": len(stop_data.X),
        "maximum_epochs": epochs,
        "completed_epochs": len(history),
        "elapsed_seconds": elapsed,
        "quality_gate": gated.quality_audit.as_dict(),
        "execution": {"fit_calls": 1, "predict_proba_calls": 1},
    }
    write_json(temporary / "pilot_result.json", result)
    write_json(temporary / "manifest.json", result)
    publish_stage(temporary, destination)
    return validate_realmlp_pilot(paths)


def lock_realmlp_fixed_epoch(
    run_id: str,
    fixed_epoch: int,
    *,
    artifact_root: str | Path,
    rationale: str,
) -> dict[str, Any]:
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    pilot = validate_realmlp_pilot(paths)
    maximum = int(pilot["maximum_epochs"])
    if isinstance(fixed_epoch, bool) or not 1 <= fixed_epoch <= maximum:
        raise RealMLPFinalTrainingError(f"fixed_epoch must be within 1..{maximum}.")
    if not rationale.strip():
        raise RealMLPFinalTrainingError("A fixed-epoch decision requires a rationale.")
    decision_path = paths.stage("pilot") / "decision.json"
    if decision_path.exists():
        raise RealMLPFinalTrainingError("RealMLP pilot decision already exists.")
    decision = {
        **base_manifest(run_id, "pilot_decision"),
        "decision": "FIXED_EPOCH",
        "fixed_epoch": fixed_epoch,
        "rationale": rationale.strip(),
        "source_completed_epochs": pilot["completed_epochs"],
    }
    write_json(decision_path, decision)
    return decision


def run_realmlp_cv(
    run_id: str,
    *,
    artifact_root: str | Path,
    batch_size: int = 256,
    device_index: int = 0,
    fit_model: FitModel = _fit_model,
) -> dict[str, Any]:
    """Run five fresh Train-only folds and publish exactly-once OOF predictions."""
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    temporary, destination = begin_stage(paths, "cv")
    fixed_epoch = _fixed_epoch(paths)
    gated = load_quality_gated_partition("train")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full((len(gated.X), 2), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (fit_positions, validation_positions) in enumerate(
        splitter.split(gated.X, gated.y), start=1
    ):
        fold_dir = temporary / f"fold_{fold:02d}"
        fold_dir.mkdir()
        fit_gated = _subset(gated, fit_positions)
        validation_gated = _subset(gated, validation_positions)
        preprocessing, fit_data = fit_native_preprocessing(fit_gated)
        validation_data = transform_native_preprocessing(preprocessing, validation_gated)
        model, probabilities, history, elapsed = fit_model(
            fit_data.X,
            fit_data.y,
            directory=fold_dir,
            stage=f"cv_fold_{fold:02d}",
            epochs=fixed_epoch,
            batch_size=batch_size,
            device_index=device_index,
            X_predict=validation_data.X,
            X_val=None,
            y_val=None,
        )
        if probabilities is None:
            raise RealMLPFinalTrainingError(f"CV fold {fold} produced no probabilities.")
        oof[validation_positions] = probabilities
        joblib.dump(
            {"estimator": model, "preprocessing": preprocessing, "scope": f"cv_fold_{fold:02d}"},
            fold_dir / "fitted_model.joblib",
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(fit_positions),
                "fold_validation_rows": len(validation_positions),
                "roc_auc": float(roc_auc_score(validation_data.y, probabilities[:, 1])),
                "fit_seconds": elapsed,
                "completed_epochs": len(history),
            }
        )
    validate_probabilities(oof, len(gated.X))
    _write_predictions(
        temporary / "oof_predictions.csv",
        tuple(gated.record_ids.astype(str).tolist()),
        gated.y.to_numpy(dtype=np.int8),
        oof,
    )
    result = {
        **base_manifest(run_id, "cv"),
        "fixed_epoch": fixed_epoch,
        "fold_count": 5,
        "oof_rows": len(oof),
        "oof_unique_record_ids": int(gated.record_ids.astype(str).nunique()),
        "oof_coverage": "exactly_once",
        "folds": fold_rows,
        "mean_roc_auc": float(np.mean([row["roc_auc"] for row in fold_rows])),
        "std_roc_auc": float(np.std([row["roc_auc"] for row in fold_rows], ddof=0)),
        "quality_gate": gated.quality_audit.as_dict(),
    }
    write_json(temporary / "cv_results.json", result)
    write_json(temporary / "manifest.json", result)
    publish_stage(temporary, destination)
    return validate_realmlp_cv(paths)


def run_realmlp_full_train(
    run_id: str,
    *,
    artifact_root: str | Path,
    batch_size: int = 256,
    device_index: int = 0,
    fit_model: FitModel = _fit_model,
) -> dict[str, Any]:
    """Fit one fresh final estimator on all immutable Train rows."""
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    validate_realmlp_cv(paths)
    temporary, destination = begin_stage(paths, "full_train")
    fixed_epoch = _fixed_epoch(paths)
    gated = load_quality_gated_partition("train")
    preprocessing, train_data = fit_native_preprocessing(gated)
    model, probabilities, history, elapsed = fit_model(
        train_data.X,
        train_data.y,
        directory=temporary,
        stage="full_train",
        epochs=fixed_epoch,
        batch_size=batch_size,
        device_index=device_index,
        X_predict=None,
        X_val=None,
        y_val=None,
    )
    if probabilities is not None:
        raise RealMLPFinalTrainingError("Full-Train must not create holdout predictions.")
    joblib.dump(
        {"estimator": model, "preprocessing": preprocessing, "scope": "all_immutable_train_rows"},
        temporary / "fitted_model.joblib",
    )
    result = {
        **base_manifest(run_id, "full_train"),
        "fixed_epoch": fixed_epoch,
        "train_rows": len(train_data.X),
        "unique_train_record_ids": len(set(train_data.record_ids)),
        "completed_epochs": len(history),
        "fit_seconds": elapsed,
        "quality_gate": gated.quality_audit.as_dict(),
        "execution": {"fit_calls": 1, "predict_proba_calls": 0},
    }
    write_json(temporary / "full_train_result.json", result)
    write_json(temporary / "manifest.json", result)
    publish_stage(temporary, destination)
    return validate_realmlp_full_train(paths)


def validate_realmlp_pilot(paths: RealMLPFinalPaths) -> dict[str, Any]:
    payload = read_json(paths.stage("pilot") / "manifest.json")
    expected = base_manifest(paths.run_id, "pilot", status="AWAITING_DECISION")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RealMLPFinalTrainingError("RealMLP pilot manifest is invalid.")
    if payload.get("completed_epochs") != payload.get("maximum_epochs"):
        raise RealMLPFinalTrainingError("RealMLP pilot history is incomplete.")
    fit_ids = _read_ids(paths.stage("pilot") / "pilot_fit_ids.csv")
    stop_ids = _read_ids(paths.stage("pilot") / "pilot_inner_stop_ids.csv")
    if (
        len(fit_ids) != payload.get("pilot_fit_rows")
        or len(stop_ids) != payload.get("pilot_inner_stop_rows")
        or len(set(fit_ids)) != len(fit_ids)
        or len(set(stop_ids)) != len(stop_ids)
        or set(fit_ids).intersection(stop_ids)
    ):
        raise RealMLPFinalTrainingError("RealMLP pilot ID membership is invalid.")
    prediction_ids = [
        row["record_id"]
        for row in _read_predictions(paths.stage("pilot") / "pilot_predictions.csv")
    ]
    if prediction_ids != stop_ids:
        raise RealMLPFinalTrainingError(
            "Pilot predictions do not preserve inner-stop record_id order."
        )
    return payload


def validate_realmlp_cv(paths: RealMLPFinalPaths) -> dict[str, Any]:
    root = paths.stage("cv")
    payload = validate_manifest(read_json(root / "manifest.json"), run_id=paths.run_id, stage="cv")
    rows = _read_predictions(root / "oof_predictions.csv")
    if (
        payload.get("fold_count") != 5
        or payload.get("oof_rows") != EXPECTED_ROWS["train"]
        or len(rows) != EXPECTED_ROWS["train"]
        or len({row["record_id"] for row in rows}) != EXPECTED_ROWS["train"]
    ):
        raise RealMLPFinalTrainingError("RealMLP OOF coverage is invalid.")
    return payload


def validate_realmlp_full_train(paths: RealMLPFinalPaths) -> dict[str, Any]:
    root = paths.stage("full_train")
    payload = validate_manifest(
        read_json(root / "manifest.json"), run_id=paths.run_id, stage="full_train"
    )
    if (
        payload.get("train_rows") != EXPECTED_ROWS["train"]
        or payload.get("unique_train_record_ids") != EXPECTED_ROWS["train"]
        or payload.get("completed_epochs") != payload.get("fixed_epoch")
        or not (root / "fitted_model.joblib").is_file()
    ):
        raise RealMLPFinalTrainingError("RealMLP full-Train artifact is invalid.")
    return payload


def _fixed_epoch(paths: RealMLPFinalPaths) -> int:
    decision = validate_manifest(
        read_json(paths.stage("pilot") / "decision.json"),
        run_id=paths.run_id,
        stage="pilot_decision",
    )
    value = decision.get("fixed_epoch")
    if decision.get("decision") != "FIXED_EPOCH" or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RealMLPFinalTrainingError("RealMLP fixed-epoch decision is invalid.")
    return value


def _subset(gated: Any, positions: Sequence[int]) -> Any:
    indexes = list(np.asarray(positions, dtype=np.int64))
    return type(gated)(
        name=gated.name,
        X=gated.X.iloc[indexes].reset_index(drop=True),
        y=gated.y.iloc[indexes].reset_index(drop=True),
        record_ids=gated.record_ids.iloc[indexes].reset_index(drop=True),
        audit=gated.audit.iloc[indexes].reset_index(drop=True),
        action_taken=gated.action_taken.iloc[indexes].reset_index(drop=True),
        quality_audit=gated.quality_audit,
    )


def _write_predictions(
    path: Path,
    record_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("record_id", "y_approved", "p_denied", "p_approved"))
        for record_id, label, row in zip(record_ids, labels, probabilities, strict=True):
            writer.writerow((record_id, int(label), repr(float(row[0])), repr(float(row[1]))))


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _write_ids(path: Path, record_ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("record_id",))
        writer.writerows((record_id,) for record_id in record_ids)


def _read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["record_id"] for row in csv.DictReader(handle)]
