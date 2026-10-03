"""Train-only PlainMLP CV and full-Train orchestration for PMLP-F04."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix_csr

from src.dataset_b.models.plain_mlp import PlainMLPClassifier
from src.dataset_b.preprocessing import (
    DatasetBInputQualityGate,
    DatasetBSplitLoader,
    make_plain_mlp_preprocessing_pipeline,
)
from src.dataset_b.splits import DatasetBSplitter
from src.utils.path import CONFIG_DIR, PROJECT_ROOT

from .plain_mlp_engine import (
    BATCH_FALLBACK_SEQUENCE,
    PlainMLPEpochRecord,
    _run_selector_epochs,
    _select_best_epoch,
    make_plain_mlp_final_refit_estimator,
    make_plain_mlp_inner_split,
)


EXPECTED_TRAIN_ROWS = 205_310
EXPECTED_FEATURES = 62
DEFAULT_BATCH_SIZE = 256
FIXED_RELOAD_SUBSET_ROWS = 256
CV_CONTRACT_ID = "PMLP-F04-CV"
FULL_TRAIN_CONTRACT_ID = "PMLP-F04-FULL-TRAIN"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/dataset_b/plain_mlp_v1"
DEFAULT_CV_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "cv"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "full_train"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b/plain_mlp_v1.yaml"
DEFAULT_SPLIT_MANIFEST_PATH = DatasetBSplitLoader.DEFAULT_MANIFEST_PATH
DEFAULT_SPLIT_DIR = DatasetBSplitLoader.DEFAULT_SPLIT_DIR
DEFAULT_FEATURE_CONTRACT_PATH = DatasetBSplitLoader.DEFAULT_FEATURE_CONTRACT_PATH
CHECKPOINT_SCOPE = "Train-only PlainMLP final evidence; no Validation or Test access."


class PlainMLPTrainingError(RuntimeError):
    """Raised when PMLP-F04 orchestration or artifact validation fails."""


PlainMLPFullTrainError = PlainMLPTrainingError
PlainMLPCVError = PlainMLPTrainingError


@dataclass(frozen=True)
class PlainMLPTrainPopulation:
    X: pd.DataFrame
    y: np.ndarray
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlainMLPSelectorEvidence:
    best_epoch: int
    best_inner_stop_bce_loss: float
    stopped_after_epoch: int
    stop_reason: str
    history: tuple[PlainMLPEpochRecord, ...]
    pos_weight: float


SelectorRunner = Callable[
    [csr_matrix, np.ndarray, csr_matrix, np.ndarray, int],
    PlainMLPSelectorEvidence,
]


@dataclass(frozen=True)
class PlainMLPRunnerDependencies:
    load_train: Callable[[], PlainMLPTrainPopulation]
    outer_folds: Callable[[pd.DataFrame, np.ndarray], Iterable[tuple[np.ndarray, np.ndarray]]]
    preprocessor_factory: Callable[[], Any]
    selector_runner: SelectorRunner
    final_estimator_factory: Callable[[int, int], Any]
    checkpoint_loader: Callable[[dict[str, Any]], Any]
    clock: Callable[[], float] = time.perf_counter


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _lineage_matches_feature_provenance(
    stored: Any,
    expected: dict[str, str],
) -> bool:
    """Retain the semantic-lineage helper imported by downstream stages."""
    feature_key = "feature_contract_sha256"
    strict_expected = {
        name: value for name, value in expected.items() if name != feature_key
    }
    if not isinstance(stored, dict) or set(stored) != {
        *strict_expected,
        feature_key,
    }:
        return False
    provenance = stored.get(feature_key)
    return (
        isinstance(provenance, str)
        and len(provenance) == 64
        and all(character in "0123456789abcdef" for character in provenance)
        and all(stored.get(name) == value for name, value in strict_expected.items())
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlainMLPTrainingError(f"Cannot read PlainMLP JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PlainMLPTrainingError(f"PlainMLP JSON artifact must contain an object: {path}")
    return value


def _default_load_train() -> PlainMLPTrainPopulation:
    loader = DatasetBSplitLoader()
    raw_train = loader.load("train")
    train = DatasetBInputQualityGate().apply(raw_train)
    if train.quality_audit.rejected_rows != 0:
        raise PlainMLPTrainingError("PlainMLP quality gate rejected one or more Train rows.")
    return PlainMLPTrainPopulation(
        X=train.X.reset_index(drop=True),
        y=train.y.to_numpy(dtype=np.float64),
        record_ids=tuple(train.record_ids.astype(str).tolist()),
    )


def _default_outer_folds(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    splitter = DatasetBSplitter.from_yaml().cv_splitter()
    return splitter.split(X, y)


def _default_selector_runner(
    X_fit: csr_matrix,
    y_fit: np.ndarray,
    X_stop: csr_matrix,
    y_stop: np.ndarray,
    batch_size: int,
) -> PlainMLPSelectorEvidence:
    stream = _run_selector_epochs(X_fit, y_fit, X_stop, y_stop, batch_size)
    try:
        best_epoch, best_loss, history, stop_reason = _select_best_epoch(stream)
    finally:
        stream.close()
    return PlainMLPSelectorEvidence(
        best_epoch=best_epoch,
        best_inner_stop_bce_loss=best_loss,
        stopped_after_epoch=history[-1].epoch,
        stop_reason=stop_reason,
        history=history,
        pos_weight=PlainMLPClassifier._fit_population_pos_weight(y_fit),
    )


def _default_final_estimator_factory(best_epoch: int, batch_size: int) -> PlainMLPClassifier:
    return make_plain_mlp_final_refit_estimator(best_epoch, batch_size=batch_size)


def default_runner_dependencies() -> PlainMLPRunnerDependencies:
    return PlainMLPRunnerDependencies(
        load_train=_default_load_train,
        outer_folds=_default_outer_folds,
        preprocessor_factory=make_plain_mlp_preprocessing_pipeline,
        selector_runner=_default_selector_runner,
        final_estimator_factory=_default_final_estimator_factory,
        checkpoint_loader=PlainMLPClassifier.from_checkpoint_payload,
    )


def _validated_train_population(
    dependencies: PlainMLPRunnerDependencies,
    expected_rows: int,
) -> PlainMLPTrainPopulation:
    train = dependencies.load_train()
    if not isinstance(train, PlainMLPTrainPopulation):
        raise PlainMLPTrainingError("PlainMLP Train loader returned an invalid population.")
    ids = tuple(str(value) for value in train.record_ids)
    labels = PlainMLPClassifier._validated_labels(train.y, len(ids))
    if (
        len(ids) != expected_rows
        or len(set(ids)) != expected_rows
        or len(train.X) != expected_rows
        or not isinstance(train.X, pd.DataFrame)
    ):
        raise PlainMLPTrainingError("PlainMLP Train membership/order/cardinality is invalid.")
    return PlainMLPTrainPopulation(
        X=train.X.reset_index(drop=True),
        y=labels,
        record_ids=ids,
    )


def _validate_matrix(matrix: Any, expected_rows: int) -> csr_matrix:
    if (
        not isspmatrix_csr(matrix)
        or matrix.shape != (expected_rows, EXPECTED_FEATURES)
        or matrix.dtype != np.float64
        or not np.isfinite(matrix.data).all()
    ):
        raise PlainMLPTrainingError(
            "PlainMLP preprocessing must produce finite CSR float64 with 62 columns."
        )
    return matrix


def _prepare_atomic_directory(destination: Path) -> Path:
    if destination.exists():
        raise PlainMLPTrainingError(
            f"PlainMLP refuses existing historical destination: {destination}"
        )
    working = destination.with_name(destination.name + ".working")
    if working.exists():
        raise PlainMLPTrainingError(
            f"Incomplete PlainMLP working directory was preserved; fresh restart required: {working}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    working.mkdir()
    return working


def _selector_for_raw_population(
    X: pd.DataFrame,
    y: np.ndarray,
    record_ids: tuple[str, ...],
    *,
    batch_size: int,
    dependencies: PlainMLPRunnerDependencies,
    outer_validation_record_ids: Iterable[str] = (),
) -> tuple[PlainMLPSelectorEvidence, dict[str, int], float, list[str]]:
    split = make_plain_mlp_inner_split(
        record_ids,
        y,
        outer_validation_record_ids=outer_validation_record_ids,
    )
    fit_positions = np.asarray(split.fit_positions, dtype=np.int64)
    stop_positions = np.asarray(split.stop_positions, dtype=np.int64)
    selector_preprocessor = dependencies.preprocessor_factory()
    started = dependencies.clock()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_fit = _validate_matrix(
            selector_preprocessor.fit_transform(X.iloc[fit_positions], y[fit_positions]),
            len(fit_positions),
        )
        X_stop = _validate_matrix(
            selector_preprocessor.transform(X.iloc[stop_positions]),
            len(stop_positions),
        )
        evidence = dependencies.selector_runner(
            X_fit,
            y[fit_positions],
            X_stop,
            y[stop_positions],
            batch_size,
        )
    elapsed = dependencies.clock() - started
    return (
        evidence,
        {"inner_fit_rows": len(fit_positions), "inner_stop_rows": len(stop_positions)},
        elapsed,
        [str(item.message) for item in caught],
    )


def _fit_final_estimator(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    best_epoch: int,
    batch_size: int,
    dependencies: PlainMLPRunnerDependencies,
) -> tuple[Any, Any, csr_matrix, float, list[str]]:
    preprocessor = dependencies.preprocessor_factory()
    estimator = dependencies.final_estimator_factory(best_epoch, batch_size)
    started = dependencies.clock()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = _validate_matrix(preprocessor.fit_transform(X, y), len(X))
        estimator.fit(matrix, y)
    return (
        preprocessor,
        estimator,
        matrix,
        dependencies.clock() - started,
        [str(item.message) for item in caught],
    )


def _selector_payload(evidence: PlainMLPSelectorEvidence) -> dict[str, Any]:
    return {
        "best_epoch": evidence.best_epoch,
        "best_inner_stop_bce_loss": evidence.best_inner_stop_bce_loss,
        "stopped_after_epoch": evidence.stopped_after_epoch,
        "stop_reason": evidence.stop_reason,
        "pos_weight": evidence.pos_weight,
        "history": [asdict(record) for record in evidence.history],
        "temporary_state_published": False,
    }


def _dump_bundle(path: Path, preprocessor: Any, estimator: Any, scope: dict[str, Any]) -> None:
    joblib.dump(
        {
            "preprocessing": preprocessor,
            "model_checkpoint": estimator.checkpoint_payload(),
            "scope": scope,
        },
        path,
        compress=3,
    )


def _load_bundle(
    path: Path,
    checkpoint_loader: Callable[[dict[str, Any]], Any],
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise PlainMLPTrainingError("Cannot load PlainMLP fitted bundle.") from exc
    if not isinstance(bundle, dict) or set(bundle) != {
        "preprocessing",
        "model_checkpoint",
        "scope",
    }:
        raise PlainMLPTrainingError("PlainMLP fitted bundle structure is invalid.")
    scope = bundle["scope"]
    if not isinstance(scope, dict) or scope.get("description") != CHECKPOINT_SCOPE:
        raise PlainMLPTrainingError("PlainMLP fitted bundle scope is invalid.")
    return bundle["preprocessing"], checkpoint_loader(bundle["model_checkpoint"]), scope


def load_plain_mlp_bundle(path: str | Path) -> tuple[Any, PlainMLPClassifier, dict[str, Any]]:
    """Load one trusted final-evidence bundle; selector state is never accepted."""
    preprocessor, estimator, scope = _load_bundle(
        Path(path), PlainMLPClassifier.from_checkpoint_payload
    )
    return preprocessor, estimator, scope


def _artifact_manifest(
    working: Path,
    *,
    contract_id: str,
    result_file: str,
) -> dict[str, Any]:
    payload_files = sorted(
        path.relative_to(working).as_posix()
        for path in working.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED"}
    )
    manifest = {
        "schema_version": 1,
        "contract_id": contract_id,
        "status": "PASS",
        "source_partition": "train",
        "project_validation_accessed": False,
        "project_test_accessed": False,
        "result_file": result_file,
        "artifact_sha256": {
            relative: _sha256_file(working / relative) for relative in payload_files
        },
    }
    _write_json(working / "manifest.json", manifest)
    (working / "COMPLETED").write_text(f"{contract_id} PASS\n", encoding="utf-8")
    return manifest


def _validate_artifact_set(directory: Path, contract_id: str) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    if (
        manifest.get("contract_id") != contract_id
        or manifest.get("status") != "PASS"
        or manifest.get("source_partition") != "train"
        or manifest.get("project_validation_accessed") is not False
        or manifest.get("project_test_accessed") is not False
        or (directory / "COMPLETED").read_text(encoding="utf-8")
        != f"{contract_id} PASS\n"
    ):
        raise PlainMLPTrainingError("PlainMLP artifact manifest is invalid.")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise PlainMLPTrainingError("PlainMLP artifact hash coverage is invalid.")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED"}
    }
    if actual != set(hashes):
        raise PlainMLPTrainingError("PlainMLP artifact file set is invalid.")
    for relative, expected in hashes.items():
        if _sha256_file(directory / relative) != expected:
            raise PlainMLPTrainingError(f"PlainMLP artifact hash mismatch: {relative}")
    return manifest


def run_plain_mlp_cv(
    output_dir: str | Path = DEFAULT_CV_OUTPUT_DIR,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dependencies: PlainMLPRunnerDependencies | None = None,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
) -> dict[str, Any]:
    """Run five Train-only outer folds and publish exact-once OOF evidence."""
    if batch_size not in BATCH_FALLBACK_SEQUENCE:
        raise PlainMLPTrainingError("PlainMLP CV batch size is outside the locked sequence.")
    deps = dependencies or default_runner_dependencies()
    destination = Path(output_dir)
    working = _prepare_atomic_directory(destination)
    try:
        train = _validated_train_population(deps, expected_train_rows)
        folds = [
            (np.asarray(fit, dtype=np.int64), np.asarray(valid, dtype=np.int64))
            for fit, valid in deps.outer_folds(train.X, train.y)
        ]
        if len(folds) != 5:
            raise PlainMLPTrainingError("PlainMLP CV requires exactly five outer folds.")
        coverage = np.zeros(expected_train_rows, dtype=np.int8)
        probabilities = np.full((expected_train_rows, 2), np.nan, dtype=np.float64)
        fold_results: list[dict[str, Any]] = []
        for fold_number, (fit_positions, valid_positions) in enumerate(folds, start=1):
            if (
                np.intersect1d(fit_positions, valid_positions).size
                or not np.array_equal(
                    np.sort(np.concatenate((fit_positions, valid_positions))),
                    np.arange(expected_train_rows, dtype=np.int64),
                )
                or np.any(coverage[valid_positions] != 0)
            ):
                raise PlainMLPTrainingError("PlainMLP outer fold partition/coverage is invalid.")
            coverage[valid_positions] += 1
            fit_ids = tuple(train.record_ids[position] for position in fit_positions)
            valid_ids = tuple(train.record_ids[position] for position in valid_positions)
            selector, inner_counts, selector_seconds, selector_warnings = (
                _selector_for_raw_population(
                    train.X.iloc[fit_positions].reset_index(drop=True),
                    train.y[fit_positions],
                    fit_ids,
                    batch_size=batch_size,
                    dependencies=deps,
                    outer_validation_record_ids=valid_ids,
                )
            )
            preprocessor, estimator, _, final_seconds, final_warnings = _fit_final_estimator(
                train.X.iloc[fit_positions].reset_index(drop=True),
                train.y[fit_positions],
                best_epoch=selector.best_epoch,
                batch_size=batch_size,
                dependencies=deps,
            )
            predict_started = deps.clock()
            validation_matrix = _validate_matrix(
                preprocessor.transform(train.X.iloc[valid_positions]), len(valid_positions)
            )
            fold_probabilities = np.asarray(
                estimator.predict_proba(validation_matrix), dtype=np.float64
            )
            predict_seconds = deps.clock() - predict_started
            PlainMLPClassifier._validate_probabilities(fold_probabilities)
            if fold_probabilities.shape != (len(valid_positions), 2):
                raise PlainMLPTrainingError("PlainMLP outer prediction shape is invalid.")
            probabilities[valid_positions] = fold_probabilities

            fold_dir = working / f"fold_{fold_number:02d}"
            fold_dir.mkdir()
            scope = {
                "description": CHECKPOINT_SCOPE,
                "stage": "outer_cv_final_estimator",
                "fold": fold_number,
                "fit_rows": len(fit_positions),
                "fold_validation_rows": len(valid_positions),
                "best_epoch": selector.best_epoch,
                "temporary_selector_state_reused": False,
            }
            _dump_bundle(fold_dir / "final_bundle.joblib", preprocessor, estimator, scope)
            fold_results.append(
                {
                    "fold": fold_number,
                    "fit_rows": len(fit_positions),
                    "fold_validation_rows": len(valid_positions),
                    **inner_counts,
                    "selector": _selector_payload(selector),
                    "final_estimator": {
                        "best_epoch": selector.best_epoch,
                        "pos_weight": float(estimator.pos_weight_),
                        "training_history": list(estimator.training_history_),
                        "evidence": True,
                        "predict_proba_calls": 1,
                    },
                    "timing_seconds": {
                        "selector": selector_seconds,
                        "final_fit": final_seconds,
                        "outer_prediction": predict_seconds,
                    },
                    "warnings": selector_warnings + final_warnings,
                }
            )
        if (
            not np.all(coverage == 1)
            or np.isnan(probabilities).any()
            or len(set(train.record_ids)) != expected_train_rows
        ):
            raise PlainMLPTrainingError("PlainMLP OOF coverage is not exactly 1/1.")
        np.savez_compressed(
            working / "oof_predictions.npz",
            record_ids=np.asarray(train.record_ids, dtype=str),
            y=train.y.astype(np.int8),
            probabilities=probabilities,
            coverage=coverage,
        )
        result = {
            "schema_version": 1,
            "contract_id": CV_CONTRACT_ID,
            "status": "PASS",
            "train_rows": expected_train_rows,
            "unique_record_ids": len(set(train.record_ids)),
            "record_ids_sha256": _sha256_ids(train.record_ids),
            "outer_folds": 5,
            "coverage": {"minimum": 1, "maximum": 1, "exactly_once": True},
            "initializations": {"temporary_selectors": 5, "final_estimators": 5},
            "fit_call_accounting": {
                "selector_preprocessing_fit": 5,
                "selector_model_fit": 5,
                "final_preprocessing_fit": 5,
                "final_model_fit": 5,
                "outer_fold_validation_predict_proba": 5,
            },
            "final_evidence_estimators": 5,
            "historical_artifacts_used": False,
            "folds": fold_results,
        }
        _write_json(working / "cv_result.json", result)
        _artifact_manifest(
            working, contract_id=CV_CONTRACT_ID, result_file="cv_result.json"
        )
        _validate_plain_mlp_cv(working, expected_train_rows, deps.checkpoint_loader)
        os.replace(working, destination)
        return _validate_plain_mlp_cv(
            destination, expected_train_rows, deps.checkpoint_loader
        )
    except PlainMLPTrainingError:
        raise
    except Exception as exc:
        raise PlainMLPTrainingError("PlainMLP CV orchestration failed.") from exc


def _validate_plain_mlp_cv(
    directory: Path,
    expected_rows: int,
    checkpoint_loader: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    manifest = _validate_artifact_set(directory, CV_CONTRACT_ID)
    result = _read_json(directory / "cv_result.json")
    with np.load(directory / "oof_predictions.npz", allow_pickle=False) as payload:
        ids = payload["record_ids"].astype(str)
        probabilities = payload["probabilities"]
        coverage = payload["coverage"]
    if (
        result.get("status") != "PASS"
        or result.get("outer_folds") != 5
        or result.get("train_rows") != expected_rows
        or result.get("unique_record_ids") != expected_rows
        or len(ids) != expected_rows
        or len(set(ids.tolist())) != expected_rows
        or probabilities.shape != (expected_rows, 2)
        or probabilities.dtype != np.float64
        or not np.isfinite(probabilities).all()
        or not np.all(coverage == 1)
        or result.get("final_evidence_estimators") != 5
    ):
        raise PlainMLPTrainingError("PlainMLP CV result/OOF evidence is invalid.")
    PlainMLPClassifier._validate_probabilities(probabilities)
    for fold_number in range(1, 6):
        _, _, scope = _load_bundle(
            directory / f"fold_{fold_number:02d}" / "final_bundle.joblib",
            checkpoint_loader,
        )
        if scope.get("fold") != fold_number or scope.get("temporary_selector_state_reused"):
            raise PlainMLPTrainingError("PlainMLP CV final bundle scope is invalid.")
    return manifest


def validate_plain_mlp_cv(
    output_dir: str | Path = DEFAULT_CV_OUTPUT_DIR,
    *,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
) -> dict[str, Any]:
    return _validate_plain_mlp_cv(
        Path(output_dir), expected_train_rows, PlainMLPClassifier.from_checkpoint_payload
    )


def run_plain_mlp_full_train(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dependencies: PlainMLPRunnerDependencies | None = None,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    **_: Any,
) -> dict[str, Any]:
    """Select an epoch on Train-only inner rows, then refit fresh on all Train."""
    if batch_size not in BATCH_FALLBACK_SEQUENCE:
        raise PlainMLPTrainingError("PlainMLP full-Train batch size is outside the locked sequence.")
    deps = dependencies or default_runner_dependencies()
    destination = Path(output_dir)
    working = _prepare_atomic_directory(destination)
    try:
        train = _validated_train_population(deps, expected_train_rows)
        selector, inner_counts, selector_seconds, selector_warnings = (
            _selector_for_raw_population(
                train.X,
                train.y,
                train.record_ids,
                batch_size=batch_size,
                dependencies=deps,
            )
        )
        preprocessor, estimator, matrix, final_seconds, final_warnings = _fit_final_estimator(
            train.X,
            train.y,
            best_epoch=selector.best_epoch,
            batch_size=batch_size,
            dependencies=deps,
        )
        subset_positions = np.arange(
            min(FIXED_RELOAD_SUBSET_ROWS, expected_train_rows), dtype=np.int64
        )
        before = np.asarray(
            estimator.predict_proba(matrix[subset_positions]), dtype=np.float64
        )
        (working / "checkpoints").mkdir()
        scope = {
            "description": CHECKPOINT_SCOPE,
            "stage": "full_train_final_estimator",
            "train_rows": expected_train_rows,
            "best_epoch": selector.best_epoch,
            "validation_accessed": False,
            "test_accessed": False,
            "temporary_selector_state_reused": False,
        }
        bundle_path = working / "checkpoints" / "fitted_bundle.joblib"
        _dump_bundle(bundle_path, preprocessor, estimator, scope)
        restored_preprocessor, restored_estimator, restored_scope = _load_bundle(
            bundle_path, deps.checkpoint_loader
        )
        restored_matrix = _validate_matrix(
            restored_preprocessor.transform(train.X.iloc[subset_positions]),
            len(subset_positions),
        )
        after = np.asarray(restored_estimator.predict_proba(restored_matrix), dtype=np.float64)
        if not np.array_equal(before, after):
            raise PlainMLPTrainingError(
                "PlainMLP checkpoint reload predictions differ on the fixed ordered Train subset."
            )
        result = {
            "schema_version": 1,
            "contract_id": FULL_TRAIN_CONTRACT_ID,
            "status": "PASS",
            "train_rows": expected_train_rows,
            "unique_record_ids": len(set(train.record_ids)),
            "record_ids_sha256": _sha256_ids(train.record_ids),
            **inner_counts,
            "selector": _selector_payload(selector),
            "final_estimator": {
                "best_epoch": selector.best_epoch,
                "fit_rows": expected_train_rows,
                "pos_weight": float(estimator.pos_weight_),
                "training_history": list(estimator.training_history_),
                "evidence": True,
            },
            "initializations": {"temporary_selectors": 1, "final_estimators": 1},
            "fit_call_accounting": {
                "selector_preprocessing_fit": 1,
                "selector_model_fit": 1,
                "final_preprocessing_fit": 1,
                "final_model_fit": 1,
                "fixed_subset_predict_proba_before_reload": 1,
                "fixed_subset_predict_proba_after_reload": 1,
            },
            "final_evidence_estimators": 1,
            "checkpoint_reload": {
                "performed": True,
                "fixed_ordered_subset_rows": len(subset_positions),
                "positions": subset_positions.tolist(),
                "predictions_exactly_equal": True,
                "scope_equal": restored_scope == scope,
            },
            "timing_seconds": {"selector": selector_seconds, "final_fit": final_seconds},
            "warnings": selector_warnings + final_warnings,
            "historical_artifacts_used": False,
            "project_validation_accessed": False,
            "project_test_accessed": False,
        }
        _write_json(working / "full_train_result.json", result)
        _artifact_manifest(
            working,
            contract_id=FULL_TRAIN_CONTRACT_ID,
            result_file="full_train_result.json",
        )
        _validate_plain_mlp_full_train(working, expected_train_rows, deps.checkpoint_loader)
        os.replace(working, destination)
        return _validate_plain_mlp_full_train(
            destination, expected_train_rows, deps.checkpoint_loader
        )
    except PlainMLPTrainingError:
        raise
    except Exception as exc:
        raise PlainMLPTrainingError("PlainMLP full-Train orchestration failed.") from exc


def _validate_plain_mlp_full_train(
    directory: Path,
    expected_rows: int,
    checkpoint_loader: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    manifest = _validate_artifact_set(directory, FULL_TRAIN_CONTRACT_ID)
    result = _read_json(directory / "full_train_result.json")
    _, estimator, scope = _load_bundle(
        directory / "checkpoints" / "fitted_bundle.joblib", checkpoint_loader
    )
    if (
        result.get("status") != "PASS"
        or result.get("train_rows") != expected_rows
        or result.get("unique_record_ids") != expected_rows
        or result.get("final_evidence_estimators") != 1
        or result.get("historical_artifacts_used") is not False
        or result.get("checkpoint_reload", {}).get("predictions_exactly_equal") is not True
        or scope.get("train_rows") != expected_rows
        or scope.get("temporary_selector_state_reused") is not False
        or len(estimator.training_history_) != result.get("final_estimator", {}).get("best_epoch")
    ):
        raise PlainMLPTrainingError("PlainMLP full-Train result/bundle is invalid.")
    return manifest


def validate_plain_mlp_full_train(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    **_: Any,
) -> dict[str, Any]:
    return _validate_plain_mlp_full_train(
        Path(output_dir),
        expected_train_rows,
        PlainMLPClassifier.from_checkpoint_payload,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate PMLP-F04 stages.")
    parser.add_argument("stage", choices=("cv", "full-train"))
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir or (
        DEFAULT_CV_OUTPUT_DIR if args.stage == "cv" else DEFAULT_OUTPUT_DIR
    )
    try:
        if args.stage == "cv":
            value = run_plain_mlp_cv(output) if args.command == "run" else validate_plain_mlp_cv(output)
        else:
            value = (
                run_plain_mlp_full_train(output)
                if args.command == "run"
                else validate_plain_mlp_full_train(output)
            )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except PlainMLPTrainingError as exc:
        print(f"PMLP-F04 failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
