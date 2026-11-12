"""LR-07: Train-only CV and fresh full-Train Logistic Regression fitting.

Provenance and frozen-model notice
----------------------------------
The frozen LR-07 bundle ``artifacts/dataset_b/lr/{working,final}/
lr_final_15f_20261002_01`` was produced on 2026-10-02 by an inline Python
script that was piped to the interpreter and never saved as a repository file.
This module was reconstructed on 2026-11-12, after the model was frozen, from
that original script as recorded in the execution session log. It preserves
the original procedure, parameters, checks and output schema; the differences
are engineering-only (atomic write to a temporary directory, validate instead
of overwrite, canonical import paths, no ``FAILED.json`` on error).

This module was NOT used to create the frozen bundle, and its ``run`` function
has NOT been executed on the real data. LR-07 is frozen: do not retrain it. Use
``validate_lr_final_training`` (read-only, no fitting, no model loading) to
check the frozen bundle. With the default output root, ``run`` for
``lr_final_15f_20261002_01`` only validates the existing working bundle and
returns without fitting.

Procedure
---------
The original LR-07 procedure and output schema are:

* five fold-local fits on the run-scoped LR preprocessing export
  (``lr_train.npz`` / ``lr_fold_validation.npz``), one fresh estimator per fold;
* Train-only out-of-fold (OOF) predictions with exactly-once coverage;
* one fresh full-Train ``Pipeline(make_preprocessing_pipeline("lr"),
  LogisticRegression)`` fit on the quality-gated Train partition;
* persisted pipeline, preprocessor and model checkpoints with reload checks;
* a manifest with ``source_sha256`` and ``artifact_sha256`` lineage.

The fixed protocol configuration from ``model_protocol_v1.yaml`` is used
without hyperparameter search. Validation and Test are never loaded here.
An existing run directory is never overwritten: ``run`` validates it instead.

Usage::

    python -m src.dataset_b.models.classical.lr.training run --run-id <run_id>
    python -m src.dataset_b.models.classical.lr.training validate \
        --run-id <run_id> --run-dir artifacts/dataset_b/lr/final/<run_id>
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.dataset_b.data import DatasetBInputQualityGate, DatasetBSplitLoader
from src.dataset_b.models.classical import make_preprocessing_pipeline
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


CONTRACT_ID = "LR-07"
MODEL_KEY = "lr"
PREPROCESSING_VERSION = "dataset_b_preprocessing_v2"
EXPECTED_TRAIN_ROWS = 205_310
EXPECTED_RAW_FEATURES = 15
EXPECTED_SEMANTIC_FEATURES = 19
EXPECTED_ENCODED_FEATURES = 62
RELOAD_CHECK_ROWS = 4096
EXPECTED_PARAMETERS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "fit_intercept": True,
    "class_weight": "balanced",
}
EXPECTED_CROSS_VALIDATION = {
    "population": "train_only",
    "splitter": "StratifiedKFold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "trial_budget_per_model": 1,
    "primary_metric": "roc_auc",
    "stability_statistic": "standard_deviation_roc_auc",
    "hyperparameter_search": False,
}
DEFAULT_PREPROCESSING_ROOT = PROJECT_ROOT / "data/preprocessing/dataset_b_v1/cv_preprocessing_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/dataset_b/lr/working"
DEFAULT_PROTOCOL_PATH = CONFIG_DIR / "dataset_b/model_protocol_v1.yaml"
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
DEFAULT_SPLIT_CONTRACT_PATH = CONFIG_DIR / "dataset_b/split_v1.yaml"
CHECKPOINT_NAMES = ("fitted_pipeline.joblib", "fitted_preprocessor.joblib", "lr_model.joblib")
PAYLOAD_NAMES = (
    "cv_results.json",
    "cv_oof_predictions.csv",
    "full_train_result.json",
    "warnings.json",
    "checkpoints/fitted_pipeline.joblib",
    "checkpoints/fitted_preprocessor.joblib",
    "checkpoints/lr_model.joblib",
)
REQUIRED_BUNDLE_FILES = frozenset({"manifest.json", "COMPLETED", *[name for name in PAYLOAD_NAMES if "/" not in name]})
SOURCE_HASH_KEYS = ("model_protocol_v1.yaml", "feature_set_v1.yaml", "split_v1.yaml", "preprocessing_manifest.json")
OOF_COLUMNS = ("record_id", "y_approved", "p_approved", "fold")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class LRTrainingError(RuntimeError):
    """Raised when LR-07 violates its Train-only or artifact contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LRTrainingError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LRTrainingError(f"Cannot read JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_run_id(run_id: str) -> str:
    _require(isinstance(run_id, str) and bool(_RUN_ID.fullmatch(run_id)), "Invalid explicit LR run_id.")
    _require("legacy" not in run_id.lower(), "Legacy LR run IDs are prohibited.")
    return run_id


def _display_path(path: Path) -> str:
    """Return a project-relative POSIX path when possible, as LR-07 recorded it."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_protocol(path: Path) -> dict[str, Any]:
    try:
        protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LRTrainingError(f"Cannot read model protocol: {path}") from exc
    _require(isinstance(protocol, dict), "LR-07 model protocol must be a mapping.")
    _require(
        protocol.get("dataset") == "dataset_b" and protocol.get("protocol_version") == "model_protocol_v1",
        "LR-07 protocol identity mismatch.",
    )
    parameters = protocol.get("models", {}).get(MODEL_KEY, {}).get("parameters")
    _require(parameters == EXPECTED_PARAMETERS, "LR parameters differ from the fixed balanced protocol.")
    _require(protocol.get("cross_validation") == EXPECTED_CROSS_VALIDATION, "LR-07 CV protocol mismatch.")
    return protocol


def _validate_preprocessing_manifest(directory: Path, run_id: str) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    required = {
        "run_id": run_id,
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_version": "split_v1",
        "feature_version": "feature_set_v1",
        "source_partition": "train",
        "fit_scope": "fold_train_only",
        "use": "cross_validation_only",
        "raw_feature_count": EXPECTED_RAW_FEATURES,
        "semantic_feature_count": EXPECTED_SEMANTIC_FEATURES,
        "encoded_feature_count": EXPECTED_ENCODED_FEATURES,
    }
    for key, expected in required.items():
        _require(manifest.get(key) == expected, f"LR-07 preprocessing manifest mismatch for {key}.")
    _require(manifest.get("blank_row_audit", {}).get("status") == "PASS", "LR-07 preprocessing quality gate did not pass.")
    normalization = manifest.get("normalization_audit", {})
    _require(normalization.get("raw_equals_normalized") is True, "LR-07 normalization equality gate failed.")
    _require(normalization.get("semantic_1111_count") == 0, "LR-07 semantic 1111 values remain.")
    _require(
        manifest.get("fold_validation_coverage") == {
            "rows": EXPECTED_TRAIN_ROWS, "minimum_assignments": 1, "maximum_assignments": 1, "exactly_once": True,
        },
        "LR-07 preprocessing fold coverage mismatch.",
    )
    folds = manifest.get("folds")
    _require(isinstance(folds, list) and [item.get("fold") for item in folds] == [1, 2, 3, 4, 5], "LR-07 requires exactly folds 1-5.")
    for key in ("raw_feature_columns", "semantic_columns", "output_columns"):
        _require(isinstance(manifest.get(key), list), f"LR-07 preprocessing manifest is missing {key}.")
    _require(len(manifest["output_columns"]) == EXPECTED_ENCODED_FEATURES, "LR-07 encoded column list differs.")
    return manifest


def _load_fold(directory: Path, fold: Mapping[str, Any]) -> tuple[Any, Any, dict[str, np.ndarray]]:
    fold_dir = directory / str(fold["directory"])
    try:
        train = sparse.load_npz(fold_dir / "lr_train.npz")
        validation = sparse.load_npz(fold_dir / "lr_fold_validation.npz")
        with np.load(fold_dir / "rows.npz", allow_pickle=False) as loaded:
            rows = {name: loaded[name] for name in loaded.files}
    except Exception as exc:
        raise LRTrainingError(f"Cannot read LR fold artifact: {fold_dir}") from exc
    number = int(fold["fold"])
    for matrix, expected_rows in ((train, len(rows["train_positions"])), (validation, len(rows["fold_validation_positions"]))):
        _require(sparse.isspmatrix_csr(matrix), f"LR-07 fold {number} matrix is not CSR.")
        _require(matrix.dtype == np.float64, f"LR-07 fold {number} matrix dtype mismatch.")
        _require(matrix.shape == (expected_rows, EXPECTED_ENCODED_FEATURES), f"LR-07 fold {number} matrix shape mismatch.")
        _require(np.isfinite(matrix.data).all(), f"LR-07 fold {number} contains non-finite matrix values.")
    return train, validation, rows


def _new_estimator(parameters: Mapping[str, Any]) -> LogisticRegression:
    """Return a new balanced LR instance for one and only one fit."""
    _require(dict(parameters) == EXPECTED_PARAMETERS, "Every LR estimator must use the fixed protocol parameters.")
    return LogisticRegression(**dict(parameters))


def _warning_records(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    return [
        {
            "category": item.category.__name__,
            "message": str(item.message),
            "is_convergence_warning": issubclass(item.category, ConvergenceWarning),
        }
        for item in caught
    ]


def _n_iter(estimator: LogisticRegression) -> list[int]:
    return [int(value) for value in np.asarray(estimator.n_iter_).tolist()]


def validate_lr_final_training(run_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate an existing LR-07 bundle without fitting, loading models or inference."""
    run_id = _validated_run_id(run_id)
    root = Path(run_dir)
    observed = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()
    _require(observed in (set(REQUIRED_BUNDLE_FILES), set(REQUIRED_BUNDLE_FILES) | {"FROZEN"}), "LR-07 bundle is incomplete or unexpected.")
    checkpoint_dir = root / "checkpoints"
    _require(
        checkpoint_dir.is_dir() and {p.name for p in checkpoint_dir.iterdir() if p.is_file()} == set(CHECKPOINT_NAMES),
        "LR-07 checkpoint set is incomplete or unexpected.",
    )
    _require((root / "COMPLETED").read_text(encoding="utf-8") == f"{CONTRACT_ID} PASS\n", "LR-07 completion marker differs.")

    manifest = _read_json(root / "manifest.json")
    _require(manifest.get("status") == "PASS" and manifest.get("contract_id") == CONTRACT_ID, "LR-07 manifest is not PASS.")
    _require(manifest.get("run_id") == run_id and manifest.get("model_key") == MODEL_KEY, "LR-07 lineage differs.")
    _require(manifest.get("dataset") == "dataset_b" and manifest.get("protocol_version") == "model_protocol_v1", "LR-07 protocol identity differs.")
    _require(manifest.get("feature_version") == "feature_set_v1" and manifest.get("split_version") == "split_v1", "LR-07 feature/split version differs.")
    _require(manifest.get("preprocessing_version") == PREPROCESSING_VERSION, "LR-07 preprocessing version differs.")
    _require(manifest.get("preprocessing_run_id") == run_id, "LR-07 preprocessing run lineage differs.")
    _require(manifest.get("source_partition") == "train", "LR-07 is not Train-only.")
    _require(manifest.get("validation_accessed") is False and manifest.get("test_accessed") is False, "LR-07 crossed a holdout boundary.")
    _require(manifest.get("configuration_confirmed_from_fixed_protocol") is True, "LR-07 fixed-protocol confirmation is missing.")
    _require(manifest.get("hyperparameter_search") is False, "LR-07 may not perform hyperparameter search.")
    _require(manifest.get("model_parameters") == EXPECTED_PARAMETERS, "LR-07 model parameters differ.")
    schemas = manifest.get("schemas", {})
    _require(
        (schemas.get("raw_feature_count"), schemas.get("semantic_feature_count"), schemas.get("encoded_feature_count"))
        == (EXPECTED_RAW_FEATURES, EXPECTED_SEMANTIC_FEATURES, EXPECTED_ENCODED_FEATURES),
        "LR-07 schema evidence differs.",
    )
    _require(len(schemas.get("encoded_columns", [])) == EXPECTED_ENCODED_FEATURES, "LR-07 encoded column list differs.")
    quality = manifest.get("quality_audit", {})
    _require(quality.get("blank_row_audit", {}).get("status") == "PASS", "LR-07 quality audit differs.")
    _require(
        manifest.get("persisted_objects") == {
            "pipeline": "checkpoints/fitted_pipeline.joblib",
            "preprocessor": "checkpoints/fitted_preprocessor.joblib",
            "model": "checkpoints/lr_model.joblib",
        },
        "LR-07 persisted-object inventory differs.",
    )
    _require(isinstance(manifest.get("library_versions"), dict), "LR-07 library-version evidence is missing.")
    source_hashes = manifest.get("source_sha256", {})
    _require(isinstance(source_hashes, dict) and tuple(source_hashes) == SOURCE_HASH_KEYS, "LR-07 source hash inventory differs.")
    artifact_hashes = manifest.get("artifact_sha256", {})
    _require(isinstance(artifact_hashes, dict) and tuple(artifact_hashes) == PAYLOAD_NAMES, "LR-07 artifact hash inventory differs.")
    for name in PAYLOAD_NAMES:
        _require(artifact_hashes[name] == _sha256(root / name), f"LR-07 artifact hash differs: {name}.")

    cv = _read_json(root / "cv_results.json")
    _require(cv == manifest.get("cv"), "LR-07 CV summary differs from manifest.")
    _require(cv.get("status") == "PASS" and cv.get("stage") == "LR-07_train_only_cv" and cv.get("run_id") == run_id, "LR-07 CV identity differs.")
    _require(cv.get("fold_count") == 5 and [fold.get("fold") for fold in cv.get("folds", [])] == [1, 2, 3, 4, 5], "LR-07 does not contain five folds.")
    _require(cv.get("fresh_estimator_per_fold") is True, "LR-07 does not prove fresh fold estimators.")
    _require(cv.get("hyperparameter_search") is False and cv.get("trial_budget_per_model") == 1, "LR-07 CV search boundary differs.")
    _require(cv.get("validation_accessed") is False and cv.get("test_accessed") is False, "LR CV crossed a holdout boundary.")
    _require(
        (cv.get("oof_rows"), cv.get("oof_unique_record_ids"), cv.get("oof_coverage_min"), cv.get("oof_coverage_max"))
        == (EXPECTED_TRAIN_ROWS, EXPECTED_TRAIN_ROWS, 1, 1),
        "LR-07 OOF summary differs.",
    )
    oof = pd.read_csv(root / "cv_oof_predictions.csv", dtype={"record_id": str}, keep_default_na=False)
    _require(tuple(oof.columns) == OOF_COLUMNS, "LR-07 OOF columns differ.")
    _require(len(oof) == oof["record_id"].nunique() == EXPECTED_TRAIN_ROWS, "LR-07 OOF IDs are incomplete or duplicated.")
    oof_y = oof["y_approved"].to_numpy(dtype=np.int8)
    oof_p = oof["p_approved"].to_numpy(dtype=np.float64)
    _require(set(np.unique(oof_y).tolist()) == {0, 1}, "LR-07 OOF labels are invalid.")
    _require(np.isfinite(oof_p).all() and np.all((oof_p >= 0) & (oof_p <= 1)), "LR-07 OOF probabilities are invalid.")
    _require(sorted(oof["fold"].astype(int).unique().tolist()) == [1, 2, 3, 4, 5], "LR-07 OOF folds differ.")
    _require(np.isclose(roc_auc_score(oof_y, oof_p), cv.get("oof_roc_auc"), rtol=0, atol=1e-12), "LR-07 OOF ROC AUC does not reproduce.")

    full = _read_json(root / "full_train_result.json")
    _require(full == manifest.get("full_train"), "LR-07 full-Train summary differs from manifest.")
    _require(full.get("status") == "PASS" and full.get("stage") == "LR-07_fresh_full_train_fit" and full.get("run_id") == run_id, "LR-07 full-Train identity differs.")
    _require(full.get("retained_train_rows") == EXPECTED_TRAIN_ROWS, "LR-07 full-Train population differs.")
    _require(
        (full.get("raw_feature_count"), full.get("semantic_feature_count"), full.get("encoded_feature_count"))
        == (EXPECTED_RAW_FEATURES, EXPECTED_SEMANTIC_FEATURES, EXPECTED_ENCODED_FEATURES),
        "LR-07 full-Train feature counts differ.",
    )
    _require(full.get("matrix_format") == "csr" and full.get("matrix_dtype") == "float64" and full.get("matrix_finite") is True, "LR-07 full-Train matrix evidence differs.")
    _require(full.get("fresh_preprocessor") is True and full.get("fresh_estimator") is True, "LR-07 full-Train objects were not fresh.")
    _require(full.get("model_parameters") == EXPECTED_PARAMETERS and full.get("class_order") == [0, 1], "LR-07 full-Train estimator evidence differs.")
    _require(
        full.get("reload_predictions_exact") is True and full.get("reload_prediction_rows_checked") == min(RELOAD_CHECK_ROWS, EXPECTED_TRAIN_ROWS),
        "LR-07 reload prediction verification is missing.",
    )
    _require(full.get("validation_accessed") is False and full.get("test_accessed") is False, "LR full-Train crossed a holdout boundary.")
    return manifest


def run_lr_final_training(
    run_id: str,
    *,
    preprocessing_root: str | Path = DEFAULT_PREPROCESSING_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    split_contract_path: str | Path = DEFAULT_SPLIT_CONTRACT_PATH,
    split_manifest_path: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create the LR-07 bundle; this is the only function that fits LR models."""
    run_id = _validated_run_id(run_id)
    pre_dir = Path(preprocessing_root) / run_id
    _require(pre_dir.is_dir(), "Run-scoped LR preprocessing directory is missing.")
    out_dir = Path(output_root) / run_id
    if out_dir.exists():
        return {**validate_lr_final_training(out_dir, run_id=run_id), "resume_action": "SKIPPED_VALID_COMPLETED"}
    protocol = _load_protocol(Path(protocol_path))
    params = dict(protocol["models"][MODEL_KEY]["parameters"])
    prep_manifest = _validate_preprocessing_manifest(pre_dir, run_id)

    loader_kwargs: dict[str, Any] = {}
    if split_manifest_path is not None:
        loader_kwargs["manifest_path"] = split_manifest_path
    if split_dir is not None:
        loader_kwargs["split_dir"] = split_dir
    if feature_contract_path != DEFAULT_FEATURE_CONTRACT_PATH:
        loader_kwargs["feature_contract_path"] = feature_contract_path
    train = DatasetBInputQualityGate().apply(DatasetBSplitLoader(**loader_kwargs).load("train"))
    n_rows = len(train.X)
    _require(
        n_rows == EXPECTED_TRAIN_ROWS and len(train.record_ids) == n_rows and len(train.y) == n_rows,
        "LR-07 retained Train row count mismatch.",
    )
    source_ids = train.record_ids.astype(str).to_numpy()
    source_y = train.y.to_numpy(dtype=np.int8)
    _require(len(np.unique(source_ids)) == n_rows, "LR-07 Train record IDs are not unique.")

    oof = np.full(n_rows, np.nan, dtype=np.float64)
    oof_fold = np.zeros(n_rows, dtype=np.int8)
    fold_results: list[dict[str, Any]] = []
    all_warning_records: list[dict[str, Any]] = []
    cv_started = time.perf_counter()
    for fold_entry in prep_manifest["folds"]:
        fold = int(fold_entry["fold"])
        X_train, X_val, rows = _load_fold(pre_dir, fold_entry)
        train_pos = rows["train_positions"].astype(np.int64)
        val_pos = rows["fold_validation_positions"].astype(np.int64)
        _require(np.array_equal(rows["train_record_ids"].astype(str), source_ids[train_pos]), f"LR-07 fold {fold} train ID alignment mismatch.")
        _require(np.array_equal(rows["fold_validation_record_ids"].astype(str), source_ids[val_pos]), f"LR-07 fold {fold} validation ID alignment mismatch.")
        _require(np.array_equal(rows["train_y"], source_y[train_pos]), f"LR-07 fold {fold} train label alignment mismatch.")
        _require(np.array_equal(rows["fold_validation_y"], source_y[val_pos]), f"LR-07 fold {fold} validation label alignment mismatch.")
        _require(not np.isfinite(oof[val_pos]).any() and not (oof_fold[val_pos] != 0).any(), f"LR-07 fold {fold} OOF overlap.")

        estimator = _new_estimator(params)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X_train, source_y[train_pos])
        elapsed = time.perf_counter() - started
        _require(np.array_equal(estimator.classes_, np.array([0, 1])), f"LR-07 fold {fold} estimator class order mismatch.")
        probabilities = estimator.predict_proba(X_val)[:, 1]
        _require(
            np.isfinite(probabilities).all() and not ((probabilities < 0) | (probabilities > 1)).any(),
            f"LR-07 fold {fold} probability contract failed.",
        )
        auc = float(roc_auc_score(source_y[val_pos], probabilities))
        warning_records = _warning_records(caught)
        all_warning_records.extend({"stage": "cv", "fold": fold, **item} for item in warning_records)
        oof[val_pos] = probabilities
        oof_fold[val_pos] = fold
        fold_results.append({
            "fold": fold,
            "train_rows": int(len(train_pos)),
            "fold_validation_rows": int(len(val_pos)),
            "roc_auc": auc,
            "fit_seconds": elapsed,
            "n_iter": _n_iter(estimator),
            "warnings": warning_records,
            "convergence_warning_count": sum(item["is_convergence_warning"] for item in warning_records),
        })
        print(f"LR-07 fold={fold} roc_auc={auc:.10f} fit_seconds={elapsed:.3f} warnings={len(warning_records)}", flush=True)
    cv_elapsed = time.perf_counter() - cv_started
    _require(np.isfinite(oof).all() and int((oof_fold > 0).sum()) == n_rows, "LR-07 OOF coverage is incomplete.")
    _require(set(np.unique(oof_fold).tolist()) == {1, 2, 3, 4, 5}, "LR-07 OOF fold assignment mismatch.")

    fold_auc = np.asarray([item["roc_auc"] for item in fold_results], dtype=np.float64)
    cv_result = {
        "status": "PASS",
        "stage": "LR-07_train_only_cv",
        "run_id": run_id,
        "source_partition": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "fold_count": 5,
        "oof_rows": n_rows,
        "oof_unique_record_ids": int(len(np.unique(source_ids))),
        "oof_coverage_min": int((oof_fold > 0).min()),
        "oof_coverage_max": int((oof_fold > 0).max()),
        "oof_roc_auc": float(roc_auc_score(source_y, oof)),
        "mean_fold_roc_auc": float(fold_auc.mean()),
        "std_fold_roc_auc": float(fold_auc.std(ddof=0)),
        "cv_elapsed_seconds": cv_elapsed,
        "fresh_estimator_per_fold": True,
        "hyperparameter_search": False,
        "trial_budget_per_model": 1,
        "model_parameters": params,
        "folds": fold_results,
    }

    full_pipeline = Pipeline([
        ("preprocess", make_preprocessing_pipeline(MODEL_KEY)),
        ("estimator", _new_estimator(params)),
    ])
    full_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        full_pipeline.fit(train.X, train.y)
    full_elapsed = time.perf_counter() - full_started
    full_warning_records = _warning_records(caught)
    all_warning_records.extend({"stage": "full_train", "fold": None, **item} for item in full_warning_records)
    preprocessor = full_pipeline.named_steps["preprocess"]
    estimator = full_pipeline.named_steps["estimator"]
    transformed = preprocessor.transform(train.X)
    _require(
        sparse.isspmatrix_csr(transformed) and transformed.shape == (n_rows, EXPECTED_ENCODED_FEATURES),
        "LR-07 full-Train transformed matrix contract failed.",
    )
    _require(transformed.dtype == np.float64 and np.isfinite(transformed.data).all(), "LR-07 full-Train dtype/finiteness contract failed.")
    _require(np.array_equal(estimator.classes_, np.array([0, 1])), "LR-07 full-Train estimator class order mismatch.")

    tmp = out_dir.with_name(f".{run_id}.tmp")
    _require(not tmp.exists(), "LR-07 temporary destination already exists.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir()
    (tmp / "checkpoints").mkdir()
    try:
        _write_json(tmp / "cv_results.json", cv_result)
        pd.DataFrame({
            "record_id": source_ids,
            "y_approved": source_y,
            "p_approved": oof,
            "fold": oof_fold,
        }).to_csv(tmp / "cv_oof_predictions.csv", index=False, float_format="%.17g")

        pipeline_path = tmp / "checkpoints/fitted_pipeline.joblib"
        preprocessor_path = tmp / "checkpoints/fitted_preprocessor.joblib"
        model_path = tmp / "checkpoints/lr_model.joblib"
        joblib.dump(full_pipeline, pipeline_path)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(estimator, model_path)

        sample_positions = np.linspace(0, n_rows - 1, num=min(RELOAD_CHECK_ROWS, n_rows), dtype=np.int64)
        sample_X = train.X.iloc[sample_positions]
        expected_sample = full_pipeline.predict_proba(sample_X)[:, 1]
        reloaded_sample = joblib.load(pipeline_path).predict_proba(sample_X)[:, 1]
        _require(np.array_equal(expected_sample, reloaded_sample), "LR-07 reloaded pipeline predictions differ.")
        reloaded_matrix = joblib.load(preprocessor_path).transform(sample_X)
        separate_sample = joblib.load(model_path).predict_proba(reloaded_matrix)[:, 1]
        _require(np.array_equal(expected_sample, separate_sample), "LR-07 separately persisted preprocessor/model predictions differ.")

        full_result = {
            "status": "PASS",
            "stage": "LR-07_fresh_full_train_fit",
            "run_id": run_id,
            "source_partition": "train",
            "validation_accessed": False,
            "test_accessed": False,
            "retained_train_rows": n_rows,
            "raw_feature_count": EXPECTED_RAW_FEATURES,
            "semantic_feature_count": EXPECTED_SEMANTIC_FEATURES,
            "encoded_feature_count": EXPECTED_ENCODED_FEATURES,
            "matrix_format": "csr",
            "matrix_dtype": "float64",
            "matrix_finite": True,
            "fresh_preprocessor": True,
            "fresh_estimator": True,
            "model_parameters": params,
            "class_order": [int(v) for v in estimator.classes_.tolist()],
            "n_iter": _n_iter(estimator),
            "fit_seconds": full_elapsed,
            "warnings": full_warning_records,
            "convergence_warning_count": sum(item["is_convergence_warning"] for item in full_warning_records),
            "reload_prediction_rows_checked": int(len(sample_positions)),
            "reload_predictions_exact": True,
        }
        _write_json(tmp / "full_train_result.json", full_result)
        _write_json(tmp / "warnings.json", {
            "run_id": run_id,
            "warning_count": len(all_warning_records),
            "convergence_warning_count": sum(item["is_convergence_warning"] for item in all_warning_records),
            "warnings": all_warning_records,
        })

        manifest = {
            "status": "PASS",
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "dataset": "dataset_b",
            "model_key": MODEL_KEY,
            "protocol_version": "model_protocol_v1",
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_run_id": prep_manifest["run_id"],
            "preprocessing_manifest": _display_path(pre_dir / "manifest.json"),
            "source_partition": "train",
            "validation_accessed": False,
            "test_accessed": False,
            "configuration_confirmed_from_fixed_protocol": True,
            "hyperparameter_search": False,
            "model_parameters": params,
            "schemas": {
                "raw_feature_count": EXPECTED_RAW_FEATURES,
                "semantic_feature_count": EXPECTED_SEMANTIC_FEATURES,
                "encoded_feature_count": EXPECTED_ENCODED_FEATURES,
                "raw_feature_columns": prep_manifest["raw_feature_columns"],
                "semantic_columns": prep_manifest["semantic_columns"],
                "encoded_columns": prep_manifest["output_columns"],
            },
            "quality_audit": {
                "blank_row_audit": prep_manifest["blank_row_audit"],
                "normalization_audit": prep_manifest["normalization_audit"],
            },
            "cv": cv_result,
            "full_train": full_result,
            "persisted_objects": {
                "pipeline": "checkpoints/fitted_pipeline.joblib",
                "preprocessor": "checkpoints/fitted_preprocessor.joblib",
                "model": "checkpoints/lr_model.joblib",
            },
            "library_versions": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "scipy": version("scipy"),
                "joblib": version("joblib"),
            },
            "source_sha256": {
                "model_protocol_v1.yaml": _sha256(Path(protocol_path)),
                "feature_set_v1.yaml": _sha256(Path(feature_contract_path)),
                "split_v1.yaml": _sha256(Path(split_contract_path)),
                "preprocessing_manifest.json": _sha256(pre_dir / "manifest.json"),
            },
            "artifact_sha256": {name: _sha256(tmp / name) for name in PAYLOAD_NAMES},
        }
        _write_json(tmp / "manifest.json", manifest)
        (tmp / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        os.replace(tmp, out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return validate_lr_final_training(out_dir, run_id=run_id)


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run or validate LR-07 (Train-only CV and full-Train LR fit).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--preprocessing-root", type=Path, default=DEFAULT_PREPROCESSING_ROOT)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate = sub.add_parser("validate")
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        value = run_lr_final_training(args.run_id, preprocessing_root=args.preprocessing_root, output_root=args.output_root)
    else:
        value = validate_lr_final_training(args.run_dir, run_id=args.run_id)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
