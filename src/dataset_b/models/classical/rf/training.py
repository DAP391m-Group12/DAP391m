"""RF-07: Train-only CV and fresh full-Train Random Forest fitting.

The entry point requires an explicit run_id.  It consumes only the matching
run-scoped preprocessing export for CV and the Train partition for the final
pipeline; Validation and Test are never loaded here.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.sparse import isspmatrix_csr, load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.dataset_b.data import DatasetBInputQualityGate, DatasetBSplitLoader
from src.dataset_b.models.classical import make_preprocessing_pipeline
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


CONTRACT_ID = "RF-07"
EXPECTED_TRAIN_ROWS = 205_310
EXPECTED_RAW_FEATURES = 15
EXPECTED_SEMANTIC_FEATURES = 19
EXPECTED_ENCODED_FEATURES = 62
DEFAULT_PREPROCESSING_ROOT = PROJECT_ROOT / "data/preprocessing/dataset_b_v1/cv_preprocessing_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/dataset_b/rf/working"
DEFAULT_PROTOCOL_PATH = CONFIG_DIR / "dataset_b/model_protocol_v1.yaml"
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
DEFAULT_SPLIT_CONTRACT_PATH = CONFIG_DIR / "dataset_b/split_v1.yaml"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RFTrainingError(RuntimeError):
    """Raised when RF-07 violates its Train-only or artifact contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RFTrainingError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RFTrainingError(f"Cannot read JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validated_run_id(run_id: str) -> str:
    _require(isinstance(run_id, str) and bool(_RUN_ID.fullmatch(run_id)), "Invalid explicit RF run_id.")
    _require("legacy" not in run_id.lower(), "Legacy RF run IDs are prohibited.")
    return run_id


def _load_protocol(path: Path) -> dict[str, Any]:
    try:
        protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RFTrainingError(f"Cannot read model protocol: {path}") from exc
    params = protocol.get("models", {}).get("rf", {}).get("parameters")
    expected = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced",
        "random_state": 42,
    }
    _require(params == expected, "RF parameters differ from the fixed balanced protocol.")
    _require(protocol.get("cross_validation", {}).get("hyperparameter_search") is False, "RF-07 forbids hyperparameter search.")
    _require(protocol.get("class_handling", {}).get("sample_weight") is False, "RF-07 forbids sample weights.")
    _require(protocol.get("class_handling", {}).get("resampling") is False, "RF-07 forbids resampling.")
    return protocol


def _validate_preprocessing_manifest(directory: Path, run_id: str) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    _require(manifest.get("run_id") == run_id, "Preprocessing run_id differs from RF-07 run_id.")
    _require(manifest.get("source_partition") == "train", "RF-07 preprocessing is not Train-only.")
    _require(manifest.get("fit_scope") == "fold_train_only", "RF-07 preprocessing is not fold-local.")
    _require(manifest.get("use") == "cross_validation_only", "RF-07 received a non-CV preprocessing export.")
    _require(manifest.get("feature_version") == "feature_set_v1", "RF-07 refuses legacy feature artifacts.")
    _require(manifest.get("split_version") == "split_v1", "RF-07 requires split_v1.")
    _require(manifest.get("raw_feature_count") == EXPECTED_RAW_FEATURES, "RF-07 requires the 15-field raw schema.")
    _require(manifest.get("semantic_feature_count") == EXPECTED_SEMANTIC_FEATURES, "RF-07 requires the 19-field semantic schema.")
    _require(manifest.get("encoded_feature_count") == EXPECTED_ENCODED_FEATURES, "RF-07 requires the 62-field encoded schema.")
    transforms = manifest.get("numeric_transforms", {}).get("rf", {})
    _require(transforms.get("log1p_columns") == [], "RF preprocessing must not apply log transforms.")
    _require(transforms.get("robust_scaled_columns") == [], "RF preprocessing must not apply scaling.")
    _require(manifest.get("sample_weights_exported") is False, "RF-07 refuses sample-weight exports.")
    blank = manifest.get("blank_row_audit", {})
    normalization = manifest.get("normalization_audit", {})
    _require(blank.get("status") == "PASS" and blank.get("retained_rows") == EXPECTED_TRAIN_ROWS, "RF preprocessing blank-row gate differs.")
    _require(normalization.get("raw_equals_normalized") is True and normalization.get("semantic_1111_count") == 0, "RF preprocessing normalization gate differs.")
    folds = manifest.get("folds")
    _require(isinstance(folds, list) and [item.get("fold") for item in folds] == [1, 2, 3, 4, 5], "RF-07 requires exactly folds 1-5.")
    coverage = manifest.get("fold_validation_coverage", {})
    _require(coverage.get("exactly_once") is True, "RF fold-validation coverage is not exactly once.")
    return manifest


def _load_fold(directory: Path, fold: Mapping[str, Any]) -> tuple[Any, Any, dict[str, np.ndarray]]:
    fold_dir = directory / str(fold["directory"])
    try:
        train = load_npz(fold_dir / "tree_train.npz")
        validation = load_npz(fold_dir / "tree_fold_validation.npz")
        with np.load(fold_dir / "rows.npz", allow_pickle=False) as loaded:
            rows = {name: loaded[name] for name in loaded.files}
    except Exception as exc:
        raise RFTrainingError(f"Cannot read RF fold artifact: {fold_dir}") from exc
    for matrix, expected_rows in ((train, len(rows["train_y"])), (validation, len(rows["fold_validation_y"]))):
        _require(isspmatrix_csr(matrix), "RF fold matrix must be CSR.")
        _require(matrix.shape == (expected_rows, EXPECTED_ENCODED_FEATURES), "RF fold matrix shape differs.")
        _require(matrix.dtype == np.float64 and np.isfinite(matrix.data).all(), "RF fold matrix dtype/finiteness differs.")
    _require(len(rows["train_positions"]) == len(rows["train_y"]), "RF fold train rows are misaligned.")
    _require(len(rows["fold_validation_positions"]) == len(rows["fold_validation_y"]), "RF fold validation rows are misaligned.")
    return train, validation, rows


def _new_estimator(parameters: Mapping[str, Any]) -> RandomForestClassifier:
    """Return a new balanced RF instance for one and only one fit."""
    _require(parameters.get("class_weight") == "balanced", "Every RF estimator must use class_weight='balanced'.")
    return RandomForestClassifier(**dict(parameters))


def _validate_tree_pipeline(pipeline: Pipeline) -> None:
    preprocess = pipeline.named_steps["preprocess"]
    _require(preprocess.named_steps.get("selective_log1p") == "passthrough", "RF pipeline must not log-transform values.")
    _require(preprocess.named_steps.get("scale_numeric") == "passthrough", "RF pipeline must not scale values.")
    _require(pipeline.named_steps["estimator"].get_params().get("class_weight") == "balanced", "Full-Train RF is not balanced.")


def validate_rf_final_training(run_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate an existing RF-07 bundle without fitting or inference."""
    run_id = _validated_run_id(run_id)
    root = Path(run_dir)
    required = {"manifest.json", "cv_results.json", "oof_predictions.npz", "full_train_result.json", "warnings.json", "COMPLETED"}
    observed = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()
    _require(observed in (required, required | {"FROZEN"}), "RF-07 bundle is incomplete or unexpected.")
    checkpoint_names = {"fitted_pipeline.joblib", "fitted_preprocessor.joblib", "rf_model.joblib"}
    checkpoint_dir = root / "checkpoints"
    _require(checkpoint_dir.is_dir() and {p.name for p in checkpoint_dir.iterdir() if p.is_file()} == checkpoint_names, "RF-07 checkpoint set is incomplete or unexpected.")
    pipeline_path = checkpoint_dir / "fitted_pipeline.joblib"
    manifest = _read_json(root / "manifest.json")
    _require(manifest.get("contract_id") == CONTRACT_ID and manifest.get("status") == "PASS", "RF-07 manifest is not PASS.")
    _require(manifest.get("run_id") == run_id and manifest.get("model_key") == "rf", "RF-07 lineage differs.")
    _require(manifest.get("dataset") == "dataset_b" and manifest.get("protocol_version") == "model_protocol_v1", "RF-07 protocol identity differs.")
    _require(manifest.get("source_partition") == "train", "RF-07 source partition differs.")
    _require(manifest.get("configuration_confirmed_from_fixed_protocol") is True, "RF-07 fixed-protocol confirmation is missing.")
    _require(manifest.get("hyperparameter_search") is False, "RF-07 may not perform hyperparameter search.")
    schemas = manifest.get("schemas", {})
    _require({key: schemas.get(key) for key in ("raw_feature_count", "semantic_feature_count", "encoded_feature_count")} == {"raw_feature_count": 15, "semantic_feature_count": 19, "encoded_feature_count": 62}, "RF-07 schema evidence differs.")
    _require(manifest.get("validation_accessed") is False and manifest.get("test_accessed") is False, "RF-07 crossed a holdout boundary.")
    _require(manifest.get("model_parameters", {}).get("class_weight") == "balanced", "RF-07 is not balanced.")
    _require(manifest.get("preprocessing_run_id") == run_id, "RF-07 preprocessing run lineage differs.")
    quality = manifest.get("quality_audit", {})
    _require(quality.get("status") == "PASS" and quality.get("retained_rows") == EXPECTED_TRAIN_ROWS, "RF-07 quality audit differs.")
    _require(isinstance(manifest.get("library_versions"), dict), "RF-07 library-version evidence is missing.")
    persisted = manifest.get("persisted_objects", {})
    _require(set(persisted.values()) == {f"checkpoints/{name}" for name in checkpoint_names}, "RF-07 persisted-object inventory differs.")
    source_manifest = Path(str(manifest.get("preprocessing_manifest", "")))
    _require(source_manifest.is_file(), "RF-07 preprocessing lineage source is missing.")
    source_paths = manifest.get("source_paths", {})
    for key in ("model_protocol_v1.yaml", "feature_set_v1.yaml", "split_v1.yaml", "split_manifest.json"):
        source_path = Path(str(source_paths.get(key, "")))
        _require(source_path.is_file(), f"RF-07 source is missing: {key}.")
    for relative in ("cv_results.json", "oof_predictions.npz", "full_train_result.json", "warnings.json", "checkpoints/fitted_pipeline.joblib", "checkpoints/fitted_preprocessor.joblib", "checkpoints/rf_model.joblib"):
        _require((root / relative).is_file(), f"RF-07 artifact is missing: {relative}.")
    cv = _read_json(root / "cv_results.json")
    _require(cv.get("fold_count") == 5 and len(cv.get("folds", [])) == 5, "RF-07 does not contain five folds.")
    _require(cv.get("fresh_estimator_per_fold") is True, "RF-07 does not prove fresh fold estimators.")
    _require(cv.get("hyperparameter_search") is False and cv.get("trial_budget_per_model") == 1, "RF-07 CV search boundary differs.")
    _require(cv.get("validation_accessed") is False and cv.get("test_accessed") is False, "RF CV crossed a holdout boundary.")
    with np.load(root / "oof_predictions.npz", allow_pickle=False) as loaded:
        _require(set(loaded.files) == {"record_ids", "y_approved", "p_approved", "coverage"}, "RF-07 OOF schema differs.")
        oof_ids = loaded["record_ids"].astype(str); oof_y = loaded["y_approved"]; oof_p = loaded["p_approved"]; coverage = loaded["coverage"]
    _require(len(oof_ids) == len(set(oof_ids.tolist())) == EXPECTED_TRAIN_ROWS, "RF-07 OOF IDs are incomplete or duplicated.")
    _require(np.array_equal(np.unique(oof_y), np.asarray([0, 1])) and np.isfinite(oof_p).all() and np.all((oof_p >= 0) & (oof_p <= 1)), "RF-07 OOF values are invalid.")
    _require(np.all(coverage == 1), "RF-07 OOF coverage is not exactly once.")
    _require(cv.get("oof_rows") == cv.get("oof_unique_record_ids") == EXPECTED_TRAIN_ROWS and cv.get("oof_coverage_min") == cv.get("oof_coverage_max") == 1, "RF-07 OOF summary differs.")
    _require(np.isclose(cv.get("oof_roc_auc"), roc_auc_score(oof_y, oof_p), rtol=0, atol=1e-15), "RF-07 OOF ROC AUC differs.")
    full = _read_json(root / "full_train_result.json")
    _require(full.get("fresh_preprocessor") is True and full.get("fresh_estimator") is True, "RF full-Train components are not fresh.")
    _require(full.get("class_weight") == "balanced", "RF full-Train estimator is not balanced.")
    _require(full.get("log1p_columns") == [] and full.get("scaled_columns") == [], "RF full-Train preprocessing changed values.")
    _require(full.get("reload_prediction_rows_checked") == min(4096, EXPECTED_TRAIN_ROWS) and full.get("reload_predictions_exact") is True, "RF reload prediction verification is missing.")
    return manifest


def run_rf_final_training(
    run_id: str,
    *,
    preprocessing_root: str | Path = DEFAULT_PREPROCESSING_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    split_manifest_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    feature_contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the RF-07 bundle; this is the only function that fits RF models."""
    run_id = _validated_run_id(run_id)
    pre_dir = Path(preprocessing_root) / run_id
    _require(pre_dir.is_dir(), "Run-scoped preprocessing directory is missing.")
    out_dir = Path(output_root) / run_id
    if out_dir.exists():
        return {**validate_rf_final_training(out_dir, run_id=run_id), "resume_action": "SKIPPED_VALID_COMPLETED"}
    protocol = _load_protocol(Path(protocol_path))
    parameters = protocol["models"]["rf"]["parameters"]
    pre_manifest = _validate_preprocessing_manifest(pre_dir, run_id)
    coverage = np.zeros(EXPECTED_TRAIN_ROWS, dtype=np.int8)
    oof = np.full(EXPECTED_TRAIN_ROWS, np.nan, dtype=np.float64)
    oof_y = np.full(EXPECTED_TRAIN_ROWS, -1, dtype=np.int8)
    oof_ids = np.full(EXPECTED_TRAIN_ROWS, "", dtype=object)
    fold_rows: list[dict[str, Any]] = []
    captured_warnings: list[dict[str, Any]] = []
    estimator_count = 0
    for fold in pre_manifest["folds"]:
        train, validation, rows = _load_fold(pre_dir, fold)
        estimator = _new_estimator(parameters)
        estimator_count += 1
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as fold_warnings:
            warnings.simplefilter("always")
            estimator.fit(train, rows["train_y"])
        fit_seconds = time.perf_counter() - started
        _require(np.array_equal(estimator.classes_, np.asarray([0, 1])), "RF fold class order differs.")
        started = time.perf_counter()
        probabilities = np.asarray(estimator.predict_proba(validation), dtype=np.float64)[:, 1]
        predict_seconds = time.perf_counter() - started
        positions = rows["fold_validation_positions"].astype(np.int64)
        _require(np.all(coverage[positions] == 0), "RF fold validation positions overlap.")
        coverage[positions] += 1
        oof[positions] = probabilities
        oof_y[positions] = rows["fold_validation_y"].astype(np.int8)
        oof_ids[positions] = rows["fold_validation_record_ids"].astype(str)
        fold_rows.append({
            "fold": int(fold["fold"]), "train_rows": len(rows["train_y"]),
            "fold_validation_rows": len(rows["fold_validation_y"]),
            "roc_auc": float(roc_auc_score(rows["fold_validation_y"], probabilities)),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "class_weight": "balanced", "fresh_estimator": True,
            "warnings": [str(item.message) for item in fold_warnings], "warning_count": len(fold_warnings),
        })
        captured_warnings.extend({"stage": "cv", "fold": int(fold["fold"]), "category": item.category.__name__, "message": str(item.message)} for item in fold_warnings)
    _require(estimator_count == 5 and np.all(coverage == 1) and np.isfinite(oof).all(), "RF fold-local fit/OOF coverage failed.")
    loader_kwargs = {}
    if split_manifest_path is not None: loader_kwargs["manifest_path"] = split_manifest_path
    if split_dir is not None: loader_kwargs["split_dir"] = split_dir
    if feature_contract_path is not None: loader_kwargs["feature_contract_path"] = feature_contract_path
    loader = DatasetBSplitLoader(**loader_kwargs)
    gated = DatasetBInputQualityGate().apply(loader.load("train"))
    _require(len(gated.X) == EXPECTED_TRAIN_ROWS and gated.X.shape[1] == EXPECTED_RAW_FEATURES, "RF full-Train population/schema differs.")
    _require(np.array_equal(oof_y, gated.y.to_numpy(dtype=np.int8)), "RF OOF labels are not aligned to full Train order.")
    _require(tuple(oof_ids.tolist()) == tuple(gated.record_ids.astype(str).tolist()), "RF OOF record IDs are not aligned to full Train order.")
    pipeline = Pipeline([("preprocess", make_preprocessing_pipeline("rf")), ("estimator", _new_estimator(parameters))])
    _validate_tree_pipeline(pipeline)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as full_warnings:
        warnings.simplefilter("always")
        pipeline.fit(gated.X.reset_index(drop=True), gated.y.to_numpy(dtype=np.int8))
    fit_seconds = time.perf_counter() - started
    _require(len(pipeline.named_steps["preprocess"].get_feature_names_out()) == EXPECTED_ENCODED_FEATURES, "RF full-Train encoded schema differs.")
    full_matrix = pipeline.named_steps["preprocess"].transform(gated.X.reset_index(drop=True))
    _require(isspmatrix_csr(full_matrix), "RF full-Train matrix must be CSR.")
    _require(full_matrix.shape == (EXPECTED_TRAIN_ROWS, EXPECTED_ENCODED_FEATURES), "RF full-Train matrix shape differs.")
    _require(full_matrix.dtype == np.float64 and np.isfinite(full_matrix.data).all(), "RF full-Train matrix dtype/finiteness differs.")
    quality_audit = gated.quality_audit.as_dict()
    _require(quality_audit.get("status") == "PASS", "RF full-Train quality gate did not PASS.")

    tmp = out_dir.with_name(f".{run_id}.tmp")
    _require(not tmp.exists(), "RF-07 temporary destination already exists.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir()
    (tmp / "checkpoints").mkdir()
    try:
        np.savez_compressed(tmp / "oof_predictions.npz", record_ids=oof_ids.astype(str), y_approved=oof_y, p_approved=oof, coverage=coverage)
        cv = {
            "contract_id": CONTRACT_ID, "run_id": run_id, "fold_count": 5, "folds": fold_rows,
            "mean_fold_roc_auc": float(np.mean([row["roc_auc"] for row in fold_rows])),
            "std_fold_roc_auc": float(np.std([row["roc_auc"] for row in fold_rows])),
            "oof_roc_auc": float(roc_auc_score(gated.y.to_numpy(dtype=np.int8), oof)),
            "oof_rows": len(oof), "oof_unique_record_ids": len(set(oof_ids.tolist())),
            "oof_coverage_min": int(coverage.min()), "oof_coverage_max": int(coverage.max()),
            "fresh_estimator_per_fold": True, "hyperparameter_search": False, "trial_budget_per_model": 1,
            "model_parameters": dict(parameters), "source_partition": "train",
            "validation_accessed": False, "test_accessed": False,
        }
        _write_json(tmp / "cv_results.json", cv)
        joblib.dump(pipeline, tmp / "checkpoints/fitted_pipeline.joblib")
        joblib.dump(pipeline.named_steps["preprocess"], tmp / "checkpoints/fitted_preprocessor.joblib")
        joblib.dump(pipeline.named_steps["estimator"], tmp / "checkpoints/rf_model.joblib")
        reload_rows = min(4096, EXPECTED_TRAIN_ROWS)
        reference_predictions = np.asarray(pipeline.predict_proba(gated.X.iloc[:reload_rows].reset_index(drop=True)), dtype=np.float64)
        reloaded = joblib.load(tmp / "checkpoints/fitted_pipeline.joblib")
        reload_predictions = np.asarray(reloaded.predict_proba(gated.X.iloc[:reload_rows].reset_index(drop=True)), dtype=np.float64)
        reload_exact = bool(np.array_equal(reference_predictions, reload_predictions))
        _require(reload_exact, "RF reloaded pipeline predictions differ.")
        full_train = {
            "contract_id": CONTRACT_ID, "run_id": run_id, "retained_train_rows": len(gated.X),
            "status": "PASS", "stage": "RF-07_fresh_full_train_fit", "source_partition": "train",
            "raw_feature_count": 15, "semantic_feature_count": 19, "encoded_feature_count": 62,
            "matrix_format": "csr", "matrix_dtype": "float64", "matrix_finite": True,
            "fresh_preprocessor": True, "fresh_estimator": True, "class_weight": "balanced",
            "model_parameters": dict(parameters), "class_order": [0, 1],
            "log1p_columns": [], "scaled_columns": [], "fit_seconds": fit_seconds,
            "warnings": [str(item.message) for item in full_warnings], "warning_count": len(full_warnings),
            "reload_prediction_rows_checked": reload_rows, "reload_predictions_exact": reload_exact,
            "validation_accessed": False, "test_accessed": False,
        }
        _write_json(tmp / "full_train_result.json", full_train)
        captured_warnings.extend({"stage": "full_train", "category": item.category.__name__, "message": str(item.message)} for item in full_warnings)
        _write_json(tmp / "warnings.json", {"contract_id": CONTRACT_ID, "run_id": run_id, "warnings": captured_warnings})
        resolved_protocol = Path(protocol_path).resolve(strict=True)
        resolved_feature = Path(feature_contract_path or DEFAULT_FEATURE_CONTRACT_PATH).resolve(strict=True)
        resolved_split_manifest = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH).resolve(strict=True)
        resolved_split_contract = DEFAULT_SPLIT_CONTRACT_PATH.resolve(strict=True)
        manifest = {
            "schema_version": 1, "contract_id": CONTRACT_ID, "status": "PASS", "run_id": run_id,
            "dataset": "dataset_b", "protocol_version": "model_protocol_v1", "source_partition": "train",
            "model_key": "rf", "feature_version": "feature_set_v1", "split_version": "split_v1",
            "preprocessing_run_id": run_id,
            "preprocessing_version": pre_manifest.get("preprocessing_version"),
            "preprocessing_manifest": str((pre_dir / "manifest.json").resolve()),
            "configuration_confirmed_from_fixed_protocol": True, "hyperparameter_search": False,
            "model_parameters": dict(parameters), "quality_audit": quality_audit,
            "cv": cv, "full_train": full_train,
            "schemas": {
                "raw_feature_count": 15, "semantic_feature_count": 19, "encoded_feature_count": 62,
                "raw_feature_columns": list(gated.X.columns),
                "encoded_columns": list(pipeline.named_steps["preprocess"].get_feature_names_out()),
            },
            "persisted_objects": {
                "pipeline": "checkpoints/fitted_pipeline.joblib",
                "preprocessor": "checkpoints/fitted_preprocessor.joblib",
                "model": "checkpoints/rf_model.joblib",
            },
            "library_versions": {
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            },
            "source_paths": {
                "model_protocol_v1.yaml": str(resolved_protocol),
                "feature_set_v1.yaml": str(resolved_feature),
                "split_v1.yaml": str(resolved_split_contract),
                "split_manifest.json": str(resolved_split_manifest),
            },
            "validation_accessed": False, "test_accessed": False,
        }
        _write_json(tmp / "manifest.json", manifest)
        (tmp / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        os.replace(tmp, out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return validate_rf_final_training(out_dir, run_id=run_id)


def parse_args(argv: Sequence[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="Run or validate RF-07.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--run-id", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--run-id", required=True); validate.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = run_rf_final_training(args.run_id) if args.command == "run" else validate_rf_final_training(args.run_dir, run_id=args.run_id)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
