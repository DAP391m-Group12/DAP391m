"""Generate and validate one ordered Validation prediction artifact for PlainMLP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.dataset_b.preprocessing import DatasetBSplitLoader
from src.dataset_b.runtime.plain_mlp_full_train import (
    DEFAULT_FEATURE_CONTRACT_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR as DEFAULT_FULL_TRAIN_DIR,
    _sha256_file,
    _sha256_ids,
    load_plain_mlp_bundle,
    validate_plain_mlp_full_train,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/dataset_b/plain_mlp_v1/validation"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b/plain_mlp_v1.yaml"
EXPECTED_VALIDATION_ROWS = 43_995
CONTRACT_ID = "PMLP-04A"
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_approved")
PAYLOAD_FILES = {"validation_predictions.csv", "validation_result.json"}
FINAL_FILES = PAYLOAD_FILES | {"run_manifest.json", "COMPLETED"}


class PlainMLPValidationError(RuntimeError):
    """Raised when PlainMLP Validation inference violates its frozen boundary."""


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlainMLPValidationError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PlainMLPValidationError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_predictions(
    path: Path, ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        for record_id, label, probability in zip(ids, labels, probabilities, strict=True):
            writer.writerow([record_id, int(label), repr(float(probability))])


def _read_predictions(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise PlainMLPValidationError("Validation prediction columns are invalid.")
            return list(reader)
    except OSError as exc:
        raise PlainMLPValidationError("Cannot read Validation predictions.") from exc


def _lineage(
    full_train_dir: Path,
    *,
    split_manifest_path: Path,
    validation_csv: Path,
    feature_contract_path: Path,
    config_path: Path,
) -> dict[str, str]:
    paths = {
        "full_train_manifest_sha256": full_train_dir / "manifest.json",
        "full_train_result_sha256": full_train_dir / "full_train_result.json",
        "fitted_bundle_sha256": full_train_dir / "checkpoints/fitted_bundle.joblib",
        "split_manifest_sha256": split_manifest_path,
        "validation_source_sha256": validation_csv,
        "feature_contract_sha256": feature_contract_path,
        "plain_mlp_config_sha256": config_path,
    }
    try:
        return {name: _sha256_file(path) for name, path in paths.items()}
    except OSError as exc:
        raise PlainMLPValidationError("Cannot construct Validation lineage.") from exc


def run_plain_mlp_validation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Transform and predict Validation once without exposing any fitting API."""
    destination = Path(output_dir)
    kwargs = {
        "full_train_dir": Path(full_train_dir),
        "split_manifest_path": Path(split_manifest_path),
        "split_dir": Path(split_dir),
        "feature_contract_path": Path(feature_contract_path),
        "config_path": Path(config_path),
    }
    if destination.exists():
        result = validate_plain_mlp_validation(destination, **kwargs)
        return {**result, "resume_action": "SKIPPED_VALID_COMPLETED"}
    working = destination.with_name(destination.name + ".tmp")
    if working.exists():
        raise PlainMLPValidationError(
            f"Temporary directory already exists and was preserved: {working}"
        )
    try:
        full_manifest = validate_plain_mlp_full_train(kwargs["full_train_dir"])
        loader = DatasetBSplitLoader(
            manifest_path=kwargs["split_manifest_path"],
            split_dir=kwargs["split_dir"],
            feature_contract_path=kwargs["feature_contract_path"],
        )
        partition = loader.load("validation")
        ids = tuple(partition.record_ids.astype(str).tolist())
        labels = partition.y.to_numpy(dtype=np.int8)
        if (
            len(ids) != EXPECTED_VALIDATION_ROWS
            or len(set(ids)) != EXPECTED_VALIDATION_ROWS
            or labels.shape != (EXPECTED_VALIDATION_ROWS,)
            or set(np.unique(labels).tolist()) != {0, 1}
        ):
            raise PlainMLPValidationError("Validation IDs or labels violate split_v1.")
        preprocessing, estimator, scope = load_plain_mlp_bundle(
            kwargs["full_train_dir"] / "checkpoints/fitted_bundle.joblib"
        )
        transformed = preprocessing.transform(partition.X.reset_index(drop=True))
        probabilities = np.asarray(estimator.predict_proba(transformed), dtype=np.float64)
        classes = np.asarray(estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if (
            probabilities.shape != (EXPECTED_VALIDATION_ROWS, 2)
            or positive.shape != (1,)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0.0) | (probabilities > 1.0))
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise PlainMLPValidationError("Validation probability matrix is invalid.")
        approved = probabilities[:, int(positive[0])]

        destination.parent.mkdir(parents=True, exist_ok=True)
        working.mkdir()
        _write_predictions(working / "validation_predictions.csv", ids, labels, approved)
        lineage = _lineage(
            kwargs["full_train_dir"],
            split_manifest_path=kwargs["split_manifest_path"],
            validation_csv=kwargs["split_dir"] / "validation.csv",
            feature_contract_path=kwargs["feature_contract_path"],
            config_path=kwargs["config_path"],
        )
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "stage": "validation_prediction",
            "prediction_partition": "validation",
            "prediction_file": "validation_predictions.csv",
            "prediction_columns": list(PREDICTION_COLUMNS),
            "source_full_train_contract_id": full_manifest["contract_id"],
            "prediction_summary": {
                "rows": len(ids),
                "unique_record_ids": len(set(ids)),
                "approved_rows": int(np.count_nonzero(labels == 1)),
                "denied_rows": int(np.count_nonzero(labels == 0)),
                "probability_min": float(np.min(approved)),
                "probability_max": float(np.max(approved)),
                "probability_mean": float(np.mean(approved)),
            },
            "execution": {
                "model_fit_calls": 0,
                "preprocessing_fit_calls": 0,
                "preprocessing_transform_calls": 1,
                "predict_proba_calls": 1,
            },
            "data_boundary": {
                "source_partition": "validation",
                "validation_rows": len(ids),
                "validation_record_ids_sha256": _sha256_ids(ids),
                "project_validation_accessed": True,
                "project_test_accessed": False,
                "model_fit_performed": False,
                "preprocessing_fit_performed": False,
            },
            "bundle_scope": scope,
            "lineage_sha256": lineage,
        }
        _write_json(working / "validation_result.json", result)
        manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "rows": len(ids),
            "unique_record_ids": len(set(ids)),
            "source_order_preserved": True,
            "lineage_sha256": lineage,
            "artifact_sha256": {
                name: _sha256_file(working / name) for name in sorted(PAYLOAD_FILES)
            },
        }
        _write_json(working / "run_manifest.json", manifest)
        (working / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        validate_plain_mlp_validation(working, **kwargs)
        os.replace(working, destination)
        return validate_plain_mlp_validation(destination, **kwargs)
    except PlainMLPValidationError:
        raise
    except Exception as exc:
        raise PlainMLPValidationError(f"Cannot publish PlainMLP Validation artifacts: {exc}") from exc


def validate_plain_mlp_validation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Validate saved predictions and lineage without invoking the model."""
    directory = Path(output_dir)
    full_train_dir = Path(full_train_dir)
    split_manifest_path = Path(split_manifest_path)
    split_dir = Path(split_dir)
    feature_contract_path = Path(feature_contract_path)
    config_path = Path(config_path)
    validate_plain_mlp_full_train(full_train_dir)
    actual = {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
    allowed_dirs = {"threshold_selection", "threshold_selection.tmp"}
    directories = {path.name for path in directory.iterdir() if path.is_dir()} if directory.is_dir() else set()
    if actual != FINAL_FILES or directories - allowed_dirs:
        raise PlainMLPValidationError("Validation artifact set is invalid.")
    if (directory / "COMPLETED").read_text(encoding="utf-8") != f"{CONTRACT_ID} PASS\n":
        raise PlainMLPValidationError("Validation completion marker is invalid.")
    manifest = _read_json(directory / "run_manifest.json")
    result = _read_json(directory / "validation_result.json")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != PAYLOAD_FILES:
        raise PlainMLPValidationError("Validation artifact hash coverage is invalid.")
    for name, expected in hashes.items():
        if _sha256_file(directory / name) != expected:
            raise PlainMLPValidationError(f"Validation artifact hash mismatch: {name}")
    lineage = _lineage(
        full_train_dir,
        split_manifest_path=split_manifest_path,
        validation_csv=split_dir / "validation.csv",
        feature_contract_path=feature_contract_path,
        config_path=config_path,
    )
    if (
        manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("rows") != EXPECTED_VALIDATION_ROWS
        or manifest.get("unique_record_ids") != EXPECTED_VALIDATION_ROWS
        or manifest.get("source_order_preserved") is not True
        or manifest.get("lineage_sha256") != lineage
        or result.get("contract_id") != CONTRACT_ID
        or result.get("status") != "PASS"
        or result.get("lineage_sha256") != lineage
        or result.get("execution") != {
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "preprocessing_transform_calls": 1,
            "predict_proba_calls": 1,
        }
        or result.get("data_boundary", {}).get("project_test_accessed") is not False
        or result.get("data_boundary", {}).get("model_fit_performed") is not False
        or result.get("data_boundary", {}).get("preprocessing_fit_performed") is not False
    ):
        raise PlainMLPValidationError("Validation status, boundary, or lineage is invalid.")

    loader = DatasetBSplitLoader(
        manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )
    partition = loader.load("validation")
    expected_ids = tuple(partition.record_ids.astype(str).tolist())
    expected_labels = partition.y.to_numpy(dtype=np.int8)
    rows = _read_predictions(directory / "validation_predictions.csv")
    try:
        observed_ids = tuple(row["record_id"] for row in rows)
        labels = np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8)
        probabilities = np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64)
    except (KeyError, ValueError) as exc:
        raise PlainMLPValidationError("Validation prediction values are invalid.") from exc
    if (
        observed_ids != expected_ids
        or not np.array_equal(labels, expected_labels)
        or len(rows) != EXPECTED_VALIDATION_ROWS
        or len(set(observed_ids)) != EXPECTED_VALIDATION_ROWS
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or result.get("data_boundary", {}).get("validation_record_ids_sha256")
        != _sha256_ids(expected_ids)
    ):
        raise PlainMLPValidationError("Validation IDs, order, labels, or probabilities are invalid.")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate PlainMLP Validation inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        sub.add_argument("--full-train-dir", type=Path, default=DEFAULT_FULL_TRAIN_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = (
            run_plain_mlp_validation(args.output_dir, full_train_dir=args.full_train_dir)
            if args.command == "run"
            else validate_plain_mlp_validation(args.output_dir, full_train_dir=args.full_train_dir)
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except PlainMLPValidationError as exc:
        print(f"PlainMLP Validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
