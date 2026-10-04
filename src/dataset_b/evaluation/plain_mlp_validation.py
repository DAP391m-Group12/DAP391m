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

from src.dataset_b.contracts import (
    DatasetBSemanticValidationError,
    FeatureContractValidation,
    validate_feature_contract,
)
from src.dataset_b.preprocessing import (
    DatasetBInputQualityGate,
    DatasetBSplitLoader,
    InputQualityError,
)
from src.dataset_b.runtime.plain_mlp_full_train import (
    DEFAULT_FEATURE_CONTRACT_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR as DEFAULT_FULL_TRAIN_DIR,
    _sha256_file,
    _sha256_ids,
    _lineage_matches_feature_provenance,
    load_plain_mlp_bundle,
    validate_plain_mlp_full_train,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/dataset_b/plain_mlp_v1/validation"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b/plain_mlp_v1.yaml"
EXPECTED_VALIDATION_ROWS = 43_995
CONTRACT_ID = "PMLP-F05-VALIDATION"
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_denied", "p_approved")
PROBABILITY_SUM_ATOL = 1e-7
PAYLOAD_FILES = {"validation_predictions.csv", "validation_result.json"}
FINAL_FILES = PAYLOAD_FILES | {"run_manifest.json", "COMPLETED"}
LINEAGE_KEYS = {
    "full_train_manifest_sha256",
    "full_train_result_sha256",
    "fitted_bundle_sha256",
    "split_manifest_sha256",
    "validation_source_sha256",
    "feature_contract_sha256",
    "plain_mlp_config_sha256",
}


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
        for record_id, label, row in zip(ids, labels, probabilities, strict=True):
            writer.writerow(
                [record_id, int(label), repr(float(row[0])), repr(float(row[1]))]
            )


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
    config_path: Path,
    feature_contract_provenance: str | None = None,
) -> dict[str, str]:
    paths = {
        "full_train_manifest_sha256": full_train_dir / "manifest.json",
        "full_train_result_sha256": full_train_dir / "full_train_result.json",
        "fitted_bundle_sha256": full_train_dir / "checkpoints/fitted_bundle.joblib",
        "split_manifest_sha256": split_manifest_path,
        "validation_source_sha256": validation_csv,
        "plain_mlp_config_sha256": config_path,
    }
    try:
        lineage = {name: _sha256_file(path) for name, path in paths.items()}
    except OSError as exc:
        raise PlainMLPValidationError("Cannot construct Validation lineage.") from exc
    if feature_contract_provenance is not None:
        lineage["feature_contract_sha256"] = feature_contract_provenance
    return lineage


def _semantic_feature_contract(path: Path) -> FeatureContractValidation:
    try:
        return validate_feature_contract(path)
    except DatasetBSemanticValidationError as exc:
        raise PlainMLPValidationError(
            "Dataset B semantic feature contract is invalid."
        ) from exc


def run_plain_mlp_validation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_id: str,
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
        feature_contract = _semantic_feature_contract(kwargs["feature_contract_path"])
        full_manifest = validate_plain_mlp_full_train(
            kwargs["full_train_dir"],
            split_manifest_path=kwargs["split_manifest_path"],
            split_dir=kwargs["split_dir"],
            feature_contract_path=kwargs["feature_contract_path"],
            config_path=kwargs["config_path"],
        )
        if full_manifest.get("run_id") != run_id:
            raise PlainMLPValidationError("Full-Train and Validation run_id differ.")
        loader = DatasetBSplitLoader(
            manifest_path=kwargs["split_manifest_path"],
            split_dir=kwargs["split_dir"],
            feature_contract_path=kwargs["feature_contract_path"],
        )
        try:
            partition = DatasetBInputQualityGate().apply(loader.load("validation"))
        except InputQualityError as exc:
            raise PlainMLPValidationError(
                f"Validation input quality gate failed: {exc}"
            ) from exc
        quality_audit = partition.quality_audit.as_dict()
        if (
            quality_audit.get("status") != "PASS"
            or quality_audit.get("input_rows") != EXPECTED_VALIDATION_ROWS
            or quality_audit.get("retained_rows") != EXPECTED_VALIDATION_ROWS
            or quality_audit.get("rejected_rows") != 0
            or quality_audit.get("rejected_record_ids") != []
            or quality_audit.get("semantic_1111_count") != 0
            or quality_audit.get("row_count_preserved_after_normalization") is not True
            or quality_audit.get("row_order_preserved") is not True
            or quality_audit.get("target_alignment_preserved") is not True
            or quality_audit.get("record_id_alignment_preserved") is not True
        ):
            raise PlainMLPValidationError("Validation input quality gate did not pass exactly.")
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
            or not np.allclose(
                probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
            )
        ):
            raise PlainMLPValidationError("Validation probability matrix is invalid.")
        approved_index = int(positive[0])
        denied_index = 1 - approved_index
        ordered_probabilities = probabilities[:, [denied_index, approved_index]]
        approved = ordered_probabilities[:, 1]

        destination.parent.mkdir(parents=True, exist_ok=True)
        working.mkdir()
        _write_predictions(
            working / "validation_predictions.csv", ids, labels, ordered_probabilities
        )
        lineage = _lineage(
            kwargs["full_train_dir"],
            split_manifest_path=kwargs["split_manifest_path"],
            validation_csv=kwargs["split_dir"] / "validation.csv",
            config_path=kwargs["config_path"],
            feature_contract_provenance=feature_contract.raw_sha256,
        )
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
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
                "quality_gate": quality_audit,
            },
            "bundle_scope": scope,
            "lineage_sha256": lineage,
        }
        _write_json(working / "validation_result.json", result)
        manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
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
        _validate_plain_mlp_validation_artifact(
            working,
            expected_lineage=lineage,
            expected_ids=ids,
            expected_labels=labels,
        )
        os.replace(working, destination)
        return _validate_plain_mlp_validation_artifact(
            destination,
            expected_lineage=lineage,
            expected_ids=ids,
            expected_labels=labels,
        )
    except PlainMLPValidationError:
        raise
    except Exception as exc:
        raise PlainMLPValidationError(f"Cannot publish PlainMLP Validation artifacts: {exc}") from exc


def _validate_plain_mlp_validation_artifact(
    directory: Path,
    *,
    expected_lineage: dict[str, str] | None = None,
    expected_ids: tuple[str, ...] | None = None,
    expected_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate a published package without reopening protected source data."""
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
    lineage = result.get("lineage_sha256")
    if (
        not isinstance(lineage, dict)
        or set(lineage) != LINEAGE_KEYS
        or (
            expected_lineage is not None
            and not _lineage_matches_feature_provenance(lineage, expected_lineage)
        )
        or manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("status") != "PASS"
        or not isinstance(manifest.get("run_id"), str)
        or manifest.get("run_id") != result.get("run_id")
        or manifest.get("rows") != EXPECTED_VALIDATION_ROWS
        or manifest.get("unique_record_ids") != EXPECTED_VALIDATION_ROWS
        or manifest.get("source_order_preserved") is not True
        or manifest.get("lineage_sha256") != lineage
        or result.get("contract_id") != CONTRACT_ID
        or result.get("status") != "PASS"
        or result.get("stage") != "validation_prediction"
        or result.get("prediction_partition") != "validation"
        or result.get("prediction_file") != "validation_predictions.csv"
        or result.get("prediction_columns") != list(PREDICTION_COLUMNS)
        or result.get("source_full_train_contract_id") != "PMLP-F04-FULL-TRAIN"
        or result.get("lineage_sha256") != lineage
        or result.get("execution") != {
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "preprocessing_transform_calls": 1,
            "predict_proba_calls": 1,
        }
        or result.get("data_boundary", {}).get("project_test_accessed") is not False
        or result.get("data_boundary", {}).get("source_partition") != "validation"
        or result.get("data_boundary", {}).get("validation_rows") != EXPECTED_VALIDATION_ROWS
        or result.get("data_boundary", {}).get("project_validation_accessed") is not True
        or result.get("data_boundary", {}).get("model_fit_performed") is not False
        or result.get("data_boundary", {}).get("preprocessing_fit_performed") is not False
        or result.get("data_boundary", {}).get("quality_gate", {}).get("status") != "PASS"
        or result.get("data_boundary", {}).get("quality_gate", {}).get("input_rows")
        != EXPECTED_VALIDATION_ROWS
        or result.get("data_boundary", {}).get("quality_gate", {}).get("retained_rows")
        != EXPECTED_VALIDATION_ROWS
        or result.get("data_boundary", {}).get("quality_gate", {}).get("rejected_rows") != 0
        or result.get("data_boundary", {}).get("quality_gate", {}).get("rejected_record_ids") != []
        or result.get("data_boundary", {}).get("quality_gate", {}).get("semantic_1111_count") != 0
        or result.get("data_boundary", {})
        .get("quality_gate", {})
        .get("row_count_preserved_after_normalization")
        is not True
        or result.get("data_boundary", {}).get("quality_gate", {}).get("row_order_preserved")
        is not True
        or result.get("data_boundary", {})
        .get("quality_gate", {})
        .get("target_alignment_preserved")
        is not True
        or result.get("data_boundary", {})
        .get("quality_gate", {})
        .get("record_id_alignment_preserved")
        is not True
        or result.get("bundle_scope", {}).get("stage") != "full_train_final_estimator"
        or result.get("bundle_scope", {}).get("validation_accessed") is not False
        or result.get("bundle_scope", {}).get("test_accessed") is not False
        or result.get("bundle_scope", {}).get("temporary_selector_state_reused") is not False
    ):
        raise PlainMLPValidationError("Validation status, boundary, or lineage is invalid.")

    rows = _read_predictions(directory / "validation_predictions.csv")
    try:
        observed_ids = tuple(row["record_id"] for row in rows)
        labels = np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8)
        probabilities = np.asarray(
            [[float(row["p_denied"]), float(row["p_approved"])] for row in rows],
            dtype=np.float64,
        )
    except (KeyError, ValueError) as exc:
        raise PlainMLPValidationError("Validation prediction values are invalid.") from exc
    summary = result.get("prediction_summary", {})
    if (
        len(rows) != EXPECTED_VALIDATION_ROWS
        or len(set(observed_ids)) != EXPECTED_VALIDATION_ROWS
        or labels.shape != (EXPECTED_VALIDATION_ROWS,)
        or set(np.unique(labels).tolist()) != {0, 1}
        or probabilities.shape != (EXPECTED_VALIDATION_ROWS, 2)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
        )
        or result.get("data_boundary", {}).get("validation_record_ids_sha256")
        != _sha256_ids(observed_ids)
        or summary.get("rows") != EXPECTED_VALIDATION_ROWS
        or summary.get("unique_record_ids") != EXPECTED_VALIDATION_ROWS
        or summary.get("approved_rows") != int(np.count_nonzero(labels == 1))
        or summary.get("denied_rows") != int(np.count_nonzero(labels == 0))
        or not np.isclose(summary.get("probability_min", np.nan), np.min(probabilities[:, 1]))
        or not np.isclose(summary.get("probability_max", np.nan), np.max(probabilities[:, 1]))
        or not np.isclose(summary.get("probability_mean", np.nan), np.mean(probabilities[:, 1]))
        or (expected_ids is not None and observed_ids != expected_ids)
        or (expected_labels is not None and not np.array_equal(labels, expected_labels))
    ):
        raise PlainMLPValidationError("Validation IDs, order, labels, or probabilities are invalid.")
    return manifest


def validate_plain_mlp_validation_artifact(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate frozen Validation files without loading a bundle or source partition."""
    return _validate_plain_mlp_validation_artifact(Path(output_dir))


def validate_plain_mlp_validation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Deep-validate saved predictions against the protected source exactly once."""
    full_train_dir = Path(full_train_dir)
    split_manifest_path = Path(split_manifest_path)
    split_dir = Path(split_dir)
    feature_contract_path = Path(feature_contract_path)
    config_path = Path(config_path)
    _semantic_feature_contract(feature_contract_path)
    full_manifest = validate_plain_mlp_full_train(
        full_train_dir,
        split_manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
        config_path=config_path,
    )
    lineage = _lineage(
        full_train_dir,
        split_manifest_path=split_manifest_path,
        validation_csv=split_dir / "validation.csv",
        config_path=config_path,
    )
    loader = DatasetBSplitLoader(
        manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )
    try:
        partition = DatasetBInputQualityGate().apply(loader.load("validation"))
    except InputQualityError as exc:
        raise PlainMLPValidationError(f"Validation input quality gate failed: {exc}") from exc
    expected_ids = tuple(partition.record_ids.astype(str).tolist())
    expected_labels = partition.y.to_numpy(dtype=np.int8)
    if (
        len(expected_ids) != EXPECTED_VALIDATION_ROWS
        or len(set(expected_ids)) != EXPECTED_VALIDATION_ROWS
        or expected_labels.shape != (EXPECTED_VALIDATION_ROWS,)
        or set(np.unique(expected_labels).tolist()) != {0, 1}
    ):
        raise PlainMLPValidationError("Validation IDs or labels violate split_v1.")
    manifest = _validate_plain_mlp_validation_artifact(
        Path(output_dir),
        expected_lineage=lineage,
        expected_ids=expected_ids,
        expected_labels=expected_labels,
    )
    expected_run_id = run_id or full_manifest.get("run_id")
    if manifest.get("run_id") != expected_run_id:
        raise PlainMLPValidationError("Full-Train and Validation run_id differ.")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate PlainMLP Validation inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        sub.add_argument("--full-train-dir", type=Path, default=DEFAULT_FULL_TRAIN_DIR)
        if command == "run":
            sub.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = (
            run_plain_mlp_validation(
                args.output_dir,
                full_train_dir=args.full_train_dir,
                run_id=args.run_id,
            )
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
