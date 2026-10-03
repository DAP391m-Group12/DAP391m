"""Run and validate LR-08 Validation inference and threshold selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.dataset_b.preprocessing import DatasetBInputQualityGate, DatasetBSplitLoader
from src.utils.path import PROJECT_ROOT


RUN_ID = "lr_final_15f_20261002_01"
CONTRACT_ID = "LR-08"
EXPECTED_VALIDATION_ROWS = 43_995
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifacts/dataset_b/lr/working" / RUN_ID
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "validation"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data/dataset_b/split_v1"
DEFAULT_FEATURE_CONTRACT_PATH = PROJECT_ROOT / "config/dataset_b/feature_set_v1.yaml"
HISTORICAL_LR_DIR = PROJECT_ROOT / "artifacts/dataset_b/lr_baseline_v1"

CANDIDATE_THRESHOLDS = tuple(value / 1000 for value in range(1, 1000))
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_approved")
CANDIDATE_COLUMNS = (
    "threshold",
    "balanced_accuracy",
    "recall_approved",
    "recall_denied",
    "tp_approved",
    "fn_approved",
    "tn_denied",
    "fp_denied",
)
PAYLOAD_FILES = {
    "validation_predictions.csv",
    "threshold_candidates.csv",
    "threshold_selection.json",
    "prediction_manifest.json",
}
FINAL_FILES = PAYLOAD_FILES | {"manifest.json", "COMPLETED"}


class LRValidationError(RuntimeError):
    """Raised when LR-08 violates its frozen Validation boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LRValidationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LRValidationError(f"Cannot read JSON artifact: {path}") from exc
    _require(isinstance(payload, dict), f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LRValidationError(f"Cannot hash required source: {path}") from exc
    return digest.hexdigest()


def _sha256_ids(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolved_paths(run_dir: Path, output_dir: Path) -> dict[str, Path]:
    root = run_dir.resolve(strict=True)
    destination = output_dir.resolve(strict=False)
    expected = (root / "validation").resolve(strict=False)
    _require(destination == expected, "LR-08 output must be current run_dir/validation.")
    _require(destination != HISTORICAL_LR_DIR.resolve(strict=False), "Historical LR output is prohibited.")
    _require(HISTORICAL_LR_DIR.resolve(strict=False) not in destination.parents, "Historical LR output is prohibited.")
    _require(destination.parent == root, "LR-08 destination escaped the current run directory.")
    paths = {
        "run_dir": root,
        "output_dir": destination,
        "temporary_dir": destination.with_name(destination.name + ".tmp"),
        "training_manifest": root / "manifest.json",
        "cv_results": root / "cv_results.json",
        "full_train_result": root / "full_train_result.json",
        "fitted_pipeline": root / "checkpoints/fitted_pipeline.joblib",
    }
    for name in ("training_manifest", "cv_results", "full_train_result", "fitted_pipeline"):
        _require(paths[name].is_file(), f"Missing frozen LR source: {paths[name]}")
    return paths


def _validate_training_lineage(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _read_json(paths["training_manifest"])
    _require(manifest.get("status") == "PASS", "Frozen LR training manifest is not PASS.")
    _require(manifest.get("contract_id") == "LR-07", "Frozen source is not LR-07.")
    _require(manifest.get("run_id") == RUN_ID, "Frozen LR source run_id differs.")
    _require(manifest.get("model_key") == "lr", "Frozen source model is not LR.")
    _require(manifest.get("feature_version") == "feature_set_v1", "Frozen source is not final-feature LR.")
    _require(manifest.get("split_version") == "split_v1", "Frozen source does not use split_v1.")
    _require(manifest.get("validation_accessed") is False, "LR-07 manifest already records Validation access.")
    _require(manifest.get("test_accessed") is False, "LR-07 manifest records Test access.")
    _require(manifest.get("schemas", {}).get("encoded_feature_count") == 62, "Frozen LR source is not 62-feature.")
    recorded = manifest.get("artifact_sha256")
    _require(isinstance(recorded, dict), "LR-07 artifact hashes are missing.")
    source_hashes = {
        "training_manifest_sha256": _sha256_file(paths["training_manifest"]),
        "cv_results_sha256": _sha256_file(paths["cv_results"]),
        "full_train_result_sha256": _sha256_file(paths["full_train_result"]),
        "fitted_pipeline_sha256": _sha256_file(paths["fitted_pipeline"]),
    }
    _require(recorded.get("cv_results.json") == source_hashes["cv_results_sha256"], "LR CV hash differs from manifest.")
    _require(recorded.get("full_train_result.json") == source_hashes["full_train_result_sha256"], "LR full-Train hash differs from manifest.")
    _require(recorded.get("checkpoints/fitted_pipeline.joblib") == source_hashes["fitted_pipeline_sha256"], "Frozen pipeline hash differs from manifest.")
    return manifest, source_hashes


def _load_frozen_pipeline(path: Path):
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise LRValidationError("Cannot load frozen LR full-Train pipeline.") from exc
    _require(tuple(getattr(pipeline, "named_steps", {})) == ("preprocess", "estimator"), "Frozen LR pipeline steps differ.")
    estimator = pipeline.named_steps["estimator"]
    _require(np.array_equal(getattr(estimator, "classes_", None), np.asarray([0, 1])), "Frozen LR class order differs.")
    return pipeline


def _predict_validation_once(pipeline: Any, frame: Any) -> np.ndarray:
    """Call the frozen pipeline exactly once and validate its probability matrix."""

    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=np.float64)
    _require(probabilities.shape == (len(frame), 2), "Validation probability matrix has the wrong shape.")
    _require(np.isfinite(probabilities).all(), "Validation probabilities are not finite.")
    _require(np.all((probabilities >= 0.0) & (probabilities <= 1.0)), "Validation probabilities are outside [0,1].")
    _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), "Validation probability rows do not sum to one.")
    return probabilities[:, 1]


def _candidate_rows(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in CANDIDATE_THRESHOLDS:
        predicted = probabilities >= threshold
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "recall_approved": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
                "recall_denied": float(recall_score(labels, predicted, pos_label=0, zero_division=0)),
                "tp_approved": int(np.count_nonzero((labels == 1) & predicted)),
                "fn_approved": int(np.count_nonzero((labels == 1) & ~predicted)),
                "tn_denied": int(np.count_nonzero((labels == 0) & ~predicted)),
                "fp_denied": int(np.count_nonzero((labels == 0) & predicted)),
            }
        )
    return rows


def _select_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(bool(rows), "Threshold candidate table is empty.")
    best = max(float(row["balanced_accuracy"]) for row in rows)
    maximizing = [dict(row) for row in rows if float(row["balanced_accuracy"]) == best]
    selected = min(
        maximizing,
        key=lambda row: (abs(float(row["threshold"]) - 0.5), float(row["threshold"])),
    )
    return selected, maximizing


def _write_predictions(path: Path, ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        for record_id, label, probability in zip(ids, labels, probabilities, strict=True):
            writer.writerow((record_id, int(label), repr(float(probability))))


def _read_predictions(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS, "Validation prediction columns differ.")
            rows = list(reader)
        ids = tuple(row["record_id"] for row in rows)
        labels = np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8)
        probabilities = np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise LRValidationError("Cannot read Validation prediction artifact.") from exc
    return ids, labels, probabilities


def _write_candidates(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == CANDIDATE_COLUMNS, "Threshold candidate columns differ.")
            return [
                {
                    key: float(row[key]) if key in CANDIDATE_COLUMNS[:4] else int(row[key])
                    for key in CANDIDATE_COLUMNS
                }
                for row in reader
            ]
    except (OSError, KeyError, ValueError) as exc:
        raise LRValidationError("Cannot read threshold candidate artifact.") from exc


def _same_candidates(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    for observed, expected in zip(left, right, strict=True):
        for key in CANDIDATE_COLUMNS:
            if key in CANDIDATE_COLUMNS[:4]:
                if not np.isclose(float(observed[key]), float(expected[key]), rtol=0.0, atol=1e-15):
                    return False
            elif int(observed[key]) != int(expected[key]):
                return False
    return True


def run_lr_validation_threshold(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Publish LR-08 after one frozen-pipeline inference on ordered Validation."""

    paths = _resolved_paths(Path(run_dir), Path(output_dir))
    if paths["output_dir"].exists():
        result = validate_lr_validation_threshold(paths["output_dir"])
        return {**result, "resume_action": "SKIPPED_VALID_COMPLETED"}
    _require(not paths["temporary_dir"].exists(), f"Temporary directory was preserved: {paths['temporary_dir']}")
    training_manifest, source_hashes = _validate_training_lineage(paths)
    pipeline = _load_frozen_pipeline(paths["fitted_pipeline"])

    loader = DatasetBSplitLoader(
        manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )
    raw_partition = loader.load("validation")
    gated = DatasetBInputQualityGate().apply(raw_partition)
    ids = tuple(gated.record_ids.astype(str).tolist())
    labels = gated.y.to_numpy(dtype=np.int8)
    _require(len(ids) == EXPECTED_VALIDATION_ROWS, "Validation quality gate changed the required row count.")
    _require(len(set(ids)) == EXPECTED_VALIDATION_ROWS, "Validation record IDs are not unique.")
    _require(labels.shape == (EXPECTED_VALIDATION_ROWS,), "Validation labels have the wrong shape.")
    _require(set(np.unique(labels).tolist()) == {0, 1}, "Validation labels are not binary.")

    approved = _predict_validation_once(pipeline, gated.X.reset_index(drop=True))
    candidates = _candidate_rows(labels, approved)
    selected, maximizing = _select_threshold(candidates)

    validation_csv = Path(split_dir) / "validation.csv"
    lineage = {
        **source_hashes,
        "preprocessing_manifest_sha256": str(training_manifest["source_sha256"]["preprocessing_manifest.json"]),
        "validation_source_sha256": _sha256_file(validation_csv),
        "split_manifest_sha256": _sha256_file(Path(split_manifest_path)),
        "feature_contract_sha256": _sha256_file(Path(feature_contract_path)),
        "validation_record_ids_sha256": _sha256_ids(ids),
    }
    paths["temporary_dir"].mkdir(parents=False)
    try:
        _write_predictions(paths["temporary_dir"] / "validation_predictions.csv", ids, labels, approved)
        _write_candidates(paths["temporary_dir"] / "threshold_candidates.csv", candidates)
        prediction_manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "stage": "validation_prediction",
            "run_id": RUN_ID,
            "model_key": "lr",
            "source_partition": "validation",
            "rows": len(ids),
            "unique_record_ids": len(set(ids)),
            "ordered_record_ids_sha256": lineage["validation_record_ids_sha256"],
            "prediction_columns": list(PREDICTION_COLUMNS),
            "probability_column": "p_approved",
            "probability_min": float(np.min(approved)),
            "probability_max": float(np.max(approved)),
            "probability_mean": float(np.mean(approved)),
            "quality_audit": gated.quality_audit.as_dict(),
            "execution": {
                "model_fit_calls": 0,
                "preprocessing_fit_calls": 0,
                "predict_proba_calls": 1,
                "project_test_accessed": False,
            },
            "lineage_sha256": lineage,
        }
        _write_json(paths["temporary_dir"] / "prediction_manifest.json", prediction_manifest)
        threshold_result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "stage": "validation_threshold_selection",
            "run_id": RUN_ID,
            "objective": "maximize_validation_balanced_accuracy",
            "candidate_grid": {
                "start": 0.001,
                "stop": 0.999,
                "step": 0.001,
                "inclusive": True,
                "count": len(CANDIDATE_THRESHOLDS),
            },
            "prediction_rule": "p_approved >= threshold -> Approve (y_approved=1)",
            "tie_break": ["minimum_absolute_distance_to_0.5", "lower_threshold"],
            "maximizer_count": len(maximizing),
            "maximizing_thresholds": [row["threshold"] for row in maximizing],
            "selected_threshold": selected["threshold"],
            "selected_metrics": {
                "balanced_accuracy": selected["balanced_accuracy"],
                "recall_approved": selected["recall_approved"],
                "recall_denied": selected["recall_denied"],
            },
            "validation_counts": {
                "rows": len(labels),
                "n_approved": int(np.count_nonzero(labels == 1)),
                "n_denied": int(np.count_nonzero(labels == 0)),
                **{key: selected[key] for key in CANDIDATE_COLUMNS[4:]},
            },
            "source_prediction_sha256": _sha256_file(paths["temporary_dir"] / "validation_predictions.csv"),
            "lineage_sha256": lineage,
            "data_boundary": {
                "source_partition": "validation",
                "project_validation_accessed": True,
                "project_test_accessed": False,
                "model_fit_performed": False,
                "preprocessing_fit_performed": False,
                "predict_proba_calls": 0,
            },
        }
        _write_json(paths["temporary_dir"] / "threshold_selection.json", threshold_result)
        manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "run_id": RUN_ID,
            "model_key": "lr",
            "feature_version": "feature_set_v1",
            "split_version": "split_v1",
            "preprocessing_version": training_manifest["preprocessing_version"],
            "output_path": str(paths["output_dir"]),
            "resolved_output_paths": {
                name: str(paths["output_dir"] / name) for name in sorted(FINAL_FILES)
            },
            "rows": len(ids),
            "unique_record_ids": len(set(ids)),
            "source_order_preserved": True,
            "selected_threshold": selected["threshold"],
            "lineage_sha256": lineage,
            "artifact_sha256": {
                name: _sha256_file(paths["temporary_dir"] / name) for name in sorted(PAYLOAD_FILES)
            },
            "prohibited_outputs_written": {
                "historical_lr_baseline_v1": False,
                "legacy_12_feature_artifacts": False,
                "cv_or_full_train_artifacts": False,
                "test_artifacts": False,
            },
        }
        _write_json(paths["temporary_dir"] / "manifest.json", manifest)
        (paths["temporary_dir"] / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        _validate_lr_validation_artifact(
            paths["temporary_dir"], expected_ids=ids, expected_labels=labels
        )
        os.replace(paths["temporary_dir"], paths["output_dir"])
        return _validate_lr_validation_artifact(
            paths["output_dir"], expected_ids=ids, expected_labels=labels
        )
    except Exception:
        raise


def _validate_lr_validation_artifact(
    directory: Path,
    *,
    expected_ids: tuple[str, ...] | None = None,
    expected_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    _require(directory.is_dir(), f"LR-08 artifact directory is missing: {directory}")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    actual_dirs = {path.name for path in directory.iterdir() if path.is_dir()}
    _require(actual_files == FINAL_FILES and not actual_dirs, "LR-08 artifact set differs.")
    _require((directory / "COMPLETED").read_text(encoding="utf-8") == f"{CONTRACT_ID} PASS\n", "LR-08 completion marker differs.")
    manifest = _read_json(directory / "manifest.json")
    prediction_manifest = _read_json(directory / "prediction_manifest.json")
    threshold = _read_json(directory / "threshold_selection.json")
    hashes = manifest.get("artifact_sha256")
    _require(isinstance(hashes, dict) and set(hashes) == PAYLOAD_FILES, "LR-08 artifact hash coverage differs.")
    for name, expected in hashes.items():
        _require(_sha256_file(directory / name) == expected, f"LR-08 artifact hash mismatch: {name}")

    ids, labels, probabilities = _read_predictions(directory / "validation_predictions.csv")
    _require(len(ids) == EXPECTED_VALIDATION_ROWS, "LR-08 prediction row count differs.")
    _require(len(set(ids)) == EXPECTED_VALIDATION_ROWS, "LR-08 prediction IDs are not unique.")
    _require(labels.shape == probabilities.shape == (EXPECTED_VALIDATION_ROWS,), "LR-08 prediction shapes differ.")
    _require(np.isfinite(probabilities).all(), "LR-08 predictions are not finite.")
    _require(np.all((probabilities >= 0.0) & (probabilities <= 1.0)), "LR-08 predictions are outside [0,1].")
    _require(expected_ids is None or ids == expected_ids, "LR-08 prediction order differs from Validation.")
    _require(expected_labels is None or np.array_equal(labels, expected_labels), "LR-08 labels differ from Validation.")

    recomputed = _candidate_rows(labels, probabilities)
    stored = _read_candidates(directory / "threshold_candidates.csv")
    _require(_same_candidates(stored, recomputed), "LR-08 threshold candidates are not reproducible.")
    selected, maximizing = _select_threshold(recomputed)
    lineage = manifest.get("lineage_sha256")
    _require(isinstance(lineage, dict), "LR-08 lineage is missing.")
    _require(
        manifest.get("contract_id") == prediction_manifest.get("contract_id") == threshold.get("contract_id") == CONTRACT_ID,
        "LR-08 contract lineage differs.",
    )
    _require(
        manifest.get("run_id") == prediction_manifest.get("run_id") == threshold.get("run_id") == RUN_ID,
        "LR-08 run_id lineage differs.",
    )
    _require(manifest.get("status") == prediction_manifest.get("status") == threshold.get("status") == "PASS", "LR-08 status differs.")
    _require(manifest.get("rows") == manifest.get("unique_record_ids") == EXPECTED_VALIDATION_ROWS, "LR-08 manifest row counts differ.")
    _require(manifest.get("source_order_preserved") is True, "LR-08 order-preservation flag differs.")
    _require(manifest.get("selected_threshold") == threshold.get("selected_threshold") == selected["threshold"], "LR-08 selected threshold differs.")
    _require(threshold.get("maximizer_count") == len(maximizing), "LR-08 maximizer count differs.")
    _require(threshold.get("maximizing_thresholds") == [row["threshold"] for row in maximizing], "LR-08 maximizing thresholds differ.")
    _require(threshold.get("tie_break") == ["minimum_absolute_distance_to_0.5", "lower_threshold"], "LR-08 tie-break differs.")
    _require(threshold.get("source_prediction_sha256") == _sha256_file(directory / "validation_predictions.csv"), "LR-08 threshold source hash differs.")
    _require(manifest.get("lineage_sha256") == prediction_manifest.get("lineage_sha256") == threshold.get("lineage_sha256"), "LR-08 lineage payloads differ.")
    _require(prediction_manifest.get("ordered_record_ids_sha256") == _sha256_ids(ids), "LR-08 ordered ID hash differs.")
    _require(prediction_manifest.get("execution") == {
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "predict_proba_calls": 1,
        "project_test_accessed": False,
    }, "LR-08 prediction execution boundary differs.")
    _require(threshold.get("data_boundary") == {
        "source_partition": "validation",
        "project_validation_accessed": True,
        "project_test_accessed": False,
        "model_fit_performed": False,
        "preprocessing_fit_performed": False,
        "predict_proba_calls": 0,
    }, "LR-08 threshold boundary differs.")
    return manifest


def validate_lr_validation_threshold(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate frozen LR-08 artifacts without loading model or holdout source data."""

    return _validate_lr_validation_artifact(Path(output_dir))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate LR-08 Validation and threshold artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = (
            run_lr_validation_threshold(args.output_dir, run_dir=args.run_dir)
            if args.command == "run"
            else validate_lr_validation_threshold(args.output_dir)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except LRValidationError as exc:
        print(f"LR-08 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
