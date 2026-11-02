"""XGB-08: one Validation inference and frozen threshold selection for XGBoost."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.dataset_b.data import DatasetBInputQualityGate, DatasetBSplitLoader
from src.dataset_b.models.classical.xgboost.training import _validated_run_id, validate_xgb_final_training
from src.utils.path import CONFIG_DIR, DATASET_B_DIR, PROJECT_ROOT


CONTRACT_ID = "XGB-08"
EXPECTED_VALIDATION_ROWS = 43_995
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
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = DATASET_B_DIR / "split_v1"
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"


class XGBValidationError(RuntimeError):
    """Raised when XGB-08 violates its frozen Validation boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XGBValidationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise XGBValidationError(f"Cannot read JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _paths(run_dir: Path, output_dir: Path, run_id: str) -> dict[str, Path]:
    _validated_run_id(run_id)
    root = run_dir.resolve(strict=True)
    _require(root.name == run_id, "XGB-08 run_dir must end with the explicit run_id.")
    destination = output_dir.resolve(strict=False)
    _require(destination == (root / "validation").resolve(strict=False), "XGB-08 output must be run_dir/validation.")
    paths = {
        "run_dir": root,
        "output_dir": destination,
        "temporary_dir": destination.with_name("validation.tmp"),
        "pipeline": root / "checkpoints/fitted_pipeline.joblib",
    }
    _require(paths["pipeline"].is_file(), "XGB-07 fitted pipeline is missing.")
    return paths


def _load_pipeline(path: Path, expected_scale_pos_weight: float):
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise XGBValidationError("Cannot load frozen XGB-07 pipeline.") from exc
    _require(tuple(getattr(pipeline, "named_steps", {})) == ("preprocess", "estimator"), "Frozen XGBoost pipeline steps differ.")
    estimator = pipeline.named_steps["estimator"]
    _require(np.array_equal(getattr(estimator, "classes_", None), np.asarray([0, 1])), "Frozen XGBoost class order differs.")
    _require(estimator.get_params().get("scale_pos_weight") == expected_scale_pos_weight, "Frozen XGBoost weighting differs from XGB-07.")
    return pipeline


def _predict_validation_once(pipeline: Any, frame: Any) -> np.ndarray:
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=np.float64)
    _require(probabilities.shape == (len(frame), 2), "Validation probability shape differs.")
    _require(
        np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)),
        "Validation probabilities are invalid.",
    )
    _require(
        np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-7),
        "Validation probability rows do not sum to one.",
    )
    return probabilities[:, 1]


def _candidate_rows(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in CANDIDATE_THRESHOLDS:
        predicted = probabilities >= threshold
        approved = labels == 1
        denied = ~approved
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "recall_approved": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
                "recall_denied": float(recall_score(labels, predicted, pos_label=0, zero_division=0)),
                "tp_approved": int(np.count_nonzero(approved & predicted)),
                "fn_approved": int(np.count_nonzero(approved & ~predicted)),
                "tn_denied": int(np.count_nonzero(denied & ~predicted)),
                "fp_denied": int(np.count_nonzero(denied & predicted)),
            }
        )
    return rows


def _select_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(bool(rows), "Threshold candidates are empty.")
    best = max(float(row["balanced_accuracy"]) for row in rows)
    maximizing = [dict(row) for row in rows if float(row["balanced_accuracy"]) == best]
    selected = min(maximizing, key=lambda row: (abs(float(row["threshold"]) - 0.5), float(row["threshold"])))
    return selected, maximizing


def _write_predictions(path: Path, ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        writer.writerows(
            (record_id, int(label), repr(float(probability)))
            for record_id, label, probability in zip(ids, labels, probabilities, strict=True)
        )


def _read_predictions(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS, "Validation prediction columns differ.")
        rows = list(reader)
    return (
        tuple(row["record_id"] for row in rows),
        np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8),
        np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64),
    )


def _validate_training_lineage(training: Mapping[str, Any], run_id: str) -> float:
    _require(training.get("run_id") == run_id, "XGB-08 training run_id lineage differs.")
    _require(training.get("model_key") == "xgb" and training.get("reporting_key") == "xgboost", "XGB-08 training model lineage differs.")
    _require(training.get("feature_version") == "feature_set_v1", "XGB-08 feature lineage differs.")
    _require(training.get("split_version") == "split_v1", "XGB-08 split lineage differs.")
    _require(training.get("preprocessing_version") == "dataset_b_preprocessing_v2", "XGB-08 preprocessing lineage differs.")
    _require(training.get("validation_accessed") is False and training.get("test_accessed") is False, "XGB-07 holdout boundary differs.")
    effective = training.get("full_train_effective_parameters", {})
    value = effective.get("scale_pos_weight")
    _require(isinstance(value, (int, float)) and np.isfinite(value) and value > 0, "XGB-07 full-Train weighting evidence differs.")
    return float(value)


def validate_xgb_validation_threshold(
    output_dir: str | Path,
    *,
    run_dir: str | Path,
    run_id: str,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Self-validate frozen XGB-08 output without fitting or inference."""
    paths = _paths(Path(run_dir), Path(output_dir), run_id)
    root = paths["output_dir"]
    required = {
        "validation_predictions.csv",
        "threshold_candidates.csv",
        "threshold_selection.json",
        "prediction_manifest.json",
        "manifest.json",
        "COMPLETED",
    }
    _require(root.is_dir() and {p.name for p in root.iterdir() if p.is_file()} == required, "XGB-08 artifact set is incomplete or unexpected.")
    training = validate_xgb_final_training(paths["run_dir"], run_id=run_id)
    expected_scale_pos_weight = _validate_training_lineage(training, run_id)
    manifest = _read_json(root / "manifest.json")
    selection = _read_json(root / "threshold_selection.json")
    prediction = _read_json(root / "prediction_manifest.json")
    _require(manifest.get("contract_id") == selection.get("contract_id") == prediction.get("contract_id") == CONTRACT_ID, "XGB-08 contract lineage differs.")
    _require(manifest.get("status") == selection.get("status") == prediction.get("status") == "PASS", "XGB-08 is not PASS.")
    _require(manifest.get("run_id") == selection.get("run_id") == prediction.get("run_id") == run_id, "XGB-08 run_id lineage differs.")
    _require(manifest.get("model_key") == prediction.get("model_key") == "xgb", "XGB-08 model key differs.")
    _require(manifest.get("reporting_key") == prediction.get("reporting_key") == "xgboost", "XGB-08 reporting key differs.")
    _require(selection.get("tie_break") == ["minimum_absolute_distance_to_0.5", "lower_threshold"], "XGB-08 tie rule differs.")
    _require(selection.get("fitted_pipeline") == str(paths["pipeline"].resolve()), "XGB-08 threshold belongs to another pipeline.")
    _require(selection.get("source_predictions") == str((root / "validation_predictions.csv").resolve()), "XGB-08 threshold source differs.")
    _require(selection.get("baseline_threshold_reused") is False, "XGB-08 reused a historical threshold.")
    _require(prediction.get("execution") == {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1}, "XGB-08 execution boundary differs.")
    _require(prediction.get("scale_pos_weight") == expected_scale_pos_weight, "XGB-08 weighting lineage differs.")
    ids, labels, probabilities = _read_predictions(root / "validation_predictions.csv")
    _require(np.array_equal(np.unique(labels), np.asarray([0, 1])), "XGB-08 Validation labels differ from the binary contract.")
    _require(np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)), "XGB-08 Validation probabilities are invalid.")
    gated = DatasetBInputQualityGate().apply(
        DatasetBSplitLoader(
            manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        ).load("validation")
    )
    expected_ids = tuple(gated.record_ids.astype(str).tolist())
    expected_labels = gated.y.to_numpy(dtype=np.int8)
    _require(ids == expected_ids and np.array_equal(labels, expected_labels), "XGB-08 predictions do not align to the Validation source.")
    _require(len(ids) == len(set(ids)) == EXPECTED_VALIDATION_ROWS, "XGB-08 Validation population differs.")
    expected_lineage = {
        "training_manifest": str((paths["run_dir"] / "manifest.json").resolve()),
        "cv_results": str((paths["run_dir"] / "cv_results.json").resolve()),
        "full_train_result": str((paths["run_dir"] / "full_train_result.json").resolve()),
        "fitted_pipeline": str(paths["pipeline"].resolve()),
        "preprocessing_manifest": str(Path(training["preprocessing_manifest"]).resolve()),
        "split_manifest": str(Path(split_manifest_path).resolve()),
        "feature_contract": str(Path(feature_contract_path).resolve()),
        "validation_source": str((Path(split_dir) / "validation.csv").resolve()),
        "validation_record_ids_verified": True,
    }
    _require(manifest.get("lineage") == expected_lineage, "XGB-08 source lineage differs.")
    with (root / "threshold_candidates.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [{key: float(row[key]) for key in CANDIDATE_COLUMNS} for row in rows]
    recomputed = _candidate_rows(labels, probabilities)
    _require(len(candidates) == len(recomputed) == len(CANDIDATE_THRESHOLDS), "XGB-08 threshold candidate count differs.")
    for stored, expected in zip(candidates, recomputed, strict=True):
        _require(
            all(np.isclose(stored[key], expected[key], rtol=0.0, atol=1e-15) for key in CANDIDATE_COLUMNS),
            "XGB-08 threshold candidates are not reproducible.",
        )
    selected, maximizing = _select_threshold(candidates)
    _require(float(selection["selected_threshold"]) == float(selected["threshold"]), "XGB-08 selected threshold is not reproducible.")
    _require(selection.get("maximizer_count") == len(maximizing), "XGB-08 maximizer count differs.")
    _require(selection.get("selected_metrics") == {key: selected[key] for key in CANDIDATE_COLUMNS[1:]}, "XGB-08 selected metrics differ.")
    _require(len(ids) == len(labels) == len(probabilities) == manifest.get("rows"), "XGB-08 rows are misaligned.")
    _require(prediction.get("probability_min") == float(probabilities.min()) and prediction.get("probability_max") == float(probabilities.max()), "XGB-08 probability summary differs.")
    _require(prediction.get("ordered_record_ids_verified") is True, "XGB-08 ordered IDs are not verified.")
    _require(manifest.get("quality_audit") == gated.quality_audit.as_dict(), "XGB-08 quality audit differs.")
    _require(manifest.get("test_accessed") is False, "XGB-08 accessed Test.")
    return manifest


def run_xgb_validation_threshold(
    run_id: str,
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Run exactly one frozen XGB-07 Validation inference and select threshold."""
    root = Path(run_dir)
    destination = Path(output_dir) if output_dir is not None else root / "validation"
    paths = _paths(root, destination, run_id)
    if paths["output_dir"].exists():
        return {
            **validate_xgb_validation_threshold(
                paths["output_dir"],
                run_dir=root,
                run_id=run_id,
                split_manifest_path=split_manifest_path,
                split_dir=split_dir,
                feature_contract_path=feature_contract_path,
            ),
            "resume_action": "SKIPPED_VALID_COMPLETED",
        }
    training = validate_xgb_final_training(root, run_id=run_id)
    expected_scale_pos_weight = _validate_training_lineage(training, run_id)
    pipeline = _load_pipeline(paths["pipeline"], expected_scale_pos_weight)
    loader = DatasetBSplitLoader(
        manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )
    gated = DatasetBInputQualityGate().apply(loader.load("validation"))
    ids = tuple(gated.record_ids.astype(str).tolist())
    labels = gated.y.to_numpy(dtype=np.int8)
    _require(len(ids) == len(set(ids)) == EXPECTED_VALIDATION_ROWS, "Validation population differs.")
    probabilities = _predict_validation_once(pipeline, gated.X.reset_index(drop=True))
    candidates = _candidate_rows(labels, probabilities)
    selected, maximizing = _select_threshold(candidates)
    tmp = paths["temporary_dir"]
    _require(not tmp.exists(), "XGB-08 temporary directory exists.")
    tmp.mkdir()
    try:
        _write_predictions(tmp / "validation_predictions.csv", ids, labels, probabilities)
        with (tmp / "threshold_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(candidates)
        _write_json(
            tmp / "threshold_selection.json",
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "status": "PASS",
                "run_id": run_id,
                "objective": "maximize_validation_balanced_accuracy",
                "selected_threshold": selected["threshold"],
                "selected_metrics": {key: selected[key] for key in CANDIDATE_COLUMNS[1:]},
                "tie_break": ["minimum_absolute_distance_to_0.5", "lower_threshold"],
                "maximizer_count": len(maximizing),
                "maximizing_thresholds": [row["threshold"] for row in maximizing],
                "source_predictions": str((paths["output_dir"] / "validation_predictions.csv").resolve()),
                "fitted_pipeline": str(paths["pipeline"].resolve()),
                "baseline_threshold_reused": False,
                "data_boundary": {
                    "source_partition": "validation",
                    "model_fit_performed": False,
                    "preprocessing_fit_performed": False,
                    "test_accessed": False,
                },
            },
        )
        quality_audit = gated.quality_audit.as_dict()
        lineage = {
            "training_manifest": str((root / "manifest.json").resolve()),
            "cv_results": str((root / "cv_results.json").resolve()),
            "full_train_result": str((root / "full_train_result.json").resolve()),
            "fitted_pipeline": str(paths["pipeline"].resolve()),
            "preprocessing_manifest": str(Path(training["preprocessing_manifest"]).resolve()),
            "split_manifest": str(Path(split_manifest_path).resolve()),
            "feature_contract": str(Path(feature_contract_path).resolve()),
            "validation_source": str((Path(split_dir) / "validation.csv").resolve()),
            "validation_record_ids_verified": True,
        }
        _write_json(
            tmp / "prediction_manifest.json",
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "status": "PASS",
                "run_id": run_id,
                "model_key": "xgb",
                "reporting_key": "xgboost",
                "rows": len(ids),
                "ordered_record_ids_verified": True,
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
                "scale_pos_weight": expected_scale_pos_weight,
                "execution": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "predict_proba_calls": 1},
                "training_manifest": str((root / "manifest.json").resolve()),
                "fitted_pipeline": str(paths["pipeline"].resolve()),
            },
        )
        _write_json(
            tmp / "manifest.json",
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "status": "PASS",
                "run_id": run_id,
                "model_key": "xgb",
                "reporting_key": "xgboost",
                "feature_version": training["feature_version"],
                "split_version": training["split_version"],
                "preprocessing_version": training["preprocessing_version"],
                "rows": len(ids),
                "unique_record_ids": len(set(ids)),
                "source_order_preserved": True,
                "selected_threshold": selected["threshold"],
                "quality_audit": quality_audit,
                "lineage": lineage,
                "test_accessed": False,
                "prohibited_outputs_written": {
                    "cv_or_full_train_artifacts": False,
                    "test_artifacts": False,
                    "legacy_xgb_artifacts": False,
                },
            },
        )
        (tmp / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        os.replace(tmp, paths["output_dir"])
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return validate_xgb_validation_threshold(
        paths["output_dir"],
        run_dir=root,
        run_id=run_id,
        split_manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run or validate XGB-08.")
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir or args.run_dir / "validation"
    value = (
        run_xgb_validation_threshold(args.run_id, run_dir=args.run_dir, output_dir=output)
        if args.command == "run"
        else validate_xgb_validation_threshold(output, run_dir=args.run_dir, run_id=args.run_id)
    )
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
