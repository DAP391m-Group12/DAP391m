"""Canonical PlainMLP frozen-threshold Validation lifecycle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.dataset_b.models.neural.plain_mlp.validation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_VALIDATION_DIR,
    PlainMLPValidationError,
    _read_json as _read_validation_json,
    validate_plain_mlp_validation_artifact,
)
from src.dataset_b.models.neural.plain_mlp.full_train import _sha256_file


DEFAULT_OUTPUT_DIR = DEFAULT_VALIDATION_DIR / "threshold_selection"
CONTRACT_ID = "PMLP-F05-THRESHOLD"
CANDIDATE_THRESHOLDS = tuple(value / 1000 for value in range(1, 1000))
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_denied", "p_approved")
CANDIDATE_COLUMNS = (
    "threshold", "balanced_accuracy", "recall_approved", "recall_denied",
    "tp_approved", "fn_approved", "tn_denied", "fp_denied",
)
PAYLOAD_FILES = {"threshold_candidates.csv", "threshold_selection.json"}
FINAL_FILES = PAYLOAD_FILES | {"threshold_manifest.json", "COMPLETED"}


class PlainMLPThresholdError(RuntimeError):
    """Raised when PlainMLP threshold selection violates its contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlainMLPThresholdError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PlainMLPThresholdError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise PlainMLPThresholdError("Validation prediction columns are invalid.")
            rows = list(reader)
        labels = np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8)
        probabilities = np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise PlainMLPThresholdError("Cannot read Validation predictions.") from exc
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or not len(labels)
        or set(np.unique(labels).tolist()) != {0, 1}
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise PlainMLPThresholdError("Validation labels or probabilities are invalid.")
    return labels, probabilities


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


def _select(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best = max(row["balanced_accuracy"] for row in rows)
    maximizing = [row for row in rows if row["balanced_accuracy"] == best]
    selected = min(maximizing, key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]))
    return selected, maximizing


def _write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANDIDATE_COLUMNS:
            raise PlainMLPThresholdError("Threshold candidate columns are invalid.")
        return [
            {
                key: float(row[key]) if key in CANDIDATE_COLUMNS[:4] else int(row[key])
                for key in CANDIDATE_COLUMNS
            }
            for row in reader
        ]


def _same_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
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


def run_plain_mlp_validation_threshold(
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
) -> dict[str, Any]:
    """Select Balanced Accuracy threshold without fitting or predicting again."""
    validation_path = Path(validation_dir)
    destination = validation_path / "threshold_selection"
    if destination.exists():
        result = validate_plain_mlp_validation_threshold(validation_path)
        return {**result, "resume_action": "SKIPPED_VALID_COMPLETED"}
    working = destination.with_name(destination.name + ".tmp")
    if working.exists():
        raise PlainMLPThresholdError(
            f"Temporary directory already exists and was preserved: {working}"
        )
    try:
        source_manifest = validate_plain_mlp_validation_artifact(validation_path)
        source_result = _read_validation_json(validation_path / "validation_result.json")
        labels, probabilities = _read_predictions(validation_path / "validation_predictions.csv")
        rows = _candidate_rows(labels, probabilities)
        selected, maximizing = _select(rows)
        prediction_hash = _sha256_file(validation_path / "validation_predictions.csv")
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "run_id": source_result.get("run_id"),
            "status": "PASS",
            "stage": "validation_threshold_selection",
            "objective": "maximize_validation_balanced_accuracy",
            "candidate_grid": {
                "start": 0.001, "stop": 0.999, "step": 0.001,
                "inclusive": True, "count": len(CANDIDATE_THRESHOLDS),
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
            "prediction_file": "../validation_predictions.csv",
            "prediction_sha256": prediction_hash,
            "source_validation_contract_id": source_manifest["contract_id"],
            "source_validation_result_sha256": _sha256_file(
                validation_path / "validation_result.json"
            ),
            "source_validation_record_ids_sha256": source_result["data_boundary"][
                "validation_record_ids_sha256"
            ],
            "data_boundary": {
                "source_partition": "validation",
                "project_test_accessed": False,
                "model_fit_performed": False,
                "preprocessing_fit_performed": False,
                "predict_proba_calls": 0,
            },
        }
        working.mkdir(parents=True)
        _write_candidates(working / "threshold_candidates.csv", rows)
        _write_json(working / "threshold_selection.json", result)
        manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "run_id": source_result.get("run_id"),
            "status": "PASS",
            "rows": len(labels),
            "candidate_rows": len(rows),
            "selected_threshold": selected["threshold"],
            "source_prediction_sha256": prediction_hash,
            "artifact_sha256": {
                name: _sha256_file(working / name) for name in sorted(PAYLOAD_FILES)
            },
        }
        _write_json(working / "threshold_manifest.json", manifest)
        (working / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        validate_plain_mlp_validation_threshold(validation_path, threshold_output_dir=working)
        os.replace(working, destination)
        return validate_plain_mlp_validation_threshold(validation_path)
    except (PlainMLPValidationError, PlainMLPThresholdError):
        raise
    except Exception as exc:
        raise PlainMLPThresholdError(f"Cannot publish PlainMLP threshold artifacts: {exc}") from exc


def validate_plain_mlp_validation_threshold(
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    *,
    threshold_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the grid and verify the frozen threshold and its lineage."""
    validation_path = Path(validation_dir)
    validate_plain_mlp_validation_artifact(validation_path)
    directory = (
        Path(threshold_output_dir)
        if threshold_output_dir is not None
        else validation_path / "threshold_selection"
    )
    actual = {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
    if actual != FINAL_FILES:
        raise PlainMLPThresholdError("Threshold artifact set is invalid.")
    if (directory / "COMPLETED").read_text(encoding="utf-8") != f"{CONTRACT_ID} PASS\n":
        raise PlainMLPThresholdError("Threshold completion marker is invalid.")
    result = _read_json(directory / "threshold_selection.json")
    manifest = _read_json(directory / "threshold_manifest.json")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != PAYLOAD_FILES:
        raise PlainMLPThresholdError("Threshold artifact hash coverage is invalid.")
    for name, expected in hashes.items():
        if _sha256_file(directory / name) != expected:
            raise PlainMLPThresholdError(f"Threshold artifact hash mismatch: {name}")
    labels, probabilities = _read_predictions(validation_path / "validation_predictions.csv")
    expected_rows = _candidate_rows(labels, probabilities)
    stored_rows = _read_candidates(directory / "threshold_candidates.csv")
    if not _same_rows(stored_rows, expected_rows):
        raise PlainMLPThresholdError("Threshold candidate table is not reproducible.")
    selected, maximizing = _select(expected_rows)
    source_result = _read_validation_json(validation_path / "validation_result.json")
    prediction_hash = _sha256_file(validation_path / "validation_predictions.csv")
    expected_counts = {
        "rows": len(labels),
        "n_approved": int(np.count_nonzero(labels == 1)),
        "n_denied": int(np.count_nonzero(labels == 0)),
        **{key: selected[key] for key in CANDIDATE_COLUMNS[4:]},
    }
    if (
        manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("run_id") != source_result.get("run_id")
        or result.get("run_id") != source_result.get("run_id")
        or manifest.get("candidate_rows") != len(CANDIDATE_THRESHOLDS)
        or manifest.get("selected_threshold") != selected["threshold"]
        or manifest.get("source_prediction_sha256") != prediction_hash
        or result.get("contract_id") != CONTRACT_ID
        or result.get("status") != "PASS"
        or result.get("objective") != "maximize_validation_balanced_accuracy"
        or result.get("tie_break") != ["minimum_absolute_distance_to_0.5", "lower_threshold"]
        or result.get("selected_threshold") != selected["threshold"]
        or result.get("maximizer_count") != len(maximizing)
        or result.get("maximizing_thresholds") != [row["threshold"] for row in maximizing]
        or result.get("validation_counts") != expected_counts
        or result.get("prediction_sha256") != prediction_hash
        or result.get("source_validation_result_sha256")
        != _sha256_file(validation_path / "validation_result.json")
        or result.get("source_validation_record_ids_sha256")
        != source_result.get("data_boundary", {}).get("validation_record_ids_sha256")
        or result.get("data_boundary") != {
            "source_partition": "validation",
            "project_test_accessed": False,
            "model_fit_performed": False,
            "preprocessing_fit_performed": False,
            "predict_proba_calls": 0,
        }
    ):
        raise PlainMLPThresholdError("Threshold decision or lineage is invalid.")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate PlainMLP threshold selection.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = (
            run_plain_mlp_validation_threshold(args.validation_dir)
            if args.command == "run"
            else validate_plain_mlp_validation_threshold(args.validation_dir)
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except (PlainMLPValidationError, PlainMLPThresholdError) as exc:
        print(f"PlainMLP threshold selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
