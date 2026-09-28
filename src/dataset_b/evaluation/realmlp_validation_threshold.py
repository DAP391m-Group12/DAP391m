"""Chọn và kiểm tra ngưỡng Validation cho RealMLP."""

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

from src.dataset_b.evaluation.realmlp_validation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_VALIDATION_DIR,
    RealMLPValidationError,
    validate_realmlp_validation,
)
from src.dataset_b.runtime.realmlp_full_train import _sha256_file


DEFAULT_OUTPUT_DIR = DEFAULT_VALIDATION_DIR / "threshold_selection"
CANDIDATE_THRESHOLDS = tuple(value / 1000 for value in range(1, 1000))
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_approved")
BASE_FILES = {"threshold_candidates.csv", "threshold_selection.json", "threshold_manifest.json"}
COMPLETED_FILES = BASE_FILES | {"COMPLETED"}
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


class RealMLPThresholdError(RuntimeError):
    """Lỗi vi phạm hợp đồng chọn ngưỡng Validation của RealMLP."""


def run_realmlp_validation_threshold(
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
) -> dict[str, Any]:
    """Chọn một ngưỡng theo Balanced Accuracy mà không dự đoán hoặc huấn luyện."""
    validation_path = Path(validation_dir)
    output_dir = validation_path / "threshold_selection"
    if output_dir.exists():
        return {**validate_realmlp_validation_threshold(validation_path), "resume_action": "SKIPPED_VALID_COMPLETED"}

    working = output_dir.with_name(output_dir.name + ".tmp")
    if working.exists():
        raise RealMLPThresholdError(f"Thư mục tạm đã tồn tại; giữ nguyên để kiểm tra: {working}")
    try:
        source_manifest = validate_realmlp_validation(validation_path)
        source_result = _read_json(validation_path / "validation_result.json")
        predictions = _read_predictions(validation_path / "validation_predictions.csv")
        labels = np.asarray([int(row["y_approved"]) for row in predictions], dtype=np.int8)
        probabilities = np.asarray([float(row["p_approved"]) for row in predictions], dtype=np.float64)
        if len(labels) != source_manifest["rows"] or not _valid_input(labels, probabilities):
            raise RealMLPThresholdError("Dữ liệu đầu vào cho chọn ngưỡng không hợp lệ.")

        candidate_rows = _candidate_rows(labels, probabilities)
        best_score = max(row["balanced_accuracy"] for row in candidate_rows)
        maximizing = [row for row in candidate_rows if row["balanced_accuracy"] == best_score]
        selected = min(maximizing, key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]))
        prediction_sha256 = _sha256_file(validation_path / "validation_predictions.csv")
        result = {
            "schema_version": 1,
            "contract_id": "RMLP-08B",
            "status": "PASS",
            "stage": "validation_threshold_selection",
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
                "tp_approved": selected["tp_approved"],
                "fn_approved": selected["fn_approved"],
                "tn_denied": selected["tn_denied"],
                "fp_denied": selected["fp_denied"],
            },
            "prediction_file": "../validation_predictions.csv",
            "prediction_sha256": prediction_sha256,
            "source_validation_contract_id": source_manifest["contract_id"],
            "source_validation_result_sha256": _sha256_file(validation_path / "validation_result.json"),
            "source_validation_record_ids_sha256": source_result["data_boundary"]["validation_record_ids_sha256"],
            "data_boundary": {
                "source_partition": "validation",
                "project_test_accessed": False,
                "model_fit_performed": False,
                "predict_proba_calls": 0,
            },
        }
        working.mkdir(parents=True)
        _write_candidates(working / "threshold_candidates.csv", candidate_rows)
        _write_json(working / "threshold_selection.json", result)
        manifest = {
            "schema_version": 1,
            "contract_id": "RMLP-08B",
            "status": "PASS",
            "rows": len(labels),
            "candidate_rows": len(candidate_rows),
            "selected_threshold": selected["threshold"],
            "source_prediction_sha256": prediction_sha256,
            "artifact_sha256": {
                name: _sha256_file(working / name)
                for name in ("threshold_candidates.csv", "threshold_selection.json")
            },
        }
        _write_json(working / "threshold_manifest.json", manifest)
        (working / "COMPLETED").write_text("RMLP-08B PASS\n", encoding="utf-8")
        validate_realmlp_validation_threshold(validation_path, threshold_output_dir=working)
        os.replace(working, output_dir)
        return validate_realmlp_validation_threshold(validation_path)
    except (RealMLPValidationError, RealMLPThresholdError):
        raise
    except Exception as exc:
        raise RealMLPThresholdError(f"Không thể công bố quyết định ngưỡng Validation: {exc}") from exc


def validate_realmlp_validation_threshold(
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    *,
    threshold_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Đối chiếu độc lập toàn bộ lưới, lựa chọn, mã băm và nguồn dự đoán."""
    validation_path = Path(validation_dir)
    validate_realmlp_validation(validation_path)
    output_dir = (
        Path(threshold_output_dir)
        if threshold_output_dir is not None
        else validation_path / "threshold_selection"
    )
    actual = {path.name for path in output_dir.iterdir() if path.is_file()} if output_dir.is_dir() else set()
    if actual != COMPLETED_FILES:
        raise RealMLPThresholdError(f"Tập tệp kết quả ngưỡng sai: {sorted(actual ^ COMPLETED_FILES)}")
    result = _read_json(output_dir / "threshold_selection.json")
    manifest = _read_json(output_dir / "threshold_manifest.json")
    if (output_dir / "COMPLETED").read_text(encoding="utf-8") != "RMLP-08B PASS\n":
        raise RealMLPThresholdError("Dấu hoàn tất RMLP-08B không hợp lệ.")
    predictions = _read_predictions(validation_path / "validation_predictions.csv")
    labels = np.asarray([int(row["y_approved"]) for row in predictions], dtype=np.int8)
    probabilities = np.asarray([float(row["p_approved"]) for row in predictions], dtype=np.float64)
    expected_rows = _candidate_rows(labels, probabilities)
    stored_rows = _read_candidates(output_dir / "threshold_candidates.csv")
    if len(stored_rows) != len(expected_rows) or not _candidate_rows_equal(stored_rows, expected_rows):
        raise RealMLPThresholdError("Bảng ngưỡng không khớp dữ liệu Validation.")
    best_score = max(row["balanced_accuracy"] for row in expected_rows)
    maximizing = [row for row in expected_rows if row["balanced_accuracy"] == best_score]
    selected = min(maximizing, key=lambda row: (abs(row["threshold"] - 0.5), row["threshold"]))
    prediction_hash = _sha256_file(validation_path / "validation_predictions.csv")
    expected_artifacts = {"threshold_candidates.csv", "threshold_selection.json"}
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_artifacts:
        raise RealMLPThresholdError("Danh sách mã băm hiện vật chọn ngưỡng không hợp lệ.")
    for name, expected_hash in hashes.items():
        if _sha256_file(output_dir / name) != expected_hash:
            raise RealMLPThresholdError(f"Mã băm hiện vật chọn ngưỡng sai: {name}")
    source_result = _read_json(validation_path / "validation_result.json")
    expected_counts = {
        "rows": len(labels),
        "n_approved": int(np.count_nonzero(labels == 1)),
        "n_denied": int(np.count_nonzero(labels == 0)),
        "tp_approved": selected["tp_approved"],
        "fn_approved": selected["fn_approved"],
        "tn_denied": selected["tn_denied"],
        "fp_denied": selected["fp_denied"],
    }
    if (
        result.get("schema_version") != 1
        or manifest.get("schema_version") != 1
        or manifest.get("contract_id") != "RMLP-08B"
        or manifest.get("status") != "PASS"
        or result.get("contract_id") != "RMLP-08B"
        or result.get("status") != "PASS"
        or result.get("objective") != "maximize_validation_balanced_accuracy"
        or result.get("candidate_grid") != {
            "start": 0.001, "stop": 0.999, "step": 0.001,
            "inclusive": True, "count": len(CANDIDATE_THRESHOLDS),
        }
        or result.get("prediction_rule") != "p_approved >= threshold -> Approve (y_approved=1)"
        or result.get("tie_break") != ["minimum_absolute_distance_to_0.5", "lower_threshold"]
        or result.get("selected_threshold") != selected["threshold"]
        or result.get("maximizer_count") != len(maximizing)
        or result.get("maximizing_thresholds") != [row["threshold"] for row in maximizing]
        or result.get("selected_metrics") != {
            "balanced_accuracy": selected["balanced_accuracy"],
            "recall_approved": selected["recall_approved"],
            "recall_denied": selected["recall_denied"],
        }
        or result.get("validation_counts") != expected_counts
        or result.get("source_validation_contract_id") != "RMLP-08A"
        or result.get("source_validation_result_sha256") != _sha256_file(validation_path / "validation_result.json")
        or result.get("source_validation_record_ids_sha256") != source_result.get("data_boundary", {}).get("validation_record_ids_sha256")
        or result.get("prediction_sha256") != prediction_hash
        or result.get("data_boundary") != {
            "source_partition": "validation", "project_test_accessed": False,
            "model_fit_performed": False, "predict_proba_calls": 0,
        }
        or manifest.get("rows") != len(labels)
        or manifest.get("candidate_rows") != len(CANDIDATE_THRESHOLDS)
        or manifest.get("selected_threshold") != selected["threshold"]
        or manifest.get("source_prediction_sha256") != prediction_hash
    ):
        raise RealMLPThresholdError("Quyết định ngưỡng hoặc dấu vết nguồn không khớp.")
    return manifest


def _candidate_rows(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for threshold in CANDIDATE_THRESHOLDS:
        predicted = probabilities >= threshold
        recall_approved = float(recall_score(labels, predicted, pos_label=1, zero_division=0))
        recall_denied = float(recall_score(labels, predicted, pos_label=0, zero_division=0))
        rows.append({
            "threshold": threshold,
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "recall_approved": recall_approved,
            "recall_denied": recall_denied,
            "tp_approved": int(np.count_nonzero((labels == 1) & predicted)),
            "fn_approved": int(np.count_nonzero((labels == 1) & ~predicted)),
            "tn_denied": int(np.count_nonzero((labels == 0) & ~predicted)),
            "fp_denied": int(np.count_nonzero((labels == 0) & predicted)),
        })
    return rows


def _candidate_rows_equal(observed: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    for left, right in zip(observed, expected, strict=True):
        for key in CANDIDATE_COLUMNS:
            if key in {"threshold", "balanced_accuracy", "recall_approved", "recall_denied"}:
                if not np.isclose(float(left[key]), float(right[key]), rtol=0.0, atol=1e-15):
                    return False
            elif int(left[key]) != right[key]:
                return False
    return True


def _valid_input(labels: np.ndarray, probabilities: np.ndarray) -> bool:
    return bool(
        labels.ndim == 1
        and probabilities.shape == labels.shape
        and len(labels) > 0
        and np.isin(labels, (0, 1)).all()
        and np.isfinite(probabilities).all()
        and np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        and np.unique(labels).size == 2
    )


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise RealMLPThresholdError("Cột của hiện vật dự đoán Validation không đúng.")
        return list(reader)


def _write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANDIDATE_COLUMNS:
            raise RealMLPThresholdError("Cột trong bảng ngưỡng không đúng.")
        parsed = []
        for row in reader:
            parsed.append({key: float(row[key]) if key in CANDIDATE_COLUMNS[:4] else int(row[key]) for key in CANDIDATE_COLUMNS})
        return parsed


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RealMLPThresholdError(f"Tệp JSON phải chứa một đối tượng: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chọn và kiểm tra ngưỡng Validation RealMLP.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = (
            run_realmlp_validation_threshold(args.validation_dir)
            if args.command == "run"
            else validate_realmlp_validation_threshold(args.validation_dir)
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (RealMLPValidationError, RealMLPThresholdError) as exc:
        print(f"Chọn ngưỡng Validation RealMLP thất bại: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
