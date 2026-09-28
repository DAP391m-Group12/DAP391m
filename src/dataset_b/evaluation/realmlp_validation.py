"""Tạo và kiểm tra đúng một lượt dự đoán Validation của RealMLP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from src.dataset_b.preprocessing import DatasetBSplitLoader
from src.dataset_b.runtime.realmlp_full_train import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FULL_TRAIN_DIR,
    _sha256_file,
    _sha256_ids,
    validate_realmlp_full_train,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp_v1" / "validation"
)
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "dataset_b" / "split_v1.json"
DEFAULT_SPLIT_DIR = DatasetBSplitLoader.DEFAULT_SPLIT_DIR
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b" / "feature_set_v1.yaml"
DEFAULT_MODEL_PROTOCOL_PATH = CONFIG_DIR / "dataset_b" / "model_protocol_v1.yaml"
EXPECTED_VALIDATION_ROWS = 43_995
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_approved")
BASE_FILES = {
    "run_manifest.json",
    "validation_predictions.csv",
    "validation_result.json",
}
FINAL_FILES = BASE_FILES | {"COMPLETED"}


class RealMLPValidationError(RuntimeError):
    """Lỗi vi phạm hợp đồng dự đoán Validation của RealMLP."""


def run_realmlp_validation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Dự đoán Validation một lần rồi công bố gói kết quả theo cách nguyên tử."""
    destination = Path(output_dir)
    full_train_path = Path(full_train_dir)
    if destination.exists():
        manifest = validate_realmlp_validation(
            destination,
            full_train_dir=full_train_path,
            split_manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
            model_protocol_path=model_protocol_path,
        )
        return {**manifest, "resume_action": "SKIPPED_VALID_COMPLETED"}

    working = destination.with_name(destination.name + ".tmp")
    if working.exists():
        raise RealMLPValidationError(
            f"Thư mục tạm đã tồn tại; giữ nguyên để kiểm tra thủ công: {working}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    full_train_manifest = validate_realmlp_full_train(full_train_path)
    try:
        loader = DatasetBSplitLoader(
            manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        )
        partition = loader.load("validation")
    except Exception as exc:
        raise RealMLPValidationError("Không thể nạp phân vùng Validation theo split đã khóa.") from exc

    record_ids = tuple(partition.record_ids.astype(str).tolist())
    labels = partition.y.to_numpy(dtype=np.int64)
    if (
        len(record_ids) != EXPECTED_VALIDATION_ROWS
        or len(set(record_ids)) != EXPECTED_VALIDATION_ROWS
        or labels.shape != (EXPECTED_VALIDATION_ROWS,)
        or not np.isin(labels, (0, 1)).all()
    ):
        raise RealMLPValidationError(
            "Số dòng, mã bản ghi duy nhất hoặc nhãn Validation không đúng hợp đồng."
        )

    checkpoint_path = full_train_path / "checkpoints" / "fitted_model.pkl"
    try:
        bundle = joblib.load(checkpoint_path)
    except Exception as exc:
        raise RealMLPValidationError("Không thể nạp gói mô hình đã khớp trên Train.") from exc
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"estimator", "preprocessing", "scope"}
        or bundle.get("estimator") is None
        or bundle.get("preprocessing") is None
    ):
        raise RealMLPValidationError("Cấu trúc gói mô hình đã lưu không hợp lệ.")

    estimator = bundle["estimator"]
    preprocessing = bundle["preprocessing"]
    try:
        transformed = preprocessing.transform(partition.X.reset_index(drop=True))
        probabilities = np.asarray(estimator.predict_proba(transformed), dtype=np.float64)
    except Exception as exc:
        raise RealMLPValidationError("Dự đoán Validation thất bại.") from exc
    if (
        probabilities.shape != (EXPECTED_VALIDATION_ROWS, 2)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise RealMLPValidationError("Ma trận xác suất Validation không hợp lệ.")

    classes = np.asarray(getattr(estimator, "classes_", ()))
    positive_positions = np.flatnonzero(classes == 1)
    if classes.shape != (2,) or positive_positions.shape != (1,):
        raise RealMLPValidationError("Thứ tự nhãn lớp của mô hình không đúng hợp đồng [0, 1].")
    positive_probabilities = probabilities[:, int(positive_positions[0])]

    working.mkdir(parents=True)
    try:
        _write_predictions(
            working / "validation_predictions.csv",
            record_ids,
            labels,
            positive_probabilities,
        )
        prediction_summary = _prediction_summary(record_ids, labels, positive_probabilities)
        result = {
            "schema_version": 1,
            "contract_id": "RMLP-08A",
            "status": "PASS",
            "prediction_partition": "validation",
            "prediction_file": "validation_predictions.csv",
            "prediction_columns": list(PREDICTION_COLUMNS),
            "split_version": "split_v1",
            "feature_version": "temporary_training_lock_v0_2",
            "model_protocol_version": "model_protocol_v1",
            "source_full_train_contract_id": full_train_manifest["contract_id"],
            "prediction_summary": prediction_summary,
            "execution": {
                "model_fit_calls": 0,
                "preprocessing_transform_calls": 1,
                "predict_proba_calls": 1,
            },
            "data_boundary": {
                "source_partition": "validation",
                "validation_rows": len(record_ids),
                "validation_record_ids_sha256": _sha256_ids(record_ids),
                "project_validation_accessed": True,
                "project_test_accessed": False,
                "model_fit_performed": False,
            },
            "lineage": _lineage(
                full_train_path,
                split_manifest_path=Path(split_manifest_path),
                validation_csv=Path(split_dir) / "validation.csv",
                feature_contract_path=Path(feature_contract_path),
                model_protocol_path=Path(model_protocol_path),
            ),
        }
        _write_json(working / "validation_result.json", result)
        hashes = {
            name: _sha256_file(working / name)
            for name in ("validation_predictions.csv", "validation_result.json")
        }
        manifest = {
            "schema_version": 1,
            "dataset": "dataset_b",
            "model": "realmlp",
            "artifact_version": "realmlp_v1",
            "contract_id": "RMLP-08A",
            "status": "PASS",
            "prediction_partition": "validation",
            "rows": len(record_ids),
            "unique_record_ids": len(set(record_ids)),
            "source_order_preserved": True,
            "artifact_sha256": hashes,
        }
        _write_json(working / "run_manifest.json", manifest)
        (working / "COMPLETED").write_text("RMLP-08A PASS\n", encoding="utf-8")
        validate_realmlp_validation(
            working,
            full_train_dir=full_train_path,
            split_manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
            model_protocol_path=model_protocol_path,
        )
        os.replace(working, destination)
        return validate_realmlp_validation(
            destination,
            full_train_dir=full_train_path,
            split_manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
            model_protocol_path=model_protocol_path,
        )
    except Exception as exc:
        if isinstance(exc, RealMLPValidationError):
            raise
        raise RealMLPValidationError(f"Không thể công bố gói dự đoán Validation: {exc}") from exc


def validate_realmlp_validation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Kiểm tra gói Validation đã công bố mà không gọi mô hình dự đoán."""
    directory = Path(output_dir)
    validate_realmlp_full_train(Path(full_train_dir))
    actual_files = {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
    actual_directories = {path.name for path in directory.iterdir() if path.is_dir()} if directory.is_dir() else set()
    if actual_files != FINAL_FILES or actual_directories - {"threshold_selection", "threshold_selection.tmp"}:
        raise RealMLPValidationError("Tập tệp hoặc thư mục cấp cao của gói Validation sai.")
    manifest = _read_json(directory / "run_manifest.json")
    result = _read_json(directory / "validation_result.json")
    marker = (directory / "COMPLETED").read_text(encoding="utf-8")
    if (
        manifest.get("contract_id") != "RMLP-08A"
        or manifest.get("status") != "PASS"
        or manifest.get("prediction_partition") != "validation"
        or manifest.get("rows") != EXPECTED_VALIDATION_ROWS
        or manifest.get("unique_record_ids") != EXPECTED_VALIDATION_ROWS
        or manifest.get("source_order_preserved") is not True
        or result.get("contract_id") != "RMLP-08A"
        or result.get("status") != "PASS"
        or result.get("prediction_partition") != "validation"
        or result.get("prediction_columns") != list(PREDICTION_COLUMNS)
        or result.get("execution")
        != {
            "model_fit_calls": 0,
            "preprocessing_transform_calls": 1,
            "predict_proba_calls": 1,
        }
        or result.get("data_boundary", {}).get("project_test_accessed") is not False
        or result.get("data_boundary", {}).get("model_fit_performed") is not False
        or marker != "RMLP-08A PASS\n"
    ):
        raise RealMLPValidationError("Trạng thái hoặc ranh giới gói Validation không hợp lệ.")

    expected_lineage = _lineage(
        Path(full_train_dir),
        split_manifest_path=Path(split_manifest_path),
        validation_csv=Path(split_dir) / "validation.csv",
        feature_contract_path=Path(feature_contract_path),
        model_protocol_path=Path(model_protocol_path),
    )
    if result.get("lineage") != expected_lineage:
        raise RealMLPValidationError("Mã băm nguồn gốc của gói Validation không khớp.")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        "validation_predictions.csv",
        "validation_result.json",
    }:
        raise RealMLPValidationError("Danh sách mã băm hiện vật Validation không hợp lệ.")
    for name, expected_hash in hashes.items():
        if _sha256_file(directory / name) != expected_hash:
            raise RealMLPValidationError(f"Mã băm hiện vật Validation sai: {name}")

    loader = DatasetBSplitLoader(
        manifest_path=split_manifest_path,
        split_dir=split_dir,
        feature_contract_path=feature_contract_path,
    )
    partition = loader.load("validation")
    expected_ids = tuple(partition.record_ids.astype(str).tolist())
    expected_labels = partition.y.to_numpy(dtype=np.int64)
    if (
        len(expected_ids) != EXPECTED_VALIDATION_ROWS
        or len(set(expected_ids)) != EXPECTED_VALIDATION_ROWS
        or result.get("data_boundary", {}).get("validation_record_ids_sha256")
        != _sha256_ids(expected_ids)
    ):
        raise RealMLPValidationError("Thành viên Validation không khớp split đã khóa.")
    rows = _read_predictions(directory / "validation_predictions.csv")
    observed_ids = tuple(row["record_id"] for row in rows)
    try:
        observed_labels = np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int64)
        observed_probabilities = np.asarray(
            [float(row["p_approved"]) for row in rows], dtype=np.float64
        )
    except (KeyError, ValueError) as exc:
        raise RealMLPValidationError("Giá trị trong tệp dự đoán Validation không hợp lệ.") from exc
    if (
        observed_ids != expected_ids
        or not np.array_equal(observed_labels, expected_labels)
        or not _valid_probabilities(observed_probabilities)
    ):
        raise RealMLPValidationError(
            "Thứ tự, nhãn hoặc xác suất dự đoán không khớp Validation đã khóa."
        )
    if manifest.get("rows") != len(rows):
        raise RealMLPValidationError("Số dòng dự đoán không khớp tệp kê khai.")
    return manifest


def _prediction_summary(
    record_ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    return {
        "rows": len(record_ids),
        "unique_record_ids": len(set(record_ids)),
        "approved_rows": int(np.count_nonzero(labels == 1)),
        "denied_rows": int(np.count_nonzero(labels == 0)),
        "probability_min": float(np.min(probabilities)),
        "probability_q25": float(np.quantile(probabilities, 0.25)),
        "probability_median": float(np.median(probabilities)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_q75": float(np.quantile(probabilities, 0.75)),
        "probability_max": float(np.max(probabilities)),
    }


def _valid_probabilities(probabilities: np.ndarray) -> bool:
    return bool(
        probabilities.shape == (EXPECTED_VALIDATION_ROWS,)
        and np.isfinite(probabilities).all()
        and np.all(probabilities >= 0.0)
        and np.all(probabilities <= 1.0)
    )


def _write_predictions(
    path: Path,
    record_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        for record_id, label, probability in zip(
            record_ids, labels, probabilities, strict=True
        ):
            writer.writerow([record_id, int(label), repr(float(probability))])


def _read_predictions(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise RealMLPValidationError("Cột trong tệp dự đoán Validation không đúng.")
            rows = list(reader)
    except OSError as exc:
        raise RealMLPValidationError(f"Không thể đọc hiện vật dự đoán: {path}") from exc
    return rows


def _lineage(
    full_train_dir: Path,
    *,
    split_manifest_path: Path,
    validation_csv: Path,
    feature_contract_path: Path,
    model_protocol_path: Path,
) -> dict[str, str]:
    paths = {
        "full_train_manifest_sha256": full_train_dir / "manifest.json",
        "full_train_result_sha256": full_train_dir / "full_train_result.json",
        "fitted_model_sha256": full_train_dir / "checkpoints" / "fitted_model.pkl",
        "split_manifest_sha256": split_manifest_path,
        "validation_source_sha256": validation_csv,
        "feature_contract_sha256": feature_contract_path,
        "model_protocol_sha256": model_protocol_path,
    }
    try:
        return {name: _sha256_file(path) for name, path in paths.items()}
    except OSError as exc:
        raise RealMLPValidationError("Không thể tạo mã băm nguồn gốc Validation.") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPValidationError(f"Không thể đọc tệp JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RealMLPValidationError(f"Tệp JSON phải chứa một đối tượng: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc lệnh chạy hoặc kiểm tra gói dự đoán Validation."""
    parser = argparse.ArgumentParser(description="Dự đoán và kiểm tra Validation RealMLP.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--full-train-dir", type=Path, default=DEFAULT_FULL_TRAIN_DIR)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate_parser.add_argument("--full-train-dir", type=Path, default=DEFAULT_FULL_TRAIN_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy lệnh được yêu cầu và in kết quả kiểm tra."""
    args = parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_realmlp_validation(
                args.output_dir, full_train_dir=args.full_train_dir
            )
        else:
            manifest = validate_realmlp_validation(
                args.output_dir, full_train_dir=args.full_train_dir
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except RealMLPValidationError as exc:
        print(f"Dự đoán Validation RealMLP thất bại: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
