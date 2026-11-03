"""Run one frozen checkpoint on the ordered Test partition; runs in the model's profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.dataset_b.data import DatasetBInputQualityGate, DatasetBSplitLoader
from src.utils.path import PROJECT_ROOT

from .manifest import ReproduceError
from .spec import (
    BOOTSTRAP_INDICES,
    BOOTSTRAP_MANIFEST,
    METRIC_NAMES,
    MODELS,
    OUTPUT_DIRECTORY,
    ModelSpec,
    check_record_order,
    check_run_identity,
    check_split_digest,
    check_threshold,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids_digest(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _predict_classical(checkpoint: Path, frame: Any) -> np.ndarray:
    import joblib

    pipeline = joblib.load(checkpoint)
    if not np.array_equal(pipeline.named_steps["estimator"].classes_, np.asarray([0, 1])):
        raise ReproduceError("Frozen pipeline class order differs")
    return np.asarray(pipeline.predict_proba(frame), dtype=np.float64)[:, 1]


def _predict_plain_mlp(checkpoint: Path, frame: Any) -> np.ndarray:
    from src.dataset_b.models.neural.plain_mlp.training import load_plain_mlp_bundle

    preprocessing, estimator, _ = load_plain_mlp_bundle(checkpoint)
    matrix = np.asarray(estimator.predict_proba(preprocessing.transform(frame)), dtype=np.float64)
    return matrix[:, int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0])]


def _predict_realmlp(checkpoint: Path, _frame: Any) -> np.ndarray:
    import joblib

    from src.dataset_b.models.neural.realmlp.final_contract import (
        load_quality_gated_partition,
        transform_native_preprocessing,
        validate_probabilities,
    )

    bundle = joblib.load(checkpoint)
    prepared = transform_native_preprocessing(bundle["preprocessing"], load_quality_gated_partition("test"))
    return validate_probabilities(bundle["estimator"].predict_proba(prepared.X), len(prepared.X))[:, 1]


def _predict_tabmmini(checkpoint: Path, frame: Any) -> np.ndarray:
    import joblib
    import torch

    from src.dataset_b.models.neural.tabmmini.estimator import TabMMiniModel
    from src.dataset_b.models.neural.tabmmini.training import _default_prediction_runner

    model, payload = TabMMiniModel.load_inference_bundle(checkpoint, map_location="cuda:0")
    model.to(torch.device("cuda:0"))
    preprocessor = joblib.load(checkpoint.with_name("preprocessor.joblib"))
    return _default_prediction_runner(model, preprocessor.transform(frame), payload["effective_batch_size"])


PREDICTORS: dict[str, Callable[[Path, Any], np.ndarray]] = {
    "logistic_regression": _predict_classical,
    "random_forest": _predict_classical,
    "xgboost": _predict_classical,
    "plain_mlp": _predict_plain_mlp,
    "realmlp": _predict_realmlp,
    "tabmmini": _predict_tabmmini,
}


def _metric_function(key: str) -> Callable[..., Any]:
    modules = {
        "logistic_regression": "src.dataset_b.models.classical.lr.test_evaluation",
        "random_forest": "src.dataset_b.models.classical.rf.test_evaluation",
        "xgboost": "src.dataset_b.models.classical.xgboost.test_evaluation",
        "plain_mlp": "src.dataset_b.models.neural.plain_mlp.test_evaluation",
        "realmlp": "src.dataset_b.models.neural.plain_mlp.test_evaluation",
        "tabmmini": "src.dataset_b.models.neural.plain_mlp.test_evaluation",
    }
    return getattr(__import__(modules[key], fromlist=["_metrics"]), "_metrics")


def _require_cuda(spec: ModelSpec) -> None:
    if not spec.requires_cuda:
        return
    import torch

    if not torch.cuda.is_available():
        raise ReproduceError(f"{spec.key}: CUDA is required and not available; no CPU fallback is allowed")
    # The peak counter exists only after the device has been initialized.
    torch.ones(1, device="cuda:0")
    torch.cuda.reset_peak_memory_stats(0)


def _cuda_evidence(spec: ModelSpec) -> dict[str, Any]:
    if not spec.requires_cuda:
        return {"device": "cpu"}
    import torch

    peak = int(torch.cuda.max_memory_allocated(0))
    if peak <= 0:
        raise ReproduceError(f"{spec.key}: inference allocated no CUDA memory, so it did not run on the GPU")
    return {
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_memory_allocated_bytes": peak,
    }


def _intervals(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[name] = {
            "lower_2_5": float(np.quantile(finite, 0.025, method="linear")) if len(finite) else None,
            "upper_97_5": float(np.quantile(finite, 0.975, method="linear")) if len(finite) else None,
            "valid_replicates": int(len(finite)),
            "invalid_replicates": int(len(values) - len(finite)),
        }
    return result


def evaluate(key: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    spec = MODELS[key]
    root = project_root.resolve()
    run_root = root / spec.run_root
    if not (run_root / "FROZEN").is_file():
        raise ReproduceError(f"{key}: FROZEN marker is missing")
    test_directory = run_root / spec.test_directory
    selection = _read_json(run_root / spec.threshold_file)
    stored = _read_json(test_directory / "rq1_test_metrics.json")
    check_run_identity(spec, selection, stored)
    threshold = check_threshold(spec, selection, stored)

    partition = DatasetBInputQualityGate().apply(DatasetBSplitLoader().load("test"))
    ids = partition.record_ids.astype(str).tolist()
    labels = partition.y.to_numpy(dtype=np.int8)
    with (test_directory / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
        stored_ids = [row["record_id"] for row in csv.DictReader(handle)]
    check_record_order(spec, ids, stored_ids)
    bootstrap_manifest_path = root / BOOTSTRAP_MANIFEST
    check_split_digest(spec, _ids_digest(ids), _read_json(bootstrap_manifest_path))

    checkpoint = run_root / spec.checkpoint
    checkpoint_before = _sha256(checkpoint)
    _require_cuda(spec)
    started = time.perf_counter()
    probabilities = PREDICTORS[key](checkpoint, partition.X.reset_index(drop=True))
    inference_seconds = time.perf_counter() - started
    device = _cuda_evidence(spec)
    if probabilities.shape != (len(ids),) or not np.isfinite(probabilities).all():
        raise ReproduceError(f"{key}: Test probabilities are invalid")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ReproduceError(f"{key}: Test probabilities are outside [0, 1]")
    if _sha256(checkpoint) != checkpoint_before:
        raise ReproduceError(f"{key}: checkpoint changed during evaluation")

    metrics = _metric_function(key)
    values64 = probabilities.astype(np.float64)
    point, invalid, counts = metrics(labels, values64, threshold)
    if invalid:
        raise ReproduceError(f"{key}: point metrics are invalid: {invalid}")
    with np.load(root / BOOTSTRAP_INDICES, allow_pickle=False) as payload:
        indices = payload["indices"]
    rows = []
    for replicate_id, positions in enumerate(indices):
        replicate, problems, _ = metrics(labels[positions], values64[positions], threshold)
        rows.append(
            {
                "replicate_id": replicate_id,
                **replicate,
                "invalid_metrics": json.dumps(problems, ensure_ascii=False, sort_keys=True),
            }
        )

    output = root / OUTPUT_DIRECTORY / key
    working = output.with_name(output.name + ".tmp")
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True)
    with (working / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("record_id", "y_approved", "p_approved"))
        for record_id, label, probability in zip(ids, labels, probabilities, strict=True):
            writer.writerow((record_id, int(label), str(probability) if probabilities.dtype == np.float32 else repr(float(probability))))
    with (working / "rq1_bootstrap_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("replicate_id", *METRIC_NAMES, "invalid_metrics"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": 1,
        "model": key,
        "run_id": spec.run_id,
        "rows": len(ids),
        "threshold": threshold,
        "point_metrics": point,
        "counts": counts,
        "confidence_intervals_95_percentile": _intervals(rows),
        "test_inference_seconds": inference_seconds,
    }
    (working / "rq1_test_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    receipt = {
        "classification": "checkpoint-based",
        "status": "EVALUATED",
        "model": key,
        "accepted_run_ids": [spec.run_id],
        "execution": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "checkpoint_inference_passes": 1, "training_code_called": False},
        "compute": device,
        "input_hashes": {
            spec.checkpoint: checkpoint_before,
            "split_v1/test.csv": _sha256(root / "data/dataset_b/split_v1/test.csv"),
            spec.threshold_file: _sha256(run_root / spec.threshold_file),
            BOOTSTRAP_INDICES: _sha256(root / BOOTSTRAP_INDICES),
        },
        "output_hashes": {
            name: _sha256(working / name)
            for name in ("test_predictions.csv", "rq1_bootstrap_metrics.csv", "rq1_test_metrics.json")
        },
        "limitations": ["Stored Test predictions are read only for record order and for comparison."],
    }
    (working / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if output.exists():
        shutil.rmtree(output)
    working.rename(output)
    return {"status": "EVALUATED", "model": key, "run_id": spec.run_id, "rows": len(ids), "compute": device}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
