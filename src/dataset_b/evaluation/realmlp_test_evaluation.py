"""Đánh giá RQ1 trên Test bằng hiện vật RealMLP đã khóa."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, root
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset_b.evaluation.realmlp_validation import DEFAULT_FULL_TRAIN_DIR
from src.dataset_b.evaluation.realmlp_validation_threshold import DEFAULT_OUTPUT_DIR as DEFAULT_VALIDATION_DIR
from src.dataset_b.models.realmlp import load_realmlp_v1_config
from src.dataset_b.preprocessing import DatasetBSplitLoader
from src.dataset_b.runtime.realmlp_full_train import (
    _sha256_file,
    _sha256_ids,
    validate_realmlp_full_train,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/dataset_b/realmlp_v1/test"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = DatasetBSplitLoader.DEFAULT_SPLIT_DIR
DEFAULT_FEATURE_CONTRACT = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
DEFAULT_MODEL_PROTOCOL = CONFIG_DIR / "dataset_b/model_protocol_v1.yaml"
DEFAULT_BOOTSTRAP_INDICES = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"
DEFAULT_BOOTSTRAP_MANIFEST = PROJECT_ROOT / "artifacts/dataset_b/rq1_test_bootstrap_manifest.json"
EXPECTED_TEST_ROWS = 43_996
LOCKED_THRESHOLD = 0.771
PREDICTION_COLUMNS = (
    "record_id", "y_approved", "p_approved", "p_denied",
    "predicted_approved", "predicted_denied",
)
METRIC_NAMES = (
    "roc_auc_approved", "average_precision_denied", "precision_denied",
    "recall_denied", "f1_denied", "recall_approved", "balanced_accuracy",
    "brier_approved", "calibration_intercept", "calibration_slope",
)
BOOTSTRAP_COLUMNS = ("replicate_id", *METRIC_NAMES, "invalid_metrics")
FINAL_FILES = {"COMPLETED", "run_manifest.json", "test_predictions.csv", "rq1_test_metrics.json", "rq1_bootstrap_metrics.csv"}


class RealMLPTestEvaluationError(RuntimeError):
    """Lỗi vi phạm hợp đồng đánh giá Test RealMLP."""


def run_realmlp_test_evaluation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    full_train_dir: str | Path = DEFAULT_FULL_TRAIN_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR.parent,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT,
    model_protocol_path: str | Path = DEFAULT_MODEL_PROTOCOL,
    bootstrap_indices_path: str | Path = DEFAULT_BOOTSTRAP_INDICES,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST,
) -> dict[str, Any]:
    """Dự báo Test đúng một lượt, tính chỉ số và công bố gói bất biến."""
    destination = Path(output_dir)
    paths = _input_paths(full_train_dir, validation_dir, split_manifest_path, split_dir,
                         feature_contract_path, model_protocol_path, bootstrap_indices_path,
                         bootstrap_manifest_path)
    if destination.exists():
        return {**validate_realmlp_test_evaluation(destination, **_kwargs(paths)), "resume_action": "SKIPPED_VALID_COMPLETED"}
    working = destination.with_name(destination.name + ".tmp")
    if working.exists():
        # Chỉ khôi phục thư mục tạm do lần chạy trước tạo nếu toàn bộ gói vượt validator.
        validate_realmlp_test_evaluation(working, **_kwargs(paths))
        os.replace(working, destination)
        return validate_realmlp_test_evaluation(destination, **_kwargs(paths))
    try:
        full_manifest = validate_realmlp_full_train(Path(full_train_dir))
        preprocessing_version = _realmlp_preprocessing_version()
        from src.dataset_b.evaluation.realmlp_validation_threshold import validate_realmlp_validation_threshold
        validate_realmlp_validation_threshold(Path(validation_dir))
        selection_path = Path(validation_dir) / "threshold_selection/threshold_selection.json"
        selection = _read_json(selection_path)
        if selection.get("selected_threshold") != LOCKED_THRESHOLD:
            raise RealMLPTestEvaluationError("Ngưỡng không khớp quyết định Validation đã khóa.")
        loader = DatasetBSplitLoader(manifest_path=split_manifest_path, split_dir=split_dir,
                                     feature_contract_path=feature_contract_path)
        partition = loader.load("test")
        ids = tuple(partition.record_ids.astype(str).tolist())
        y = partition.y.to_numpy(dtype=np.int8)
        if len(ids) != EXPECTED_TEST_ROWS or len(set(ids)) != EXPECTED_TEST_ROWS or not np.isin(y, (0, 1)).all():
            raise RealMLPTestEvaluationError("Kích thước, tính duy nhất của ID hoặc nhãn Test sai.")
        if _sha256_ids(ids) != _read_json(bootstrap_manifest_path).get("test_record_id_sha256"):
            raise RealMLPTestEvaluationError("ID Test không khớp dấu vết bootstrap chung.")
        bundle = joblib.load(Path(full_train_dir) / "checkpoints/fitted_model.pkl")
        if not isinstance(bundle, dict) or set(bundle) != {"estimator", "preprocessing", "scope"}:
            raise RealMLPTestEvaluationError("Gói mô hình đã lưu không hợp lệ.")
        started = time.perf_counter()
        transformed = bundle["preprocessing"].transform(partition.X.reset_index(drop=True))
        probabilities = np.asarray(bundle["estimator"].predict_proba(transformed), dtype=np.float64)
        inference_seconds = time.perf_counter() - started
        classes = np.asarray(getattr(bundle["estimator"], "classes_", ()))
        positive = np.flatnonzero(classes == 1)
        if probabilities.shape != (EXPECTED_TEST_ROWS, 2) or positive.shape != (1,):
            raise RealMLPTestEvaluationError("Ma trận xác suất hoặc thứ tự lớp mô hình không hợp lệ.")
        p = probabilities[:, int(positive[0])]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise RealMLPTestEvaluationError("Xác suất Test không hữu hạn hoặc ngoài [0, 1].")
        predictions = pd.DataFrame({"record_id": ids, "y_approved": y, "p_approved": p,
                                    "p_denied": 1-p, "predicted_approved": (p >= LOCKED_THRESHOLD).astype(np.int8),
                                    "predicted_denied": (p < LOCKED_THRESHOLD).astype(np.int8)})
        point, invalid, counts = _metrics(y, p, LOCKED_THRESHOLD)
        if invalid:
            raise RealMLPTestEvaluationError(f"Chỉ số điểm không hợp lệ: {invalid}")
        curve = _calibration_curve(y, p)
        with np.load(bootstrap_indices_path, allow_pickle=False) as data:
            indices = data["indices"]
        bootstrap_meta = _read_json(bootstrap_manifest_path)
        if (_sha256_file(Path(bootstrap_indices_path)) != bootstrap_meta.get("indices_sha256")
                or indices.shape != (1000, EXPECTED_TEST_ROWS) or indices.dtype != np.int32
                or not np.all((indices >= 0) & (indices < EXPECTED_TEST_ROWS))
                or not np.all(y[indices[:, :counts["n_denied"]]] == 0)
                or not np.all(y[indices[:, counts["n_denied"]:]] == 1)):
            raise RealMLPTestEvaluationError("Ma trận bootstrap không khớp hợp đồng chung đã khóa.")
        rows = []
        for rep, row_indices in enumerate(indices):
            values, bad, _ = _metrics(y[row_indices], p[row_indices], LOCKED_THRESHOLD)
            rows.append({"replicate_id": rep, **values,
                         "invalid_metrics": json.dumps(bad, ensure_ascii=False, sort_keys=True)})
        frame = pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)
        intervals = {}
        for name in METRIC_NAMES:
            values = frame[name].to_numpy(dtype=np.float64)
            valid = values[np.isfinite(values)]
            intervals[name] = {"lower_2_5": float(np.quantile(valid, .025, method="linear")) if len(valid) else None,
                                "upper_97_5": float(np.quantile(valid, .975, method="linear")) if len(valid) else None,
                                "valid_replicates": int(len(valid)), "invalid_replicates": int(len(values)-len(valid))}
        destination.parent.mkdir(parents=True, exist_ok=True)
        working.mkdir()
        pred_path = working / "test_predictions.csv"
        pred_path.write_text(predictions.to_csv(index=False, float_format="%.17g", lineterminator="\n"), encoding="utf-8", newline="")
        boot_path = working / "rq1_bootstrap_metrics.csv"
        boot_path.write_text(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n"), encoding="utf-8", newline="")
        lineage = {
            "full_train_manifest_sha256": _sha256_file(Path(full_train_dir)/"manifest.json"),
            "fitted_model_sha256": _sha256_file(Path(full_train_dir)/"checkpoints/fitted_model.pkl"),
            "validation_threshold_sha256": _sha256_file(selection_path),
            "split_manifest_sha256": _sha256_file(Path(split_manifest_path)),
            "test_source_sha256": _sha256_file(Path(split_dir)/"test.csv"),
            "feature_contract_sha256": _sha256_file(Path(feature_contract_path)),
            "model_protocol_sha256": _sha256_file(Path(model_protocol_path)),
            "bootstrap_manifest_sha256": _sha256_file(Path(bootstrap_manifest_path)),
            "bootstrap_indices_sha256": _sha256_file(Path(bootstrap_indices_path)),
            "test_record_ids_sha256": _sha256_ids(ids),
        }
        result = {
            "schema_version": 1, "run_id": "realmlp_v1", "evaluation_label": "realmlp_v1 / experimental feature lock v0.2",
            "evaluation_status": "complete_rq1_evaluation_experimental_feature_v0_2",
            "feature_version": "temporary_training_lock_v0_2", "split_version": "split_v1",
            "preprocessing_version": preprocessing_version, "model_protocol_version": "model_protocol_v1",
            "partition": "test", "rows": len(y), "threshold": LOCKED_THRESHOLD,
            "prediction_file": "test_predictions.csv", "prediction_sha256": _sha256_file(pred_path),
            "model_sha256": lineage["fitted_model_sha256"], "bootstrap_indices_file": Path(bootstrap_indices_path).name,
            "bootstrap_indices_sha256": lineage["bootstrap_indices_sha256"], "bootstrap_replicates_file": boot_path.name,
            "bootstrap_replicates_sha256": _sha256_file(boot_path), "counts": counts, "point_metrics": point,
            "confidence_intervals_95_percentile": intervals, "calibration_curve": curve,
            "computational_metrics_seconds": {
                "full_train_runner_elapsed": _read_json(Path(full_train_dir)/"full_train_result.json")["execution"]["elapsed_seconds"],
                "test_inference_including_preprocessing": inference_seconds,
            },
            "methods": _methods(), "lineage_sha256": lineage,
            "execution": {"model_fit_calls": 0, "preprocessing_transform_calls": 1, "predict_proba_calls": 1},
            "data_boundary": {"source_partition": "test", "model_fit_performed": False,
                              "threshold_changed": False, "shared_bootstrap_regenerated": False},
            "full_train_source_status": full_manifest.get("status"),
        }
        metrics_path = working / "rq1_test_metrics.json"
        _write_json(metrics_path, result)
        manifest = {"schema_version": 1, "contract_id": "RMLP-09", "status": "PASS",
                    "rows": len(y), "threshold": LOCKED_THRESHOLD, "lineage_sha256": lineage,
                    "artifact_sha256": {name: _sha256_file(working/name) for name in
                                        ("test_predictions.csv", "rq1_test_metrics.json", "rq1_bootstrap_metrics.csv")}}
        _write_json(working / "run_manifest.json", manifest)
        (working / "COMPLETED").write_text("RMLP-09 PASS\n", encoding="utf-8")
        validate_realmlp_test_evaluation(working, **_kwargs(paths))
        os.replace(working, destination)
        return validate_realmlp_test_evaluation(destination, **_kwargs(paths))
    except RealMLPTestEvaluationError:
        raise
    except Exception as exc:
        raise RealMLPTestEvaluationError(f"Không thể công bố đánh giá Test RealMLP: {exc}") from exc


def validate_realmlp_test_evaluation(output_dir: str | Path = DEFAULT_OUTPUT_DIR, **kwargs: Any) -> dict[str, Any]:
    """Kiểm tra cấu trúc, mã băm, dự đoán và chỉ số đã công bố."""
    root = Path(output_dir)
    actual = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()
    if actual != FINAL_FILES or (root/"COMPLETED").read_text(encoding="utf-8") != "RMLP-09 PASS\n":
        raise RealMLPTestEvaluationError("Tập tệp hoặc dấu hoàn tất RMLP-09 sai.")
    manifest = _read_json(root/"run_manifest.json")
    metrics = _read_json(root/"rq1_test_metrics.json")
    expected_artifacts = {"test_predictions.csv", "rq1_test_metrics.json", "rq1_bootstrap_metrics.csv"}
    if set(manifest.get("artifact_sha256", {})) != expected_artifacts:
        raise RealMLPTestEvaluationError("Danh sách mã băm hiện vật không đầy đủ.")
    for name, digest in manifest.get("artifact_sha256", {}).items():
        if _sha256_file(root/name) != digest:
            raise RealMLPTestEvaluationError(f"Mã băm hiện vật sai: {name}")
    frame = pd.read_csv(root/"test_predictions.csv", dtype={"record_id": str})
    if tuple(frame.columns) != PREDICTION_COLUMNS or len(frame) != EXPECTED_TEST_ROWS or not frame.record_id.is_unique:
        raise RealMLPTestEvaluationError("Bảng dự đoán sai cấu trúc hoặc số dòng.")
    y = frame.y_approved.to_numpy(dtype=np.int8); p = frame.p_approved.to_numpy(dtype=np.float64)
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or not np.allclose(frame.p_denied, 1-p, atol=1e-15, rtol=0):
        raise RealMLPTestEvaluationError("Nhãn hoặc xác suất trong bảng dự đoán sai.")
    if (not np.array_equal(frame.predicted_approved, (p >= LOCKED_THRESHOLD).astype(np.int8))
            or not np.array_equal(frame.predicted_denied, (p < LOCKED_THRESHOLD).astype(np.int8))):
        raise RealMLPTestEvaluationError("Nhãn dự đoán không tuân theo ngưỡng đã khóa.")
    point, invalid, counts = _metrics(y, p, LOCKED_THRESHOLD)
    stored_point = metrics.get("point_metrics", {})
    if (invalid or counts != metrics.get("counts") or set(point) != set(stored_point)
            or any(not np.isclose(point[k], stored_point[k], rtol=0.0, atol=1e-12) for k in point)):
        raise RealMLPTestEvaluationError("Chỉ số điểm không khớp dữ liệu dự đoán.")
    bootstrap = pd.read_csv(root/"rq1_bootstrap_metrics.csv")
    if len(bootstrap) != 1000 or tuple(bootstrap.columns) != BOOTSTRAP_COLUMNS:
        raise RealMLPTestEvaluationError("Bảng bootstrap không khớp hợp đồng.")
    if metrics.get("prediction_sha256") != _sha256_file(root/"test_predictions.csv"):
        raise RealMLPTestEvaluationError("Mã băm dự đoán trong tệp chỉ số sai.")
    if metrics.get("bootstrap_replicates_sha256") != _sha256_file(root/"rq1_bootstrap_metrics.csv"):
        raise RealMLPTestEvaluationError("Mã băm bảng bootstrap trong tệp chỉ số sai.")
    for name in METRIC_NAMES:
        values = bootstrap[name].to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        ci = metrics["confidence_intervals_95_percentile"][name]
        low = float(np.quantile(valid, .025, method="linear")) if len(valid) else None
        high = float(np.quantile(valid, .975, method="linear")) if len(valid) else None
        if (ci.get("valid_replicates") != len(valid) or ci.get("invalid_replicates") != len(values)-len(valid)
                or (low is None) != (ci.get("lower_2_5") is None)
                or (low is not None and not np.isclose(low, ci["lower_2_5"], rtol=0, atol=1e-14))
                or (high is not None and not np.isclose(high, ci["upper_97_5"], rtol=0, atol=1e-14))):
            raise RealMLPTestEvaluationError(f"Khoảng tin cậy bootstrap sai: {name}")
    resolved = _input_paths(
        kwargs.get("full_train_dir", DEFAULT_FULL_TRAIN_DIR),
        kwargs.get("validation_dir", DEFAULT_VALIDATION_DIR.parent),
        kwargs.get("split_manifest_path", DEFAULT_SPLIT_MANIFEST),
        kwargs.get("split_dir", DEFAULT_SPLIT_DIR),
        kwargs.get("feature_contract_path", DEFAULT_FEATURE_CONTRACT),
        kwargs.get("model_protocol_path", DEFAULT_MODEL_PROTOCOL),
        kwargs.get("bootstrap_indices_path", DEFAULT_BOOTSTRAP_INDICES),
        kwargs.get("bootstrap_manifest_path", DEFAULT_BOOTSTRAP_MANIFEST),
    )
    loader = DatasetBSplitLoader(manifest_path=resolved["split_manifest_path"],
                                 split_dir=resolved["split_dir"],
                                 feature_contract_path=resolved["feature_contract_path"])
    source = loader.load("test")
    ids = tuple(source.record_ids.astype(str).tolist())
    if ids != tuple(frame.record_id.astype(str).tolist()) or not np.array_equal(source.y.to_numpy(dtype=np.int8), y):
        raise RealMLPTestEvaluationError("Dự đoán không khớp ID, thứ tự hoặc nhãn trong Test đã khóa.")
    expected_lineage = {
        "full_train_manifest_sha256": _sha256_file(resolved["full_train_dir"] / "manifest.json"),
        "fitted_model_sha256": _sha256_file(resolved["full_train_dir"] / "checkpoints/fitted_model.pkl"),
        "validation_threshold_sha256": _sha256_file(resolved["validation_dir"] / "threshold_selection/threshold_selection.json"),
        "split_manifest_sha256": _sha256_file(resolved["split_manifest_path"]),
        "test_source_sha256": _sha256_file(resolved["split_dir"] / "test.csv"),
        "feature_contract_sha256": _sha256_file(resolved["feature_contract_path"]),
        "model_protocol_sha256": _sha256_file(resolved["model_protocol_path"]),
        "bootstrap_manifest_sha256": _sha256_file(resolved["bootstrap_manifest_path"]),
        "bootstrap_indices_sha256": _sha256_file(resolved["bootstrap_indices_path"]),
        "test_record_ids_sha256": _sha256_ids(ids),
    }
    if (manifest.get("lineage_sha256") != expected_lineage
            or metrics.get("lineage_sha256") != expected_lineage
            or metrics.get("model_sha256") != expected_lineage["fitted_model_sha256"]
            or manifest.get("rows") != EXPECTED_TEST_ROWS
            or metrics.get("execution") != {"model_fit_calls": 0, "preprocessing_transform_calls": 1, "predict_proba_calls": 1}
            or metrics.get("data_boundary", {}).get("model_fit_performed") is not False
            or metrics.get("data_boundary", {}).get("threshold_changed") is not False
            or metrics.get("data_boundary", {}).get("shared_bootstrap_regenerated") is not False
            or metrics.get("preprocessing_version") != _realmlp_preprocessing_version()):
        raise RealMLPTestEvaluationError("Dấu vết nguồn hoặc ranh giới chạy không khớp.")
    if manifest.get("contract_id") != "RMLP-09" or manifest.get("status") != "PASS" or metrics.get("threshold") != LOCKED_THRESHOLD:
        raise RealMLPTestEvaluationError("Trạng thái, ngưỡng hoặc hợp đồng không khớp.")
    return manifest


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    pred = p >= threshold; approved = y == 1; denied = ~approved
    tp = int(np.count_nonzero(approved & pred)); fn = int(np.count_nonzero(approved & ~pred))
    tn = int(np.count_nonzero(denied & ~pred)); fp = int(np.count_nonzero(denied & pred))
    na, nd = tp+fn, tn+fp
    if na == 0 or nd == 0: raise ArithmeticError("Cần có đủ hai lớp thật.")
    vals = {"roc_auc_approved": float(roc_auc_score(y,p)),
            "average_precision_denied": float(average_precision_score(1-y,1-p)),
            "precision_denied": tn/(tn+fn) if tn+fn else float("nan"), "recall_denied": tn/nd,
            "f1_denied": 2*tn/(2*tn+fn+fp), "recall_approved": tp/na,
            "balanced_accuracy": (tp/na+tn/nd)/2, "brier_approved": float(np.mean((p-y)**2))}
    vals["calibration_intercept"], vals["calibration_slope"], bad = _calibration(y,p)
    invalid = dict(bad)
    if not np.isfinite(vals["precision_denied"]): invalid["precision_denied"] = "no_predicted_denied"
    for k,v in vals.items():
        if not np.isfinite(v): invalid.setdefault(k,"nonfinite_metric")
    return vals, invalid, {"tp_approved":tp,"fn_approved":fn,"tn_denied":tn,"fp_denied":fp,"n_approved":na,"n_denied":nd}


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float, dict[str,str]]:
    z=logit(np.clip(p,1e-15,1-1e-15)); target=y.astype(np.float64); bad={}; intercept=float("nan"); slope=float("nan")
    try: intercept=float(brentq(lambda a: float(np.mean(expit(z+a)-target)),-80,80,xtol=1e-10))
    except (ArithmeticError,ValueError,RuntimeError) as exc: bad["calibration_intercept"]=str(exc)
    def fn(theta):
        eta=theta[0]+theta[1]*z; err=expit(eta)-target
        return float(np.mean(np.logaddexp(0,eta)-target*eta)),np.array([np.mean(err),np.mean(err*z)],dtype=np.float64)
    try:
        fit=minimize(fn,[0.,1.],method="L-BFGS-B",jac=True,options={"maxiter":100,"ftol":1e-12,"gtol":1e-9}); theta=fit.x
        if not fit.success or not np.isfinite(theta).all(): theta=root(lambda t:fn(t)[1],theta if np.isfinite(theta).all() else [0.,1.],method="hybr",tol=1e-10).x
        if not np.isfinite(theta).all() or np.max(np.abs(fn(theta)[1]))>1e-7: raise ArithmeticError("Điểm số hiệu chuẩn chưa hội tụ.")
        slope=float(theta[1])
    except (ArithmeticError,ValueError,RuntimeError) as exc: bad["calibration_slope"]=str(exc)
    return intercept,slope,bad


def _calibration_curve(y: np.ndarray,p: np.ndarray) -> list[dict[str,Any]]:
    edges=np.unique(np.quantile(p,np.linspace(0,1,11)))
    if len(edges)<2: raise ArithmeticError("Đường hiệu chuẩn cần ít nhất hai biên khác nhau.")
    ids=np.searchsorted(edges[1:-1],p,side="right"); rows=[]
    for i in range(len(edges)-1):
        mask=ids==i
        if mask.any(): rows.append({"bin":i,"lower_edge":float(edges[i]),"upper_edge":float(edges[i+1]),"count":int(mask.sum()),"mean_p_approved":float(p[mask].mean()),"observed_approval_rate":float(y[mask].mean())})
    if sum(r["count"] for r in rows)!=len(y): raise ArithmeticError("Các bin không bao phủ Test.")
    return rows


def _methods() -> dict[str,str]:
    return {"roc_auc_approved":"roc_auc_score(y_approved, p_approved)","average_precision_denied":"average_precision_score(1-y_approved, 1-p_approved)","predicted_approved":"p_approved >= locked threshold","brier_approved":"mean((p_approved-y_approved)^2)","calibration_curve":"10 quantile bins from Test probabilities; duplicate edges collapsed; no bootstrap CI","calibration_intercept":"logistic offset recalibration; slope fixed at 1","calibration_slope":"unpenalized logistic recalibration with free intercept and slope","bootstrap":"1,000 shared stratified row-index replicates; model and threshold fixed; percentile 2.5 and 97.5","timing_ci":"no bootstrap CI for timing"}


def _realmlp_preprocessing_version() -> str:
    """Return the preprocessing version from the validated RealMLP input contract."""
    config = load_realmlp_v1_config()
    value = config["input_contract"].get("preprocessing_version")
    if not isinstance(value, str) or not value:
        raise RealMLPTestEvaluationError(
            "Hợp đồng đầu vào RealMLP thiếu preprocessing_version hợp lệ."
        )
    return value


def _input_paths(full,validation,split_manifest,split_dir,feature,protocol,indices,bootstrap_manifest):
    return {"full_train_dir":Path(full),"validation_dir":Path(validation),"split_manifest_path":Path(split_manifest),"split_dir":Path(split_dir),"feature_contract_path":Path(feature),"model_protocol_path":Path(protocol),"bootstrap_indices_path":Path(indices),"bootstrap_manifest_path":Path(bootstrap_manifest)}

def _kwargs(paths): return {k:v for k,v in paths.items() if k!="split_dir"} | {"split_dir":paths["split_dir"]}
def _read_json(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise RealMLPTestEvaluationError(f"JSON không phải đối tượng: {path}")
    return value
def _write_json(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def parse_args(argv: Sequence[str]|None=None):
    parser=argparse.ArgumentParser(description="Đánh giá RQ1 Test của RealMLP."); subs=parser.add_subparsers(dest="command",required=True)
    for command in ("run","validate"):
        sub=subs.add_parser(command); sub.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)
def main(argv: Sequence[str]|None=None)->int:
    args=parse_args(argv)
    try:
        result=run_realmlp_test_evaluation(args.output_dir) if args.command=="run" else validate_realmlp_test_evaluation(args.output_dir)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
    except RealMLPTestEvaluationError as exc:
        print(f"Đánh giá Test RealMLP thất bại: {exc}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
