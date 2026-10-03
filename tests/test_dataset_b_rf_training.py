"""Synthetic contract tests for RF-07; no real Random Forest is fitted."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from src.dataset_b.training import rf_final_training as training


def _preprocessing_export(root: Path, run_id: str, rows: int = 10) -> Path:
    destination = root / run_id; destination.mkdir(parents=True)
    folds = []
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), rows // 2)
    for fold in range(5):
        validation_positions = np.asarray([2 * fold, 2 * fold + 1], dtype=np.int64)
        train_positions = np.asarray([i for i in range(rows) if i not in validation_positions], dtype=np.int64)
        fold_dir = destination / f"fold_{fold + 1:02d}"; fold_dir.mkdir()
        save_npz(fold_dir / "tree_train.npz", csr_matrix(np.ones((len(train_positions), 62), dtype=np.float64)))
        save_npz(fold_dir / "tree_fold_validation.npz", csr_matrix(np.ones((2, 62), dtype=np.float64)))
        np.savez_compressed(
            fold_dir / "rows.npz", train_positions=train_positions,
            fold_validation_positions=validation_positions,
            train_record_ids=np.asarray([f"r{i}" for i in train_positions]),
            fold_validation_record_ids=np.asarray([f"r{i}" for i in validation_positions]),
            train_y=labels[train_positions], fold_validation_y=labels[validation_positions],
        )
        folds.append({"fold": fold + 1, "directory": fold_dir.name})
    manifest = {
        "run_id": run_id, "preprocessing_version": "dataset_b_preprocessing_v2",
        "source_partition": "train", "fit_scope": "fold_train_only", "use": "cross_validation_only",
        "feature_version": "feature_set_v1", "split_version": "split_v1",
        "raw_feature_count": 15, "semantic_feature_count": 19, "encoded_feature_count": 62,
        "numeric_transforms": {"rf": {"log1p_columns": [], "robust_scaled_columns": []}},
        "blank_row_audit": {"status": "PASS", "retained_rows": rows},
        "normalization_audit": {"raw_equals_normalized": True, "semantic_1111_count": 0},
        "sample_weights_exported": False, "fold_validation_coverage": {"exactly_once": True}, "folds": folds,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def test_protocol_and_tree_profile_are_fixed_balanced_without_log_or_scaling():
    protocol = training._load_protocol(training.DEFAULT_PROTOCOL_PATH)
    assert protocol["models"]["rf"]["parameters"]["class_weight"] == "balanced"
    pipeline = training.Pipeline([
        ("preprocess", training.make_preprocessing_pipeline("rf")),
        ("estimator", training._new_estimator(protocol["models"]["rf"]["parameters"])),
    ])
    training._validate_tree_pipeline(pipeline)
    assert pipeline.named_steps["preprocess"].named_steps["selective_log1p"] == "passthrough"
    assert pipeline.named_steps["preprocess"].named_steps["scale_numeric"] == "passthrough"


def test_preprocessing_manifest_enforces_15_19_62_and_nonlegacy(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "EXPECTED_TRAIN_ROWS", 10)
    export = _preprocessing_export(tmp_path, "rf_run")
    assert training._validate_preprocessing_manifest(export, "rf_run")["encoded_feature_count"] == 62
    payload = json.loads((export / "manifest.json").read_text(encoding="utf-8")); payload["semantic_feature_count"] = 18
    (export / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(training.RFTrainingError, match="19-field"):
        training._validate_preprocessing_manifest(export, "rf_run")
    with pytest.raises(training.RFTrainingError, match="Legacy"):
        training._validated_run_id("legacy_rf")


def test_training_bundle_inventory_allows_only_optional_frozen_marker(tmp_path, monkeypatch):
    required = {"manifest.json", "cv_results.json", "oof_predictions.npz", "full_train_result.json", "warnings.json", "COMPLETED"}
    root = tmp_path / "rf_run"
    root.mkdir()
    for name in required | {"FROZEN"}:
        (root / name).write_text("{}\n", encoding="utf-8")
    with pytest.raises(training.RFTrainingError, match="checkpoint"):
        training.validate_rf_final_training(root, run_id="rf_run")
    (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(training.RFTrainingError, match="incomplete or unexpected"):
        training.validate_rf_final_training(root, run_id="rf_run")


def test_runner_uses_five_fresh_fold_estimators_and_train_only(tmp_path, monkeypatch):
    run_id = "rf_test_run"; preprocessing_root = tmp_path / "pre"; _preprocessing_export(preprocessing_root, run_id)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 5)
    class Audit:
        def as_dict(self): return {"status": "PASS", "retained_rows": 10}
    gated = SimpleNamespace(
        X=pd.DataFrame(np.ones((10, 15)), columns=[f"x{i}" for i in range(15)]),
        y=pd.Series(labels), record_ids=pd.Series([f"r{i}" for i in range(10)]), quality_audit=Audit(),
    )
    loaded = []
    class Loader:
        feature_version = "feature_set_v1"
        def __init__(self, **_kwargs): pass
        def load(self, partition): loaded.append(partition); return gated
    class Gate:
        def apply(self, value): return value
    estimators = []
    class Estimator:
        def __init__(self, **params): self.params = params; self.classes_ = np.asarray([0, 1]); self.fit_rows = None; estimators.append(self)
        def fit(self, X, y): self.fit_rows = len(y); return self
        def predict_proba(self, X):
            rows = X.shape[0]
            approved = np.tile(np.asarray([0.25, 0.75]), (rows + 1) // 2)[:rows]
            return np.column_stack((1 - approved, approved))
        def get_params(self): return self.params
    class Preprocess:
        named_steps = {"selective_log1p": "passthrough", "scale_numeric": "passthrough"}
        def get_feature_names_out(self): return np.asarray([f"x{i}" for i in range(62)])
        def transform(self, X): return csr_matrix(np.ones((len(X), 62), dtype=np.float64))
    class FakePipeline:
        def __init__(self, steps): self.named_steps = {name: value for name, value in steps}
        def fit(self, X, y): self.named_steps["estimator"].fit(X, y); return self
        def predict_proba(self, X): return self.named_steps["estimator"].predict_proba(X)
    monkeypatch.setattr(training, "EXPECTED_TRAIN_ROWS", 10)
    monkeypatch.setattr(training, "DatasetBSplitLoader", Loader); monkeypatch.setattr(training, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(training, "RandomForestClassifier", Estimator); monkeypatch.setattr(training, "make_preprocessing_pipeline", lambda profile: Preprocess())
    monkeypatch.setattr(training, "Pipeline", FakePipeline)
    persisted = {}
    def dump(value, path): persisted[str(path)] = value; Path(path).write_bytes(b"synthetic-object")
    monkeypatch.setattr(training.joblib, "dump", dump)
    monkeypatch.setattr(training.joblib, "load", lambda path: persisted[str(path)])
    result = training.run_rf_final_training(run_id, preprocessing_root=preprocessing_root, output_root=tmp_path / "out")
    assert result["status"] == "PASS" and loaded == ["train"]
    assert len(estimators) == 6 and len({id(value) for value in estimators}) == 6
    assert [value.fit_rows for value in estimators[:5]] == [8] * 5
    assert estimators[-1].fit_rows == 10
    assert all(value.params["class_weight"] == "balanced" for value in estimators)
    manifest = json.loads((tmp_path / "out" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration_confirmed_from_fixed_protocol"] is True
    assert manifest["hyperparameter_search"] is False
    assert manifest["model_parameters"]["class_weight"] == "balanced"
    assert set(manifest["persisted_objects"]) == {"pipeline", "preprocessor", "model"}
    full = json.loads((tmp_path / "out" / run_id / "full_train_result.json").read_text(encoding="utf-8"))
    assert full["reload_predictions_exact"] is True and full["reload_prediction_rows_checked"] == 10
