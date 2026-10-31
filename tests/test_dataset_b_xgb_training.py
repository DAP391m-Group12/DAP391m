"""Synthetic contract tests for XGB-07; no real XGBoost model is fitted."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from src.dataset_b.models.classical.xgboost import training
from src.dataset_b.training import xgb_final_training as compatibility


LABELS = np.asarray(
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
    dtype=np.int8,
)


def test_legacy_training_module_delegates_to_canonical():
    assert compatibility.XGBTrainingError is training.XGBTrainingError
    assert compatibility.run_xgb_final_training is training.run_xgb_final_training


def _preprocessing_export(root: Path, run_id: str, rows: int = 15) -> Path:
    destination = root / run_id
    destination.mkdir(parents=True)
    labels = LABELS[:rows]
    folds = []
    for fold in range(5):
        validation_positions = np.asarray([3 * fold, 3 * fold + 1, 3 * fold + 2], dtype=np.int64)
        train_positions = np.asarray([i for i in range(rows) if i not in validation_positions], dtype=np.int64)
        fold_dir = destination / f"fold_{fold + 1:02d}"
        fold_dir.mkdir()
        save_npz(fold_dir / "tree_train.npz", csr_matrix(np.ones((len(train_positions), 62), dtype=np.float64)))
        save_npz(
            fold_dir / "tree_fold_validation.npz",
            csr_matrix(np.ones((len(validation_positions), 62), dtype=np.float64)),
        )
        np.savez_compressed(
            fold_dir / "rows.npz",
            train_positions=train_positions,
            fold_validation_positions=validation_positions,
            train_record_ids=np.asarray([f"r{i}" for i in train_positions]),
            fold_validation_record_ids=np.asarray([f"r{i}" for i in validation_positions]),
            train_y=labels[train_positions],
            fold_validation_y=labels[validation_positions],
        )
        folds.append({"fold": fold + 1, "directory": fold_dir.name})
    manifest = {
        "run_id": run_id,
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "source_partition": "train",
        "fit_scope": "fold_train_only",
        "use": "cross_validation_only",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "raw_feature_count": 15,
        "semantic_feature_count": 19,
        "encoded_feature_count": 62,
        "tree_profile_shared_by": ["rf", "xgb"],
        "numeric_transforms": {"xgb": {"log1p_columns": [], "robust_scaled_columns": []}},
        "blank_row_audit": {"status": "PASS", "retained_rows": rows},
        "normalization_audit": {"raw_equals_normalized": True, "semantic_1111_count": 0},
        "sample_weights_exported": False,
        "fold_validation_coverage": {"exactly_once": True},
        "folds": folds,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def test_protocol_has_fixed_base_parameters_and_executable_fit_local_weighting():
    protocol = training._load_protocol(training.DEFAULT_PROTOCOL_PATH)
    model = protocol["models"]["xgb"]
    assert "scale_pos_weight" not in model["parameters"]
    assert model["reporting_key"] == "xgboost"
    assert protocol["class_handling"]["xgb_weighting_policy"]["fit_populations"] == [
        "cv_fold_train",
        "full_train",
    ]
    effective, ratio = training._effective_parameters(model["parameters"], np.asarray([0, 0, 1]))
    assert ratio == {"n_denied": 2, "n_approved": 1, "scale_pos_weight": 2.0}
    assert effective["scale_pos_weight"] == 2.0


def test_class_ratio_refuses_single_class_and_fixed_weight():
    with pytest.raises(training.XGBTrainingError, match="both target classes"):
        training._class_ratio(np.asarray([0, 0], dtype=np.int8))
    with pytest.raises(training.XGBTrainingError, match="must not be fixed"):
        training._effective_parameters({"scale_pos_weight": 1.0}, np.asarray([0, 1], dtype=np.int8))


def test_preprocessing_manifest_enforces_15_19_62_run_id_and_tree_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "EXPECTED_TRAIN_ROWS", 15)
    export = _preprocessing_export(tmp_path, "xgb_run")
    assert training._validate_preprocessing_manifest(export, "xgb_run")["encoded_feature_count"] == 62
    with pytest.raises(training.XGBTrainingError, match="run_id"):
        training._validate_preprocessing_manifest(export, "other_run")
    payload = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    payload["semantic_feature_count"] = 18
    (export / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(training.XGBTrainingError, match="19-field"):
        training._validate_preprocessing_manifest(export, "xgb_run")


def test_runner_uses_six_fresh_estimators_and_fit_population_ratios(tmp_path, monkeypatch):
    run_id = "xgb_test_run"
    preprocessing_root = tmp_path / "pre"
    _preprocessing_export(preprocessing_root, run_id)

    class Audit:
        def as_dict(self):
            return {"status": "PASS", "retained_rows": 15}

    gated = SimpleNamespace(
        X=pd.DataFrame(np.ones((15, 15)), columns=[f"x{i}" for i in range(15)]),
        y=pd.Series(LABELS),
        record_ids=pd.Series([f"r{i}" for i in range(15)]),
        quality_audit=Audit(),
    )
    loaded = []

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def load(self, partition):
            loaded.append(partition)
            return gated

    class Gate:
        def apply(self, value):
            return value

    estimators = []

    class Estimator:
        def __init__(self, **params):
            self.params = params
            self.classes_ = np.asarray([0, 1])
            self.fit_labels = None
            estimators.append(self)

        def fit(self, _X, y):
            self.fit_labels = np.asarray(y)
            return self

        def predict_proba(self, X):
            rows = X.shape[0]
            approved = np.tile(np.asarray([0.25, 0.75]), (rows + 1) // 2)[:rows]
            return np.column_stack((1 - approved, approved))

        def get_params(self):
            return self.params

    class Preprocess:
        named_steps = {"selective_log1p": "passthrough", "scale_numeric": "passthrough"}

        def get_feature_names_out(self):
            return np.asarray([f"x{i}" for i in range(62)])

        def transform(self, X):
            return csr_matrix(np.ones((len(X), 62), dtype=np.float64))

    class FakePipeline:
        def __init__(self, steps):
            self.named_steps = {name: value for name, value in steps}

        def fit(self, X, y):
            self.named_steps["estimator"].fit(X, y)
            return self

        def predict_proba(self, X):
            return self.named_steps["estimator"].predict_proba(X)

    monkeypatch.setattr(training, "EXPECTED_TRAIN_ROWS", 15)
    monkeypatch.setattr(training, "DatasetBSplitLoader", Loader)
    monkeypatch.setattr(training, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(training, "XGBClassifier", Estimator)
    monkeypatch.setattr(training, "make_preprocessing_pipeline", lambda profile: Preprocess())
    monkeypatch.setattr(training, "Pipeline", FakePipeline)
    persisted = {}

    def dump(value, path):
        persisted[str(path)] = value
        Path(path).write_bytes(b"synthetic-object")

    monkeypatch.setattr(training.joblib, "dump", dump)
    monkeypatch.setattr(training.joblib, "load", lambda path: persisted[str(path)])
    result = training.run_xgb_final_training(
        run_id,
        preprocessing_root=preprocessing_root,
        output_root=tmp_path / "out",
    )
    assert result["status"] == "PASS"
    assert loaded == ["train"]
    assert len(estimators) == 6 and len({id(value) for value in estimators}) == 6
    expected_fold_ratios = [7 / 5, 7 / 5, 7 / 5, 7 / 5, 2.0]
    assert [value.params["scale_pos_weight"] for value in estimators[:5]] == pytest.approx(expected_fold_ratios)
    assert estimators[-1].params["scale_pos_weight"] == pytest.approx(9 / 6)
    assert [len(value.fit_labels) for value in estimators[:5]] == [12] * 5
    assert len(estimators[-1].fit_labels) == 15
    manifest = json.loads((tmp_path / "out" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_key"] == "xgb" and manifest["reporting_key"] == "xgboost"
    assert manifest["fresh_estimator_count"] == 6
    assert "scale_pos_weight" not in manifest["model_base_parameters"]
    assert manifest["full_train_effective_parameters"]["scale_pos_weight"] == pytest.approx(1.5)
    assert set(manifest["persisted_objects"]) == {"pipeline", "preprocessor", "model"}
    full = json.loads((tmp_path / "out" / run_id / "full_train_result.json").read_text(encoding="utf-8"))
    assert full["reload_predictions_exact"] is True
    assert full["reload_prediction_rows_checked"] == 15
    cv = json.loads((tmp_path / "out" / run_id / "cv_results.json").read_text(encoding="utf-8"))
    assert cv["fresh_estimator_count"] == 5


def test_existing_partial_destination_is_rejected(tmp_path):
    run_id = "xgb_partial"
    preprocessing_root = tmp_path / "pre"
    _preprocessing_export(preprocessing_root, run_id)
    partial = tmp_path / "out" / run_id
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(training.XGBTrainingError, match="incomplete or unexpected"):
        training.run_xgb_final_training(run_id, preprocessing_root=preprocessing_root, output_root=tmp_path / "out")


def test_training_source_has_no_private_rf_or_lr_dependency():
    source = inspect.getsource(training)
    assert "rf_final_training" not in source
    assert "lr_" not in source
