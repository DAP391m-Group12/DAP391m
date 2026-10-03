"""RF-08 boundary tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import rf_validation_threshold as validation
from src.dataset_b.evaluation.rf_validation_threshold import _predict_validation_once, _select_threshold


class CountingPipeline:
    def __init__(self): self.predict_calls = 0; self.fit_calls = 0
    def fit(self, *_args, **_kwargs): self.fit_calls += 1; raise AssertionError("RF-08 must not fit")
    def predict_proba(self, frame):
        self.predict_calls += 1; approved = np.linspace(0.1, 0.9, len(frame)); return np.column_stack((1 - approved, approved))


def test_validation_inference_occurs_once_and_never_fits():
    pipeline = CountingPipeline(); values = _predict_validation_once(pipeline, pd.DataFrame({"x": [1, 2, 3]}))
    assert pipeline.predict_calls == 1 and pipeline.fit_calls == 0
    assert values.tolist() == pytest.approx([0.1, 0.5, 0.9])


def test_tie_rule_prefers_nearest_half_then_lower():
    selected, maximizing = _select_threshold([
        {"threshold": 0.4, "balanced_accuracy": 0.8},
        {"threshold": 0.6, "balanced_accuracy": 0.8},
        {"threshold": 0.7, "balanced_accuracy": 0.7},
    ])
    assert len(maximizing) == 2 and selected["threshold"] == 0.4


def test_full_runner_self_validates_and_rejects_validation_source_tampering(tmp_path, monkeypatch):
    run_id = "rf_run"; run_dir = tmp_path / run_id; (run_dir / "checkpoints").mkdir(parents=True)
    for relative in ("manifest.json", "cv_results.json", "full_train_result.json", "checkpoints/fitted_pipeline.joblib"):
        (run_dir / relative).write_text("{}\n", encoding="utf-8")
    split_manifest = tmp_path / "split.json"; split_manifest.write_text("{}\n", encoding="utf-8")
    feature = tmp_path / "feature.yaml"; feature.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "split"; split_dir.mkdir(); source = split_dir / "validation.csv"; source.write_text("source-v1\n", encoding="utf-8")
    ids = ("r1", "r2", "r3", "r4"); labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    class Audit:
        def as_dict(self): return {"status": "PASS", "retained_rows": 4}
    gated = SimpleNamespace(X=pd.DataFrame({"x": range(4)}), y=pd.Series(labels), record_ids=pd.Series(ids), quality_audit=Audit())
    class Loader:
        def __init__(self, **_kwargs): pass
        def load(self, partition): assert partition == "validation"; return gated
    class Gate:
        def apply(self, value): return value
    pipeline = CountingPipeline()
    training = {
        "feature_version": "feature_set_v1", "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "preprocessing_manifest": str((run_dir / "manifest.json").resolve()),
    }
    monkeypatch.setattr(validation, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(validation, "DatasetBSplitLoader", Loader); monkeypatch.setattr(validation, "DatasetBInputQualityGate", Gate)
    monkeypatch.setattr(validation, "validate_rf_final_training", lambda *_a, **_k: training)
    monkeypatch.setattr(validation, "_load_pipeline", lambda _path: pipeline)
    result = validation.run_rf_validation_threshold(
        run_id, run_dir=run_dir, split_manifest_path=split_manifest,
        split_dir=split_dir, feature_contract_path=feature,
    )
    assert result["status"] == "PASS" and result["rows"] == 4
    assert pipeline.predict_calls == 1 and pipeline.fit_calls == 0
    prediction = json.loads((run_dir / "validation/prediction_manifest.json").read_text(encoding="utf-8"))
    assert prediction["execution"]["predict_proba_calls"] == 1
    manifest_path = run_dir / "validation/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lineage"]["validation_source"] = str((tmp_path / "wrong.csv").resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(validation.RFValidationError, match="source lineage"):
        validation.validate_rf_validation_threshold(
            run_dir / "validation", run_dir=run_dir, run_id=run_id,
            split_manifest_path=split_manifest, split_dir=split_dir, feature_contract_path=feature,
        )
