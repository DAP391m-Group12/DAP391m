"""LR-08 Validation prediction and threshold-selection contract tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.models.classical.lr.validation import (
    LRValidationError,
    _predict_validation_once,
    _resolved_paths,
    _select_threshold,
)
from src.dataset_b.evaluation import lr_validation_threshold as compatibility


def test_legacy_validation_module_delegates_to_canonical():
    assert compatibility.LRValidationError is LRValidationError
    assert compatibility._predict_validation_once is _predict_validation_once


class _CountingPipeline:
    def __init__(self):
        self.predict_calls = 0
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("LR-08 must not fit.")

    def predict_proba(self, frame):
        self.predict_calls += 1
        approved = np.linspace(0.1, 0.9, len(frame))
        return np.column_stack((1.0 - approved, approved))


def test_prediction_helper_calls_frozen_pipeline_exactly_once():
    pipeline = _CountingPipeline()
    frame = pd.DataFrame({"x": [1, 2, 3]})

    approved = _predict_validation_once(pipeline, frame)

    assert pipeline.predict_calls == 1
    assert pipeline.fit_calls == 0
    assert approved.tolist() == pytest.approx([0.1, 0.5, 0.9])


def test_threshold_tie_break_prefers_nearest_to_half_then_lower():
    selected, maximizing = _select_threshold(
        [
            {"threshold": 0.4, "balanced_accuracy": 0.8},
            {"threshold": 0.6, "balanced_accuracy": 0.8},
            {"threshold": 0.7, "balanced_accuracy": 0.7},
        ]
    )
    assert len(maximizing) == 2
    assert selected["threshold"] == 0.4


def test_output_path_must_be_current_run_validation(tmp_path):
    run_dir = tmp_path / "current_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    for relative in (
        "manifest.json",
        "cv_results.json",
        "full_train_result.json",
        "checkpoints/fitted_pipeline.joblib",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    paths = _resolved_paths(run_dir, run_dir / "validation")
    assert paths["output_dir"] == (run_dir / "validation").resolve()

    with pytest.raises(LRValidationError, match="current run_dir/validation"):
        _resolved_paths(run_dir, tmp_path / "other")
