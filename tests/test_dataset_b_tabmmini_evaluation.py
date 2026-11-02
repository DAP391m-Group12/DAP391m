"""Synthetic tests for TabMmini holdout and evaluation interfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation import tabmmini_final_evaluation as evaluation


RUN_ID = "tabmmini_synthetic_run"


class _NoFitPreprocessor:
    def __init__(self) -> None:
        self.transform_calls = 0

    def fit(self, *_args, **_kwargs):  # pragma: no cover - must never be called
        raise AssertionError("holdout fit is forbidden")

    def fit_transform(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("holdout fit_transform is forbidden")

    def partial_fit(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("holdout partial_fit is forbidden")

    def transform(self, X):
        self.transform_calls += 1
        return np.asarray(X)


def _population() -> evaluation.TabMMiniHoldoutPopulation:
    return evaluation.TabMMiniHoldoutPopulation(
        X=pd.DataFrame({"x": [0, 1, 2, 3]}),
        y_approved=np.asarray([0, 0, 1, 1], dtype=np.int8),
        record_ids=("r0", "r1", "r2", "r3"),
    )


def _probabilities() -> np.ndarray:
    return np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float32)


def _locked_indices(path: Path) -> np.ndarray:
    indices = np.tile(np.arange(4, dtype=np.int32), (1_000, 1))
    np.savez_compressed(path, indices=indices)
    return indices


def test_probability_adapter_preserves_canonical_order_and_float32():
    approved = _probabilities()
    matrix = evaluation.as_two_class_probabilities(approved)
    assert matrix.dtype == np.float32
    assert matrix.shape == (4, 2)
    np.testing.assert_array_equal(matrix[:, 1], approved)
    np.testing.assert_allclose(matrix[:, 0], 1.0 - approved, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
    with pytest.raises(evaluation.TabMMiniFinalEvaluationError, match="float32"):
        evaluation.as_two_class_probabilities(approved.astype(np.float64))


def test_shared_threshold_grid_and_lower_symmetric_tie_break(monkeypatch):
    selected, candidates = evaluation.select_shared_threshold(
        np.asarray([0, 0, 1, 1]), _probabilities()
    )
    assert len(candidates) == 999
    assert candidates[0]["threshold"] == 0.001
    assert candidates[-1]["threshold"] == 0.999
    assert selected["threshold"] == 0.5

    monkeypatch.setattr(
        evaluation, "THRESHOLD_GRID", np.asarray([0.4, 0.6], dtype=np.float64)
    )
    tied, _ = evaluation.select_shared_threshold(
        np.asarray([0, 1]), np.asarray([0.2, 0.8], dtype=np.float32)
    )
    assert tied["threshold"] == 0.4


def test_validation_is_one_transform_and_one_prediction_with_no_fit(tmp_path):
    preprocessor = _NoFitPreprocessor()
    calls = []

    def predict(_model, native, batch_size):
        calls.append((native.shape, batch_size))
        return _probabilities()

    manifest = evaluation.run_tabmmini_validation(
        RUN_ID,
        _population(),
        fitted_run_id=RUN_ID,
        preprocessor=preprocessor,
        model=object(),
        prediction_runner=predict,
        effective_batch_size=128,
        artifact_root=tmp_path,
    )
    assert preprocessor.transform_calls == 1
    assert calls == [((4, 1), 128)]
    assert manifest["execution"] == {
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "preprocessing_transform_calls": 1,
        "prediction_calls": 1,
    }
    with np.load(
        tmp_path / "working" / RUN_ID / "validation/validation_predictions.npz",
        allow_pickle=False,
    ) as payload:
        assert payload["p_approved"].dtype == np.float32
        assert "p_denied" not in payload.files
    with pytest.raises(FileExistsError):
        evaluation.run_tabmmini_validation(
            RUN_ID,
            _population(),
            fitted_run_id=RUN_ID,
            preprocessor=preprocessor,
            model=object(),
            prediction_runner=predict,
            effective_batch_size=128,
            artifact_root=tmp_path,
        )
    assert preprocessor.transform_calls == 1
    assert len(calls) == 1


def test_validation_rejects_fitted_bundle_from_another_run(tmp_path):
    preprocessor = _NoFitPreprocessor()
    with pytest.raises(evaluation.TabMMiniFinalEvaluationError, match="run_id"):
        evaluation.run_tabmmini_validation(
            RUN_ID,
            _population(),
            fitted_run_id="another_run",
            preprocessor=preprocessor,
            model=object(),
            prediction_runner=lambda *_args: _probabilities(),
            effective_batch_size=128,
            artifact_root=tmp_path,
        )
    assert preprocessor.transform_calls == 0


def test_locked_bootstrap_is_loaded_exactly_and_has_no_generation_path(tmp_path):
    path = tmp_path / "locked.npz"
    expected = _locked_indices(path)
    observed = evaluation.load_locked_bootstrap_indices(path, 4)
    assert np.array_equal(observed, expected)
    assert observed.dtype == np.int32
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, indices=expected.astype(np.int64))
    with pytest.raises(evaluation.TabMMiniFinalEvaluationError, match="invalid"):
        evaluation.load_locked_bootstrap_indices(bad, 4)


def test_test_evaluation_uses_one_prediction_and_existing_indices(
    tmp_path, monkeypatch
):
    preprocessor = _NoFitPreprocessor()
    calls = []

    def predict(_model, _native, _batch_size):
        calls.append("predict")
        return _probabilities()

    evaluation.run_tabmmini_validation(
        RUN_ID,
        _population(),
        fitted_run_id=RUN_ID,
        preprocessor=preprocessor,
        model=object(),
        prediction_runner=predict,
        effective_batch_size=64,
        artifact_root=tmp_path,
    )
    evaluation.select_tabmmini_threshold(RUN_ID, artifact_root=tmp_path)
    validation_root = tmp_path / "working" / RUN_ID / "validation"
    assert (validation_root / "threshold_selection.json").is_file()
    assert (validation_root / "threshold_candidates.csv").is_file()
    assert not (tmp_path / "working" / RUN_ID / "threshold").exists()
    locked_manifest = json.loads(
        (validation_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert locked_manifest["threshold_status"] == "LOCKED"
    assert locked_manifest["selected_threshold"] == 0.5
    with pytest.raises(FileExistsError, match="already locked"):
        evaluation.select_tabmmini_threshold(RUN_ID, artifact_root=tmp_path)
    indices_path = tmp_path / "locked.npz"
    _locked_indices(indices_path)
    metric_names = evaluation.METRIC_NAMES

    def fast_metrics(labels, probabilities, threshold):
        del probabilities, threshold
        values = {name: 0.5 for name in metric_names}
        counts = {
            "tp_approved": int(np.count_nonzero(labels == 1)),
            "fn_approved": 0,
            "tn_denied": int(np.count_nonzero(labels == 0)),
            "fp_denied": 0,
            "n_approved": int(np.count_nonzero(labels == 1)),
            "n_denied": int(np.count_nonzero(labels == 0)),
        }
        return values, {}, counts

    monkeypatch.setattr(evaluation, "_metrics", fast_metrics)
    monkeypatch.setattr(
        evaluation,
        "_calibration_curve",
        lambda *_args: [
            {
                "bin": 0,
                "lower_edge": 0.0,
                "upper_edge": 1.0,
                "count": 4,
                "mean_p_approved": 0.5,
                "observed_approval_rate": 0.5,
            }
        ],
    )
    result = evaluation.run_tabmmini_test_evaluation(
        RUN_ID,
        _population(),
        fitted_run_id=RUN_ID,
        preprocessor=preprocessor,
        model=object(),
        prediction_runner=predict,
        effective_batch_size=64,
        artifact_root=tmp_path,
        bootstrap_indices_path=indices_path,
    )
    assert calls == ["predict", "predict"]
    assert preprocessor.transform_calls == 2
    assert result["bootstrap_replicates"] == 1_000
    assert result["execution"]["bootstrap_indices_regenerated"] is False
    assert set(result["point_metrics"]) == set(metric_names)


def test_final_notebook_is_executed_and_artifact_read_only():
    path = Path(__file__).parents[1] / "notebooks/dataset_b/final/models/06_tabmmini_final.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in code_cells
    )
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert [cell.get("execution_count") for cell in code_cells] == [1, 2, 3, 4, 5]
    assert all(cell.get("outputs") for cell in code_cells[1:])
    assert errors == []
    assert all(
        token not in code
        for token in (
            ".fit(",
            ".predict(",
            ".transform(",
            "publish_",
            "write_text",
            "to_csv",
            "to_json",
            "FROZEN",
        )
    )
    assert "read_text" in code and "read_csv" in code
