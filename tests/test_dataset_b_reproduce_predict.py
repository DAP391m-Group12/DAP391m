"""Synthetic unit tests for new-record scoring; these are not production evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset_b.data.split_inputs import InputQualityError
from src.dataset_b.reproduce import model_worker, predict_worker
from src.dataset_b.reproduce.evaluate import predict_model
from src.dataset_b.reproduce.manifest import ReproduceError
from src.dataset_b.reproduce.spec import MODELS
from tests.test_dataset_b_reproduce_evaluate import fake_python, model_manifest


REPO = Path(__file__).resolve().parents[1]
HEADER = "record_id,loan_type,loan_purpose,lien_status,reverse_mortgage,open-end_line_of_credit,business_or_commercial_purpose,income,debt_to_income_ratio,loan_amount,loan_to_value_ratio,loan_term,occupancy_type,preapproval,interest_only_payment,balloon_payment"
ROW_A = "a,1,2,2,2,1,2,88,20%-<30%,75000.0,42.2,360,1,2,2,2"
ROW_B = "b,1,4,2,2,1,2,286,48,85000.0,62.268,360,1,2,2,2"
SPEC = MODELS["logistic_regression"]


def input_file(root: Path, *rows: str, header: str = HEADER) -> Path:
    path = root / "input.csv"
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8")
    return path


def frozen_run(root: Path, run_id: str = SPEC.run_id) -> None:
    selection = root / SPEC.path(SPEC.threshold_file)
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps({"run_id": run_id, "selected_threshold": 0.5}), encoding="utf-8")


def test_real_quality_gate_accepts_valid_records_and_keeps_their_order(tmp_path: Path) -> None:
    gated = predict_worker.gate_input(predict_worker.read_input(input_file(tmp_path, ROW_A, ROW_B)))
    assert gated.record_ids.tolist() == ["a", "b"]
    assert len(gated.X) == 2


def test_blank_input_is_rejected_and_never_imputed(tmp_path: Path) -> None:
    blank = "a,1,2,2,2,1,2,,20%-<30%,75000.0,42.2,360,1,2,2,2"
    with pytest.raises(InputQualityError, match="rejected"):
        predict_worker.gate_input(predict_worker.read_input(input_file(tmp_path, blank)))


def test_unknown_category_is_rejected(tmp_path: Path) -> None:
    unknown = "a,99,2,2,2,1,2,88,20%-<30%,75000.0,42.2,360,1,2,2,2"
    with pytest.raises(InputQualityError):
        predict_worker.gate_input(predict_worker.read_input(input_file(tmp_path, unknown)))


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReproduceError, match="missing required columns"):
        predict_worker.read_input(input_file(tmp_path, "a,1", header="record_id,loan_type"))


def test_predict_writes_probability_and_frozen_threshold_decision(tmp_path: Path, monkeypatch) -> None:
    frozen_run(tmp_path)
    monkeypatch.setitem(model_worker.PREDICTORS, SPEC.key, lambda _checkpoint, _frame: np.asarray([0.25, 0.75]))
    output = tmp_path / "out" / "scores.csv"
    result = predict_worker.predict(SPEC.key, input_file(tmp_path, ROW_A, ROW_B), output, project_root=tmp_path)
    assert result["status"] == "PASS" and result["threshold"] == 0.5
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["record_id"], row["predicted_approved"]) for row in rows] == [("a", "0"), ("b", "1")]


def test_predict_rejects_another_run_identity(tmp_path: Path) -> None:
    frozen_run(tmp_path, run_id="other_run")
    with pytest.raises(ReproduceError, match="run identity"):
        predict_worker.predict(SPEC.key, input_file(tmp_path, ROW_A), tmp_path / "out.csv", project_root=tmp_path)


def test_predict_rejects_non_finite_probabilities(tmp_path: Path, monkeypatch) -> None:
    frozen_run(tmp_path)
    monkeypatch.setitem(model_worker.PREDICTORS, SPEC.key, lambda _checkpoint, _frame: np.asarray([np.nan]))
    with pytest.raises(ReproduceError, match="invalid"):
        predict_worker.predict(SPEC.key, input_file(tmp_path, ROW_A), tmp_path / "out.csv", project_root=tmp_path)


def test_dispatch_requires_input_assets_and_environment(tmp_path: Path) -> None:
    calls: list[str] = []
    worker = lambda *a: calls.append("x") or 0
    assert predict_model(tmp_path, tmp_path / "envs", SPEC.key, tmp_path / "absent.csv", tmp_path / "o.csv", worker=worker)["status"] == "FAIL"
    source = input_file(tmp_path, ROW_A)
    model_manifest(tmp_path, tamper=True)
    assert predict_model(tmp_path, tmp_path / "envs", SPEC.key, source, tmp_path / "o.csv", worker=worker)["error"] == "assets are not verified"
    model_manifest(tmp_path)
    assert "not installed" in predict_model(tmp_path, tmp_path / "envs", SPEC.key, source, tmp_path / "o.csv", worker=worker)["error"]
    assert calls == []
    fake_python(tmp_path / "envs", SPEC.profile)
    assert predict_model(tmp_path, tmp_path / "envs", SPEC.key, source, tmp_path / "o.csv", worker=worker)["status"] == "PASS"
    assert calls == ["x"]


def test_predict_source_contains_no_fit_call() -> None:
    source = (REPO / "src/dataset_b/reproduce/predict_worker.py").read_text(encoding="utf-8")
    for forbidden in (".fit(", "fit_transform(", "partial_fit(", 'map_location="cpu"'):
        assert forbidden not in source
