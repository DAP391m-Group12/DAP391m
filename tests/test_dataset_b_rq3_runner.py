from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_b.fairness.audit_data import ModelAuditInput, RQ3PreflightResult
from src.dataset_b.fairness.runner import run_from_preflight, validate_output_directory


def _toy_protocol() -> dict:
    return {
        "audit_data": {"test_rows": 8},
        "eligible_final_runs": {"toy": {"run_id": "toy_run"}},
        "attributes": {
            "applicant_sex": {
                "reference_group": "Male",
                "primary_comparisons": ["Female"],
                "coverage_only_groups": {"Information not provided": "3"},
            },
            "derived_race": {
                "reference_group": "White",
                "primary_comparisons": ["Black or African American", "Asian"],
                "secondary_comparisons": [],
                "coverage_only_groups": ["Race Not Available"],
            },
        },
        "denominators": {
            "minimum_group_rows": 2,
            "minimum_actual_approved": 1,
            "minimum_actual_denied": 1,
        },
        "uncertainty": {
            "minimum_valid_replicates": 2,
            "confidence_level": 0.95,
        },
        "output": {
            "run_id": "toy_rq3",
            "files": [
                "audit_metadata.csv",
                "group_metrics.csv",
                "gap_metrics.csv",
                "bootstrap_summary.csv",
                "runtime.csv",
                "status.json",
            ],
        },
    }


def _toy_preflight() -> RQ3PreflightResult:
    audit = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(8)],
            "y_approved": [1, 0, 1, 0, 1, 0, 1, 0],
            "applicant_sex_code": ["1", "1", "2", "2", "1", "1", "3", "3"],
            "applicant_sex_group": [
                "Male", "Male", "Female", "Female", "Male", "Male",
                "Information not provided", "Information not provided",
            ],
            "derived_race": [
                "White", "White", "Black or African American", "Black or African American",
                "Asian", "Asian", "Race Not Available", "Race Not Available",
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "record_id": audit["record_id"],
            "y_approved": audit["y_approved"],
            "p_approved": [0.9, 0.2, 0.4, 0.7, 0.8, 0.3, 0.6, 0.1],
            "predicted_approved": [1, 0, 0, 1, 1, 0, 1, 0],
        }
    )
    model = ModelAuditInput("toy", "toy_run", 0.5, predictions)
    indices = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
        ],
        dtype=np.int32,
    )
    return RQ3PreflightResult(audit, {"toy": model}, indices)


def test_toy_runner_writes_and_validates_minimal_artifacts(tmp_path: Path) -> None:
    protocol = _toy_protocol()
    output = tmp_path / "rq3_toy"
    validation = run_from_preflight(
        _toy_preflight(), protocol, output, progress=None
    )
    assert validation["status"] == "PASS"
    assert validation["model_count"] == 1
    assert validate_output_directory(output, protocol, expected_rows=8)["status"] == "PASS"

    groups = pd.read_csv(output / "group_metrics.csv")
    gaps = pd.read_csv(output / "gap_metrics.csv")
    bootstrap = pd.read_csv(output / "bootstrap_summary.csv")
    assert set(groups.loc[groups["group_role"] == "coverage_only", "status"]) == {"coverage_only"}
    assert groups.loc[groups["group_role"] == "coverage_only", "value"].isna().all()
    assert "Race Not Available" not in "|".join(gaps["groups_used"].dropna().astype(str))
    assert len(bootstrap) == len(groups) + len(gaps)
    assert set(bootstrap["replicate_count"]) == {3}

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["model_fit_calls"] == 0
    assert status["prediction_calls"] == 0
    assert status["bootstrap_indices_generated"] is False


def test_runner_source_has_no_model_or_random_execution() -> None:
    source = Path("src/dataset_b/fairness/runner.py").read_text(encoding="utf-8")
    for forbidden in ("predict_proba(", ".predict(", ".fit(", "np.random", "default_rng", "sha256"):
        assert forbidden not in source
