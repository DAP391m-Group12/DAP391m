from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq3_protocol.yaml"
CONTRACT_PATH = PROJECT_ROOT / "docs" / "handbook" / "contracts" / "rq3_fairness.md"
DATA_CONTRACT_PATH = PROJECT_ROOT / "docs" / "handbook" / "contracts" / "data_contract.md"
SPLIT_CONTRACT_PATH = PROJECT_ROOT / "docs" / "handbook" / "contracts" / "split_and_evaluation.md"
STATUS_PATH = PROJECT_ROOT / "docs" / "handbook" / "STATUS.md"


def _protocol() -> dict:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_completed_protocol_keeps_six_final_runs_without_model_execution() -> None:
    protocol = _protocol()
    assert protocol["status"] == "complete"
    assert list(protocol["eligible_final_runs"]) == [
        "lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini"
    ]
    assert protocol["audit_data"] == {
        "partition": "test",
        "test_rows": 43996,
        "test_view": "data/dataset_b/split_v1/test.csv",
        "raw_source": "data/raw/state_NY_2024.csv",
        "join_key": "record_id",
        "join_method": "one_based_source_row_number_from_record_id_suffix",
        "join_cardinality": "one_to_one_complete",
        "protected_attributes_are_model_inputs": False,
        "allow_retraining_or_reinference": False,
        "allow_group_specific_thresholds": False,
        "allow_fairness_mitigation": False,
    }
    assert protocol["target_orientation"] == {
        "positive_outcome": "Approved",
        "label": "y_approved",
        "probability": "p_approved",
        "predicted_class": "predicted_approved",
        "allow_denied_label_inversion": False,
    }


def test_group_reference_and_gap_direction_are_locked() -> None:
    attributes = _protocol()["attributes"]
    assert set(attributes) == {"applicant_sex", "derived_race"}
    sex = attributes["applicant_sex"]
    assert sex["reference_group"] == "Male"
    assert sex["gap_direction"] == "target_minus_reference"
    assert sex["primary_groups"] == {"Male": "1", "Female": "2"}
    assert sex["primary_comparisons"] == ["Female"]

    race = attributes["derived_race"]
    assert race["reference_group"] == "White"
    assert race["gap_direction"] == "target_minus_reference"
    assert race["primary_comparisons"] == ["Black or African American", "Asian"]
    assert race["secondary_comparisons"] == [
        "American Indian or Alaska Native", "2 or more minority races", "Joint"
    ]


def test_rq3_only_audit_view_keeps_race_out_of_model_inputs() -> None:
    protocol = _protocol()
    audit_view = protocol["audit_view"]
    assert audit_view["name"] == "rq3_only_audit_view"
    assert audit_view["controlled_model_view_unchanged"] is True
    assert audit_view["controlled_model_view_role_arithmetic_unchanged"] is True
    assert audit_view["derived_race"] == {
        "source": "immutable_raw_source",
        "allowed_in_rq3_audit_view": True,
        "allowed_in_controlled_model_view": False,
        "allowed_in_predictors": False,
    }
    assert audit_view["raw_applicant_race_slots"]["used_for_rq3_metrics"] is False
    assert audit_view["raw_applicant_race_slots"]["retained_for_traceability"] is True
    assert audit_view["raw_applicant_race_slots"]["fields"] == [
        "applicant_race-1",
        "applicant_race-2",
        "applicant_race-3",
        "applicant_race-4",
        "applicant_race-5",
    ]


def test_metrics_denominators_and_bootstrap_are_locked() -> None:
    protocol = _protocol()
    metrics = protocol["metrics"]
    assert metrics["group_metrics"] == [
        "selection_rate", "tpr_approved", "fpr_approved", "fnr_approved",
        "brier_score", "calibration_residual", "absolute_calibration_residual",
    ]
    assert metrics["calibration_residual"] == "mean_p_approved_minus_y_approved"
    assert metrics["pairwise_gap_metrics"] == [
        "selection_rate_gap_signed", "selection_rate_gap_absolute",
        "tpr_approved_gap_signed", "tpr_approved_gap_absolute",
        "fpr_approved_gap_signed", "fpr_approved_gap_absolute",
        "fnr_approved_gap_signed", "fnr_approved_gap_absolute",
        "calibration_residual_gap_signed", "calibration_residual_gap_absolute",
        "equalized_odds_gap",
    ]
    assert metrics["equalized_odds_gap"] == "max_absolute_tpr_fpr_gap"
    assert metrics["global_race_metrics"] == [
        "demographic_parity_difference_range", "tpr_approved_range",
        "fpr_approved_range", "equalized_odds_difference_range",
    ]
    assert metrics["global_race_excludes_coverage_only_groups"] is True
    assert protocol["denominators"] == {
        "minimum_group_rows": 100,
        "minimum_actual_approved": 30,
        "minimum_actual_denied": 30,
        "insufficient_result": "undefined",
        "undefined_storage": None,
        "never_replace_undefined_with_zero": True,
    }
    uncertainty = protocol["uncertainty"]
    assert uncertainty["replicates"] == 1000
    assert uncertainty["minimum_valid_replicates"] == 950
    assert uncertainty["paired_across_models_attributes_and_groups"] is True
    assert uncertainty["apply_denominator_rules_per_replicate"] is True
    assert uncertainty["may_regenerate_indices"] is False


def test_contract_is_minimal_and_completed_status_stays_bounded() -> None:
    protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8").lower()
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8").lower()
    status_text = STATUS_PATH.read_text(encoding="utf-8")
    for forbidden in ("sha256", "sha-256", "digest", "canonical hash"):
        assert forbidden not in protocol_text
        assert forbidden not in contract_text
    assert "RQ3 | COMPLETE" in status_text
    assert "rq3_fairness_20261010_01" in status_text
    assert "Results are descriptive model-audit evidence only" in status_text
    assert "Female minus Male" in CONTRACT_PATH.read_text(encoding="utf-8")
    assert "target group - reference group" in CONTRACT_PATH.read_text(encoding="utf-8")
    assert "does not invert labels" in CONTRACT_PATH.read_text(encoding="utf-8")


def test_authority_docs_record_the_derived_race_exception_without_scope_drift() -> None:
    data_contract = DATA_CONTRACT_PATH.read_text(encoding="utf-8")
    split_contract = SPLIT_CONTRACT_PATH.read_text(encoding="utf-8")
    assert "separate RQ3-only audit view" in data_contract
    assert "does not add `derived_race` to the" in data_contract
    assert "99-column role" in data_contract
    assert "approved race scope uses `derived_race`" in split_contract
    assert "does not place `derived_race` in the controlled model view" in split_contract
