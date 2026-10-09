from __future__ import annotations

from pathlib import Path

import yaml

from src.dataset_b.gsv_sl.predictors import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq2_protocol.yaml"
EXPLANATION_PATH = PROJECT_ROOT / "docs" / "handbook" / "contracts" / "rq2_explanation.md"
STATUS_PATH = PROJECT_ROOT / "docs" / "handbook" / "STATUS.md"


def _protocol() -> dict:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_protocol_preserves_six_frozen_runs_and_ordered_players() -> None:
    protocol = _protocol()
    assert protocol["status"] == "production_complete"
    assert protocol["eligible_final_runs"] == {
        "lr": "lr_final_15f_20261002_01",
        "rf": "rf_final_15f_20261002_01",
        "xgboost": "xgb_final_15f_20261003_01",
        "plain_mlp": "plain_mlp_final_15f_20261004_03",
        "realmlp": "realmlp_final_15f_20261004_01",
        "tabmmini": "tabmmini_final_15f_20261004_01",
    }
    players = protocol["scope"]["ordered_players"]
    assert len(players) == 12
    assert [field for player in players for field in player["raw_fields"]] == list(FEATURE_COLUMNS)
    assert protocol["scope"]["output_scale"] == "probability_approved"
    assert protocol["scope"]["explanation_partition"] == "validation"
    assert protocol["scope"]["test_partition_allowed_for_explanation"] is False
    assert protocol["execution"] == {
        "device": "cpu",
        "allow_cuda_fallback": False,
        "smoke_required_for_all_models": True,
        "coalition_batch_size": 16,
    }


def test_sampling_bootstrap_and_agreement_parameters_are_unchanged() -> None:
    protocol = _protocol()
    assert protocol["sampling"] == {
        "strata": [0, 1],
        "explanation": {"partition": "validation", "n_rows": 256, "seed": 4201},
        "background": {
            "partition": "train",
            "pool_rows": 128,
            "seed": 4202,
            "assignment_seed": 4203,
        },
    }
    assert protocol["uncertainty"]["bootstrap_replicates"] == 1000
    assert protocol["uncertainty"]["bootstrap_seed"] == 4204
    assert protocol["uncertainty"]["undefined_metrics"]["minimum_valid_replicates"] == 950
    assert protocol["uncertainty"]["shared_indices_across_methods"] is True
    assert protocol["uncertainty"]["shared_indices_across_models"] is True
    assert protocol["uncertainty"]["may_retrain_or_reinfer"] is False
    assert protocol["uncertainty"]["checkpoint"] == {
        "enabled": True,
        "chunk_size": 100,
        "chunk_count": 10,
        "resume_existing_run": True,
        "safe_file_replace": True,
        "retain_completed_chunks": True,
    }
    assert protocol["toy_fixtures"]["production_bootstrap_gate"] == {
        "expected_shape": [1000, 256],
        "one_artifact_for_all_six_models": True,
    }
    assert protocol["ranking_and_agreement"]["top_k"] == 3
    assert protocol["ranking_and_agreement"]["top_k_ties"] == "include_all_at_cutoff"


def test_contract_contains_no_sha_integrity_or_atomic_publication() -> None:
    yaml_text = PROTOCOL_PATH.read_text(encoding="utf-8").lower()
    handbook_text = EXPLANATION_PATH.read_text(encoding="utf-8").lower()
    forbidden = ("sha256", "sha-256", "integrity manifest", "artifact_integrity", ".working-", ".failed-")
    for value in forbidden:
        assert value not in yaml_text
        assert value not in handbook_text
    assert "schema_version" not in yaml_text
    assert "required_fields" not in yaml_text


def test_minimal_output_contract_and_smoke_gate_are_recorded() -> None:
    protocol = _protocol()
    assert protocol["output"]["root_template"] == "artifacts/dataset_b/rq2/{run_id}"
    assert protocol["smoke"] == {"rows": 8, "atol": 1.0e-6}
    text = EXPLANATION_PATH.read_text(encoding="utf-8")
    for required in (
        "sample.csv",
        "donors.csv",
        "bootstrap_indices.npy",
        "runtime.csv",
        "cross_model_stability.csv",
        "cube.npy",
        "gsv_local.npy",
        "importance.csv",
        "bootstrap.csv",
        "status.json",
        "bootstrap_checkpoints",
        "frozen stored Validation predictions",
    ):
        assert required in text


def test_verified_production_closure_is_recorded_consistently() -> None:
    protocol = _protocol()
    evidence = protocol["production_evidence"]
    assert evidence == {
        "run_id": "rq2_cpu_production_20261009_01",
        "output_root": "artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01",
        "model_order": ["lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini"],
        "model_status_pass": 6,
        "model_status_required": 6,
        "checkpoint_chunks_per_model": 10,
        "importance_rows_per_model": 24,
        "bootstrap_metric_rows_per_model": 4,
        "minimum_valid_replicates_observed": 1000,
        "cross_model_stability_rows": 60,
        "test_partition_used": False,
        "benchmark_imported": False,
        "frozen_artifacts_rewritten": False,
        "validation_checks_passed": 91,
        "validation_checks_failed": 0,
        "runtime_seconds": {
            "lr": 782.9453709,
            "rf": 865.1570737999999,
            "xgboost": 748.4194796,
            "plain_mlp": 713.7663306000001,
            "realmlp": 908.9360968000001,
            "tabmmini": 798.1936332,
            "total": 4817.4179849,
        },
        "notebook": "notebooks/dataset_b/08_rq2_gsv_sl_results.ipynb",
        "notebook_verified": True,
    }

    explanation = EXPLANATION_PATH.read_text(encoding="utf-8")
    status = STATUS_PATH.read_text(encoding="utf-8")
    for text in (explanation, status):
        assert "rq2_cpu_production_20261009_01" in text
        assert "08_rq2_gsv_sl_results.ipynb" in text
        assert "4817.4179849" in text
    assert "RQ2 | COMPLETE" in status
    assert "does not establish explanation correctness or ground truth" in status
