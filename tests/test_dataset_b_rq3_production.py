from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load(
    (PROJECT_ROOT / "config/dataset_b/rq3_protocol.yaml").read_text(encoding="utf-8")
)
OUTPUT = PROJECT_ROOT / PROTOCOL["output"]["root"]


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(OUTPUT / "audit_metadata.csv"),
        pd.read_csv(OUTPUT / "group_metrics.csv"),
        pd.read_csv(OUTPUT / "gap_metrics.csv"),
        pd.read_csv(OUTPUT / "bootstrap_summary.csv"),
        pd.read_csv(OUTPUT / "runtime.csv"),
    )


def test_production_file_set_status_and_row_contract() -> None:
    assert {path.name for path in OUTPUT.iterdir() if path.is_file()} == set(
        PROTOCOL["output"]["files"]
    )
    status = json.loads((OUTPUT / "status.json").read_text(encoding="utf-8"))
    assert status == {
        "bootstrap_indices_generated": False,
        "model_fit_calls": 0,
        "model_status": {
            "lr": "PASS",
            "plain_mlp": "PASS",
            "realmlp": "PASS",
            "rf": "PASS",
            "tabmmini": "PASS",
            "xgboost": "PASS",
        },
        "prediction_calls": 0,
        "run_id": "rq3_fairness_20261010_01",
        "status": "PASS",
        "target_orientation": "Approved-positive",
    }
    audit, groups, gaps, bootstrap, runtime = _tables()
    assert len(audit) == 43996 and audit["record_id"].is_unique
    assert len(groups) == 588
    assert len(gaps) == 420
    assert len(bootstrap) == 1008 == len(groups) + len(gaps)
    assert len(runtime) == 6 and set(runtime["status"]) == {"PASS"}
    assert (runtime[["point_metric_seconds", "bootstrap_seconds", "total_seconds"]] > 0).all().all()
    assert not any("denied" in column.lower() for column in audit.columns)


def test_audit_group_counts_match_locked_test_view() -> None:
    audit, *_ = _tables()
    assert audit["applicant_sex_code"].astype(str).value_counts().to_dict() == {
        "1": 24799,
        "2": 15347,
        "3": 2384,
        "4": 1439,
        "6": 27,
    }
    assert audit["derived_race"].value_counts().to_dict() == {
        "White": 27791,
        "Race Not Available": 7589,
        "Asian": 3976,
        "Black or African American": 3456,
        "Joint": 765,
        "American Indian or Alaska Native": 189,
        "2 or more minority races": 112,
        "Native Hawaiian or Other Pacific Islander": 97,
        "Free Form Text Only": 21,
    }


def _direct_metrics(y: np.ndarray, p: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    approved = y == 1
    denied = y == 0
    residual = float(np.mean(p - y))
    return {
        "selection_rate": float(np.mean(predicted)),
        "tpr_approved": float(np.mean(predicted[approved])),
        "fpr_approved": float(np.mean(predicted[denied])),
        "fnr_approved": float(np.mean(1 - predicted[approved])),
        "brier_score": float(np.mean((p - y) ** 2)),
        "calibration_residual": residual,
        "absolute_calibration_residual": abs(residual),
    }


def test_all_defined_group_point_metrics_match_direct_formulas() -> None:
    audit, groups, *_ = _tables()
    for model_name, specification in PROTOCOL["eligible_final_runs"].items():
        predictions = pd.read_csv(PROJECT_ROOT / specification["predictions"])
        threshold_payload = json.loads(
            (PROJECT_ROOT / specification["threshold"]).read_text(encoding="utf-8")
        )
        threshold = float(threshold_payload["selected_threshold"])
        assert predictions["record_id"].astype(str).tolist() == audit["record_id"].astype(str).tolist()
        model_rows = groups[groups["model"] == model_name]
        for (attribute, group), rows in model_rows.groupby(["attribute", "group"], sort=False):
            column = "applicant_sex_group" if attribute == "applicant_sex" else "derived_race"
            mask = audit[column].astype(str).to_numpy() == str(group)
            y = predictions.loc[mask, "y_approved"].to_numpy(dtype=np.int8)
            p = predictions.loc[mask, "p_approved"].to_numpy(dtype=np.float64)
            predicted = (p >= threshold).astype(np.int8)
            assert (rows["group_rows"] == len(y)).all()
            assert (rows["actual_approved"] == int(np.sum(y == 1))).all()
            assert (rows["actual_denied"] == int(np.sum(y == 0))).all()
            if set(rows["group_role"]) == {"coverage_only"}:
                assert set(rows["status"]) == {"coverage_only"}
                assert rows["value"].isna().all()
                continue
            expected = _direct_metrics(y, p, predicted)
            assert set(rows["status"]) == {"defined"}
            for row in rows.itertuples(index=False):
                assert row.value == pytest.approx(expected[row.metric], abs=1e-12)


def test_pairwise_and_global_point_metrics_reconcile_to_group_rows() -> None:
    _, groups, gaps, *_ = _tables()
    lookup = {
        (row.model, row.attribute, row.group, row.metric): row.value
        for row in groups.itertuples(index=False)
        if row.status == "defined"
    }
    for row in gaps[gaps["result_type"] == "pairwise"].itertuples(index=False):
        if row.metric == "equalized_odds_gap":
            tpr = gaps[
                (gaps["model"] == row.model)
                & (gaps["attribute"] == row.attribute)
                & (gaps["target_group"] == row.target_group)
                & (gaps["metric"] == "tpr_approved_gap_signed")
            ]["value"].iloc[0]
            fpr = gaps[
                (gaps["model"] == row.model)
                & (gaps["attribute"] == row.attribute)
                & (gaps["target_group"] == row.target_group)
                & (gaps["metric"] == "fpr_approved_gap_signed")
            ]["value"].iloc[0]
            assert row.value == pytest.approx(max(abs(tpr), abs(fpr)), abs=1e-12)
            continue
        base, suffix = row.metric.rsplit("_gap_", 1)
        signed = lookup[(row.model, row.attribute, row.target_group, base)] - lookup[
            (row.model, row.attribute, row.reference_group, base)
        ]
        expected = signed if suffix == "signed" else abs(signed)
        assert row.value == pytest.approx(expected, abs=1e-12)

    coverage = set(
        groups.loc[groups["group_role"] == "coverage_only", "group"].astype(str)
    )
    for row in gaps[gaps["result_type"] == "global_range"].itertuples(index=False):
        used = set(str(row.groups_used).split("|"))
        assert used.isdisjoint(coverage)
        if row.metric == "equalized_odds_difference_range":
            components = gaps[
                (gaps["model"] == row.model)
                & (gaps["result_type"] == "global_range")
                & (gaps["metric"].isin(["tpr_approved_range", "fpr_approved_range"]))
            ]["value"]
            assert row.value == pytest.approx(components.max(), abs=1e-12)
        else:
            base = {
                "demographic_parity_difference_range": "selection_rate",
                "tpr_approved_range": "tpr_approved",
                "fpr_approved_range": "fpr_approved",
            }[row.metric]
            values = [lookup[(row.model, "derived_race", group, base)] for group in used]
            assert row.value == pytest.approx(max(values) - min(values), abs=1e-12)


def test_bootstrap_summary_has_complete_validity_payload() -> None:
    _, groups, gaps, bootstrap, _ = _tables()
    assert set(bootstrap["replicate_count"]) == {1000}
    assert ((bootstrap["valid_replicates"] + bootstrap["invalid_replicates"]) == 1000).all()
    point = pd.concat([groups, gaps], ignore_index=True)
    keys = ["result_type", "model", "attribute", "group", "target_group", "reference_group", "metric"]
    merged = bootstrap.merge(
        point[keys + ["value", "status"]], on=keys, how="left", validate="one_to_one"
    )
    np.testing.assert_allclose(
        merged["point_estimate"].to_numpy(dtype=float),
        merged["value"].to_numpy(dtype=float),
        equal_nan=True,
    )
    coverage = merged[merged["status"] == "coverage_only"]
    assert (coverage["valid_replicates"] == 0).all()
    assert coverage["ci_lower"].isna().all() and coverage["ci_upper"].isna().all()
    defined = merged[merged["ci_status"] == "defined"]
    assert (defined["valid_replicates"] >= 950).all()
    assert np.isfinite(defined[["ci_lower", "ci_upper"]]).all().all()
    assert (defined["ci_lower"] <= defined["ci_upper"]).all()
