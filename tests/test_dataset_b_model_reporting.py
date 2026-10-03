"""Tests for the stable Dataset B model-reporting contract."""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.reporting import model_reporting as reporting

from src.dataset_b.reporting import (
    FINAL_MODEL_KEYS,
    ReportingContractError,
    cv_fold_frame,
    plot_calibration_curve,
    plot_threshold_balanced_accuracy,
    validate_cross_model_summaries,
    validate_model_summary,
    validate_reporting_directory,
)


def _summary(model_key="lr", *, status="PARTIAL", test_rows=43996, predictions="test_predictions.csv", bootstrap="bootstrap_indices.npz"):
    complete = status == "COMPLETE"
    return {
        "schema_version": 1,
        "reporting_status": status,
        "model_key": model_key,
        "run_id": f"{model_key}_final_15f_test",
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "data_boundaries": {
            "train_rows": 205310,
            "validation_rows": 43995 if complete else None,
            "test_rows": test_rows if complete else None,
        },
        "validation": {"threshold": 0.5} if complete else None,
        "test": {
            "ordered_record_ids_verified": True,
            "bootstrap_indices": str(bootstrap),
        } if complete else None,
        "artifact_sources": {"test_predictions": str(predictions)} if complete else {},
    }


def test_partial_summary_is_valid_but_not_final_comparison_ready():
    summary = _summary()
    validate_model_summary(summary, require_complete=False)
    with pytest.raises(ReportingContractError, match="COMPLETE"):
        validate_model_summary(summary, require_complete=True)


def test_cross_model_gate_requires_six_complete_aligned_models(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(reporting, "EXPECTED_BOOTSTRAP_REPLICATES", 3)
    ids = ["r1", "r2", "r3", "r4"]
    indices = np.tile(np.arange(4, dtype=np.int32), (3, 1))
    bootstrap = tmp_path / "bootstrap_indices.npz"; np.savez_compressed(bootstrap, indices=indices)
    summaries = []
    for model in FINAL_MODEL_KEYS:
        predictions = tmp_path / f"{model}_predictions.csv"
        pd.DataFrame({"record_id": ids}).to_csv(predictions, index=False)
        summaries.append(_summary(model, status="COMPLETE", test_rows=4, predictions=predictions, bootstrap=bootstrap))
    validate_cross_model_summaries(summaries)

    mismatched = deepcopy(summaries)
    different = tmp_path / "different_indices.npz"
    changed = indices.copy(); changed[0, 0] = 1; np.savez_compressed(different, indices=changed)
    mismatched[-1]["test"]["bootstrap_indices"] = str(different)
    with pytest.raises(ReportingContractError, match="Bootstrap index lineage"):
        validate_cross_model_summaries(mismatched)

    mismatched_ids = deepcopy(summaries)
    changed_predictions = tmp_path / "changed_predictions.csv"
    pd.DataFrame({"record_id": ["r1", "r2", "r4", "r3"]}).to_csv(changed_predictions, index=False)
    mismatched_ids[-1]["artifact_sources"]["test_predictions"] = str(changed_predictions)
    with pytest.raises(ReportingContractError, match="Test membership/order lineage"):
        validate_cross_model_summaries(mismatched_ids)


def test_cross_model_gate_rejects_legacy_feature_version(tmp_path):
    summaries = [_summary(model, status="COMPLETE", predictions=tmp_path / "missing.csv", bootstrap=tmp_path / "missing.npz") for model in FINAL_MODEL_KEYS]
    summaries[0]["feature_version"] = "temporary_training_lock_v0_2"
    with pytest.raises(ReportingContractError, match="final-feature"):
        validate_cross_model_summaries(summaries)


def test_cv_fold_frame_returns_stable_tidy_schema():
    frame = cv_fold_frame(
        {
            "folds": [
                {"fold": 2, "train_rows": 8, "fold_validation_rows": 2, "roc_auc": 0.82, "fit_seconds": 1.2},
                {"fold": 1, "train_rows": 8, "fold_validation_rows": 2, "roc_auc": 0.81, "fit_seconds": 1.0},
            ]
        },
        model_key="lr",
    )
    assert frame.columns.tolist() == [
        "model_key",
        "fold",
        "train_rows",
        "fold_validation_rows",
        "roc_auc",
        "fit_seconds",
        "convergence_warning_count",
    ]
    assert frame["fold"].tolist() == [1, 2]
    assert frame["roc_auc"].dtype == pd.Series([0.1]).dtype


def test_threshold_plot_requires_exact_selected_candidate():
    frame = pd.DataFrame(
        {"threshold": [0.4, 0.5, 0.6], "balanced_accuracy": [0.7, 0.8, 0.75]}
    )
    figure, _axis = plot_threshold_balanced_accuracy(
        frame, model_key="lr", selected_threshold=0.5
    )
    assert figure is not None

    with pytest.raises(ReportingContractError, match="absent or duplicated"):
        plot_threshold_balanced_accuracy(frame, model_key="lr", selected_threshold=0.55)


def test_calibration_plot_uses_requested_model_label():
    rows = [
        {
            "mean_p_approved": 0.25,
            "observed_approval_rate": 0.3,
            "count": 10,
        },
        {
            "mean_p_approved": 0.75,
            "observed_approval_rate": 0.7,
            "count": 12,
        },
    ]

    figure, axis = plot_calibration_curve(rows, model_key="rf")

    assert figure is not None
    assert "RF Test bins" in axis.get_legend_handles_labels()[1]


def test_complete_reporting_directory_contract(tmp_path):
    summary = _summary(status="COMPLETE")
    (tmp_path / "model.joblib").write_bytes(b"model")
    (tmp_path / "model_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "model_key": "lr",
                "run_id": summary["run_id"],
                "feature_version": "feature_set_v1",
                "split_version": "split_v1",
                "fold": fold,
                "roc_auc": 0.8,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "warning_count": 0,
            }
            for fold in range(1, 6)
        ]
    ).to_csv(tmp_path / "cv_fold_metrics.csv", index=False)
    metric_names = [
        "roc_auc_approved",
        "average_precision_denied",
        "precision_denied",
        "recall_denied",
        "f1_denied",
        "recall_approved",
        "balanced_accuracy",
        "brier_score_approved",
        "calibration_intercept",
        "calibration_slope",
    ]
    pd.DataFrame(
        [
            {
                "model_key": "lr",
                "run_id": summary["run_id"],
                "feature_version": "feature_set_v1",
                "split_version": "split_v1",
                "partition": "test",
                "metric": metric,
                "estimate": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "unit": "rate",
                "higher_is_better": metric != "brier_score_approved",
                "valid_replicates": 1000,
            }
            for metric in metric_names
        ]
    ).to_csv(tmp_path / "rq1_metrics_long.csv", index=False)
    pd.DataFrame(
        [
            {"model_key": "lr", "run_id": summary["run_id"], "actual_class": actual, "predicted_class": predicted, "count": 1}
            for actual in ("approved", "denied")
            for predicted in ("approved", "denied")
        ]
    ).to_csv(tmp_path / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_key": "lr",
                "run_id": summary["run_id"],
                "bin": 1,
                "lower_edge": 0.0,
                "upper_edge": 0.1,
                "count": 100,
                "mean_p_approved": 0.05,
                "observed_approval_rate": 0.06,
            }
        ]
    ).to_csv(tmp_path / "calibration_curve.csv", index=False)
    (tmp_path / "artifact_index.json").write_text(
        json.dumps({"schema_version": 1, "model_key": "lr", "run_id": summary["run_id"], "artifacts": [{"path": "model.joblib"}]}),
        encoding="utf-8",
    )

    loaded = validate_reporting_directory(tmp_path)
    assert loaded["run_id"] == summary["run_id"]


def test_reporting_directory_rejects_missing_indexed_artifact(tmp_path):
    test_complete_reporting_directory_contract(tmp_path)
    index_path = tmp_path / "artifact_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"][0]["path"] = "missing.joblib"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ReportingContractError, match="Indexed artifact is missing"):
        validate_reporting_directory(tmp_path)


def _xgboost_summary(run_dir: Path):
    stored_names = (
        "roc_auc_approved",
        "average_precision_denied",
        "precision_denied",
        "recall_denied",
        "f1_denied",
        "recall_approved",
        "balanced_accuracy",
        "brier_approved",
        "calibration_intercept",
        "calibration_slope",
    )
    intervals = {
        name: {"lower_2_5": 0.4, "upper_97_5": 0.6, "valid_replicates": 1000, "invalid_replicates": 0}
        for name in stored_names
    }
    return {
        "schema_version": 1,
        "contract_id": "XGB-10",
        "status": "PASS",
        "reporting_status": "COMPLETE",
        "model_key": "xgb",
        "run_id": run_dir.name,
        "feature_version": "feature_set_v1",
        "split_version": "split_v1",
        "preprocessing_version": "dataset_b_preprocessing_v2",
        "data_boundaries": {"train_rows": 205310, "validation_rows": 43995, "test_rows": 43996},
        "validation": {"threshold": 0.5},
        "test": {
            "ordered_record_ids_verified": True,
            "bootstrap_indices": "bootstrap_indices.npz",
            "threshold": 0.5,
            "point_metrics": {name: 0.5 for name in stored_names},
            "confidence_intervals_95_percentile": intervals,
            "counts": {"tp_approved": 10, "fn_approved": 2, "tn_denied": 8, "fp_denied": 3},
            "calibration_curve": [
                {
                    "bin": 0,
                    "lower_edge": 0.0,
                    "upper_edge": 1.0,
                    "count": 23,
                    "mean_p_approved": 0.5,
                    "observed_approval_rate": 12 / 23,
                }
            ],
        },
        "artifact_sources": {
            "cv_results": str(run_dir / "cv_results.json"),
            "test_predictions": str(run_dir / "test/test_predictions.csv"),
        },
        "execution": {
            "fit_calls": 0,
            "predict_proba_calls": 0,
            "bootstrap_regenerated": False,
            "bootstrap_recomputation_verified": True,
        },
    }


def test_xgboost_adapter_maps_keys_without_mutating_source(tmp_path):
    run_dir = tmp_path / "xgb_final"
    run_dir.mkdir()
    source = _xgboost_summary(run_dir)
    adapted = reporting.adapt_xgboost_reporting_summary(source)

    assert source["model_key"] == "xgb"
    assert "brier_approved" in source["test"]["point_metrics"]
    assert adapted["model_key"] == "xgboost"
    assert adapted["protocol_model_key"] == "xgb"
    assert "brier_score_approved" in adapted["test"]["point_metrics"]
    assert "brier_approved" not in adapted["test"]["point_metrics"]


def test_synthetic_xgboost_reporting_package_passes_contract(tmp_path):
    run_dir = tmp_path / "xgb_final"
    (run_dir / "test").mkdir(parents=True)
    (run_dir / "test/test_predictions.csv").write_text("record_id\nr1\n", encoding="utf-8")
    cv = {
        "run_id": run_dir.name,
        "folds": [
            {
                "fold": fold,
                "roc_auc": 0.8,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "warning_count": 0,
            }
            for fold in range(1, 6)
        ],
    }
    (run_dir / "cv_results.json").write_text(json.dumps(cv), encoding="utf-8")

    loaded = reporting.write_xgboost_reporting_package(_xgboost_summary(run_dir), run_dir=run_dir)
    reporting_dir = run_dir / "reporting"

    assert set(path.name for path in reporting_dir.iterdir()) == set(reporting.EXPECTED_REPORTING_FILES)
    assert loaded["model_key"] == "xgboost"
    assert loaded["protocol_model_key"] == "xgb"
    assert set(pd.read_csv(reporting_dir / "rq1_metrics_long.csv")["metric"]) == reporting.REQUIRED_RQ1_METRICS
    for filename in reporting.REPORTING_CSV_COLUMNS:
        assert set(pd.read_csv(reporting_dir / filename)["model_key"]) == {"xgboost"}


def test_xgboost_reporting_does_not_change_lr_or_rf_model_keys(tmp_path):
    for model_key in ("lr", "rf"):
        summary = _summary(model_key=model_key)
        validate_model_summary(summary, require_complete=False)
        assert summary["model_key"] == model_key


def test_xgboost_notebook_has_complete_non_executing_stage_map():
    notebook_path = Path(__file__).parents[1] / "notebooks/dataset_b/03_xgboost_final.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    stage_cells = [cell for cell in notebook["cells"] if "xgb-stage-map" in cell.get("metadata", {}).get("tags", [])]
    assert len(stage_cells) == 1

    stage_map = "".join(stage_cells[0]["source"])
    expected_stages = {
        "XGB-07 — training",
        "XGB-08 — Validation threshold",
        "XGB-09 — Test evaluation",
        "XGB-10 — final review",
        "XGB-11 — reporting",
        "XGB-12 — publication",
    }
    assert all(stage in stage_map for stage in expected_stages)

    code_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "RUN_ID = 'SET_XGBOOST_FINAL_RUN_ID'" in code_source
    assert all(call not in code_source for call in (".fit(", ".predict(", ".predict_proba("))
    assert code_source.count("publish_xgboost_final(") == 1
    assert "artifacts/dataset_b/xgboost/legacy" not in code_source


def _publication_function():
    notebook_path = Path(__file__).parents[1] / "notebooks/dataset_b/03_xgboost_final.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if "xgb-publication" in cell.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1
    namespace = {"__name__": "xgb_publication_test"}
    exec("".join(cells[0]["source"]), namespace)
    return namespace["publish_xgboost_final"], notebook


def _complete_publication_validators(run_id, calls):
    def bundle_validator(candidate_run_id, *, run_dir):
        calls.append(("bundle", Path(run_dir)))
        return {"status": "PASS", "run_id": candidate_run_id}

    def reporting_validator(reporting_dir):
        calls.append(("reporting", Path(reporting_dir)))
        return {"reporting_status": "COMPLETE", "run_id": run_id, "model_key": "xgboost"}

    return bundle_validator, reporting_validator


def test_xgboost_publication_refuses_incomplete_working_bundle(tmp_path):
    publish, _notebook = _publication_function()
    run_id = "xgb_run"
    working_root = tmp_path / "working"
    final_root = tmp_path / "final"
    (working_root / run_id / "reporting").mkdir(parents=True)
    final_root.mkdir()

    def incomplete(*_args, **_kwargs):
        raise ReportingContractError("incomplete working bundle")

    with pytest.raises(ReportingContractError, match="incomplete"):
        publish(
            run_id,
            working_root=working_root,
            final_root=final_root,
            bundle_validator=incomplete,
            reporting_validator=lambda _path: pytest.fail("reporting validation must not run"),
        )
    assert (working_root / run_id).is_dir()
    assert not (final_root / run_id).exists()


def test_xgboost_publication_refuses_existing_final_destination(tmp_path):
    publish, _notebook = _publication_function()
    run_id = "xgb_run"
    working_root = tmp_path / "working"
    final_root = tmp_path / "final"
    (working_root / run_id).mkdir(parents=True)
    (final_root / run_id).mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        publish(
            run_id,
            working_root=working_root,
            final_root=final_root,
            bundle_validator=lambda *_args, **_kwargs: pytest.fail("validator must not run"),
            reporting_validator=lambda _path: pytest.fail("validator must not run"),
        )
    assert (working_root / run_id).is_dir()
    assert (final_root / run_id).is_dir()


def test_xgboost_publication_is_atomic_and_revalidates_final_reporting(tmp_path, monkeypatch):
    publish, _notebook = _publication_function()
    run_id = "xgb_run"
    working_root = tmp_path / "working"
    final_root = tmp_path / "final"
    (working_root / run_id / "reporting").mkdir(parents=True)
    (working_root / run_id / "reporting/model_summary.json").write_text("{}", encoding="utf-8")
    final_root.mkdir()
    calls = []
    bundle_validator, reporting_validator = _complete_publication_validators(run_id, calls)
    replace_calls = []
    real_replace = os.replace

    def tracked_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracked_replace)
    destination = publish(
        run_id,
        working_root=working_root,
        final_root=final_root,
        bundle_validator=bundle_validator,
        reporting_validator=reporting_validator,
    )

    assert replace_calls == [(working_root / run_id, final_root / run_id)]
    assert destination == final_root / run_id
    assert not (working_root / run_id).exists()
    assert (destination / "reporting/model_summary.json").is_file()
    assert calls == [
        ("bundle", working_root / run_id),
        ("reporting", working_root / run_id / "reporting"),
        ("reporting", final_root / run_id / "reporting"),
    ]


def test_xgboost_publication_notebook_does_not_update_notebook_07():
    _publish, notebook = _publication_function()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "07_model_total_review" not in source
    assert "MODEL_SUMMARY_PATHS" not in source
