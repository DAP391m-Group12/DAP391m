"""Synthetic reporting and publication-boundary tests for TabMmini."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.evaluation.tabmmini_final_evaluation import (
    TabMMiniFinalEvaluationError,
)
from src.dataset_b.reporting import model_reporting
from src.dataset_b.reporting import tabmmini_reporting as reporting
from src.dataset_b.review import tabmmini_final_review as final_review
from src.dataset_b.review import tabmmini_publication as publication
from src.dataset_b.review.tabmmini_publication import publish_tabmmini_final


RUN_ID = "tabmmini_reporting_run"


def test_reporting_rejects_inconsistent_run_id_before_publication(tmp_path, monkeypatch):
    run_root = tmp_path / "working" / RUN_ID
    (run_root / "full_train").mkdir(parents=True)
    (run_root / "cv").mkdir()
    (run_root / "full_train/full_train_result.json").write_text(
        json.dumps({"run_id": RUN_ID, "train_rows": 4}), encoding="utf-8"
    )
    (run_root / "cv/cv_result.json").write_text(
        json.dumps(
            {
                "run_id": "different_run",
                "folds": [
                    {"fold": fold, "roc_auc": 0.8, "fit_seconds": 1.0}
                    for fold in range(1, 6)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reporting,
        "validate_tabmmini_prediction_stage",
        lambda *_args, **_kwargs: {"run_id": RUN_ID, "stage": "validation", "rows": 4},
    )
    monkeypatch.setattr(
        reporting,
        "validate_tabmmini_threshold",
        lambda *_args, **_kwargs: {"run_id": RUN_ID, "selected_threshold": 0.5},
    )
    monkeypatch.setattr(
        reporting,
        "validate_tabmmini_test_evaluation",
        lambda *_args, **_kwargs: {"run_id": RUN_ID, "rows": 4},
    )
    with pytest.raises(reporting.TabMMiniReportingError, match="run_id"):
        reporting.write_tabmmini_reporting(RUN_ID, artifact_root=tmp_path)
    assert not (run_root / "reporting").exists()


def test_publication_failure_paths_preserve_working_and_create_no_frozen(
    tmp_path, monkeypatch
):
    with pytest.raises(TabMMiniFinalEvaluationError, match="missing"):
        publish_tabmmini_final("missing_run", artifact_root=tmp_path)

    working = tmp_path / "working" / RUN_ID
    working.mkdir(parents=True)

    def failed_review(*_args, **_kwargs):
        return {"status": "FAIL", "run_id": RUN_ID}

    with pytest.raises(TabMMiniFinalEvaluationError, match="did not pass"):
        publish_tabmmini_final(
            RUN_ID,
            artifact_root=tmp_path,
            reviewer=failed_review,
            reporting_validator=lambda _path: pytest.fail("must not validate reporting"),
        )
    assert working.is_dir()
    assert not (working / "FROZEN").exists()
    assert not (tmp_path / "final" / RUN_ID).exists()

    def passed_review(*_args, **_kwargs):
        return {"status": "PASS", "run_id": RUN_ID}

    with pytest.raises(RuntimeError, match="reporting failure"):
        publish_tabmmini_final(
            RUN_ID,
            artifact_root=tmp_path,
            reviewer=passed_review,
            reporting_validator=lambda _path: (_ for _ in ()).throw(
                RuntimeError("reporting failure")
            ),
        )
    assert working.is_dir()
    assert not (working / "FROZEN").exists()

    monkeypatch.setattr(
        publication.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("atomic rename failure")),
    )
    with pytest.raises(OSError, match="atomic rename failure"):
        publish_tabmmini_final(
            RUN_ID,
            artifact_root=tmp_path,
            reviewer=passed_review,
            reporting_validator=lambda _path: {
                "run_id": RUN_ID,
                "reporting_status": "COMPLETE",
            },
        )
    assert working.is_dir()
    assert not (working / "FROZEN").exists()
    assert not (tmp_path / "final" / RUN_ID).exists()

    (tmp_path / "final" / RUN_ID).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_tabmmini_final(
            RUN_ID,
            artifact_root=tmp_path,
            reviewer=lambda *_args, **_kwargs: pytest.fail("must not review"),
        )
    assert not (working / "FROZEN").exists()


def test_publication_validates_final_before_creating_frozen(tmp_path):
    working = tmp_path / "working" / RUN_ID
    (working / "reporting").mkdir(parents=True)
    final = tmp_path / "final" / RUN_ID
    events = []

    def reviewer(run_id, *, artifact_root, published=False):
        assert run_id == RUN_ID
        assert Path(artifact_root) == tmp_path
        root = final if published else working
        assert root.is_dir()
        assert not (root / "FROZEN").exists()
        events.append(("review", "final" if published else "working"))
        return {"status": "PASS", "run_id": RUN_ID}

    def reporting_validator(path):
        path = Path(path)
        root = path.parent
        assert path.is_dir()
        assert not (root / "FROZEN").exists()
        events.append(("reporting", "final" if root == final else "working"))
        return {"run_id": RUN_ID, "reporting_status": "COMPLETE"}

    destination = publish_tabmmini_final(
        RUN_ID,
        artifact_root=tmp_path,
        reviewer=reviewer,
        reporting_validator=reporting_validator,
    )

    assert destination == final
    assert events == [
        ("review", "working"),
        ("reporting", "working"),
        ("review", "final"),
        ("reporting", "final"),
    ]
    assert (final / "FROZEN").read_text(encoding="utf-8") == (
        "TABMMINI FINAL FROZEN\n"
    )
    assert not working.exists()


def test_post_move_failures_leave_final_unfrozen(tmp_path):
    review_root = tmp_path / "review_failure"
    review_working = review_root / "working" / RUN_ID
    (review_working / "reporting").mkdir(parents=True)
    review_final = review_root / "final" / RUN_ID

    def failed_published_review(run_id, *, artifact_root, published=False):
        return {"status": "FAIL" if published else "PASS", "run_id": run_id}

    with pytest.raises(TabMMiniFinalEvaluationError, match="Published TabMmini direct review"):
        publish_tabmmini_final(
            RUN_ID,
            artifact_root=review_root,
            reviewer=failed_published_review,
            reporting_validator=lambda _path: {
                "run_id": RUN_ID,
                "reporting_status": "COMPLETE",
            },
        )
    assert review_final.is_dir()
    assert not (review_final / "FROZEN").exists()

    reporting_root = tmp_path / "reporting_failure"
    reporting_working = reporting_root / "working" / RUN_ID
    (reporting_working / "reporting").mkdir(parents=True)
    reporting_final = reporting_root / "final" / RUN_ID

    def reporting_fails_after_rename(path):
        if Path(path).parent == reporting_final:
            raise RuntimeError("post-move reporting failure")
        return {"run_id": RUN_ID, "reporting_status": "COMPLETE"}

    with pytest.raises(RuntimeError, match="post-move reporting failure"):
        publish_tabmmini_final(
            RUN_ID,
            artifact_root=reporting_root,
            reviewer=lambda run_id, **_kwargs: {
                "status": "PASS",
                "run_id": run_id,
            },
            reporting_validator=reporting_fails_after_rename,
        )
    assert reporting_final.is_dir()
    assert not (reporting_final / "FROZEN").exists()


def test_reporting_writer_emits_the_shared_complete_schema(tmp_path, monkeypatch):
    run_root = tmp_path / "working" / RUN_ID
    (run_root / "full_train/model").mkdir(parents=True)
    (run_root / "cv").mkdir()
    (run_root / "test_evaluation").mkdir()
    (run_root / "full_train/full_train_result.json").write_text(
        json.dumps({"run_id": RUN_ID, "train_rows": 4, "training_seconds": 2.0}),
        encoding="utf-8",
    )
    (run_root / "full_train/model/model_state.pt").write_bytes(b"synthetic")
    (run_root / "cv/cv_result.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
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
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"record_id": ["r0", "r1", "r2", "r3"]}).to_csv(
        run_root / "test_evaluation/test_predictions.csv", index=False
    )
    (run_root / "test_evaluation/test_predictions.npz").write_bytes(b"synthetic")
    (run_root / "test_evaluation/rq1_bootstrap_metrics.csv").write_text(
        "replicate_id\n", encoding="utf-8"
    )
    metric_names = (
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
        name: {
            "lower_2_5": 0.4,
            "upper_97_5": 0.6,
            "valid_replicates": 1_000,
            "invalid_replicates": 0,
        }
        for name in metric_names
    }
    test_payload = {
        "run_id": RUN_ID,
        "rows": 4,
        "threshold": 0.5,
        "bootstrap_indices": str(tmp_path / "locked.npz"),
        "point_metrics": {name: 0.5 for name in metric_names},
        "confidence_intervals_95_percentile": intervals,
        "counts": {
            "tp_approved": 2,
            "fn_approved": 0,
            "tn_denied": 2,
            "fp_denied": 0,
        },
        "calibration_curve": [
            {
                "bin": 0,
                "lower_edge": 0.0,
                "upper_edge": 1.0,
                "count": 4,
                "mean_p_approved": 0.5,
                "observed_approval_rate": 0.5,
            }
        ],
        "test_inference_seconds": 0.25,
    }
    monkeypatch.setattr(model_reporting, "EXPECTED_TRAIN_ROWS", 4)
    monkeypatch.setattr(model_reporting, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(model_reporting, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(
        reporting,
        "validate_tabmmini_prediction_stage",
        lambda *_args, **_kwargs: {"run_id": RUN_ID, "stage": "validation", "rows": 4},
    )
    monkeypatch.setattr(
        reporting,
        "validate_tabmmini_threshold",
        lambda *_args, **_kwargs: {"run_id": RUN_ID, "selected_threshold": 0.5},
    )
    monkeypatch.setattr(
        reporting,
        "validate_tabmmini_test_evaluation",
        lambda *_args, **_kwargs: test_payload,
    )
    summary = reporting.write_tabmmini_reporting(RUN_ID, artifact_root=tmp_path)
    reporting_root = run_root / "reporting"
    assert summary["model_key"] == "tabmmini"
    assert summary["run_id"] == RUN_ID
    assert set(path.name for path in reporting_root.iterdir()) == set(
        model_reporting.EXPECTED_REPORTING_FILES
    )
    assert set(pd.read_csv(reporting_root / "rq1_metrics_long.csv")["metric"]) == model_reporting.REQUIRED_RQ1_METRICS


def test_direct_review_recomputes_stored_predictions_without_inference(tmp_path, monkeypatch):
    root = tmp_path / "working" / RUN_ID
    test_root = root / "test_evaluation"
    test_root.mkdir(parents=True)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    np.savez_compressed(
        test_root / "test_predictions.npz",
        record_ids=np.asarray(["r0", "r1", "r2", "r3"]),
        y_approved=labels,
        p_approved=probabilities,
    )
    names = final_review.METRIC_NAMES
    pd.DataFrame(
        [
            {
                "replicate_id": replicate,
                **{name: 0.5 for name in names},
                "invalid_metrics": "{}",
            }
            for replicate in range(3)
        ]
    ).to_csv(test_root / "rq1_bootstrap_metrics.csv", index=False)
    counts = {
        "tp_approved": 2,
        "fn_approved": 0,
        "tn_denied": 2,
        "fp_denied": 0,
        "n_approved": 2,
        "n_denied": 2,
    }
    result = {
        "run_id": RUN_ID,
        "threshold": 0.5,
        "bootstrap_indices": str(tmp_path / "locked.npz"),
        "point_metrics": {name: 0.5 for name in names},
        "counts": counts,
    }
    monkeypatch.setattr(
        final_review,
        "validate_tabmmini_threshold",
        lambda *_args, **_kwargs: {"run_id": RUN_ID},
    )
    monkeypatch.setattr(
        final_review,
        "validate_tabmmini_test_evaluation",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        final_review,
        "load_locked_bootstrap_indices",
        lambda *_args, **_kwargs: np.tile(np.arange(4, dtype=np.int32), (3, 1)),
    )
    monkeypatch.setattr(
        final_review,
        "_metrics",
        lambda *_args, **_kwargs: ({name: 0.5 for name in names}, {}, counts),
    )
    monkeypatch.setattr(
        final_review,
        "validate_reporting_directory",
        lambda _path: {"run_id": RUN_ID, "reporting_status": "COMPLETE"},
    )
    reviewed = final_review.review_tabmmini_final(RUN_ID, artifact_root=tmp_path)
    assert reviewed["status"] == "PASS"
    assert reviewed["metric_count"] == 10
    assert reviewed["bootstrap_replicates"] == 3
    assert reviewed["execution"] == {"fit_calls": 0, "prediction_calls": 0}


def test_tabmmini_is_part_of_shared_six_model_reporting_contract():
    assert "tabmmini" in model_reporting.FINAL_MODEL_KEYS
    assert len(model_reporting.REQUIRED_RQ1_METRICS) == 10
    assert model_reporting.MODEL_COLORS["tabmmini"]
