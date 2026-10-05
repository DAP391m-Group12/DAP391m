"""Direct content and numerical review for one completed TabMmini final run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset_b.evaluation.plain_mlp_test_evaluation import (
    METRIC_NAMES,
    _bootstrap_metric_matches,
    _metrics,
)
from src.dataset_b.evaluation.tabmmini_final_evaluation import (
    DEFAULT_ARTIFACT_ROOT,
    TabMMiniFinalPaths,
    _read_prediction_npz,
    load_locked_bootstrap_indices,
    validate_tabmmini_test_evaluation,
    validate_tabmmini_threshold,
)
from src.dataset_b.reporting.model_reporting import validate_reporting_directory


class TabMMiniFinalReviewError(RuntimeError):
    """Raised when persisted TabMmini evidence differs on direct review."""


def review_tabmmini_final(
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    published: bool = False,
) -> dict[str, Any]:
    """Recompute metrics from stored predictions; never fit or infer."""
    paths = TabMMiniFinalPaths.for_run(run_id, artifact_root=artifact_root)
    threshold = validate_tabmmini_threshold(paths, published=published)
    result = validate_tabmmini_test_evaluation(paths, published=published)
    root = paths.stage("test_evaluation", published=published)
    labels, probabilities, _ = _read_prediction_npz(root / "test_predictions.npz")
    stored = pd.read_csv(root / "rq1_bootstrap_metrics.csv")
    indices = load_locked_bootstrap_indices(result["bootstrap_indices"], len(labels))
    if threshold.get("run_id") != run_id or result.get("run_id") != run_id:
        raise TabMMiniFinalReviewError("TabMmini run_id lineage differs.")
    point, invalid, counts = _metrics(
        labels, probabilities.astype(np.float64), float(result["threshold"])
    )
    if invalid or counts != result.get("counts"):
        raise TabMMiniFinalReviewError("TabMmini point metric inputs differ.")
    for name in METRIC_NAMES:
        if not bool(
            _bootstrap_metric_matches(
                name,
                np.asarray([result["point_metrics"][name]]),
                np.asarray([point[name]]),
            ).all()
        ):
            raise TabMMiniFinalReviewError(f"TabMmini point metric differs: {name}")
    if len(stored) != len(indices) or not np.array_equal(
        stored["replicate_id"].to_numpy(), np.arange(len(indices))
    ):
        raise TabMMiniFinalReviewError("TabMmini bootstrap replicate IDs differ.")
    for replicate_id, positions in enumerate(indices):
        values, problems, _ = _metrics(
            labels[positions],
            probabilities[positions].astype(np.float64),
            float(result["threshold"]),
        )
        expected_invalid = json.dumps(problems, ensure_ascii=False, sort_keys=True)
        if stored.loc[replicate_id, "invalid_metrics"] != expected_invalid:
            raise TabMMiniFinalReviewError("TabMmini bootstrap invalid payload differs.")
        for name in METRIC_NAMES:
            if not bool(
                _bootstrap_metric_matches(
                    name,
                    np.asarray([stored.loc[replicate_id, name]]),
                    np.asarray([values[name]]),
                ).all()
            ):
                raise TabMMiniFinalReviewError(
                    f"TabMmini bootstrap metric differs: {name}"
                )
    validate_reporting_directory(paths.stage("reporting", published=published))
    return {
        "schema_version": 1,
        "status": "PASS",
        "model_key": "tabmmini",
        "run_id": run_id,
        "acceptance_policy": "direct_content_and_numerical_recomputation",
        "metric_count": len(METRIC_NAMES),
        "bootstrap_replicates": len(indices),
        "execution": {"fit_calls": 0, "prediction_calls": 0},
    }
