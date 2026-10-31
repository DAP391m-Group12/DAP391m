"""Canonical Direct numerical final review for one completed RealMLP final run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset_b.models.neural.plain_mlp.test_evaluation import (
    METRIC_NAMES,
    _bootstrap_metric_matches,
    _metrics,
)
from src.dataset_b.research.rq1.reporting import validate_reporting_directory
from src.dataset_b.models.neural.realmlp.final_contract import RealMLPFinalPaths, read_json
from src.utils.path import PROJECT_ROOT


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"


class RealMLPFinalReviewError(RuntimeError):
    """Raised when direct RealMLP content or numerical recomputation differs."""


def review_realmlp_final(
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    published: bool = False,
) -> dict[str, Any]:
    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    if published:
        paths = RealMLPFinalPaths(run_id, paths.final_root, paths.final_root)
    threshold = read_json(paths.stage("threshold") / "threshold_selection.json")
    result = read_json(paths.stage("test_evaluation") / "rq1_test_metrics.json")
    predictions = pd.read_csv(
        paths.stage("test_evaluation") / "test_predictions.csv", dtype={"record_id": str}
    )
    stored = pd.read_csv(paths.stage("test_evaluation") / "rq1_bootstrap_metrics.csv")
    with np.load(result["bootstrap_indices"], allow_pickle=False) as payload:
        indices = payload["indices"]
    if {threshold.get("run_id"), result.get("run_id")} != {run_id}:
        raise RealMLPFinalReviewError("RealMLP run_id linkage differs.")
    labels = predictions["y_approved"].to_numpy(dtype=np.int8)
    probabilities = predictions["p_approved"].to_numpy(dtype=np.float64)
    point, invalid, counts = _metrics(labels, probabilities, float(result["threshold"]))
    if invalid or counts != result.get("counts"):
        raise RealMLPFinalReviewError("RealMLP point metric inputs differ.")
    for name in METRIC_NAMES:
        if not bool(
            _bootstrap_metric_matches(
                name,
                np.asarray([result["point_metrics"][name]]),
                np.asarray([point[name]]),
            ).all()
        ):
            raise RealMLPFinalReviewError(f"RealMLP point metric differs: {name}")
    if len(stored) != len(indices) or not np.array_equal(
        stored["replicate_id"].to_numpy(), np.arange(len(indices))
    ):
        raise RealMLPFinalReviewError("RealMLP bootstrap replicate identifiers differ.")
    for replicate_id, positions in enumerate(indices):
        values, problems, _ = _metrics(labels[positions], probabilities[positions], float(result["threshold"]))
        if stored.loc[replicate_id, "invalid_metrics"] != json.dumps(problems, ensure_ascii=False, sort_keys=True):
            raise RealMLPFinalReviewError("RealMLP bootstrap invalid payload differs.")
        for name in METRIC_NAMES:
            if not bool(
                _bootstrap_metric_matches(
                    name,
                    np.asarray([stored.loc[replicate_id, name]]),
                    np.asarray([values[name]]),
                ).all()
            ):
                raise RealMLPFinalReviewError(f"RealMLP bootstrap metric differs: {name}")
    validate_reporting_directory(paths.stage("reporting"))
    return {
        "schema_version": 1,
        "status": "PASS",
        "run_id": run_id,
        "acceptance_policy": "direct_content_and_numerical_recomputation",
        "metric_count": len(METRIC_NAMES),
        "bootstrap_replicates": len(indices),
        "execution": {"fit_calls": 0, "predict_proba_calls": 0},
    }
