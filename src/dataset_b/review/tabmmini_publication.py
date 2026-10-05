"""Atomic working-to-final publication for a reviewed TabMmini run."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.dataset_b.evaluation.tabmmini_final_evaluation import (
    DEFAULT_ARTIFACT_ROOT,
    TabMMiniFinalEvaluationError,
    TabMMiniFinalPaths,
    _write_json,
)
from src.dataset_b.reporting.model_reporting import validate_reporting_directory

from .tabmmini_final_review import review_tabmmini_final


Reviewer = Callable[..., Mapping[str, Any]]
ReportingValidator = Callable[[str | Path], Mapping[str, Any]]


def publish_tabmmini_final(
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    reviewer: Reviewer = review_tabmmini_final,
    reporting_validator: ReportingValidator = validate_reporting_directory,
) -> Path:
    """Publish only after working and post-move final validation both pass."""
    paths = TabMMiniFinalPaths.for_run(run_id, artifact_root=artifact_root)
    if not paths.working_root.is_dir():
        raise TabMMiniFinalEvaluationError("TabMmini working bundle is missing.")
    if paths.final_root.exists():
        raise FileExistsError("TabMmini final destination already exists.")
    review = dict(reviewer(run_id, artifact_root=artifact_root, published=False))
    if review.get("status") != "PASS" or review.get("run_id") != run_id:
        raise TabMMiniFinalEvaluationError("TabMmini direct review did not pass.")
    reporting = reporting_validator(paths.stage("reporting"))
    if reporting.get("run_id") != run_id or reporting.get("reporting_status") != "COMPLETE":
        raise TabMMiniFinalEvaluationError("TabMmini reporting package is incomplete.")
    _write_json(paths.working_root / "final_review.json", review)
    paths.final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(paths.working_root, paths.final_root)
    published_review = dict(
        reviewer(run_id, artifact_root=artifact_root, published=True)
    )
    if (
        published_review.get("status") != "PASS"
        or published_review.get("run_id") != run_id
    ):
        raise TabMMiniFinalEvaluationError(
            "Published TabMmini direct review did not pass."
        )
    published_reporting = reporting_validator(
        paths.stage("reporting", published=True)
    )
    if (
        published_reporting.get("run_id") != run_id
        or published_reporting.get("reporting_status") != "COMPLETE"
    ):
        raise TabMMiniFinalEvaluationError(
            "Published TabMmini reporting package is incomplete."
        )
    (paths.final_root / "FROZEN").write_text(
        "TABMMINI FINAL FROZEN\n", encoding="utf-8"
    )
    return paths.final_root
