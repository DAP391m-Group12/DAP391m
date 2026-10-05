"""Atomic publication of a directly reviewed RealMLP final bundle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.dataset_b.reporting.model_reporting import validate_reporting_directory
from src.dataset_b.runtime.realmlp_final_contract import (
    RealMLPFinalContractError,
    RealMLPFinalPaths,
    read_json,
    write_json,
)
from src.utils.path import PROJECT_ROOT

from .realmlp_final_review import review_realmlp_final


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"


def publish_realmlp_final(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> Path:
    """Publish only after working and post-move final validation both pass."""

    paths = RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    if not paths.working_root.is_dir() or paths.final_root.exists():
        raise RealMLPFinalContractError("RealMLP working/final publication boundary is invalid.")
    review = review_realmlp_final(run_id, artifact_root=artifact_root)
    if review.get("status") != "PASS" or review.get("run_id") != run_id:
        raise RealMLPFinalContractError("RealMLP working review did not pass.")
    write_json(paths.working_root / "final_review.json", review)
    reporting = validate_reporting_directory(paths.stage("reporting"))
    if (
        reporting.get("run_id") != run_id
        or reporting.get("reporting_status") != "COMPLETE"
    ):
        raise RealMLPFinalContractError("RealMLP working reporting package is incomplete.")
    paths.final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(paths.working_root, paths.final_root)
    published_review = review_realmlp_final(
        run_id, artifact_root=artifact_root, published=True
    )
    stored = read_json(paths.final_root / "final_review.json")
    if (
        stored.get("run_id") != run_id
        or published_review.get("run_id") != run_id
        or published_review.get("status") != "PASS"
    ):
        raise RealMLPFinalContractError("Published RealMLP final review is invalid.")
    published_reporting = validate_reporting_directory(paths.final_root / "reporting")
    if (
        published_reporting.get("run_id") != run_id
        or published_reporting.get("reporting_status") != "COMPLETE"
    ):
        raise RealMLPFinalContractError(
            "Published RealMLP reporting package is incomplete."
        )
    (paths.final_root / "FROZEN").write_text(
        "REALMLP FINAL FROZEN\n", encoding="utf-8"
    )
    return paths.final_root
