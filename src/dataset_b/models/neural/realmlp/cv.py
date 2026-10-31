"""Canonical CLI and public facade for final RealMLP five-fold Train-only CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.path import PROJECT_ROOT

from .final_contract import EXPECTED_ROWS, RealMLPFinalPaths
from .final_training import (
    RealMLPFinalTrainingError as RealMLPCVError,
    run_realmlp_cv,
    validate_realmlp_cv,
)


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"
EXPECTED_TRAIN_ROWS = EXPECTED_ROWS["train"]


def run_realmlp_production_cv(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict:
    return run_realmlp_cv(run_id, artifact_root=artifact_root)


def validate_realmlp_production_cv(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict:
    return validate_realmlp_cv(
        RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate final RealMLP CV.")
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    result = (
        run_realmlp_production_cv(args.run_id, artifact_root=args.artifact_root)
        if args.command == "run"
        else validate_realmlp_production_cv(args.run_id, artifact_root=args.artifact_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
