"""Canonical CLI and compatibility facade for the final RealMLP Train-only pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.path import PROJECT_ROOT

from .final_contract import RealMLPFinalPaths
from .final_training import (
    RealMLPFinalTrainingError as RealMLPPilotRunError,
    lock_realmlp_fixed_epoch,
    run_realmlp_pilot,
    validate_realmlp_pilot,
)


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp"


def validate_realmlp_pilot_run(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict:
    return validate_realmlp_pilot(
        RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or decide the final RealMLP Train-only pilot.")
    parser.add_argument("command", choices=("run", "decide", "validate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--fixed-epoch", type=int)
    parser.add_argument("--rationale")
    args = parser.parse_args()
    if args.command == "run":
        result = run_realmlp_pilot(args.run_id, artifact_root=args.artifact_root)
    elif args.command == "decide":
        result = lock_realmlp_fixed_epoch(
            args.run_id,
            args.fixed_epoch,
            artifact_root=args.artifact_root,
            rationale=args.rationale or "",
        )
    else:
        result = validate_realmlp_pilot_run(args.run_id, artifact_root=args.artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
