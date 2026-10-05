"""Public final RealMLP Validation interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset_b.runtime.realmlp_final_contract import RealMLPFinalPaths

from .realmlp_final_evaluation import (
    DEFAULT_ARTIFACT_ROOT,
    RealMLPFinalEvaluationError as RealMLPValidationError,
    run_realmlp_validation,
    validate_realmlp_validation as _validate,
)


def validate_realmlp_validation(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict:
    return _validate(RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate final RealMLP Validation inference.")
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    result = (
        run_realmlp_validation(args.run_id, artifact_root=args.artifact_root)
        if args.command == "run"
        else validate_realmlp_validation(args.run_id, artifact_root=args.artifact_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
