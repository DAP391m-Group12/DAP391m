"""Public final RealMLP Test-evaluation interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset_b.runtime.realmlp_final_contract import RealMLPFinalPaths

from .realmlp_final_evaluation import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BOOTSTRAP_INDICES,
    RealMLPFinalEvaluationError as RealMLPTestEvaluationError,
    run_realmlp_test_evaluation,
    validate_realmlp_test_evaluation as _validate,
)


def validate_realmlp_test_evaluation(
    run_id: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> dict:
    return _validate(RealMLPFinalPaths.for_run(run_id, artifact_root=artifact_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate final RealMLP Test evaluation.")
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--bootstrap-indices", type=Path, default=DEFAULT_BOOTSTRAP_INDICES)
    args = parser.parse_args()
    result = (
        run_realmlp_test_evaluation(
            args.run_id,
            artifact_root=args.artifact_root,
            bootstrap_indices_path=args.bootstrap_indices,
        )
        if args.command == "run"
        else validate_realmlp_test_evaluation(args.run_id, artifact_root=args.artifact_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
