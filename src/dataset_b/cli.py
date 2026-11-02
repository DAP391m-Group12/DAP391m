"""CPU-safe Dataset B handover navigation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.utils.path import PROJECT_ROOT

from .core.config import ConfigResolutionError
from .core.navigation import (
    NavigationError,
    report_index,
    status_snapshot,
    validate_configuration,
    validate_frozen_artifacts,
    validate_notebooks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset_b.cli",
        description="Read-only navigation and validation for the Dataset B handover.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show Dataset B handover status.")
    subparsers.add_parser("validate-config", help="Validate project config and catalog targets.")

    artifacts = subparsers.add_parser(
        "validate-artifacts",
        help="Validate FROZEN markers and small locked-file hashes.",
    )
    artifacts.add_argument("--scope", choices=("frozen",), default="frozen")

    notebooks = subparsers.add_parser(
        "validate-notebooks",
        help="Parse saved notebook JSON without execution.",
    )
    notebooks.add_argument("--scope", choices=("final",), default="final")

    subparsers.add_parser("report-index", help="List final notebooks and durable report files.")
    return parser


def _emit(payload: dict[str, Any], *, stream: Any | None = None) -> None:
    output = sys.stdout if stream is None else stream
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=output)


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            payload = status_snapshot(project_root)
        elif args.command == "validate-config":
            payload = validate_configuration(project_root)
        elif args.command == "validate-artifacts":
            payload = validate_frozen_artifacts(project_root)
        elif args.command == "validate-notebooks":
            payload = validate_notebooks(project_root, scope=args.scope)
        elif args.command == "report-index":
            payload = report_index(project_root)
        else:  # pragma: no cover - argparse prevents this branch.
            raise NavigationError(f"Unsupported command: {args.command}")
    except (ConfigResolutionError, NavigationError, OSError, UnicodeError) as exc:
        _emit(
            {"status": "FAIL", "command": args.command, "error": str(exc)},
            stream=sys.stderr,
        )
        return 1
    _emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
