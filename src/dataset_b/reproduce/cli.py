"""Public Dataset B reproduction CLI.

Exit codes: 0 PASS, 1 FAIL or INCOMPLETE, 2 usage error, 3 gate BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.utils.path import PROJECT_ROOT

from .assets import download_assets, verify_assets
from .compare import compare_all
from .environment import DEFAULT_ENV_DIRECTORY, check_environments
from .evaluate import evaluate_models, predict_model
from .manifest import MANIFEST_RELATIVE_PATH, ReproduceError, load_manifest
from .reports import build_reports
from .research import ROUTES, run_routes
from .spec import MODELS


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 3
RECEIPT_DIRECTORY = "reports/reproducibility/receipts"
MODEL_CHOICES = ("all", *MODELS)


def _env_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-root", type=Path, help=f"Environment directory (default: {DEFAULT_ENV_DIRECTORY}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset_b.reproduce",
        description="Reproduce Dataset B from public assets and frozen checkpoints.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="Download, verify, and place every asset.")
    subparsers.add_parser("verify-assets", help="Hash every declared asset without changing anything.")
    environment = subparsers.add_parser("environment", help="Record and check the runtime environments.")
    environment.add_argument("--install", action="store_true", help="Create or update each profile with uv first.")
    _env_root_argument(environment)
    evaluate = subparsers.add_parser("evaluate", help="Run frozen-checkpoint Test evaluation.")
    evaluate.add_argument("--model", choices=MODEL_CHOICES, default="all")
    _env_root_argument(evaluate)
    predict = subparsers.add_parser("predict", help="Score new records with one frozen checkpoint; nothing is fitted.")
    predict.add_argument("--model", choices=tuple(MODELS), required=True)
    predict.add_argument("--input", type=Path, required=True, help="CSV with record_id and the 15 raw model inputs.")
    predict.add_argument("--output", type=Path, required=True)
    _env_root_argument(predict)
    subparsers.add_parser("compare", help="Compare observed metrics with accepted references.")
    research = subparsers.add_parser("research", help="Run the RQ2-RQ4 routes under their contracts.")
    research.add_argument("--route", choices=("all", *ROUTES), default="all")
    _env_root_argument(research)
    subparsers.add_parser("reports", help="Regenerate reports from reproduction outputs.")
    run_all = subparsers.add_parser("all", help="Run every stage in order and stop at the first failure.")
    _env_root_argument(run_all)

    build = subparsers.add_parser("maintainer-build", help="Build bundles and an unpublished manifest.")
    build.add_argument("--source-root", type=Path, required=True)
    bind = subparsers.add_parser("maintainer-bind-release", help="Record the published release location.")
    bind.add_argument("--base-url", required=True)
    links = subparsers.add_parser("maintainer-bind-links", help="Record one download link per bundle.")
    links.add_argument("--links-file", type=Path, required=True, help="JSON object: bundle filename to link.")
    split_links = subparsers.add_parser(
        "maintainer-bind-split-links", help="Record direct-download links for the three locked split CSVs."
    )
    split_links.add_argument(
        "--links-file", type=Path, required=True, help="JSON object: split CSV filename to link."
    )
    subparsers.add_parser("maintainer-check-publication", help="Check anonymous access to each bundle.")
    local = subparsers.add_parser("maintainer-install-local", help="Place assets from local bundles; not a public route.")
    local.add_argument("--bundle-directory", type=Path, default=Path("dist/reproducibility"))
    return parser


def _emit(payload: dict[str, Any], *, stream: Any | None = None) -> None:
    output = sys.stdout if stream is None else stream
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=output)


def _exit_code(payload: dict[str, Any]) -> int:
    status = payload.get("status")
    if status == "PASS":
        return EXIT_PASS
    if status == "BLOCKED":
        return EXIT_BLOCKED
    return EXIT_FAIL


def _write_receipt(project_root: Path, command: str, payload: dict[str, Any]) -> str:
    created_at = datetime.now(timezone.utc)
    manifest_path = project_root / MANIFEST_RELATIVE_PATH
    receipt = {
        **payload,
        "command": command,
        "created_at_utc": created_at.isoformat(),
        "manifest_sha256": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
        ),
    }
    directory = project_root / RECEIPT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{command}_{created_at.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return path.relative_to(project_root).as_posix()


def _public_error(exc: Exception) -> str:
    # OSError text can embed an absolute path, and receipts are kept.
    if isinstance(exc, OSError):
        return f"{type(exc).__name__}: {exc.strerror or 'I/O failure'}"
    return str(exc)


def _run_public(command: str, project_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if command == "reports":
        return build_reports(project_root)
    env_root = getattr(args, "env_root", None) or project_root / DEFAULT_ENV_DIRECTORY
    if command == "environment":
        return check_environments(project_root, env_root, install=getattr(args, "install", False))
    if command == "evaluate":
        return evaluate_models(project_root, env_root, getattr(args, "model", "all"))
    if command == "predict":
        return predict_model(project_root, env_root, args.model, args.input.resolve(), args.output.resolve())
    if command == "compare":
        return compare_all(project_root)
    if command == "research":
        return run_routes(project_root, env_root, getattr(args, "route", "all"))
    manifest = load_manifest(project_root)
    if command == "download":
        return download_assets(manifest, project_root)
    return verify_assets(manifest, project_root)


def _run_maintainer(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    from .release import (
        bind_links,
        bind_release,
        bind_split_links,
        build_release,
        check_publication,
        install_local_bundles,
        write_manifest,
    )

    if args.command == "maintainer-build":
        return build_release(args.source_root, project_root)
    manifest = load_manifest(project_root)
    if args.command == "maintainer-bind-links":
        links = json.loads(args.links_file.read_text(encoding="utf-8"))
        write_manifest(bind_links(manifest, links), project_root)
        return {"status": "PASS", "bundle_count": len(manifest["bundles"])}
    if args.command == "maintainer-bind-split-links":
        links = json.loads(args.links_file.read_text(encoding="utf-8"))
        write_manifest(bind_split_links(manifest, links), project_root)
        return {"status": "PASS", "split_count": len(links)}
    if args.command == "maintainer-bind-release":
        write_manifest(bind_release(manifest, args.base_url), project_root)
        return {"status": "PASS", "bundle_count": len(manifest["bundles"])}
    if args.command == "maintainer-install-local":
        return install_local_bundles(manifest, project_root, project_root / args.bundle_directory)
    return check_publication(manifest)


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> int:
    args = build_parser().parse_args(argv)
    root = project_root.resolve()
    try:
        if args.command.startswith("maintainer-"):
            payload = _run_maintainer(args, root)
            _emit(payload)
            return _exit_code(payload)

        stages = (
            ("download", "verify-assets", "environment", "evaluate", "compare", "research", "reports")
            if args.command == "all"
            else (args.command,)
        )
        stage_results: dict[str, Any] = {}
        for stage in stages:
            payload = _run_public(stage, root, args)
            stage_results[stage] = payload
            if payload.get("status") != "PASS":
                break
    except (ReproduceError, OSError, UnicodeError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        payload = {"status": "FAIL", "command": args.command, "error": _public_error(exc)}
        payload["receipt"] = _write_receipt(root, args.command, payload)
        _emit(payload, stream=sys.stderr)
        return EXIT_FAIL

    final = dict(payload)
    if args.command == "all":
        final = {"status": payload.get("status"), "stopped_at": stage, "stages": stage_results}
    final["receipt"] = _write_receipt(root, args.command, final)
    _emit(final)
    return _exit_code(final)


if __name__ == "__main__":
    raise SystemExit(main())
