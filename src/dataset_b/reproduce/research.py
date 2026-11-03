"""Dispatch the RQ2-RQ4 routes to the rq_cpu environment profile."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from .environment import EXECUTION_ENV, python_path


ROUTES = ("rq3", "rq4a", "rq4b", "rq2")
RQ_PROFILE = "rq_cpu"
OUTPUT_ROOT = "reports/reproducibility/research"
ALLOWED_CLASSIFICATIONS = frozenset({"checkpoint-based", "artifact-based", "report-only", "historical-evidence-only"})
REQUIRED_RECEIPT_FIELDS = frozenset({"classification", "input_hashes", "output_hashes", "accepted_run_ids", "comparison_result", "limitations"})


def run_worker(python: Path, project_root: Path, route: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            [str(python), "-B", "-m", "src.dataset_b.reproduce.rq_worker", "--route", route],
            cwd=project_root,
            env={**os.environ, **EXECUTION_ENV},
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def check_receipt(project_root: Path, route: str) -> list[str]:
    """Return every reason a route receipt cannot support a PASS."""
    path = project_root / OUTPUT_ROOT / route / "receipt.json"
    if not path.is_file():
        return [f"{route}: receipt is missing"]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    problems = [f"{route}: receipt field is missing: {name}" for name in sorted(REQUIRED_RECEIPT_FIELDS - set(receipt))]
    if receipt.get("classification") not in ALLOWED_CLASSIFICATIONS:
        problems.append(f"{route}: classification is not allowed: {receipt.get('classification')!r}")
    result = receipt.get("comparison_result", {})
    if result.get("status") != "PASS" or not result.get("row_count"):
        problems.append(f"{route}: comparison did not pass")
    return problems


def run_routes(
    project_root: Path,
    env_root: Path,
    route: str = "all",
    *,
    worker: Callable[[Path, Path, str, Path], int] = run_worker,
) -> dict[str, Any]:
    python = python_path(env_root, RQ_PROFILE)
    if not python.is_file():
        return {"status": "FAIL", "error": f"environment {RQ_PROFILE} is not installed"}
    selected = ROUTES if route == "all" else (route,)
    results: dict[str, Any] = {}
    for name in selected:
        log_path = project_root / OUTPUT_ROOT / f"{name}.log"
        code = worker(python, project_root, name, log_path)
        problems = [f"{name}: worker exit code {code}"] if code else []
        problems += check_receipt(project_root, name)
        results[name] = {
            "status": "FAIL" if problems else "PASS",
            "exit_code": code,
            "problems": problems,
            "log": log_path.relative_to(project_root).as_posix(),
        }
    failed = [name for name, result in results.items() if result["status"] != "PASS"]
    return {"status": "FAIL" if failed else "PASS", "routes": results, "failed": failed}
