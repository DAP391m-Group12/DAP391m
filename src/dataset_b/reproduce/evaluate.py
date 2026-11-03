"""Dispatch frozen-checkpoint evaluation to each model's environment profile."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from .assets import asset_state
from .environment import EXECUTION_ENV, python_path
from .manifest import load_manifest
from .spec import MODELS, OUTPUT_DIRECTORY, ModelSpec


SHARED_SCOPES = frozenset({"core", "data"})


def scope_problems(manifest: Mapping[str, Any], project_root: Path, spec: ModelSpec) -> list[str]:
    """Return required assets for one model that are missing or have a wrong hash."""
    problems = []
    for asset in manifest["assets"]:
        if asset["model_or_rq_scope"] not in SHARED_SCOPES | {spec.scope} or not asset["required"]:
            continue
        state = asset_state(asset, project_root)
        if state != "verified":
            problems.append(f"{asset['asset_id']}: {state}")
    return problems


def run_worker(python: Path, project_root: Path, spec: ModelSpec, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            [str(python), "-B", "-m", "src.dataset_b.reproduce.model_worker", "--model", spec.key],
            cwd=project_root,
            env={**os.environ, **EXECUTION_ENV},
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def run_predict_worker(python: Path, project_root: Path, spec: ModelSpec, input_path: Path, output_path: Path) -> int:
    return subprocess.run(
        [str(python), "-B", "-m", "src.dataset_b.reproduce.predict_worker", "--model", spec.key, "--input", str(input_path), "--output", str(output_path)],
        cwd=project_root,
        env={**os.environ, **EXECUTION_ENV},
    ).returncode


def predict_model(
    project_root: Path,
    env_root: Path,
    model: str,
    input_path: Path,
    output_path: Path,
    *,
    worker: Callable[[Path, Path, ModelSpec, Path, Path], int] = run_predict_worker,
) -> dict[str, Any]:
    """Score new records with one frozen checkpoint after its assets verify."""
    spec = MODELS[model]
    if not input_path.is_file():
        return {"status": "FAIL", "error": "input file does not exist"}
    problems = scope_problems(load_manifest(project_root), project_root, spec)
    if problems:
        return {"status": "FAIL", "error": "assets are not verified", "problem_count": len(problems), "first_problems": problems[:5]}
    python = python_path(env_root, spec.profile)
    if not python.is_file():
        return {"status": "FAIL", "error": f"environment {spec.profile} is not installed"}
    code = worker(python, project_root, spec, input_path, output_path)
    return {"status": "PASS" if code == 0 else "FAIL", "exit_code": code, "model": model}


def evaluate_models(
    project_root: Path,
    env_root: Path,
    model: str = "all",
    *,
    worker: Callable[[Path, Path, ModelSpec, Path], int] = run_worker,
) -> dict[str, Any]:
    """Evaluate the selected models; a failed precondition never falls back to stored output."""
    manifest = load_manifest(project_root)
    selected = list(MODELS) if model == "all" else [model]
    results: dict[str, Any] = {}
    for key in selected:
        spec = MODELS[key]
        problems = scope_problems(manifest, project_root, spec)
        if problems:
            results[key] = {"status": "FAIL", "error": "assets are not verified", "problem_count": len(problems), "first_problems": problems[:5]}
            continue
        python = python_path(env_root, spec.profile)
        if not python.is_file():
            results[key] = {"status": "FAIL", "error": f"environment {spec.profile} is not installed"}
            continue
        log_path = project_root / OUTPUT_DIRECTORY / f"{key}.log"
        code = worker(python, project_root, spec, log_path)
        results[key] = {
            "status": "EVALUATED" if code == 0 else "FAIL",
            "exit_code": code,
            "profile": spec.profile,
            "log": log_path.relative_to(project_root).as_posix(),
        }
    failed = [key for key, result in results.items() if result["status"] != "EVALUATED"]
    return {"status": "FAIL" if failed else "PASS", "models": results, "failed": failed}
