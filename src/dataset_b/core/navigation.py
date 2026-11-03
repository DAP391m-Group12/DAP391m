"""Read-only repository navigation and handover validation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.path import PROJECT_ROOT

from .config import validate_catalog


FINAL_MODELS = {
    "lr": "artifacts/dataset_b/lr/final/lr_final_15f_20261002_01/FROZEN",
    "rf": "artifacts/dataset_b/rf/final/rf_final_15f_20261002_01/FROZEN",
    "xgboost": "artifacts/dataset_b/xgboost/final/xgb_final_15f_20261003_01/FROZEN",
    "plain_mlp": "artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03/FROZEN",
    "realmlp": "artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/FROZEN",
    "tabmmini": "artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01/FROZEN",
}

LOCKED_FILES = {
    "artifacts/dataset_b/rq1_test_bootstrap_indices.npz":
        "2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1",
    "artifacts/dataset_b/rq1_test_bootstrap_manifest.json":
        "5544af4ebe91bfdf8c5b3b574fa2ef24f3da8c14827f353970cbe9d759e9d21e",
    "artifacts/dataset_b/split_v1.json":
        "66816ba47da64bfd85c010a90a14b9d80443a46d319258bf9fc1b3657f731596",
}


class NavigationError(ValueError):
    """Raised when a read-only handover validation fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "project.yaml"
    if not path.is_file():
        raise NavigationError(f"Project config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NavigationError("Project config must be a YAML mapping")
    return payload


def status_snapshot(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return handover status without importing or running experiment code."""
    root = project_root.resolve()
    project = _project_config(root)
    markers = {
        model: (root / relative).is_file()
        for model, relative in FINAL_MODELS.items()
    }
    return {
        "status": "PASS" if all(markers.values()) else "INCOMPLETE",
        "project": project.get("project", {}).get("name"),
        "active_dataset": project.get("project", {}).get("active_dataset"),
        "frozen_markers": markers,
        "safety": {
            "training_allowed": False,
            "inference_allowed": False,
            "notebook_execution_allowed": False,
        },
    }


def validate_frozen_artifacts(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate markers and small locked files without loading checkpoints."""
    root = project_root.resolve()
    missing_markers = [
        model for model, relative in FINAL_MODELS.items()
        if not (root / relative).is_file()
    ]
    if missing_markers:
        raise NavigationError(f"Missing FROZEN markers: {', '.join(missing_markers)}")

    verified: dict[str, str] = {}
    for relative, expected_hash in LOCKED_FILES.items():
        path = root / relative
        if not path.is_file():
            raise NavigationError(f"Locked file does not exist: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise NavigationError(f"Locked file hash mismatch: {relative}")
        verified[relative] = actual_hash
    return {
        "status": "PASS",
        "scope": "frozen",
        "frozen_marker_count": len(FINAL_MODELS),
        "locked_files": verified,
        "checkpoints_loaded": 0,
        "inference_calls": 0,
    }


def validate_notebooks(
    project_root: Path = PROJECT_ROOT,
    *,
    scope: str = "final",
) -> dict[str, Any]:
    """Parse saved notebooks as JSON without executing or rewriting them."""
    if scope != "final":
        raise NavigationError(f"Unsupported notebook scope: {scope}")
    root = project_root.resolve()
    notebook_root = root / "notebooks" / "dataset_b" / "final"
    paths = sorted(notebook_root.rglob("*.ipynb"))
    if not paths:
        raise NavigationError(f"No final notebooks found: {notebook_root}")

    saved_errors = 0
    code_cells = 0
    for path in paths:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NavigationError(f"Invalid notebook JSON: {path}") from exc
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise NavigationError(f"Unsupported notebook structure: {path}")
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            saved_errors += sum(
                output.get("output_type") == "error"
                for output in cell.get("outputs", [])
            )
    return {
        "status": "PASS" if saved_errors == 0 else "REVIEW",
        "scope": scope,
        "notebook_count": len(paths),
        "code_cell_count": code_cells,
        "saved_error_count": saved_errors,
        "executed": False,
        "paths": [path.relative_to(root).as_posix() for path in paths],
    }


def report_index(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """List current reporting notebooks and durable report documents."""
    root = project_root.resolve()
    notebook_root = root / "notebooks" / "dataset_b" / "final"
    report_root = root / "reports"
    notebooks = sorted(
        path.relative_to(root).as_posix()
        for path in notebook_root.rglob("*.ipynb")
    )
    report_suffixes = {".md", ".txt", ".pdf", ".docx", ".html"}
    reports = sorted(
        path.relative_to(root).as_posix()
        for path in report_root.rglob("*")
        if path.is_file() and path.suffix.lower() in report_suffixes
    ) if report_root.is_dir() else []
    return {
        "status": "PASS",
        "final_notebooks": notebooks,
        "reports": reports,
        "notebook_count": len(notebooks),
        "report_count": len(reports),
    }


def validate_configuration(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate the project config and all catalog targets."""
    root = project_root.resolve()
    project = _project_config(root)
    if project.get("project", {}).get("active_dataset") != "dataset_b":
        raise NavigationError("Project config must declare dataset_b as active_dataset")
    result = validate_catalog(root)
    result["project_config"] = "config/project.yaml"
    return result
