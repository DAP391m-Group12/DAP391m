"""Resolve Dataset B configuration roles through the paths-only catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

from src.utils.path import PROJECT_ROOT


CATALOG_GROUPS = ("contracts", "models", "research", "runtime")


class ConfigResolutionError(ValueError):
    """Raised when the catalog or a requested configuration role is invalid."""


def _catalog_path(project_root: Path, catalog_path: Path | None) -> Path:
    if catalog_path is None:
        return project_root / "config" / "dataset_b" / "catalog.yaml"
    return catalog_path if catalog_path.is_absolute() else project_root / catalog_path


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigResolutionError(f"{label} does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigResolutionError(f"{label} is not valid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigResolutionError(f"{label} must be a YAML mapping: {path}")
    return payload


def load_catalog(
    project_root: Path = PROJECT_ROOT,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Load and minimally validate the Dataset B catalog."""
    root = project_root.resolve()
    path = _catalog_path(root, catalog_path)
    catalog = _load_yaml_mapping(path, label="Dataset B catalog")
    if catalog.get("schema_version") != 1:
        raise ConfigResolutionError("Dataset B catalog schema_version must equal 1")
    if catalog.get("dataset") != "dataset_b":
        raise ConfigResolutionError("Dataset B catalog must declare dataset: dataset_b")
    for group in CATALOG_GROUPS:
        if not isinstance(catalog.get(group), dict) or not catalog[group]:
            raise ConfigResolutionError(f"Catalog group must be a non-empty mapping: {group}")
    return catalog


def iter_catalog_entries(catalog: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield stable dotted role keys and repository-relative target paths."""
    for group in CATALOG_GROUPS:
        entries = catalog[group]
        for name in sorted(entries):
            target = entries[name]
            if not isinstance(target, str) or not target.strip():
                raise ConfigResolutionError(f"Catalog target must be a path string: {group}.{name}")
            yield f"{group}.{name}", target


def _resolve_repository_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ConfigResolutionError(f"Catalog target must be repository-relative: {relative_path}")
    root = project_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigResolutionError(f"Catalog target escapes the repository: {relative_path}") from exc
    return resolved


def config_path(
    role: str,
    *,
    project_root: Path = PROJECT_ROOT,
    catalog_path: Path | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve a dotted role such as ``models.realmlp`` to one YAML file."""
    catalog = load_catalog(project_root, catalog_path)
    try:
        group, name = role.split(".", maxsplit=1)
    except ValueError as exc:
        raise ConfigResolutionError(f"Config role must use '<group>.<name>': {role}") from exc
    if group not in CATALOG_GROUPS or name not in catalog[group]:
        raise ConfigResolutionError(f"Unknown Dataset B config role: {role}")
    resolved = _resolve_repository_path(project_root, catalog[group][name])
    if require_exists and not resolved.is_file():
        raise ConfigResolutionError(f"Config target does not exist for {role}: {resolved}")
    return resolved


def validate_catalog(
    project_root: Path = PROJECT_ROOT,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Validate every catalog target without importing any experiment code."""
    root = project_root.resolve()
    catalog = load_catalog(root, catalog_path)
    resolved_targets: dict[str, str] = {}
    seen: set[Path] = set()
    for role, target in iter_catalog_entries(catalog):
        resolved = _resolve_repository_path(root, target)
        if resolved in seen:
            raise ConfigResolutionError(f"Duplicate catalog target: {target}")
        seen.add(resolved)
        _load_yaml_mapping(resolved, label=f"Config target {role}")
        resolved_targets[role] = resolved.relative_to(root).as_posix()
    return {
        "status": "PASS",
        "dataset": "dataset_b",
        "target_count": len(resolved_targets),
        "targets": resolved_targets,
    }
