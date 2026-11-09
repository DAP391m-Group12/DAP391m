"""Asset manifest schema, path safety, and file identity helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


MANIFEST_RELATIVE_PATH = "config/dataset_b/reproducibility/asset_manifest.json"
SCHEMA_VERSION = 1

ROLES = frozenset(
    {
        "raw_source",
        "derived_split",
        "split_manifest",
        "bootstrap_indices",
        "bootstrap_manifest",
        "checkpoint",
        "preprocessor",
        "threshold",
        "reference_prediction",
        "reference_metric",
        "rq_input",
        "reference_report",
        "run_lineage",
    }
)
OFFICIAL_SOURCE = "official_source"
DIRECT_DOWNLOAD = "direct_download"
DERIVED_FROM_RAW = "derived_from_raw"
PROJECT_BUNDLE = "project_bundle"
GIT_TRACKED = "git_tracked"
DISTRIBUTIONS = frozenset(
    {OFFICIAL_SOURCE, DIRECT_DOWNLOAD, DERIVED_FROM_RAW, PROJECT_BUNDLE, GIT_TRACKED}
)
HASH_MODES = frozenset({"bytes", "text_lf"})
RQ1_SCOPES = frozenset({"lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini"})
ASSET_FIELDS = frozenset(
    {
        "asset_id",
        "model_or_rq_scope",
        "url",
        "repository_relative_destination",
        "byte_size",
        "sha256",
        "hash_mode",
        "required",
        "distribution_classification",
        "bundle_id",
        "accepted_run_id",
        "role",
    }
)
BUNDLE_FIELDS = frozenset({"bundle_id", "filename", "url", "byte_size", "sha256"})
SOURCE_PROVENANCE_FIELDS = frozenset(
    {"source_revision", "inventory_sha256", "excluded_volatile_paths"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.IGNORECASE)


class ReproduceError(ValueError):
    """Raised when a reproduction gate cannot be satisfied."""


def check_relative_path(relative: str) -> PurePosixPath:
    """Reject any path that could leave the repository root."""
    if not isinstance(relative, str) or not relative or relative != relative.strip():
        raise ReproduceError(f"Unsafe repository path: {relative!r}")
    if "\\" in relative or ":" in relative or relative.startswith("/"):
        raise ReproduceError(f"Unsafe repository path: {relative!r}")
    path = PurePosixPath(relative)
    if path.as_posix() != relative:
        raise ReproduceError(f"Unsafe repository path: {relative!r}")
    for part in path.parts:
        if part in {"", ".", ".."} or _WINDOWS_RESERVED.match(part) or part.endswith((" ", ".")):
            raise ReproduceError(f"Unsafe repository path: {relative!r}")
    if ".git" in {part.lower() for part in path.parts}:
        raise ReproduceError(f"Unsafe repository path: {relative!r}")
    return path


def safe_destination(project_root: Path, relative: str) -> Path:
    """Resolve a manifest destination and confirm it stays inside the root."""
    checked = check_relative_path(relative)
    root = project_root.resolve()
    destination = root.joinpath(*checked.parts)
    if not destination.resolve().is_relative_to(root):
        raise ReproduceError(f"Unsafe repository path: {relative!r}")
    return destination


def file_identity(path: Path, *, hash_mode: str = "bytes") -> tuple[int, str]:
    """Return size and SHA-256; ``text_lf`` hashes content with LF line endings.

    Git-tracked text is checked out with host-specific line endings, so its
    identity is defined on normalized content.
    """
    digest = hashlib.sha256()
    if hash_mode == "text_lf":
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(content)
        return len(content), digest.hexdigest()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _url_errors(owner: str, url: Any) -> list[str]:
    if url is None:
        return []
    if not isinstance(url, str) or not url.startswith("https://"):
        return [f"{owner}: url must be https or null"]
    return []


def _identity_errors(owner: str, record: Mapping[str, Any]) -> list[str]:
    errors = []
    size = record.get("byte_size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        errors.append(f"{owner}: byte_size must be a non-negative integer")
    sha256 = record.get("sha256")
    if not isinstance(sha256, str) or not _SHA256.match(sha256):
        errors.append(f"{owner}: sha256 must be 64 lowercase hex characters")
    return errors


def _asset_errors(asset: Any, bundles: Mapping[str, Mapping[str, Any]]) -> list[str]:
    if not isinstance(asset, dict):
        return ["asset entry must be an object"]
    owner = f"asset {asset.get('asset_id')!r}"
    if set(asset) != ASSET_FIELDS:
        return [f"{owner}: fields must equal the schema; got {sorted(asset)}"]
    errors = _identity_errors(owner, asset) + _url_errors(owner, asset["url"])
    if not isinstance(asset["asset_id"], str) or not asset["asset_id"]:
        errors.append(f"{owner}: asset_id must be a non-empty string")
    try:
        check_relative_path(asset["repository_relative_destination"])
    except ReproduceError as exc:
        errors.append(f"{owner}: {exc}")
    if asset["role"] not in ROLES:
        errors.append(f"{owner}: unknown role {asset['role']!r}")
    if asset["hash_mode"] not in HASH_MODES:
        errors.append(f"{owner}: unknown hash_mode {asset['hash_mode']!r}")
    if not isinstance(asset["required"], bool):
        errors.append(f"{owner}: required must be boolean")
    if asset["model_or_rq_scope"] in RQ1_SCOPES and not asset["accepted_run_id"]:
        errors.append(f"{owner}: RQ1 scope requires accepted_run_id")

    distribution = asset["distribution_classification"]
    if distribution not in DISTRIBUTIONS:
        errors.append(f"{owner}: unknown distribution {distribution!r}")
    elif distribution == OFFICIAL_SOURCE:
        if not asset["url"] or asset["bundle_id"] is not None:
            errors.append(f"{owner}: official source needs a url and no bundle")
    elif distribution == DIRECT_DOWNLOAD:
        if asset["bundle_id"] is not None:
            errors.append(f"{owner}: direct download must not name a bundle")
    elif distribution == PROJECT_BUNDLE:
        bundle = bundles.get(asset["bundle_id"])
        if bundle is None:
            errors.append(f"{owner}: unknown bundle {asset['bundle_id']!r}")
        elif asset["url"] != bundle["url"]:
            errors.append(f"{owner}: url must equal its bundle url")
    elif asset["url"] is not None or asset["bundle_id"] is not None:
        errors.append(f"{owner}: {distribution} must have no url and no bundle")
    return errors


def validate_manifest(manifest: Any) -> None:
    """Raise with every schema violation found in one manifest."""
    if not isinstance(manifest, dict):
        raise ReproduceError("Asset manifest must be a JSON object")
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if manifest.get("dataset") != "dataset_b":
        errors.append("dataset must be dataset_b")

    provenance = manifest.get("source_provenance")
    if provenance is not None:
        if not isinstance(provenance, dict) or set(provenance) != SOURCE_PROVENANCE_FIELDS:
            errors.append("source_provenance fields must equal the schema")
        else:
            revision = provenance["source_revision"]
            if revision is not None and (
                not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            ):
                errors.append("source_provenance source_revision must be null or a full Git SHA")
            inventory_sha256 = provenance["inventory_sha256"]
            if not isinstance(inventory_sha256, str) or _SHA256.match(inventory_sha256) is None:
                errors.append("source_provenance inventory_sha256 must be 64 lowercase hex characters")
            excluded = provenance["excluded_volatile_paths"]
            if not isinstance(excluded, list):
                errors.append("source_provenance excluded_volatile_paths must be a list")
            else:
                for relative in excluded:
                    try:
                        check_relative_path(relative)
                    except ReproduceError as exc:
                        errors.append(f"source_provenance excluded path: {exc}")

    bundles: dict[str, Mapping[str, Any]] = {}
    raw_bundles = manifest.get("bundles")
    if not isinstance(raw_bundles, list):
        errors.append("bundles must be a list")
        raw_bundles = []
    for bundle in raw_bundles:
        if not isinstance(bundle, dict) or set(bundle) != BUNDLE_FIELDS:
            errors.append(f"bundle fields must equal the schema: {bundle!r}")
            continue
        owner = f"bundle {bundle['bundle_id']!r}"
        errors += _identity_errors(owner, bundle) + _url_errors(owner, bundle["url"])
        if bundle["bundle_id"] in bundles:
            errors.append(f"{owner}: duplicate bundle_id")
        try:
            check_relative_path(bundle["filename"])
        except ReproduceError as exc:
            errors.append(f"{owner}: {exc}")
        bundles[bundle["bundle_id"]] = bundle

    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        errors.append("assets must be a non-empty list")
        assets = []
    seen_ids: set[Any] = set()
    seen_destinations: set[str] = set()
    for asset in assets:
        errors += _asset_errors(asset, bundles)
        if not isinstance(asset, dict):
            continue
        if asset.get("asset_id") in seen_ids:
            errors.append(f"duplicate asset_id: {asset.get('asset_id')!r}")
        seen_ids.add(asset.get("asset_id"))
        destination = str(asset.get("repository_relative_destination")).lower()
        if destination in seen_destinations:
            errors.append(f"duplicate destination: {destination}")
        seen_destinations.add(destination)
    if errors:
        raise ReproduceError("Invalid asset manifest: " + "; ".join(errors))


def load_manifest(project_root: Path, relative: str = MANIFEST_RELATIVE_PATH) -> dict[str, Any]:
    """Load and validate the repository asset manifest."""
    path = project_root / relative
    if not path.is_file():
        raise ReproduceError(f"Asset manifest does not exist: {relative}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def unpublished_assets(assets: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return required assets that have no public download route."""
    return [
        asset["asset_id"]
        for asset in assets
        if asset["required"]
        and asset["distribution_classification"] in {PROJECT_BUNDLE, DIRECT_DOWNLOAD}
        and asset["url"] is None
    ]
