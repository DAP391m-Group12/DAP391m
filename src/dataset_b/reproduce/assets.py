"""Download, extract, derive, and verify reproduction assets."""

from __future__ import annotations

import http.client
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .manifest import (
    DERIVED_FROM_RAW,
    GIT_TRACKED,
    OFFICIAL_SOURCE,
    ReproduceError,
    check_relative_path,
    file_identity,
    safe_destination,
    unpublished_assets,
    validate_manifest,
)


DOWNLOAD_DIRECTORY = "dist/reproducibility/downloads"
RAW_DESTINATION = "data/raw/state_NY_2024.csv"
SPLIT_MANIFEST_DESTINATION = "artifacts/dataset_b/split_v1.json"
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
_CHUNK_BYTES = 1024 * 1024


def _fetch_into_part(
    url: str,
    part: Path,
    opener: Callable[..., Any],
) -> None:
    offset = part.stat().st_size if part.exists() else 0
    # The HMDA endpoint answers 403 to a custom User-Agent, so keep the library default.
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=60) as response:
        final_url = response.geturl() if hasattr(response, "geturl") else url
        if not final_url.startswith("https://"):
            raise ReproduceError(f"Download redirected away from https: {final_url}")
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type.startswith("text/html"):
            # Share pages and virus-scan notices must never enter a partial file.
            raise ReproduceError(f"Link returned an HTML page instead of a file: {part.name}")
        status = getattr(response, "status", 200)
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise OSError(f"Unexpected Content-Range: {content_range!r}")
            mode = "ab"
        elif status == 200:
            # The server ignored the Range request, so restart from zero.
            mode = "wb"
        else:
            raise OSError(f"Unexpected HTTP status: {status}")
        with part.open(mode) as stream:
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                stream.write(chunk)


def download_file(
    url: str,
    destination: Path,
    *,
    byte_size: int,
    sha256: str,
    opener: Callable[..., Any] | None = None,
    retries: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Fetch one file with resume; publish it only after its hash matches.

    Returns ``True`` when a new file was published and ``False`` when a verified
    file already existed. A local file with the wrong identity is never replaced.
    """
    if not url.startswith("https://"):
        raise ReproduceError(f"Only https downloads are allowed: {url}")
    if destination.exists():
        if file_identity(destination) == (byte_size, sha256):
            return False
        raise ReproduceError(f"Refusing to overwrite local file with a wrong hash: {destination.name}")

    opener = urllib.request.urlopen if opener is None else opener
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    last_error: object = None
    for attempt in range(retries):
        if attempt:
            sleep(min(2.0**attempt, 30.0))
        if part.exists() and part.stat().st_size > byte_size:
            part.unlink()
        if not part.exists() or part.stat().st_size < byte_size:
            try:
                _fetch_into_part(url, part, opener)
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code == 416:
                    part.unlink(missing_ok=True)
                elif exc.code not in _RETRYABLE_HTTP:
                    raise ReproduceError(f"Download failed for {destination.name}: HTTP {exc.code}") from exc
                continue
            except (OSError, http.client.HTTPException) as exc:
                # Keep the partial file so the next attempt can resume it.
                last_error = exc
                continue
        current_size = part.stat().st_size if part.exists() else 0
        if current_size < byte_size:
            last_error = f"short read: {current_size} of {byte_size} bytes"
            continue
        if file_identity(part) == (byte_size, sha256):
            os.replace(part, destination)
            return True
        part.unlink()
        last_error = "checksum mismatch"
    raise ReproduceError(f"Download failed for {destination.name} after {retries} attempts: {last_error}")


def asset_state(asset: Mapping[str, Any], project_root: Path) -> str:
    """Classify one asset on disk as ``verified``, ``missing``, or ``mismatch``."""
    path = safe_destination(project_root, asset["repository_relative_destination"])
    if not path.is_file():
        return "missing"
    identity = file_identity(path, hash_mode=asset["hash_mode"])
    return "verified" if identity == (asset["byte_size"], asset["sha256"]) else "mismatch"


def extract_bundle(
    bundle_path: Path,
    bundle_assets: Sequence[Mapping[str, Any]],
    project_root: Path,
) -> int:
    """Publish the missing members of a verified bundle after hashing each one."""
    expected = {asset["repository_relative_destination"]: asset for asset in bundle_assets}
    extracted = 0
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        for name in names:
            check_relative_path(name)
        unexpected = sorted(set(names) - set(expected))
        absent = sorted(set(expected) - set(names))
        if unexpected or absent or len(names) != len(set(names)):
            raise ReproduceError(
                f"Bundle {bundle_path.name} does not match the manifest: "
                f"unexpected={unexpected[:5]} absent={absent[:5]}"
            )
        for name, asset in sorted(expected.items()):
            if asset_state(asset, project_root) != "missing":
                continue
            destination = safe_destination(project_root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.with_name(destination.name + ".part")
            with archive.open(name) as source, part.open("wb") as target:
                shutil.copyfileobj(source, target, _CHUNK_BYTES)
            if file_identity(part) != (asset["byte_size"], asset["sha256"]):
                part.unlink()
                raise ReproduceError(f"Bundle member failed its hash check: {name}")
            os.replace(part, destination)
            extracted += 1
    return extracted


def derive_split_csvs(project_root: Path, assets: Sequence[Mapping[str, Any]]) -> int:
    """Rebuild the split CSVs from raw data and the locked split manifest."""
    # Imported here because this route needs the installed project dependencies.
    from src.dataset_b.data import DatasetBSplitter, DatasetBViewBuilder
    from src.dataset_b.workflows.create_split import write_split_csvs

    root = project_root.resolve()
    view = DatasetBViewBuilder.from_yaml(root / "config/dataset_b/feature_set_v1.yaml").build(
        root / RAW_DESTINATION
    )
    manifest = DatasetBSplitter.load_manifest(root / SPLIT_MANIFEST_DESTINATION)
    published = 0
    (root / "data/dataset_b").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "data/dataset_b", prefix=".split_rebuild_") as temporary:
        written = {path.name: path for path in write_split_csvs(view, manifest, Path(temporary)).values()}
        for asset in assets:
            destination = safe_destination(root, asset["repository_relative_destination"])
            candidate = written.get(destination.name)
            if candidate is None or file_identity(candidate) != (asset["byte_size"], asset["sha256"]):
                raise ReproduceError(f"Derived file failed its hash check: {destination.name}")
            if asset_state(asset, root) == "missing":
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, destination)
                published += 1
    return published


def verify_assets(manifest: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Hash every declared asset without changing or downloading anything."""
    validate_manifest(manifest)
    states = {
        asset["asset_id"]: (asset, asset_state(asset, project_root))
        for asset in manifest["assets"]
    }
    mismatch = sorted(
        asset_id for asset_id, (_, state) in states.items() if state == "mismatch"
    )
    missing_required = sorted(
        asset_id for asset_id, (asset, state) in states.items()
        if state == "missing" and asset["required"]
    )
    return {
        "status": "FAIL" if mismatch or missing_required else "PASS",
        "asset_count": len(states),
        "verified_count": sum(state == "verified" for _, state in states.values()),
        "missing_optional_count": sum(
            state == "missing" and not asset["required"] for asset, state in states.values()
        ),
        "missing_required": missing_required,
        "mismatch": mismatch,
    }


def download_assets(
    manifest: Mapping[str, Any],
    project_root: Path,
    *,
    opener: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    deriver: Callable[[Path, Sequence[Mapping[str, Any]]], int] = derive_split_csvs,
) -> dict[str, Any]:
    """Acquire every asset, or report exactly which gate stops the route."""
    validate_manifest(manifest)
    root = project_root.resolve()
    assets = manifest["assets"]
    conflicts = sorted(a["asset_id"] for a in assets if asset_state(a, root) == "mismatch")
    if conflicts:
        return {"status": "FAIL", "error": "local files have wrong hashes", "mismatch": conflicts}

    result: dict[str, Any] = {"downloaded": [], "extracted_count": 0, "derived_count": 0}
    for asset in assets:
        if asset["distribution_classification"] == OFFICIAL_SOURCE:
            if download_file(
                asset["url"],
                safe_destination(root, asset["repository_relative_destination"]),
                byte_size=asset["byte_size"],
                sha256=asset["sha256"],
                opener=opener,
                sleep=sleep,
            ):
                result["downloaded"].append(asset["asset_id"])

    # Local copies never stand in for a public route, so this ignores on-disk state.
    blocked = unpublished_assets(assets)
    for bundle in manifest["bundles"]:
        members = [a for a in assets if a["bundle_id"] == bundle["bundle_id"]]
        if bundle["url"] is None or all(asset_state(a, root) != "missing" for a in members):
            continue
        bundle_path = safe_destination(root, f"{DOWNLOAD_DIRECTORY}/{bundle['filename']}")
        if download_file(
            bundle["url"],
            bundle_path,
            byte_size=bundle["byte_size"],
            sha256=bundle["sha256"],
            opener=opener,
            sleep=sleep,
        ):
            result["downloaded"].append(bundle["bundle_id"])
        result["extracted_count"] += extract_bundle(bundle_path, members, root)

    missing_tracked = sorted(
        a["asset_id"] for a in assets
        if a["distribution_classification"] == GIT_TRACKED and asset_state(a, root) == "missing"
    )
    if missing_tracked:
        return {**result, "status": "FAIL", "error": "git-tracked files are missing", "missing": missing_tracked}

    derived = [a for a in assets if a["distribution_classification"] == DERIVED_FROM_RAW]
    if any(asset_state(a, root) == "missing" for a in derived):
        prerequisites = [
            a for a in assets
            if a["repository_relative_destination"] in {RAW_DESTINATION, SPLIT_MANIFEST_DESTINATION}
        ]
        if all(asset_state(a, root) == "verified" for a in prerequisites) and len(prerequisites) == 2:
            result["derived_count"] = deriver(root, derived)

    if blocked:
        return {
            **result,
            "status": "BLOCKED",
            "gate": "G1",
            "error": "required assets have no public download route",
            "unpublished_required_count": len(blocked),
        }
    return {**result, **verify_assets(manifest, root)}
