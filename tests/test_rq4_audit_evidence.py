from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from src.utils.path import PROJECT_ROOT


MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "dataset_b"
    / "reproducibility"
    / "rq4_audit_evidence_manifest.json"
)
CORE_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "dataset_b"
    / "reproducibility"
    / "asset_manifest.json"
)
LINK_TEMPLATE = (
    PROJECT_ROOT
    / "config"
    / "dataset_b"
    / "reproducibility"
    / "rq4_audit_evidence_links.example.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scratch_is_ignored_without_entering_the_core_manifest() -> None:
    ignore_lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/scratch/**" in ignore_lines

    if (PROJECT_ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "check-ignore", "-q", "scratch/rq4b_implementation/example"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 0

    core = _load(CORE_MANIFEST)
    destinations = [asset["repository_relative_destination"] for asset in core["assets"]]
    assert not any(destination.startswith("scratch/") for destination in destinations)


def test_audit_manifest_is_optional_curated_and_self_consistent() -> None:
    manifest = _load(MANIFEST)
    assets = manifest["assets"]

    assert manifest["schema_version"] == 1
    assert manifest["distribution"] == "optional_external_audit_evidence"
    assert manifest["required_for_core_reproduction"] is False
    assert manifest["public_url"] is None
    assert manifest["file_count"] == len(assets) == 243
    assert manifest["uncompressed_byte_size"] == sum(asset["byte_size"] for asset in assets)
    assert len({asset["path"] for asset in assets}) == len(assets)

    for asset in assets:
        path = asset["path"]
        assert path.startswith(("scratch/rq4b_spike/", "scratch/rq4b_implementation/"))
        assert "/basetemp/" not in path
        assert "/pytest/" not in path
        assert len(asset["sha256"]) == 64

    links = _load(LINK_TEMPLATE)
    assert links == {manifest["filename"]: ""}


def test_retained_local_sources_match_the_audit_manifest_when_present() -> None:
    manifest = _load(MANIFEST)
    for asset in manifest["assets"]:
        source = PROJECT_ROOT / asset["path"]
        if not source.is_file():
            continue
        assert source.stat().st_size == asset["byte_size"]
        assert _sha256(source) == asset["sha256"]


def test_local_optional_bundle_matches_manifest_when_present() -> None:
    manifest = _load(MANIFEST)
    bundle = PROJECT_ROOT / "dist" / "reproducibility" / "evidence" / manifest["filename"]
    if not bundle.is_file():
        return

    assert bundle.stat().st_size == manifest["bundle_byte_size"]
    assert _sha256(bundle) == manifest["bundle_sha256"]

    expected = {asset["path"]: asset for asset in manifest["assets"]}
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == set(expected)
        for name, asset in expected.items():
            payload = archive.read(name)
            assert len(payload) == asset["byte_size"]
            assert hashlib.sha256(payload).hexdigest() == asset["sha256"]
