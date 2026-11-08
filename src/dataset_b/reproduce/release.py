"""Maintainer-only inventory, deterministic bundling, and release binding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping

from .assets import extract_bundle
from .manifest import (
    DERIVED_FROM_RAW,
    GIT_TRACKED,
    MANIFEST_RELATIVE_PATH,
    OFFICIAL_SOURCE,
    PROJECT_BUNDLE,
    SCHEMA_VERSION,
    ReproduceError,
    file_identity,
    validate_manifest,
)


REPOSITORY_URL = "https://github.com/DAP391m-Group12/DAP391m.git"
BUNDLE_DIRECTORY = "dist/reproducibility"
MAX_BUNDLE_BYTES = 1_900_000_000
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

OFFICIAL_RAW_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=NY&years=2024"
RAW_ASSET = "data/raw/state_NY_2024.csv"
DERIVED_SPLIT_ASSETS = (
    "data/dataset_b/split_v1/train.csv",
    "data/dataset_b/split_v1/validation.csv",
    "data/dataset_b/split_v1/test.csv",
)
CORE_ASSETS = {
    "artifacts/dataset_b/split_v1.json": "split_manifest",
    "artifacts/dataset_b/rq1_test_bootstrap_indices.npz": "bootstrap_indices",
    "artifacts/dataset_b/rq1_test_bootstrap_manifest.json": "bootstrap_manifest",
}
REFERENCE_REPORTS = (
    "reports/RQ1_Research_Metrics_Final.txt",
    "reports/RQ2_Research_Metrics_Final.txt",
    "reports/RQ3_RQ4a_RQ4b_Research_Metrics_Final.txt",
)
RQ1_RUNS = {
    "lr": "artifacts/dataset_b/lr/final/lr_final_15f_20261002_01",
    "rf": "artifacts/dataset_b/rf/final/rf_final_15f_20261002_01",
    "xgboost": "artifacts/dataset_b/xgboost/final/xgb_final_15f_20261003_01",
    "plain_mlp": "artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03",
    "realmlp": "artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01",
    "tabmmini": "artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01",
}
RQ_RUNS = {
    "rq2": {"artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01": "rq2_cpu_production_20261009_01"},
    "rq3": {"artifacts/dataset_b/rq3/rq3_fairness_20261010_01": "rq3_fairness_20261010_01"},
    "rq4a": {
        f"artifacts/dataset_b/rq4/rq4a/{run_id}": run_id
        for run_id in ("rq4a3_postproc_20261017_01", "rq4a6_postproc_20261017_01")
    },
    "rq4b": {
        f"artifacts/dataset_b/rq4/rq4b/{relative}": PurePosixPath(relative).name
        for relative in (
            "rq4b3_adv_20261019_01",
            "rq4b3_adv_20261019_02",
            "rq4b3_adv_20261021_01",
            "rq4b3_adv_20261022_01",
            "gate_c_aggregation/rq4b3_gatec_20261024_01",
            "test_evaluation/rq4b3_test_20261024_01",
            "sex_supplementary/rq4b3_sex_supplementary_20261024_01",
            "sex_supplementary/rq4b3_sex_supplementary_20261024_02",
        )
    },
}
# Retained on disk but not accepted evidence, so a reproduction does not need them.
NOT_ACCEPTED_RUNS = frozenset({"rq4b3_adv_20261019_01", "rq4b3_sex_supplementary_20261024_01"})
VOLATILE_RQ_METADATA = frozenset(
    {"artifacts/dataset_b/rq3/rq3_fairness_20261010_01/runtime.csv"}
)
CHECKPOINT_PINS = {
    "artifacts/dataset_b/lr/final/lr_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib":
        "69770bc11558f56f147e2af89f79747c903e74e779aeae17d1347a25de620859",
    "artifacts/dataset_b/rf/final/rf_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib":
        "2e2378596ba62f4c82b7d657d45cb733223612fb38943a5e121a818183b6c802",
    "artifacts/dataset_b/xgboost/final/xgb_final_15f_20261003_01/checkpoints/fitted_pipeline.joblib":
        "d24db9bb642e6e7810596e4a16632b1be88e68eeb030ad05130b134ea65ee834",
    "artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03/full_train/checkpoints/fitted_bundle.joblib":
        "bad3761eeb72f41b6cb2784530a1f46fce6b9bb2948c812f7d6e5e4337ae1968",
    "artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/full_train/fitted_model.joblib":
        "b5b79dcd3200f989ecd9cdae850fccaeb2f98ae0487046c68cebbdbcf8ea2754",
    "artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01/full_train/model/inference_bundle.pt":
        "0d3eacadb7a2e07ae5a6fb7396f87b00317a21711a0b0db8b6aaf4ec1e4d2229",
}
RAW_PIN = "e973343e8bf60c9ad9cefe9f7d0a9321b8e04813c8ec4379f746679b0ddc847d"
# Values pinned by the accepted run manifests.
LINEAGE_PINS = {
    "artifacts/dataset_b/split_v1.json":
        "66816ba47da64bfd85c010a90a14b9d80443a46d319258bf9fc1b3657f731596",
    "artifacts/dataset_b/rq1_test_bootstrap_indices.npz":
        "2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1",
    "artifacts/dataset_b/rq1_test_bootstrap_manifest.json":
        "5544af4ebe91bfdf8c5b3b574fa2ef24f3da8c14827f353970cbe9d759e9d21e",
}


def classify_role(scope: str, relative: PurePosixPath) -> str:
    """Map one retained file to its manifest role from its accepted-run path."""
    name = relative.name.lower()
    if name == "rq1_test_bootstrap_indices.npz":
        return "bootstrap_indices"
    if name == "rq1_test_bootstrap_manifest.json":
        return "bootstrap_manifest"
    if scope == "rq3" and name == "runtime.csv":
        return "reference_report"
    if scope in RQ_RUNS:
        return "reference_report" if "reporting" in relative.parts else "rq_input"
    if relative.suffix in {".joblib", ".pt"}:
        return "preprocessor" if "preprocessor" in name else "checkpoint"
    if name.startswith("threshold_"):
        return "threshold"
    if "predictions" in name and relative.suffix in {".csv", ".npz"}:
        return "reference_prediction"
    if name in {"rq1_test_metrics.json", "rq1_bootstrap_metrics.csv", "rq1_metrics_long.csv"}:
        return "reference_metric"
    return "run_lineage"


def _walk_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        raise ReproduceError(f"Accepted run root is missing: {directory.name}")
    for current, folders, files in os.walk(directory):
        folders.sort()
        for name in sorted(files):
            yield Path(current) / name


def _asset(
    *,
    asset_id: str,
    scope: str,
    destination: str,
    identity: tuple[int, str],
    required: bool = True,
    url: str | None = None,
    distribution: str = PROJECT_BUNDLE,
    accepted_run_id: str | None = None,
    role: str,
    hash_mode: str = "bytes",
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "model_or_rq_scope": scope,
        "url": url,
        "repository_relative_destination": destination,
        "byte_size": identity[0],
        "sha256": identity[1],
        "hash_mode": hash_mode,
        "required": required,
        "distribution_classification": distribution,
        "bundle_id": None,
        "accepted_run_id": accepted_run_id,
        "role": role,
    }


def _run_assets(source_root: Path, scope: str, run_root: str, run_id: str, *, required: bool) -> list[dict[str, Any]]:
    assets = []
    for path in _walk_files(source_root / run_root):
        relative = PurePosixPath(path.relative_to(source_root / run_root).as_posix())
        destination = f"{run_root}/{relative}"
        if destination in VOLATILE_RQ_METADATA:
            continue
        assets.append(
            _asset(
                asset_id=f"{run_id}:{relative}",
                scope=scope,
                destination=destination,
                identity=file_identity(path),
                required=required,
                accepted_run_id=run_id if required else None,
                role=classify_role(scope, relative),
            )
        )
    return assets


def source_provenance(source_root: Path, assets: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the retained source inventory without publishing a machine-local path."""
    inventory = [
        {
            "destination": asset["repository_relative_destination"],
            "byte_size": asset["byte_size"],
            "sha256": asset["sha256"],
            "hash_mode": asset["hash_mode"],
        }
        for asset in sorted(assets, key=lambda item: item["repository_relative_destination"])
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    revision = None
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        candidate = result.stdout.strip().lower()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", candidate):
            revision = candidate
    except OSError:
        pass
    return {
        "source_revision": revision,
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "excluded_volatile_paths": sorted(VOLATILE_RQ_METADATA),
    }


def build_inventory(source_root: Path, project_root: Path) -> list[dict[str, Any]]:
    """Enumerate the accepted evidence and check it against the recorded pins."""
    assets = [
        _asset(
            asset_id="hmda_ny_2024_raw",
            scope="data",
            destination=RAW_ASSET,
            identity=file_identity(source_root / RAW_ASSET),
            url=OFFICIAL_RAW_URL,
            distribution=OFFICIAL_SOURCE,
            role="raw_source",
        )
    ]
    for destination in DERIVED_SPLIT_ASSETS:
        assets.append(
            _asset(
                asset_id=f"split_v1:{PurePosixPath(destination).name}",
                scope="data",
                destination=destination,
                identity=file_identity(source_root / destination),
                distribution=DERIVED_FROM_RAW,
                role="derived_split",
            )
        )
    for destination, role in CORE_ASSETS.items():
        assets.append(
            _asset(
                asset_id=f"core:{PurePosixPath(destination).name}",
                scope="core",
                destination=destination,
                identity=file_identity(source_root / destination),
                role=role,
            )
        )
    for destination in REFERENCE_REPORTS:
        assets.append(
            _asset(
                asset_id=f"report:{PurePosixPath(destination).name}",
                scope="reports",
                destination=destination,
                identity=file_identity(project_root / destination, hash_mode="text_lf"),
                distribution=GIT_TRACKED,
                role="reference_report",
                hash_mode="text_lf",
            )
        )
    for scope, run_root in RQ1_RUNS.items():
        assets += _run_assets(source_root, scope, run_root, PurePosixPath(run_root).name, required=True)
    for scope, runs in RQ_RUNS.items():
        for run_root, run_id in runs.items():
            assets += _run_assets(source_root, scope, run_root, run_id, required=run_id not in NOT_ACCEPTED_RUNS)

    by_destination = {asset["repository_relative_destination"]: asset for asset in assets}
    pins = {RAW_ASSET: RAW_PIN, **LINEAGE_PINS, **CHECKPOINT_PINS}
    for destination, expected in pins.items():
        asset = by_destination.get(destination)
        if asset is None or asset["sha256"] != expected:
            raise ReproduceError(f"Accepted evidence does not match its recorded pin: {destination}")
    return assets


def _write_bundle(path: Path, members: list[dict[str, Any]], source_root: Path) -> None:
    temporary = path.with_name(path.name + ".part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for asset in sorted(members, key=lambda item: item["repository_relative_destination"]):
            destination = asset["repository_relative_destination"]
            info = zipfile.ZipInfo(destination, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            with (source_root / destination).open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
    os.replace(temporary, path)


def build_bundles(
    assets: list[dict[str, Any]],
    source_root: Path,
    output_directory: Path,
    *,
    max_bundle_bytes: int = MAX_BUNDLE_BYTES,
) -> list[dict[str, Any]]:
    """Write per-scope ZIP bundles and link each asset to its bundle."""
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[list[dict[str, Any]]]] = {}
    for asset in sorted(assets, key=lambda item: item["repository_relative_destination"]):
        if asset["distribution_classification"] != PROJECT_BUNDLE:
            continue
        if asset["byte_size"] > max_bundle_bytes:
            raise ReproduceError(f"Asset exceeds the bundle limit: {asset['asset_id']}")
        parts = grouped.setdefault(asset["model_or_rq_scope"], [[]])
        if sum(item["byte_size"] for item in parts[-1]) + asset["byte_size"] > max_bundle_bytes:
            parts.append([])
        parts[-1].append(asset)

    bundles = []
    for scope in sorted(grouped):
        for index, members in enumerate(grouped[scope], start=1):
            bundle_id = f"dataset_b_{scope}_part{index:02d}"
            path = output_directory / f"{bundle_id}.zip"
            _write_bundle(path, members, source_root)
            size, digest = file_identity(path)
            bundles.append(
                {"bundle_id": bundle_id, "filename": path.name, "url": None, "byte_size": size, "sha256": digest}
            )
            for asset in members:
                asset["bundle_id"] = bundle_id
    return bundles


def write_manifest(manifest: Mapping[str, Any], project_root: Path) -> Path:
    validate_manifest(manifest)
    path = project_root / MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def build_release(source_root: Path, project_root: Path) -> dict[str, Any]:
    """Create the bundles and an unpublished manifest from retained evidence."""
    source = source_root.resolve()
    project = project_root.resolve()
    assets = build_inventory(source, project)
    bundles = build_bundles(assets, source, project / BUNDLE_DIRECTORY)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "dataset_b",
        "source_provenance": source_provenance(source, assets),
        "bundles": bundles,
        "assets": assets,
    }
    write_manifest(manifest, project)
    return {
        "status": "PASS",
        "asset_count": len(assets),
        "required_count": sum(asset["required"] for asset in assets),
        "bundle_count": len(bundles),
        "bundle_total_bytes": sum(bundle["byte_size"] for bundle in bundles),
        "bundles": [{"filename": b["filename"], "byte_size": b["byte_size"]} for b in bundles],
        "longest_destination_length": max(len(a["repository_relative_destination"]) for a in assets),
    }


def bind_release(manifest: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Record a release location on every bundle and its assets."""
    if not base_url.startswith("https://"):
        raise ReproduceError("Release base URL must be https")
    urls = {}
    for bundle in manifest["bundles"]:
        bundle["url"] = f"{base_url.rstrip('/')}/{bundle['filename']}"
        urls[bundle["bundle_id"]] = bundle["url"]
    for asset in manifest["assets"]:
        if asset["distribution_classification"] == PROJECT_BUNDLE:
            asset["url"] = urls[asset["bundle_id"]]
    validate_manifest(manifest)
    return manifest


def install_local_bundles(manifest: Mapping[str, Any], project_root: Path, directory: Path) -> dict[str, Any]:
    """Place assets from verified local bundles; this is local evidence, not a public route."""
    extracted = 0
    for bundle in manifest["bundles"]:
        path = directory / bundle["filename"]
        if file_identity(path) != (bundle["byte_size"], bundle["sha256"]):
            raise ReproduceError(f"Local bundle failed its hash check: {bundle['filename']}")
        members = [asset for asset in manifest["assets"] if asset["bundle_id"] == bundle["bundle_id"]]
        extracted += extract_bundle(path, members, project_root)
    return {"status": "PASS", "extracted_count": extracted, "route": "local_bundles_not_public"}


def direct_download_url(url: str) -> str:
    """Turn a Google Drive share link into its direct-download form."""
    if not url.startswith("https://"):
        raise ReproduceError(f"Bundle link must be https: {url}")
    host = urllib.parse.urlsplit(url).netloc.lower()
    if host not in {"drive.google.com", "drive.usercontent.google.com"}:
        return url
    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", url) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if match is None:
        raise ReproduceError(f"Google Drive link has no file id: {url}")
    return f"https://drive.usercontent.google.com/download?id={match.group(1)}&export=download&confirm=t"


def bind_links(manifest: dict[str, Any], links: Mapping[str, str]) -> dict[str, Any]:
    """Record one download link per bundle filename; every bundle must be named."""
    filenames = {bundle["filename"] for bundle in manifest["bundles"]}
    if set(links) != filenames:
        raise ReproduceError(
            f"Links must name every bundle exactly: missing={sorted(filenames - set(links))} "
            f"unknown={sorted(set(links) - filenames)}"
        )
    urls = {}
    for bundle in manifest["bundles"]:
        bundle["url"] = direct_download_url(links[bundle["filename"]])
        urls[bundle["bundle_id"]] = bundle["url"]
    for asset in manifest["assets"]:
        if asset["distribution_classification"] == PROJECT_BUNDLE:
            asset["url"] = urls[asset["bundle_id"]]
    validate_manifest(manifest)
    return manifest


def check_publication(
    manifest: Mapping[str, Any],
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    repository_url: str = REPOSITORY_URL,
) -> dict[str, Any]:
    """Confirm the repository and every bundle are reachable without credentials."""
    failures: dict[str, str] = {}
    if not manifest["bundles"]:
        failures["bundles"] = "no bundles declared"
    try:
        with opener(urllib.request.Request(f"{repository_url}/info/refs?service=git-upload-pack"), timeout=60):
            pass
    except OSError as exc:
        failures["repository"] = str(exc)
    for bundle in manifest["bundles"]:
        if bundle["url"] is None:
            failures[bundle["bundle_id"]] = "no public url"
            continue
        request = urllib.request.Request(bundle["url"], headers={"Range": "bytes=0-0"})
        try:
            with opener(request, timeout=60) as response:
                total = response.headers.get("Content-Range", "").rpartition("/")[2]
                if str(bundle["byte_size"]) != total:
                    failures[bundle["bundle_id"]] = f"size mismatch: {total!r}"
        except OSError as exc:
            failures[bundle["bundle_id"]] = str(exc)
    return {"status": "FAIL" if failures else "PASS", "bundle_count": len(manifest["bundles"]), "failures": failures}
