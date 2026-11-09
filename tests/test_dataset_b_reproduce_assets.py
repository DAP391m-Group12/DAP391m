"""Synthetic unit tests for the Dataset B asset route (G1 scope only).

These fixtures are not production reproduction evidence.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import urllib.error
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from src.dataset_b.reproduce import cli
from src.dataset_b.reproduce.assets import (
    DOWNLOAD_DIRECTORY,
    download_assets,
    download_file,
    extract_bundle,
    verify_assets,
)
from src.dataset_b.reproduce.manifest import (
    MANIFEST_RELATIVE_PATH,
    ReproduceError,
    check_relative_path,
    file_identity,
    load_manifest,
    safe_destination,
    unpublished_assets,
    validate_manifest,
)
from src.dataset_b.reproduce.release import (
    bind_split_links,
    CHECKPOINT_PINS,
    LINEAGE_PINS,
    RAW_PIN,
    bind_links,
    bind_release,
    build_bundles,
    check_publication,
    classify_role,
    direct_download_url,
    source_provenance,
)


URL ="https://example.test/file.bin"
PAYLOAD = b"0123456789" * 100


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, status: int = 200, headers: dict | None = None, url: str = URL):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}
        self._url = url

    def geturl(self) -> str:
        return self._url


class FlakyStream(FakeResponse):
    def read(self, *args):
        chunk = super().read(*args)
        if not chunk:
            raise ConnectionResetError("dropped")
        return chunk


class RangeServer:
    """Serve bytes with Range support; scripted entries run before normal serving."""

    def __init__(self, content: bytes, script: list | None = None, honor_range: bool = True):
        self.content = content
        self.script = list(script or [])
        self.honor_range = honor_range
        self.ranges: list[str | None] = []

    def __call__(self, request, timeout=None):
        self.ranges.append(request.get_header("Range"))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        header = request.get_header("Range")
        if header and self.honor_range:
            start = int(header.split("=")[1].split("-")[0])
            total = len(self.content)
            return FakeResponse(
                self.content[start:],
                status=206,
                headers={"Content-Range": f"bytes {start}-{total - 1}/{total}"},
            )
        return FakeResponse(self.content)


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "error", {}, None)


def no_sleep(_seconds: float) -> None:
    return None


def fetch(destination: Path, opener, content: bytes = PAYLOAD, **kwargs) -> bool:
    return download_file(
        URL,
        destination,
        byte_size=len(content),
        sha256=sha(content),
        opener=opener,
        sleep=no_sleep,
        **kwargs,
    )


def make_asset(destination: str, content: bytes, **overrides) -> dict:
    asset = {
        "asset_id": destination,
        "model_or_rq_scope": "core",
        "url": None,
        "repository_relative_destination": destination,
        "byte_size": len(content),
        "sha256": sha(content),
        "hash_mode": "bytes",
        "required": True,
        "distribution_classification": "project_bundle",
        "bundle_id": "bundle_a",
        "accepted_run_id": None,
        "role": "split_manifest",
    }
    asset.update(overrides)
    return asset


def make_bundle(url: str | None = None, content: bytes = b"zip") -> dict:
    return {"bundle_id": "bundle_a", "filename": "bundle_a.zip", "url": url, "byte_size": len(content), "sha256": sha(content)}


def make_manifest(assets: list[dict], bundles: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "dataset": "dataset_b",
        "bundles": [make_bundle()] if bundles is None else bundles,
        "assets": assets,
    }


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def write_project_manifest(root: Path, manifest: dict) -> None:
    path = root / MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# Manifest schema


def test_valid_manifest_is_accepted() -> None:
    validate_manifest(make_manifest([make_asset("artifacts/a.json", b"a")]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "ABC"),
        ("byte_size", -1),
        ("byte_size", True),
        ("role", "unknown"),
        ("hash_mode", "crc"),
        ("required", "yes"),
        ("distribution_classification", "unknown"),
        ("url", "http://example.test/a"),
        ("repository_relative_destination", "../escape.json"),
        ("bundle_id", "missing_bundle"),
    ],
)
def test_manifest_rejects_invalid_field(field: str, value: object) -> None:
    asset = make_asset("artifacts/a.json", b"a")
    asset[field] = value
    with pytest.raises(ReproduceError):
        validate_manifest(make_manifest([asset]))


def test_manifest_rejects_missing_and_extra_fields() -> None:
    missing = make_asset("artifacts/a.json", b"a")
    del missing["role"]
    extra = make_asset("artifacts/b.json", b"b", note="x")
    for asset in (missing, extra):
        with pytest.raises(ReproduceError):
            validate_manifest(make_manifest([asset]))


def test_manifest_rejects_duplicates_including_case_folded_destination() -> None:
    first = make_asset("artifacts/a.json", b"a")
    with pytest.raises(ReproduceError, match="duplicate asset_id"):
        validate_manifest(make_manifest([first, copy.deepcopy(first)]))
    second = make_asset("artifacts/A.json", b"a", asset_id="other")
    with pytest.raises(ReproduceError, match="duplicate destination"):
        validate_manifest(make_manifest([first, second]))


def test_manifest_rejects_rq1_asset_without_run_identity() -> None:
    asset = make_asset("artifacts/a.json", b"a", model_or_rq_scope="lr")
    with pytest.raises(ReproduceError, match="accepted_run_id"):
        validate_manifest(make_manifest([asset]))


def test_manifest_rejects_official_source_without_url_and_url_mismatch() -> None:
    official = make_asset("data/raw/a.csv", b"a", distribution_classification="official_source", bundle_id=None)
    with pytest.raises(ReproduceError, match="official source"):
        validate_manifest(make_manifest([official]))
    mismatch = make_asset("artifacts/a.json", b"a", url="https://example.test/other.zip")
    with pytest.raises(ReproduceError, match="bundle url"):
        validate_manifest(make_manifest([mismatch]))


def test_direct_download_allows_a_pending_https_link_but_never_a_bundle() -> None:
    pending = make_asset(
        "data/dataset_b/split_v1/test.csv",
        b"test",
        distribution_classification="direct_download",
        bundle_id=None,
    )
    validate_manifest(make_manifest([pending], []))
    pending["url"] = URL
    validate_manifest(make_manifest([pending], []))
    pending["bundle_id"] = "bundle_a"
    with pytest.raises(ReproduceError, match="no bundle"):
        validate_manifest(make_manifest([pending]))


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "a/../../escape",
        "/absolute/path",
        "C:/private/path",
        "D:\\private\\path",
        "a\\b",
        "",
        " leading",
        "a//b/./c",
        ".git/config",
        ".GIT/config",
        "a/nul",
        "a/CON.txt",
        "trailing./file",
    ],
)
def test_unsafe_paths_are_rejected(relative: str) -> None:
    with pytest.raises(ReproduceError):
        safe_destination(Path("."), relative)


def test_safe_path_stays_inside_root(tmp_path: Path) -> None:
    assert check_relative_path("artifacts/a.json") == PurePosixPath("artifacts/a.json")
    assert safe_destination(tmp_path, "artifacts/a.json") == tmp_path.resolve() / "artifacts" / "a.json"


# Download retry and resume


def test_download_publishes_verified_file_and_leaves_no_part(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    assert fetch(destination, RangeServer(PAYLOAD)) is True
    assert destination.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.bin.part").exists()


def test_download_does_not_set_a_custom_user_agent(tmp_path: Path) -> None:
    seen: list[str | None] = []
    server = RangeServer(PAYLOAD)

    def opener(request, timeout=None):
        seen.append(request.get_header("User-agent"))
        return server(request, timeout=timeout)

    fetch(tmp_path / "file.bin", opener)
    assert seen == [None]


def test_download_resumes_after_dropped_connection(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    server = RangeServer(PAYLOAD, script=[FlakyStream(PAYLOAD[:400])])
    assert fetch(destination, server) is True
    assert server.ranges == [None, "bytes=400-"]
    assert destination.read_bytes() == PAYLOAD


def test_download_resumes_existing_partial_from_an_earlier_run(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:250])
    server = RangeServer(PAYLOAD)
    fetch(destination, server)
    assert server.ranges == ["bytes=250-"]
    assert destination.read_bytes() == PAYLOAD


def test_download_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:250])
    fetch(destination, RangeServer(PAYLOAD, honor_range=False))
    assert destination.read_bytes() == PAYLOAD


def test_download_discards_partial_after_416(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:250])
    server = RangeServer(PAYLOAD, script=[http_error(416)])
    fetch(destination, server)
    assert server.ranges == ["bytes=250-", None]
    assert destination.read_bytes() == PAYLOAD


def test_download_retries_transient_http_error(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD, script=[http_error(503), http_error(429)])
    assert fetch(tmp_path / "file.bin", server) is True
    assert len(server.ranges) == 3


def test_download_does_not_retry_not_found(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD, script=[http_error(404)])
    with pytest.raises(ReproduceError, match="HTTP 404"):
        fetch(tmp_path / "file.bin", server)
    assert len(server.ranges) == 1


def test_download_gives_up_after_retry_limit_and_keeps_partial(tmp_path: Path) -> None:
    server = RangeServer(PAYLOAD, script=[FlakyStream(PAYLOAD[:100])] + [OSError("down")] * 9)
    with pytest.raises(ReproduceError, match="after 3 attempts"):
        fetch(tmp_path / "file.bin", server, retries=3)
    assert not (tmp_path / "file.bin").exists()
    assert (tmp_path / "file.bin.part").read_bytes() == PAYLOAD[:100]


def test_checksum_mismatch_is_rejected_and_nothing_is_published(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    corrupt = b"x" * len(PAYLOAD)
    with pytest.raises(ReproduceError, match="checksum mismatch"):
        fetch(destination, RangeServer(corrupt))
    assert not destination.exists()
    assert not (tmp_path / "file.bin.part").exists()


def test_corrupt_download_recovers_when_a_retry_serves_good_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    server = RangeServer(PAYLOAD, script=[FakeResponse(b"x" * len(PAYLOAD))])
    assert fetch(destination, server) is True
    assert destination.read_bytes() == PAYLOAD


def test_oversized_partial_is_discarded(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD + b"extra")
    fetch(destination, RangeServer(PAYLOAD))
    assert destination.read_bytes() == PAYLOAD


def test_existing_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    destination.write_bytes(PAYLOAD)
    server = RangeServer(PAYLOAD)
    assert fetch(destination, server) is False
    assert server.ranges == []


def test_local_file_with_wrong_hash_is_not_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "file.bin"
    destination.write_bytes(b"user data")
    server = RangeServer(PAYLOAD)
    with pytest.raises(ReproduceError, match="Refusing to overwrite"):
        fetch(destination, server)
    assert destination.read_bytes() == b"user data"
    assert server.ranges == []


def test_plain_http_and_https_downgrade_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReproduceError, match="Only https"):
        download_file("http://example.test/a", tmp_path / "a", byte_size=1, sha256=sha(b"a"), opener=RangeServer(b"a"))
    downgrade = RangeServer(PAYLOAD, script=[FakeResponse(PAYLOAD, url="http://example.test/a")])
    with pytest.raises(ReproduceError, match="redirected"):
        fetch(tmp_path / "file.bin", downgrade)


# Bundle extraction


def test_extract_bundle_publishes_verified_members(tmp_path: Path) -> None:
    bundle = tmp_path / "b.zip"
    bundle.write_bytes(zip_bytes({"artifacts/a.json": b"a", "artifacts/b.json": b"b"}))
    assets = [make_asset("artifacts/a.json", b"a"), make_asset("artifacts/b.json", b"b")]
    assert extract_bundle(bundle, assets, tmp_path) == 2
    assert (tmp_path / "artifacts/a.json").read_bytes() == b"a"


@pytest.mark.parametrize("name", ["../evil.txt", "C:/evil.txt", "/evil.txt"])
def test_extract_rejects_unsafe_member_before_writing(tmp_path: Path, name: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    bundle = tmp_path / "b.zip"
    bundle.write_bytes(zip_bytes({"artifacts/a.json": b"a", name: b"evil"}))
    with pytest.raises(ReproduceError, match="Unsafe"):
        extract_bundle(bundle, [make_asset("artifacts/a.json", b"a")], root)
    assert list(root.iterdir()) == []


def test_extract_rejects_unexpected_and_absent_members(tmp_path: Path) -> None:
    bundle = tmp_path / "b.zip"
    bundle.write_bytes(zip_bytes({"artifacts/extra.json": b"x"}))
    with pytest.raises(ReproduceError, match="does not match the manifest"):
        extract_bundle(bundle, [make_asset("artifacts/a.json", b"a")], tmp_path)


def test_extract_rejects_member_with_wrong_hash(tmp_path: Path) -> None:
    bundle = tmp_path / "b.zip"
    bundle.write_bytes(zip_bytes({"artifacts/a.json": b"tampered"}))
    with pytest.raises(ReproduceError, match="failed its hash check"):
        extract_bundle(bundle, [make_asset("artifacts/a.json", b"a")], tmp_path)
    assert not (tmp_path / "artifacts/a.json").exists()
    assert not (tmp_path / "artifacts/a.json.part").exists()


# Download orchestration


def published_project(root: Path) -> tuple[dict, RangeServer]:
    content = zip_bytes({"artifacts/a.json": b"a"})
    url = "https://example.test/bundle_a.zip"
    manifest = make_manifest([make_asset("artifacts/a.json", b"a", url=url)], [make_bundle(url, content)])
    return manifest, RangeServer(content)


def test_download_assets_passes_with_published_bundle(tmp_path: Path) -> None:
    manifest, server = published_project(tmp_path)
    result = download_assets(manifest, tmp_path, opener=server, sleep=no_sleep)
    assert result["status"] == "PASS"
    assert result["extracted_count"] == 1
    assert (tmp_path / DOWNLOAD_DIRECTORY / "bundle_a.zip").is_file()
    assert verify_assets(manifest, tmp_path)["status"] == "PASS"


def test_download_assets_is_idempotent(tmp_path: Path) -> None:
    manifest, server = published_project(tmp_path)
    download_assets(manifest, tmp_path, opener=server, sleep=no_sleep)
    calls = len(server.ranges)
    assert download_assets(manifest, tmp_path, opener=server, sleep=no_sleep)["status"] == "PASS"
    assert len(server.ranges) == calls


def test_unpublished_required_asset_blocks_g1(tmp_path: Path) -> None:
    manifest = make_manifest([make_asset("artifacts/a.json", b"a")])
    assert unpublished_assets(manifest["assets"]) == ["artifacts/a.json"]
    result = download_assets(manifest, tmp_path, opener=RangeServer(b""), sleep=no_sleep)
    assert result["status"] == "BLOCKED"
    assert result["gate"] == "G1"
    assert result["unpublished_required_count"] == 1


def test_local_copy_does_not_replace_a_public_route(tmp_path: Path) -> None:
    manifest = make_manifest([make_asset("artifacts/a.json", b"a")])
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/a.json").write_bytes(b"a")
    result = download_assets(manifest, tmp_path, opener=RangeServer(b""), sleep=no_sleep)
    assert result["status"] == "BLOCKED"
    assert result["unpublished_required_count"] == 1


def test_unpublished_optional_asset_does_not_block(tmp_path: Path) -> None:
    manifest = make_manifest([make_asset("artifacts/a.json", b"a", required=False)])
    result = download_assets(manifest, tmp_path, opener=RangeServer(b""), sleep=no_sleep)
    assert result["status"] == "PASS"
    assert result["missing_optional_count"] == 1


def test_local_mismatch_stops_before_any_network_call(tmp_path: Path) -> None:
    manifest, server = published_project(tmp_path)
    target = tmp_path / "artifacts/a.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user edit")
    result = download_assets(manifest, tmp_path, opener=server, sleep=no_sleep)
    assert result["status"] == "FAIL"
    assert result["mismatch"] == ["artifacts/a.json"]
    assert server.ranges == []
    assert target.read_bytes() == b"user edit"


def test_missing_git_tracked_asset_fails(tmp_path: Path) -> None:
    asset = make_asset(
        "reports/a.txt", b"a", distribution_classification="git_tracked", bundle_id=None, hash_mode="text_lf"
    )
    result = download_assets(make_manifest([asset], []), tmp_path, opener=RangeServer(b""), sleep=no_sleep)
    assert result["status"] == "FAIL"
    assert result["missing"] == ["reports/a.txt"]


def test_direct_split_download_is_verified_without_fetching_optional_raw(tmp_path: Path) -> None:
    raw = make_asset(
        "data/raw/state_NY_2024.csv",
        b"raw",
        url="https://example.test/raw.csv",
        distribution_classification="official_source",
        bundle_id=None,
        required=False,
        role="raw_source",
    )
    split = make_asset(
        "data/dataset_b/split_v1/test.csv",
        b"test",
        url=URL,
        distribution_classification="direct_download",
        bundle_id=None,
        role="derived_split",
    )
    server = RangeServer(b"test")
    result = download_assets(make_manifest([raw, split], []), tmp_path, opener=server, sleep=no_sleep)
    assert result["status"] == "PASS"
    assert result["downloaded"] == [split["asset_id"]]
    assert not (tmp_path / "data/raw/state_NY_2024.csv").exists()
    assert (tmp_path / "data/dataset_b/split_v1/test.csv").read_bytes() == b"test"


def derived_manifest(*, split_url: str | None) -> tuple[dict, RangeServer, list[dict]]:
    raw = b"raw"
    split = b"split"
    derived = make_asset(
        "data/dataset_b/split_v1/test.csv", b"test", distribution_classification="derived_from_raw", bundle_id=None
    )
    content = zip_bytes({"artifacts/dataset_b/split_v1.json": split})
    manifest = make_manifest(
        [
            make_asset(
                "data/raw/state_NY_2024.csv",
                raw,
                url=URL,
                distribution_classification="official_source",
                bundle_id=None,
                role="raw_source",
            ),
            make_asset("artifacts/dataset_b/split_v1.json", split, url=split_url),
            derived,
        ],
        [make_bundle(split_url, content)],
    )
    return manifest, RangeServer(raw if split_url is None else content), [derived]


def test_derived_split_waits_for_prerequisites_and_reports_blocked(tmp_path: Path) -> None:
    manifest, server, _ = derived_manifest(split_url=None)
    calls: list[int] = []
    result = download_assets(
        manifest, tmp_path, opener=server, sleep=no_sleep, deriver=lambda *_: calls.append(1) or 0
    )
    assert result["status"] == "BLOCKED"
    assert calls == []
    assert (tmp_path / "data/raw/state_NY_2024.csv").read_bytes() == b"raw"


def test_derived_split_runs_after_prerequisites_verify(tmp_path: Path) -> None:
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/state_NY_2024.csv").write_bytes(b"raw")
    manifest, server, derived = derived_manifest(split_url="https://example.test/bundle_a.zip")

    def deriver(root: Path, assets: list[dict]) -> int:
        assert assets == derived
        target = root / assets[0]["repository_relative_destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test")
        return 1

    result = download_assets(manifest, tmp_path, opener=server, sleep=no_sleep, deriver=deriver)
    assert result["status"] == "PASS"
    assert result["derived_count"] == 1


# Verification


def test_verify_reports_missing_and_mismatch_without_changing_files(tmp_path: Path) -> None:
    good = make_asset("artifacts/good.json", b"good")
    bad = make_asset("artifacts/bad.json", b"expected")
    missing = make_asset("artifacts/missing.json", b"m")
    optional = make_asset("artifacts/optional.json", b"o", required=False)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/good.json").write_bytes(b"good")
    (tmp_path / "artifacts/bad.json").write_bytes(b"actual")
    result = verify_assets(make_manifest([good, bad, missing, optional]), tmp_path)
    assert result["status"] == "FAIL"
    assert result["verified_count"] == 1
    assert result["mismatch"] == ["artifacts/bad.json"]
    assert result["missing_required"] == ["artifacts/missing.json"]
    assert result["missing_optional_count"] == 1
    assert (tmp_path / "artifacts/bad.json").read_bytes() == b"actual"


def test_text_lf_identity_ignores_checkout_line_endings(tmp_path: Path) -> None:
    (tmp_path / "lf.txt").write_bytes(b"a\nb\n")
    (tmp_path / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    assert file_identity(tmp_path / "lf.txt", hash_mode="text_lf") == file_identity(
        tmp_path / "crlf.txt", hash_mode="text_lf"
    )
    assert file_identity(tmp_path / "lf.txt") != file_identity(tmp_path / "crlf.txt")


# Release building


@pytest.mark.parametrize(
    ("scope", "relative", "role"),
    [
        ("lr", "checkpoints/fitted_pipeline.joblib", "checkpoint"),
        ("lr", "checkpoints/fitted_preprocessor.joblib", "preprocessor"),
        ("tabmmini", "full_train/model/inference_bundle.pt", "checkpoint"),
        ("tabmmini", "full_train/model/preprocessor.joblib", "preprocessor"),
        ("lr", "validation/threshold_selection.json", "threshold"),
        ("plain_mlp", "validation/threshold_selection/threshold_manifest.json", "threshold"),
        ("lr", "test/test_predictions.csv", "reference_prediction"),
        ("lr", "test/rq1_test_metrics.json", "reference_metric"),
        ("lr", "test/rq1_bootstrap_metrics.csv", "reference_metric"),
        ("lr", "shared_bootstrap/rq1_test_bootstrap_indices.npz", "bootstrap_indices"),
        ("lr", "shared_bootstrap/rq1_test_bootstrap_manifest.json", "bootstrap_manifest"),
        ("lr", "manifest.json", "run_lineage"),
        ("rq3", "group_metrics.csv", "rq_input"),
        ("rq3", "runtime.csv", "reference_report"),
        ("rq4a", "reporting/summary.csv", "reference_report"),
    ],
)
def test_role_classification(scope: str, relative: str, role: str) -> None:
    assert classify_role(scope, PurePosixPath(relative)) == role


def test_release_source_provenance_is_path_free_and_inventory_sensitive(tmp_path: Path) -> None:
    first = [make_asset("artifacts/a.json", b"a", bundle_id=None)]
    second = [make_asset("artifacts/a.json", b"changed", bundle_id=None)]

    provenance = source_provenance(tmp_path, first)

    assert provenance["source_revision"] is None
    assert len(provenance["inventory_sha256"]) == 64
    assert provenance["inventory_sha256"] != source_provenance(tmp_path, second)["inventory_sha256"]
    assert provenance["excluded_volatile_paths"] == [
        "artifacts/dataset_b/rq3/rq3_fairness_20261010_01/runtime.csv"
    ]


def bundle_fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    source = tmp_path / "source"
    assets = []
    for name, content in {"artifacts/a.bin": b"a" * 40, "artifacts/b.bin": b"b" * 40, "artifacts/c.bin": b"c" * 40}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        assets.append(make_asset(name, content, bundle_id=None))
    return source, assets


def test_bundles_are_byte_deterministic_and_extractable(tmp_path: Path) -> None:
    source, assets = bundle_fixture(tmp_path)
    first = build_bundles(copy.deepcopy(assets), source, tmp_path / "out1")
    second = build_bundles(list(reversed(copy.deepcopy(assets))), source, tmp_path / "out2")
    assert [bundle["sha256"] for bundle in first] == [bundle["sha256"] for bundle in second]
    linked = copy.deepcopy(assets)
    build_bundles(linked, source, tmp_path / "out3")
    root = tmp_path / "root"
    root.mkdir()
    assert extract_bundle(tmp_path / "out3" / first[0]["filename"], linked, root) == 3


def test_bundles_split_at_the_size_limit(tmp_path: Path) -> None:
    source, assets = bundle_fixture(tmp_path)
    bundles = build_bundles(assets, source, tmp_path / "out", max_bundle_bytes=80)
    assert [bundle["bundle_id"] for bundle in bundles] == ["dataset_b_core_part01", "dataset_b_core_part02"]
    assert {asset["bundle_id"] for asset in assets} == {bundle["bundle_id"] for bundle in bundles}


def test_single_asset_over_limit_is_rejected(tmp_path: Path) -> None:
    source, assets = bundle_fixture(tmp_path)
    with pytest.raises(ReproduceError, match="exceeds the bundle limit"):
        build_bundles(assets, source, tmp_path / "out", max_bundle_bytes=10)


def test_bind_release_sets_every_url_and_rejects_http() -> None:
    manifest = make_manifest([make_asset("artifacts/a.json", b"a")])
    with pytest.raises(ReproduceError, match="https"):
        bind_release(copy.deepcopy(manifest), "http://example.test/release")
    bound = bind_release(manifest, "https://example.test/release/")
    assert bound["bundles"][0]["url"] == "https://example.test/release/bundle_a.zip"
    assert bound["assets"][0]["url"] == bound["bundles"][0]["url"]
    assert unpublished_assets(bound["assets"]) == []


def test_html_page_never_enters_a_download(tmp_path: Path) -> None:
    page = FakeResponse(b"<html>scan warning</html>", headers={"Content-Type": "text/html; charset=utf-8"})
    with pytest.raises(ReproduceError, match="HTML page"):
        fetch(tmp_path / "file.bin", RangeServer(PAYLOAD, script=[page]))
    assert not (tmp_path / "file.bin").exists()
    assert not (tmp_path / "file.bin.part").exists()


@pytest.mark.parametrize(
    "link",
    [
        "https://drive.google.com/file/d/AbC_1-x/view?usp=sharing",
        "https://drive.google.com/open?id=AbC_1-x",
        "https://drive.google.com/uc?export=download&id=AbC_1-x",
    ],
)
def test_drive_share_links_become_direct_downloads(link: str) -> None:
    assert direct_download_url(link) == (
        "https://drive.usercontent.google.com/download?id=AbC_1-x&export=download&confirm=t"
    )


def test_other_links_are_kept_and_unusable_links_are_rejected() -> None:
    assert direct_download_url("https://example.test/a.zip") == "https://example.test/a.zip"
    for bad in ("http://example.test/a.zip", "https://drive.google.com/drive/folders"):
        with pytest.raises(ReproduceError):
            direct_download_url(bad)


def test_bind_links_requires_every_bundle_exactly() -> None:
    manifest = make_manifest([make_asset("artifacts/a.json", b"a")])
    for links in ({}, {"bundle_a.zip": URL, "other.zip": URL}):
        with pytest.raises(ReproduceError, match="every bundle"):
            bind_links(copy.deepcopy(manifest), links)
    bound = bind_links(manifest, {"bundle_a.zip": "https://drive.google.com/file/d/AbC/view"})
    assert bound["assets"][0]["url"] == bound["bundles"][0]["url"]
    assert unpublished_assets(bound["assets"]) == []


def test_bind_split_links_requires_all_three_locked_csv_names() -> None:
    assets = [
        make_asset(
            f"data/dataset_b/split_v1/{name}",
            name.encode(),
            distribution_classification="direct_download",
            bundle_id=None,
            role="derived_split",
        )
        for name in ("train.csv", "validation.csv", "test.csv")
    ]
    manifest = make_manifest(assets, [])
    links = {name: f"https://example.test/{name}" for name in ("train.csv", "validation.csv", "test.csv")}
    bound = bind_split_links(manifest, links)
    assert {asset["url"] for asset in bound["assets"]} == set(links.values())
    assert unpublished_assets(bound["assets"]) == []
    with pytest.raises(ReproduceError, match="exactly"):
        bind_split_links(manifest, {"train.csv": links["train.csv"]})


def test_check_publication_requires_url_declared_size_and_public_repository() -> None:
    good = lambda *_a, **_k: FakeResponse(b"z", 206, {"Content-Range": "bytes 0-0/3"})
    wrong = lambda *_a, **_k: FakeResponse(b"z", 206, {"Content-Range": "bytes 0-0/9"})
    unbound = make_manifest([make_asset("artifacts/a.json", b"a")])
    assert check_publication(unbound, opener=good)["status"] == "FAIL"
    url = "https://example.test/bundle_a.zip"
    manifest = make_manifest([make_asset("artifacts/a.json", b"a", url=url)], [make_bundle(url, b"zip")])
    assert check_publication(manifest, opener=good)["status"] == "PASS"
    assert check_publication(manifest, opener=wrong)["status"] == "FAIL"

    def denied(*_a, **_k):
        raise http_error(404)

    assert check_publication(manifest, opener=denied)["status"] == "FAIL"

    def private_repository(request, timeout=None):
        if "info/refs" in request.full_url:
            raise http_error(401)
        return good()

    result = check_publication(manifest, opener=private_repository)
    assert result["status"] == "FAIL"
    assert list(result["failures"]) == ["repository"]


def test_check_publication_rejects_an_empty_bundle_list() -> None:
    manifest = make_manifest([make_asset("reports/a.txt", b"a", distribution_classification="git_tracked", bundle_id=None)], [])
    result = check_publication(manifest, opener=lambda *_a, **_k: FakeResponse(b""))
    assert result["failures"] == {"bundles": "no bundles declared"}


def test_check_publication_checks_direct_asset_identity() -> None:
    url = "https://example.test/train.csv"
    direct = make_asset(
        "data/dataset_b/split_v1/train.csv",
        b"train",
        url=url,
        distribution_classification="direct_download",
        bundle_id=None,
    )
    manifest = make_manifest([direct], [])

    def opener(request, timeout=None):
        if "info/refs" in request.full_url:
            return FakeResponse(b"")
        return FakeResponse(b"train", headers={"Content-Type": "text/csv"})

    result = check_publication(manifest, opener=opener)
    assert result["direct_asset_count"] == 1
    assert result["failures"] == {"bundles": "no bundles declared"}


def test_check_publication_rejects_wrong_direct_asset_identity() -> None:
    direct = make_asset(
        "data/dataset_b/split_v1/train.csv",
        b"train",
        url="https://example.test/train.csv",
        distribution_classification="direct_download",
        bundle_id=None,
    )
    manifest = make_manifest([direct], [make_bundle("https://example.test/bundle.zip", b"zip")])

    def opener(request, timeout=None):
        if "info/refs" in request.full_url:
            return FakeResponse(b"")
        if request.headers.get("Range"):
            return FakeResponse(b"z", 206, {"Content-Range": "bytes 0-0/3"})
        return FakeResponse(b"wrong", headers={"Content-Type": "text/csv"})

    result = check_publication(manifest, opener=opener)
    assert result["status"] == "FAIL"
    assert direct["asset_id"] in result["failures"]


# Committed manifest


def test_committed_manifest_is_valid_and_covers_the_pinned_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root)
    by_destination = {a["repository_relative_destination"]: a for a in manifest["assets"]}
    assert "artifacts/dataset_b/rq3/rq3_fairness_20261010_01/runtime.csv" not in by_destination
    assert manifest["source_provenance"]["excluded_volatile_paths"] == [
        "artifacts/dataset_b/rq3/rq3_fairness_20261010_01/runtime.csv"
    ]
    raw = by_destination["data/raw/state_NY_2024.csv"]
    # RQ3 reads derived_race from the locked raw file, so the clean-clone route needs it.
    raw_url = "https://drive.usercontent.google.com/download?id=18UuN7nm7TCoMm8FxPQytl5cFb3QgFKbg&export=download&confirm=t"
    assert (raw["url"], raw["sha256"]) == (raw_url, RAW_PIN)
    assert raw["distribution_classification"] == "direct_download"
    assert raw["required"] is True
    split_urls = {
        "train.csv": "https://drive.usercontent.google.com/download?id=1VDDEO3uSQUq7dWj-3ir6_g4bmGVoposo&export=download&confirm=t",
        "validation.csv": "https://drive.usercontent.google.com/download?id=1t_qV893lbcrdlFDb22mCvadvNwGXzWTV&export=download&confirm=t",
        "test.csv": "https://drive.usercontent.google.com/download?id=16z2rvUUoRzJx6n3wB61fiqeMjSDJfDGX&export=download&confirm=t",
    }
    for filename in ("train.csv", "validation.csv", "test.csv"):
        split = by_destination[f"data/dataset_b/split_v1/{filename}"]
        assert split["distribution_classification"] == "direct_download"
        assert split["url"] == split_urls[filename]
    for destination, pin in {**LINEAGE_PINS, **CHECKPOINT_PINS}.items():
        assert by_destination[destination]["sha256"] == pin
        assert by_destination[destination]["required"] is True
    roles ={asset["role"] for asset in manifest["assets"]}
    assert {
        "raw_source",
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
    } <= roles
    scopes = {asset["model_or_rq_scope"] for asset in manifest["assets"]}
    assert {"lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini", "rq2", "rq3", "rq4a", "rq4b"} <= scopes
    for asset in manifest["assets"]:
        if not asset["required"]:
            assert asset["accepted_run_id"] is None
    serialized = json.dumps(manifest).lower()
    assert "d:\\\\" not in serialized and "c:\\\\users" not in serialized


# CLI


def test_cli_exit_codes_for_pass_fail_and_blocked(tmp_path: Path, capsys) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/a.json").write_bytes(b"a")
    write_project_manifest(tmp_path, make_manifest([make_asset("artifacts/a.json", b"a")]))
    assert cli.main(["verify-assets"], project_root=tmp_path) == cli.EXIT_PASS
    (tmp_path / "artifacts/a.json").write_bytes(b"tampered")
    assert cli.main(["verify-assets"], project_root=tmp_path) == cli.EXIT_FAIL
    (tmp_path / "artifacts/a.json").unlink()
    assert cli.main(["download"], project_root=tmp_path) == cli.EXIT_BLOCKED


def test_cli_missing_manifest_fails_with_receipt(tmp_path: Path, capsys) -> None:
    assert cli.main(["verify-assets"], project_root=tmp_path) == cli.EXIT_FAIL
    assert len(list((tmp_path / cli.RECEIPT_DIRECTORY).glob("verify-assets_*.json"))) == 1


def test_all_stops_at_first_non_pass_stage(tmp_path: Path, capsys) -> None:
    write_project_manifest(tmp_path, make_manifest([make_asset("artifacts/a.json", b"a")]))
    assert cli.main(["all"], project_root=tmp_path) == cli.EXIT_BLOCKED
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped_at"] == "download"
    assert list(payload["stages"]) == ["download"]


def test_all_cannot_pass_while_later_gates_are_unimplemented(tmp_path: Path, capsys) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/a.json").write_bytes(b"a")
    url = "https://example.test/bundle_a.zip"
    manifest = make_manifest([make_asset("artifacts/a.json", b"a", url=url)], [make_bundle(url)])
    write_project_manifest(tmp_path, manifest)
    assert cli.main(["all"], project_root=tmp_path) == cli.EXIT_FAIL
    assert "Environment profiles do not exist" in json.loads(capsys.readouterr().err)["error"]


def test_receipt_records_manifest_identity_without_private_paths(tmp_path: Path, capsys) -> None:
    write_project_manifest(tmp_path, make_manifest([make_asset("artifacts/a.json", b"a")]))
    cli.main(["verify-assets"], project_root=tmp_path)
    receipt_name = json.loads(capsys.readouterr().out)["receipt"]
    receipt = (tmp_path / receipt_name).read_text(encoding="utf-8")
    assert json.loads(receipt)["manifest_sha256"] == sha((tmp_path / MANIFEST_RELATIVE_PATH).read_bytes())
    assert str(tmp_path) not in receipt


def test_receipt_does_not_embed_paths_from_os_errors(tmp_path: Path, capsys, monkeypatch) -> None:
    def denied(_root):
        raise PermissionError(13, "Permission denied", str(tmp_path / "private.bin"))

    monkeypatch.setattr(cli, "load_manifest", denied)
    assert cli.main(["verify-assets"], project_root=tmp_path) == cli.EXIT_FAIL
    receipt = next((tmp_path / cli.RECEIPT_DIRECTORY).glob("verify-assets_*.json")).read_text(encoding="utf-8")
    assert "Permission denied" in receipt
    assert "private.bin" not in receipt


def test_corrupt_bundle_fails_with_receipt(tmp_path: Path, capsys, monkeypatch) -> None:
    def corrupt(_root):
        raise zipfile.BadZipFile("File is not a zip file")

    monkeypatch.setattr(cli, "load_manifest", corrupt)
    assert cli.main(["download"], project_root=tmp_path) == cli.EXIT_FAIL
    assert len(list((tmp_path / cli.RECEIPT_DIRECTORY).glob("download_*.json"))) == 1


def test_cli_rejects_unknown_model(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", "--model", "unknown"], project_root=tmp_path)
    assert error.value.code == 2
