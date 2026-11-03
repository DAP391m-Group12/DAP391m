"""Synthetic unit tests for environment, evaluation dispatch, and comparison.

These fixtures are not production reproduction evidence.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from src.dataset_b.reproduce import cli
from src.dataset_b.reproduce.compare import (
    compare_bootstrap,
    compare_intervals,
    compare_model,
    compare_point_metrics,
    compare_predictions,
    load_policy,
    slope_match,
    values_match,
)
from src.dataset_b.reproduce.environment import (
    PROFILE_DIRECTORY,
    assess_fingerprint,
    check_environments,
    load_profiles,
    locked_versions,
)
from src.dataset_b.reproduce.evaluate import evaluate_models, scope_problems
from src.dataset_b.reproduce.manifest import MANIFEST_RELATIVE_PATH, ReproduceError
from src.dataset_b.reproduce.spec import (
    METRIC_NAMES,
    MODELS,
    OUTPUT_DIRECTORY,
    check_record_order,
    check_run_identity,
    check_split_digest,
    check_threshold,
)
from tests.test_dataset_b_reproduce_assets import make_asset, make_bundle, make_manifest, sha


REPO = Path(__file__).resolve().parents[1]
POLICY = {
    "point_atol": 1e-12,
    "bootstrap_atol": 1e-14,
    "ci_atol": 1e-14,
    "calibration_slope_decimals": 6,
    "prediction_atol": 1e-12,
}
SPEC = MODELS["logistic_regression"]


def statuses(rows) -> set[str]:
    return {row["status"] for row in rows}


# Model dispatch and accepted identity


def test_dispatch_covers_six_models_with_accepted_runs_and_profiles() -> None:
    assert set(MODELS) == {"logistic_regression", "random_forest", "xgboost", "plain_mlp", "realmlp", "tabmmini"}
    assert {key: spec.requires_cuda for key, spec in MODELS.items()} == {
        "logistic_regression": False,
        "random_forest": False,
        "xgboost": False,
        "plain_mlp": True,
        "realmlp": True,
        "tabmmini": True,
    }
    profiles = load_profiles(REPO)
    for key, spec in MODELS.items():
        assert key in profiles[spec.profile]["models"]
        assert spec.run_id in spec.run_root
        assert (profiles[spec.profile]["compute"] == "cuda") is spec.requires_cuda


def test_committed_manifest_contains_every_primary_checkpoint() -> None:
    assets = json.loads((REPO / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))["assets"]
    destinations = {asset["repository_relative_destination"]: asset for asset in assets}
    for spec in MODELS.values():
        asset = destinations[spec.path(spec.checkpoint)]
        assert asset["role"] == "checkpoint" and asset["accepted_run_id"] == spec.run_id
        assert destinations[spec.path(spec.threshold_file)]["role"] == "threshold"


def test_run_identity_mismatch_is_rejected() -> None:
    check_run_identity(SPEC, {"run_id": SPEC.run_id})
    with pytest.raises(ReproduceError, match="run identity"):
        check_run_identity(SPEC, {"run_id": SPEC.run_id}, {"run_id": "other_run"})


@pytest.mark.parametrize(
    ("selected", "stored", "message"),
    [
        (0.493, 0.5, "threshold differs"),
        (None, None, "invalid"),
        (True, True, "invalid"),
        (0.0, 0.0, "invalid"),
        (1.0, 1.0, "invalid"),
    ],
)
def test_threshold_mismatch_is_rejected(selected, stored, message: str) -> None:
    with pytest.raises(ReproduceError, match=message):
        check_threshold(SPEC, {"selected_threshold": selected}, {"threshold": stored})


def test_matching_threshold_is_returned() -> None:
    assert check_threshold(SPEC, {"selected_threshold": 0.493}, {"threshold": 0.493}) == 0.493


@pytest.mark.parametrize(
    ("observed", "expected", "message"),
    [
        (["a", "b"], ["b", "a"], "order differs"),
        (["a"], ["a", "b"], "row count"),
        (["a", "a"], ["a", "a"], "not unique"),
    ],
)
def test_record_order_mismatch_is_rejected(observed, expected, message: str) -> None:
    with pytest.raises(ReproduceError, match=message):
        check_record_order(SPEC, observed, expected)


def test_split_mismatch_is_rejected() -> None:
    check_split_digest(SPEC, "abc", {"test_record_id_sha256": "abc"})
    with pytest.raises(ReproduceError, match="split differs"):
        check_split_digest(SPEC, "abc", {"test_record_id_sha256": "def"})


# Environment and CUDA enforcement


def write_profile(root: Path, compute: str = "cuda", pins: str = "numpy==2.5.2\n") -> dict:
    directory = root / PROFILE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "demo.txt").write_text(f"--extra-index-url https://example.test\n{pins}", encoding="utf-8")
    profile = {
        "python": "3.12.13",
        "requirements": "demo.txt",
        "compute": compute,
        "models": ["plain_mlp"],
        "smoke_imports": ["module_a"],
        "version_evidence": "synthetic",
    }
    (directory / "profiles.json").write_text(json.dumps({"profiles": {"demo": profile}}), encoding="utf-8")
    return profile


def good_fingerprint(**torch_overrides) -> dict:
    torch_state = {"cuda_available": True, "finite_cuda_operation": True, **torch_overrides}
    return {"python": "3.12.13", "packages": {"NumPy": "2.5.2"}, "import_errors": {}, "torch": torch_state}


def test_good_environment_has_no_problems(tmp_path: Path) -> None:
    assert assess_fingerprint(tmp_path, "demo", write_profile(tmp_path), good_fingerprint()) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"python": "3.13.0"}, "Python"),
        ({"packages": {"numpy": "1.0"}}, "numpy is 1.0"),
        ({"packages": {}}, "numpy is None"),
        ({"import_errors": {"module_a": "ImportError: x"}}, "import of module_a"),
    ],
)
def test_environment_differences_are_reported(tmp_path: Path, change: dict, message: str) -> None:
    errors = assess_fingerprint(tmp_path, "demo", write_profile(tmp_path), {**good_fingerprint(), **change})
    assert any(message in error for error in errors)


def test_cuda_profile_fails_without_cuda(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    errors = assess_fingerprint(tmp_path, "demo", profile, good_fingerprint(cuda_available=False))
    assert errors == ["demo: CUDA is required and not available"]
    errors = assess_fingerprint(tmp_path, "demo", profile, good_fingerprint(finite_cuda_operation=False))
    assert errors == ["demo: finite CUDA tensor operation failed"]


def test_cpu_profile_does_not_require_cuda(tmp_path: Path) -> None:
    profile = write_profile(tmp_path, compute="cpu")
    assert assess_fingerprint(tmp_path, "demo", profile, good_fingerprint(cuda_available=False)) == []


def test_unpinned_requirement_is_rejected(tmp_path: Path) -> None:
    profile = write_profile(tmp_path, pins="numpy>=2\n")
    with pytest.raises(ReproduceError, match="exact version"):
        locked_versions(tmp_path, profile)


def test_committed_profiles_pin_every_requirement() -> None:
    profiles = load_profiles(REPO)
    assert set(profiles) >= {"classical_cpu", "plain_mlp_cuda", "realmlp_cuda", "tabmmini_cuda"}
    for profile in profiles.values():
        assert locked_versions(REPO, profile)


def test_missing_environment_fails(tmp_path: Path) -> None:
    write_profile(tmp_path)
    result = check_environments(tmp_path, tmp_path / "envs")
    assert result["status"] == "FAIL"
    assert result["errors"] == ["demo: environment is not installed"]


# Evaluation preconditions


def model_manifest(root: Path, *, tamper: bool = False) -> dict:
    url = "https://example.test/b.zip"
    asset = make_asset(SPEC.path(SPEC.checkpoint), b"model", url=url, model_or_rq_scope=SPEC.scope, accepted_run_id=SPEC.run_id)
    other = make_asset("artifacts/other.bin", b"other", url=url, model_or_rq_scope="rf", accepted_run_id="rf_run")
    target = root / asset["repository_relative_destination"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tampered" if tamper else b"model")
    manifest = make_manifest([asset, other], [make_bundle(url)])
    path = root / MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def fake_python(root: Path, profile: str) -> None:
    from src.dataset_b.reproduce.environment import python_path

    path = python_path(root, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_scope_check_covers_only_required_assets_of_that_model(tmp_path: Path) -> None:
    manifest = model_manifest(tmp_path)
    assert scope_problems(manifest, tmp_path, SPEC) == []
    assert scope_problems(manifest, tmp_path, MODELS["random_forest"]) == ["artifacts/other.bin: missing"]


def test_wrong_checkpoint_hash_stops_before_any_worker(tmp_path: Path) -> None:
    model_manifest(tmp_path, tamper=True)
    fake_python(tmp_path / "envs", SPEC.profile)
    calls: list[str] = []
    result = evaluate_models(tmp_path, tmp_path / "envs", SPEC.key, worker=lambda *a: calls.append("x") or 0)
    assert result["status"] == "FAIL"
    assert calls == []
    assert "mismatch" in result["models"][SPEC.key]["first_problems"][0]


def test_missing_environment_stops_before_any_worker(tmp_path: Path) -> None:
    model_manifest(tmp_path)
    calls: list[str] = []
    result = evaluate_models(tmp_path, tmp_path / "envs", SPEC.key, worker=lambda *a: calls.append("x") or 0)
    assert result["status"] == "FAIL"
    assert calls == []


def test_worker_exit_code_decides_status_and_no_fallback_is_used(tmp_path: Path) -> None:
    model_manifest(tmp_path)
    fake_python(tmp_path / "envs", SPEC.profile)
    assert evaluate_models(tmp_path, tmp_path / "envs", SPEC.key, worker=lambda *a: 1)["status"] == "FAIL"
    assert evaluate_models(tmp_path, tmp_path / "envs", SPEC.key, worker=lambda *a: 0)["status"] == "PASS"


def test_worker_source_has_no_cpu_fallback_or_training_call() -> None:
    source = (REPO / "src/dataset_b/reproduce/model_worker.py").read_text(encoding="utf-8")
    for forbidden in (".fit(", "fit_transform(", 'map_location="cpu"', "partial_fit"):
        assert forbidden not in source


# Metric tolerance boundaries


def test_values_match_at_the_boundary() -> None:
    assert values_match(1.0, 1.0 + 1e-12, 1e-11)[0] is True
    assert values_match(0.0, 1e-12, 1e-12)[0] is True
    assert values_match(0.0, 2e-12, 1e-12)[0] is False


def test_non_finite_and_missing_values_match_only_the_same_state() -> None:
    assert values_match(math.nan, math.nan, 1e-12)[0] is True
    assert values_match(math.nan, 0.5, 1e-12)[0] is False
    assert values_match(None, None, 1e-12)[0] is True
    assert values_match(None, 0.5, 1e-12)[0] is False


def test_calibration_slope_uses_six_decimal_rounding() -> None:
    assert slope_match(1.2345671, 1.2345674, 6)[0] is True
    assert slope_match(1.234567, 1.234569, 6)[0] is False
    assert slope_match(math.nan, math.nan, 6)[0] is True
    assert slope_match(math.nan, 1.0, 6)[0] is False


def points(value: float = 0.5) -> dict:
    return {name: value for name in METRIC_NAMES}


def test_missing_point_metric_fails() -> None:
    observed = points()
    del observed["balanced_accuracy"]
    rows = list(compare_point_metrics("m", points(), observed, POLICY, "s"))
    failed = [row["metric"] for row in rows if row["status"] == "FAIL"]
    assert failed == ["point:balanced_accuracy"]


def test_point_metric_outside_tolerance_fails() -> None:
    assert statuses(compare_point_metrics("m", points(), points(0.5 + 1e-9), POLICY, "s")) == {"FAIL"}
    assert statuses(compare_point_metrics("m", points(), points(0.5), POLICY, "s")) == {"PASS"}


def interval(lower: float = 0.1, valid: int = 1000) -> dict:
    return {"lower_2_5": lower, "upper_97_5": 0.9, "valid_replicates": valid, "invalid_replicates": 1000 - valid}


def test_interval_endpoint_and_validity_counts_are_checked() -> None:
    expected = {name: interval() for name in METRIC_NAMES}
    assert statuses(compare_intervals("m", expected, dict(expected), POLICY, "s")) == {"PASS"}
    changed = {**expected, "f1_denied": interval(lower=0.2)}
    assert "FAIL" in statuses(compare_intervals("m", expected, changed, POLICY, "s"))
    counts = {**expected, "f1_denied": interval(valid=998)}
    failed = [row["metric"] for row in compare_intervals("m", expected, counts, POLICY, "s") if row["status"] == "FAIL"]
    assert failed == ["ci:f1_denied:valid_replicates", "ci:f1_denied:invalid_replicates"]
    missing = dict(expected)
    del missing["f1_denied"]
    assert "FAIL" in statuses(compare_intervals("m", expected, missing, POLICY, "s"))


def replicates(value: str = "0.5", payload: str = "{}", count: int = 1000) -> list[dict]:
    return [{"replicate_id": str(i), **{name: value for name in METRIC_NAMES}, "invalid_metrics": payload} for i in range(count)]


def test_bootstrap_identifiers_and_payload_are_exact() -> None:
    assert statuses(compare_bootstrap("m", replicates(), replicates(), POLICY, "s")) == {"PASS"}
    assert statuses(compare_bootstrap("m", replicates(), replicates(count=999), POLICY, "s")) == {"FAIL"}
    rows = list(compare_bootstrap("m", replicates(), replicates(payload='{"a": "b"}'), POLICY, "s"))
    assert [row["metric"] for row in rows if row["status"] == "FAIL"] == ["bootstrap:invalid_payload"]


def test_short_bootstrap_table_fails_even_when_both_sides_agree() -> None:
    rows = list(compare_bootstrap("m", replicates(count=10), replicates(count=10), POLICY, "s"))
    assert rows[0]["status"] == "FAIL"


def test_bootstrap_slope_rounds_while_other_metrics_use_absolute_tolerance() -> None:
    observed = replicates()
    observed[3][ "calibration_slope"] = "0.5000000001"
    assert statuses(compare_bootstrap("m", replicates(), observed, POLICY, "s")) == {"PASS"}
    observed = replicates()
    observed[3]["brier_approved"] = "0.5000000001"
    failed = [row["metric"] for row in compare_bootstrap("m", replicates(), observed, POLICY, "s") if row["status"] == "FAIL"]
    assert failed == ["bootstrap:brier_approved"]


def predictions(probability: str = "0.5", ids: tuple[str, ...] = ("a", "b")) -> list[dict]:
    return [{"record_id": record, "y_approved": "1", "p_approved": probability} for record in ids]


def test_prediction_comparison_checks_order_and_values() -> None:
    assert statuses(compare_predictions("m", predictions(), predictions(), POLICY, "s")) == {"PASS"}
    assert "FAIL" in statuses(compare_predictions("m", predictions(), predictions(ids=("b", "a")), POLICY, "s"))
    assert "FAIL" in statuses(compare_predictions("m", predictions(), predictions("0.6"), POLICY, "s"))


def test_missing_reproduction_output_fails_the_model(tmp_path: Path) -> None:
    rows = compare_model(tmp_path, SPEC, POLICY)
    assert [(row["metric"], row["status"]) for row in rows] == [("output", "FAIL")]


def test_committed_policy_has_an_entry_for_each_model() -> None:
    policy = load_policy(REPO)
    assert set(policy) == set(MODELS)
    for values in policy.values():
        assert values == POLICY


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_package(directory: Path, run_id: str, threshold: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "threshold": threshold,
        "point_metrics": points(),
        "confidence_intervals_95_percentile": {name: interval() for name in METRIC_NAMES},
    }
    (directory / "rq1_test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    write_rows(directory / "rq1_bootstrap_metrics.csv", ("replicate_id", *METRIC_NAMES, "invalid_metrics"), replicates())
    write_rows(directory / "test_predictions.csv", ("record_id", "y_approved", "p_approved"), predictions())


def write_output_receipt(directory: Path, classification: str = "checkpoint-based") -> None:
    hashes = {
        name: sha((directory / name).read_bytes())
        for name in ("rq1_test_metrics.json", "rq1_bootstrap_metrics.csv", "test_predictions.csv")
    }
    payload = {"classification": classification, "output_hashes": hashes}
    (directory / "receipt.json").write_text(json.dumps(payload), encoding="utf-8")


def test_full_comparison_passes_and_detects_identity_and_threshold_drift(tmp_path: Path) -> None:
    reference = tmp_path / SPEC.run_root / SPEC.test_directory
    output = tmp_path / OUTPUT_DIRECTORY / SPEC.key
    write_package(reference, SPEC.run_id, 0.493)
    write_package(output, SPEC.run_id, 0.493)
    write_output_receipt(output)
    assert statuses(compare_model(tmp_path, SPEC, POLICY)) == {"PASS"}
    write_package(output, "other_run", 0.5)
    write_output_receipt(output)
    failed = {row["metric"] for row in compare_model(tmp_path, SPEC, POLICY) if row["status"] == "FAIL"}
    assert failed == {"run_id", "threshold"}


def test_copied_stored_files_without_a_receipt_cannot_pass(tmp_path: Path) -> None:
    reference = tmp_path / SPEC.run_root / SPEC.test_directory
    output = tmp_path / OUTPUT_DIRECTORY / SPEC.key
    write_package(reference, SPEC.run_id, 0.493)
    write_package(output, SPEC.run_id, 0.493)
    rows = compare_model(tmp_path, SPEC, POLICY)
    assert [(row["metric"], row["status"]) for row in rows] == [("output_receipt", "FAIL")]


def test_output_changed_after_its_receipt_or_with_another_classification_fails(tmp_path: Path) -> None:
    reference = tmp_path / SPEC.run_root / SPEC.test_directory
    output = tmp_path / OUTPUT_DIRECTORY / SPEC.key
    write_package(reference, SPEC.run_id, 0.493)
    write_package(output, SPEC.run_id, 0.493)
    write_output_receipt(output, classification="artifact-based")
    assert statuses(compare_model(tmp_path, SPEC, POLICY)) == {"FAIL"}
    write_output_receipt(output)
    write_rows(output / "test_predictions.csv", ("record_id", "y_approved", "p_approved"), predictions("0.7"))
    assert statuses(compare_model(tmp_path, SPEC, POLICY)) == {"FAIL"}


# CLI


def test_compare_exit_code_is_nonzero_when_outputs_are_missing(tmp_path: Path, capsys) -> None:
    policy = tmp_path / "config/dataset_b/reproducibility/numerical_policy.json"
    policy.parent.mkdir(parents=True)
    policy.write_text((REPO / "config/dataset_b/reproducibility/numerical_policy.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert cli.main(["compare"], project_root=tmp_path) == cli.EXIT_FAIL
    assert json.loads(capsys.readouterr().out)["failed_count"] == 6


def test_environment_exit_code_is_nonzero_when_not_installed(tmp_path: Path, capsys) -> None:
    write_profile(tmp_path)
    assert cli.main(["environment", "--env-root", str(tmp_path / "envs")], project_root=tmp_path) == cli.EXIT_FAIL


def test_evaluate_exit_code_is_nonzero_when_assets_are_missing(tmp_path: Path, capsys) -> None:
    model_manifest(tmp_path)
    assert cli.main(["evaluate", "--model", "random_forest", "--env-root", str(tmp_path / "envs")], project_root=tmp_path) == cli.EXIT_FAIL


def test_sha_helper_is_stable() -> None:
    assert sha(b"a") == sha(b"a")
