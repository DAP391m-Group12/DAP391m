"""Synthetic unit tests for the RQ2-RQ4 dispatch and table comparison.

These fixtures are not production reproduction evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset_b.reproduce.environment import load_profiles, python_path
from src.dataset_b.reproduce.research import OUTPUT_ROOT, RQ_PROFILE, ROUTES, check_receipt, run_routes
from src.dataset_b.reproduce.rq_tables import compare_csv


REPO = Path(__file__).resolve().parents[1]


def table(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def compare(tmp_path: Path, left: str, right: str, **kwargs) -> dict:
    return compare_csv("r", "t.csv", table(tmp_path / "e.csv", left), table(tmp_path / "o.csv", right), 1e-12, **kwargs)


def test_identical_table_passes(tmp_path: Path) -> None:
    assert compare(tmp_path, "a,b\nx,1\n", "a,b\nx,1\n")["status"] == "PASS"


def test_numeric_tolerance_boundary(tmp_path: Path) -> None:
    assert compare(tmp_path, "a\n0.0\n", "a\n1e-12\n")["status"] == "PASS"
    assert compare(tmp_path, "a\n0.0\n", "a\n2e-12\n")["status"] == "FAIL"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("a,b\nx,1\n", "a,c\nx,1\n"),
        ("a\n1\n2\n", "a\n1\n"),
        ("a\nx\n", "a\ny\n"),
        ("a\nnan\n", "a\n1\n"),
        ("a\n1\n", "a\nnan\n"),
    ],
)
def test_shape_column_text_and_missing_value_differences_fail(tmp_path: Path, left: str, right: str) -> None:
    assert compare(tmp_path, left, right)["status"] == "FAIL"


def test_nan_matches_only_nan(tmp_path: Path) -> None:
    assert compare(tmp_path, "a\nnan\n", "a\nNaN\n")["status"] == "PASS"


def test_ignored_column_is_limited_to_the_named_column(tmp_path: Path) -> None:
    left, right = "run_id,a\nold,1\n", "run_id,a\nnew,1\n"
    assert compare(tmp_path, left, right)["status"] == "FAIL"
    assert compare(tmp_path, left, right, ignore_columns=frozenset({"run_id"}))["status"] == "PASS"
    assert compare(tmp_path, left, "run_id,a\nnew,2\n", ignore_columns=frozenset({"run_id"}))["status"] == "FAIL"


def test_missing_table_fails(tmp_path: Path) -> None:
    expected = table(tmp_path / "e.csv", "a\n1\n")
    assert compare_csv("r", "t.csv", expected, tmp_path / "absent.csv", 1e-12)["status"] == "FAIL"


def write_receipt(root: Path, route: str, **changes) -> None:
    receipt = {
        "classification": "artifact-based",
        "input_hashes": {},
        "output_hashes": {},
        "accepted_run_ids": ["run"],
        "comparison_result": {"status": "PASS", "row_count": 3, "failed_count": 0},
        "limitations": [],
        **changes,
    }
    path = root / OUTPUT_ROOT / route / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_complete_receipt_has_no_problems(tmp_path: Path) -> None:
    write_receipt(tmp_path, "rq3")
    assert check_receipt(tmp_path, "rq3") == []


def test_receipt_integrity_failures(tmp_path: Path) -> None:
    assert check_receipt(tmp_path, "rq3") == ["rq3: receipt is missing"]
    write_receipt(tmp_path, "rq3", classification="retrained")
    assert "classification is not allowed" in check_receipt(tmp_path, "rq3")[0]
    write_receipt(tmp_path, "rq3", comparison_result={"status": "FAIL", "row_count": 3})
    assert check_receipt(tmp_path, "rq3") == ["rq3: comparison did not pass"]
    write_receipt(tmp_path, "rq3", comparison_result={"status": "PASS", "row_count": 0})
    assert check_receipt(tmp_path, "rq3") == ["rq3: comparison did not pass"]


def fake_env(root: Path) -> Path:
    path = python_path(root, RQ_PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return root


def test_missing_environment_fails_before_any_route(tmp_path: Path) -> None:
    calls: list[str] = []
    result = run_routes(tmp_path, tmp_path / "envs", worker=lambda *a: calls.append("x") or 0)
    assert result["status"] == "FAIL" and calls == []


def test_worker_exit_and_receipt_both_decide_status(tmp_path: Path) -> None:
    envs = fake_env(tmp_path / "envs")
    assert run_routes(tmp_path, envs, "rq3", worker=lambda *a: 0)["status"] == "FAIL"
    write_receipt(tmp_path, "rq3")
    assert run_routes(tmp_path, envs, "rq3", worker=lambda *a: 0)["status"] == "PASS"
    assert run_routes(tmp_path, envs, "rq3", worker=lambda *a: 1)["status"] == "FAIL"


def test_all_routes_are_dispatched(tmp_path: Path) -> None:
    envs = fake_env(tmp_path / "envs")
    seen: list[str] = []
    for route in ROUTES:
        write_receipt(tmp_path, route)
    assert run_routes(tmp_path, envs, worker=lambda _p, _r, route, _l: seen.append(route) or 0)["status"] == "PASS"
    assert set(seen) == {"rq2", "rq3", "rq4a", "rq4b"}


def test_committed_profile_covers_every_route() -> None:
    assert set(load_profiles(REPO)[RQ_PROFILE]["models"]) == set(ROUTES)


def test_worker_source_has_no_training_or_threshold_change() -> None:
    source = (REPO / "src/dataset_b/reproduce/rq_worker.py").read_text(encoding="utf-8")
    for forbidden in (".fit(", "partial_fit", "--confirm-production", "evaluate_cross_run_test(", "ProductionBackend", "import run_rq4a", "execute_run"):
        assert forbidden not in source


@pytest.mark.parametrize(
    "relative",
    [
        "src/dataset_b/reproduce/model_worker.py",
        "src/dataset_b/research/rq2/runner.py",
        "src/dataset_b/research/rq2/predictors.py",
        "src/dataset_b/research/rq2/cpu_inference.py",
        "src/dataset_b/research/rq2/exact_methods.py",
        "src/dataset_b/research/rq3/runner.py",
        "src/dataset_b/research/rq3/metrics.py",
        "src/dataset_b/mitigation/rq4b_sex_supplementary.py",
        "src/dataset_b/mitigation/rq4b_evaluation.py",
    ],
)
def test_reproduction_routes_contain_no_fit_call(relative: str) -> None:
    source = (REPO / relative).read_text(encoding="utf-8")
    for forbidden in (".fit(", "fit_transform(", "partial_fit("):
        assert forbidden not in source
