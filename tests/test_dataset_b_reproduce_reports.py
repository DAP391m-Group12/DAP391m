"""Synthetic unit tests for the report mapping.

These fixtures are not production reproduction evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.dataset_b.reproduce import cli
from src.dataset_b.reproduce.compare import POLICY_PATH
from src.dataset_b.reproduce.reports import MAPPING_FIELDS, REPORT_DIRECTORY, RESEARCH_OBJECTS, build_reports
from src.dataset_b.reproduce.research import OUTPUT_ROOT
from src.dataset_b.reproduce.spec import METRIC_NAMES, MODELS, OUTPUT_DIRECTORY


REPO = Path(__file__).resolve().parents[1]


def setup_project(root: Path, *, estimate: float = 0.5) -> None:
    policy = root / POLICY_PATH
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text((REPO / POLICY_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    interval = {"lower_2_5": 0.4, "upper_97_5": 0.6}
    for key, spec in MODELS.items():
        output = root / OUTPUT_DIRECTORY / key
        output.mkdir(parents=True)
        payload = {
            "point_metrics": {name: estimate for name in METRIC_NAMES},
            "confidence_intervals_95_percentile": {name: interval for name in METRIC_NAMES},
        }
        metrics_file = output / "rq1_test_metrics.json"
        metrics_file.write_text(json.dumps(payload), encoding="utf-8")
        receipt = {
            "classification": "checkpoint-based",
            "output_hashes": {metrics_file.name: hashlib.sha256(metrics_file.read_bytes()).hexdigest()},
        }
        (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        reference =root / spec.path("reporting/rq1_metrics_long.csv")
        reference.parent.mkdir(parents=True)
        with reference.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("metric", "estimate", "ci_low", "ci_high"))
            writer.writeheader()
            for name in METRIC_NAMES:
                label = "brier_score_approved" if name == "brier_approved" else name
                writer.writerow({"metric": label, "estimate": 0.5, "ci_low": 0.4, "ci_high": 0.6})


def write_receipt(root: Path, route: str, status: str = "PASS") -> None:
    path = root / OUTPUT_ROOT / route / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = path.with_name("comparison.csv")
    table.write_text("a\n1\n", encoding="utf-8")
    hashes = {table.relative_to(root).as_posix(): hashlib.sha256(table.read_bytes()).hexdigest()}
    payload = {"comparison_result": {"status": status}, "limitations": ["note"], "output_hashes": hashes}
    path.write_text(json.dumps(payload), encoding="utf-8")


def mapping(root: Path) -> dict[str, dict[str, str]]:
    with (root / REPORT_DIRECTORY / "report_mapping.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) == set(MAPPING_FIELDS)
    return {row["object_id"]: row for row in rows}


def test_route_without_a_receipt_is_not_run_and_never_pass(tmp_path: Path) -> None:
    setup_project(tmp_path)
    for _, _, route, _, _ in RESEARCH_OBJECTS:
        if route != "rq2":
            write_receipt(tmp_path, route)
    result = build_reports(tmp_path)
    assert result["status"] == "INCOMPLETE"
    rows = mapping(tmp_path)
    assert rows["RQ2_EXPLANATIONS"]["comparison_status"] == "NOT_RUN"
    assert rows["RQ2_EXPLANATIONS"]["generated_output"] == ""


def test_mapping_is_complete_for_every_object(tmp_path: Path) -> None:
    setup_project(tmp_path)
    build_reports(tmp_path)
    rows = mapping(tmp_path)
    assert {"RQ1_METRICS", "RQ2_EXPLANATIONS", "RQ3_FAIRNESS", "RQ4A_POSTPROCESSING", "RQ4B_ADVERSARIAL", "FIGURE_02", "FIGURE_03", "RQ_REPORTS"} == set(rows)
    assert rows["RQ1_METRICS"]["comparison_status"] == "PASS"
    assert rows["FIGURE_02"]["comparison_status"] == "NOT_REGENERATED"
    assert all(row["expected_reference"] for row in rows.values())


def test_generated_rq1_table_differing_from_reference_fails(tmp_path: Path) -> None:
    setup_project(tmp_path, estimate=0.6)
    assert build_reports(tmp_path)["status"] == "FAIL"
    assert mapping(tmp_path)["RQ1_METRICS"]["comparison_status"] == "FAIL"


def test_failed_route_receipt_fails_the_stage(tmp_path: Path) -> None:
    setup_project(tmp_path)
    write_receipt(tmp_path, "rq3", status="FAIL")
    assert build_reports(tmp_path)["status"] == "FAIL"


def test_stale_receipt_cannot_report_pass(tmp_path: Path) -> None:
    setup_project(tmp_path)
    write_receipt(tmp_path, "rq3")
    (tmp_path / OUTPUT_ROOT / "rq3" / "comparison.csv").write_text("changed\n", encoding="utf-8")
    build_reports(tmp_path)
    assert mapping(tmp_path)["RQ3_FAIRNESS"]["comparison_status"] == "FAIL"


def test_rq1_output_without_a_receipt_fails_the_table(tmp_path: Path) -> None:
    setup_project(tmp_path)
    (tmp_path / OUTPUT_DIRECTORY / "xgboost" / "receipt.json").unlink()
    build_reports(tmp_path)
    assert mapping(tmp_path)["RQ1_METRICS"]["comparison_status"] == "FAIL"


def test_clean_clone_orchestration_runs_every_stage_in_order(tmp_path: Path, capsys, monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(cli, "_run_public", lambda stage, _root, _args: seen.append(stage) or {"status": "PASS"})
    assert cli.main(["all"], project_root=tmp_path) == cli.EXIT_PASS
    assert seen == ["download", "verify-assets", "environment", "evaluate", "compare", "research", "reports"]


def test_clean_clone_orchestration_stops_at_the_first_failed_stage(tmp_path: Path, capsys, monkeypatch) -> None:
    seen: list[str] = []

    def stage_result(stage, _root, _args):
        seen.append(stage)
        return {"status": "FAIL" if stage == "compare" else "PASS"}

    monkeypatch.setattr(cli, "_run_public", stage_result)
    assert cli.main(["all"], project_root=tmp_path) == cli.EXIT_FAIL
    assert seen[-1] == "compare" and "research" not in seen
    assert json.loads(capsys.readouterr().out)["stopped_at"] == "compare"


def test_cli_exit_code_is_nonzero_when_objects_are_missing(tmp_path: Path, capsys) -> None:
    setup_project(tmp_path)
    assert cli.main(["reports"], project_root=tmp_path) == cli.EXIT_FAIL
    assert json.loads(capsys.readouterr().out)["status"] == "INCOMPLETE"


def test_reports_module_starts_no_process_or_model() -> None:
    source = (REPO / "src/dataset_b/reproduce/reports.py").read_text(encoding="utf-8")
    for forbidden in ("subprocess", "joblib", "torch", ".fit(", "predict"):
        assert forbidden not in source
