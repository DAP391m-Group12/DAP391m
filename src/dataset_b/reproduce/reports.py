"""Build report tables and the object mapping from existing reproduction outputs.

This stage reads outputs only. It starts no model, fit, inference, or notebook.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .compare import load_policy, output_binding_problems, values_match
from .research import OUTPUT_ROOT as RESEARCH_ROOT
from .spec import METRIC_NAMES, MODELS, OUTPUT_DIRECTORY


REPORT_DIRECTORY = "reports/reproducibility/reports"
MAPPING_FIELDS = (
    "object_id",
    "table_or_figure",
    "command",
    "config",
    "accepted_run",
    "inputs",
    "generated_output",
    "expected_reference",
    "comparison_status",
    "limitations",
)
REPORTING_METRIC_NAMES = {"brier_approved": "brier_score_approved"}
RESEARCH_OBJECTS = (
    ("RQ2_EXPLANATIONS", "RQ2 importance, bootstrap, and stability tables", "rq2", "config/dataset_b/rq2_protocol.yaml", "artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01"),
    ("RQ3_FAIRNESS", "RQ3 group, gap, and bootstrap tables", "rq3", "config/dataset_b/rq3_protocol.yaml", "artifacts/dataset_b/rq3/rq3_fairness_20261010_01"),
    ("RQ4A_POSTPROCESSING", "RQ4a stored reporting package", "rq4a", "config/dataset_b/rq4a_postprocessing.yaml", "artifacts/dataset_b/rq4/rq4a"),
    ("RQ4B_ADVERSARIAL", "RQ4b Test and applicant-sex tables", "rq4b", "config/dataset_b/rq4b_adversarial.yaml", "artifacts/dataset_b/rq4/rq4b/test_evaluation/rq4b3_test_20261024_01"),
)
NOT_REGENERATED_OBJECTS = (
    ("FIGURE_02", "Figure 2 post-processing group error rates", "reports/paper_figures/fig02_postprocessing_group_error_rates.svg"),
    ("FIGURE_03", "Figure 3 adversarial group error rates", "reports/paper_figures/fig03_adversarial_group_error_rates.svg"),
    ("RQ_REPORTS", "Assembled RQ1, RQ2, and RQ3-RQ4 text reports", "reports/RQ1_Research_Metrics_Final.txt"),
)


def _rq1_table(project_root: Path, policy: dict[str, dict[str, Any]], table_path: Path) -> tuple[str, str]:
    """Write the RQ1 long table from reproduction output and compare it with accepted reporting."""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for key, spec in MODELS.items():
        metrics_path = project_root / OUTPUT_DIRECTORY / key / "rq1_test_metrics.json"
        reference_path = project_root / spec.path("reporting/rq1_metrics_long.csv")
        if not metrics_path.is_file() or not reference_path.is_file():
            failures.append(f"{key}: output or reference is missing")
            continue
        problems = output_binding_problems(metrics_path.parent)
        if problems:
            failures.append(f"{key}: {'; '.join(problems)}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with reference_path.open(encoding="utf-8", newline="") as handle:
            reference = {row["metric"]: row for row in csv.DictReader(handle)}
        for name in METRIC_NAMES:
            label = REPORTING_METRIC_NAMES.get(name, name)
            estimate = metrics["point_metrics"].get(name)
            interval = metrics["confidence_intervals_95_percentile"].get(name, {})
            rows.append({"model": key, "run_id": spec.run_id, "metric": label, "estimate": estimate, "ci_low": interval.get("lower_2_5"), "ci_high": interval.get("upper_97_5")})
            expected = reference.get(label)
            if expected is None:
                failures.append(f"{key}: {label} is missing from the reference")
                continue
            checks = (
                (expected["estimate"], estimate, policy[key]["point_atol"]),
                (expected["ci_low"], interval.get("lower_2_5"), policy[key]["ci_atol"]),
                (expected["ci_high"], interval.get("upper_97_5"), policy[key]["ci_atol"]),
            )
            for left, right, tolerance in checks:
                if right is None or not values_match(float(left), float(right), tolerance)[0]:
                    failures.append(f"{key}: {label} differs from the reference")
                    break
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "run_id", "metric", "estimate", "ci_low", "ci_high"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return ("FAIL" if failures else "PASS"), "; ".join(failures[:5])


def _research_object(project_root: Path, route: str) -> tuple[str, str, str]:
    """Report a route only from its receipt; a route that did not run stays NOT_RUN."""
    receipt_path = project_root / RESEARCH_ROOT / route / "receipt.json"
    if not receipt_path.is_file():
        return "NOT_RUN", "", "The route has no receipt in this workspace, so no result is claimed."
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    status = receipt.get("comparison_result", {}).get("status", "FAIL")
    hashes = receipt.get("output_hashes", {})
    stale = not hashes or any(
        not (project_root / name).is_file() or hashlib.sha256((project_root / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    )
    if stale:
        return "FAIL", "", "Route outputs do not match the hashes in the receipt."
    return status, f"{RESEARCH_ROOT}/{route}/comparison.csv", " ".join(receipt.get("limitations", []))


def build_reports(project_root: Path) -> dict[str, Any]:
    """Write the mapping; PASS needs every object to be generated and to match its reference."""
    policy = load_policy(project_root)
    table = project_root / REPORT_DIRECTORY / "rq1_metrics_long.csv"
    rq1_status, rq1_note = _rq1_table(project_root, policy, table)
    mapping = [
        {
            "object_id": "RQ1_METRICS",
            "table_or_figure": "RQ1 ten-metric table with 95% percentile intervals",
            "command": "python -m src.dataset_b.reproduce evaluate --model all",
            "config": "config/dataset_b/reproducibility/numerical_policy.json",
            "accepted_run": ";".join(spec.run_id for spec in MODELS.values()),
            "inputs": f"{OUTPUT_DIRECTORY}/<model>/rq1_test_metrics.json",
            "generated_output": table.relative_to(project_root).as_posix(),
            "expected_reference": "<accepted run>/reporting/rq1_metrics_long.csv",
            "comparison_status": rq1_status,
            "limitations": rq1_note,
        }
    ]
    for object_id, label, route, config, accepted in RESEARCH_OBJECTS:
        status, output, limitations = _research_object(project_root, route)
        mapping.append(
            {
                "object_id": object_id,
                "table_or_figure": label,
                "command": f"python -m src.dataset_b.reproduce research --route {route}",
                "config": config,
                "accepted_run": accepted,
                "inputs": f"{RESEARCH_ROOT}/{route}/receipt.json",
                "generated_output": output,
                "expected_reference": accepted,
                "comparison_status": status,
                "limitations": limitations,
            }
        )
    for object_id, label, reference in NOT_REGENERATED_OBJECTS:
        mapping.append(
            {
                "object_id": object_id,
                "table_or_figure": label,
                "command": "",
                "config": "",
                "accepted_run": "",
                "inputs": "",
                "generated_output": "",
                "expected_reference": reference,
                "comparison_status": "NOT_REGENERATED",
                "limitations": "This object comes from a notebook; no source-backed builder exists and notebooks were not executed.",
            }
        )
    path = project_root / REPORT_DIRECTORY / "report_mapping.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(mapping)
    counts: dict[str, int] = {}
    for row in mapping:
        counts[row["comparison_status"]] = counts.get(row["comparison_status"], 0) + 1
    status = "PASS" if set(counts) == {"PASS"} else ("FAIL" if "FAIL" in counts else "INCOMPLETE")
    return {"status": status, "object_counts": counts, "mapping": path.relative_to(project_root).as_posix()}
