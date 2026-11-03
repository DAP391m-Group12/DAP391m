"""Reproduce one RQ2-RQ4 route under its owning contract; runs in the rq_cpu profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.utils.path import PROJECT_ROOT

from .rq_tables import FIELDS, compare_csv, comparison_row


OUTPUT_ROOT = "reports/reproducibility/research"
REPRODUCTION_RUN_DATE = "20261103"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy() -> float:
    payload = json.loads((PROJECT_ROOT / "config/dataset_b/reproducibility/numerical_policy.json").read_text(encoding="utf-8"))
    return float(payload["research"]["table_atol"])


def _clean(directory: Path) -> Path:
    """Remove an earlier reproduction output; this path is never an accepted run."""
    resolved = directory.resolve()
    if not resolved.is_relative_to((PROJECT_ROOT / OUTPUT_ROOT).resolve()):
        raise ValueError(f"Refusing to clean outside the reproduction output root: {directory}")
    if resolved.exists():
        shutil.rmtree(resolved)
    return resolved


def _compare_directory(route: str, expected: Path, observed: Path, names: Iterable[str], atol: float, ignore: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    return [
        compare_csv(route, name, expected / name, observed / name, atol, ignore_columns=ignore)
        for name in names
    ]


def _compare_array(route: str, name: str, expected: Path, observed: Path, atol: float) -> dict[str, Any]:
    source = expected.as_posix()
    if not observed.is_file():
        return comparison_row(route, name, "present", "missing", None, "exact", False, source)
    left, right = np.load(expected, allow_pickle=False), np.load(observed, allow_pickle=False)
    if left.shape != right.shape or left.dtype != right.dtype:
        return comparison_row(route, name, f"{left.shape} {left.dtype}", f"{right.shape} {right.dtype}", None, "exact", False, source)
    delta = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
    return comparison_row(route, name, left.size, right.size, delta, atol, delta <= atol, source)


def run_rq3(atol: float) -> dict[str, Any]:
    from src.dataset_b.research.rq3.runner import DEFAULT_PROTOCOL_PATH, load_protocol, run_from_preflight, run_preflight

    run_id = "rq3_fairness_20261010_01"
    output = _clean(PROJECT_ROOT / OUTPUT_ROOT / "rq3" / "output")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    run_from_preflight(run_preflight(protocol, project_root=PROJECT_ROOT), protocol, output, progress=None)
    rows = _compare_directory("rq3", PROJECT_ROOT / protocol["output"]["root"], output, ("audit_metadata.csv", "group_metrics.csv", "gap_metrics.csv", "bootstrap_summary.csv"), atol)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    boundary = {name: status.get(name) for name in ("model_fit_calls", "prediction_calls", "bootstrap_indices_generated")}
    rows.append(comparison_row("rq3", "execution_boundary", {"model_fit_calls": 0, "prediction_calls": 0, "bootstrap_indices_generated": False}, boundary, None, "exact", boundary == {"model_fit_calls": 0, "prediction_calls": 0, "bootstrap_indices_generated": False}, "status.json"))
    return {
        "classification": "artifact-based",
        "accepted_run_ids": [run_id],
        "output": output,
        "rows": rows,
        "limitations": ["Audit of saved RQ1 predictions with locked thresholds and bootstrap indices; no prediction call."],
    }


def run_rq4a(_atol: float) -> dict[str, Any]:
    """Validate the accepted packages only; the owner prohibited any new fit on 2026-11-03."""
    from src.dataset_b.research.rq4.rq4a.runner import validate_run_directory

    output = _clean(PROJECT_ROOT / OUTPUT_ROOT / "rq4a" / "output")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    accepted = {"RQ4a-3": "rq4a3_postproc_20261017_01", "RQ4a-6": "rq4a6_postproc_20261017_01"}
    for configuration, run_id in accepted.items():
        root = PROJECT_ROOT / "artifacts/dataset_b/rq4/rq4a" / run_id
        result = validate_run_directory(root, replicates=1000, rows={"validation": 43995, "test": 43996})
        (output / f"{run_id}_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        rows.append(comparison_row(configuration, "package_validation", "PASS", result["status"], None, "exact", result["status"] == "PASS", root.as_posix()))
        status = json.loads((root / "test_evaluation" / "status.json").read_text(encoding="utf-8"))
        rows.append(comparison_row(configuration, "model_statuses", 2, len(status["model_statuses"]), None, "exact", len(status["model_statuses"]) == 2, root.as_posix()))
    return {
        "classification": "report-only",
        "accepted_run_ids": list(accepted.values()),
        "output": output,
        "rows": rows,
        "limitations": [
            "The owner prohibited any fit, so the Fairlearn post-processor was not fitted again and decisions were not regenerated.",
            "This route checks the stored package layout, counts, statuses, and file index; it does not reproduce the post-processing result.",
        ],
    }


def run_rq4b(atol: float) -> dict[str, Any]:
    from src.dataset_b.mitigation import rq4b_inputs as ri
    from src.dataset_b.mitigation import rq4b_runner as rr
    from src.dataset_b.mitigation import rq4b_sex_supplementary as sex

    output = _clean(PROJECT_ROOT / OUTPUT_ROOT / "rq4b" / "output")
    output.mkdir(parents=True)
    contract = ri.load_contract()
    arms = rr._resolve_test_arms(contract.artifact_namespace, contract, rr.MODE_PRODUCTION)
    stage = Path(contract.test_namespace) / contract.test_run_id / "stages" / rr.TEST_STAGE
    saved = {arm.artifact_id: rr.read_predictions(stage / arm.prediction_name) for arm in arms}
    validation = {arm.artifact_id: rr.read_predictions(arm.fit_dir.parent / arm.validation_key / "validation_predictions.csv") for arm in arms}
    reference = next(iter(saved.values()))
    tables = rr._compute_test_tables(arms, saved, validation, rr._load_test_indices(contract, reference))
    rows: list[dict[str, Any]] = []
    for name, table in tables.items():
        columns = list(contract.test_required_output_files[f"stages/{rr.TEST_STAGE}/{name}"])
        (output / name).write_text(rr._csv_content(table, columns), encoding="utf-8", newline="")
        rows.append(compare_csv("rq4b_test", name, stage / name, output / name, atol))

    sex_output = output / "sex_supplementary"
    sex_output.mkdir()
    sex_arms, _ = sex._verify_sources()
    sex_stage = sex.ROOT / sex.RUN_ID / "stages" / sex.STAGE
    manifest = json.loads((sex.ROOT / sex.RUN_ID / "run_manifest.json").read_text(encoding="utf-8"))
    test, sex_validation, indices = sex._load_saved(sex_arms)
    for name, table in sex._tables(sex_arms, test, sex_validation, indices).items():
        (sex_output / name).write_bytes(sex._csv_bytes(table, manifest["output_schema"][name]))
        rows.append(compare_csv("rq4b_sex", name, sex_stage / name, sex_output / name, atol))

    gate_receipt = Path(contract.aggregation_namespace) / "rq4b3_gatec_20261024_01" / "stages" / "gate_c_aggregation" / "receipt.json"
    selection = json.loads((gate_receipt.parent / "gate_c_selection.json").read_text(encoding="utf-8"))
    rows.append(comparison_row("rq4b_gate_c", "selected_lambda", 0.0, selection.get("selected_lambda"), None, "exact", selection.get("selected_lambda") == 0.0, gate_receipt.parent.as_posix()))
    return {
        "classification": "artifact-based",
        "accepted_run_ids": [contract.test_run_id, sex.RUN_ID, "rq4b3_gatec_20261024_01"],
        "output": output,
        "rows": rows,
        "limitations": [
            "Test and applicant-sex tables are recomputed from saved predictions; no fit, refit, CV, or inference.",
            "Gate C is read only: lambda selection is not reopened and its stored selection is checked, not recomputed.",
            "No research interpretation is produced.",
        ],
    }


def run_rq2(atol: float) -> dict[str, Any]:
    from src.dataset_b.research.rq2.runner import run_rq2 as execute

    run_id = "rq2_cpu_production_20261009_01"
    output = _clean(PROJECT_ROOT / OUTPUT_ROOT / "rq2" / "output")
    output.parent.mkdir(parents=True, exist_ok=True)
    execute(output_directory=output, run_id=f"rq2_reproduction_{REPRODUCTION_RUN_DATE}_01", progress_callback=lambda message: print(message, flush=True))
    reference = PROJECT_ROOT / "artifacts/dataset_b/rq2" / run_id
    ignore = frozenset({"run_id"})
    rows = _compare_directory("rq2", reference, output, ("sample.csv", "donors.csv", "cross_model_stability.csv"), atol, ignore=ignore)
    rows.append(_compare_array("rq2", "bootstrap_indices.npy", reference / "bootstrap_indices.npy", output / "bootstrap_indices.npy", 0.0))
    for model in ("lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini"):
        rows += _compare_directory(f"rq2_{model}", reference / "models" / model, output / "models" / model, ("importance.csv", "bootstrap.csv"), atol, ignore=ignore)
        for name in ("cube.npy", "gsv_local.npy"):
            rows.append(_compare_array(f"rq2_{model}", name, reference / "models" / model / name, output / "models" / model / name, atol))
    return {
        "classification": "checkpoint-based",
        "accepted_run_ids": [run_id],
        "output": output,
        "rows": rows,
        "limitations": [
            "CPU frozen-model inference over Validation with locked seeds; the protocol does not read Test.",
            "Output run ID differs from the accepted run ID and is ignored in comparison.",
        ],
    }


ROUTES: dict[str, Callable[[float], dict[str, Any]]] = {"rq2": run_rq2, "rq3": run_rq3, "rq4a": run_rq4a, "rq4b": run_rq4b}


def run_route(route: str) -> dict[str, Any]:
    result = ROUTES[route](_policy())
    output: Path = result["output"]
    rows = result["rows"]
    input_hashes: dict[str, str] = {}
    for row in rows:
        source = Path(row["source"])
        if source.is_absolute() and source.resolve().is_relative_to(PROJECT_ROOT.resolve()):
            row["source"] = source.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        if (PROJECT_ROOT / row["source"]).is_file():
            input_hashes[row["source"]] = _sha256(PROJECT_ROOT / row["source"])
    table = pd.DataFrame(rows, columns=FIELDS)
    directory = PROJECT_ROOT / OUTPUT_ROOT / route
    table.to_csv(directory / "comparison.csv", index=False, lineterminator="\n")
    failed = [row for row in rows if row["status"] != "PASS"]
    output_files = sorted(path for path in output.rglob("*") if path.is_file())
    receipt = {
        "route": route,
        "classification": result["classification"],
        "accepted_run_ids": result["accepted_run_ids"],
        "input_hashes": input_hashes,
        "output_hashes": {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in output_files},
        "comparison_result": {"status": "FAIL" if failed else "PASS", "row_count": len(rows), "failed_count": len(failed)},
        "failed": [{"name": row["metric"], "route": row["model"], "delta": row["delta"]} for row in failed[:30]],
        "limitations": result["limitations"],
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return receipt["comparison_result"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=sorted(ROUTES), required=True)
    args = parser.parse_args()
    result = run_route(args.route)
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
