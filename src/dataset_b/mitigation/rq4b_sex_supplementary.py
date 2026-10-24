"""Read-only sex supplementary evaluation for the closed RQ4b-3 Test run.

This module has a deliberately narrow boundary: it reads the nine stored Test
predictions, the corresponding stored Validation predictions and the locked
RQ1 Test indices.  It never constructs a backend, fits, or performs inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.dataset_b.mitigation import metrics as m
from src.dataset_b.mitigation import rq4b_evaluation as ev
from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.mitigation import rq4b_runner as rr


RUN_ID = "rq4b3_sex_supplementary_20261024_02"
ROOT = rr.PROJECT_ROOT / "artifacts" / "dataset_b" / "rq4" / "rq4b" / "sex_supplementary"
SOURCE_TEST_ROOT = ri.TEST_RUN_ID
SOURCE_TEST_MANIFEST_SHA256 = "d05fc2980986452428b5fc656aaba89c8bad698923cab7da67f5ff241e7595a4"
SOURCE_TEST_RECEIPT_SHA256 = "2f7c1ae8b31d8df82b9342182cb40fb9b3c18aa783a31ff53eeeaf54e64632c4"
SOURCE_PACKAGE_RECEIPT_SHA256 = "d20fa9b13cd4f9b9e5e52e4c387f49a63791f4c95af3fa9903caecdbf1e5800a"
SEX_GROUPS = ev.SEX_LABELS
SEX_ROSTER = ev.SEX_ROSTER
STAGE = "sex_evaluation"
PACKAGE_STAGE = "package_validation"
OUTPUTS = (
    "sex_arm_metrics.csv", "sex_group_metrics.csv", "sex_matched_deltas.csv",
    "sex_decision_changes.csv", "sex_bootstrap_replicates.csv",
    "sex_bootstrap_summary.csv", "sex_across_seed_summary.csv",
    "sex_generalization_gaps.csv", "execution_status.json",
)


class SexSupplementaryError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise SexSupplementaryError(message)


def _sha(path: Path) -> str:
    return rr.sha256_file(path)


def _json(path: Path) -> dict[str, Any]:
    return rr.read_json(path)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    import io
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})
    return out.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.write_bytes(_csv_bytes(rows, columns))


def _schemas() -> dict[str, list[str]]:
    return {
        "sex_arm_metrics.csv": ["seed", "adversary_lambda", "role", "source_run_id", "threshold", "partition", "rows", "record_id_sha256", "sex_eod", "sex_eod_valid", "sex_eod_reason", "tpr_range", "fpr_range"],
        "sex_group_metrics.csv": ["seed", "adversary_lambda", "partition", "group", "row_count", "actual_approved_count", "actual_denied_count", "predicted_approved_count", "predicted_denied_count", "tpr_approved", "fpr_approved", "selection_rate", "tpr_valid", "fpr_valid", "selection_rate_valid", "invalid_reason"],
        "sex_matched_deltas.csv": ["seed", "comparator_lambda", "control_lambda", "delta_sex_eod", "valid", "invalid_reason"],
        "sex_decision_changes.csv": ["seed", "comparator_lambda", "control_lambda", "scope", "group", "row_count", "unchanged_count", "changed_count", "denied_to_approved_count", "approved_to_denied_count", "change_rate"],
        "sex_bootstrap_replicates.csv": ["seed", "comparator_lambda", "control_lambda", "replicate_id", "sample_size", "delta_sex_eod", "valid", "invalid_reason"],
        "sex_bootstrap_summary.csv": ["seed", "comparator_lambda", "control_lambda", "quantity", "replicates", "valid_replicates", "invalid_replicates", "confidence_level", "lower", "upper", "quantile_method"],
        "sex_across_seed_summary.csv": ["scope", "adversary_lambda", "comparator_lambda", "control_lambda", "quantity", "seed_count", "mean", "standard_deviation", "ddof", "min", "max", "range"],
        "sex_generalization_gaps.csv": ["seed", "adversary_lambda", "quantity", "validation_value", "test_value", "test_minus_validation", "valid", "invalid_reason"],
    }


def _source_layout() -> rr.RunLayout:
    return rr.RunLayout(ri.load_contract().test_namespace, SOURCE_TEST_ROOT, rr.TEST_LAYOUT_MODES[rr.MODE_PRODUCTION])


def _verify_sources() -> tuple[list[rr.TestArmSource], dict[str, str]]:
    """Fail closed on every closed production pin before any output is created."""
    contract = ri.load_contract()
    gate_pins = rr._verify_gate_c_for_test(contract.aggregation_namespace, contract, rr.MODE_PRODUCTION)
    arms = rr._resolve_test_arms(contract.artifact_namespace, contract, rr.MODE_PRODUCTION)
    layout = _source_layout()
    _require(_sha(layout.run_dir / rr.MANIFEST_FILE) == SOURCE_TEST_MANIFEST_SHA256, "Source Test manifest digest differs.")
    _require(_sha(layout.stage_dir(rr.TEST_STAGE) / rr.RECEIPT_FILE) == SOURCE_TEST_RECEIPT_SHA256, "Source Test receipt digest differs.")
    _require(_sha(layout.stage_dir(rr.PACKAGE_STAGE) / rr.RECEIPT_FILE) == SOURCE_PACKAGE_RECEIPT_SHA256, "Source Test package receipt digest differs.")
    test_receipt = rr.read_stage(layout, rr.TEST_STAGE)
    package_receipt = rr.read_stage(layout, rr.PACKAGE_STAGE)
    _require(test_receipt["status"] == package_receipt["status"] == "PASS", "Source Test receipts are not PASS.")
    _require(test_receipt["counts"]["cv_fold_fits"] == test_receipt["counts"]["refit_fits"] == 0, "Source Test contains local fitting.")
    pins = {"source_test_manifest": SOURCE_TEST_MANIFEST_SHA256, "source_test_receipt": SOURCE_TEST_RECEIPT_SHA256,
            "source_package_receipt": SOURCE_PACKAGE_RECEIPT_SHA256, "locked_bootstrap": contract.test_bootstrap_sha256,
            "gate_c_manifest": gate_pins["manifest"], "gate_c_receipt": gate_pins["receipt"], "gate_c_selection": gate_pins["selection"]}
    for run_id, source in contract.aggregation_source_pins.items():
        pins[f"source_manifest:{run_id}"] = source["manifest_sha256"]
        for relative, digest in source["receipt_sha256"].items():
            pins[f"source_receipt:{run_id}:{relative}"] = digest
    return arms, pins


def _sex_metrics(pred: ev.PartitionPredictions, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d = ev.decisions_at(pred.p_approved, threshold)
    labels = np.asarray(pred.sex).astype(str)
    metrics = m.compute_roster_group_metrics(pred.y, pred.p_approved, d, labels, SEX_GROUPS, rules=ev.RULES)
    eod = m.compute_fixed_roster_eod({key: metrics[key] for key in SEX_ROSTER}, SEX_ROSTER)
    rows = []
    for group in SEX_GROUPS:
        mask = labels == group
        tpr, fpr, selection = (metrics[group][key] for key in ("tpr_approved", "fpr_approved", "selection_rate"))
        reasons = sorted({x.status for x in (tpr, fpr, selection) if not x.is_defined})
        rows.append({"group": group, "row_count": int(mask.sum()), "actual_approved_count": int(np.sum(pred.y[mask] == 1)),
                     "actual_denied_count": int(np.sum(pred.y[mask] == 0)), "predicted_approved_count": int(np.sum(d[mask] == 1)),
                     "predicted_denied_count": int(np.sum(d[mask] == 0)), "tpr_approved": tpr.value, "fpr_approved": fpr.value,
                     "selection_rate": selection.value, "tpr_valid": tpr.is_defined, "fpr_valid": fpr.is_defined,
                     "selection_rate_valid": selection.is_defined, "invalid_reason": "|".join(reasons) if reasons else None})
    return {"value": eod.value, "valid": eod.valid, "reason": eod.reason_code, "tpr_range": eod.tpr_range, "fpr_range": eod.fpr_range}, rows


def _sex_eod_sample(pred: ev.PartitionPredictions, threshold: float, indices: np.ndarray) -> dict[str, Any]:
    """Fast fixed-roster replicate metric, equivalent to the shared metric path.

    This intentionally operates only on a resample.  Point estimates and all
    per-group outputs use ``_sex_metrics`` and therefore the shared RQ3/RQ4a
    implementation directly.
    """
    labels = np.asarray(pred.sex).astype(str)
    full_codes = np.where(labels == "Male", 0, np.where(labels == "Female", 1, 2))
    sample = indices[full_codes[indices] < 2]
    codes = full_codes[sample]
    y = pred.y[sample]
    approved = ev.decisions_at(pred.p_approved[sample], threshold)
    actual_positive = np.bincount(codes, weights=(y == 1), minlength=2)
    actual_negative = np.bincount(codes, weights=(y == 0), minlength=2)
    group_rows = np.bincount(codes, minlength=2)
    if (group_rows < ev.RULES.minimum_group_rows).any() or (actual_positive < ev.RULES.minimum_actual_approved).any() or (actual_negative < ev.RULES.minimum_actual_denied).any():
        return {"value": None, "valid": False, "reason": "required_group_metric_invalid"}
    tp = np.bincount(codes, weights=((y == 1) & (approved == 1)), minlength=2)
    fp = np.bincount(codes, weights=((y == 0) & (approved == 1)), minlength=2)
    tpr, fpr = tp / actual_positive, fp / actual_negative
    return {"value": float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min())), "valid": True, "reason": "none"}


def _bootstrap_arm_values(pred: ev.PartitionPredictions, threshold: float, indices: np.ndarray) -> np.ndarray:
    """Compute all replicate EODs in bounded vectorized batches.

    Batch evaluation keeps the immutable 1,000-by-N index layout intact while
    avoiding a Python-level metric pass for every replicate.
    """
    labels = np.asarray(pred.sex).astype(str)
    codes = np.where(labels == "Male", 0, np.where(labels == "Female", 1, 2)).astype(np.int8)
    values = np.full(len(indices), np.nan, dtype=np.float64)
    for start in range(0, len(indices), 50):
        stop, sample = min(start + 50, len(indices)), indices[start:start + 50]
        c, y = codes[sample], pred.y[sample]
        rows = np.stack([(c == group).sum(axis=1) for group in (0, 1)], axis=1)
        positive = np.stack([((c == group) & (y == 1)).sum(axis=1) for group in (0, 1)], axis=1)
        negative = np.stack([((c == group) & (y == 0)).sum(axis=1) for group in (0, 1)], axis=1)
        valid = ((rows >= ev.RULES.minimum_group_rows).all(axis=1) & (positive >= ev.RULES.minimum_actual_approved).all(axis=1) & (negative >= ev.RULES.minimum_actual_denied).all(axis=1))
        approved = pred.p_approved[sample] >= threshold
        tp = np.stack([((c == group) & (y == 1) & approved).sum(axis=1) for group in (0, 1)], axis=1)
        fp = np.stack([((c == group) & (y == 0) & approved).sum(axis=1) for group in (0, 1)], axis=1)
        tpr, fpr = tp / positive, fp / negative
        values[start:stop] = np.where(valid, np.maximum(tpr.max(axis=1) - tpr.min(axis=1), fpr.max(axis=1) - fpr.min(axis=1)), np.nan)
    return values


def _decision_rows(control: ev.PartitionPredictions, comparator: ev.PartitionPredictions, ct: float, at: float, seed: int, lam: float) -> list[dict[str, Any]]:
    ev.require_paired(control, comparator)
    base, alt, labels = ev.decisions_at(control.p_approved, ct), ev.decisions_at(comparator.p_approved, at), np.asarray(control.sex).astype(str)
    scopes = [("overall", "__all__", np.ones(len(labels), dtype=bool))] + [("sex_group", g, labels == g) for g in SEX_GROUPS]
    out = []
    for scope, group, mask in scopes:
        changed, n = base[mask] != alt[mask], int(mask.sum())
        out.append({"seed": seed, "comparator_lambda": lam, "control_lambda": 0.0, "scope": scope, "group": group, "row_count": n,
                    "unchanged_count": int(n - changed.sum()), "changed_count": int(changed.sum()),
                    "denied_to_approved_count": int(np.sum((base[mask] == 0) & (alt[mask] == 1))),
                    "approved_to_denied_count": int(np.sum((base[mask] == 1) & (alt[mask] == 0))), "change_rate": float(changed.mean()) if n else None})
    return out


def _summary(scope: str, lam: float | None, comparator: float | None, quantity: str, values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    return {"scope": scope, "adversary_lambda": lam, "comparator_lambda": comparator, "control_lambda": 0.0 if comparator is not None else None,
            "quantity": quantity, "seed_count": len(a), "mean": float(a.mean()), "standard_deviation": float(a.std(ddof=1)), "ddof": 1,
            "min": float(a.min()), "max": float(a.max()), "range": float(a.max() - a.min())}


def _tables(arms: Sequence[rr.TestArmSource], test: Mapping[str, ev.PartitionPredictions], validation: Mapping[str, ev.PartitionPredictions], indices: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    point, arm_rows, group_rows, gaps = {}, [], [], []
    for arm in arms:
        for partition, pred in (("test", test[arm.artifact_id]), ("validation", validation[arm.artifact_id])):
            metric, groups = _sex_metrics(pred, arm.threshold)
            if partition == "test":
                point[(arm.adversary_lambda, arm.seed)] = metric
            arm_rows.append({"seed": arm.seed, "adversary_lambda": arm.adversary_lambda, "role": arm.role, "source_run_id": arm.run_id,
                             "threshold": arm.threshold, "partition": partition, "rows": len(pred.y), "record_id_sha256": rr.sha256_ids(pred.record_ids),
                             "sex_eod": metric["value"], "sex_eod_valid": metric["valid"], "sex_eod_reason": metric["reason"], "tpr_range": metric["tpr_range"], "fpr_range": metric["fpr_range"]})
            group_rows.extend([{"seed": arm.seed, "adversary_lambda": arm.adversary_lambda, "partition": partition, **row} for row in groups])
        test_m, val_m = _sex_metrics(test[arm.artifact_id], arm.threshold)[0], _sex_metrics(validation[arm.artifact_id], arm.threshold)[0]
        valid = test_m["valid"] and val_m["valid"]
        gaps.append({"seed": arm.seed, "adversary_lambda": arm.adversary_lambda, "quantity": "sex_eod", "validation_value": val_m["value"], "test_value": test_m["value"], "test_minus_validation": (test_m["value"] - val_m["value"]) if valid else None, "valid": valid, "invalid_reason": None if valid else "required_group_metric_invalid"})
    deltas, decisions, reps, boots = [], [], [], []
    bootstrap_values = {arm.artifact_id: _bootstrap_arm_values(test[arm.artifact_id], arm.threshold, indices) for arm in arms}
    for seed, lam, _ in ri.TEST_MATCHED_COMPARISONS:
        c, a = next(x for x in arms if (x.adversary_lambda, x.seed) == (0.0, seed)), next(x for x in arms if (x.adversary_lambda, x.seed) == (lam, seed))
        cm, am = point[(0.0, seed)], point[(lam, seed)]
        valid = cm["valid"] and am["valid"]
        deltas.append({"seed": seed, "comparator_lambda": lam, "control_lambda": 0.0, "delta_sex_eod": (am["value"] - cm["value"]) if valid else None, "valid": valid, "invalid_reason": None if valid else "required_group_metric_invalid"})
        decisions.extend(_decision_rows(test[c.artifact_id], test[a.artifact_id], c.threshold, a.threshold, seed, lam))
        raw = bootstrap_values[a.artifact_id] - bootstrap_values[c.artifact_id]
        values, invalid = [], []
        for replicate_id, raw_value in enumerate(raw):
            ok = bool(np.isfinite(raw_value))
            value, reason = (float(raw_value), None) if ok else (None, "required_group_metric_invalid")
            values.append(value)
            if not ok: invalid.append(replicate_id)
            reps.append({"seed": seed, "comparator_lambda": lam, "control_lambda": 0.0, "replicate_id": replicate_id, "sample_size": indices.shape[1], "delta_sex_eod": value, "valid": ok, "invalid_reason": reason})
        valid_values = [x for x in values if x is not None]
        boots.append({"seed": seed, "comparator_lambda": lam, "control_lambda": 0.0, "quantity": "delta_sex_eod", "replicates": len(values), "valid_replicates": len(valid_values), "invalid_replicates": len(invalid), "confidence_level": 0.95, "lower": float(np.quantile(valid_values, .025, method="linear")) if len(valid_values) == len(values) else None, "upper": float(np.quantile(valid_values, .975, method="linear")) if len(valid_values) == len(values) else None, "quantile_method": "linear"})
    across = []
    for lam in (0.0, .01):
        values = [point[(lam, seed)]["value"] for seed in (42, 43, 44)]
        if all(x is not None for x in values): across.append(_summary("per_arm", lam, None, "sex_eod", values))
    values = [x["delta_sex_eod"] for x in deltas if x["comparator_lambda"] == .01]
    if all(x is not None for x in values) and len(values) == 3: across.append(_summary("matched_delta", None, .01, "delta_sex_eod", values))
    return {"sex_arm_metrics.csv": arm_rows, "sex_group_metrics.csv": group_rows, "sex_matched_deltas.csv": deltas, "sex_decision_changes.csv": decisions, "sex_bootstrap_replicates.csv": reps, "sex_bootstrap_summary.csv": boots, "sex_across_seed_summary.csv": across, "sex_generalization_gaps.csv": gaps}


def _load_saved(arms: Sequence[rr.TestArmSource]) -> tuple[dict[str, ev.PartitionPredictions], dict[str, ev.PartitionPredictions], np.ndarray]:
    layout, contract = _source_layout(), ri.load_contract()
    test = {arm.artifact_id: rr.read_predictions(layout.stage_dir(rr.TEST_STAGE) / arm.prediction_name) for arm in arms}
    reference = next(iter(test.values()))
    _require(all(x.record_ids == reference.record_ids for x in test.values()), "Saved Test predictions have different ordered IDs.")
    validation = {arm.artifact_id: rr.read_predictions(arm.fit_dir.parent / arm.validation_key / "validation_predictions.csv") for arm in arms}
    return test, validation, rr._load_test_indices(contract, reference)


def _manifest(arms: Sequence[rr.TestArmSource], inputs: Mapping[str, str]) -> dict[str, Any]:
    return {"run_id": RUN_ID, "stage": STAGE, "protocol": "rq4b3_applicant_sex_supplementary_v1", "source_test_run_id": SOURCE_TEST_ROOT,
            "source_pins": dict(inputs), "sex_attribute": "applicant_sex", "sex_mapping": ev.RQ3_SEX_CODES, "primary_roster": list(SEX_ROSTER),
            "coverage_groups": list(SEX_GROUPS), "matched_comparisons": [list(x) for x in ri.TEST_MATCHED_COMPARISONS], "bootstrap_replicates": 1000,
            "prohibit_fit": True, "prohibit_inference": True, "research_interpretation_owner": "user", "output_schema": _schemas(),
            "arms": [{"lambda": a.adversary_lambda, "seed": a.seed, "artifact_id": a.artifact_id, "threshold": a.threshold} for a in arms]}


def _publish(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]], inputs: Mapping[str, str]) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    target, temp = ROOT / RUN_ID, ROOT / f".tmp_{RUN_ID}_{uuid.uuid4().hex}"
    _require(not target.exists(), "Supplementary run already exists and is immutable.")
    try:
        stage = temp / "stages" / STAGE; stage.mkdir(parents=True); (temp / "failures").mkdir()
        (temp / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name, rows in tables.items(): _write_csv(stage / name, rows, manifest["output_schema"][name])
        status = {"execution_status": "completed", "inference_performed": False, "fit_count": 0, "refit_count": 0, "cv_fit_count": 0, "research_interpretation_owner": "user"}
        (stage / "execution_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs = {x.name: _sha(x) for x in sorted(stage.iterdir())}
        receipt = {"run_id": RUN_ID, "stage": STAGE, "status": "PASS", "inputs": dict(inputs), "outputs": outputs, "counts": {"cv_fold_fits": 0, "refit_fits": 0, "inference_count": 0, "bootstrap_replicates": 1000}}
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(temp, target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True); raise
    return target


def evaluate() -> Path:
    arms, inputs = _verify_sources(); test, validation, indices = _load_saved(arms)
    _require(len(arms) == 9 and len(indices) == 1000, "Supplementary evaluation requires nine arms and 1,000 locked indices.")
    return _publish(_manifest(arms, inputs), _tables(arms, test, validation, indices), inputs)


def validate_package() -> dict[str, Any]:
    arms, inputs = _verify_sources(); run, stage = ROOT / RUN_ID, ROOT / RUN_ID / "stages" / STAGE
    manifest = _json(run / "run_manifest.json")
    _require(manifest == _manifest(arms, inputs), "Supplementary manifest differs from the locked contract.")
    receipt = _json(stage / "receipt.json")
    _require(receipt.get("status") == "PASS" and receipt.get("inputs") == inputs and set(receipt.get("outputs", {})) == set(OUTPUTS), "Supplementary receipt differs from the locked package.")
    for name, digest in receipt["outputs"].items(): _require(_sha(stage / name) == digest, f"Supplementary output digest differs: {name}.")
    test, validation, indices = _load_saved(arms); expected = _tables(arms, test, validation, indices)
    for name, rows in expected.items(): _require((stage / name).read_bytes() == _csv_bytes(rows, manifest["output_schema"][name]), f"Saved {name} does not match recomputation.")
    status = {"execution_status": "completed", "inference_performed": False, "fit_count": 0, "refit_count": 0, "cv_fit_count": 0, "research_interpretation_owner": "user"}
    _require(_json(stage / "execution_status.json") == status, "Supplementary execution status differs.")
    pkg = run / "stages" / PACKAGE_STAGE
    _require(not pkg.exists(), "Supplementary package validation already exists and is immutable.")
    temp = run / f".tmp_{PACKAGE_STAGE}_{uuid.uuid4().hex}"
    try:
        temp.mkdir(); report = {"execution_status": "completed", "recomputed_from_saved_predictions": True, "inference_performed": False, "local_fit_counts": {"cv_fold_fits": 0, "refit_fits": 0}, "verified_outputs": list(OUTPUTS)}
        (temp / "package_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rec = {"run_id": RUN_ID, "stage": PACKAGE_STAGE, "status": "PASS", "inputs": {STAGE: _sha(stage / "receipt.json")}, "outputs": {"package_report.json": _sha(temp / "package_report.json")}, "counts": {"cv_fold_fits": 0, "refit_fits": 0, "inference_count": 0}}
        (temp / "receipt.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.rename(temp, pkg)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True); raise
    return _json(pkg / "receipt.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("evaluate", "package_validation"), required=True); args = parser.parse_args(argv)
    result = evaluate() if args.stage == "evaluate" else validate_package()
    print(json.dumps({"status": "PASS", "result": str(result)}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
