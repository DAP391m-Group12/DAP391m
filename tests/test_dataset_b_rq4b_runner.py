"""Mocked end-to-end tests for the RQ4b staged runner.

All runs use a synthetic fake backend and live under
scratch/rq4b_implementation/T5_R1_runs_<uuid>/. No production artifact is
published, no model is trained and no project data is read. Synthetic
receipts are marked ``synthetic_test`` and cannot be validated as production.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.dataset_b.mitigation import rq4b_evaluation as ev
from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.mitigation import rq4b_runner as rr


ALL_GROUPS = (*ri.G6_GROUPS, *ri.COVERAGE_ONLY_GROUPS)
RUN_ID = "rq4b3_adv_20261018_01"


# ---------------------------------------------------------------------------
# Synthetic fake backend
# ---------------------------------------------------------------------------


def _partition(seed: int, prefix: str) -> ev.PartitionPredictions:
    rng = np.random.default_rng(seed)
    y = np.tile(np.repeat(np.array([0, 1], dtype=np.int8), 60), len(ALL_GROUPS))
    race = np.repeat(np.asarray(ALL_GROUPS, dtype=object), 120)
    sex = np.where(rng.random(len(y)) < 0.5, "Male", "Female").astype(object)
    gap = np.where(np.isin(race, ["Asian", "American Indian or Alaska Native"]), -0.08, 0.0)  # baseline disparity
    p = np.clip(0.3 + 0.4 * y + gap + rng.normal(0, 0.15, len(y)), 0.0, 1.0)
    ids = tuple(f"dataset_b-{prefix}{i:08d}" for i in range(len(y)))
    return ev.PartitionPredictions(ids, y, p, race, sex)


class FakeBackend:
    kind = "fake_synthetic"

    def __init__(self, *, statuses=None, gate="PASS", fail_units=(), adversary_helps=True, partial_failure=False):
        self.statuses = statuses or {"t5_runner_acceptance": "PASS", "cuda0": "PASS"}
        self.gate = gate
        self.fail_units = set(fail_units)
        self.adversary_helps = adversary_helps
        self.partial_failure = partial_failure
        self.validation = _partition(1, "1")
        self.test = _partition(2, "2")
        self.calls: list[tuple] = []

    def preflight(self):
        self.calls.append(("preflight",))
        return dict(self.statuses)

    def capability_evidence(self):
        return {"synthetic_gate_status": self.gate, "evidence_path": "synthetic", "evidence_sha256": "0" * 64}

    def fit_unit(self, adversary_lambda, seed, out_dir: Path):
        self.calls.append(("fit", adversary_lambda, seed))
        if self.partial_failure:
            (out_dir / "partial.bin").write_bytes(b"partial")
        if (adversary_lambda, seed) in self.fail_units:
            raise FloatingPointError(f"simulated nonfinite fit lambda={adversary_lambda} seed={seed}")
        (out_dir / "model.json").write_text(json.dumps({"adversary_lambda": adversary_lambda, "model_seed": seed}), encoding="utf-8")
        epochs = np.arange(1, 257)
        curves = np.vstack([1.0 + 0.001 * fold + 1e-4 * (epochs - 40) ** 2 for fold in range(5)])
        return {"cv_fold_fits": 5, "refit_fits": 1, "curves": curves, "evidence_kind": "test_double",
                "selection_record": {"selected_epoch": 40, "evidence_kind": "test_double"}, "diagnostics": {"fake": True}}

    def predict(self, fit_dir: Path, partition: str):
        model = json.loads((fit_dir / "model.json").read_text(encoding="utf-8"))
        self.calls.append(("predict", fit_dir.name, partition))
        base = self.validation if partition == "validation" else self.test
        lam = float(model["adversary_lambda"])
        rng = np.random.default_rng(model["model_seed"])
        if self.adversary_helps or lam == 0.0:
            # Lambda moves Asian/AIAN approval rates toward the other groups.
            effect = np.where(np.isin(base.race, ["Asian", "American Indian or Alaska Native"]), 0.08 * lam, 0.0)
            p = np.clip(base.p_approved + effect + rng.normal(0, 0.005, len(base.y)), 0.0, 1.0)
        else:
            # Any lambda > 0 destroys utility, so only lambda = 0 stays BA-eligible.
            p = np.clip(0.5 * base.p_approved + 0.5 * rng.uniform(size=len(base.y)), 0.0, 1.0)
        return ev.PartitionPredictions(base.record_ids, base.y, p, base.race, base.sex)

    def load_test_indices(self, y, record_ids):
        self.calls.append(("indices",))
        rng = np.random.default_rng(42)
        denied, approved = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        return np.asarray([np.concatenate([rng.choice(denied, len(denied)), rng.choice(approved, len(approved))]) for _ in range(1000)], dtype=np.int32)


@pytest.fixture(scope="module")
def contract():
    return ri.load_contract()


def _new_root() -> Path:
    root = rr.SYNTHETIC_ROOT / f"T5_R1_runs_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


def _runner(contract, backend=None, root=None, run_id=RUN_ID):
    root = root or _new_root()
    layout = rr.create_run(root, run_id, rr.MODE_SYNTHETIC, contract)
    return rr.RQ4bRunner(layout, backend or FakeBackend(), contract)


def _production_contract_with_test_sources(contract, root: Path):
    evidence = {}
    for label, source in contract.source_run_evidence.items():
        run_dir = root / source["run_id"]
        run_dir.mkdir(parents=True)
        manifest = run_dir / rr.MANIFEST_FILE
        manifest.write_text(json.dumps({"run_id": source["run_id"]}), encoding="utf-8")
        updated = {**source, "manifest_sha256": rr.sha256_file(manifest)}
        for digest_field in ("receipt_sha256", "failure_receipt_sha256"):
            receipts = {}
            for relative in source.get(digest_field, {}):
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"source": label, "path": relative}), encoding="utf-8")
                receipts[relative] = rr.sha256_file(path)
            if receipts:
                updated[digest_field] = receipts
        evidence[label] = updated
    return dataclasses.replace(contract, artifact_namespace=root, source_run_evidence=evidence)


def _development(runner):
    for lam, seed in ev.ADDITIONAL_DEVELOPMENT_UNITS:
        runner.development(lam, seed)
        runner.development_validation(lam, seed)


def _complete_track_b(runner):
    runner.preflight()
    runner.capability()
    _development(runner)
    return runner


@pytest.fixture(scope="module")
def track_b_completed(contract):
    return _complete_track_b(_runner(contract))


@pytest.fixture(scope="module")
def completed():
    pytest.skip("Legacy post-Gate-C package tests are inactive until the three-seed selection policy is explicitly approved.")


# ---------------------------------------------------------------------------
# End-to-end progression
# ---------------------------------------------------------------------------


def test_track_b_runs_only_the_four_additional_units(track_b_completed):
    runner = track_b_completed
    dev_stages = sorted(p.name for p in runner.layout.stages.glob("dev_*") if not p.name.startswith("devval_"))
    val_stages = sorted(p.name for p in runner.layout.stages.glob("devval_*"))
    assert len(dev_stages) == len(val_stages) == 4
    receipts = [rr.read_stage(runner.layout, rr.unit_key("dev", lam, seed)) for lam, seed in ev.ADDITIONAL_DEVELOPMENT_UNITS]
    assert sum(item["counts"]["cv_fold_fits"] for item in receipts) == 20
    assert sum(item["counts"]["refit_fits"] for item in receipts) == 4
    assert not runner.layout.stage_dir("selection").exists()
    assert not runner.layout.stage_dir("test_evaluation").exists()
    assert not list(runner.layout.stages.glob(".tmp_*"))


def test_required_track_b_evidence_is_exported(track_b_completed):
    runner = track_b_completed
    dev = runner.layout.stage_dir("dev_l0_s43")
    assert {"cv_curves.json", "cv_fold_outputs.json", "selection_record.json", "diagnostics.json", "model.json",
            "run.log", "training_history.jsonl", "receipt.json"} == {p.name for p in dev.iterdir()}
    assert np.asarray(json.loads((dev / "cv_curves.json").read_text())["curves"]).shape == (5, 256)
    val = runner.layout.stage_dir("devval_l0_s43")
    assert {"threshold.json", "validation_predictions.csv", "receipt.json"} == {p.name for p in val.iterdir()}


def test_selection_final_and_test_remain_closed_pending_policy(track_b_completed):
    runner = track_b_completed
    with pytest.raises(rr.RQ4bStageOrderError, match="methodological approval"):
        runner.validation_selection()
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive"):
        runner.final_fit(0.0, 43)
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive"):
        runner.final_validation(0.0, 43)
    with pytest.raises(rr.RQ4bStageOrderError, match="methodological approval"):
        runner.test_evaluation()
    assert not any(c[0] == "predict" and c[2] == "test" for c in runner.backend.calls)


def test_retained_units_are_not_refit_by_track_b(contract):
    runner = _runner(contract)
    runner.preflight()
    runner.capability()
    for lam, seed in ev.RETAINED_DEVELOPMENT_UNITS:
        with pytest.raises(rr.RQ4bStageOrderError, match="retained seed-42"):
            runner.development(lam, seed)
    assert not any(c[0] == "fit" for c in runner.backend.calls)


# ---------------------------------------------------------------------------
# Stage ordering and prerequisites
# ---------------------------------------------------------------------------


def test_forbidden_stage_ordering_fails_closed(contract):
    runner = _runner(contract)
    with pytest.raises(rr.RQ4bStageOrderError):
        runner.capability()
    runner.preflight()
    with pytest.raises(rr.RQ4bStageOrderError):
        runner.development(0.01, 43)
    runner.capability()
    with pytest.raises(rr.RQ4bStageOrderError):
        runner.development_validation(0.01, 43)
    runner.development(0.01, 43)
    runner.development_validation(0.01, 43)
    for call in (runner.validation_selection, runner.test_evaluation, runner.package_validation, lambda: runner.final_fit(0.01, 43)):
        with pytest.raises(rr.RQ4bStageOrderError):
            call()
    with pytest.raises(rr.RQ4bRunnerError):
        runner.development(0.2, 42)
    with pytest.raises(rr.RQ4bRunnerError):
        runner.development(0.1, 45)
    assert not any(c[0] == "predict" and c[2] == "test" for c in runner.backend.calls)


def test_gate_c_stays_locked_after_all_track_b_units(contract):
    runner = _runner(contract)
    runner.preflight()
    runner.capability()
    _development(runner)
    with pytest.raises(rr.RQ4bStageOrderError, match="methodological approval"):
        runner.validation_selection()
    with pytest.raises(rr.RQ4bStageOrderError, match="methodological approval"):
        runner.test_evaluation()
    assert not any(c[0] in ("indices",) or (c[0] == "predict" and c[2] == "test") for c in runner.backend.calls)


@pytest.mark.parametrize("statuses", [{"t5_runner_acceptance": None}, {"t5_runner_acceptance": "FAIL"}, {"cuda0": "PASS", "x": "NOT_RUN"}])
def test_unaccepted_gates_fail_preflight_with_receipt(contract, statuses):
    runner = _runner(contract, FakeBackend(statuses=statuses))
    with pytest.raises(rr.RQ4bRunnerError, match="not PASS"):
        runner.preflight()
    failures = list(runner.layout.failures.iterdir())
    assert len(failures) == 1 and failures[0].name.startswith("preflight.") and not runner.layout.stage_dir("preflight").exists()


def test_unaccepted_capability_gate_blocks_development(contract):
    runner = _runner(contract, FakeBackend(gate="FAIL"))
    runner.preflight()
    with pytest.raises(rr.RQ4bRunnerError, match="not an accepted PASS"):
        runner.capability()
    with pytest.raises(rr.RQ4bStageOrderError):
        runner.development(0.0, 43)


def test_real_probe_reporting_is_published_without_capability_promotion(contract):
    backend = FakeBackend()
    backend.capability_evidence = lambda: {"synthetic_gate_status": "PASS", "real_probe": {
        "status": "completed", "capability_status": "capability_inconclusive", "evidence_sha256": "a" * 64,
        "evidence": {"holdout": {"ce_gain_nats": -0.1}}}}
    runner = _runner(contract, backend)
    runner.preflight()
    runner.capability()
    report = rr.read_json(runner.layout.stage_dir("capability") / "capability.json")
    assert report["real_probe"]["status"] == "completed"
    assert report["real_probe"]["evidence"]["holdout"]["ce_gain_nats"] == -0.1
    assert report["capability_status"] == "capability_inconclusive"
    assert not any(c[0] == "fit" for c in backend.calls)
    # Inconclusive is not a new production gate: a synthetic development unit can proceed.
    assert runner.development(0.0, 43)["status"] == "PASS"


def test_invalid_real_evidence_retains_capability_failure_and_no_publication(contract):
    backend = FakeBackend()
    def invalid():
        raise ValueError("invalid real evidence")
    backend.capability_evidence = invalid
    runner = _runner(contract, backend)
    runner.preflight()
    with pytest.raises(ValueError, match="invalid real evidence"):
        runner.capability()
    assert not runner.layout.stage_dir("capability").exists()
    assert len(list(runner.layout.failures.glob("capability.*/FAILURE_RECEIPT.json"))) == 1
    with pytest.raises(rr.RQ4bStageOrderError, match="failed attempts"):
        runner.capability()
    with pytest.raises(rr.RQ4bStageOrderError):
        runner.development(0.0, 43)


# ---------------------------------------------------------------------------
# Collision, failure receipts and atomic publication
# ---------------------------------------------------------------------------


def test_collision_and_non_overwrite(contract):
    root = _new_root()
    runner = _runner(contract, root=root)
    runner.preflight()
    before = (runner.layout.stage_dir("preflight") / "receipt.json").read_bytes()
    with pytest.raises(rr.RQ4bStageOrderError, match="already published"):
        runner.preflight()
    assert (runner.layout.stage_dir("preflight") / "receipt.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        rr.create_run(root, RUN_ID, rr.MODE_SYNTHETIC, contract)
    with pytest.raises(rr.RQ4bRunnerError):
        rr.create_run(root, "rq4b3_adv_bad", rr.MODE_SYNTHETIC, contract)


def test_failed_attempt_is_retained_and_not_silently_retried(contract):
    backend = FakeBackend(fail_units={(0.01, 43)}, partial_failure=True)
    runner = _runner(contract, backend)
    runner.preflight()
    runner.capability()
    with pytest.raises(FloatingPointError, match="simulated nonfinite"):
        runner.development(0.01, 43)
    failure, = list(runner.layout.failures.iterdir())
    receipt = json.loads((failure / rr.FAILURE_FILE).read_text())
    assert receipt["execution_status"] == "failed" and receipt["exception_type"] == "FloatingPointError" and receipt["automatic_retry"] is False
    assert set(receipt["partial_outputs"]) == {"partial.bin", "run.log", "training_history.jsonl"}
    assert not runner.layout.stage_dir("dev_l0p01_s43").exists()
    assert receipt["exception_chain"] and receipt["counts"]["fit_accounting"]["unknown"] is True
    assert receipt["last_position"]["phase"] == "backend_dispatch" and receipt["last_progress_utc"]
    assert not list(runner.layout.stages.glob(".tmp_*"))  # atomic: nothing half-published
    with pytest.raises(rr.RQ4bStageOrderError, match="failed attempts"):
        runner.development(0.01, 43)
    backend.fail_units.clear()
    published = runner.development(0.01, 43, allow_new_attempt_after_failure=True)
    assert published["prior_failed_attempts"] == [failure.name] and (failure / rr.FAILURE_FILE).exists()
    assert [c for c in backend.calls if c[0] == "fit" and c[1:] == (0.01, 43)] == [("fit", 0.01, 43)] * 2


def test_failed_attempt_uses_live_fit_ledger_instead_of_zero_compute(contract):
    class LedgerFailureBackend(FakeBackend):
        def fit_unit(self, adversary_lambda, seed, out_dir: Path):
            (out_dir / "fit_ledger.json").write_text(json.dumps({
                "planned": {"cv_fold_fits": 5, "refit_fits": 1},
                "started": {"cv_fold_fits": 2, "refit_fits": 0},
                "completed": {"cv_fold_fits": 1, "refit_fits": 0},
                "partial": {"cv_fold_fits": 1, "refit_fits": 0},
                "unknown": False,
                "last_position": {"phase": "cv", "fold": 2, "epoch": 17, "batch": 44, "event": "heartbeat"},
                "last_progress_utc": "2026-10-19T03:00:00+00:00",
            }), encoding="utf-8")
            raise RuntimeError("interrupted after durable progress")

    runner = _runner(contract, LedgerFailureBackend())
    runner.preflight()
    runner.capability()
    with pytest.raises(RuntimeError, match="interrupted after durable progress"):
        runner.development(0.01, 43)
    receipt_path, = runner.layout.failures.glob("dev_l0p01_s43.*/FAILURE_RECEIPT.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    accounting = receipt["counts"]["fit_accounting"]
    assert accounting["started"] == {"cv_fold_fits": 2, "refit_fits": 0}
    assert accounting["completed"] == {"cv_fold_fits": 1, "refit_fits": 0}
    assert accounting["partial"] == {"cv_fold_fits": 1, "refit_fits": 0}
    assert accounting["unknown"] is False and receipt["last_position"]["epoch"] == 17


def test_publication_failure_retains_attempt_and_preserves_exception(contract, monkeypatch):
    runner = _runner(contract)
    runner.preflight()

    def failing_rename(src, dst):
        raise PermissionError("simulated rename failure")

    original = rr.os.rename
    monkeypatch.setattr(rr.os, "rename", lambda src, dst: failing_rename(src, dst) if Path(dst).parent == runner.layout.stages else original(src, dst))
    with pytest.raises(PermissionError, match="simulated rename failure"):
        runner.capability()
    assert not runner.layout.stage_dir("capability").exists()
    failure, = list(runner.layout.failures.iterdir())
    assert "capability.json" in json.loads((failure / rr.FAILURE_FILE).read_text())["partial_outputs"]


# ---------------------------------------------------------------------------
# Package integrity and report tracing
# ---------------------------------------------------------------------------


def _copy_run(completed) -> rr.RQ4bRunner:
    runner, _ = completed
    root = _new_root()
    shutil.copytree(runner.layout.run_dir, root / RUN_ID)
    for path in (root / RUN_ID).rglob("*"):
        if path.is_file():
            path.chmod(0o666)
    shutil.rmtree(root / RUN_ID / "stages" / "package_validation")
    layout = rr.open_run(root, RUN_ID, rr.MODE_SYNTHETIC, runner.contract)
    return rr.RQ4bRunner(layout, FakeBackend(), runner.contract)


@pytest.mark.parametrize("damage", ["corrupt", "missing", "extra"])
def test_missing_corrupt_or_extra_package_files_are_rejected(completed, damage):
    copy = _copy_run(completed)
    stage = copy.layout.stage_dir("test_evaluation")
    if damage == "corrupt":
        (stage / "seed_43.json").write_text("{}", encoding="utf-8")
    elif damage == "missing":
        (stage / "assessment.json").unlink()
    else:
        (stage / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError):
        copy.package_validation()
    assert not copy.layout.stage_dir("package_validation").exists()


def test_report_values_must_trace_to_saved_inputs(completed):
    copy = _copy_run(completed)
    stage = copy.layout.stage_dir("test_evaluation")
    assessment = json.loads((stage / "assessment.json").read_text())
    assessment["strong_support"]["supported_count"] += 1
    (stage / "assessment.json").write_text(json.dumps(assessment), encoding="utf-8")
    receipt = json.loads((stage / "receipt.json").read_text())
    receipt["outputs"]["assessment.json"] = rr.sha256_file(stage / "assessment.json")  # forge a consistent receipt
    (stage / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="does not trace"):
        copy.package_validation()


def _forge(copy: rr.RQ4bRunner, name: str, mutate) -> None:
    """Edit one test_evaluation output and rewrite its receipt hash so read_stage still passes."""
    stage = copy.layout.stage_dir("test_evaluation")
    payload = json.loads((stage / name).read_text(encoding="utf-8"))
    mutate(payload)
    (stage / name).write_text(json.dumps(payload), encoding="utf-8")
    receipt = json.loads((stage / "receipt.json").read_text(encoding="utf-8"))
    receipt["outputs"][name] = rr.sha256_file(stage / name)
    (stage / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    rr.read_stage(copy.layout, "test_evaluation")  # the forgery is internally consistent


def _set(path, value):
    def mutate(payload):
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


@pytest.mark.parametrize(
    "name, path, value, message",
    [
        ("status.json", ("execution_status",), "failed", "status.json"),
        ("status.json", ("strong_support",), "supported_by_fiat", "status.json"),
        ("status.json", ("stability_status",), "stable", "status.json"),
        ("status.json", ("primary_criterion_per_seed", "42"), "pass_by_fiat", "status.json"),
        ("seed_42.json", ("lineage", "control_threshold"), 0.123, "lineage"),
        ("seed_43.json", ("lineage", "adversarial_artifact"), "forged_artifact_s43", "lineage"),
        ("seed_44.json", ("lineage", "test_record_id_sha256"), "0" * 64, "lineage"),
        ("seed_45.json", ("lineage", "deduplicated_lambda_zero"), True, "lineage"),
        ("seed_42.json", ("fairness_generalization_gap", "control", "eod3", "value"), 0.5, "fairness_generalization_gap"),
        ("seed_46.json", ("fairness_generalization_gap", "adversarial", "eod6", "validation_eod"), None, "fairness_generalization_gap"),
        ("seed_43.json", ("fairness_generalization_gap", "adversarial", "threshold"), 0.9, "fairness_generalization_gap"),
    ],
    ids=["exec", "support", "stability", "criterion", "threshold", "artifact", "test_ids", "dedup", "gap_value", "gap_null", "gap_threshold"],
)
def test_forged_status_lineage_or_gap_is_rejected(completed, name, path, value, message):
    copy = _copy_run(completed)
    stage = copy.layout.stage_dir("test_evaluation")
    before = (stage / name).read_bytes()
    _forge(copy, name, _set(path, value))
    assert json.loads((stage / name).read_text(encoding="utf-8")) != json.loads(before)  # the forgery really changed a value
    with pytest.raises(rr.RQ4bPackageError, match=message):
        copy.package_validation()
    assert not copy.layout.stage_dir("package_validation").exists()


@pytest.mark.parametrize("which", ["first", "last"])
def test_forged_test_prediction_ids_are_rejected_per_artifact(completed, which):
    copy = _copy_run(completed)
    stage = copy.layout.stage_dir("test_evaluation")
    files = sorted(stage.glob("test_predictions_*.csv"))
    target = files[0] if which == "first" else files[-1]
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    target.write_text(lines[0] + "".join(lines[1:][::-1]), encoding="utf-8")  # same rows, different Test ID order
    receipt = json.loads((stage / "receipt.json").read_text(encoding="utf-8"))
    receipt["outputs"][target.name] = rr.sha256_file(target)
    (stage / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    rr.read_stage(copy.layout, "test_evaluation")
    with pytest.raises(rr.RQ4bPackageError, match="record IDs differ|different ordered Test IDs"):
        copy.package_validation()


def test_fairness_generalization_gap_is_exported_from_locked_thresholds(completed):
    runner, _ = completed
    seed = json.loads((runner.layout.stage_dir("test_evaluation") / "seed_42.json").read_text(encoding="utf-8"))
    gap = seed["fairness_generalization_gap"]
    assert set(gap) == {"control", "adversarial"}
    for role, threshold_key in (("control", "control_threshold"), ("adversarial", "adversarial_threshold")):
        entry = gap[role]
        assert entry["threshold"] == seed["lineage"][threshold_key] and entry["development_partition"] == "validation"
        for name in ("eod3", "eod6"):
            assert entry[name]["value"] == entry[name]["test_eod"] - entry[name]["validation_eod"]
        assert entry["eod3"]["test_eod"] == seed[role]["eod3"]["value"]  # same Test EOD as the reported arm metric
    validation = rr.read_predictions(runner.layout.stage_dir(seed["lineage"]["control_artifact"].replace("dev_", "devval_", 1)) / "validation_predictions.csv")
    expected = ev.core_quantities(validation, seed["lineage"]["control_threshold"])["eod3"].value
    assert gap["control"]["eod3"]["validation_eod"] == expected
    predict_calls = [c for c in runner.backend.calls if c[0] == "predict"]
    assert len(predict_calls) == 32  # 22 Validation + 10 Test: the gap adds no inference


def test_gap_preserves_undefined_values_and_reasons():
    rng = np.random.default_rng(3)
    groups = tuple(g for g in ALL_GROUPS if g != "Asian")
    y = np.tile(np.repeat(np.array([0, 1], dtype=np.int8), 60), len(groups))
    race = np.repeat(np.asarray(groups, dtype=object), 120)
    sex = np.where(rng.random(len(y)) < 0.5, "Male", "Female").astype(object)
    test = ev.PartitionPredictions(tuple(f"t{i}" for i in range(len(y))), y, rng.uniform(size=len(y)), race, sex)
    validation = _partition(1, "1")
    gap = rr.generalization_gap(test, validation, 0.5)
    assert gap["eod3"]["value"] is None and gap["eod3"]["test_eod"] is None and gap["eod3"]["validation_eod"] is not None
    assert gap["eod3"]["undefined_reasons"] == {"test": {"reason": "required_group_metric_invalid", "invalid_groups": ["Asian"]}}
    assert gap["eod6"]["value"] is None and "test" in gap["eod6"]["undefined_reasons"]


def test_changed_upstream_receipt_breaks_lineage(completed):
    copy = _copy_run(completed)
    receipt_path = copy.layout.stage_dir("selection") / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["finished_utc"] = "tampered"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="changed after use"):
        copy.package_validation()


def test_swapped_final_model_is_rejected_before_test_is_opened(contract):
    runner = _runner(contract)
    runner.preflight()
    runner.capability()
    _development(runner)
    with pytest.raises(rr.RQ4bStageOrderError, match="methodological approval"):
        runner.validation_selection()
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive"):
        runner.final_fit(0.0, 43)
    with pytest.raises(rr.RQ4bStageOrderError, match="methodological approval"):
        runner.test_evaluation()
    assert not any(c[0] == "indices" or (c[0] == "predict" and c[2] == "test") for c in runner.backend.calls)


def test_intact_copy_still_validates(completed):
    copy = _copy_run(completed)
    assert copy.package_validation()["status"] == "PASS"


# ---------------------------------------------------------------------------
# Inference guard, modes and CLI
# ---------------------------------------------------------------------------


def test_inference_guard_refuses_second_validation_prediction(track_b_completed):
    runner = track_b_completed
    writer = rr.StageWriter(runner.layout, "probe_guard", stage="guard", unit={}, inputs={})
    with pytest.raises(rr.RQ4bStageOrderError, match="already predicted"):
        with writer:
            runner._predict_once(writer, "dev_l0_s43", "validation", "again.csv")


def test_modes_cannot_masquerade(contract):
    with pytest.raises(rr.RQ4bRunnerError, match="continuation run"):
        rr.create_run(_new_root(), contract.track_b_child_run_id, rr.MODE_PRODUCTION, contract)
    with pytest.raises(rr.RQ4bRunnerError, match="scratch"):
        rr.create_run(contract.artifact_namespace, RUN_ID, rr.MODE_SYNTHETIC, contract)
    production_layout = rr.RunLayout(contract.artifact_namespace, RUN_ID, rr.MODE_PRODUCTION)
    with pytest.raises(rr.RQ4bRunnerError, match="ProductionBackend"):
        rr.RQ4bRunner(production_layout, FakeBackend(), contract)
    synthetic = _runner(contract)
    with pytest.raises(rr.RQ4bRunnerError, match="may not use the ProductionBackend"):
        rr.RQ4bRunner(synthetic.layout, rr.ProductionBackend(contract), contract)
    with pytest.raises(rr.RQ4bPackageError):
        rr.open_run(synthetic.layout.root, "rq4b3_adv_20261018_99", rr.MODE_SYNTHETIC, contract)  # no such run
    assert not (contract.artifact_namespace / RUN_ID).exists()


def test_production_run_is_locked_to_seed44_continuation_and_records_pinned_sources(contract):
    root = _new_root()
    scoped = _production_contract_with_test_sources(contract, root)
    with pytest.raises(rr.RQ4bRunnerError, match="approved continuation run"):
        rr.create_run(root, contract.track_a_source_run_id, rr.MODE_PRODUCTION, scoped)
    layout = rr.create_run(root, contract.production_run_id, rr.MODE_PRODUCTION, scoped)
    manifest = rr.read_json(layout.run_dir / rr.MANIFEST_FILE)
    assert manifest["execution_lineage"] == rr.execution_lineage(scoped)
    assert manifest["execution_lineage"]["track_b_seed43_source_run_id"] == contract.track_b_child_run_id
    assert manifest["execution_lineage"]["source_run_evidence"] == scoped.source_run_evidence
    assert rr.open_run(root, contract.production_run_id, rr.MODE_PRODUCTION, scoped).run_dir == layout.run_dir
    production = rr.RQ4bRunner(layout, rr.ProductionBackend(scoped), scoped)
    with pytest.raises(rr.RQ4bStageOrderError, match="only the two seed-44 continuation units"):
        production.development(0.0, 43)
    with pytest.raises(rr.RQ4bStageOrderError):
        production.development(0.01, 44)
    manifest["execution_lineage"]["source_runs_are_immutable"] = False
    (layout.run_dir / rr.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="execution lineage"):
        rr.open_run(root, contract.production_run_id, rr.MODE_PRODUCTION, scoped)


def test_source_evidence_digest_mismatch_blocks_continuation_creation(contract):
    root = _new_root()
    scoped = _production_contract_with_test_sources(contract, root)
    source = next(iter(scoped.source_run_evidence.values()))
    (root / source["run_id"] / rr.MANIFEST_FILE).write_text("tampered", encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="source manifest digest differs"):
        rr.create_run(root, contract.production_run_id, rr.MODE_PRODUCTION, scoped)
    assert not (root / contract.production_run_id).exists()


def test_published_source_evidence_matches_locked_continuation_contract(contract):
    rr.verify_source_run_evidence(contract.artifact_namespace, contract)


def test_seed44_lambda_0_01_requires_verified_lambda_zero_artifacts(contract, monkeypatch):
    root = _new_root()
    scoped = _production_contract_with_test_sources(contract, root)
    fake = FakeBackend()
    scoped = dataclasses.replace(scoped, split_rows={**scoped.split_rows, "validation": len(fake.validation.y)})
    layout = rr.create_run(root, scoped.production_run_id, rr.MODE_PRODUCTION, scoped)
    backend = rr.ProductionBackend(scoped)
    monkeypatch.setattr(backend, "preflight", fake.preflight)
    monkeypatch.setattr(backend, "capability_evidence", fake.capability_evidence)
    monkeypatch.setattr(backend, "fit_unit", fake.fit_unit)
    monkeypatch.setattr(backend, "predict", fake.predict)
    runner = rr.RQ4bRunner(layout, backend, scoped)
    runner.preflight()
    runner.capability()
    with pytest.raises(rr.RQ4bStageOrderError):
        runner.development(0.01, 44)
    runner.development(0.0, 44)
    runner.development_validation(0.0, 44)
    receipt = runner.development(0.01, 44)
    assert set(receipt["inputs"]) >= {"dev_l0_s44", "devval_l0_s44"}


def test_code_change_blocks_resuming_a_run(contract):
    runner = _runner(contract)
    manifest_path = runner.layout.run_dir / rr.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["code_sha256"]["rq4b_runner.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="code changed"):
        rr.open_run(runner.layout.root, RUN_ID, rr.MODE_SYNTHETIC, contract)


def test_config_change_blocks_resuming_a_run(contract):
    runner = _runner(contract)
    manifest_path = runner.layout.run_dir / rr.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["config_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="config changed"):
        rr.open_run(runner.layout.root, RUN_ID, rr.MODE_SYNTHETIC, contract)


@pytest.mark.parametrize(
    "argv",
    [[], ["--stage", "all", "--run-id", RUN_ID, "--confirm-production"], ["--stage", "preflight", "--run-id", RUN_ID],
     ["--run-id", RUN_ID, "--confirm-production"]],
    ids=["no_args", "no_whole_grid_stage", "missing_confirmation", "missing_stage"],
)
def test_cli_requires_explicit_production_stage_commands(argv):
    with pytest.raises(SystemExit):
        rr.main(argv)
    assert "all" not in rr.STAGE_COMMANDS
