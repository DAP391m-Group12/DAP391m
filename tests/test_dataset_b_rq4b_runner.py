"""Mocked end-to-end tests for the RQ4b staged runner.

All runs use a synthetic fake backend and live under
scratch/rq4b_implementation/T5_R1_runs_<uuid>/. No production artifact is
published, no model is trained and no project data is read. Synthetic
receipts are marked ``synthetic_test`` and cannot be validated as production.
"""

from __future__ import annotations

import dataclasses
import inspect
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
    y = np.tile(np.repeat(np.array([0, 1], dtype=np.int8), 30), len(ALL_GROUPS))
    race = np.repeat(np.asarray(ALL_GROUPS, dtype=object), 60)
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


def test_in_run_selection_final_and_test_remain_closed_under_nine_unit_contract(track_b_completed):
    runner = track_b_completed
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive under the nine-unit contract"):
        runner.validation_selection()
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive"):
        runner.final_fit(0.0, 43)
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive"):
        runner.final_validation(0.0, 43)
    with pytest.raises(rr.RQ4bStageOrderError, match="local-run Test route is retired"):
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
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive under the nine-unit contract"):
        runner.validation_selection()
    with pytest.raises(rr.RQ4bStageOrderError, match="local-run Test route is retired"):
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
# Legacy local-run package route
# ---------------------------------------------------------------------------


def test_legacy_local_test_and_package_routes_are_retired(contract):
    runner = _runner(contract)
    with pytest.raises(rr.RQ4bStageOrderError, match="local-run Test route is retired"):
        runner.test_evaluation()
    with pytest.raises(rr.RQ4bStageOrderError, match="local-run package route is retired"):
        runner.package_validation()


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


# ---------------------------------------------------------------------------
# Gate C evaluation-only aggregation (synthetic pinned source runs)
# ---------------------------------------------------------------------------


UNAUTHORIZED_GATE_C_ROOT = rr.PROJECT_ROOT / "artifacts" / "dataset_b" / "rq4" / "rq4b" / "unauthorized_gate_c"


def _gate_c_predictions(lam: float, seed: int, base: ev.PartitionPredictions) -> ev.PartitionPredictions:
    rng = np.random.default_rng(1000 * seed + int(round(lam * 100)))
    effect = np.where(np.isin(base.race, ["Asian", "American Indian or Alaska Native"]), 0.08 * lam, 0.0)
    p = np.clip(base.p_approved + effect + rng.normal(0, 0.005, len(base.y)), 0.0, 1.0)
    return ev.PartitionPredictions(base.record_ids, base.y, p, base.race, base.sex)


def _write_gate_c_unit(layout: rr.RunLayout, lam: float, seed: int, predictions: ev.PartitionPredictions) -> None:
    """One completed source unit as the runner publishes it: fit receipt (5 CV + 1 refit) and Validation receipt."""
    fit_key, val_key = rr.unit_key("dev", lam, seed), rr.unit_key("devval", lam, seed)
    unit = {"adversary_lambda": lam, "model_seed": seed}
    with rr.StageWriter(layout, fit_key, stage="synthetic_gate_c_source_fit_record", unit=unit, inputs={}) as writer:
        writer.counts.update({"cv_fold_fits": 5, "refit_fits": 1})
        writer.write_json("model.json", unit)
        writer.path("estimator.joblib").write_bytes(f"estimator:{lam}:{seed}".encode())
        writer.path("preprocessor.joblib").write_bytes(f"preprocessor:{lam}:{seed}".encode())
    with rr.StageWriter(layout, val_key, stage="synthetic_gate_c_source_validation_record", unit=unit,
                        inputs={fit_key: rr.receipt_sha(layout, fit_key)}) as writer:
        writer.counts["predictions"]["validation_predictions.csv"] = {
            "artifact": fit_key, "partition": "validation", "rows": len(predictions.record_ids),
            "record_id_sha256": rr.sha256_ids(predictions.record_ids)}
        writer.write_predictions("validation_predictions.csv", predictions)
        writer.write_json("threshold.json", {"threshold": ev.calibrate_threshold(predictions.y, predictions.p_approved).threshold})


def _build_gate_c_sources(contract, root: Path, predictions_for=_gate_c_predictions) -> Path:
    base = _partition(1, "1")
    for run_id, seed in rr.gate_c_source_seeds(contract).items():
        layout = rr.create_run(root, run_id, rr.MODE_SYNTHETIC, contract)
        for lam, unit_seed in ev.DEVELOPMENT_UNITS:
            if unit_seed == seed:
                _write_gate_c_unit(layout, lam, seed, predictions_for(lam, seed, base))
    return root


def _pin_gate_c_sources(contract, root: Path):
    """Exact pins computed from the synthetic sources, in the same shape as the production contract."""
    pins = {}
    for run_id, seed in rr.gate_c_source_seeds(contract).items():
        run_dir = root / run_id
        receipts = {}
        for lam, unit_seed in ev.DEVELOPMENT_UNITS:
            for prefix in ("dev", "devval"):
                if unit_seed == seed:
                    relative = f"stages/{rr.unit_key(prefix, lam, seed)}/{rr.RECEIPT_FILE}"
                    receipts[relative] = rr.sha256_file(run_dir / relative)
        pins[run_id] = {"manifest_sha256": rr.sha256_file(run_dir / rr.MANIFEST_FILE), "receipt_sha256": receipts}
    base = _partition(1, "1")
    return dataclasses.replace(contract, aggregation_source_pins=pins, aggregation_validation_rows=len(base.record_ids),
                               aggregation_validation_record_id_sha256=rr.sha256_ids(base.record_ids))


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): rr.sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _tree_state(root: Path) -> tuple[bool, dict[str, str]]:
    """Snapshot existence and file digests without assuming publication state."""
    return root.exists(), _tree(root) if root.exists() else {}


def _forbid_backend(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Gate C aggregation must not construct a backend, fit, infer or open a partition.")

    class ForbiddenProductionBackend:
        def __init__(self, *args, **kwargs):
            forbidden()

    monkeypatch.setattr(rr, "ProductionBackend", ForbiddenProductionBackend)
    monkeypatch.setattr(ri, "StageAccess", forbidden)
    monkeypatch.setattr(ri, "load_partition", forbidden)
    for name in ("preflight", "fit_unit", "predict", "load_test_indices"):
        monkeypatch.setattr(FakeBackend, name, forbidden)


@pytest.fixture(scope="module")
def gate_c_template(contract):
    return _build_gate_c_sources(contract, _new_root() / "sources")


def _gate_c_case(contract, template: Path):
    base = _new_root()
    sources = base / "sources"
    shutil.copytree(template, sources)
    return sources, base / "gate_c_aggregation", _pin_gate_c_sources(contract, sources)


def test_gate_c_aggregation_publishes_from_exactly_three_pinned_sources(contract, gate_c_template, monkeypatch):
    sources, target, scoped = _gate_c_case(contract, gate_c_template)
    before = _tree(sources)
    _forbid_backend(monkeypatch)
    layout = rr.aggregate_gate_c_sources(sources, target, scoped)
    assert {p.name for p in target.iterdir()} == {scoped.aggregation_run_id} and layout.mode == "evaluation_aggregation_synthetic_test"
    assert _tree(layout.run_dir).keys() == {rr.MANIFEST_FILE, "stages/gate_c_aggregation/receipt.json", "stages/gate_c_aggregation/gate_c_selection.json"}
    receipt = rr.read_stage(layout, rr.GATE_C_STAGE)
    assert receipt["counts"]["cv_fold_fits"] == receipt["counts"]["refit_fits"] == receipt["counts"]["probe_fits"] == 0
    assert receipt["counts"]["predictions"] == {}
    assert receipt["counts"]["source_fit_accounting"] == {"cv_fold_fits": 45, "refit_fits": 9}
    expected_inputs = {f"{run_id}:{relative.split('/')[1]}": digest
                       for run_id, pin in scoped.aggregation_source_pins.items() for relative, digest in pin["receipt_sha256"].items()}
    assert receipt["inputs"] == expected_inputs and len(expected_inputs) == 18
    manifest = rr.read_json(layout.run_dir / rr.MANIFEST_FILE)
    assert manifest["source_run_ids"] == list(scoped.aggregation_source_run_ids)
    assert manifest["source_pins"] == rr.jsonable(scoped.aggregation_source_pins) and manifest["source_mode"] == rr.MODE_SYNTHETIC
    assert (manifest["evaluation_only"], manifest["prohibit_fit"], manifest["prohibit_inference"]) == (True, True, True)
    selection = rr.read_json(layout.stage_dir(rr.GATE_C_STAGE) / rr.GATE_C_SELECTION_FILE)
    # Independent recomputation from the saved Validation predictions under the unchanged locked policy.
    run_for_seed = {seed: run_id for run_id, seed in rr.gate_c_source_seeds(scoped).items()}
    evaluations = []
    for lam, seed in ev.DEVELOPMENT_UNITS:
        saved = rr.read_predictions(sources / run_for_seed[seed] / "stages" / rr.unit_key("devval", lam, seed) / "validation_predictions.csv")
        evaluations.append(ev.evaluate_candidate(saved, adversary_lambda=lam, model_seed=seed))
    expected = rr.jsonable(ev.select_lambda(evaluations))
    assert {key: selection[key] for key in expected} == expected
    assert [row["adversary_lambda"] for row in selection["candidates"]] == [0.0, 0.01]
    assert selection["screening_only_lambdas"] == [0.03, 0.1, 0.3] and selection["eod6_role"] == "audit_only"
    assert selection["selected_lambda"] in (0.0, 0.01) and len(selection["units"]) == 9
    assert selection["validation_record_id_sha256"] == scoped.aggregation_validation_record_id_sha256
    assert selection["test_status"] == manifest["test_status"] == "blocked_pending_separate_methodological_gate"
    assert _tree(sources) == before  # source runs are read-only evidence
    with pytest.raises(rr.RQ4bStageOrderError, match="already published"):
        rr.aggregate_gate_c_sources(sources, target, scoped)


def test_gate_c_aggregation_has_no_backend_fit_inference_or_test_path(contract, gate_c_template, monkeypatch):
    assert list(inspect.signature(rr.aggregate_gate_c_sources).parameters) == ["source_root", "aggregation_root", "contract"]
    sources, target, scoped = _gate_c_case(contract, gate_c_template)
    _forbid_backend(monkeypatch)
    layout = rr.aggregate_gate_c_sources(sources, target, scoped)
    assert not any("test" in name.split("/")[-1] for name in _tree(layout.run_dir))
    assert not any("test" in name for name in _tree(sources))  # no Test prediction or index in any source either
    selection = rr.read_json(layout.stage_dir(rr.GATE_C_STAGE) / rr.GATE_C_SELECTION_FILE)
    assert selection["test_status"] == "blocked_pending_separate_methodological_gate"
    runner = rr.RQ4bRunner(rr.RunLayout(sources, scoped.production_run_id, rr.MODE_SYNTHETIC), FakeBackend(), scoped)
    with pytest.raises(rr.RQ4bStageOrderError, match="local-run Test route is retired"):
        runner.test_evaluation()
    with pytest.raises(rr.RQ4bStageOrderError, match="inactive under the nine-unit contract"):
        runner.validation_selection()


def _missing_source_run(sources, scoped):
    shutil.rmtree(sources / scoped.production_run_id)
    return scoped


def _rewrite_manifest_identity(sources, scoped):
    path = sources / scoped.track_b_child_run_id / rr.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["run_id"] = scoped.production_run_id
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return _pin_gate_c_sources(scoped, sources)  # pins match: only the identity is wrong


def _reorder_source_run_ids(sources, scoped):
    return dataclasses.replace(scoped, aggregation_source_run_ids=tuple(reversed(scoped.aggregation_source_run_ids)))


def _tamper_manifest(sources, scoped):
    path = sources / scoped.track_a_source_run_id / rr.MANIFEST_FILE
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    return scoped


def _tamper_receipt(sources, scoped):
    path = sources / scoped.production_run_id / "stages" / "dev_l0p01_s44" / rr.RECEIPT_FILE
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["finished_utc"] = "tampered"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return scoped


def _tamper_output(sources, scoped):
    path = sources / scoped.track_b_child_run_id / "stages" / "devval_l0_s43" / "validation_predictions.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fields = lines[1].split(",")
    fields[2] = "0.5"
    path.write_text(lines[0] + ",".join(fields) + "".join(lines[2:]), encoding="utf-8")
    return scoped


def _add_unit(lam, seed):
    def damage(sources, scoped):
        layout = rr.RunLayout(sources, scoped.track_b_child_run_id, rr.MODE_SYNTHETIC)
        _write_gate_c_unit(layout, lam, seed, _gate_c_predictions(lam, seed, _partition(1, "1")))
        return scoped
    return damage


def _remove_stages(*keys):
    def damage(sources, scoped):
        for key in keys:
            shutil.rmtree(sources / scoped.production_run_id / "stages" / key)
        return scoped
    return damage


def _wrong_pinned_population(sources, scoped):
    return dataclasses.replace(scoped, aggregation_validation_record_id_sha256="0" * 64)


GATE_C_DAMAGE = {
    "missing_source_run": (_missing_source_run, "source run rq4b3_adv_20261022_01 is missing"),
    "source_run_id_mismatch": (_rewrite_manifest_identity, "identity or mode differs"),
    "source_run_order_mismatch": (_reorder_source_run_ids, "source run ids differ"),
    "manifest_digest_mismatch": (_tamper_manifest, "manifest digest differs"),
    "receipt_digest_mismatch": (_tamper_receipt, "receipt digest differs"),
    "output_changed": (_tamper_output, "corrupt"),
    "duplicate_development_unit": (_add_unit(0.0, 42), "Duplicate Gate C development units"),
    "unexpected_development_unit": (_add_unit(0.03, 43), "Unexpected Gate C development units"),
    "missing_development_unit": (_remove_stages("dev_l0p01_s44", "devval_l0p01_s44"), "Missing Gate C development units"),
    "unpaired_development_unit": (_remove_stages("devval_l0_s44"), "unpaired"),
    "pinned_population_mismatch": (_wrong_pinned_population, "pinned ordered population"),
}


@pytest.mark.parametrize("case", sorted(GATE_C_DAMAGE))
def test_gate_c_source_validation_failures_publish_nothing(contract, gate_c_template, monkeypatch, case):
    sources, target, scoped = _gate_c_case(contract, gate_c_template)
    damage, message = GATE_C_DAMAGE[case]
    scoped = damage(sources, scoped)
    before = _tree(sources)
    _forbid_backend(monkeypatch)
    with pytest.raises(rr.RQ4bRunnerError, match=message):
        rr.aggregate_gate_c_sources(sources, target, scoped)
    assert not target.exists()  # validation fails before any aggregation file or directory is created
    assert _tree(sources) == before


def _variant_predictions(kind: str):
    def predictions_for(lam, seed, base):
        out = _gate_c_predictions(lam, seed, base)
        if (lam, seed) != (0.01, 44):
            return out
        if kind == "population":
            return ev.PartitionPredictions((*out.record_ids[:-1], "dataset_b-999999999"), out.y, out.p_approved, out.race, out.sex)
        order = np.arange(len(out.y))[::-1]
        return ev.PartitionPredictions(tuple(np.asarray(out.record_ids)[order]), out.y[order], out.p_approved[order], out.race[order], out.sex[order])
    return predictions_for


@pytest.mark.parametrize("kind, message", [("population", "Validation population differs"), ("ordering", "Validation record ID ordering differs")])
def test_gate_c_rejects_inconsistent_validation_population_or_order(contract, monkeypatch, kind, message):
    base = _new_root()
    sources = _build_gate_c_sources(contract, base / "sources", _variant_predictions(kind))
    scoped = _pin_gate_c_sources(contract, sources)  # receipts and outputs are internally consistent
    target = base / "gate_c_aggregation"
    _forbid_backend(monkeypatch)
    with pytest.raises(rr.RQ4bPackageError, match=message):
        rr.aggregate_gate_c_sources(sources, target, scoped)
    assert not target.exists()


def test_gate_c_publication_failure_leaves_no_run_or_partial_manifest(contract, gate_c_template, monkeypatch):
    sources, target, scoped = _gate_c_case(contract, gate_c_template)
    original = rr.os.rename

    def rename(src, dst):
        if Path(dst).parent == target.resolve():
            raise PermissionError("simulated publish failure")
        return original(src, dst)

    monkeypatch.setattr(rr.os, "rename", rename)
    with pytest.raises(PermissionError, match="simulated publish failure"):
        rr.aggregate_gate_c_sources(sources, target, scoped)
    assert list(target.iterdir()) == []  # no run directory, temporary tree or manifest survives


def test_gate_c_rejects_unauthorized_roots(contract, gate_c_template):
    sources, target, scoped = _gate_c_case(contract, gate_c_template)
    production_run = scoped.aggregation_namespace / scoped.aggregation_run_id
    production_before = _tree_state(production_run)
    with pytest.raises(rr.RQ4bRunnerError, match="not authorized"):
        rr.aggregate_gate_c_sources(sources, UNAUTHORIZED_GATE_C_ROOT, scoped)
    with pytest.raises(rr.RQ4bRunnerError, match="declared RQ4b artifact namespace"):
        rr.aggregate_gate_c_sources(sources, scoped.aggregation_namespace, scoped)
    assert not UNAUTHORIZED_GATE_C_ROOT.exists()
    assert _tree_state(production_run) == production_before


def test_published_gate_c_sources_match_exact_pins(contract):
    """Read-only: manifests, roster and all 18 receipt digests of the real source runs equal the locked pins."""
    production_run = contract.aggregation_namespace / contract.aggregation_run_id
    production_before = _tree_state(production_run)
    layouts = rr.verify_gate_c_source_pins(contract.artifact_namespace, contract)
    assert list(layouts) == list(contract.aggregation_source_run_ids)
    assert sum(len(pin["receipt_sha256"]) for pin in contract.aggregation_source_pins.values()) == 18
    assert _tree_state(production_run) == production_before


def test_cli_gate_c_stage_is_the_official_evaluation_only_path(contract, gate_c_template, monkeypatch, capsys):
    sources, target, scoped = _gate_c_case(contract, gate_c_template)
    production_run = contract.aggregation_namespace / contract.aggregation_run_id
    production_before = _tree_state(production_run)
    calls = []
    real = rr.aggregate_gate_c_sources

    def recording(source_root, aggregation_root, active):
        calls.append((source_root, aggregation_root, active))
        return real(sources, target, scoped)  # redirected to synthetic roots; production is never touched

    monkeypatch.setattr(rr, "aggregate_gate_c_sources", recording)
    monkeypatch.setattr(rr.ri, "load_contract", lambda: contract)
    _forbid_backend(monkeypatch)
    base = ["--stage", rr.GATE_C_STAGE, "--confirm-production"]
    rejected = (
        (["--run-id", RUN_ID], "pinned run id"),
        (["--run-id", contract.aggregation_run_id, "--lambda", "0", "--seed", "42"], "takes no"),
        (["--run-id", contract.aggregation_run_id, "--new-attempt-after-failure"], "takes no"),
    )
    for argv, message in rejected:
        with pytest.raises(rr.RQ4bRunnerError, match=message):
            rr.main(base + argv)
    assert calls == []
    assert rr.main(base + ["--run-id", contract.aggregation_run_id]) == 0
    assert calls == [(contract.artifact_namespace, contract.aggregation_namespace, contract)]
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == rr.GATE_C_STAGE and report["status"] == "PASS" and report["counts"]["refit_fits"] == 0
    assert _tree_state(production_run) == production_before


# ---------------------------------------------------------------------------
# Cross-run Test evaluation and package validation
# ---------------------------------------------------------------------------


def _test_case(contract):
    base = _new_root()
    sources = _build_gate_c_sources(contract, base / "sources")
    scoped = _pin_gate_c_sources(contract, sources)
    gate_root, test_root = base / "gate", base / "test"
    gate_layout = rr.aggregate_gate_c_sources(sources, gate_root, scoped)
    test = FakeBackend().test
    rng = np.random.default_rng(42)
    denied, approved = np.flatnonzero(test.y == 0), np.flatnonzero(test.y == 1)
    indices = np.asarray([np.concatenate([rng.choice(denied, len(denied)), rng.choice(approved, len(approved))])
                          for _ in range(1000)], dtype=np.int32)
    bootstrap = base / "rq1_test_bootstrap_indices.npz"
    np.savez(bootstrap, indices=indices)
    bootstrap_sha = rr.sha256_file(bootstrap)
    (base / "rq1_test_bootstrap_manifest.json").write_text(json.dumps({
        "indices_sha256": bootstrap_sha, "test_record_id_sha256": rr.sha256_ids(test.record_ids), "test_rows": len(test.record_ids)
    }), encoding="utf-8")
    test_pins = {run_id: {"manifest_sha256": pin["manifest_sha256"],
                          "evaluated_arm_receipt_sha256": pin["receipt_sha256"]}
                 for run_id, pin in scoped.aggregation_source_pins.items()}
    scoped = dataclasses.replace(
        scoped, test_source_pins=test_pins, test_bootstrap_path=bootstrap,
        test_bootstrap_sha256=bootstrap_sha,
        test_gate_c_manifest_sha256=rr.sha256_file(gate_layout.run_dir / rr.MANIFEST_FILE),
        test_gate_c_receipt_sha256=rr.sha256_file(gate_layout.stage_dir(rr.GATE_C_STAGE) / rr.RECEIPT_FILE),
        test_gate_c_selection_sha256=rr.sha256_file(gate_layout.stage_dir(rr.GATE_C_STAGE) / rr.GATE_C_SELECTION_FILE),
    )
    return base, sources, gate_root, test_root, scoped


@pytest.fixture(scope="module")
def cross_run_completed(contract):
    base, sources, gate_root, test_root, scoped = _test_case(contract)
    backend = FakeBackend()
    source_before = _tree(sources)
    layout = rr.evaluate_cross_run_test(sources, gate_root, test_root, backend, scoped)
    return base, sources, gate_root, test_root, scoped, backend, layout, source_before


def _copy_cross_run(case):
    _, sources, gate_root, test_root, scoped, _, _, _ = case
    base = _new_root()
    copied_sources, copied_gate, copied_test = base / "sources", base / "gate", base / "test"
    shutil.copytree(sources, copied_sources)
    shutil.copytree(gate_root, copied_gate)
    shutil.copytree(test_root, copied_test)
    bootstrap = base / scoped.test_bootstrap_path.name
    shutil.copy2(scoped.test_bootstrap_path, bootstrap)
    shutil.copy2(scoped.test_bootstrap_path.parent / "rq1_test_bootstrap_manifest.json", base / "rq1_test_bootstrap_manifest.json")
    return copied_sources, copied_gate, copied_test, dataclasses.replace(scoped, test_bootstrap_path=bootstrap)


def test_cross_run_test_publishes_nine_arms_without_fit_and_preserves_sources(cross_run_completed):
    _, sources, _, _, _, backend, layout, source_before = cross_run_completed
    receipt = rr.read_stage(layout, rr.TEST_STAGE)
    predict_calls = [call for call in backend.calls if call[0] == "predict" and call[2] == "test"]
    assert len(predict_calls) == len(set(predict_calls)) == 9
    assert not any(call[0] == "fit" for call in backend.calls)
    assert receipt["counts"]["cv_fold_fits"] == receipt["counts"]["refit_fits"] == 0
    assert receipt["counts"]["source_fit_accounting"] == {"cv_fold_fits": 45, "refit_fits": 9}
    assert len(receipt["counts"]["predictions"]) == 9 and _tree(sources) == source_before
    stage = layout.stage_dir(rr.TEST_STAGE)
    assert len(list(stage.glob("test_predictions_*.csv"))) == 9
    assert sum(1 for _ in (stage / "bootstrap_replicates.csv").open(encoding="utf-8")) == 6001
    assert not (layout.stage_dir(rr.PACKAGE_STAGE)).exists()


def test_cross_run_outputs_are_neutral_and_complete(cross_run_completed):
    *_, layout, _ = cross_run_completed
    stage = layout.stage_dir(rr.TEST_STAGE)
    manifest = rr.read_json(layout.run_dir / rr.MANIFEST_FILE)
    assert manifest["research_interpretation_owner"] == "user"
    assert not any(name in {"assessment.json", "status.json", "research_interpretation.json"} for name in rr.read_stage(layout, rr.TEST_STAGE)["outputs"])
    assert len(__import__("pandas").read_csv(stage / "arm_metrics.csv")) == 9
    assert len(__import__("pandas").read_csv(stage / "matched_deltas.csv")) == 6
    assert set(__import__("pandas").read_csv(stage / "generalization_gaps.csv")["quantity"]) == {"balanced_accuracy", "eod3", "eod6"}


@pytest.mark.parametrize("pin", ["manifest", "receipt", "selection"])
def test_gate_c_pin_mismatch_blocks_test_and_publishes_nothing(contract, pin):
    _, sources, gate_root, test_root, scoped = _test_case(contract)
    scoped = dataclasses.replace(scoped, **{f"test_gate_c_{pin}_sha256": "0" * 64})
    with pytest.raises(rr.RQ4bPackageError, match="Gate C"):
        rr.evaluate_cross_run_test(sources, gate_root, test_root, FakeBackend(), scoped)
    assert not test_root.exists()


@pytest.mark.parametrize("kind", ["model", "receipt"])
def test_changed_model_or_receipt_blocks_test_before_inference(contract, kind):
    _, sources, gate_root, test_root, scoped = _test_case(contract)
    fit = sources / scoped.track_a_source_run_id / "stages" / "dev_l0_s42"
    target = fit / ("estimator.joblib" if kind == "model" else rr.RECEIPT_FILE)
    target.write_bytes(target.read_bytes() + b"tamper")
    backend = FakeBackend()
    with pytest.raises(rr.RQ4bPackageError, match="corrupt|receipt digest differs"):
        rr.evaluate_cross_run_test(sources, gate_root, test_root, backend, scoped)
    assert not any(call[0] == "predict" for call in backend.calls) and not test_root.exists()


def test_missing_arm_or_seed_blocks_test(contract):
    _, sources, gate_root, test_root, scoped = _test_case(contract)
    shutil.rmtree(sources / scoped.track_b_child_run_id / "stages" / "devval_l0p01_s43")
    with pytest.raises(rr.RQ4bPackageError, match="unpaired|Missing"):
        rr.evaluate_cross_run_test(sources, gate_root, test_root, FakeBackend(), scoped)
    assert not test_root.exists()


@pytest.mark.parametrize("kind", ["order", "population"])
def test_test_id_or_order_mismatch_fails_without_publication(contract, kind):
    _, sources, gate_root, test_root, scoped = _test_case(contract)

    class BadIds(FakeBackend):
        def predict(self, fit_dir, partition):
            out = super().predict(fit_dir, partition)
            if fit_dir.name != "dev_l0p3_s42":
                return out
            order = np.arange(len(out.y))[::-1] if kind == "order" else np.arange(len(out.y))
            ids = tuple(np.asarray(out.record_ids)[order])
            if kind == "population":
                ids = (*ids[:-1], "dataset_b-999999999")
            return ev.PartitionPredictions(ids, out.y[order], out.p_approved[order], out.race[order], out.sex[order])

    with pytest.raises(rr.RQ4bPackageError, match="different ordered Test IDs"):
        rr.evaluate_cross_run_test(sources, gate_root, test_root, BadIds(), scoped)
    assert not test_root.exists()


def test_bootstrap_digest_mismatch_blocks_before_inference(contract):
    _, sources, gate_root, test_root, scoped = _test_case(contract)
    scoped.test_bootstrap_path.write_bytes(scoped.test_bootstrap_path.read_bytes() + b"tamper")
    backend = FakeBackend()
    with pytest.raises(rr.RQ4bPackageError, match="bootstrap digest"):
        rr.evaluate_cross_run_test(sources, gate_root, test_root, backend, scoped)
    assert not any(call[0] == "predict" for call in backend.calls) and not test_root.exists()


def test_inference_failure_leaves_no_test_run(contract):
    _, sources, gate_root, test_root, scoped = _test_case(contract)

    class Failing(FakeBackend):
        def predict(self, fit_dir, partition):
            if fit_dir.name == "dev_l0p03_s42":
                raise RuntimeError("simulated inference failure")
            return super().predict(fit_dir, partition)

    with pytest.raises(RuntimeError, match="simulated inference failure"):
        rr.evaluate_cross_run_test(sources, gate_root, test_root, Failing(), scoped)
    assert not test_root.exists()


def test_package_recomputes_all_outputs_without_inference(cross_run_completed, monkeypatch):
    sources, gate, test, scoped = _copy_cross_run(cross_run_completed)
    monkeypatch.setattr(rr.ProductionBackend, "predict", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference forbidden")))
    receipt = rr.validate_cross_run_test_package(sources, gate, test, scoped)
    assert receipt["status"] == "PASS" and receipt["counts"]["cv_fold_fits"] == receipt["counts"]["refit_fits"] == 0
    report = rr.read_json(rr.RunLayout(test, scoped.test_run_id, rr.TEST_LAYOUT_MODES[rr.MODE_SYNTHETIC]).stage_dir(rr.PACKAGE_STAGE) / "package_report.json")
    assert report["inference_performed"] is False and report["recomputed_from_saved_predictions"] is True


@pytest.mark.parametrize("name", ["arm_metrics.csv", "matched_deltas.csv", "bootstrap_summary.csv", "generalization_gaps.csv"])
def test_forged_output_and_receipt_are_rejected(cross_run_completed, name):
    sources, gate, test, scoped = _copy_cross_run(cross_run_completed)
    layout = rr.RunLayout(test, scoped.test_run_id, rr.TEST_LAYOUT_MODES[rr.MODE_SYNTHETIC])
    stage = layout.stage_dir(rr.TEST_STAGE)
    lines = (stage / name).read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[-1] = "999"
    (stage / name).write_text("\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n", encoding="utf-8")
    receipt = rr.read_json(stage / rr.RECEIPT_FILE)
    receipt["outputs"][name] = rr.sha256_file(stage / name)
    (stage / rr.RECEIPT_FILE).write_text(json.dumps(receipt), encoding="utf-8")
    rr.read_stage(layout, rr.TEST_STAGE)
    with pytest.raises(rr.RQ4bPackageError, match="does not match recomputation"):
        rr.validate_cross_run_test_package(sources, gate, test, scoped)
    assert not layout.stage_dir(rr.PACKAGE_STAGE).exists()


def test_forged_prediction_receipt_is_rejected(cross_run_completed):
    sources, gate, test, scoped = _copy_cross_run(cross_run_completed)
    layout = rr.RunLayout(test, scoped.test_run_id, rr.TEST_LAYOUT_MODES[rr.MODE_SYNTHETIC])
    stage = layout.stage_dir(rr.TEST_STAGE)
    target = sorted(stage.glob("test_predictions_*.csv"))[0]
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8")
    receipt = rr.read_json(stage / rr.RECEIPT_FILE)
    receipt["outputs"][target.name] = rr.sha256_file(target)
    (stage / rr.RECEIPT_FILE).write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(rr.RQ4bPackageError, match="different ordered IDs|bootstrap manifest"):
        rr.validate_cross_run_test_package(sources, gate, test, scoped)
    assert not layout.stage_dir(rr.PACKAGE_STAGE).exists()
