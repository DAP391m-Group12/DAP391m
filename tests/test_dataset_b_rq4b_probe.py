"""Tests for the RQ4b capability probe.

Fits here use small purpose-built fixtures, never the locked 6,000-row
synthetic gate (which runs once as a separate retained attempt). The real
frozen-Validation pipeline is exercised only with synthetic files under a
temporary project root; no project data or frozen artifact is read.
"""

from __future__ import annotations

import hashlib
import dataclasses
import json
import stat
import uuid
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.mitigation import rq4b_probe as rp
from src.utils.path import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Synthetic fixture and split
# ---------------------------------------------------------------------------


def test_synthetic_fixture_exact_cells_and_deterministic_generation():
    data = rp.make_synthetic_fixture()
    assert len(data.record_ids) == 6000 and data.source == rp.SOURCE_SYNTHETIC
    assert rp.cell_counts(data) == {f"group{g}_y{y}": 1000 for g in range(3) for y in (0, 1)}
    assert {int(g): int((data.group == g).sum()) for g in range(3)} == {0: 2000, 1: 2000, 2: 2000}
    expected = np.asarray([-2.0, 0.0, 2.0])[data.group] + np.random.default_rng(42).normal(0.0, 0.25, size=6000)
    np.testing.assert_array_equal(data.margin, expected)
    again = rp.make_synthetic_fixture()
    assert again.record_ids == data.record_ids and np.array_equal(again.margin, data.margin)
    assert data.record_ids[0] == "rq4b_synthetic_000001" and len(set(data.record_ids)) == 6000


def test_synthetic_split_exact_per_cell_disjoint_and_deterministic():
    data = rp.make_synthetic_fixture()
    split = rp.split_probe(data)
    expected = {"train": 600, "validation": 200, "holdout": 200}
    for name, size in expected.items():
        assert rp.cell_counts(data.subset(split[name])) == {f"group{g}_y{y}": size for g in range(3) for y in (0, 1)}
    ids = {name: set(np.asarray(data.record_ids, dtype=object)[split[name]]) for name in split}
    assert not ids["train"] & ids["validation"] and not ids["train"] & ids["holdout"] and not ids["validation"] & ids["holdout"]
    assert set().union(*ids.values()) == set(data.record_ids)
    again = rp.split_probe(rp.make_synthetic_fixture())
    assert all(np.array_equal(split[name], again[name]) for name in split)


def test_uneven_strata_split_is_stratified_and_complete():
    rng = np.random.default_rng(3)
    n = 1037
    data = rp.ProbeData(tuple(f"r{i}" for i in range(n)), rng.normal(size=n), rng.integers(0, 2, n), rng.choice(3, n, p=[0.7, 0.2, 0.1]), rp.SOURCE_REAL)
    split = rp.split_probe(data)
    sizes = {name: len(split[name]) for name in split}
    assert sum(sizes.values()) == n and abs(sizes["train"] - 0.6 * n) <= 1 and abs(sizes["validation"] - sizes["holdout"]) <= 1
    for name in split:
        part = data.subset(split[name])
        for stratum in np.unique(data.strata):
            assert abs((part.strata == stratum).mean() - (data.strata == stratum).mean()) < 0.02
    with pytest.raises(rp.RQ4bProbeError):
        rp._check_split({"train": split["train"], "validation": split["validation"], "holdout": split["holdout"][1:]}, n)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"margin": np.array([0.0, np.nan])}, rp.RQ4bProbeNonFiniteError),
        ({"margin": np.array([0.0, np.inf])}, rp.RQ4bProbeNonFiniteError),
        ({"y": np.array([0, 2])}, rp.RQ4bProbeError),
        ({"group": np.array([0, 3])}, rp.RQ4bProbeError),
        ({"record_ids": ("a", "a")}, rp.RQ4bProbeError),
        ({"source": "other"}, rp.RQ4bProbeError),
    ],
)
def test_invalid_probe_inputs_are_rejected(kwargs, error):
    base = {"record_ids": ("a", "b"), "margin": np.array([0.0, 1.0]), "y": np.array([0, 1]), "group": np.array([0, 1]), "source": rp.SOURCE_REAL}
    with pytest.raises(error):
        rp.ProbeData(**{**base, **kwargs})


# ---------------------------------------------------------------------------
# Comparator and leakage
# ---------------------------------------------------------------------------


def test_y_only_comparator_laplace_counts_from_train_only():
    train = rp.ProbeData(tuple("abcdefg"), np.zeros(7), np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 2, 1, 1, 1, 2]), rp.SOURCE_REAL)
    comparator = rp.fit_y_only(train)
    assert comparator.counts.tolist() == [[2, 0, 1], [0, 3, 1]] and comparator.alpha == 1.0
    np.testing.assert_array_equal(comparator.probabilities, np.array([[3 / 6, 1 / 6, 2 / 6], [1 / 7, 4 / 7, 2 / 7]]))
    np.testing.assert_array_equal(comparator.row_ce(np.array([0, 1]), np.array([1, 2])), -np.log([1 / 6, 2 / 7]))
    data = rp.make_synthetic_fixture()
    split = rp.split_probe(data)
    fitted = rp.fit_y_only(data.subset(split["train"]))
    assert fitted.counts.tolist() == [[600, 600, 600], [600, 600, 600]]  # probe-train rows only
    shuffled = rp.ProbeData(data.record_ids, data.margin, data.y, data.group.copy(), data.source)
    rest = np.concatenate([split["validation"], split["holdout"]])
    shuffled.group[rest] = np.random.default_rng(0).permutation(shuffled.group[rest])
    assert np.array_equal(rp.fit_y_only(shuffled.subset(split["train"])).counts, fitted.counts)


def test_probe_inputs_are_only_margin_and_y():
    margin, y = np.array([-1.5, 0.25, 2.0]), np.array([0, 1, 1])
    features = rp.probe_features(margin, y)
    assert features.shape == (3, 2) and features.dtype == np.float32
    np.testing.assert_array_equal(features, np.stack([margin, y], axis=1).astype(np.float32))
    model = rp.build_probe(42).cpu()
    assert model[0].in_features == 2 and model[2].out_features == 3 and isinstance(model[1], torch.nn.ReLU) and len(model) == 3


# ---------------------------------------------------------------------------
# Optimizer, architecture and stopping
# ---------------------------------------------------------------------------


def test_optimizer_and_architecture_use_locked_settings():
    model = rp.build_probe(42)
    optimizer = rp.make_optimizer(model)
    group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert (group["lr"], group["betas"], group["eps"], group["weight_decay"]) == (0.001, (0.9, 0.999), 1e-08, 0.01)
    assert all(group[key] is False for key in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"))
    assert rp.FITTING["batch_size"] == 256 and rp.FITTING["patience"] == 15 and rp.FITTING["maximum_epochs"] == 100
    assert str(next(model.parameters()).device) == "cuda:0"


def test_probe_initialization_is_seeded_and_rng_isolated():
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state(0)
    first, second = rp.build_probe(42), rp.build_probe(42)
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state(0))
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    assert not torch.equal(rp.build_probe(43)[0].weight, first[0].weight)


def test_stopping_strict_improvement_earliest_tie_and_patience():
    tracker = rp.StoppingTracker()
    assert tracker.update(1, 1.0) and tracker.update(2, 0.5)
    assert not tracker.update(3, 0.5)  # exact tie keeps the earliest checkpoint
    assert tracker.best_epoch == 2 and tracker.epochs_without_improvement == 1
    for epoch in range(4, 17):
        assert not tracker.update(epoch, 0.6)
    assert not tracker.should_stop and tracker.epochs_without_improvement == 14
    assert not tracker.update(17, 0.5) and tracker.should_stop
    assert tracker.stop_reason == "patience_15_consecutive_non_improving_epochs" and tracker.best_epoch == 2
    with pytest.raises(rp.RQ4bProbeError):
        tracker.update(18, 0.1)


def test_stopping_resets_on_improvement_and_caps_at_100():
    tracker = rp.StoppingTracker()
    value = 10.0
    for epoch in range(1, 101):
        if epoch % 10 == 0:
            value -= 1.0
        tracker.update(epoch, value)
    # improvements every 10 epochs (incl. epoch 100) keep patience from firing; the cap stops at 100
    assert tracker.stop_reason == "maximum_epochs_100" and tracker.best_epoch == 100 and tracker.last_epoch == 100
    with pytest.raises(rp.RQ4bProbeNonFiniteError):
        rp.StoppingTracker().update(1, float("nan"))
    with pytest.raises(rp.RQ4bProbeError):
        rp.StoppingTracker().update(2, 1.0)


def _small(seed: int, n: int, *, informative: bool) -> rp.ProbeData:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, n)
    y = rng.integers(0, 2, n)
    margin = (group - 1.0) * 1.5 + rng.normal(0, 0.5, n) if informative else rng.normal(size=n)
    return rp.ProbeData(tuple(f"s{seed}_{i}" for i in range(n)), margin, y, group, rp.SOURCE_SYNTHETIC)


def test_probe_fit_restores_best_checkpoint_and_records_history():
    train, validation = _small(11, 700, informative=False), _small(12, 300, informative=False)
    fit = rp.fit_probe(train, validation)
    values = [entry["validation_ce"] for entry in fit.history]
    best = int(np.argmin(values))  # numpy argmin returns the earliest minimum
    assert fit.best_epoch == best + 1 and fit.best_validation_ce == values[best]
    assert fit.stopped_epoch == len(fit.history) <= 100
    if fit.stop_reason.startswith("patience"):
        assert fit.stopped_epoch - fit.best_epoch == 15
    assert float(rp.row_ce(fit.model, validation).mean()) == fit.best_validation_ce  # restored before holdout
    assert all(entry["batches"] == 3 for entry in fit.history)  # 700 rows / 256, final partial batch kept
    assert fit.rng_isolation == {"cpu_rng_state_unchanged": True, "cuda_rng_state_unchanged": True}
    again = rp.fit_probe(train, validation)
    stable = ("epoch", "mean_train_batch_loss", "batches", "validation_ce", "improved", "learning_rate", "status")
    assert again.best_epoch == fit.best_epoch
    assert [[row[key] for key in stable] for row in again.history] == [[row[key] for key in stable] for row in fit.history]


def test_probe_sees_only_margin_and_y_during_fit(monkeypatch):
    train, validation = _small(21, 300, informative=True), _small(22, 120, informative=True)
    seen = []
    original = rp.build_probe

    def hooked(seed):
        model = original(seed)
        model[0].register_forward_hook(lambda module, inputs, output: seen.append(inputs[0].detach().cpu().numpy()))
        return model

    monkeypatch.setattr(rp, "build_probe", hooked)
    rp.fit_probe(train, validation)
    allowed = {tuple(row) for row in rp.probe_features(np.concatenate([train.margin, validation.margin]), np.concatenate([train.y, validation.y]))}
    assert all(batch.shape[1] == 2 for batch in seen)
    assert all(tuple(row) in allowed for batch in seen[:5] for row in batch)


@pytest.mark.parametrize("kind", ["nonfinite_loss", "nonfinite_gradient"])
def test_nonfinite_probe_training_stops_the_attempt(monkeypatch, kind):
    steps = {"count": 0}
    original_step = torch.optim.AdamW.step

    def counting(self, *args, **kwargs):
        steps["count"] += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting)
    original = rp._batch_loss
    if kind == "nonfinite_loss":
        monkeypatch.setattr(rp, "_batch_loss", lambda logits, target: original(logits, target) * float("nan"))
        message = "loss is not finite"
    else:
        monkeypatch.setattr(rp, "_batch_loss", lambda logits, target: original(logits, target) + 0.0 * torch.sqrt(0.0 * logits.sum()))
        message = "gradients are not finite"
    with pytest.raises(rp.RQ4bProbeNonFiniteError, match=message):
        rp.fit_probe(_small(31, 300, informative=True), _small(32, 100, informative=True))
    assert steps["count"] == 0


def test_cuda_unavailable_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(rp.RQ4bProbeError, match="cuda:0"):
        rp.fit_probe(_small(41, 50, informative=True), _small(42, 20, informative=True))


# ---------------------------------------------------------------------------
# CE gain, bootstrap and gate boundaries
# ---------------------------------------------------------------------------


def test_ce_gain_definition():
    assert rp.ce_gain(np.array([1.0, 1.2]), np.array([0.2, 0.4])) == pytest.approx(0.8)


def test_bootstrap_is_paired_and_stratified():
    rng = np.random.default_rng(5)
    strata = np.repeat(np.arange(6), [40, 35, 30, 25, 20, 50])
    y_only = rng.uniform(0.5, 1.5, len(strata))
    paired = rp.paired_stratified_bootstrap(y_only, y_only - 0.3, strata)
    assert paired["valid_replicates"] == 1000 and np.allclose(paired["gains"], 0.3, rtol=0, atol=1e-12)
    constant_per_stratum = strata.astype(float)
    stratified = rp.paired_stratified_bootstrap(constant_per_stratum, np.zeros(len(strata)), strata)
    assert np.allclose(stratified["y_only_means"], constant_per_stratum.mean(), rtol=0, atol=1e-12)  # stratum sizes preserved
    assert stratified["strata_sizes"] == {0: 40, 1: 35, 2: 30, 3: 25, 4: 20, 5: 50}
    again = rp.paired_stratified_bootstrap(y_only, y_only * 0.5, strata)
    assert again["gains"] == rp.paired_stratified_bootstrap(y_only, y_only * 0.5, strata)["gains"]  # seed 42 deterministic
    assert again["replicates"] == 1000 and len(set(again["gains"])) > 900


def test_bootstrap_requires_all_1000_valid_without_replacement():
    strata = np.repeat(np.arange(2), 30)
    y_only = np.ones(60)
    y_only[3] = np.nan
    summary = rp.paired_stratified_bootstrap(y_only, np.zeros(60), strata)
    assert 0 < len(summary["invalid_replicates"]) < 1000 and summary["replicates"] == 1000
    assert summary["valid_replicates"] == 1000 - len(summary["invalid_replicates"])
    assert summary["ci_evaluable"] is False and summary["ci_lower"] is None and summary["ci_reason"] == "ci_not_evaluable_no_replacement_replicates"
    assert all(item["reason"] == "nonfinite_ce" for item in summary["invalid_replicates"])
    short = rp.summarize_bootstrap(np.ones(999), [])
    assert short["ci_evaluable"] is False


def test_percentile_ci_uses_linear_quantiles_without_rounding():
    gains = np.arange(1000, dtype=np.float64) / 7.0
    summary = rp.summarize_bootstrap(gains, [])
    lower, upper = np.quantile(gains, [0.025, 0.975], method="linear")
    assert (summary["ci_lower"], summary["ci_upper"]) == (lower, upper)
    assert summary["ci_lower"] == pytest.approx(24.975 / 7.0, abs=1e-12) and summary["quantile_method"] == "linear"


@pytest.mark.parametrize(
    "gain, lower, evaluable, status",
    [
        (0.05, 1e-12, True, "PASS"),
        (0.05, 0.0, True, "FAIL"),
        (0.0499999999, 0.2, True, "FAIL"),
        (1.0, -1e-12, True, "FAIL"),
        (1.0, None, False, "FAIL"),
        (np.nextafter(0.05, 1.0), np.nextafter(0.0, 1.0), True, "PASS"),
    ],
)
def test_synthetic_gate_endpoint_boundaries(gain, lower, evaluable, status):
    decision = rp.synthetic_gate_decision(gain, {"ci_lower": lower, "ci_evaluable": evaluable})
    assert decision["status"] == status


# ---------------------------------------------------------------------------
# Real frozen-Validation pipeline (synthetic files only)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_fixture_root(tmp_path):
    config = yaml.safe_load(ri.RQ4B_CONFIG_PATH.read_text(encoding="utf-8"))
    rq3 = yaml.safe_load((PROJECT_ROOT / "config/dataset_b/rq3_protocol.yaml").read_text(encoding="utf-8"))
    contract = ri.validate_contract(config, rq3, project_root=tmp_path)
    n = 43_995
    rng = np.random.default_rng(9)
    ids = [f"dataset_b-{index:09d}" for index in range(1, n + 1)]
    p_approved = rng.uniform(0.01, 0.99, n)
    p_approved[:5] = 1.0  # stored boundary probabilities with a positive denied probability
    p_denied = 1.0 - p_approved
    p_denied[:5] = 1e-9
    y = rng.integers(0, 2, n)
    race = np.asarray((*ri.G6_GROUPS, *ri.COVERAGE_ONLY_GROUPS), dtype=object)[rng.integers(0, 9, n)]
    predictions = contract.resources["frozen_probe_validation_predictions"]
    predictions.parent.mkdir(parents=True)
    pd.DataFrame({"record_id": ids, "y_approved": y, "p_denied": p_denied, "p_approved": p_approved}).to_csv(predictions, index=False)
    raw = contract.resources["derived_race_source"]
    raw.parent.mkdir(parents=True)
    pd.DataFrame({"action_taken": np.where(y == 1, "1", "3"), "derived_race": race}).to_csv(raw, index=False)
    return contract, ids, y, race, p_approved, p_denied


def test_real_probe_population_pipeline_with_fixture_files(real_fixture_root):
    contract, ids, y, race, p_approved, p_denied = real_fixture_root
    data = rp.prepare_real_probe_population(ri.StageAccess("capability_probe", contract))
    keep = np.isin(race, ri.G3_GROUPS)
    assert data.source == rp.SOURCE_REAL and data.record_ids == tuple(np.asarray(ids, dtype=object)[keep])
    np.testing.assert_array_equal(data.margin, (np.log(p_approved) - np.log(p_denied))[keep])
    assert np.array_equal(data.y, y[keep]) and set(np.unique(data.group)) == {0, 1, 2}
    boundary = keep[:5]  # p_approved == 1 with positive p_denied stays finite, without 1 - p
    assert np.isfinite(data.margin).all()
    assert np.all(data.margin[: int(boundary.sum())] == -np.log(1e-9))
    with pytest.raises(ri.RQ4bAccessError):
        rp.prepare_real_probe_population(ri.StageAccess("cv_epoch_selection", contract))


def test_real_probe_fit_requires_separate_authorization(monkeypatch):
    calls = []
    monkeypatch.setattr(rp, "fit_probe", lambda *a, **k: calls.append(1))
    real = rp.ProbeData(("a", "b"), np.zeros(2), np.array([0, 1]), np.array([0, 1]), rp.SOURCE_REAL)
    for flag in (False, "yes", 1):
        with pytest.raises(rp.RQ4bProbeAuthorizationError):
            rp.run_probe_experiment(real, real_probe_authorized=flag)
    assert calls == []


def test_attempt_root_must_be_new_direct_child():
    with pytest.raises(rp.RQ4bProbeError):
        rp.create_attempt_root("../escape")


# ---------------------------------------------------------------------------
# Attempt boundary and failure receipts (T3-R1)
# ---------------------------------------------------------------------------


@pytest.fixture
def attempt_dir():
    """New single-use attempt directory, always prefixed T3_R2_attempt_."""
    return rp.create_attempt_root(f"T3_R2_attempt_{uuid.uuid4().hex}")


def _small_real(seed: int = 51, n: int = 600) -> rp.ProbeData:
    data = _small(seed, n, informative=True)
    return rp.ProbeData(data.record_ids, data.margin, data.y, data.group, rp.SOURCE_REAL)


def _count_fits(monkeypatch) -> list:
    calls = []
    original = rp.fit_probe

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rp, "fit_probe", counting)
    return calls


def _receipt(path):
    receipt = path / rp.RECEIPT_FILE
    assert not (receipt.stat().st_mode & stat.S_IWRITE)  # immutable
    return json.loads(receipt.read_text(encoding="utf-8"))


def _inject(monkeypatch, kind: str) -> str:
    original = rp._batch_loss
    if kind == "loss":
        monkeypatch.setattr(rp, "_batch_loss", lambda logits, target: original(logits, target) * float("nan"))
        return "loss is not finite"
    monkeypatch.setattr(rp, "_batch_loss", lambda logits, target: original(logits, target) + 0.0 * torch.sqrt(0.0 * logits.sum()))
    return "gradients are not finite"


@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_real_attempt_nonfinite_failure_retains_receipt_without_retry(monkeypatch, attempt_dir, kind):
    message = _inject(monkeypatch, kind)
    fits = _count_fits(monkeypatch)
    data = _small_real()
    with pytest.raises(rp.RQ4bProbeNonFiniteError, match=message) as info:
        rp.run_real_probe_attempt(data, attempt_dir, real_probe_authorized=True)
    assert fits == [1]  # no automatic retry
    assert sorted(p.name for p in attempt_dir.iterdir()) == [rp.RECEIPT_FILE, rp.INVOCATION_FILE, "run.log", "training_history.jsonl"]
    receipt = _receipt(attempt_dir)
    assert receipt["attempt_kind"] == "real_probe" and receipt["source"] == rp.SOURCE_REAL and receipt["rows"] == 600
    assert receipt["category"] == "nonfinite_loss_or_gradient" and receipt["exception_type"] == "RQ4bProbeNonFiniteError"
    assert receipt["exception_message"] == str(info.value) and "attempt stopped" in receipt["exception_message"]
    assert receipt["contract_action"] == "stop_attempt_and_write_failure_receipt"
    assert receipt["automatic_retry"] is False and receipt["bad_batch_skipped"] is False and receipt["success_evidence_written"] is False
    assert receipt["record_id_sha256"] == hashlib.sha256("\n".join(data.record_ids).encode()).hexdigest()
    assert receipt["exception_chain"] and receipt["last_position"]["phase"] == "probe_fit"
    assert receipt["last_progress_utc"] is None  # first batch failed before a completed epoch
    live = [json.loads(line) for line in (attempt_dir / "training_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert live[-1]["event"] == "heartbeat" and live[-1]["optimizer_progress_claimed"] is False


def test_real_attempt_reraises_the_original_exception_object(monkeypatch, attempt_dir):
    sentinel = rp.RQ4bProbeNonFiniteError("sentinel failure")

    def failing(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(rp, "fit_probe", failing)
    with pytest.raises(rp.RQ4bProbeNonFiniteError) as info:
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    assert info.value is sentinel and info.value.__cause__ is None
    assert _receipt(attempt_dir)["exception_message"] == "sentinel failure"


def test_failed_attempt_cannot_be_reused_or_overwritten(monkeypatch, attempt_dir):
    _inject(monkeypatch, "loss")
    with pytest.raises(rp.RQ4bProbeNonFiniteError):
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    before = (attempt_dir / rp.RECEIPT_FILE).read_bytes()
    fits = _count_fits(monkeypatch)
    with pytest.raises(rp.RQ4bProbeAttemptError, match="not empty"):
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    assert fits == [] and (attempt_dir / rp.RECEIPT_FILE).read_bytes() == before
    with pytest.raises(FileExistsError):
        rp.create_attempt_root(attempt_dir.name)
    with pytest.raises(FileExistsError):
        rp.write_json_once(attempt_dir / rp.RECEIPT_FILE, {"overwrite": True})
    assert (attempt_dir / rp.RECEIPT_FILE).read_bytes() == before


def test_attempt_boundary_rejects_nonempty_missing_or_outside_directories(monkeypatch, attempt_dir, tmp_path):
    fits = _count_fits(monkeypatch)
    (attempt_dir / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(rp.RQ4bProbeAttemptError, match="not empty"):
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    with pytest.raises(rp.RQ4bProbeAttemptError):
        rp.open_attempt(rp.ATTEMPT_ROOT / f"T3_R2_attempt_missing_{uuid.uuid4().hex}", rp.ATTEMPT_REAL)
    with pytest.raises(rp.RQ4bProbeAttemptError):
        rp.open_attempt(tmp_path, rp.ATTEMPT_REAL)
    with pytest.raises(rp.RQ4bProbeAttemptError):
        rp.open_attempt(attempt_dir, "other")
    assert fits == []


@pytest.mark.parametrize("flag", [False, "yes", 1, None])
def test_unauthorized_real_attempt_is_rejected_before_any_write_or_fit(monkeypatch, attempt_dir, flag):
    fits = _count_fits(monkeypatch)
    with pytest.raises(rp.RQ4bProbeAuthorizationError):
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=flag)
    assert fits == [] and list(attempt_dir.iterdir()) == []


def test_real_fit_requires_an_open_real_attempt_context(monkeypatch, attempt_dir):
    fits = _count_fits(monkeypatch)
    data = _small_real()
    with pytest.raises(rp.RQ4bProbeAttemptError):
        rp.run_probe_experiment(data, real_probe_authorized=True)
    with pytest.raises(rp.RQ4bProbeAttemptError):
        rp.run_probe_experiment(data, real_probe_authorized=True, attempt=rp.ProbeAttempt(attempt_dir, rp.ATTEMPT_SYNTHETIC))
    with pytest.raises(rp.RQ4bProbeError, match="real frozen-Validation probe data"):
        rp.run_real_probe_attempt(_small(52, 600, informative=True), attempt_dir, real_probe_authorized=True)
    # a directly constructed (or properly opened) attempt that is not executing cannot record failure: rejected
    for attempt in (rp.ProbeAttempt(attempt_dir, rp.ATTEMPT_REAL), rp.open_attempt(attempt_dir, rp.ATTEMPT_REAL)):
        with pytest.raises(rp.RQ4bProbeAttemptError, match="executing real-probe attempt"):
            rp.run_probe_experiment(data, real_probe_authorized=True, attempt=attempt)
    assert fits == [] and list(attempt_dir.iterdir()) == [] and rp._ACTIVE_ATTEMPTS == {}


def test_authorized_real_attempt_success_writes_evidence_only(attempt_dir):
    evidence = rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    assert sorted(p.name for p in attempt_dir.iterdir()) == [rp.INVOCATION_FILE, "real_probe_evidence.json", "run.log", "training_history.json", "training_history.jsonl"]
    stored = json.loads((attempt_dir / "real_probe_evidence.json").read_text(encoding="utf-8"))
    assert stored["source"] == rp.SOURCE_REAL and stored["holdout"]["ce_gain_nats"] == evidence["holdout"]["ce_gain_nats"]
    assert "gate" not in stored  # real probe is descriptive only


@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_synthetic_gate_failure_branch_writes_receipt(monkeypatch, attempt_dir, kind):
    small = _small(61, 600, informative=True)  # replaces the locked fixture; the 6,000-row gate is not rerun
    monkeypatch.setattr(rp, "make_synthetic_fixture", lambda: small)
    message = _inject(monkeypatch, kind)
    fits = _count_fits(monkeypatch)
    with pytest.raises(rp.RQ4bProbeNonFiniteError, match=message):
        rp.run_synthetic_gate(attempt_dir)
    assert fits == [1]
    assert sorted(p.name for p in attempt_dir.iterdir()) == [rp.RECEIPT_FILE, rp.INVOCATION_FILE, "run.log", "training_history.jsonl"]
    receipt = _receipt(attempt_dir)
    assert receipt["attempt_kind"] == "synthetic_gate" and receipt["category"] == "nonfinite_loss_or_gradient" and receipt["rows"] == 600


def test_synthetic_gate_fixture_or_finalize_failures_write_receipt_not_evidence(monkeypatch):
    first = rp.create_attempt_root(f"T3_R2_attempt_{uuid.uuid4().hex}")

    def broken_fixture():
        raise rp.RQ4bProbeError("fixture generation failed")

    monkeypatch.setattr(rp, "make_synthetic_fixture", broken_fixture)
    with pytest.raises(rp.RQ4bProbeError, match="fixture generation failed"):
        rp.run_synthetic_gate(first)
    receipt = _receipt(first)
    assert receipt["source"] is None and receipt["category"] == "attempt_failure" and [p.name for p in first.iterdir()] == [rp.RECEIPT_FILE]

    second = rp.create_attempt_root(f"T3_R2_attempt_{uuid.uuid4().hex}")
    small = _small(62, 600, informative=True)
    monkeypatch.setattr(rp, "make_synthetic_fixture", lambda: small)

    def broken_decision(*args, **kwargs):
        raise RuntimeError("decision failed")

    monkeypatch.setattr(rp, "synthetic_gate_decision", broken_decision)
    with pytest.raises(RuntimeError, match="decision failed"):
        rp.run_synthetic_gate(second)
    assert sorted(p.name for p in second.iterdir()) == [rp.RECEIPT_FILE, rp.INVOCATION_FILE, "run.log", "training_history.jsonl"]
    assert _receipt(second)["exception_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Atomic publication and exception preservation (T3-R2)
# ---------------------------------------------------------------------------


def _published(root) -> list[str]:
    return sorted(p.name for p in root.iterdir() if not p.name.endswith(rp.PARTIAL_SUFFIX))


def _partials(root) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.name.endswith(rp.PARTIAL_SUFFIX))


def _fail_writes_for(monkeypatch, target_name: str, *, after_bytes: int | None = None) -> list:
    """Make _write_payload fail for one target (optionally after writing part of it); record attempts."""
    attempts = []
    original = rp._write_payload

    def failing(target, handle, payload):
        if target.name == target_name:
            attempts.append(target.name)
            if after_bytes is not None:
                handle.write(payload[:after_bytes])
                handle.flush()
            raise OSError(f"simulated write failure for {target_name}")
        return original(target, handle, payload)

    monkeypatch.setattr(rp, "_write_payload", failing)
    return attempts


def test_write_json_once_publishes_atomically_and_never_overwrites(attempt_dir):
    target = attempt_dir / "example.json"
    rp.write_json_once(target, {"value": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1} and not (target.stat().st_mode & stat.S_IWRITE)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        rp.write_json_once(target, {"value": 2})
    assert target.read_bytes() == before and _partials(attempt_dir) == []


def test_partial_evidence_write_is_not_success_and_does_not_suppress_receipt(monkeypatch, attempt_dir):
    attempts = _fail_writes_for(monkeypatch, "real_probe_evidence.json", after_bytes=64)
    fits = _count_fits(monkeypatch)
    with pytest.raises(OSError, match="simulated write failure for real_probe_evidence.json") as info:
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    assert fits == [1] and attempts == ["real_probe_evidence.json"]  # no retry
    assert _published(attempt_dir) == [rp.RECEIPT_FILE, rp.INVOCATION_FILE, "run.log", "training_history.json", "training_history.jsonl"]
    partial, = _partials(attempt_dir)
    assert partial.startswith(".real_probe_evidence.json.") and (attempt_dir / partial).stat().st_size == 64
    receipt = _receipt(attempt_dir)
    assert receipt["exception_type"] == "OSError" and receipt["exception_message"] == str(info.value)
    assert receipt["partial_files"] == [partial] and receipt["success_evidence_written"] is False
    assert not hasattr(info.value, "rq4b_receipt_error")


def test_receipt_write_failure_preserves_original_nonfinite_exception(monkeypatch, attempt_dir):
    _inject(monkeypatch, "loss")
    attempts = _fail_writes_for(monkeypatch, rp.RECEIPT_FILE)
    fits = _count_fits(monkeypatch)
    with pytest.raises(rp.RQ4bProbeNonFiniteError, match="loss is not finite") as info:
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    secondary = info.value.rq4b_receipt_error
    assert isinstance(secondary, OSError) and "simulated write failure" in str(secondary)
    assert any("failure receipt could not be written" in note for note in info.value.__notes__)
    assert attempts == [rp.RECEIPT_FILE] and fits == [1]  # receipt write and fit each attempted once
    assert _published(attempt_dir) == [rp.INVOCATION_FILE, "run.log", "training_history.jsonl"]
    assert all(name.startswith(f".{rp.RECEIPT_FILE}.") for name in _partials(attempt_dir))


def test_receipt_write_failure_preserves_original_object_identity(monkeypatch, attempt_dir):
    sentinel = rp.RQ4bProbeNonFiniteError("sentinel nonfinite")

    def failing(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(rp, "fit_probe", failing)
    _fail_writes_for(monkeypatch, rp.RECEIPT_FILE)
    with pytest.raises(rp.RQ4bProbeNonFiniteError) as info:
        rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    assert info.value is sentinel and isinstance(sentinel.rq4b_receipt_error, OSError)


def test_receipt_write_failure_after_fixture_failure_preserves_original(monkeypatch, attempt_dir):
    sentinel = rp.RQ4bProbeError("fixture generation failed")

    def broken_fixture():
        raise sentinel

    monkeypatch.setattr(rp, "make_synthetic_fixture", broken_fixture)
    attempts = _fail_writes_for(monkeypatch, rp.RECEIPT_FILE, after_bytes=10)
    fits = _count_fits(monkeypatch)
    with pytest.raises(rp.RQ4bProbeError) as info:
        rp.run_synthetic_gate(attempt_dir)
    assert info.value is sentinel and isinstance(sentinel.rq4b_receipt_error, OSError)
    assert any("failure receipt could not be written" in note for note in sentinel.__notes__)
    assert attempts == [rp.RECEIPT_FILE] and fits == []
    assert _published(attempt_dir) == [] and len(_partials(attempt_dir)) == 1  # nothing published, no evidence
    with pytest.raises(rp.RQ4bProbeAttemptError, match="not empty"):
        rp.run_synthetic_gate(attempt_dir)  # a failed attempt is never reused


def test_completed_evidence_is_published_only_after_full_write(attempt_dir):
    rp.run_real_probe_attempt(_small_real(), attempt_dir, real_probe_authorized=True)
    assert _published(attempt_dir) == [rp.INVOCATION_FILE, "real_probe_evidence.json", "run.log", "training_history.json", "training_history.jsonl"] and _partials(attempt_dir) == []
    stored = json.loads((attempt_dir / "real_probe_evidence.json").read_text(encoding="utf-8"))
    assert stored["source"] == rp.SOURCE_REAL and not ((attempt_dir / "real_probe_evidence.json").stat().st_mode & stat.S_IWRITE)


@pytest.fixture
def frozen_route(tmp_path, monkeypatch):
    """Real-route orchestration with synthetic resources and a mocked fitter only."""
    root = tmp_path / "a"
    root.mkdir()
    monkeypatch.setattr(rp, "ATTEMPT_ROOT", root)
    contract = ri.load_contract()
    resources = dict(contract.resources)
    for key in ("frozen_probe_validation_predictions", "derived_race_source"):
        resources[key] = tmp_path / key
        resources[key].write_text("synthetic resource", encoding="utf-8")
    contract = dataclasses.replace(contract, resources=resources)
    data = _small_real()
    monkeypatch.setattr(ri, "load_contract", lambda: contract)
    monkeypatch.setattr(rp, "prepare_real_probe_population", lambda access: data)
    tracker, history = rp.StoppingTracker(), []
    for epoch in range(1, 17):
        improved = tracker.update(epoch, 2.0)
        history.append({"epoch": epoch, "validation_ce": 2.0, "mean_train_batch_loss": 2.0,
                        "batches": 2, "improved": improved, "learning_rate": 0.001,
                        "epoch_seconds": 0.01, "elapsed_seconds": epoch * 0.01, "gpu_memory_bytes": 0})
    fake = SimpleNamespace(model=object(), history=history, best_epoch=1, best_validation_ce=2.0,
                           stopped_epoch=16, stop_reason=tracker.stop_reason,
                           rng_isolation={"cpu_rng_state_unchanged": True, "cuda_rng_state_unchanged": True})
    calls = []
    def mocked_fit(*args):
        calls.append(1)
        if rp._CURRENT_PROGRESS is not None:
            for row in history:
                rp._CURRENT_PROGRESS.emit("epoch_completed", phase="probe_fit", epoch=row["epoch"], batch=row["batches"],
                    train_loss=row["mean_train_batch_loss"], validation_ce=row["validation_ce"],
                    learning_rate=row["learning_rate"], epoch_seconds=row["epoch_seconds"],
                    elapsed_seconds=row["elapsed_seconds"], gpu_memory_bytes=row["gpu_memory_bytes"], optimizer_progress_claimed=True)
        return fake
    monkeypatch.setattr(rp, "fit_probe", mocked_fit)
    monkeypatch.setattr(rp, "row_ce", lambda model, part: np.full(len(part.record_ids), 2.0))
    monkeypatch.setattr(rp, "_environment", lambda: {"mock": True})
    return SimpleNamespace(root=root, contract=contract, data=data, resources=resources, calls=calls)


def _real_evidence(route):
    root = rp.create_attempt_root("real")
    rp.run_real_probe_from_frozen(root, real_probe_authorized=True)
    return root / "real_probe_evidence.json"


def _rewrite_evidence(path, edit):
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.chmod(stat.S_IREAD | stat.S_IWRITE)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_real_cli_and_ingestion_preserve_inconclusive_without_refitting(frozen_route, monkeypatch, capsys):
    assert rp.main(["--kind", "real", "--attempt-name", "real", "--confirm-real-probe"]) == 0
    path = frozen_route.root / "real" / "real_probe_evidence.json"
    monkeypatch.setattr(rp, "fit_probe", lambda *args: pytest.fail("ingestion must not fit"))
    result = rp.load_real_probe_evidence(path, frozen_route.contract)
    assert result["status"] == "completed" and result["capability_status"] == "capability_inconclusive"
    assert result["evidence"]["holdout"]["ce_gain_nats"] < 0
    assert frozen_route.calls == [1] and '"status": "completed"' in capsys.readouterr().out


@pytest.mark.parametrize("args", [[], ["--kind", "real", "--attempt-name", "x"],
                                  ["--attempt-name", "x", "--confirm-real-probe"]])
def test_probe_cli_requires_acknowledgement_before_any_write(monkeypatch, args):
    monkeypatch.setattr(rp, "create_attempt_root", lambda *a: pytest.fail("must not write"))
    with pytest.raises(SystemExit):
        rp.main(args)


def test_probe_cli_synthetic_default_is_preserved(frozen_route, monkeypatch):
    monkeypatch.setattr(rp, "run_synthetic_gate", lambda root: {"gate": {"status": "PASS"}, "holdout": {},
        "bootstrap": {"ci_lower": 0.1, "ci_upper": 0.2, "valid_replicates": 1000},
        "probe_fit": {"best_epoch": 1, "stop_reason": "mock"}})
    monkeypatch.setattr(rp, "run_real_probe_from_frozen", lambda *a, **k: pytest.fail("must not run real"))
    assert rp.main(["--attempt-name", "synthetic"]) == 0


def test_real_preparation_failure_is_receipted_and_not_retried(frozen_route, monkeypatch):
    root = rp.create_attempt_root("broken")
    sentinel = RuntimeError("bad frozen source")
    def fail(access):
        raise sentinel
    monkeypatch.setattr(rp, "prepare_real_probe_population", fail)
    with pytest.raises(RuntimeError) as info:
        rp.run_real_probe_from_frozen(root, real_probe_authorized=True)
    assert info.value is sentinel and frozen_route.calls == []
    assert json.loads((root / rp.RECEIPT_FILE).read_text())["rows"] is None
    with pytest.raises(rp.RQ4bProbeAttemptError):
        rp.run_real_probe_from_frozen(root, real_probe_authorized=True)


def test_real_route_authorization_precedes_reads_and_writes(frozen_route, monkeypatch):
    monkeypatch.setattr(rp, "open_attempt", lambda *a: pytest.fail("must not open"))
    with pytest.raises(rp.RQ4bProbeAuthorizationError):
        rp.run_real_probe_from_frozen(frozen_route.root / "none")


@pytest.mark.parametrize("edit", [
    lambda e: e.update(source=rp.SOURCE_SYNTHETIC),
    lambda e: e.pop("lineage"),
    lambda e: e["lineage"].update(frozen_run_id="foreign"),
    lambda e: e["lineage"]["code_sha256"].update({"rq4b_probe.py": "0" * 64}),
    lambda e: e["lineage"].update(population_sha256="0" * 64),
    lambda e: e.update(invocation_sha256="0" * 64),
    lambda e: e.update(rows=1),
    lambda e: e["split_record_id_sha256"].update(holdout="0" * 64),
    lambda e: e["settings"]["fitting"].update(patience=1),
    lambda e: e["probe_fit"].update(best_epoch=2),
    lambda e: e["holdout"].update(ce_gain_nats=9),
    lambda e: e["bootstrap"].update(ci_lower=-99),
    lambda e: e["bootstrap"].update(valid_replicates=999),
    lambda e: e["bootstrap"]["gains"].pop(),
    lambda e: e["bootstrap"].update(strata_sizes={}),
])
def test_real_evidence_rejects_corrupted_lineage_and_reporting(frozen_route, edit):
    path = _real_evidence(frozen_route)
    _rewrite_evidence(path, edit)
    with pytest.raises(rp.RQ4bProbeError):
        rp.load_real_probe_evidence(path, frozen_route.contract)


@pytest.mark.parametrize("key", ["frozen_probe_validation_predictions", "derived_race_source"])
def test_real_evidence_rejects_changed_source_files(frozen_route, key):
    path = _real_evidence(frozen_route)
    frozen_route.resources[key].write_text("changed source", encoding="utf-8")
    with pytest.raises(rp.RQ4bProbeError, match="lineage"):
        rp.load_real_probe_evidence(path, frozen_route.contract)


@pytest.mark.parametrize("kind", ["missing", "failure", "partial", "invocation", "malformed"])
def test_real_evidence_missing_failed_partial_or_invalid_never_falls_back(frozen_route, kind):
    path = _real_evidence(frozen_route)
    if kind == "missing":
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
        path.unlink()
    elif kind == "failure":
        (path.parent / rp.RECEIPT_FILE).write_text("{}")
    elif kind == "partial":
        (path.parent / ".evidence.partial").write_text("partial")
    elif kind == "invocation":
        invocation = path.parent / rp.INVOCATION_FILE
        invocation.chmod(stat.S_IREAD | stat.S_IWRITE)
        invocation.write_text('{"source": "foreign"}')
    else:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
        path.write_text("invalid json")
    with pytest.raises((rp.RQ4bProbeError, FileNotFoundError, json.JSONDecodeError)):
        rp.load_real_probe_evidence(path, frozen_route.contract)


def test_real_sources_changed_during_fit_fail_without_success_evidence(frozen_route, monkeypatch):
    original = rp.fit_probe
    def changed(*args):
        frozen_route.resources["derived_race_source"].write_text("changed")
        return original(*args)
    monkeypatch.setattr(rp, "fit_probe", changed)
    root = rp.create_attempt_root("changed")
    with pytest.raises(rp.RQ4bProbeError, match="changed during fitting"):
        rp.run_real_probe_from_frozen(root, real_probe_authorized=True)
    assert (root / rp.RECEIPT_FILE).is_file() and not (root / "real_probe_evidence.json").exists()


def test_actual_ingestion_and_runner_publication_keep_complete_evidence(frozen_route, monkeypatch):
    from src.dataset_b.mitigation import rq4b_runner as rr
    path = _real_evidence(frozen_route)
    config = yaml.safe_load(ri.RQ4B_CONFIG_PATH.read_text(encoding="utf-8"))
    record = config["execution_governance"]["t3_capability_probe_acceptance"]["synthetic_gate"]
    record["evidence_path"] = "gate.json"
    project = frozen_route.root.parent
    monkeypatch.setattr(rr, "SYNTHETIC_ROOT", project)
    (project / "gate.json").write_text('{"synthetic": true}')
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(rr, "PROJECT_ROOT", project)
    monkeypatch.setattr(rp, "fit_probe", lambda *a: pytest.fail("must not fit while ingesting"))
    backend = rr.ProductionBackend(frozen_route.contract, config_path=config_path, real_probe_evidence=path)
    adapter = SimpleNamespace(kind="synthetic_ingestion_adapter", preflight=lambda: {"mock": "PASS"},
                              capability_evidence=backend.capability_evidence)
    layout = rr.create_run(project / "run", "rq4b3_adv_20261019_01", rr.MODE_SYNTHETIC,
                           frozen_route.contract, config_path=config_path)
    runner = rr.RQ4bRunner(layout, adapter, frozen_route.contract)
    runner.preflight()
    assert runner.capability()["counts"]["probe_fits"] == 0
    saved = rr.read_json(layout.stage_dir("capability") / "capability.json")
    assert saved["real_probe"]["status"] == "completed"
    assert saved["real_probe"]["evidence"] == json.loads(path.read_text())
    assert saved["real_probe"]["evidence_sha256"] == rp.file_sha256(path)
    # The stage retains the actual validated payload; later external changes do not erase it.
    path.chmod(stat.S_IREAD | stat.S_IWRITE)
    path.write_text("changed after ingestion")
    rr.read_stage(layout, "capability")
    assert rr.read_json(layout.stage_dir("capability") / "capability.json") == saved


def test_real_evidence_rejects_foreign_path(frozen_route):
    with pytest.raises(rp.RQ4bProbeError, match="direct probe attempt"):
        rp.load_real_probe_evidence(frozen_route.root / "real_probe_evidence.json", frozen_route.contract)


def test_real_consistent_unevaluable_ci_remains_descriptive(frozen_route):
    path = _real_evidence(frozen_route)
    def invalid_replicate(evidence):
        boot = evidence["bootstrap"]
        boot["y_only_means"][0] = float("nan")
        boot["gains"][0] = float("nan")
        boot.update(rp.summarize_bootstrap(boot["gains"], [{"replicate": 0, "reason": "nonfinite_ce"}]))
    _rewrite_evidence(path, invalid_replicate)
    result = rp.load_real_probe_evidence(path, frozen_route.contract)
    assert result["status"] == "completed" and result["capability_status"] == "capability_inconclusive"
    assert result["evidence"]["bootstrap"]["ci_evaluable"] is False


def test_supplied_synthetic_or_legacy_attempt_without_lineage_is_rejected(frozen_route):
    root = rp.create_attempt_root("legacy")
    rp.run_real_probe_attempt(frozen_route.data, root, real_probe_authorized=True)
    with pytest.raises(rp.RQ4bProbeError):
        rp.load_real_probe_evidence(root / "real_probe_evidence.json", frozen_route.contract)


def test_observation_does_not_replace_completed_progress(frozen_route):
    p = rp.ProbeProgress(frozen_route.root, rp.ATTEMPT_REAL)
    p.emit("epoch_completed", phase="probe_fit", epoch=1, batch=2, optimizer_progress_claimed=True)
    done = p.events[-1]["last_progress_utc"]
    p.heartbeat(2, 1)
    assert p.events[-1]["last_progress_utc"] == done
    assert p.events[-1]["timestamp_utc"] != done and p.events[-1]["optimizer_progress_claimed"] is False


def test_initial_logging_failure_retains_receipt(frozen_route, monkeypatch):
    root = rp.create_attempt_root("bad_log")
    sentinel = OSError("initial log failure")
    def broken(*args, **kwargs):
        raise sentinel
    monkeypatch.setattr(rp.ProbeProgress, "emit", broken)
    with pytest.raises(OSError) as exc:
        rp.run_real_probe_from_frozen(root, real_probe_authorized=True)
    assert exc.value is sentinel
    assert json.loads((root / rp.RECEIPT_FILE).read_text())["exception_message"] == str(sentinel)
    assert not (root / "real_probe_evidence.json").exists() and frozen_route.calls == []


@pytest.mark.parametrize("phase", ["preparation", "experiment"])
def test_time_observer_reports_while_foreground_is_waiting(frozen_route, monkeypatch, phase):
    root = rp.create_attempt_root("observer")
    monkeypatch.setattr(rp, "HEARTBEAT_SECONDS", 0.01)
    seen = threading.Event()
    original_emit = rp.ProbeProgress.emit
    def observing(self, event, **kwargs):
        original_emit(self, event, **kwargs)
        if event == "observer_heartbeat":
            seen.set()
    monkeypatch.setattr(rp.ProbeProgress, "emit", observing)
    def wait_then_fail(*args, **kwargs):
        assert seen.wait(2), "observer did not report while foreground waited"
        raise RuntimeError("foreground interrupted")
    monkeypatch.setattr(rp, "prepare_real_probe_population" if phase == "preparation" else "run_probe_experiment", wait_then_fail)
    with pytest.raises(RuntimeError, match="foreground interrupted"):
        rp.run_real_probe_from_frozen(root, real_probe_authorized=True)
    rows = [json.loads(line) for line in (root / "training_history.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "attempt_started"
    assert any(row["event"] == "observer_heartbeat" for row in rows)
    assert all(row["last_progress_utc"] is None for row in rows)
    assert (root / rp.RECEIPT_FILE).exists()


def test_failed_publication_never_logs_attempt_completed(frozen_route, monkeypatch):
    _fail_writes_for(monkeypatch, "real_probe_evidence.json")
    root = rp.create_attempt_root("publish_failure")
    with pytest.raises(OSError):
        rp.run_real_probe_from_frozen(root, real_probe_authorized=True)
    rows = [json.loads(line) for line in (root / "training_history.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "computation_completed"
    assert not any(row["event"] == "attempt_completed" for row in rows)
    assert (root / rp.RECEIPT_FILE).exists() and not (root / "real_probe_evidence.json").exists()


@pytest.mark.parametrize("name", ["run.log", "training_history.jsonl", "training_history.json"])
@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_required_logging_payload_is_verified(frozen_route, name, kind):
    path = _real_evidence(frozen_route)
    target = path.parent / name
    target.chmod(stat.S_IREAD | stat.S_IWRITE)
    if kind == "missing":
        target.unlink()
    else:
        target.write_text("corrupt")
    with pytest.raises((rp.RQ4bProbeError, FileNotFoundError)):
        rp.load_real_probe_evidence(path, frozen_route.contract)


def test_log_metrics_checked_even_if_hashes_are_rewritten(frozen_route):
    path = _real_evidence(frozen_route)
    root = path.parent
    rows = [json.loads(line) for line in (root / "training_history.jsonl").read_text().splitlines()]
    next(row for row in rows if row["event"] == "epoch_completed")["train_loss"] = 99
    lines = "".join(json.dumps(row) + "\n" for row in rows)
    for name in ("run.log", "training_history.jsonl"):
        (root / name).write_text(lines)
    summary = root / "training_history.json"
    summary.chmod(stat.S_IREAD | stat.S_IWRITE)
    summary.write_text(json.dumps({"attempt_id": root.name, "events": rows}))
    _rewrite_evidence(path, lambda e: e["logging"].update(files={name: rp.file_sha256(root / name)
        for name in ("run.log", "training_history.jsonl", "training_history.json")}))
    with pytest.raises(rp.RQ4bProbeError, match="metrics differ"):
        rp.load_real_probe_evidence(path, frozen_route.contract)


def test_observer_write_failure_propagates_without_fit(frozen_route, monkeypatch):
    p = rp.ProbeProgress(frozen_route.root, rp.ATTEMPT_REAL)
    monkeypatch.setattr(rp, "HEARTBEAT_SECONDS", 0.01)
    seen = threading.Event()
    error = OSError("background log failed")
    original = rp.ProbeProgress.emit
    def broken(self, event, **kwargs):
        if event == "observer_heartbeat":
            seen.set()
            raise error
        return original(self, event, **kwargs)
    monkeypatch.setattr(rp.ProbeProgress, "emit", broken)
    p.emit("attempt_started", phase="probe_prepare", optimizer_progress_claimed=False)
    p.start()
    assert seen.wait(2)
    with pytest.raises(OSError) as exc:
        p.stop()
    assert exc.value is error and not p._thread.is_alive() and frozen_route.calls == []
