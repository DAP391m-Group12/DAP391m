"""Tests for the RQ4b production adversarial training, CV and refit engine.

GPU tests are synthetic smoke fits of at most two executed epochs. Complete
5 x 256 bookkeeping is exercised only with test doubles whose curves are
labelled ``test_double`` and are never observed production curves. No project
Train/Validation/Test data or frozen artifact is read.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.mitigation import rq4b_training as rt
from src.dataset_b.preprocessing import split_inputs
from src.dataset_b.preprocessing.categorical_encoder import CATEGORICAL_SCHEMA, NUMERIC_VALUE_COLUMNS, TWO_CHANNEL_COLUMNS


SEED = 42
LAMBDA = 0.1
TEST_DOUBLE = "test_double"
RTOL, ATOL = rt.EQUIVALENCE_RTOL, rt.EQUIVALENCE_ATOL
ALL_GROUPS = (*ri.G6_GROUPS, *ri.COVERAGE_ONLY_GROUPS)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _ids(n: int, start: int = 1) -> list[str]:
    return [f"dataset_b-{index:09d}" for index in range(start, start + n)]


def _native(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(n, len(NUMERIC_VALUE_COLUMNS))).astype(np.float32)
    data = {}
    for column in TWO_CHANNEL_COLUMNS:
        if column in NUMERIC_VALUE_COLUMNS:
            data[column] = numeric[:, NUMERIC_VALUE_COLUMNS.index(column)]
        else:
            categories = list(CATEGORICAL_SCHEMA[column])
            data[column] = pd.Categorical.from_codes(rng.integers(0, len(categories), size=n), categories=categories)
    X = pd.DataFrame(data, columns=list(TWO_CHANNEL_COLUMNS))
    y = (numeric[:, 0] - 0.7 * numeric[:, 1] + 0.3 * rng.normal(size=n) > -0.3).astype(np.int64)
    race = np.asarray(ALL_GROUPS, dtype=object)[rng.integers(0, len(ALL_GROUPS), size=n)]
    race[:3] = ri.G3_GROUPS
    race[3] = "Joint"
    return X, y, race


@pytest.fixture(scope="session")
def contract() -> ri.RQ4bContract:
    return ri.load_contract()


@pytest.fixture(scope="session")
def train_input(contract) -> ri.RQ4bPredictorInput:
    X, y, race = _native(1100, 1)
    return ri.build_predictor_input(X, y, _ids(1100), race, contract)


@pytest.fixture(scope="session")
def fold_validation() -> rt.FoldValidation:
    X, y, _ = _native(300, 2)
    return rt.FoldValidation(X, y, tuple(_ids(300, 5001)))


@pytest.fixture(scope="session")
def alignment_input(contract) -> ri.RQ4bPredictorInput:
    """Features encode row number (col 0) and group code (col 1) so same-label rows are distinguishable."""
    X, y, race = _native(1100, 3)
    probe = ri.build_predictor_input(X, y, _ids(1100), race, contract)
    X = X.copy()
    X[NUMERIC_VALUE_COLUMNS[0]] = np.arange(1100, dtype=np.float32)
    X[NUMERIC_VALUE_COLUMNS[1]] = probe.group_codes.astype(np.float32)
    return ri.build_predictor_input(X, y, _ids(1100), race, contract)


RAW_CHOICES = {
    "loan_type": ["1", "2"], "loan_purpose": ["1", "31"], "lien_status": ["1", "2"], "reverse_mortgage": ["2", "1111"],
    "open-end_line_of_credit": ["2"], "business_or_commercial_purpose": ["2"], "debt_to_income_ratio": ["<20%", "20%-<30%", "40"],
    "loan_term": ["360", "240"], "occupancy_type": ["1", "2"], "preapproval": ["1", "2"], "interest_only_payment": ["2"],
    "balloon_payment": ["2"],
}


def _raw_population(n: int, seed: int) -> rt.TrainPopulation:
    """Synthetic raw 15-field rows passed through the real input-quality gate."""
    rng = np.random.default_rng(seed)
    raw = {}
    for column in ri.RAW_PREDICTOR_COLUMNS:
        if column in RAW_CHOICES:
            raw[column] = rng.choice(RAW_CHOICES[column], size=n)
        elif column == "income":
            raw[column] = rng.integers(20, 400, size=n).astype(str)
        elif column == "loan_amount":
            raw[column] = (rng.integers(5, 90, size=n) * 10000).astype(str)
        else:  # loan_to_value_ratio
            raw[column] = rng.integers(30, 99, size=n).astype(str)
    X = pd.DataFrame(raw, columns=list(ri.RAW_PREDICTOR_COLUMNS))
    action = rng.choice(["1", "3"], size=n)
    ids = _ids(n, 10_001)
    audit = pd.DataFrame({column: ["1"] * n for column in ("lei", "county_code", *(f"applicant_race-{i}" for i in range(1, 6)), "applicant_sex", "applicant_age")})
    partition = split_inputs.DatasetBPartition(
        name="train", X=X, y=pd.Series((action != "3").astype(np.int8)), record_ids=pd.Series(ids), audit=audit, action_taken=pd.Series(action)
    )
    gated = split_inputs.DatasetBInputQualityGate().apply(partition)
    race = np.asarray(ALL_GROUPS, dtype=object)[rng.integers(0, len(ALL_GROUPS), size=n)]
    return rt.TrainPopulation(gated.X.reset_index(drop=True), gated.y.to_numpy(dtype=np.int64), tuple(gated.record_ids), race)


@pytest.fixture(scope="session")
def population() -> rt.TrainPopulation:
    return _raw_population(1100, 7)


def _context(contract, **overrides) -> rt.ProductionContext:
    prerequisites = {"technical_spike_codex_acceptance": "PASS", "input_boundary_codex_acceptance": "PASS", "pytabkit_source_lineage": "PASS"}
    prerequisites.update(overrides)
    return rt.ProductionContext(contract=contract, prerequisites=prerequisites, cuda={"device": "cuda:0"}, lineage={})


def _tensors(estimator) -> dict[str, np.ndarray]:
    layer = estimator.backend_.alg_interface_.model.model
    out = {f"param:{n}": p.detach().cpu().numpy().copy() for n, p in layer.named_parameters()}
    out.update({f"buffer:{n}": b.detach().cpu().numpy().copy() for n, b in layer.named_buffers()})
    return out


def _assert_close(reference: dict, candidate: dict) -> None:
    assert list(reference) == list(candidate)
    for key in reference:
        np.testing.assert_allclose(candidate[key], reference[key], rtol=RTOL, atol=ATOL, err_msg=key)


# ---------------------------------------------------------------------------
# Prerequisites, CUDA and budgets (no GPU fit)
# ---------------------------------------------------------------------------


def test_installed_pytabkit_matches_pinned_lineage():
    lineage = rt.pytabkit_lineage()
    assert lineage["status"] == "PASS" and lineage["package_version"] == "1.7.3"
    assert all(item["match"] for item in lineage["files"].values()) and len(lineage["files"]) == 12


def test_lineage_mismatch_fails(monkeypatch):
    pinned = dict(rt.PYTABKIT_PINNED_BLOBS)
    pinned["pytabkit/models/data/data.py"] = "0" * 40
    monkeypatch.setattr(rt, "PYTABKIT_PINNED_BLOBS", pinned)
    assert rt.pytabkit_lineage()["status"] == "FAIL"


def test_recorded_prerequisites_are_pass():
    config = yaml.safe_load(ri.RQ4B_CONFIG_PATH.read_text(encoding="utf-8"))
    assert rt.require_prerequisites(rt.prerequisite_statuses(config)) == {
        "technical_spike_codex_acceptance": "PASS", "input_boundary_codex_acceptance": "PASS", "pytabkit_source_lineage": "PASS"}


@pytest.mark.parametrize("value", ["FAIL", "NOT_RUN", "UNRESOLVED", "pass", "PASS ", None, True])
@pytest.mark.parametrize("name", ["technical_spike_codex_acceptance", "input_boundary_codex_acceptance", "pytabkit_source_lineage"])
def test_non_pass_mandatory_prerequisite_is_rejected(name, value):
    statuses = {"technical_spike_codex_acceptance": "PASS", "input_boundary_codex_acceptance": "PASS", "pytabkit_source_lineage": "PASS", name: value}
    with pytest.raises(rt.RQ4bPrerequisiteError):
        rt.require_prerequisites(statuses)


def test_production_context_rejects_non_pass_yaml_record(tmp_path):
    config = yaml.safe_load(ri.RQ4B_CONFIG_PATH.read_text(encoding="utf-8"))
    config["execution_governance"]["t1_input_boundary_acceptance"]["result"] = "FAIL"
    path = tmp_path / "rq4b_adversarial.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(rt.RQ4bPrerequisiteError):
        rt.open_production_context(path)


def test_production_context_opens_with_recorded_pass():
    context = rt.open_production_context()
    assert dict(context.prerequisites) == {name: "PASS" for name in context.prerequisites}
    assert context.cuda["device"] == "cuda:0"


def test_cuda_unavailable_is_rejected_without_cpu_fallback(monkeypatch, train_input):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(rt.RQ4bCudaUnavailableError):
        rt.require_cuda0()
    with pytest.raises(rt.RQ4bCudaUnavailableError):
        rt.fit_smoke(train_input, adversary_lambda=LAMBDA, model_seed=SEED)


@pytest.mark.parametrize("device", ["cuda", "cpu", "cuda:1"])
def test_non_cuda0_device_is_rejected(device):
    estimator = rt.AdversarialRealMLPClassifier(**{**rt.estimator_parameters(SEED, 2), "device": device})
    with pytest.raises(rt.RQ4bCudaUnavailableError):
        estimator._validate_requested_device()


def test_fit_spec_and_smoke_budget_limits(train_input):
    with pytest.raises(rt.RQ4bBudgetError):
        rt.fit_smoke(train_input, adversary_lambda=LAMBDA, model_seed=SEED, execution_epochs=3)
    with pytest.raises(rt.RQ4bBudgetError):
        rt.FitSpec(adversary_lambda=0.0, model_seed=SEED, execution_epochs=257)
    with pytest.raises(rt.RQ4bBudgetError):
        rt.FitSpec(adversary_lambda=0.0, model_seed=SEED, execution_epochs=0)
    for bad in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(rt.RQ4bTrainingError):
            rt.FitSpec(adversary_lambda=bad, model_seed=SEED, execution_epochs=1)
    with pytest.raises(rt.RQ4bTrainingError):
        rt.FitSpec(adversary_lambda=0.0, model_seed=SEED, execution_epochs=1, module_kind="other")


def test_production_fold_fit_always_uses_full_budget_and_horizon(monkeypatch, contract, train_input, fold_validation):
    captured = {}

    def fake_engine(inputs, spec, **kwargs):
        captured["spec"], captured["kwargs"] = spec, kwargs
        history = {"fold_validation_ce": [0.5] * 256, "completed_epochs": 256, "budget": {"module_schedule_horizon_epochs": 256}}
        return rt.FitResult(None, spec, history, None, {}, kwargs["evidence_kind"])

    monkeypatch.setattr(rt, "_fit_engine", fake_engine)
    result = rt.fit_cv_fold(_context(contract), train_input, fold_validation, adversary_lambda=0.03, model_seed=45, fold_index=2)
    spec = captured["spec"]
    assert (spec.execution_epochs, spec.schedule_horizon_epochs, spec.adversary_lambda, spec.model_seed) == (256, 256, 0.03, 45)
    assert captured["kwargs"]["fold_validation"] is fold_validation and result.evidence_kind == rt.EVIDENCE_OBSERVED


@pytest.mark.parametrize(
    "overrides, lam, seed",
    [({"pytabkit_source_lineage": "FAIL"}, 0.1, 42), ({}, 0.2, 42), ({}, 0.1, 47)],
)
def test_production_fold_fit_rejects_non_pass_or_unlocked_settings(contract, train_input, fold_validation, overrides, lam, seed):
    with pytest.raises((rt.RQ4bPrerequisiteError, rt.RQ4bTrainingError)):
        rt.fit_cv_fold(_context(contract, **overrides), train_input, fold_validation, adversary_lambda=lam, model_seed=seed, fold_index=1)
    with pytest.raises(rt.RQ4bPrerequisiteError):
        rt.fit_cv_fold(object(), train_input, fold_validation, adversary_lambda=0.1, model_seed=42, fold_index=1)


# ---------------------------------------------------------------------------
# Epoch selection
# ---------------------------------------------------------------------------


def test_epoch_selector_earliest_tie_equal_weighting_and_cap():
    tie = np.ones((5, 256))
    tie[:, [9, 19]] = 0.5
    assert rt.select_epoch(tie)["selected_epoch"] == 10 and rt.select_epoch(tie)["tied_epochs"] == [10, 20]
    unequal = np.ones((5, 256))
    unequal[0, 3] = 0.0  # equal fold weighting moves the mean by 0.2
    unequal[1:, 5] = 0.76
    assert rt.select_epoch(unequal)["selected_epoch"] == 4
    cap = rt.select_epoch(np.tile(np.linspace(1.0, 0.5, 256), (5, 1)))
    assert cap["selected_epoch"] == 256 and cap["epoch_cap_hit"] is True
    assert rt.select_epoch(tie)["epoch_cap_hit"] is False and rt.select_epoch(tie)["finite_values"] == 1280


def test_epoch_selector_uses_unrounded_values():
    curves = np.ones((5, 256))
    curves[:, 7] = 0.5 + 1e-15
    curves[:, 30] = 0.5
    assert rt.select_epoch(curves)["selected_epoch"] == 31


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c[:, :255],
        lambda c: c[:4],
        lambda c: np.vstack([c, c[:1]]),
        lambda c: np.where(np.eye(5, 256, dtype=bool), np.nan, c),
        lambda c: np.where(np.eye(5, 256, dtype=bool), np.inf, c),
        lambda c: [list(row) for row in c[:, :-1]] + [list(c[0])],
        lambda c: [["x"] * 256] * 5,
    ],
    ids=["short", "four_folds", "six_folds", "nan", "inf", "ragged", "non_numeric"],
)
def test_epoch_selector_rejects_incomplete_or_nonfinite_curves(mutate):
    with pytest.raises(rt.RQ4bEpochSelectionError):
        rt.select_epoch(mutate(np.ones((5, 256))))


def test_cv_folds_are_locked_and_partition_train(population):
    first = rt.make_cv_folds(population.record_ids, population.y)
    second = rt.make_cv_folds(population.record_ids, population.y)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, second))
    assert rt.fold_membership_digest(population.record_ids, first) == rt.fold_membership_digest(population.record_ids, second)
    validation = np.concatenate([val for _, val in first])
    assert np.array_equal(np.sort(validation), np.arange(len(population.y)))
    with pytest.raises(rt.RQ4bEpochSelectionError):
        rt._check_fold_partition(first[:4], len(population.y))
    bad = [(first[0][0], np.concatenate([first[0][1], first[1][1][:1]]))] + first[1:]
    with pytest.raises(rt.RQ4bEpochSelectionError):
        rt._check_fold_partition(bad, len(population.y))


# ---------------------------------------------------------------------------
# 5 x 256 orchestration with test doubles (never observed production curves)
# ---------------------------------------------------------------------------


def _double_curve(fold_index: int, minimum_epoch: int = 37) -> list[float]:
    epochs = np.arange(1, 257)
    return list(1.0 + 0.001 * fold_index + 1e-4 * (epochs - minimum_epoch) ** 2)


class _FoldDouble:
    def __init__(self, *, minimum_epoch: int = 37, executed: int = 256, values: int = 256) -> None:
        self.calls: list[dict] = []
        self.minimum_epoch, self.executed, self.values = minimum_epoch, executed, values

    def __call__(self, context, fold_train, fold_validation, *, adversary_lambda, model_seed, fold_index):
        self.calls.append(
            {"train_ids": set(fold_train.record_ids), "val_ids": set(fold_validation.record_ids), "X_train": fold_train.X, "X_val": fold_validation.X,
             "lambda": adversary_lambda, "seed": model_seed, "group_codes": fold_train.group_codes}
        )
        curve = _double_curve(fold_index, self.minimum_epoch)[: self.values]
        history = {"fold_validation_ce": curve, "epochs": [{"epoch": 1, "fold": fold_index}]}
        return rt.FoldFitResult(fold_index, curve, self.executed, 256, TEST_DOUBLE, history=history, log={"fold": fold_index})


def test_cv_orchestration_bookkeeping_with_fold_local_preprocessing(monkeypatch, contract, population):
    created = []
    original = ri.make_fit_population_preprocessor

    def tracking(access):
        pipeline = original(access)
        created.append(pipeline)
        return pipeline

    monkeypatch.setattr(ri, "make_fit_population_preprocessor", tracking)
    double = _FoldDouble()
    record = rt.run_cv_epoch_selection(
        _context(contract), ri.StageAccess("cv_epoch_selection", contract), population, adversary_lambda=0.1, model_seed=43, fold_fitter=double
    )
    assert record.evidence_kind == TEST_DOUBLE  # never reported as observed production curves
    assert np.asarray(record.curves).shape == (5, 256) and record.selected_epoch == 37
    assert record.selection["finite_values"] == 1280 and record.selection["epoch_cap_hit"] is False
    assert len(record.fold_outputs) == 5
    assert all(output["curve_matches_history"] and output["selected_epoch"] == 37 for output in record.fold_outputs)
    assert [output["log"]["fold"] for output in record.fold_outputs] == [1, 2, 3, 4, 5]
    all_ids = set(population.record_ids)
    val_union = set()
    for call, pipeline, summary in zip(double.calls, created, record.fold_summaries):
        assert not call["train_ids"] & call["val_ids"] and call["train_ids"] | call["val_ids"] == all_ids
        val_union |= call["val_ids"]
        fitted_rows = sum(pipeline.named_steps["native"].fold_train_category_counts_["loan_type"].values())
        assert fitted_rows == len(call["train_ids"]) == summary["preprocessor_fit_rows"]  # fitted on fold-train only
        assert tuple(call["X_train"].columns) == ri.NATIVE_PREDICTOR_COLUMNS and not set(call["X_val"].columns) & ri.FORBIDDEN_PREDICTOR_COLUMNS
        assert (call["lambda"], call["seed"]) == (0.1, 43)
    assert val_union == all_ids and len({id(p) for p in created}) == 5
    again = rt.run_cv_epoch_selection(
        _context(contract), ri.StageAccess("cv_epoch_selection", contract), population, adversary_lambda=1.0, model_seed=44, fold_fitter=_FoldDouble()
    )
    assert again.fold_membership_sha256 == record.fold_membership_sha256  # identical folds across lambdas/seeds


def test_durable_progress_flushes_live_events_and_retains_interrupted_position(tmp_path):
    progress = rt.DurableProgress(
        tmp_path, run_id="rq4b3_adv_20261019_02", attempt_id="attempt-1", unit="dev_l0p1_s42",
        adversary_lambda=0.1, model_seed=42,
    )
    progress.phase_started("cv", 1)
    progress.heartbeat(phase="cv", fold=1, epoch=1, batch=1, gpu_memory_bytes=123)
    lines = [json.loads(line) for line in (tmp_path / "training_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["phase_started", "heartbeat"]
    assert lines[-1]["optimizer_progress_claimed"] is False and lines[-1]["gpu_memory_bytes"] == 123
    ledger = json.loads((tmp_path / "fit_ledger.json").read_text(encoding="utf-8"))
    assert ledger["started"]["cv_fold_fits"] == 1 and ledger["completed"]["cv_fold_fits"] == 0
    assert ledger["last_position"] == {"phase": "cv", "fold": 1, "epoch": 1, "batch": 1, "event": "heartbeat"}
    assert not (tmp_path / "training_history.json").exists()  # interrupted before finalization


def test_training_heartbeat_preserves_last_completed_progress(tmp_path):
    p = rt.DurableProgress(tmp_path, run_id="review", attempt_id="a", unit="u", adversary_lambda=0, model_seed=42)
    p.emit("epoch_completed", phase="cv", fold=1, epoch=1, batch=100, optimizer_progress_claimed=True)
    done = p.ledger["last_progress_utc"]
    p.heartbeat(phase="cv", fold=1, epoch=2, batch=1, gpu_memory_bytes=0)
    assert p.ledger["last_progress_utc"] == done
    assert p.events[-1]["timestamp_utc"] != done


def test_durable_progress_write_failure_is_not_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(rt.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("simulated log fsync failure")))
    with pytest.raises(OSError, match="simulated log fsync failure"):
        rt.DurableProgress(
            tmp_path, run_id="rq4b3_adv_20261019_02", attempt_id="attempt-1", unit="dev_l0p1_s42",
            adversary_lambda=0.1, model_seed=42,
        )


@pytest.mark.parametrize(
    "double, error",
    [(_FoldDouble(executed=2), rt.RQ4bBudgetError), (_FoldDouble(values=255), rt.RQ4bEpochSelectionError)],
    ids=["shortened_execution", "short_curve"],
)
def test_cv_orchestration_rejects_shortened_budgets(contract, population, double, error):
    with pytest.raises(error):
        rt.run_cv_epoch_selection(
            _context(contract), ri.StageAccess("cv_epoch_selection", contract), population, adversary_lambda=0.1, model_seed=42, fold_fitter=double
        )


@pytest.mark.parametrize("stage", ["full_train_refit", "validation_calibration_selection", "capability_probe"])
def test_cv_orchestration_rejects_other_stages(contract, population, stage):
    with pytest.raises(ri.RQ4bAccessError):
        rt.run_cv_epoch_selection(_context(contract), ri.StageAccess(stage, contract), population, adversary_lambda=0.1, model_seed=42, fold_fitter=_FoldDouble())


def _selection(contract, population, minimum_epoch=37, lam=0.1, seed=42) -> rt.EpochSelectionRecord:
    return rt.run_cv_epoch_selection(
        _context(contract), ri.StageAccess("cv_epoch_selection", contract), population,
        adversary_lambda=lam, model_seed=seed, fold_fitter=_FoldDouble(minimum_epoch=minimum_epoch),
    )


class _RefitDouble:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, inputs, spec):
        self.calls.append((inputs, spec))
        history = {"completed_epochs": spec.execution_epochs, "budget": {"module_schedule_horizon_epochs": spec.schedule_horizon_epochs}}
        return rt.FitResult(None, spec, history, None, {}, TEST_DOUBLE)


@pytest.mark.parametrize("minimum_epoch", [37, 256])
def test_refit_is_fresh_executes_selected_epoch_and_keeps_horizon(contract, population, minimum_epoch):
    record = _selection(contract, population, minimum_epoch)
    double = _RefitDouble()
    result = rt.run_full_train_refit(
        _context(contract), ri.StageAccess("full_train_refit", contract), population, record, adversary_lambda=0.1, model_seed=42, fitter=double
    )
    (inputs, spec), = double.calls
    assert (spec.execution_epochs, spec.schedule_horizon_epochs) == (minimum_epoch, 256)
    assert result.selected_epoch == minimum_epoch and result.epoch_cap_hit is (minimum_epoch == 256)
    assert inputs.record_ids == population.record_ids  # complete Train, no stopping holdout
    fitted_rows = sum(result.preprocessor.named_steps["native"].fold_train_category_counts_["loan_type"].values())
    assert fitted_rows == len(population.record_ids)


def test_refit_rejects_inconsistent_or_test_double_selection(contract, population):
    record = _selection(contract, population)
    access = ri.StageAccess("full_train_refit", contract)
    with pytest.raises(rt.RQ4bEpochSelectionError, match="observed"):
        rt.run_full_train_refit(_context(contract), access, population, record, adversary_lambda=0.1, model_seed=42)
    tampered = dataclasses.replace(record, selection={**record.selection, "selected_epoch": 12})
    with pytest.raises(rt.RQ4bEpochSelectionError):
        rt.run_full_train_refit(_context(contract), access, population, tampered, adversary_lambda=0.1, model_seed=42, fitter=_RefitDouble())
    with pytest.raises(rt.RQ4bEpochSelectionError):
        rt.run_full_train_refit(_context(contract), access, population, record, adversary_lambda=0.3, model_seed=42, fitter=_RefitDouble())
    with pytest.raises(ri.RQ4bAccessError):
        rt.run_full_train_refit(
            _context(contract), ri.StageAccess("cv_epoch_selection", contract), population, record, adversary_lambda=0.1, model_seed=42, fitter=_RefitDouble()
        )


def _forbid_refit_work(monkeypatch) -> _RefitDouble:
    """Fail if preprocessing is constructed; the returned double records any engine call."""

    def forbidden(access):
        raise AssertionError("preprocessing must not start before lineage is verified")

    monkeypatch.setattr(ri, "make_fit_population_preprocessor", forbidden)
    return _RefitDouble()


def _with_population(population: rt.TrainPopulation, **changes) -> rt.TrainPopulation:
    return dataclasses.replace(population, **changes)


def _lineage_mismatches(population: rt.TrainPopulation) -> dict[str, rt.TrainPopulation]:
    n = len(population.record_ids)
    order = np.random.default_rng(5).permutation(n)
    return {
        "disjoint_ids": _with_population(population, record_ids=tuple(_ids(n, 900_001))),
        "permuted_rows": _with_population(
            population,
            X_raw=population.X_raw.iloc[order].reset_index(drop=True),
            y=population.y[order],
            record_ids=tuple(np.asarray(population.record_ids, dtype=object)[order]),
            derived_race=population.derived_race[order],
        ),
        "changed_labels": _with_population(population, y=np.where(np.arange(n) < 40, 1 - population.y, population.y)),
        "dropped_row": rt.TrainPopulation(
            population.X_raw.iloc[1:].reset_index(drop=True), population.y[1:], population.record_ids[1:], population.derived_race[1:]
        ),
    }


@pytest.mark.parametrize("case", ["disjoint_ids", "permuted_rows", "changed_labels", "dropped_row"])
def test_refit_rejects_population_lineage_mismatch_before_any_work(monkeypatch, contract, population, case):
    record = _selection(contract, population)
    other = _lineage_mismatches(population)[case]
    assert rt.fold_membership_digest(other.record_ids, rt.make_cv_folds(other.record_ids, other.y)) != record.fold_membership_sha256
    double = _forbid_refit_work(monkeypatch)
    with pytest.raises(rt.RQ4bEpochSelectionError, match="fold membership differs"):
        rt.run_full_train_refit(
            _context(contract), ri.StageAccess("full_train_refit", contract), other, record, adversary_lambda=0.1, model_seed=42, fitter=double
        )
    assert double.calls == []


@pytest.mark.parametrize(
    "digest",
    [None, "", "0" * 63, "0" * 65, "G" * 64, "A" * 64, 12345, b"0" * 64],
    ids=["missing", "empty", "short", "long", "non_hex", "uppercase", "non_string", "bytes"],
)
def test_refit_rejects_missing_or_malformed_lineage_before_any_work(monkeypatch, contract, population, digest):
    record = dataclasses.replace(_selection(contract, population), fold_membership_sha256=digest)
    double = _forbid_refit_work(monkeypatch)
    with pytest.raises(rt.RQ4bEpochSelectionError, match="fold-membership SHA-256"):
        rt.run_full_train_refit(
            _context(contract), ri.StageAccess("full_train_refit", contract), population, record, adversary_lambda=0.1, model_seed=42, fitter=double
        )
    assert double.calls == []


def test_refit_rejects_corrupted_digest_before_any_work(monkeypatch, contract, population):
    record = _selection(contract, population)
    flipped = ("1" if record.fold_membership_sha256[0] != "1" else "2") + record.fold_membership_sha256[1:]
    double = _forbid_refit_work(monkeypatch)
    with pytest.raises(rt.RQ4bEpochSelectionError, match="fold membership differs"):
        rt.run_full_train_refit(
            _context(contract), ri.StageAccess("full_train_refit", contract), population,
            dataclasses.replace(record, fold_membership_sha256=flipped), adversary_lambda=0.1, model_seed=42, fitter=double,
        )
    assert double.calls == []


def test_refit_lineage_matches_same_population_and_existing_checks_still_apply(contract, population):
    record = _selection(contract, population)
    assert rt.require_population_lineage(record, population) == record.fold_membership_sha256
    double = _RefitDouble()
    result = rt.run_full_train_refit(
        _context(contract), ri.StageAccess("full_train_refit", contract), population, record, adversary_lambda=0.1, model_seed=42, fitter=double
    )
    assert len(double.calls) == 1 and result.selected_epoch == 37 and result.schedule_horizon_epochs == 256
    # lambda/seed, curve and evidence-kind checks run before the lineage check and still reject
    with pytest.raises(rt.RQ4bEpochSelectionError, match="another lambda"):
        rt.run_full_train_refit(_context(contract), ri.StageAccess("full_train_refit", contract), population, record, adversary_lambda=0.3, model_seed=42, fitter=double)
    with pytest.raises(rt.RQ4bEpochSelectionError, match="inconsistent"):
        rt.run_full_train_refit(
            _context(contract), ri.StageAccess("full_train_refit", contract), population,
            dataclasses.replace(record, selection={**record.selection, "selected_epoch": 12}), adversary_lambda=0.1, model_seed=42, fitter=double,
        )
    with pytest.raises(rt.RQ4bEpochSelectionError, match="observed"):
        rt.run_full_train_refit(_context(contract), ri.StageAccess("full_train_refit", contract), population, record, adversary_lambda=0.1, model_seed=42)
    assert len(double.calls) == 1


# ---------------------------------------------------------------------------
# GPU smoke fits (at most two executed epochs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cuda0():
    rt.require_cuda0()


@pytest.fixture(scope="session")
def diag_run(cuda0, train_input):
    return rt.fit_smoke(train_input, adversary_lambda=LAMBDA, model_seed=SEED, record_gradient_diagnostics=True)


@pytest.fixture(scope="session")
def zero_run(cuda0, train_input):
    return rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED)


def test_signed_gradients_separation_and_parameter_updates(diag_run):
    diag = diag_run.log["gradient_diagnostics"]
    assert diag["logits_shape"] == [1, 256, 2] and diag["margin_shape"] == [256]
    assert diag["adversary_input_shape"][1] == 2 and diag["adversary_output_shape"][1] == 3
    assert diag["signed_identity_holds"] and diag["additive_variant_max_abs_diff"] > 0
    assert diag["adversary_logit1_grad_nonzero_roster_rows"] > 0
    assert diag["total_minus_pred_opposes_adv_roster_rows"] == diag["adversary_logit1_grad_nonzero_roster_rows"]
    assert diag["adversary_grad_outside_roster_max_abs"] == 0.0 and diag["adversary_margin_grad_outside_roster_max_abs"] == 0.0
    assert diag["prediction_grad_outside_roster_nonzero_rows"] == diag["outside_rows"] > 0  # all rows in predictor loss
    assert diag["predictor_unchanged_by_adversary_step"] and diag["predictor_grads_empty_after_adversary_backward"]
    assert diag["adversary_max_change_by_adversary_step"] > 0 and diag["predictor_max_change_by_predictor_step"] > 0
    assert diag["adversary_unchanged_by_predictor_step"] and diag["adversary_grads_unchanged_by_predictor_backward"]
    assert not diag["adversary_params_in_predictor_optimizer"] and diag["predictor_lr_at_diagnosed_batch"] > 0


def test_seeds_devices_batches_losses_and_finite_state_recorded(diag_run):
    history, log = diag_run.history, diag_run.log
    assert history["spec"] == {"adversary_lambda": LAMBDA, "model_seed": SEED, "adversary_seed": SEED, "execution_epochs": 2,
                               "schedule_horizon_epochs": 256, "module_kind": "adversarial"}
    assert log["model_seed"] == log["adversary_seed"] == SEED
    budget = history["budget"]
    assert (budget["batch_size"], budget["drop_last"], budget["shuffle"], budget["predict_batch_size"]) == (256, True, True, 1024)
    assert budget["batches_per_epoch"] == 4 and budget["train_rows"] == 1100 and budget["iterated_samples_per_epoch"] == 1024
    assert all(value == "cuda:0" for value in log["devices"].values() if isinstance(value, str))
    assert log["devices"]["predictor_parameters"] == log["devices"]["adversary_parameters"] == ["cuda:0"]
    assert log["adversary_updates"] == log["predictor_updates"] == 8 and log["adversary_skipped_empty_mask"] == 0
    for epoch in history["epochs"]:
        assert epoch["predictor_parameters_finite"] and epoch["adversary_parameters_finite"]
        assert epoch["losses"]["batches"] == 4 and np.isfinite(epoch["losses"]["mean_prediction_loss"])
        assert np.isfinite(epoch["losses"]["mean_adversary_loss"]) and epoch["first_batch_schedule"]["sched"]["lr"]
    assert log["alignment"] == {"dataset_rows_verified": 1100, "batches_verified": 8, "batches_group_feature_verified": 0}


def test_lambda_zero_equivalence_to_preserved_recipe(cuda0, train_input, zero_run):
    stock = rt.fit_preserved_recipe_smoke(train_input, model_seed=SEED)
    extension_h2 = rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED, schedule_horizon_epochs=2)
    _assert_close(_tensors(stock), _tensors(extension_h2.estimator))
    np.testing.assert_allclose(extension_h2.estimator.predict_proba(train_input.X), stock.predict_proba(train_input.X), rtol=RTOL, atol=ATOL)
    observed_stock = rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED, module_kind="stock")
    _assert_close(_tensors(observed_stock.estimator), _tensors(zero_run.estimator))
    np.testing.assert_allclose(zero_run.estimator.predict_proba(train_input.X), observed_stock.estimator.predict_proba(train_input.X), rtol=RTOL, atol=ATOL)
    assert observed_stock.history["batch_fingerprints"] == zero_run.history["batch_fingerprints"]
    assert zero_run.log["adversary_updates"] == 0 and zero_run.log["adversary_bypassed_lambda_zero"] == 8
    assert stock.backend_parameters() == {**extension_h2.estimator.backend_parameters()}


def test_rng_isolation_and_matched_adversary_initialization(diag_run, zero_run):
    assert diag_run.history["batch_fingerprints"] == zero_run.history["batch_fingerprints"]  # loader order unchanged by adversary
    assert [b["rows"] for b in diag_run.log["batches"]] == [b["rows"] for b in zero_run.log["batches"]]
    for run in (diag_run, zero_run):
        assert run.log["adversary_rng_isolation"]["cpu_rng_state_unchanged"] and run.log["adversary_rng_isolation"]["cuda_rng_state_unchanged"]
    assert diag_run.log["adversary_rng_isolation"]["initial_state_sha256"] == zero_run.log["adversary_rng_isolation"]["initial_state_sha256"]
    _, same = rt.build_adversary(SEED)
    _, other = rt.build_adversary(SEED + 1)
    assert same["initial_state_sha256"] == diag_run.log["adversary_rng_isolation"]["initial_state_sha256"] != other["initial_state_sha256"]


def test_empty_mask_skips_only_adversary_update(cuda0, train_input, zero_run):
    empty = dataclasses.replace(train_input, group_codes=np.full(len(train_input.y), ri.OUTSIDE_ROSTER_CODE, dtype=np.int64))
    run = rt.fit_smoke(empty, adversary_lambda=LAMBDA, model_seed=SEED)
    assert run.log["adversary_updates"] == 0 and run.log["adversary_skipped_empty_mask"] == run.log["predictor_updates"] == 8
    assert all(epoch["losses"]["mean_adversary_loss"] is None for epoch in run.history["epochs"])
    _assert_close(_tensors(zero_run.estimator), _tensors(run.estimator))


def test_row_feature_group_alignment_and_same_label_corruption(cuda0, alignment_input):
    run = rt.fit_smoke(alignment_input, adversary_lambda=LAMBDA, model_seed=SEED, alignment_group_feature_column=1)
    assert run.log["alignment"]["batches_group_feature_verified"] == run.log["alignment"]["batches_verified"] == 8
    for batch in run.log["batches"]:
        rows = np.asarray(batch["rows"])
        assert np.array_equal(np.asarray(batch["groups"]), alignment_input.group_codes[rows])
    y, codes = alignment_input.y, alignment_input.group_codes
    a = int(np.flatnonzero(y == 1)[0])
    b = int(np.flatnonzero((y == 1) & (codes != codes[a]))[0])
    perm = np.arange(len(y))
    perm[[a, b]] = perm[[b, a]]
    assert np.array_equal(y[perm], y)  # label-preserving
    metadata = rt.RowMetadata.from_input(alignment_input)
    group_swap = dataclasses.replace(metadata, group_codes=codes[perm])
    row_swap = dataclasses.replace(metadata, group_codes=codes[perm], numeric_reference=metadata.numeric_reference[perm])
    for corrupted in (group_swap, row_swap):
        with pytest.raises(rt.RQ4bAlignmentError):
            rt.fit_smoke(alignment_input, adversary_lambda=LAMBDA, model_seed=SEED, metadata=corrupted, alignment_group_feature_column=1)
    with pytest.raises(rt.RQ4bAlignmentError):
        rt.fit_smoke(alignment_input, adversary_lambda=LAMBDA, model_seed=SEED, metadata=dataclasses.replace(metadata, labels=1 - metadata.labels))


def _count_steps(monkeypatch) -> dict:
    from pytabkit.models.optim.optimizers import OptimizerBase

    counts = {"predictor_steps": 0}
    original = OptimizerBase.step

    def counting(self, *args, **kwargs):
        counts["predictor_steps"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OptimizerBase, "step", counting)
    return counts


def _nan_gradient_loss(original):
    def poisoned(self, output):
        loss = original(self, output)
        return loss + 0.0 * torch.sqrt(0.0 * output["x_cont"].sum())  # finite value, NaN gradient
    return poisoned


@pytest.mark.parametrize(
    "method, factory, message",
    [
        ("_prediction_loss", lambda original: (lambda self, output: original(self, output) * float("nan")), "not finite"),
        ("_adversary_loss", lambda original: (lambda self, *args: original(self, *args) * float("inf")), "not finite"),
        ("_prediction_loss", _nan_gradient_loss, "gradients are not finite"),
    ],
    ids=["nonfinite_prediction_loss", "nonfinite_adversary_loss", "nonfinite_gradient"],
)
def test_nonfinite_losses_and_gradients_fail_before_step(monkeypatch, cuda0, train_input, method, factory, message):
    counts = _count_steps(monkeypatch)
    monkeypatch.setattr(rt.AdversarialTabNNModule, method, factory(getattr(rt.AdversarialTabNNModule, method)))
    with pytest.raises(rt.RQ4bNonFiniteError, match=message):
        rt.fit_smoke(train_input, adversary_lambda=LAMBDA, model_seed=SEED)
    assert counts["predictor_steps"] == 0


def test_nonfinite_lambda_zero_criterion_fails_before_step(monkeypatch, cuda0, train_input):
    counts = _count_steps(monkeypatch)
    original = rt.AdversarialTabNNModule.compile_model

    def compile_and_poison(self, *args, **kwargs):
        original(self, *args, **kwargs)
        criterion = self.criterion
        self.criterion = lambda logits, y: criterion(logits, y) * float("nan")

    monkeypatch.setattr(rt.AdversarialTabNNModule, "compile_model", compile_and_poison)
    with pytest.raises(rt.RQ4bNonFiniteError):
        rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED)
    assert counts["predictor_steps"] == 0


def test_fold_validation_ce_recorded_without_fitting_state(cuda0, train_input, fold_validation, zero_run):
    run = rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED, fold_validation=fold_validation)
    ce = run.history["fold_validation_ce"]
    assert len(ce) == 2 and all(np.isfinite(ce))
    assert all(epoch["fold_validation_eval_left_parameters_unchanged"] for epoch in run.history["epochs"])
    p = run.estimator.predict_proba(fold_validation.X)
    assert abs(ce[-1] - rt.stable_binary_ce_from_margin(np.log(p[:, 1]) - np.log(p[:, 0]), fold_validation.y)) < 1e-6
    _assert_close(_tensors(zero_run.estimator), _tensors(run.estimator))  # recording is training-neutral
    assert run.history["batch_fingerprints"] == zero_run.history["batch_fingerprints"]
    assert run.history["budget"]["train_rows"] == len(train_input.y)  # fold-validation rows never enter the backend dataset
    flipped = rt.FoldValidation(fold_validation.X, 1 - fold_validation.y, fold_validation.record_ids)
    flipped_run = rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED, fold_validation=flipped)
    _assert_close(_tensors(run.estimator), _tensors(flipped_run.estimator))
    assert flipped_run.history["fold_validation_ce"] != ce
    overlapping = rt.FoldValidation(train_input.X.iloc[:10].reset_index(drop=True), train_input.y[:10], tuple(train_input.record_ids[:10]))
    with pytest.raises(rt.RQ4bAlignmentError):
        rt.fit_smoke(train_input, adversary_lambda=0.0, model_seed=SEED, fold_validation=overlapping)


def test_schedule_horizon_preserved_separately_from_execution_budget(cuda0, train_input, diag_run):
    budget = diag_run.history["budget"]
    assert (budget["trainer_max_epochs"], budget["module_schedule_horizon_epochs"]) == (2, 256)
    n_batches = budget["batches_per_epoch"]
    for entry in diag_run.history["batch_schedules"]:
        assert abs(entry["fit_progress"] * 256 - (entry["epoch"] - 1 + entry["batch_idx"] / n_batches)) < 1e-9
        assert set(entry["sched"]) == {"lr", "wd", "p_drop"}
    one_epoch = rt.fit_smoke(train_input, adversary_lambda=LAMBDA, model_seed=SEED, execution_epochs=1)
    assert one_epoch.history["batch_schedules"] == diag_run.history["batch_schedules"][:n_batches]  # budget does not rescale schedules
    compressed = rt.fit_smoke(train_input, adversary_lambda=LAMBDA, model_seed=SEED, schedule_horizon_epochs=2)
    assert compressed.history["batch_schedules"][1:] != diag_run.history["batch_schedules"][1:]
    last = diag_run.history["epochs"][-1]["last_batch_schedule"]
    assert last["fit_progress"] < 2 / 256


@pytest.fixture
def smoke_model_dir():
    """Unique new directory under scratch/rq4b_implementation, independent of --basetemp."""
    import uuid

    path = rt.SMOKE_OUTPUT_ROOT / f"T2_R1_reload_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_exact_reload_without_training_metadata(cuda0, diag_run, train_input, smoke_model_dir):
    estimator = diag_run.estimator
    tmp_path = smoke_model_dir  # model files must live under the smoke root regardless of --basetemp
    assert diag_run.log["batches"]  # smoke recorded per-batch rows before stripping
    evidence = rt.save_and_verify_reload(estimator, train_input.X, tmp_path / "model.joblib")
    assert evidence["probabilities_exactly_equal"] and not evidence["training_metadata_pickled"]
    assert evidence["adversary_present_after_reload"] and evidence["interface_device_after_reload"] == "cuda:0"
    with pytest.raises(rt.RQ4bReloadError, match="overwrite"):
        rt.save_and_verify_reload(estimator, train_input.X, tmp_path / "model.joblib")

    def broken(path):
        raise OSError("simulated")

    with pytest.raises(rt.RQ4bReloadError, match="failed to reload"):
        rt.save_and_verify_reload(estimator, train_input.X, tmp_path / "broken.joblib", loader=broken)

    class _Shifted:
        def __init__(self, inner):
            self.inner = inner

        def predict_proba(self, X):
            return self.inner.predict_proba(X) + 1e-12

    import joblib

    with pytest.raises(rt.RQ4bReloadError, match="changed"):
        rt.save_and_verify_reload(estimator, train_input.X, tmp_path / "shifted.joblib", loader=lambda p: _Shifted(joblib.load(p)))
    with pytest.raises(rt.RQ4bReloadError, match="outside"):
        rt.save_and_verify_reload(estimator, train_input.X, rt.PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp" / "x.joblib")


def test_gpu_refit_route_with_two_epoch_selection_keeps_fresh_state(cuda0, contract, population, diag_run):
    record = _selection(contract, population, minimum_epoch=2)
    runs = []

    def gpu_fitter(inputs, spec):
        assert spec.schedule_horizon_epochs == 256 and spec.execution_epochs == 2
        result = rt.fit_smoke(inputs, adversary_lambda=spec.adversary_lambda, model_seed=spec.model_seed, execution_epochs=spec.execution_epochs)
        runs.append(result)
        return result

    access = ri.StageAccess("full_train_refit", contract)
    first = rt.run_full_train_refit(_context(contract), access, population, record, adversary_lambda=0.1, model_seed=42, fitter=gpu_fitter)
    second = rt.run_full_train_refit(_context(contract), access, population, record, adversary_lambda=0.1, model_seed=42, fitter=gpu_fitter)
    assert first.selected_epoch == 2 and first.history["budget"]["module_schedule_horizon_epochs"] == 256
    assert first.estimator is not second.estimator and first.preprocessor is not second.preprocessor
    assert first.estimator.backend_ is not second.estimator.backend_
    assert first.log["adversary_rng_isolation"]["initial_state_sha256"] == second.log["adversary_rng_isolation"]["initial_state_sha256"]
    assert first.log["adversary_updates"] > 0 and first.log["alignment"]["dataset_rows_verified"] == len(population.record_ids)
    _assert_close(_tensors(first.estimator), _tensors(second.estimator))  # same-seed short smoke only
