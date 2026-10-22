"""Synthetic-data tests for the Dataset B RQ4a runner (T5).

Every dataset here is generated in the test; no real prediction, raw, split, or
bootstrap-index artifact is read, and no production run is executed.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_b.mitigation import runner
from src.dataset_b.mitigation.inputs import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RQ3_PROTOCOL_PATH,
    LOCKED_G6,
    MODELS,
    RQ4AInputError,
    load_config,
    validate_config,
)
from src.dataset_b.mitigation.postprocessing import RQ4APolicyError, load_policy
from src.dataset_b.mitigation.runner import (
    ModelData,
    RQ4ARunnerError,
    RunnerInputs,
    RunSettings,
    execute_run,
    primary_criterion,
    replicate_streams,
    stability_status,
    stratified_resample,
    summarize_quantity,
)
from src.utils.path import PROJECT_ROOT


RUN_IDS = {"RQ4a-3": "rq4a3_postproc_20261017_01", "RQ4a-6": "rq4a6_postproc_20261017_01"}
OUTSIDE = ("Race Not Available", "Free Form Text Only")
GROUP_SHIFT = {
    "White": 0.5,
    "Black or African American": -0.4,
    "Asian": 0.2,
    "American Indian or Alaska Native": -0.2,
    "2 or more minority races": -0.3,
    "Joint": 0.3,
    "Race Not Available": 0.0,
    "Free Form Text Only": 0.1,
}
SIZES = {**{g: 160 for g in LOCKED_G6}, "Race Not Available": 80, "Free Form Text Only": 40}
THRESHOLDS = {"xgboost": 0.5, "realmlp": 0.6}
REPLICATES = 24
SETTINGS = RunSettings(replicates=REPLICATES, minimum_valid=20)
ROWS = {"validation": sum(SIZES.values()), "test": sum(SIZES.values())}


def _rid(n: int) -> str:
    return f"dataset_b-{n:09d}"


def _cohort(rng: np.random.Generator, start: int) -> pd.DataFrame:
    frames = []
    for group, size in SIZES.items():
        latent = rng.normal(GROUP_SHIFT[group], 1.0, size)
        y = (rng.random(size) < 1.0 / (1.0 + np.exp(-(1.3 * latent)))).astype(np.int8)
        sex = rng.choice(["Male", "Female", "Information not provided"], size=size, p=[0.55, 0.4, 0.05])
        frames.append(pd.DataFrame({"derived_race": group, "applicant_sex_group": sex, "y_approved": y, "latent": latent}))
    frame = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=int(start)).reset_index(drop=True)
    frame.insert(0, "record_id", [_rid(start + i) for i in range(len(frame))])
    return frame


def _model_frame(cohort: pd.DataFrame, rng: np.random.Generator, threshold: float, scale: float) -> pd.DataFrame:
    p = 1.0 / (1.0 + np.exp(-(scale * cohort["latent"].to_numpy() + rng.normal(0.0, 0.4, len(cohort)))))
    return pd.DataFrame(
        {
            "record_id": cohort["record_id"].to_numpy(),
            "y_approved": cohort["y_approved"].to_numpy(dtype=np.int8),
            "p_approved": p,
            "predicted_approved": (p >= threshold).astype(np.int8),
            "derived_race": cohort["derived_race"].to_numpy(),
            "applicant_sex_group": cohort["applicant_sex_group"].to_numpy(),
        }
    )


@pytest.fixture(scope="module")
def config():
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def contract(config):
    return validate_config(config)


@pytest.fixture(scope="module")
def sex():
    return runner.sex_rosters(yaml.safe_load(DEFAULT_RQ3_PROTOCOL_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def cohorts():
    rng = np.random.default_rng(20261017)
    validation = _cohort(rng, 1)
    test = _cohort(rng, 10_001)
    models = {}
    for model, scale in (("xgboost", 1.4), ("realmlp", 1.1)):
        models[model] = ModelData(
            model,
            THRESHOLDS[model],
            _model_frame(validation, rng, THRESHOLDS[model], scale),
            _model_frame(test, rng, THRESHOLDS[model], scale),
        )
    indices = np.random.default_rng(7).integers(0, len(test), size=(REPLICATES, len(test))).astype(np.int32)
    return models, indices


def _inputs(config, contract, sex, models, indices, name="RQ4a-3") -> RunnerInputs:
    return RunnerInputs(config, contract, contract.configurations[name], models, indices, sex[0], sex[1])


@pytest.fixture(scope="module")
def runs(tmp_path_factory, config, contract, sex, cohorts):
    models, indices = cohorts
    out = {}
    for name in RUN_IDS:
        root = tmp_path_factory.mktemp(f"runs_{name}")
        result = execute_run(_inputs(config, contract, sex, models, indices, name), RUN_IDS[name], root, settings=SETTINGS)
        out[name] = (Path(result["run_root"]), result)
    return out


def _read(run_root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(run_root / relative, dtype={"record_id": str}, float_precision="round_trip")


# --------------------------------------------------------------------------- pure helpers


def test_replicate_streams_are_deterministic_independent_and_follow_root_formula() -> None:
    a, b = replicate_streams(3), replicate_streams(3)
    assert a.root_seed == 10003 and a.decision_seed == b.decision_seed
    assert np.array_equal(a.calibration_rng.integers(0, 10**9, 50), b.calibration_rng.integers(0, 10**9, 50))
    assert a.calibration_stream_id != a.decision_stream_id
    assert str(a.decision_seed) in a.decision_stream_id
    assert len({replicate_streams(r).decision_seed for r in range(200)}) == 200


def test_stratified_resample_preserves_stratum_sizes_and_is_reproducible() -> None:
    race = np.array(["A"] * 40 + ["B"] * 25 + ["A"] * 10)
    y = np.array([1] * 30 + [0] * 10 + [1] * 5 + [0] * 20 + [0] * 10)
    first = stratified_resample(race, y, np.random.default_rng(1))
    second = stratified_resample(race, y, np.random.default_rng(1))
    assert np.array_equal(first, second) and len(first) == len(race)
    original = pd.Series(list(zip(race, y))).value_counts().to_dict()
    resampled = pd.Series(list(zip(race[first], y[first]))).value_counts().to_dict()
    assert original == resampled
    assert len(np.unique(first)) < len(first)


def _criterion(**overrides):
    base = dict(
        point_target_valid=True,
        point_delta_eod=-0.05,
        delta_eod_ci_upper=-0.01,
        n_valid_delta_eod=1000,
        point_ba_cost=0.004,
        ba_cost_ci_upper=0.009,
        n_valid_ba_cost=1000,
        minimum_valid=950,
        ba_budget=0.01,
    )
    base.update(overrides)
    return primary_criterion(**base)


def test_primary_criterion_pass_fail_and_not_evaluable_paths() -> None:
    assert _criterion() == ("pass", [])
    assert _criterion(ba_cost_ci_upper=0.01) == ("pass", [])
    assert _criterion(point_delta_eod=0.0) == ("fail", ["target_eod_reduction_not_demonstrated"])
    assert _criterion(delta_eod_ci_upper=0.0) == ("fail", ["target_eod_reduction_not_demonstrated"])
    assert _criterion(point_ba_cost=0.0101) == ("fail", ["ba_budget_exceeded"])
    assert _criterion(ba_cost_ci_upper=0.02) == ("fail", ["ba_budget_exceeded"])
    assert _criterion(point_delta_eod=0.1, ba_cost_ci_upper=0.02) == (
        "fail",
        ["target_eod_reduction_not_demonstrated", "ba_budget_exceeded"],
    )
    assert _criterion(point_target_valid=False, point_delta_eod=None) == ("not_evaluable", ["required_group_metric_invalid"])
    assert _criterion(n_valid_delta_eod=949, delta_eod_ci_upper=None) == ("not_evaluable", ["insufficient_valid_bootstrap_replicates"])
    assert _criterion(n_valid_ba_cost=949, ba_cost_ci_upper=None) == ("not_evaluable", ["insufficient_valid_bootstrap_replicates"])
    assert _criterion(point_target_valid=False, n_valid_delta_eod=0) == (
        "not_evaluable",
        ["required_group_metric_invalid", "insufficient_valid_bootstrap_replicates"],
    )
    with pytest.raises(RQ4ARunnerError):
        _criterion(ba_cost_ci_upper=None)


def _sensitivity(delta: list[float], cost: list[float], valid: list[bool] | None = None) -> pd.DataFrame:
    n = len(delta)
    valid = valid or [True] * n
    return pd.DataFrame(
        {"seed": list(range(42, 42 + n)), "delta_target_eod": delta, "ba_cost": cost, "target_eod_valid": valid, "ba_valid": [True] * n}
    )


def test_stability_status_stable_unstable_and_not_assessed() -> None:
    seeds = tuple(range(42, 72))
    kwargs = dict(seeds=seeds, reference_seed=42, ba_budget=0.01)
    assert stability_status(_sensitivity([-0.02] * 30, [0.005] * 30), **kwargs) == "stable"
    # Stable is not numerical invariance: values may move while indicators stay fixed.
    assert stability_status(_sensitivity(list(np.linspace(-0.05, -0.01, 30)), [0.011] * 30), **kwargs) == "stable"
    assert stability_status(_sensitivity([-0.02] * 29 + [0.001], [0.005] * 30), **kwargs) == "unstable"
    assert stability_status(_sensitivity([-0.02] * 30, [0.005] * 29 + [0.0101]), **kwargs) == "unstable"
    assert stability_status(_sensitivity([-0.02] * 29, [0.005] * 29), **kwargs) == "not_assessed"
    assert stability_status(_sensitivity([-0.02] * 30, [0.005] * 30, [True] * 29 + [False]), **kwargs) == "not_assessed"


def test_summarize_quantity_uses_valid_values_linear_percentiles_and_gate() -> None:
    values = pd.Series([0.1, 0.2, None, 0.4, 0.5, 9.9])
    valid = pd.Series([True, True, False, True, True, False])
    summary = summarize_quantity(values, valid, 0.3, minimum_valid=4, confidence=0.95)
    expected = np.quantile([0.1, 0.2, 0.4, 0.5], [(1.0 - 0.95) / 2.0, 1.0 - (1.0 - 0.95) / 2.0], method="linear")
    assert (summary["ci_lower"], summary["ci_upper"]) == (float(expected[0]), float(expected[1]))
    assert (summary["n_replicates"], summary["n_valid"], summary["valid_gate_status"]) == (6, 4, "pass")
    gated = summarize_quantity(values, valid, 0.3, minimum_valid=5, confidence=0.95)
    assert (gated["ci_lower"], gated["ci_upper"], gated["valid_gate_status"]) == (None, None, "fail")


# --------------------------------------------------------------------------- integration


@pytest.mark.parametrize("name", list(RUN_IDS))
def test_run_layout_columns_and_status_match_locked_yaml(runs, config, name) -> None:
    run_root, result = runs[name]
    assert result["status"] == "PASS" and run_root.name == RUN_IDS[name]
    assert {p.relative_to(run_root).as_posix() for p in run_root.rglob("*") if p.is_file()} == runner.expected_files(MODELS)
    for section, filename in (
        ("validation", "decisions.csv"),
        ("test_evaluation", "bootstrap_replicates.csv"),
        ("reporting", "experiment_metrics.csv"),
        ("reporting", "group_metrics.csv"),
        ("reporting", "decision_changes.csv"),
        ("reporting", "randomization_sensitivity.csv"),
        ("reporting", "bootstrap_summary.csv"),
        ("reporting", "runtime.csv"),
    ):
        assert _read(run_root, f"{section}/{filename}").columns.tolist() == config["outputs"][section][filename]["columns"]
    status = json.loads((run_root / "test_evaluation" / "status.json").read_text(encoding="utf-8"))
    assert status["run_id"] == RUN_IDS[name] and status["configuration_id"] == name
    assert list(status["model_statuses"]) == list(MODELS)
    for entry in status["model_statuses"].values():
        assert entry["execution_status"] == "completed"
        assert entry["stability_status"] in {"stable", "unstable", "not_assessed"}
        assert entry["primary_criterion"] in {"pass", "fail", "not_evaluable"}
    roles = _read(run_root, "reporting/group_metrics.csv")[["tpr_extremum_role", "fpr_extremum_role", "selection_rate_extremum_role"]]
    assert set(roles.to_numpy().ravel()) <= {"min", "max", "none", "not_evaluable"}
    assert set(_read(run_root, "reporting/group_metrics.csv")["attribute"]) == {"derived_race", "applicant_sex"}


@pytest.mark.parametrize("name", list(RUN_IDS))
def test_ordered_ids_and_outside_roster_fallback(runs, cohorts, contract, name) -> None:
    run_root, _ = runs[name]
    models, _ = cohorts
    roster = set(contract.configurations[name].roster)
    for section, partition in (("validation/decisions.csv", "validation"), ("test_evaluation/decisions.csv", "test")):
        frame = _read(run_root, section)
        for model in MODELS:
            rows = frame[frame["model"] == model]
            source = getattr(models[model], partition)
            assert rows["record_id"].tolist() == source["record_id"].tolist()
            assert np.array_equal(rows["baseline_decision"].to_numpy(), source["predicted_approved"].to_numpy())
            assert rows["in_target_roster"].tolist() == rows["derived_race"].isin(roster).tolist()
            outside = rows[~rows["in_target_roster"]]
            assert len(outside) > 0
            assert (outside["mitigated_decision"] == outside["baseline_decision"]).all()
            assert (rows["decision_changed"] == (rows["mitigated_decision"] != rows["baseline_decision"])).all()


@pytest.mark.parametrize("name", list(RUN_IDS))
def test_summary_criterion_and_stability_recompute_from_artifacts(runs, config, contract, name) -> None:
    run_root, _ = runs[name]
    reps = _read(run_root, "test_evaluation/bootstrap_replicates.csv")
    summary = _read(run_root, "reporting/bootstrap_summary.csv")
    sens = _read(run_root, "reporting/randomization_sensitivity.csv")
    experiment = _read(run_root, "reporting/experiment_metrics.csv")
    status = json.loads((run_root / "test_evaluation" / "status.json").read_text(encoding="utf-8"))["model_statuses"]
    target = contract.configurations[name].target_eod
    for model in MODELS:
        r = reps[reps["model"] == model]
        assert r["replicate_id"].tolist() == list(range(REPLICATES))
        assert (r["root_seed"] == 10000 + r["replicate_id"]).all()
        s = summary[summary["model"] == model].set_index("quantity")
        for quantity, flag in (("delta_target_eod", "target_eod_valid"), ("ba_cost", "ba_valid"), ("delta_audit_eod", "audit_eod_valid")):
            kept = r.loc[r[flag], quantity].to_numpy(dtype=float)
            assert s.loc[quantity, "n_valid"] == len(kept) and s.loc[quantity, "n_replicates"] == REPLICATES
            lower, upper = np.quantile(kept, [(1.0 - 0.95) / 2.0, 1.0 - (1.0 - 0.95) / 2.0], method="linear")
            assert s.loc[quantity, "ci_lower"] == lower
            assert s.loc[quantity, "ci_upper"] == upper
        test_rows = experiment[(experiment["model"] == model) & (experiment["partition"] == "test")].set_index("policy")
        point_delta = test_rows.loc["mitigated", target] - test_rows.loc["baseline", target]
        point_cost = test_rows.loc["baseline", "balanced_accuracy"] - test_rows.loc["mitigated", "balanced_accuracy"]
        assert s.loc["delta_target_eod", "point_estimate"] == pytest.approx(point_delta, rel=1e-12)
        seed42 = sens[(sens["model"] == model) & (sens["seed"] == 42)].iloc[0]
        assert seed42["delta_target_eod"] == pytest.approx(point_delta, rel=1e-12)
        assert seed42["ba_cost"] == pytest.approx(point_cost, rel=1e-12)
        criterion, reasons = primary_criterion(
            point_target_valid=bool(test_rows.loc["mitigated", f"{target}_valid"]),
            point_delta_eod=float(s.loc["delta_target_eod", "point_estimate"]),
            delta_eod_ci_upper=float(s.loc["delta_target_eod", "ci_upper"]),
            n_valid_delta_eod=int(s.loc["delta_target_eod", "n_valid"]),
            point_ba_cost=float(s.loc["ba_cost", "point_estimate"]),
            ba_cost_ci_upper=float(s.loc["ba_cost", "ci_upper"]),
            n_valid_ba_cost=int(s.loc["ba_cost", "n_valid"]),
            minimum_valid=SETTINGS.minimum_valid,
            ba_budget=0.01,
        )
        stability = stability_status(sens[sens["model"] == model], seeds=range(42, 72), reference_seed=42, ba_budget=0.01)
        assert status[model]["primary_criterion"] == criterion
        assert status[model]["stability_status"] == stability
        expected = reasons + (["unstable_diagnostics"] if stability == "unstable" else [])
        assert status[model]["reason_codes"] == (expected or ["none"])


def test_policy_reload_reproduces_seed_42_decisions_and_index_hashes(runs, contract) -> None:
    run_root, _ = runs["RQ4a-6"]
    roster = contract.configurations["RQ4a-6"].roster
    decisions = _read(run_root, "test_evaluation/decisions.csv")
    for model in MODELS:
        policy = load_policy(run_root / "validation" / "fitted_policies" / model)
        rows = decisions[decisions["model"] == model]
        again, _ = runner.apply_policy(policy, rows["record_id"], rows["derived_race"], rows["baseline_decision"], roster, seed=42)
        assert np.array_equal(again, rows["mitigated_decision"].to_numpy())
    index = json.loads((run_root / "reporting" / "artifact_index.json").read_text(encoding="utf-8"))
    assert index["files"] == runner.build_file_entries(run_root)
    assert {entry["relative_path"] for entry in index["files"]} == runner.expected_files(MODELS) - {"reporting/artifact_index.json"}


def test_rerun_is_reproducible_and_models_are_independent(tmp_path, runs, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    first_root, _ = runs["RQ4a-3"]
    again = Path(execute_run(_inputs(config, contract, sex, models, indices), RUN_IDS["RQ4a-3"], tmp_path / "again", settings=SETTINGS)["run_root"])
    for relative in ("test_evaluation/bootstrap_replicates.csv", "test_evaluation/decisions.csv", "validation/decisions.csv", "reporting/randomization_sensitivity.csv", "reporting/bootstrap_summary.csv"):
        assert (first_root / relative).read_bytes() == (again / relative).read_bytes(), relative

    realmlp = models["realmlp"]
    shifted_p = np.clip(realmlp.validation["p_approved"].to_numpy() ** 0.7, 0, 1)
    shifted = dataclasses.replace(
        realmlp,
        validation=realmlp.validation.assign(p_approved=shifted_p, predicted_approved=(shifted_p >= realmlp.threshold).astype(np.int8)),
    )
    changed = Path(
        execute_run(_inputs(config, contract, sex, {"xgboost": models["xgboost"], "realmlp": shifted}, indices), RUN_IDS["RQ4a-3"], tmp_path / "changed", settings=SETTINGS)["run_root"]
    )
    reps_a, reps_b = _read(first_root, "test_evaluation/bootstrap_replicates.csv"), _read(changed, "test_evaluation/bootstrap_replicates.csv")
    pd.testing.assert_frame_equal(reps_a[reps_a["model"] == "xgboost"], reps_b[reps_b["model"] == "xgboost"])
    assert not reps_a[reps_a["model"] == "realmlp"].reset_index(drop=True).equals(reps_b[reps_b["model"] == "realmlp"].reset_index(drop=True))


def test_policy_is_refit_on_every_replicate_resample(monkeypatch, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    calls = []
    original = runner.fit_policy

    def counting(adapter, record_ids, *args, **kwargs):
        calls.append(np.asarray(record_ids).copy())
        return original(adapter, record_ids, *args, **kwargs)

    monkeypatch.setattr(runner, "fit_policy", counting)
    settings = RunSettings(replicates=5, minimum_valid=5)
    inputs = _inputs(config, contract, sex, models, indices[:5])
    data = models["xgboost"]
    reps = runner.bootstrap_model(data, runner._adapter(data), inputs, settings, RUN_IDS["RQ4a-3"])
    assert len(calls) == 5 and len(reps) == 5
    for r, ids in enumerate(calls):
        stream = replicate_streams(r)
        expected = stratified_resample(data.validation["derived_race"], data.validation["y_approved"], stream.calibration_rng)
        assert np.array_equal(ids, data.validation["record_id"].to_numpy()[expected])
    assert len({tuple(ids[:50]) for ids in calls}) == 5


def test_repeated_test_ids_receive_independent_draws_in_locked_order(config, contract, sex, cohorts) -> None:
    models, _ = cohorts
    data = models["xgboost"]
    roster = contract.g3
    policy = runner.fit_policy(runner._adapter(data), data.validation["record_id"], data.validation["y_approved"], data.validation["derived_race"], roster)
    test = data.test
    in_roster = test[test["derived_race"].isin(roster)]
    pmf = policy._pmf_predict(pd.DataFrame({"record_id": in_roster["record_id"]}), sensitive_features=in_roster["derived_race"])[:, 1]
    randomized = np.flatnonzero((pmf > 0.05) & (pmf < 0.95))
    assert len(randomized) > 0
    position = int(test.index.get_loc(in_roster.index[randomized[0]]))
    indices = np.full((1, len(test)), position, dtype=np.int32)
    inputs = _inputs(config, contract, sex, models, indices)
    reps = runner.bootstrap_model(data, runner._adapter(data), inputs, RunSettings(1, 1), RUN_IDS["RQ4a-3"])
    stream = replicate_streams(0)
    ids = np.repeat(test["record_id"].to_numpy()[position], len(test))
    draws, _ = runner.apply_policy(policy, ids, np.repeat(test["derived_race"].to_numpy()[position], len(test)), np.repeat(test["predicted_approved"].to_numpy()[position], len(test)), roster, seed=stream.decision_seed)
    assert set(np.unique(draws)) == {0, 1}
    assert reps.loc[0, "decision_stream_id"] == stream.decision_stream_id
    assert reps.loc[0, "ba_valid"] in (False, np.False_)


def test_invalid_replicates_are_retained_and_gate_only_their_quantity(tmp_path, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    test = models["xgboost"].test
    white = np.flatnonzero(test["derived_race"].to_numpy() == "White")
    approved = np.flatnonzero(test["y_approved"].to_numpy() == 1)
    custom = indices.copy()
    custom[0] = np.resize(white, len(test)).astype(np.int32)
    custom[1] = np.resize(approved, len(test)).astype(np.int32)
    root = Path(execute_run(_inputs(config, contract, sex, models, custom), RUN_IDS["RQ4a-3"], tmp_path, settings=SETTINGS)["run_root"])
    reps = _read(root, "test_evaluation/bootstrap_replicates.csv")
    summary = _read(root, "reporting/bootstrap_summary.csv")
    for model in MODELS:
        r = reps[reps["model"] == model].set_index("replicate_id")
        assert len(r) == REPLICATES
        assert (not r.loc[0, "target_eod_valid"]) and r.loc[0, "ba_valid"] and pd.isna(r.loc[0, "delta_target_eod"])
        assert r.loc[0, "invalid_reason_codes"] == "target_eod:required_group_metric_invalid;audit_eod:required_group_metric_invalid"
        assert (not r.loc[1, "ba_valid"]) and pd.isna(r.loc[1, "ba_cost"])
        assert r.loc[1, "invalid_reason_codes"].startswith("ba:single_actual_class")
        s = summary[summary["model"] == model].set_index("quantity")
        assert s.loc["delta_target_eod", "n_valid"] == REPLICATES - 2
        assert s.loc["ba_cost", "n_valid"] == REPLICATES - 1
        assert (r.loc[2:, "invalid_reason_codes"] == "none").all()


def test_minimum_valid_gate_makes_criterion_not_evaluable(tmp_path, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    test = models["xgboost"].test
    custom = indices.copy()
    custom[0] = np.resize(np.flatnonzero(test["derived_race"].to_numpy() == "White"), len(test)).astype(np.int32)
    settings = RunSettings(replicates=REPLICATES, minimum_valid=REPLICATES)
    root = Path(execute_run(_inputs(config, contract, sex, models, custom), RUN_IDS["RQ4a-3"], tmp_path, settings=settings)["run_root"])
    status = json.loads((root / "test_evaluation" / "status.json").read_text(encoding="utf-8"))["model_statuses"]
    summary = _read(root, "reporting/bootstrap_summary.csv")
    for model in MODELS:
        assert status[model]["primary_criterion"] == "not_evaluable"
        assert "insufficient_valid_bootstrap_replicates" in status[model]["reason_codes"]
        s = summary[summary["model"] == model].set_index("quantity")
        assert s.loc["delta_target_eod", "valid_gate_status"] == "fail" and pd.isna(s.loc["delta_target_eod", "ci_upper"])
        assert s.loc["ba_cost", "valid_gate_status"] == "pass"


def test_point_target_eod_invalid_is_not_evaluable_and_not_assessed(tmp_path, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    def thin(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        asian_denied = frame.index[(frame["derived_race"] == "Asian") & (frame["y_approved"] == 0)]
        frame.loc[asian_denied[20:], "derived_race"] = "Race Not Available"
        return frame
    thinned = {m: dataclasses.replace(d, test=thin(d.test)) for m, d in models.items()}
    root = Path(execute_run(_inputs(config, contract, sex, thinned, indices), RUN_IDS["RQ4a-3"], tmp_path, settings=SETTINGS)["run_root"])
    status = json.loads((root / "test_evaluation" / "status.json").read_text(encoding="utf-8"))["model_statuses"]
    for model in MODELS:
        assert status[model]["primary_criterion"] == "not_evaluable"
        assert status[model]["reason_codes"][0] == "required_group_metric_invalid"
        assert status[model]["stability_status"] == "not_assessed"
    sens = _read(root, "reporting/randomization_sensitivity.csv")
    assert (sens["invalid_reason_codes"].str.contains("target_eod:required_group_metric_invalid")).all()


# --------------------------------------------------------------------------- negative paths


def test_output_collision_rq3_namespace_and_bad_identity_are_rejected(tmp_path, runs, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    inputs = _inputs(config, contract, sex, models, indices)
    existing_root, _ = runs["RQ4a-3"]
    before = runner.build_file_entries(existing_root)
    with pytest.raises(RQ4ARunnerError, match="overwrite is prohibited"):
        execute_run(inputs, RUN_IDS["RQ4a-3"], existing_root.parent, settings=SETTINGS)
    assert runner.build_file_entries(existing_root) == before
    with pytest.raises(RQ4ARunnerError, match="RQ3 namespace"):
        runner.resolve_run_root(PROJECT_ROOT / "artifacts" / "dataset_b" / "rq3", RUN_IDS["RQ4a-3"])
    for bad in ("rq4a6_postproc_20261017_01", "rq4a3_postproc_20261340_01", "rq4a3_postproc_20261017_00"):
        with pytest.raises(RQ4AInputError):
            execute_run(inputs, bad, tmp_path, settings=SETTINGS)
    assert list(tmp_path.iterdir()) == []


def test_invalid_inputs_fail_closed(tmp_path, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    xgb, realmlp = models["xgboost"], models["realmlp"]
    cases = [
        ({"realmlp": realmlp, "xgboost": xgb}, indices, SETTINGS),
        ({"xgboost": xgb, "realmlp": dataclasses.replace(realmlp, test=realmlp.test.iloc[::-1].reset_index(drop=True))}, indices, SETTINGS),
        ({"xgboost": xgb, "realmlp": dataclasses.replace(realmlp, threshold=0.4)}, indices, SETTINGS),
        (models, indices.astype(np.int64), SETTINGS),
        (models, indices[:-1], SETTINGS),
        (models, indices, RunSettings(REPLICATES, REPLICATES + 1)),
    ]
    no_asian = {
        m: dataclasses.replace(d, validation=d.validation.assign(derived_race=d.validation["derived_race"].replace("Asian", "Race Not Available")))
        for m, d in models.items()
    }
    cases.append((no_asian, indices, SETTINGS))
    for case_models, case_indices, settings in cases:
        with pytest.raises(RQ4ARunnerError):
            execute_run(_inputs(config, contract, sex, case_models, case_indices), RUN_IDS["RQ4a-3"], tmp_path, settings=settings)
    assert list(tmp_path.iterdir()) == []


def test_failure_during_run_leaves_no_partial_artifact(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    models, indices = cohorts

    def broken(*args, **kwargs):
        raise RQ4APolicyError("simulated persistence failure")

    monkeypatch.setattr(runner, "save_policy", broken)
    with pytest.raises(RQ4APolicyError, match="simulated"):
        execute_run(_inputs(config, contract, sex, models, indices[:3]), RUN_IDS["RQ4a-3"], tmp_path, settings=RunSettings(3, 3))
    # Post-boundary failure: no completed run and no temporary directory; only the failure receipt.
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == ["_failures", f"_failures/{RUN_IDS['RQ4a-3']}.json"]
    receipt = json.loads((tmp_path / "_failures" / f"{RUN_IDS['RQ4a-3']}.json").read_text(encoding="utf-8"))
    assert (receipt["failed_stage"], receipt["model"]) == ("policy_persistence_and_reload", "xgboost")


# --------------------------------------------------------------------------- runner-contract gate and provenance


def _set(path: tuple, value):
    def mutate(cfg):
        node = cfg
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
    return mutate


def _delete(path: tuple):
    def mutate(cfg):
        node = cfg
        for part in path[:-1]:
            node = node[part]
        del node[path[-1]]
    return mutate


def _rename_file(section: str, old: str, new: str):
    def mutate(cfg):
        cfg["outputs"][section][new] = cfg["outputs"][section].pop(old)
    return mutate


TAMPERS = {
    "column_dropped": lambda cfg: cfg["outputs"]["reporting"]["experiment_metrics.csv"]["columns"].pop(),
    "columns_reordered": lambda cfg: cfg["outputs"]["validation"]["decisions.csv"]["columns"].reverse(),
    "file_renamed": _rename_file("reporting", "runtime.csv", "timing.csv"),
    "file_added": _set(("outputs", "reporting", "extra.csv"), {"columns": ["run_id"]}),
    "run_root_changed": _set(("outputs", "run_root"), "artifacts/dataset_b/rq4/<run_id>"),
    "policy_dir_changed": _set(("outputs", "validation", "fitted_policies_dir"), "validation/policies"),
    "reason_code_added": lambda cfg: cfg["status_contract"]["reason_codes"].append("custom_code"),
    "reason_code_removed": lambda cfg: cfg["status_contract"]["reason_codes"].remove("runtime_failure"),
    "status_value_added": lambda cfg: cfg["status_contract"]["stability_status"].append("partially_stable"),
    "provenance_flag_false": _set(("runtime", "record_actual_interpreter_and_imported_versions"), False),
    "provenance_flag_string": _set(("runtime", "record_actual_interpreter_and_imported_versions"), "true"),
    "provenance_flag_deleted": _delete(("runtime", "record_actual_interpreter_and_imported_versions")),
    "sensitivity_seeds_shortened": lambda cfg: cfg["randomized_decisions"]["sensitivity_seeds"].pop(),
    "sensitivity_output_seeds": _set(("outputs", "reporting", "randomization_sensitivity.csv", "seeds"), "42_through_70"),
    "stability_seed_count": _set(("stability_status", "required_seed_count"), 29),
    "write_mode_direct": _set(("reporting_contract", "reporting_write_mode"), "direct"),
    "reporting_contract_deleted": _delete(("reporting_contract",)),
    "prohibition_disabled": _set(("prohibitions", "prohibit_best_randomized_seed_selection"), False),
    "prohibition_removed": _delete(("prohibitions", "prohibit_locked_test_bootstrap_regeneration")),
    "prohibition_int_for_bool": _set(("prohibitions", "prohibit_dynamic_group_roster"), 1),
    "overwrite_allowed": _set(("run_identity", "overwrite_existing_run"), True),
    "base_inference_allowed": _set(("cached_probability_adapter", "base_model_inference"), True),
    "receipt_field_removed": lambda cfg: cfg["failure_receipts"]["fields"].remove("error_summary"),
    "receipt_fields_reordered": lambda cfg: cfg["failure_receipts"]["fields"].reverse(),
    "receipt_overwrite_allowed": _set(("failure_receipts", "overwrite_existing_receipt"), True),
    "receipt_write_mode": _set(("failure_receipts", "write_mode"), "direct"),
    "receipt_directory": _set(("failure_receipts", "output_root_override_relative_directory"), "failures"),
    "receipt_infeasible_class": _set(("failure_receipts", "infeasible_exception_class"), "ValueError"),
    "receipt_infer_infeasible": _set(("failure_receipts", "never_infer_infeasible_from_generic_exception"), False),
    "receipt_summary_limit": _set(("failure_receipts", "error_summary", "maximum_characters"), 2000),
    "receipt_runtime_status": _set(("failure_receipts", "execution_status", "runtime_exception"), "infeasible"),
    "receipt_boundary_shortened": lambda cfg: cfg["failure_receipts"]["attempt_begins_after"].pop(),
    "receipt_section_deleted": _delete(("failure_receipts",)),
    "collision_targets": _set(("run_identity", "collision_targets"), ["completed_run_directory"]),
    "receipt_not_consuming_run_id": _set(("run_identity", "failure_receipt_consumes_run_id"), False),
    "infeasible_reason_removed": lambda cfg: cfg["status_contract"]["reason_codes"].remove("postprocessing_infeasible"),
    "completed_status_widened": _set(("status_contract", "completed_run_status_json_execution_status"), ["completed", "failed"]),
    "receipt_status_narrowed": _set(("status_contract", "failure_receipt_execution_status"), ["failed"]),
    "attempt_level_flag": _set(("status_contract", "failed_and_infeasible_are_attempt_level_only"), False),
}


def test_runner_contract_gate_accepts_the_locked_yaml(config) -> None:
    runner.validate_runner_contract(config)
    assert runner.LOCKED_OUTPUTS is not config["outputs"]


@pytest.mark.parametrize("label", list(TAMPERS))
def test_tampered_runner_contract_fails_before_anything_is_published(tmp_path, config, contract, sex, cohorts, label) -> None:
    models, indices = cohorts
    tampered = copy.deepcopy(config)
    TAMPERS[label](tampered)
    with pytest.raises(RQ4ARunnerError, match="runner contract"):
        runner.validate_runner_contract(tampered)
    inputs = RunnerInputs(tampered, contract, contract.configurations["RQ4a-3"], models, indices, sex[0], sex[1])
    with pytest.raises(RQ4ARunnerError, match="runner contract"):
        execute_run(inputs, RUN_IDS["RQ4a-3"], tmp_path / "out", settings=SETTINGS)
    assert not (tmp_path / "out").exists()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(RQ4ARunnerError, match="runner contract"):
        runner.load_production_inputs(tampered, "RQ4a-3", project_root=tmp_path / "absent", rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)


def test_artifact_index_records_the_test_process_interpreter_and_modules(runs) -> None:
    import platform
    import sys

    import fairlearn
    import joblib
    import scipy
    import sklearn

    run_root, _ = runs["RQ4a-3"]
    index = json.loads((run_root / "reporting" / "artifact_index.json").read_text(encoding="utf-8"))
    assert list(index) == ["runtime_provenance", "files"]
    provenance = index["runtime_provenance"]
    assert Path(provenance["sys_executable"]) == Path(sys.executable)
    assert provenance["python_version"] == "{}.{}.{}".format(*sys.version_info[:3]) == platform.python_version()
    assert provenance["packages"] == {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
        "joblib": joblib.__version__,
        "fairlearn": fairlearn.__version__,
    }
    assert all(list(entry) == ["relative_path", "sha256", "bytes", "rows"] for entry in index["files"])


def _copy_run(tmp_path: Path, runs) -> Path:
    import shutil

    source, _ = runs["RQ4a-3"]
    copy_root = tmp_path / source.name
    shutil.copytree(source, copy_root)
    return copy_root


def _rewrite_index(copy_root: Path, mutate) -> None:
    path = copy_root / "reporting" / "artifact_index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    mutate(index)
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def _other_environment() -> dict:
    return {
        "sys_executable": "C:/auditor/python.exe",
        "python_version": "3.12.0",
        "packages": {name: "0.0.1" for name in runner.PROVENANCE_PACKAGES},
    }


def test_producer_verification_rejects_mismatched_current_process_provenance() -> None:
    current = runner.runtime_provenance()
    runner.verify_producer_provenance(current)
    for mismatch in (
        {**current, "python_version": "0.0.0"},
        {**current, "sys_executable": "C:/other/python.exe"},
        {**current, "packages": {**current["packages"], "fairlearn": "9.9.9"}},
    ):
        runner.validate_runtime_provenance_structure(mismatch)
        with pytest.raises(RQ4ARunnerError, match="producing interpreter"):
            runner.verify_producer_provenance(mismatch)


def test_execute_run_refuses_to_publish_when_stored_provenance_differs_from_producer(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    original = runner.build_artifact_index

    def foreign(root, provenance):
        return original(root, _other_environment())

    monkeypatch.setattr(runner, "build_artifact_index", foreign)
    with pytest.raises(RQ4ARunnerError, match="producing interpreter"):
        execute_run(_inputs(config, contract, sex, models, indices[:3]), RUN_IDS["RQ4a-3"], tmp_path, settings=RunSettings(3, 3))
    assert not (tmp_path / RUN_IDS["RQ4a-3"]).exists()
    assert [p.name for p in tmp_path.iterdir()] == ["_failures"]
    receipt = json.loads((tmp_path / "_failures" / f"{RUN_IDS['RQ4a-3']}.json").read_text(encoding="utf-8"))
    assert (receipt["execution_status"], receipt["failed_stage"], receipt["model"]) == ("failed", "finalize_and_validate_outputs", None)


def test_later_audit_accepts_saved_provenance_under_a_different_environment(tmp_path, monkeypatch, runs) -> None:
    copy_root = _copy_run(tmp_path, runs)
    index_path = copy_root / "reporting" / "artifact_index.json"
    before = index_path.read_bytes()
    stored = json.loads(before)["runtime_provenance"]
    monkeypatch.setattr(runner, "runtime_provenance", _other_environment)
    assert runner.runtime_provenance() != stored
    assert runner.validate_run_directory(copy_root, replicates=REPLICATES, rows=ROWS)["status"] == "PASS"
    with pytest.raises(RQ4ARunnerError, match="producing interpreter"):
        runner.validate_run_directory(copy_root, replicates=REPLICATES, rows=ROWS, verify_producer=True)
    assert index_path.read_bytes() == before
    assert json.loads(index_path.read_bytes())["runtime_provenance"] == stored


def test_published_artifact_retains_original_producer_provenance(runs, monkeypatch) -> None:
    run_root, _ = runs["RQ4a-6"]
    index_path = run_root / "reporting" / "artifact_index.json"
    before = index_path.read_bytes()
    stored = json.loads(before)["runtime_provenance"]
    monkeypatch.setattr(runner, "runtime_provenance", _other_environment)
    runner.validate_run_directory(run_root, replicates=REPLICATES, rows=ROWS)
    assert runner.build_artifact_index(run_root, stored)["runtime_provenance"] == stored
    assert index_path.read_bytes() == before


def _tamper_file_bytes(copy_root: Path) -> None:
    path = copy_root / "validation" / "fitted_policies" / "xgboost" / "policy_interpolation.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"]["packages"].pop("fairlearn")),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"]["packages"].__setitem__("extra_pkg", "1.0")),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"].__setitem__("packages", dict(reversed(list(i["runtime_provenance"]["packages"].items()))))),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"]["packages"].__setitem__("numpy", "")),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"].__setitem__("python_version", "")),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"].__setitem__("sys_executable", None)),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"].__setitem__("packages", list(runner.PROVENANCE_PACKAGES))),
        lambda root: _rewrite_index(root, lambda i: i["runtime_provenance"].__setitem__("extra_field", "x")),
        lambda root: _rewrite_index(root, lambda i: i.__setitem__("runtime_provenance", "3.13.14")),
        lambda root: _rewrite_index(root, lambda i: i.pop("runtime_provenance")),
        lambda root: _rewrite_index(root, lambda i: i["files"][0].__setitem__("extra", 1)),
        lambda root: _rewrite_index(root, lambda i: i["files"][0].__setitem__("sha256", "0" * 64)),
        _tamper_file_bytes,
    ],
    ids=[
        "package_missing", "package_added", "package_order", "package_empty", "python_version_empty",
        "executable_not_string", "packages_wrong_type", "provenance_extra_field", "provenance_wrong_type",
        "provenance_missing", "per_file_field", "index_hash", "file_bytes",
    ],
)
def test_tampered_provenance_structure_or_hashes_fail_later_audit(tmp_path, monkeypatch, runs, tamper) -> None:
    copy_root = _copy_run(tmp_path, runs)
    tamper(copy_root)
    monkeypatch.setattr(runner, "runtime_provenance", _other_environment)
    with pytest.raises(RQ4ARunnerError):
        runner.validate_run_directory(copy_root, replicates=REPLICATES, rows=ROWS)


def test_untampered_copy_passes_run_directory_validation(tmp_path, runs) -> None:
    copy_root = _copy_run(tmp_path, runs)
    assert runner.validate_run_directory(copy_root, replicates=REPLICATES, rows=ROWS)["status"] == "PASS"
    assert runner.validate_run_directory(copy_root, replicates=REPLICATES, rows=ROWS, verify_producer=True)["status"] == "PASS"


# --------------------------------------------------------------------------- failure receipts (T5-R3)


RECEIPT_FIELDS = [
    "schema_version", "run_id", "configuration_id", "execution_status", "reason_codes", "failed_stage",
    "model", "started_at_utc", "failed_at_utc", "runtime_provenance", "exception_type", "error_summary",
]


def _receipt(root: Path, run_id: str) -> dict:
    return json.loads((root / "_failures" / f"{run_id}.json").read_text(encoding="utf-8"))


def _assert_receipt_basics(receipt: dict, run_id: str, configuration_id: str) -> None:
    from datetime import datetime

    assert list(receipt) == RECEIPT_FIELDS
    assert type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
    assert (receipt["run_id"], receipt["configuration_id"]) == (run_id, configuration_id)
    started = datetime.fromisoformat(receipt["started_at_utc"])
    failed = datetime.fromisoformat(receipt["failed_at_utc"])
    assert started.utcoffset().total_seconds() == 0 and failed.utcoffset().total_seconds() == 0 and failed >= started
    assert receipt["runtime_provenance"] == runner.runtime_provenance()
    assert isinstance(receipt["error_summary"], str) and "\n" not in receipt["error_summary"] and len(receipt["error_summary"]) <= 1000
    runner.validate_failure_receipt(receipt, verify_producer=True)


def _fail_in(monkeypatch, name: str, error: BaseException, *, only_model: str | None = None) -> None:
    original = getattr(runner, name)

    def broken(*args, **kwargs):
        data = args[0] if args else None
        if only_model is None or getattr(data, "model", None) == only_model:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, name, broken)


def _small(config, contract, sex, cohorts, name="RQ4a-3"):
    models, indices = cohorts
    return _inputs(config, contract, sex, models, indices[:3], name)


SMALL = RunSettings(3, 3)


def test_successful_runs_publish_no_failure_receipt(runs) -> None:
    for run_root, _ in runs.values():
        assert not (run_root.parent / "_failures").exists()
        assert sorted(p.name for p in run_root.parent.iterdir()) == [run_root.name]


def test_generic_model_failure_publishes_failed_receipt_and_reraises_original(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    error = ValueError("boom in bootstrap")
    _fail_in(monkeypatch, "bootstrap_model", error, only_model="realmlp")
    with pytest.raises(ValueError) as raised:
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    assert raised.value is error
    assert not (tmp_path / RUN_IDS["RQ4a-3"]).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["_failures"]
    assert [p.name for p in (tmp_path / "_failures").iterdir()] == [f"{RUN_IDS['RQ4a-3']}.json"]
    receipt = _receipt(tmp_path, RUN_IDS["RQ4a-3"])
    _assert_receipt_basics(receipt, RUN_IDS["RQ4a-3"], "RQ4a-3")
    assert (receipt["execution_status"], receipt["reason_codes"]) == ("failed", ["runtime_failure"])
    assert (receipt["failed_stage"], receipt["model"]) == ("bootstrap", "realmlp")
    assert (receipt["exception_type"], receipt["error_summary"]) == ("builtins.ValueError", "boom in bootstrap")


def test_explicit_infeasible_error_publishes_infeasible_receipt(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    error = runner.RQ4AInfeasibleError("no feasible equalized-odds solution")
    _fail_in(monkeypatch, "_adapter", error, only_model="xgboost")
    with pytest.raises(runner.RQ4AInfeasibleError):
        execute_run(_small(config, contract, sex, cohorts, "RQ4a-6"), RUN_IDS["RQ4a-6"], tmp_path, settings=SMALL)
    receipt = _receipt(tmp_path, RUN_IDS["RQ4a-6"])
    _assert_receipt_basics(receipt, RUN_IDS["RQ4a-6"], "RQ4a-6")
    assert (receipt["execution_status"], receipt["reason_codes"]) == ("infeasible", ["postprocessing_infeasible"])
    assert (receipt["failed_stage"], receipt["model"]) == ("model_setup", "xgboost")
    assert receipt["exception_type"] == "src.dataset_b.mitigation.runner.RQ4AInfeasibleError"
    assert not (tmp_path / RUN_IDS["RQ4a-6"]).exists()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("infeasible solution"),
        RuntimeError("Fairlearn: no feasible threshold"),
        OSError("disk"),
        RQ4APolicyError("policy"),
        type("InfeasibleSubclass", (runner.RQ4AInfeasibleError,), {})("subclass"),
    ],
    ids=["value_error", "fairlearn_like", "os_error", "policy_error", "infeasible_subclass"],
)
def test_infeasibility_is_never_inferred_from_other_exceptions(tmp_path, monkeypatch, config, contract, sex, cohorts, error) -> None:
    _fail_in(monkeypatch, "fit_policy", error)
    with pytest.raises(type(error)):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    receipt = _receipt(tmp_path, RUN_IDS["RQ4a-3"])
    assert (receipt["execution_status"], receipt["reason_codes"]) == ("failed", ["runtime_failure"])
    assert (receipt["failed_stage"], receipt["model"]) == ("fit_complete_validation", "xgboost")


def test_second_model_failure_before_its_first_stage_names_that_model(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    _fail_in(monkeypatch, "_adapter", ValueError("realmlp setup failed"), only_model="realmlp")
    with pytest.raises(ValueError, match="realmlp setup failed"):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    receipt = _receipt(tmp_path, RUN_IDS["RQ4a-3"])
    _assert_receipt_basics(receipt, RUN_IDS["RQ4a-3"], "RQ4a-3")
    assert (receipt["failed_stage"], receipt["model"]) == ("model_setup", "realmlp")


@pytest.mark.parametrize("target", ["summarize_quantity", "primary_criterion", "stability_status"])
def test_failure_after_bootstrap_is_not_labelled_bootstrap(tmp_path, monkeypatch, config, contract, sex, cohorts, target) -> None:
    calls = []
    original = getattr(runner, target)

    def broken(*args, **kwargs):
        calls.append(1)
        raise ValueError(f"{target} failed")

    monkeypatch.setattr(runner, target, broken)
    with pytest.raises(ValueError, match=f"{target} failed"):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    assert calls and original is not broken
    receipt = _receipt(tmp_path, RUN_IDS["RQ4a-3"])
    assert (receipt["failed_stage"], receipt["model"]) == ("summary_and_criterion", "xgboost")


def test_run_level_failure_records_null_model(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    _fail_in(monkeypatch, "build_artifact_index", OSError("index write failed"))
    with pytest.raises(OSError, match="index write failed"):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    receipt = _receipt(tmp_path, RUN_IDS["RQ4a-3"])
    _assert_receipt_basics(receipt, RUN_IDS["RQ4a-3"], "RQ4a-3")
    assert (receipt["failed_stage"], receipt["model"]) == ("finalize_and_validate_outputs", None)


def test_error_summary_is_single_line_and_truncated(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    message = "first line\nsecond\tline\r\n" + "x" * 1500
    _fail_in(monkeypatch, "bootstrap_model", RuntimeError(message), only_model="xgboost")
    with pytest.raises(RuntimeError):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    summary = _receipt(tmp_path, RUN_IDS["RQ4a-3"])["error_summary"]
    assert len(summary) == 1000
    assert summary.startswith("first line second line xxx")
    assert "\n" not in summary and "\r" not in summary and "\t" not in summary
    assert runner.error_summary(ValueError("a\n\n  b")) == "a b"


def test_pre_boundary_failures_publish_nothing(tmp_path, runs, config, contract, sex, cohorts) -> None:
    models, indices = cohorts
    small = indices[:3]
    tampered_contract = copy.deepcopy(config)
    tampered_contract["failure_receipts"]["write_mode"] = "direct"
    bad_config = copy.deepcopy(config)
    bad_config["postprocessor"]["grid_size"] = 500
    no_prereq = copy.deepcopy(config)
    no_prereq["prerequisites"]["rq1_paired_bootstrap_task"] = "pending"
    cases = [
        (RQ4ARunnerError, RunnerInputs(tampered_contract, contract, contract.configurations["RQ4a-3"], models, small, sex[0], sex[1]), RUN_IDS["RQ4a-3"]),
        (RQ4AInputError, RunnerInputs(bad_config, contract, contract.configurations["RQ4a-3"], models, small, sex[0], sex[1]), RUN_IDS["RQ4a-3"]),
        (RQ4ARunnerError, RunnerInputs(no_prereq, contract, contract.configurations["RQ4a-3"], models, small, sex[0], sex[1]), RUN_IDS["RQ4a-3"]),
        (RQ4AInputError, _inputs(config, contract, sex, models, small), "rq4a3_postproc_20261017_00"),
        (RQ4ARunnerError, _inputs(config, contract, sex, models, small[:2]), RUN_IDS["RQ4a-3"]),
    ]
    for expected, inputs, run_id in cases:
        with pytest.raises(expected):
            execute_run(inputs, run_id, tmp_path, settings=SMALL)
    assert list(tmp_path.iterdir()) == []
    existing_root, _ = runs["RQ4a-3"]
    with pytest.raises(RQ4ARunnerError, match="overwrite is prohibited"):
        execute_run(_inputs(config, contract, sex, models, indices), RUN_IDS["RQ4a-3"], existing_root.parent, settings=SETTINGS)
    assert not (existing_root.parent / "_failures").exists()


def test_failure_receipt_consumes_run_id_and_is_never_overwritten(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    _fail_in(monkeypatch, "bootstrap_model", ValueError("first attempt"), only_model="xgboost")
    with pytest.raises(ValueError):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    path = tmp_path / "_failures" / f"{RUN_IDS['RQ4a-3']}.json"
    before = path.read_bytes()
    monkeypatch.undo()
    calls = []
    original = runner.fit_policy
    monkeypatch.setattr(runner, "fit_policy", lambda *a, **k: calls.append(1) or original(*a, **k))
    with pytest.raises(RQ4ARunnerError, match="consumed by a failure receipt"):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    assert calls == []
    assert path.read_bytes() == before
    assert not (tmp_path / RUN_IDS["RQ4a-3"]).exists()
    with pytest.raises(RQ4ARunnerError, match="overwrite an existing failure receipt"):
        runner.publish_failure_receipt(tmp_path, json.loads(before))
    assert path.read_bytes() == before


def test_receipt_respects_output_root_override(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    override = tmp_path / "custom" / "root"
    _fail_in(monkeypatch, "bootstrap_model", ValueError("x"), only_model="xgboost")
    with pytest.raises(ValueError):
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], override, settings=SMALL)
    assert (override / "_failures" / f"{RUN_IDS['RQ4a-3']}.json").is_file()
    assert runner.failure_receipt_path(override, RUN_IDS["RQ4a-3"]) == (override / "_failures" / f"{RUN_IDS['RQ4a-3']}.json").resolve()
    assert runner.failure_receipt_path(PROJECT_ROOT / config["artifact_namespace"], "rq4a3_postproc_20261017_01") == (
        PROJECT_ROOT / config["failure_receipts"]["directory"] / "rq4a3_postproc_20261017_01.json"
    ).resolve()


def test_receipt_write_failure_leaves_no_partial_receipt_or_run(tmp_path, monkeypatch, config, contract, sex, cohorts) -> None:
    import os

    error = ValueError("original failure")
    _fail_in(monkeypatch, "bootstrap_model", error, only_model="xgboost")
    seen = []

    def partial_write(path, payload):
        path.write_text('{"schema_version": 1, "run_id"', encoding="utf-8")
        seen.append(path)
        raise OSError("disk full while writing receipt")

    monkeypatch.setattr(runner, "_write_receipt_file", partial_write)
    with pytest.raises(ValueError) as raised:
        execute_run(_small(config, contract, sex, cohorts), RUN_IDS["RQ4a-3"], tmp_path, settings=SMALL)
    assert raised.value is error
    assert any("failure receipt was not published" in note for note in getattr(raised.value, "__notes__", []))
    assert seen and seen[0].name == f".{RUN_IDS['RQ4a-3']}.tmp-{os.getpid()}.json"
    assert not seen[0].exists()
    assert not (tmp_path / RUN_IDS["RQ4a-3"]).exists()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_production_entry_runs_gates_before_preflight_and_publishes_nothing(tmp_path, config, cohorts) -> None:
    project = tmp_path / "project"
    synthetic = _write_synthetic_project(project, config, cohorts)
    config_path = tmp_path / "synthetic_rq4a.yaml"
    config_path.write_text(yaml.safe_dump(synthetic, sort_keys=False), encoding="utf-8")
    out = tmp_path / "out"
    # Production-input preflight failure (raw/view reconciliation) publishes nothing.
    view_path = project / config["audit_join"]["controlled_split_views"]["validation"]
    view = pd.read_csv(view_path, dtype=str)
    view.loc[0, "action_taken"] = "9"
    view.to_csv(view_path, index=False)
    with pytest.raises(RQ4AInputError, match="action_taken"):
        runner.run_rq4a("RQ4a-3", RUN_IDS["RQ4a-3"], config_path=config_path, project_root=project, output_root=out)
    assert not out.exists()
    # A consumed run ID stops the production entry before any input file is read.
    (out / "_failures").mkdir(parents=True)
    (out / "_failures" / f"{RUN_IDS['RQ4a-3']}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RQ4ARunnerError, match="consumed by a failure receipt"):
        runner.run_rq4a("RQ4a-3", RUN_IDS["RQ4a-3"], config_path=config_path, project_root=tmp_path / "absent", output_root=out)
    with pytest.raises(RQ4AInputError):
        runner.run_rq4a("RQ4a-3", "rq4a6_postproc_20261017_01", config_path=config_path, project_root=tmp_path / "absent", output_root=out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == ["_failures", f"_failures/{RUN_IDS['RQ4a-3']}.json"]


def _valid_receipt() -> dict:
    from datetime import datetime, timedelta, timezone

    started = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)
    return runner.build_failure_receipt(
        run_id=RUN_IDS["RQ4a-3"],
        configuration_id="RQ4a-3",
        exc=ValueError("x"),
        failed_stage="bootstrap",
        model="xgboost",
        started_at=started,
        failed_at=started + timedelta(seconds=5),
    )


def _with(**changes):
    def mutate(receipt):
        receipt.update(changes)
    return mutate


RECEIPT_TAMPERS = {
    "field_order": lambda r: r.update({"run_id": r.pop("run_id")}),
    "extra_field": _with(extra="x"),
    "missing_field": lambda r: r.pop("failed_stage"),
    "schema_bool": _with(schema_version=True),
    "schema_string": _with(schema_version="1"),
    "schema_two": _with(schema_version=2),
    "run_id_pattern": _with(run_id="rq4a6_postproc_20261017_01"),
    "run_id_date": _with(run_id="rq4a3_postproc_20261399_01"),
    "configuration": _with(configuration_id="RQ4a-9"),
    "status_reason_mismatch": _with(reason_codes=["postprocessing_infeasible"]),
    "infeasible_from_generic": _with(execution_status="infeasible", reason_codes=["postprocessing_infeasible"]),
    "completed_status": _with(execution_status="completed", reason_codes=["none"]),
    "reason_codes_string": _with(reason_codes="runtime_failure"),
    "status_unhashable": _with(execution_status=["failed"]),
    "reason_code_unhashable": _with(reason_codes=[["runtime_failure"]]),
    "model_unknown": _with(model="lightgbm"),
    "stage_empty": _with(failed_stage=""),
    "naive_timestamp": _with(started_at_utc="2026-10-17T09:00:00"),
    "non_utc_timestamp": _with(failed_at_utc="2026-10-17T16:00:05+07:00"),
    "failed_before_started": _with(failed_at_utc="2026-10-17T08:59:59+00:00"),
    "timestamp_garbage": _with(started_at_utc="yesterday"),
    "summary_newline": _with(error_summary="a\nb"),
    "summary_too_long": _with(error_summary="y" * 1001),
    "summary_not_string": _with(error_summary=None),
    "exception_type_empty": _with(exception_type=""),
    "provenance_missing_package": lambda r: r["runtime_provenance"]["packages"].pop("joblib"),
    "provenance_empty_value": lambda r: r["runtime_provenance"].__setitem__("python_version", ""),
}


def test_valid_receipt_passes_and_audit_is_environment_independent(monkeypatch) -> None:
    receipt = _valid_receipt()
    runner.validate_failure_receipt(receipt, verify_producer=True)
    monkeypatch.setattr(runner, "runtime_provenance", _other_environment)
    runner.validate_failure_receipt(receipt)
    with pytest.raises(RQ4ARunnerError, match="producing interpreter"):
        runner.validate_failure_receipt(receipt, verify_producer=True)


@pytest.mark.parametrize("label", list(RECEIPT_TAMPERS))
def test_tampered_receipts_fail_validation(label) -> None:
    receipt = _valid_receipt()
    RECEIPT_TAMPERS[label](receipt)
    with pytest.raises(RQ4ARunnerError):
        runner.validate_failure_receipt(receipt)


# --------------------------------------------------------------------------- production loader on a synthetic tree


def _write_synthetic_project(root: Path, config, cohorts_frames) -> dict:
    models, _ = cohorts_frames
    validation, test = models["xgboost"].validation, models["xgboost"].test
    raw_rows = int(test["record_id"].str[-9:].astype(int).max()) + 5
    raw = pd.DataFrame({"lei": "LEI0", "action_taken": "1", "applicant_sex": "1", "derived_race": "Race Not Available"}, index=range(raw_rows))
    sex_code = {"Male": "1", "Female": "2", "Information not provided": "3"}
    for frame in (validation, test):
        ordinals = frame["record_id"].str[-9:].astype(int).to_numpy() - 1
        raw.loc[ordinals, "derived_race"] = frame["derived_race"].to_numpy()
        raw.loc[ordinals, "applicant_sex"] = frame["applicant_sex_group"].map(sex_code).to_numpy()
        raw.loc[ordinals, "lei"] = [f"LEI{i}" for i in ordinals]
        raw.loc[ordinals, "action_taken"] = np.where(frame["y_approved"].to_numpy() == 1, "1", "3")
    (root / "data" / "raw").mkdir(parents=True)
    raw.to_csv(root / config["audit_join"]["raw_source"], index=False)
    for partition, frame in (("validation", validation), ("test", test)):
        ordinals = frame["record_id"].str[-9:].astype(int).to_numpy() - 1
        view = pd.DataFrame(
            {
                "record_id": frame["record_id"],
                "lei": raw.loc[ordinals, "lei"].to_numpy(),
                "action_taken": raw.loc[ordinals, "action_taken"].to_numpy(),
                "y_approved": frame["y_approved"],
                "applicant_sex": raw.loc[ordinals, "applicant_sex"].to_numpy(),
            }
        )
        path = root / config["audit_join"]["controlled_split_views"][partition]
        path.parent.mkdir(parents=True, exist_ok=True)
        view.to_csv(path, index=False)
    rng = np.random.default_rng(99)
    for model, spec in config["baseline_models"].items():
        threshold = float(spec["frozen_threshold"])
        for partition, frame in (("validation", validation), ("test", test)):
            p = np.clip(frame["p_approved"].to_numpy() + rng.normal(0, 0.02, len(frame)), 0, 1)
            path = root / spec[f"{partition}_predictions"]
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"record_id": frame["record_id"], "y_approved": frame["y_approved"], "p_approved": p, "predicted_approved": (p >= threshold).astype(int)}).to_csv(path, index=False)
        threshold_path = root / spec["threshold_artifact"]
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(json.dumps({"run_id": spec["run_id"], "selected_threshold": threshold}), encoding="utf-8")
    indices_path = root / config["bootstrap"]["test_indices"]
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(indices_path, indices=np.random.default_rng(5).integers(0, len(test), (1000, len(test))).astype(np.int32))
    synthetic = copy.deepcopy(config)
    synthetic["input_preflight"]["validation_rows"] = len(validation)
    synthetic["input_preflight"]["test_rows"] = len(test)
    return synthetic


def test_production_loader_on_synthetic_tree_feeds_the_runner(tmp_path, config, cohorts) -> None:
    project = tmp_path / "project"
    synthetic = _write_synthetic_project(project, config, cohorts)
    inputs = runner.load_production_inputs(synthetic, "RQ4a-6", project_root=project, rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)
    models, _ = cohorts
    assert tuple(inputs.models) == MODELS and inputs.bootstrap_indices.shape == (1000, len(models["xgboost"].test))
    for model in MODELS:
        assert inputs.models[model].threshold == float(config["baseline_models"][model]["frozen_threshold"])
        assert inputs.models[model].validation["derived_race"].tolist() == models["xgboost"].validation["derived_race"].tolist()
        assert inputs.models[model].test["applicant_sex_group"].tolist() == models["xgboost"].test["applicant_sex_group"].tolist()
    small = dataclasses.replace(inputs, bootstrap_indices=inputs.bootstrap_indices[:4])
    result = execute_run(small, RUN_IDS["RQ4a-6"], tmp_path / "out", settings=RunSettings(4, 4), project_root=project)
    assert result["status"] == "PASS"

    missing_prereq = copy.deepcopy(synthetic)
    missing_prereq["prerequisites"]["rq1_paired_bootstrap_task"] = "pending"
    with pytest.raises(RQ4ARunnerError, match="prerequisite_rq1_paired_bootstrap_missing"):
        runner.load_production_inputs(missing_prereq, "RQ4a-6", project_root=project, rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)
    with pytest.raises(RQ4ARunnerError, match="Unknown RQ4a configuration"):
        runner.load_production_inputs(synthetic, "RQ4a-9", project_root=project, rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)
    tampered = copy.deepcopy(synthetic)
    tampered["postprocessor"]["grid_size"] = 500
    with pytest.raises(RQ4AInputError):
        runner.load_production_inputs(tampered, "RQ4a-6", project_root=project, rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)
    view_path = project / config["audit_join"]["controlled_split_views"]["test"]
    view = pd.read_csv(view_path, dtype=str)
    view.loc[0, "lei"] = "MISMATCH"
    view.to_csv(view_path, index=False)
    with pytest.raises(RQ4AInputError, match="lei"):
        runner.load_production_inputs(synthetic, "RQ4a-6", project_root=project, rq3_protocol_path=DEFAULT_RQ3_PROTOCOL_PATH)
