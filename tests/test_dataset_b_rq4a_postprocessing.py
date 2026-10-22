from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.mitigation.inputs import LOCKED_G3, LOCKED_POSTPROCESSOR
from src.dataset_b.mitigation.postprocessing import (
    INTERPOLATION_FILENAME,
    POLICY_FILENAME,
    CachedProbabilityEstimator,
    RQ4APolicyError,
    apply_policy,
    assert_groups_fitted,
    baseline_decisions,
    decision_random_state,
    fit_policy,
    load_policy,
    save_policy,
)
from src.utils.path import PROJECT_ROOT


OUTSIDE = "Race Not Available"
THRESHOLD = 0.5


def _partition(rng: np.random.Generator, prefix: str, n_per_group: int) -> pd.DataFrame:
    shifts = {"White": 0.6, "Black or African American": -0.2, "Asian": 0.1, OUTSIDE: 0.0}
    frames = []
    for group, shift in shifts.items():
        latent = rng.normal(shift, 1.0, n_per_group)
        p = 1.0 / (1.0 + np.exp(-1.6 * latent))
        y = (rng.random(n_per_group) < 1.0 / (1.0 + np.exp(-(1.6 * latent + 0.3)))).astype(int)
        frames.append(pd.DataFrame({"group": group, "p_approved": p, "y_approved": y}))
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=11).reset_index(drop=True)
    df.insert(0, "record_id", [f"{prefix}-{i:06d}" for i in range(len(df))])
    return df


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(20261017)
    fit_df = _partition(rng, "val", 500)
    eval_df = _partition(rng, "test", 300)
    lookup = pd.concat([fit_df, eval_df]).set_index("record_id")["p_approved"]
    adapter = CachedProbabilityEstimator(lookup)
    policy = fit_policy(adapter, fit_df["record_id"], fit_df["y_approved"], fit_df["group"], LOCKED_G3)
    return fit_df, eval_df, adapter, policy


def test_adapter_returns_cached_columns_and_fails_closed() -> None:
    adapter = CachedProbabilityEstimator(pd.Series([0.2, 0.7], index=["a", "b"]))
    proba = adapter.predict_proba(pd.DataFrame({"record_id": ["b", "a"]}))
    assert np.array_equal(proba[:, 1], np.array([0.7, 0.2]))
    assert np.array_equal(proba[:, 0], 1.0 - np.array([0.7, 0.2]))
    for frame in (
        pd.DataFrame({"record_id": ["c"]}),
        pd.DataFrame({"record_id": ["a", "c", "a"]}),
        pd.DataFrame({"record_id": ["a", None]}),
        pd.DataFrame({"x": [1]}),
    ):
        with pytest.raises(RQ4APolicyError):
            adapter.predict_proba(frame)
    for series in (
        pd.Series([0.1, 0.2], index=["d", "d"]),
        pd.Series([0.1, 0.2], index=pd.Index([1, "1"], dtype=object)),
        pd.Series([0.1, 0.2], index=pd.Index(["a", None], dtype=object)),
        pd.Series([0.1, np.nan], index=["a", "b"]),
        pd.Series([0.1, 1.5], index=["a", "b"]),
    ):
        with pytest.raises(RQ4APolicyError):
            CachedProbabilityEstimator(series)


def test_no_model_loading_fit_or_inference(toy) -> None:
    _, _, adapter, policy = toy
    assert adapter.fit_calls == 0
    assert policy.estimator_ is adapter
    with pytest.raises(RQ4APolicyError):
        adapter.fit(None, None)
    for heavy in ("torch", "pytabkit", "xgboost"):
        assert heavy not in sys.modules


def test_fit_uses_roster_rows_only_and_locked_params(toy) -> None:
    fit_df, _, adapter, policy = toy
    assert sorted(policy.interpolated_thresholder_.interpolation_dict) == sorted(LOCKED_G3)
    assert {k: getattr(policy, k) for k in LOCKED_POSTPROCESSOR} == LOCKED_POSTPROCESSOR
    missing = fit_df[fit_df["group"] != "Asian"]
    with pytest.raises(RQ4APolicyError):
        fit_policy(adapter, missing["record_id"], missing["y_approved"], missing["group"], LOCKED_G3)
    with pytest.raises(RQ4APolicyError):
        fit_policy(adapter, fit_df["record_id"], fit_df["y_approved"], fit_df["group"], LOCKED_G3,
                   params={**LOCKED_POSTPROCESSOR, "flip": True})


def test_same_seed_reproducible_and_matches_fairlearn_semantics(toy) -> None:
    _, eval_df, _, policy = toy
    base = baseline_decisions(eval_df["p_approved"], THRESHOLD)
    args = (policy, eval_df["record_id"], eval_df["group"], base, LOCKED_G3)
    d42, mask = apply_policy(*args, seed=42)
    d42_again, _ = apply_policy(*args, seed=42)
    d43, _ = apply_policy(*args, seed=43)
    assert np.array_equal(d42, d42_again)
    assert not np.array_equal(d42, d43)
    roster_rows = eval_df[mask]
    pmf = policy._pmf_predict(roster_rows[["record_id"]], sensitive_features=roster_rows["group"])
    expected = (pmf[:, 1] >= np.random.RandomState(42).rand(len(pmf))).astype(np.int8)
    assert np.array_equal(d42[mask], expected)


def test_outside_roster_rows_keep_frozen_baseline(toy) -> None:
    _, eval_df, _, policy = toy
    base = baseline_decisions(eval_df["p_approved"], THRESHOLD)
    decisions, mask = apply_policy(policy, eval_df["record_id"], eval_df["group"], base, LOCKED_G3, seed=42)
    assert np.array_equal(mask, eval_df["group"].isin(LOCKED_G3).to_numpy())
    assert np.array_equal(decisions[~mask], base[~mask])
    assert len(decisions) == len(eval_df) and set(np.unique(decisions)) <= {0, 1}
    assert np.array_equal(base, (eval_df["p_approved"].to_numpy() >= THRESHOLD).astype(np.int8))


def test_unseen_roster_group_fails_instead_of_silent_denied(toy) -> None:
    _, eval_df, _, policy = toy
    base = baseline_decisions(eval_df["p_approved"], THRESHOLD)
    with pytest.raises(RQ4APolicyError):
        apply_policy(policy, eval_df["record_id"], eval_df["group"], base, (*LOCKED_G3, OUTSIDE), seed=42)
    with pytest.raises(RQ4APolicyError):
        assert_groups_fitted(policy, [OUTSIDE])


def test_decision_random_state_is_fresh_and_validated() -> None:
    assert np.array_equal(decision_random_state(42).rand(5), decision_random_state(42).rand(5))
    for bad in (-1, 2**32, 1.5, True):
        with pytest.raises(RQ4APolicyError):
            decision_random_state(bad)


def test_persistence_round_trip_and_no_overwrite(toy, tmp_path) -> None:
    _, eval_df, _, policy = toy
    hashes = save_policy(policy, tmp_path)
    assert set(hashes) == {POLICY_FILENAME, INTERPOLATION_FILENAME}
    summary = json.loads((tmp_path / INTERPOLATION_FILENAME).read_text(encoding="utf-8"))
    assert sorted(summary) == sorted(LOCKED_G3)
    reloaded = load_policy(tmp_path)
    assert type(reloaded.estimator_).__module__ == "src.dataset_b.mitigation.postprocessing"
    base = baseline_decisions(eval_df["p_approved"], THRESHOLD)
    before, mask = apply_policy(policy, eval_df["record_id"], eval_df["group"], base, LOCKED_G3, seed=42)
    after, _ = apply_policy(reloaded, eval_df["record_id"], eval_df["group"], base, LOCKED_G3, seed=42)
    assert np.array_equal(before, after)
    rows = eval_df[mask][["record_id"]]
    groups = eval_df[mask]["group"]
    assert np.array_equal(policy._pmf_predict(rows, sensitive_features=groups),
                          reloaded._pmf_predict(rows, sensitive_features=groups))
    with pytest.raises(RQ4APolicyError):
        save_policy(policy, tmp_path)


def test_fresh_interpreter_reload_without_main_shim(toy, tmp_path) -> None:
    _, eval_df, _, policy = toy
    save_policy(policy, tmp_path)
    rows = eval_df[eval_df["group"].isin(LOCKED_G3)].reset_index(drop=True)
    rows.to_csv(tmp_path / "rows.csv", index=False)
    expected = policy._pmf_predict(rows[["record_id"]], sensitive_features=rows["group"])
    np.save(tmp_path / "expected_pmf.npy", expected)
    code = (
        "import sys, __main__, numpy as np, pandas as pd; "
        "from src.dataset_b.mitigation.postprocessing import load_policy; "
        "p = load_policy(sys.argv[1]); "
        "df = pd.read_csv(sys.argv[1] + '/rows.csv', dtype={'record_id': str, 'group': str}); "
        "pmf = p._pmf_predict(df[['record_id']], sensitive_features=df['group']); "
        "ok = np.array_equal(pmf, np.load(sys.argv[1] + '/expected_pmf.npy')) "
        "and not hasattr(__main__, 'CachedProbabilityEstimator'); "
        "print('OK' if ok else 'MISMATCH')"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_save_rejects_main_defined_adapter(tmp_path, toy) -> None:
    fit_df, _, _, _ = toy

    class LocalAdapter(CachedProbabilityEstimator):
        pass

    LocalAdapter.__module__ = "__main__"
    adapter = LocalAdapter(fit_df.set_index("record_id")["p_approved"])
    policy = fit_policy(adapter, fit_df["record_id"], fit_df["y_approved"], fit_df["group"], LOCKED_G3)
    with pytest.raises(RQ4APolicyError):
        save_policy(policy, tmp_path)
    assert not Path(tmp_path / POLICY_FILENAME).exists()


def test_duplicate_query_returns_repeated_probabilities_in_order() -> None:
    adapter = CachedProbabilityEstimator(pd.Series([0.2, 0.7, 0.4], index=["a", "b", "c"]))
    proba = adapter.predict_proba(pd.DataFrame({"record_id": ["b", "a", "b", "b", "c", "a"]}))
    assert proba.shape == (6, 2)
    assert np.array_equal(proba[:, 1], np.array([0.7, 0.2, 0.7, 0.7, 0.4, 0.2]))
    assert np.array_equal(proba[:, 0], 1.0 - proba[:, 1])


def _stratified_resample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for _, stratum in df.groupby(["group", "y_approved"], sort=True):
        parts.append(stratum.iloc[rng.integers(0, len(stratum), len(stratum))])
    return pd.concat(parts, ignore_index=True)


def test_validation_resample_with_repeated_ids_fits(toy) -> None:
    fit_df, _, adapter, _ = toy
    resample = _stratified_resample(fit_df, np.random.default_rng(10001))
    assert resample["record_id"].duplicated().any()
    assert len(resample) == len(fit_df)
    fit_calls_before = adapter.fit_calls
    policy = fit_policy(adapter, resample["record_id"], resample["y_approved"], resample["group"], LOCKED_G3)
    assert sorted(policy.interpolated_thresholder_.interpolation_dict) == sorted(LOCKED_G3)
    assert adapter.fit_calls == fit_calls_before


def test_test_resample_with_repeated_ids_applies_independent_draws(toy) -> None:
    _, eval_df, _, policy = toy
    positions = np.random.default_rng(10002).integers(0, len(eval_df), len(eval_df)).astype(np.int32)
    rows = eval_df.iloc[positions].reset_index(drop=True)
    assert rows["record_id"].duplicated().any()
    base = baseline_decisions(rows["p_approved"], THRESHOLD)
    d1, mask = apply_policy(policy, rows["record_id"], rows["group"], base, LOCKED_G3, seed=12345)
    d2, _ = apply_policy(policy, rows["record_id"], rows["group"], base, LOCKED_G3, seed=12345)
    assert len(d1) == len(rows) and np.array_equal(d1, d2)
    assert np.array_equal(d1[~mask], base[~mask])
    roster_rows = rows[mask]
    pmf = policy._pmf_predict(roster_rows[["record_id"]], sensitive_features=roster_rows["group"])
    draws = np.random.RandomState(12345).rand(len(roster_rows))
    assert np.array_equal(d1[mask], (pmf[:, 1] >= draws).astype(np.int8))
    # Repeated occurrences share a positive probability but receive separate draws.
    roster_ids = roster_rows["record_id"].to_numpy()
    mitigated = d1[mask]
    differing = [rid for rid in np.unique(roster_ids[pd.Series(roster_ids).duplicated(keep=False).to_numpy()])
                 if len(set(mitigated[roster_ids == rid])) > 1]
    assert differing, "expected at least one repeated row with different independent draws"


def test_lookup_uniqueness_is_checked_after_string_normalization() -> None:
    with pytest.raises(RQ4APolicyError):
        CachedProbabilityEstimator(pd.Series([0.1, 0.2], index=pd.Index([1, "1"], dtype=object)))
    adapter = CachedProbabilityEstimator(pd.Series([0.1, 0.2], index=pd.Index([1, "2"], dtype=object)))
    proba = adapter.predict_proba(pd.DataFrame({"record_id": ["2", "1", "2"]}))
    assert np.array_equal(proba[:, 1], np.array([0.2, 0.1, 0.2]))
