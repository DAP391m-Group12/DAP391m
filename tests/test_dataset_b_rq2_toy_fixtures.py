from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/dataset_b/rq2/toy_fixtures.json"
PROTOCOL_PATH = PROJECT_ROOT / "config/dataset_b/rq2_protocol.yaml"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _reference_lz(values: np.ndarray, epsilon: float = 1e-12) -> float:
    z = np.asarray(values, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("non-finite prediction")
    mean_z = float(np.mean(z, dtype=np.float64))
    if mean_z <= epsilon:
        raise ValueError("prediction mean is at or below epsilon")
    ranks = _average_ranks(z)
    covariance = float(np.mean((z - mean_z) * (ranks - ranks.mean()), dtype=np.float64))
    return 2.0 * covariance / (len(z) * mean_z)


def _two_player_shapley(values: dict[str, float]) -> tuple[float, float]:
    phi_a = 0.5 * (values["A"] - values["empty"]) + 0.5 * (values["AB"] - values["B"])
    phi_b = 0.5 * (values["B"] - values["empty"]) + 0.5 * (values["AB"] - values["A"])
    return phi_a, phi_b


def test_lorenz_zonoid_numeric_fixture() -> None:
    case = _fixture()["lorenz_zonoid_numeric"]
    z = np.asarray(case["prediction_vector"], dtype=np.float64)
    ranks = _average_ranks(z)
    covariance = np.mean((z - z.mean()) * (ranks - ranks.mean()), dtype=np.float64)

    np.testing.assert_allclose(ranks, case["expected_ranks"], atol=1e-12, rtol=1e-10)
    assert z.mean() == pytest.approx(case["expected_prediction_mean"])
    assert ranks.mean() == pytest.approx(case["expected_rank_mean"])
    assert covariance == pytest.approx(case["expected_population_covariance"])
    assert _reference_lz(z) == pytest.approx(case["expected_lz"])

    tied = np.asarray(case["tied_prediction_vector"], dtype=np.float64)
    tied_ranks = _average_ranks(tied)
    tied_covariance = np.mean(
        (tied - tied.mean()) * (tied_ranks - tied_ranks.mean()), dtype=np.float64
    )
    np.testing.assert_allclose(tied_ranks, case["expected_tied_ranks"])
    assert tied.mean() == pytest.approx(case["expected_tied_prediction_mean"])
    assert tied_covariance == pytest.approx(case["expected_tied_population_covariance"])
    assert _reference_lz(tied) == pytest.approx(case["expected_tied_lz"])


def test_exact_shapley_combinatorics_fixture() -> None:
    case = _fixture()["exact_shapley_combinatorics"]
    phi_a, phi_b = _two_player_shapley(case["coalition_values"])
    assert phi_a == pytest.approx(case["expected_contributions"]["A"])
    assert phi_b == pytest.approx(case["expected_contributions"]["B"])
    assert phi_a + phi_b == pytest.approx(
        case["coalition_values"]["AB"] - case["coalition_values"]["empty"]
    )


def test_prediction_cube_gsv_sl_fixture() -> None:
    case = _fixture()["prediction_cube_gsv_sl"]
    cube = np.asarray(case["prediction_cube"], dtype=np.float64)
    empty, a, b, ab = cube

    phi_a = 0.5 * (a - empty) + 0.5 * (ab - b)
    phi_b = 0.5 * (b - empty) + 0.5 * (ab - a)
    np.testing.assert_allclose(phi_a, case["expected_gsv_local"]["A"])
    np.testing.assert_allclose(phi_b, case["expected_gsv_local"]["B"])
    np.testing.assert_allclose(phi_a + phi_b, ab - empty)

    importances = np.asarray([np.mean(np.abs(phi_a)), np.mean(np.abs(phi_b))])
    directions = np.asarray([np.mean(phi_a), np.mean(phi_b)])
    normalized = importances / importances.sum()
    np.testing.assert_allclose(importances, list(case["expected_gsv_importance"].values()))
    np.testing.assert_allclose(directions, list(case["expected_gsv_direction_mean"].values()))
    np.testing.assert_allclose(normalized, list(case["expected_gsv_normalized"].values()))

    coalition_values = {
        name: _reference_lz(cube[index])
        for index, name in enumerate(case["coalition_order"])
    }
    for name, expected in case["expected_sl_coalition_values"].items():
        assert coalition_values[name] == pytest.approx(expected)
    sl_a, sl_b = _two_player_shapley(coalition_values)
    assert sl_a == pytest.approx(case["expected_sl_signed_contribution"]["A"])
    assert sl_b == pytest.approx(case["expected_sl_signed_contribution"]["B"])
    assert sl_a + sl_b == pytest.approx(coalition_values["AB"] - coalition_values["empty"])
    sl_normalized = np.abs([sl_a, sl_b]) / np.sum(np.abs([sl_a, sl_b]))
    np.testing.assert_allclose(sl_normalized, list(case["expected_sl_normalized"].values()))


def test_ranking_and_edge_cases_fixture() -> None:
    case = _fixture()["ranking_and_edge_cases"]
    values = np.asarray(case["normalized_importances"], dtype=np.float64)
    assert values.sum() == pytest.approx(1.0)
    cutoff = np.sort(values)[::-1][case["top_k"] - 1]
    observed = list(np.flatnonzero(values >= cutoff - 1e-10) + 1)
    assert observed == case["expected_top_k_indices_one_based"]
    assert _reference_lz(np.asarray(case["constant_positive_vector"])) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="at or below epsilon"):
        _reference_lz(np.asarray(case["near_zero_mean_vector"]))
    with pytest.raises(ValueError, match="non-finite"):
        _reference_lz(np.asarray([0.1, np.nan, 0.2, 0.3]))


def test_shared_bootstrap_fixture_and_protocol_alignment() -> None:
    fixture = _fixture()
    case = fixture["shared_bootstrap"]
    indices = np.asarray(case["indices"], dtype=np.int64)
    assert indices.shape == (case["replicate_count"], case["observation_count"])
    assert indices.min() >= 0
    assert indices.max() < case["observation_count"]
    assert indices[1].tolist().count(3) == 2
    assert np.unique(indices[2]).tolist() == [2]

    row_ids = np.asarray(["r0", "r1", "r2", "r3"])
    donor_ids = np.asarray(["d0", "d1", "d2", "d3"])
    for replicate in indices:
        np.testing.assert_array_equal(donor_ids[replicate], np.asarray([donor_ids[i] for i in replicate]))
        assert len(row_ids[replicate]) == case["observation_count"]

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    uncertainty = protocol["uncertainty"]
    production = protocol["toy_fixtures"]["production_bootstrap_gate"]
    assert uncertainty["bootstrap_seed"] == case["production_seed"] == 4204
    assert uncertainty["shared_indices_across_methods"] is True
    assert uncertainty["shared_indices_across_models"] is True
    assert uncertainty["quantile_method"] == case["quantile_method"] == "linear"
    assert uncertainty["undefined_metrics"]["minimum_valid_replicates"] == 950
    assert production["expected_shape"] == case["production_shape"] == [1000, 256]
    assert production["one_artifact_for_all_six_models"] is True


def test_exact_methods_integration_matches_locked_fixture() -> None:
    from src.dataset_b.gsv_sl.exact_methods import (
        exact_shapley_values,
        gsv_from_prediction_cube,
        lorenz_zonoid_d1,
        shapley_lorenz_from_prediction_cube,
    )

    fixture = _fixture()
    lz_case = fixture["lorenz_zonoid_numeric"]
    assert lorenz_zonoid_d1(lz_case["prediction_vector"]) == pytest.approx(
        lz_case["expected_lz"]
    )

    combinatorics = fixture["exact_shapley_combinatorics"]
    values = combinatorics["coalition_values"]
    observed = exact_shapley_values(
        [values["empty"], values["A"], values["B"], values["AB"]],
        player_count=2,
    )
    np.testing.assert_allclose(observed, list(combinatorics["expected_contributions"].values()))

    cube_case = fixture["prediction_cube_gsv_sl"]
    cube = np.asarray(cube_case["prediction_cube"], dtype=np.float64)
    gsv = gsv_from_prediction_cube(cube, player_count=2)
    np.testing.assert_allclose(
        gsv["local_contributions"][:, 0], cube_case["expected_gsv_local"]["A"]
    )
    np.testing.assert_allclose(
        gsv["local_contributions"][:, 1], cube_case["expected_gsv_local"]["B"]
    )
    np.testing.assert_allclose(
        gsv["normalized_importance"], list(cube_case["expected_gsv_normalized"].values())
    )
    assert gsv["degenerate"] is False

    sl = shapley_lorenz_from_prediction_cube(cube, player_count=2)
    np.testing.assert_allclose(
        sl["coalition_values"], list(cube_case["expected_sl_coalition_values"].values())
    )
    np.testing.assert_allclose(
        sl["signed_lz_contribution"],
        list(cube_case["expected_sl_signed_contribution"].values()),
    )
    np.testing.assert_allclose(
        sl["normalized_importance"], list(cube_case["expected_sl_normalized"].values())
    )
    assert sl["degenerate"] is False


def test_exact_methods_production_edge_gates() -> None:
    from src.dataset_b.gsv_sl.exact_methods import (
        gsv_from_prediction_cube,
        lorenz_zonoid_d1,
        shapley_lorenz_from_prediction_cube,
    )

    constant_cube = np.full((4, 4), 0.2, dtype=np.float64)
    gsv = gsv_from_prediction_cube(constant_cube, player_count=2)
    sl = shapley_lorenz_from_prediction_cube(constant_cube, player_count=2)
    assert gsv["degenerate"] is True
    assert sl["degenerate"] is True
    assert np.isnan(gsv["normalized_importance"]).all()
    assert np.isnan(sl["normalized_importance"]).all()

    with pytest.raises(ValueError, match="at or below epsilon"):
        lorenz_zonoid_d1([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        lorenz_zonoid_d1([0.1, 0.2, np.nan, 0.4])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        gsv_from_prediction_cube(
            [[0.2, 0.2], [0.1, 0.2], [0.2, 0.3], [0.5, 1.1]],
            player_count=2,
        )


def test_ranking_agreement_and_cached_bootstrap_integration() -> None:
    from src.dataset_b.gsv_sl.exact_methods import (
        agreement_metrics,
        bootstrap_cached_agreement,
        generate_bootstrap_indices,
        gsv_from_prediction_cube,
        percentile_interval,
        rank_importance,
        top_k_with_ties,
    )

    fixture = _fixture()
    ranking = fixture["ranking_and_edge_cases"]
    importance = np.asarray(ranking["normalized_importances"], dtype=np.float64)
    np.testing.assert_allclose(rank_importance(importance), [1.0, 2.0, 3.5, 3.5, 5.0])
    np.testing.assert_array_equal(
        top_k_with_ties(importance, k=ranking["top_k"]) + 1,
        ranking["expected_top_k_indices_one_based"],
    )

    cube_case = fixture["prediction_cube_gsv_sl"]
    cube = np.asarray(cube_case["prediction_cube"], dtype=np.float64)
    gsv = gsv_from_prediction_cube(cube, player_count=2)
    agreement = agreement_metrics(
        list(cube_case["expected_gsv_normalized"].values()),
        list(cube_case["expected_sl_normalized"].values()),
        top_k=3,
    )
    assert agreement["spearman"] == pytest.approx(1.0)
    assert agreement["kendall_tau_b"] == pytest.approx(1.0)
    assert agreement["top_k_jaccard"] == pytest.approx(1.0)
    assert agreement["mean_absolute_rank_difference"] == pytest.approx(0.0)

    bootstrap_case = fixture["shared_bootstrap"]
    indices = np.asarray(bootstrap_case["indices"], dtype=np.int64)
    observed = bootstrap_cached_agreement(
        gsv["local_contributions"],
        cube,
        indices,
        player_count=2,
        minimum_valid_replicates=2,
    )
    assert observed["replicate_count"] == 3
    for distribution in observed["distributions"].values():
        assert distribution.shape == (3,)
    assert np.isnan(observed["distributions"]["spearman"][-1])
    spearman_ci = observed["confidence_intervals_95_percentile"]["spearman"]
    assert spearman_ci["lower"] == pytest.approx(1.0)
    assert spearman_ci["upper"] == pytest.approx(1.0)
    assert spearman_ci["valid_replicates"] == 2
    assert spearman_ci["invalid_replicates"] == 1
    assert spearman_ci["available"] is True
    for reason_counts in observed["invalid_reason_counts"].values():
        assert reason_counts == {"shapley_lorenz_degenerate_total_importance": 1}

    unavailable = percentile_interval([1.0, np.nan, np.nan], minimum_valid_replicates=2)
    assert unavailable["available"] is False
    assert np.isnan(unavailable["lower"])
    assert np.isnan(unavailable["upper"])

    generated_first = generate_bootstrap_indices(
        observation_count=4, replicate_count=3, seed=4204
    )
    generated_second = generate_bootstrap_indices(
        observation_count=4, replicate_count=3, seed=4204
    )
    np.testing.assert_array_equal(generated_first, generated_second)
