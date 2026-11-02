"""Canonical Exact numerical methods for Dataset B RQ2.

This module operates only on cached coalition predictions. It does not load,
fit, mutate, or invoke a model.
"""

from __future__ import annotations

from math import factorial
from typing import Mapping, Sequence

import numpy as np


DEFAULT_IMPORTANCE_EPSILON = 1.0e-12
DEFAULT_EFFICIENCY_ATOL = 1.0e-12
DEFAULT_EFFICIENCY_RTOL = 1.0e-10
DEFAULT_RANK_TIE_ATOL = 1.0e-10


def _as_float64_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return ascending, one-based average ranks using stable sorting."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def lorenz_zonoid_d1(
    predictions: Sequence[float] | np.ndarray,
    *,
    epsilon: float = DEFAULT_IMPORTANCE_EPSILON,
) -> float:
    """Calculate the locked one-dimensional Lorenz--Zonoid coalition value."""

    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    values = _as_float64_vector(predictions, name="predictions")
    mean_prediction = float(np.mean(values, dtype=np.float64))
    if mean_prediction <= epsilon:
        raise ValueError("prediction mean is at or below epsilon")
    ranks = _average_ranks(values)
    mean_rank = float(np.mean(ranks, dtype=np.float64))
    covariance = float(
        np.mean((values - mean_prediction) * (ranks - mean_rank), dtype=np.float64)
    )
    return float(2.0 * covariance / (values.size * mean_prediction))


def _coalition_array(
    coalition_values: Mapping[int, float] | Sequence[float] | np.ndarray,
    *,
    player_count: int,
) -> np.ndarray:
    if not isinstance(player_count, int) or isinstance(player_count, bool) or player_count < 1:
        raise ValueError("player_count must be a positive integer")
    coalition_count = 1 << player_count
    if isinstance(coalition_values, Mapping):
        expected = set(range(coalition_count))
        observed = set(coalition_values)
        if observed != expected:
            raise ValueError("coalition mapping must contain every mask exactly once")
        values = np.asarray([coalition_values[mask] for mask in range(coalition_count)], dtype=np.float64)
    else:
        values = np.asarray(coalition_values, dtype=np.float64)
    if values.shape != (coalition_count,):
        raise ValueError(f"coalition_values must have shape ({coalition_count},)")
    if not np.isfinite(values).all():
        raise ValueError("coalition_values contains a non-finite value")
    return values


def exact_shapley_values(
    coalition_values: Mapping[int, float] | Sequence[float] | np.ndarray,
    *,
    player_count: int,
) -> np.ndarray:
    """Return exact Shapley contributions for coalition values ordered by bit mask."""

    values = _coalition_array(coalition_values, player_count=player_count)
    contributions = np.zeros(player_count, dtype=np.float64)
    denominator = factorial(player_count)
    full_mask = (1 << player_count) - 1
    for player in range(player_count):
        player_bit = 1 << player
        for mask in range(full_mask + 1):
            if mask & player_bit:
                continue
            subset_size = mask.bit_count()
            weight = (
                factorial(subset_size)
                * factorial(player_count - subset_size - 1)
                / denominator
            )
            contributions[player] += weight * (values[mask | player_bit] - values[mask])
    return contributions


def _prediction_cube(
    prediction_cube: Sequence[Sequence[float]] | np.ndarray,
    *,
    player_count: int,
) -> np.ndarray:
    cube = np.asarray(prediction_cube, dtype=np.float64)
    expected_coalitions = 1 << player_count
    if cube.ndim != 2 or cube.shape[0] != expected_coalitions or cube.shape[1] == 0:
        raise ValueError(
            f"prediction_cube must have shape ({expected_coalitions}, n_observations)"
        )
    if not np.isfinite(cube).all():
        raise ValueError("prediction_cube contains a non-finite value")
    if np.any((cube < 0.0) | (cube > 1.0)):
        raise ValueError("prediction_cube probabilities must be within [0, 1]")
    return cube


def _normalize_importance(importance: np.ndarray, *, epsilon: float) -> tuple[np.ndarray, bool]:
    total = float(np.sum(importance, dtype=np.float64))
    if total <= epsilon:
        return np.full(importance.shape, np.nan, dtype=np.float64), True
    return importance / total, False


def gsv_from_prediction_cube(
    prediction_cube: Sequence[Sequence[float]] | np.ndarray,
    *,
    player_count: int,
    importance_epsilon: float = DEFAULT_IMPORTANCE_EPSILON,
    efficiency_atol: float = DEFAULT_EFFICIENCY_ATOL,
    efficiency_rtol: float = DEFAULT_EFFICIENCY_RTOL,
) -> dict[str, np.ndarray | bool]:
    """Calculate exact local and global GSV summaries from a cached cube."""

    cube = _prediction_cube(prediction_cube, player_count=player_count)
    local = np.column_stack(
        [exact_shapley_values(cube[:, row], player_count=player_count) for row in range(cube.shape[1])]
    ).T
    expected_difference = cube[-1] - cube[0]
    if not np.allclose(
        np.sum(local, axis=1),
        expected_difference,
        atol=efficiency_atol,
        rtol=efficiency_rtol,
    ):
        raise ValueError("GSV efficiency gate failed")
    importance = np.mean(np.abs(local), axis=0, dtype=np.float64)
    normalized, degenerate = _normalize_importance(importance, epsilon=importance_epsilon)
    return {
        "local_contributions": local,
        "prediction_direction_mean": np.mean(local, axis=0, dtype=np.float64),
        "global_importance": importance,
        "normalized_importance": normalized,
        "degenerate": degenerate,
    }


def shapley_lorenz_from_prediction_cube(
    prediction_cube: Sequence[Sequence[float]] | np.ndarray,
    *,
    player_count: int,
    importance_epsilon: float = DEFAULT_IMPORTANCE_EPSILON,
    efficiency_atol: float = DEFAULT_EFFICIENCY_ATOL,
    efficiency_rtol: float = DEFAULT_EFFICIENCY_RTOL,
) -> dict[str, np.ndarray | bool]:
    """Calculate the locked interventional Shapley--Lorenz adaptation."""

    cube = _prediction_cube(prediction_cube, player_count=player_count)
    coalition_values = np.asarray(
        [lorenz_zonoid_d1(row, epsilon=importance_epsilon) for row in cube],
        dtype=np.float64,
    )
    signed = exact_shapley_values(coalition_values, player_count=player_count)
    if not np.isclose(
        np.sum(signed, dtype=np.float64),
        coalition_values[-1] - coalition_values[0],
        atol=efficiency_atol,
        rtol=efficiency_rtol,
    ):
        raise ValueError("Shapley--Lorenz efficiency gate failed")
    importance = np.abs(signed)
    normalized, degenerate = _normalize_importance(importance, epsilon=importance_epsilon)
    return {
        "coalition_values": coalition_values,
        "signed_lz_contribution": signed,
        "global_importance": importance,
        "normalized_importance": normalized,
        "degenerate": degenerate,
    }


def rank_importance(
    normalized_importance: Sequence[float] | np.ndarray,
    *,
    tie_atol: float = DEFAULT_RANK_TIE_ATOL,
) -> np.ndarray:
    """Rank normalized importance descending with average ranks for tolerance ties."""

    if not np.isfinite(tie_atol) or tie_atol < 0.0:
        raise ValueError("tie_atol must be finite and non-negative")
    values = np.asarray(normalized_importance, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("normalized_importance must be a non-empty one-dimensional vector")
    if np.isnan(values).all():
        return np.full(values.shape, np.nan, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("normalized_importance must be entirely finite or entirely NaN")

    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        anchor = values[order[start]]
        while end < values.size and abs(values[order[end]] - anchor) <= tie_atol:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def top_k_with_ties(
    normalized_importance: Sequence[float] | np.ndarray,
    *,
    k: int = 3,
    tie_atol: float = DEFAULT_RANK_TIE_ATOL,
) -> np.ndarray:
    """Return zero-based player indices in a descending Top-k set with cutoff ties."""

    values = np.asarray(normalized_importance, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("normalized_importance must be a non-empty one-dimensional vector")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if np.isnan(values).all():
        return np.asarray([], dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError("normalized_importance must be entirely finite or entirely NaN")
    cutoff_position = min(k, values.size) - 1
    cutoff = np.sort(values)[::-1][cutoff_position]
    selected = np.flatnonzero(values >= cutoff - tie_atol)
    return selected[np.argsort(-values[selected], kind="mergesort")].astype(np.int64)


def _spearman_from_ranks(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    concordant = discordant = left_only_ties = right_only_ties = 0
    for first in range(left.size - 1):
        for second in range(first + 1, left.size):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            left_tie = left_delta == 0.0
            right_tie = right_delta == 0.0
            if left_tie and right_tie:
                continue
            if left_tie:
                left_only_ties += 1
            elif right_tie:
                right_only_ties += 1
            elif left_delta * right_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + left_only_ties)
        * (concordant + discordant + right_only_ties)
    )
    if denominator == 0.0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def agreement_metrics(
    left_normalized_importance: Sequence[float] | np.ndarray,
    right_normalized_importance: Sequence[float] | np.ndarray,
    *,
    top_k: int = 3,
    tie_atol: float = DEFAULT_RANK_TIE_ATOL,
) -> dict[str, float | np.ndarray]:
    """Calculate the locked rank-agreement metrics for two methods."""

    left = np.asarray(left_normalized_importance, dtype=np.float64)
    right = np.asarray(right_normalized_importance, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("importance vectors must have the same shape")
    left_rank = rank_importance(left, tie_atol=tie_atol)
    right_rank = rank_importance(right, tie_atol=tie_atol)
    if np.isnan(left_rank).any() or np.isnan(right_rank).any():
        return {
            "left_rank": left_rank,
            "right_rank": right_rank,
            "spearman": float("nan"),
            "kendall_tau_b": float("nan"),
            "top_k_jaccard": float("nan"),
            "mean_absolute_rank_difference": float("nan"),
        }
    left_top = set(top_k_with_ties(left, k=top_k, tie_atol=tie_atol).tolist())
    right_top = set(top_k_with_ties(right, k=top_k, tie_atol=tie_atol).tolist())
    union = left_top | right_top
    return {
        "left_rank": left_rank,
        "right_rank": right_rank,
        "spearman": _spearman_from_ranks(left_rank, right_rank),
        "kendall_tau_b": _kendall_tau_b(left_rank, right_rank),
        "top_k_jaccard": float(len(left_top & right_top) / len(union)) if union else float("nan"),
        "mean_absolute_rank_difference": float(np.mean(np.abs(left_rank - right_rank))),
    }


def generate_bootstrap_indices(
    *,
    observation_count: int,
    replicate_count: int = 1000,
    seed: int = 4204,
) -> np.ndarray:
    """Generate ordinary observation-bootstrap indices once for shared reuse."""

    if not isinstance(observation_count, int) or isinstance(observation_count, bool) or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    if not isinstance(replicate_count, int) or isinstance(replicate_count, bool) or replicate_count < 1:
        raise ValueError("replicate_count must be a positive integer")
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        observation_count,
        size=(replicate_count, observation_count),
        dtype=np.int64,
    )


def percentile_interval(
    values: Sequence[float] | np.ndarray,
    *,
    minimum_valid_replicates: int,
) -> dict[str, float | int | bool]:
    """Return the locked linear 95% percentile interval and validity counts."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("bootstrap metric values must be one-dimensional")
    valid = samples[np.isfinite(samples)]
    valid_count = int(valid.size)
    invalid_count = int(samples.size - valid_count)
    available = valid_count >= minimum_valid_replicates
    if available:
        lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
    else:
        lower = upper = float("nan")
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": valid_count,
        "invalid_replicates": invalid_count,
        "available": available,
    }


def bootstrap_cached_agreement(
    gsv_local_contributions: Sequence[Sequence[float]] | np.ndarray,
    coalition_prediction_cube: Sequence[Sequence[float]] | np.ndarray,
    bootstrap_indices: Sequence[Sequence[int]] | np.ndarray,
    *,
    player_count: int,
    top_k: int = 3,
    importance_epsilon: float = DEFAULT_IMPORTANCE_EPSILON,
    tie_atol: float = DEFAULT_RANK_TIE_ATOL,
    minimum_valid_replicates: int = 950,
) -> dict[str, object]:
    """Bootstrap GSV--SL agreement from cached inputs without model inference."""

    local = np.asarray(gsv_local_contributions, dtype=np.float64)
    cube = _prediction_cube(coalition_prediction_cube, player_count=player_count)
    indices = np.asarray(bootstrap_indices)
    if local.shape != (cube.shape[1], player_count):
        raise ValueError("gsv_local_contributions must have shape (n_observations, player_count)")
    if not np.isfinite(local).all():
        raise ValueError("gsv_local_contributions contains a non-finite value")
    if indices.ndim != 2 or indices.shape[1] != cube.shape[1]:
        raise ValueError("bootstrap_indices must have shape (replicates, n_observations)")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("bootstrap_indices must contain integers")
    indices = indices.astype(np.int64, copy=False)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= cube.shape[1]:
        raise ValueError("bootstrap_indices contains an out-of-range index")

    metric_names = (
        "spearman",
        "kendall_tau_b",
        "top_k_jaccard",
        "mean_absolute_rank_difference",
    )
    distributions = {
        name: np.full(indices.shape[0], np.nan, dtype=np.float64) for name in metric_names
    }
    invalid_reason_counts = {name: {} for name in metric_names}

    def mark_invalid(metric: str, reason: str) -> None:
        counts = invalid_reason_counts[metric]
        counts[reason] = counts.get(reason, 0) + 1

    for replicate_id, replicate in enumerate(indices):
        sampled_local = local[replicate]
        gsv_importance = np.mean(np.abs(sampled_local), axis=0, dtype=np.float64)
        gsv_normalized, gsv_degenerate = _normalize_importance(
            gsv_importance, epsilon=importance_epsilon
        )
        sampled_cube = cube[:, replicate]
        sl = shapley_lorenz_from_prediction_cube(
            sampled_cube,
            player_count=player_count,
            importance_epsilon=importance_epsilon,
        )
        if gsv_degenerate:
            for name in metric_names:
                mark_invalid(name, "gsv_degenerate_total_importance")
            continue
        if bool(sl["degenerate"]):
            for name in metric_names:
                mark_invalid(name, "shapley_lorenz_degenerate_total_importance")
            continue
        metrics = agreement_metrics(
            gsv_normalized,
            sl["normalized_importance"],
            top_k=top_k,
            tie_atol=tie_atol,
        )
        for name in metric_names:
            value = float(metrics[name])
            distributions[name][replicate_id] = value
            if not np.isfinite(value):
                mark_invalid(name, "undefined_rank_metric_from_constant_ranks")

    intervals = {
        name: percentile_interval(values, minimum_valid_replicates=minimum_valid_replicates)
        for name, values in distributions.items()
    }
    return {
        "distributions": distributions,
        "confidence_intervals_95_percentile": intervals,
        "invalid_reason_counts": invalid_reason_counts,
        "replicate_count": int(indices.shape[0]),
    }
