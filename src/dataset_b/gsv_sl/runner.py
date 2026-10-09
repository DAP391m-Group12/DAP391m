"""Minimal production runner for the frozen six-model Dataset B RQ2 analysis."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.dataset_b.gsv_sl.exact_methods import (
    agreement_metrics,
    bootstrap_cached_agreement,
    generate_bootstrap_indices,
    gsv_from_prediction_cube,
    percentile_interval,
    rank_importance,
    shapley_lorenz_from_prediction_cube,
)
from src.dataset_b.gsv_sl.predictors import (
    DEFAULT_ARTIFACT_ROOT,
    FEATURE_COLUMNS,
    FrozenPredictor,
    load_frozen_predictor,
)
from src.utils.path import PROJECT_ROOT

DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "config" / "dataset_b" / "rq2_protocol.yaml"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "dataset_b" / "split_v1"
BOOTSTRAP_METRICS = (
    "spearman",
    "kendall_tau_b",
    "top_k_jaccard",
    "mean_absolute_rank_difference",
)


class RQ2RunnerError(RuntimeError):
    """Raised when an RQ2 input or output gate fails."""


@dataclass(frozen=True)
class SamplingInputs:
    explanation_rows: pd.DataFrame
    explanation_X: pd.DataFrame
    paired_donor_X: pd.DataFrame
    sample_output: pd.DataFrame
    donors_output: pd.DataFrame
    background_count: int


@dataclass(frozen=True)
class CubeTiming:
    coalition_construction_seconds: float
    frozen_preprocessing_seconds: float
    frozen_inference_seconds: float


@dataclass(frozen=True)
class RQ2RunResult:
    analysis_run_id: str
    output_directory: Path
    validation_status: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ2RunnerError(message)


def _console_progress(message: str) -> None:
    print(message, flush=True)


def _emit_progress(
    callback: Callable[[str], None] | None, event: str, **fields: Any
) -> None:
    if callback is None:
        return
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    callback(f"[RQ2 {timestamp}] event={event}" + (f" {details}" if details else ""))


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_write_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RQ2RunnerError(f"Cannot read {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _csv_matches_expected(path: Path, expected: pd.DataFrame, *, label: str) -> None:
    try:
        observed = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise RQ2RunnerError(f"Cannot read {label}: {path}") from exc
    expected_text = expected.copy().astype(str)
    _require(
        tuple(observed.columns) == tuple(expected_text.columns)
        and observed.equals(expected_text),
        f"{label} differs from the deterministic RQ2 input.",
    )


def _result_csv_matches_expected(
    path: Path, expected: pd.DataFrame, *, label: str
) -> None:
    try:
        observed = pd.read_csv(path)
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except (OSError, pd.errors.ParserError, AssertionError) as exc:
        raise RQ2RunnerError(f"{label} differs from the cached RQ2 result: {path}") from exc


def _model_step(model_key: str, stage: str, action: Callable[[], Any]) -> Any:
    """Run one model-specific stage and preserve the failing model in the error."""

    try:
        return action()
    except Exception as exc:
        raise RQ2RunnerError(
            f"RQ2 model {model_key!r} failed during {stage}: {exc}"
        ) from exc


def _read_protocol(path: str | Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RQ2RunnerError("Cannot read the RQ2 protocol.") from exc
    _require(isinstance(payload, dict), "RQ2 protocol must be a YAML mapping.")
    _require(
        payload.get("status") == "production_complete",
        "RQ2 protocol status must be production_complete.",
    )
    return payload


def _validate_partition(frame: pd.DataFrame, *, partition: str) -> None:
    required = ["record_id", "y_approved", *FEATURE_COLUMNS]
    _require(all(column in frame.columns for column in required), f"{partition} lacks RQ2 columns.")
    positions = [frame.columns.get_loc(column) for column in FEATURE_COLUMNS]
    _require(
        positions == list(range(positions[0], positions[0] + len(FEATURE_COLUMNS))),
        f"{partition} does not contain the ordered final 15 input fields.",
    )
    ids = frame["record_id"].astype(str)
    _require(not ids.eq("").any(), f"{partition} contains a blank record_id.")
    _require(not ids.duplicated().any(), f"{partition} record_id values are not unique.")
    labels = frame["y_approved"].astype(str)
    _require(set(labels.unique()) == {"0", "1"}, f"{partition} y_approved must contain 0/1.")


def _deterministic_sample(
    frame: pd.DataFrame,
    *,
    partition: str,
    requested_rows: int,
    seed: int,
    strata: Sequence[int],
) -> tuple[pd.DataFrame, list[int]]:
    _validate_partition(frame, partition=partition)
    labels = frame["y_approved"].astype(str).to_numpy()
    stratum_values = tuple(str(value) for value in strata)
    population_counts = np.asarray([(labels == value).sum() for value in stratum_values])
    quotas = requested_rows * population_counts / len(frame)
    allocation = np.floor(quotas).astype(np.int64)
    remainder_order = sorted(
        range(len(stratum_values)),
        key=lambda index: (-(quotas[index] - allocation[index]), index),
    )
    for index in remainder_order[: requested_rows - int(allocation.sum())]:
        allocation[index] += 1
    _require(np.all(allocation <= population_counts), f"{partition} has an insufficient stratum.")
    rng = np.random.Generator(np.random.PCG64(seed))
    positions: list[int] = []
    for index, value in enumerate(stratum_values):
        candidates = np.flatnonzero(labels == value)
        positions.extend(rng.choice(candidates, size=int(allocation[index]), replace=False).tolist())
    positions.sort()
    selected = frame.iloc[positions].copy(deep=True).reset_index(drop=True)
    _require(len(selected) == requested_rows, f"{partition} sample row count differs.")
    return selected, positions


def prepare_sampling_inputs(
    protocol: Mapping[str, Any], *, split_dir: str | Path = DEFAULT_SPLIT_DIR
) -> SamplingInputs:
    """Create deterministic Validation observations and fixed Train donor pairs."""

    sampling = protocol["sampling"]
    split_root = Path(split_dir)

    def read_partition(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(
                split_root / f"{name}.csv", dtype=str, keep_default_na=False, na_filter=False
            )
        except (OSError, pd.errors.ParserError) as exc:
            raise RQ2RunnerError(f"Cannot read Dataset B {name} for RQ2.") from exc

    explanation_config = sampling["explanation"]
    background_config = sampling["background"]
    explanation, explanation_positions = _deterministic_sample(
        read_partition(explanation_config["partition"]),
        partition=explanation_config["partition"],
        requested_rows=int(explanation_config["n_rows"]),
        seed=int(explanation_config["seed"]),
        strata=sampling["strata"],
    )
    background, _ = _deterministic_sample(
        read_partition(background_config["partition"]),
        partition=background_config["partition"],
        requested_rows=int(background_config["pool_rows"]),
        seed=int(background_config["seed"]),
        strata=sampling["strata"],
    )
    assignment_seed = int(background_config["assignment_seed"])
    donor_indices = np.random.Generator(np.random.PCG64(assignment_seed)).integers(
        0, len(background), size=len(explanation), dtype=np.int64
    )
    paired = background.iloc[donor_indices].reset_index(drop=True)

    sample_output = explanation.loc[:, ["record_id", "y_approved", *FEATURE_COLUMNS]].copy()
    sample_output.insert(0, "sample_position", np.arange(len(sample_output), dtype=np.int64))
    sample_output.insert(1, "partition_position", explanation_positions)
    sample_output.insert(2, "partition", explanation_config["partition"])
    sample_output.insert(3, "seed", int(explanation_config["seed"]))
    donors_output = pd.DataFrame(
        {
            "explanation_position": np.arange(len(explanation), dtype=np.int64),
            "explanation_record_id": explanation["record_id"].astype(str),
            "donor_pool_index": donor_indices,
            "donor_record_id": paired["record_id"].astype(str),
            "background_partition": background_config["partition"],
            "background_rows": len(background),
            "background_seed": int(background_config["seed"]),
            "assignment_seed": assignment_seed,
        }
    )
    for column in FEATURE_COLUMNS:
        donors_output[f"donor_{column}"] = paired[column].to_numpy()
    return SamplingInputs(
        explanation,
        explanation.loc[:, FEATURE_COLUMNS].copy(deep=True),
        paired.loc[:, FEATURE_COLUMNS].copy(deep=True),
        sample_output,
        donors_output,
        len(background),
    )


def build_coalition_prediction_cube(
    predictor: FrozenPredictor,
    explanation_X: pd.DataFrame,
    paired_donor_X: pd.DataFrame,
    ordered_players: Sequence[Mapping[str, Any]],
    *,
    coalition_batch_size: int = 16,
) -> tuple[np.ndarray, CubeTiming]:
    """Construct and infer the canonical mask-ordered coalition cube in batches."""

    _require(isinstance(coalition_batch_size, int) and coalition_batch_size > 0, "Invalid batch size.")
    _require(tuple(explanation_X.columns) == FEATURE_COLUMNS, "Explanation schema differs.")
    _require(tuple(paired_donor_X.columns) == FEATURE_COLUMNS, "Donor schema differs.")
    _require(len(explanation_X) == len(paired_donor_X) > 0, "Explanation/donor rows differ.")
    field_positions = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
    player_positions = [
        [field_positions[field] for field in player["raw_fields"]] for player in ordered_players
    ]
    _require(
        [field for player in ordered_players for field in player["raw_fields"]]
        == list(FEATURE_COLUMNS),
        "Ordered players do not partition the final 15 fields.",
    )
    observation_count = len(explanation_X)
    coalition_count = 1 << len(ordered_players)
    cube = np.empty((coalition_count, observation_count), dtype=np.float64)
    explanation_values = explanation_X.to_numpy(dtype=object)
    donor_values = paired_donor_X.to_numpy(dtype=object)
    construction_seconds = preprocessing_seconds = inference_seconds = 0.0
    for first_mask in range(0, coalition_count, coalition_batch_size):
        masks = list(range(first_mask, min(first_mask + coalition_batch_size, coalition_count)))
        start = time.perf_counter_ns()
        hybrid = np.tile(donor_values, (len(masks), 1))
        for batch_index, mask in enumerate(masks):
            row_slice = slice(batch_index * observation_count, (batch_index + 1) * observation_count)
            for player_index, columns in enumerate(player_positions):
                if mask & (1 << player_index):
                    hybrid[row_slice, columns] = explanation_values[:, columns]
        hybrid_frame = pd.DataFrame(hybrid, columns=FEATURE_COLUMNS, dtype=object)
        construction_seconds += (time.perf_counter_ns() - start) / 1.0e9
        timed = predictor.predict_p_approved_timed(hybrid_frame)
        preprocessing_seconds += timed.preprocessing_seconds
        inference_seconds += timed.inference_seconds
        cube[masks, :] = timed.probabilities.reshape(len(masks), observation_count)
    _require(np.isfinite(cube).all(), "Coalition cube contains a non-finite probability.")
    _require(np.all((cube >= 0.0) & (cube <= 1.0)), "Coalition cube probability is out of range.")
    return cube, CubeTiming(construction_seconds, preprocessing_seconds, inference_seconds)


def _result_rows(
    *, analysis_run_id: str, model_key: str, model_run_id: str,
    ordered_players: Sequence[Mapping[str, Any]], method: str,
    importance: np.ndarray, normalized: np.ndarray, signed: np.ndarray,
    degenerate: bool, tie_atol: float,
) -> list[dict[str, Any]]:
    ranks = np.full(len(ordered_players), np.nan) if degenerate else rank_importance(normalized, tie_atol=tie_atol)
    return [
        {
            "run_id": analysis_run_id, "model_key": model_key, "model_run_id": model_run_id,
            "method": method, "player_index": index, "player_key": player["key"],
            "raw_fields": json.dumps(player["raw_fields"], separators=(",", ":")),
            "global_importance": float(importance[index]),
            "normalized_importance": float(normalized[index]), "rank": float(ranks[index]),
            "signed_summary": float(signed[index]),
            "result_status": "degenerate" if degenerate else "PASS",
        }
        for index, player in enumerate(ordered_players)
    ]


def _cross_model_rows(
    normalized_by_model: Mapping[str, Mapping[str, np.ndarray]], *,
    analysis_run_id: str, model_keys: Sequence[str], top_k: int, tie_atol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("gsv", "shapley_lorenz"):
        for left, right in combinations(model_keys, 2):
            metrics = agreement_metrics(
                normalized_by_model[left][method], normalized_by_model[right][method],
                top_k=top_k, tie_atol=tie_atol,
            )
            for metric in ("spearman", "kendall_tau_b"):
                rows.append({"run_id": analysis_run_id, "method": method,
                             "left_model_key": left, "right_model_key": right,
                             "metric": metric, "value": float(metrics[metric])})
    return rows


def _bootstrap_rows(
    *, run_id: str, model_key: str, model_run_id: str,
    point_estimate: Mapping[str, float], bootstrap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    intervals = bootstrap["confidence_intervals_95_percentile"]
    reasons = bootstrap["invalid_reason_counts"]
    rows = []
    for metric in ("spearman", "kendall_tau_b", "top_k_jaccard", "mean_absolute_rank_difference"):
        interval = intervals[metric]
        rows.append({
            "run_id": run_id, "model_key": model_key, "model_run_id": model_run_id,
            "metric": metric, "point_estimate": float(point_estimate[metric]),
            "ci_lower": float(interval["lower"]), "ci_upper": float(interval["upper"]),
            "ci_available": bool(interval["available"]),
            "valid_replicates": int(interval["valid_replicates"]),
            "invalid_replicates": int(interval["invalid_replicates"]),
            "invalid_reason_counts": json.dumps(reasons[metric], sort_keys=True),
        })
    return rows


def _checkpoint_identity(
    *, protocol: Mapping[str, Any], run_id: str, model_key: str,
    model_run_id: str, player_count: int, observation_count: int,
    chunk_size: int,
) -> dict[str, Any]:
    uncertainty = protocol["uncertainty"]
    return {
        "checkpoint_format": "rq2_bootstrap_chunk_v1",
        "protocol_version": protocol["protocol_version"],
        "run_id": run_id,
        "model_key": model_key,
        "model_run_id": model_run_id,
        "bootstrap_seed": int(uncertainty["bootstrap_seed"]),
        "bootstrap_replicates": int(uncertainty["bootstrap_replicates"]),
        "observation_count": observation_count,
        "player_count": player_count,
        "chunk_size": chunk_size,
    }


def _validate_checkpoint(
    *, npz_path: Path, metadata_path: Path, expected_identity: Mapping[str, Any],
    chunk_id: int, start: int, stop: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if npz_path.exists() != metadata_path.exists():
        raise RQ2RunnerError(
            f"Bootstrap checkpoint {chunk_id} is incomplete; both NPZ and JSON are required."
        )
    metadata = _read_json(metadata_path, label="bootstrap checkpoint metadata")
    expected = {
        **expected_identity,
        "chunk_id": chunk_id,
        "replicate_start": start,
        "replicate_stop": stop,
    }
    observed_identity = {key: metadata.get(key) for key in expected}
    _require(
        observed_identity == expected,
        f"Bootstrap checkpoint {chunk_id} identity differs from the current run.",
    )
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            _require(
                set(payload.files) == set(BOOTSTRAP_METRICS),
                f"Bootstrap checkpoint {chunk_id} metric keys differ.",
            )
            distributions = {
                metric: np.asarray(payload[metric])
                for metric in BOOTSTRAP_METRICS
            }
    except (OSError, ValueError) as exc:
        raise RQ2RunnerError(f"Cannot read bootstrap checkpoint {chunk_id}.") from exc

    invalid_reasons = metadata.get("invalid_reason_counts")
    valid_counts = metadata.get("valid_replicates")
    invalid_counts = metadata.get("invalid_replicates")
    _require(
        isinstance(invalid_reasons, dict)
        and isinstance(valid_counts, dict)
        and isinstance(invalid_counts, dict),
        f"Bootstrap checkpoint {chunk_id} count metadata is invalid.",
    )
    expected_rows = stop - start
    for metric, values in distributions.items():
        _require(
            values.shape == (expected_rows,)
            and values.dtype == np.float64
            and not np.isinf(values).any(),
            f"Bootstrap checkpoint {chunk_id} payload is invalid for {metric}.",
        )
        valid = int(np.isfinite(values).sum())
        invalid = int(np.isnan(values).sum())
        reasons = invalid_reasons.get(metric)
        _require(
            isinstance(reasons, dict)
            and all(isinstance(key, str) and isinstance(value, int) and value >= 0
                    for key, value in reasons.items())
            and sum(reasons.values()) == invalid
            and valid_counts.get(metric) == valid
            and invalid_counts.get(metric) == invalid
            and valid + invalid == expected_rows,
            f"Bootstrap checkpoint {chunk_id} counts differ for {metric}.",
        )
    return distributions, metadata


def _run_checkpointed_bootstrap(
    *, gsv_local_contributions: np.ndarray, coalition_prediction_cube: np.ndarray,
    bootstrap_indices: np.ndarray, checkpoint_directory: Path,
    checkpoint_identity: Mapping[str, Any], chunk_size: int, chunk_count: int,
    player_count: int, top_k: int, importance_epsilon: float,
    tie_atol: float, minimum_valid_replicates: int,
    progress_callback: Callable[[str], None] | None,
    chunk_complete_hook: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], float]:
    """Run or resume cache-only bootstrap chunks and merge them in index order."""

    replicate_count = int(bootstrap_indices.shape[0])
    _require(
        chunk_size > 0
        and chunk_count > 0
        and replicate_count == chunk_size * chunk_count,
        "Bootstrap checkpoint partition differs from the replicate matrix.",
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    expected_checkpoint_files = {
        f"chunk_{chunk_id:03d}.{suffix}"
        for chunk_id in range(chunk_count)
        for suffix in ("json", "npz")
    }
    unexpected_checkpoint_entries = sorted(
        entry.name
        for entry in checkpoint_directory.iterdir()
        if not entry.is_file() or entry.name not in expected_checkpoint_files
    )
    _require(
        not unexpected_checkpoint_entries,
        "Bootstrap checkpoint directory contains unexpected entries: "
        + ", ".join(unexpected_checkpoint_entries),
    )
    distributions_by_metric: dict[str, list[np.ndarray]] = {
        metric: [] for metric in BOOTSTRAP_METRICS
    }
    invalid_reason_counts = {metric: Counter() for metric in BOOTSTRAP_METRICS}
    total_seconds = 0.0

    for chunk_id in range(chunk_count):
        start = chunk_id * chunk_size
        stop = start + chunk_size
        npz_path = checkpoint_directory / f"chunk_{chunk_id:03d}.npz"
        metadata_path = checkpoint_directory / f"chunk_{chunk_id:03d}.json"
        if npz_path.exists() or metadata_path.exists():
            distributions, metadata = _validate_checkpoint(
                npz_path=npz_path,
                metadata_path=metadata_path,
                expected_identity=checkpoint_identity,
                chunk_id=chunk_id,
                start=start,
                stop=stop,
            )
            chunk_seconds = float(metadata.get("seconds", 0.0))
            _require(
                np.isfinite(chunk_seconds) and chunk_seconds >= 0.0,
                f"Bootstrap checkpoint {chunk_id} runtime is invalid.",
            )
            _emit_progress(
                progress_callback, "bootstrap_chunk_skipped",
                model=checkpoint_identity["model_key"], chunk=f"{chunk_id + 1}/{chunk_count}",
                replicates=f"{start}-{stop - 1}",
            )
        else:
            _emit_progress(
                progress_callback, "bootstrap_chunk_started",
                model=checkpoint_identity["model_key"], chunk=f"{chunk_id + 1}/{chunk_count}",
                replicates=f"{start}-{stop - 1}",
            )
            started = time.perf_counter_ns()
            result = bootstrap_cached_agreement(
                gsv_local_contributions,
                coalition_prediction_cube,
                bootstrap_indices[start:stop],
                player_count=player_count,
                top_k=top_k,
                importance_epsilon=importance_epsilon,
                tie_atol=tie_atol,
                minimum_valid_replicates=1,
            )
            chunk_seconds = (time.perf_counter_ns() - started) / 1.0e9
            distributions = {
                metric: np.asarray(result["distributions"][metric], dtype=np.float64)
                for metric in BOOTSTRAP_METRICS
            }
            metadata = {
                **checkpoint_identity,
                "chunk_id": chunk_id,
                "replicate_start": start,
                "replicate_stop": stop,
                "seconds": chunk_seconds,
                "valid_replicates": {
                    metric: int(np.isfinite(values).sum())
                    for metric, values in distributions.items()
                },
                "invalid_replicates": {
                    metric: int(np.isnan(values).sum())
                    for metric, values in distributions.items()
                },
                "invalid_reason_counts": result["invalid_reason_counts"],
            }
            _safe_write_npz(npz_path, distributions)
            _safe_write_json(metadata_path, metadata)
            _emit_progress(
                progress_callback, "bootstrap_chunk_passed",
                model=checkpoint_identity["model_key"], chunk=f"{chunk_id + 1}/{chunk_count}",
                seconds=f"{chunk_seconds:.3f}",
            )
            if chunk_complete_hook is not None:
                chunk_complete_hook(chunk_id)

        total_seconds += chunk_seconds
        for metric in BOOTSTRAP_METRICS:
            distributions_by_metric[metric].append(distributions[metric])
            invalid_reason_counts[metric].update(metadata["invalid_reason_counts"][metric])

    merged = {
        metric: np.concatenate(parts).astype(np.float64, copy=False)
        for metric, parts in distributions_by_metric.items()
    }
    _require(
        all(values.shape == (replicate_count,) for values in merged.values()),
        "Merged bootstrap distributions differ from the locked replicate count.",
    )
    intervals = {
        metric: percentile_interval(
            values, minimum_valid_replicates=minimum_valid_replicates
        )
        for metric, values in merged.items()
    }
    return {
        "distributions": merged,
        "confidence_intervals_95_percentile": intervals,
        "invalid_reason_counts": {
            metric: dict(sorted(counts.items()))
            for metric, counts in invalid_reason_counts.items()
        },
    }, total_seconds


def run_rq2(
    *, protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    frozen_artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    output_directory: str | Path | None = None, run_id: str | None = None,
    coalition_batch_size: int | None = None, resume: bool = False,
    predictor_loader: Callable[..., FrozenPredictor] = load_frozen_predictor,
    progress_callback: Callable[[str], None] | None = _console_progress,
    bootstrap_chunk_hook: Callable[[str, int], None] | None = None,
) -> RQ2RunResult:
    """Run or resume the locked six-model analysis with cache-only checkpoints."""

    protocol = _read_protocol(protocol_path)
    locked_batch_size = int(protocol["execution"]["coalition_batch_size"])
    if coalition_batch_size is None:
        coalition_batch_size = locked_batch_size
    _require(
        isinstance(coalition_batch_size, int)
        and not isinstance(coalition_batch_size, bool)
        and coalition_batch_size == locked_batch_size,
        f"RQ2 coalition batch size must equal the locked value {locked_batch_size}.",
    )
    run_id = run_id or (
        Path(output_directory).name if output_directory
        else datetime.now(UTC).strftime("rq2_%Y%m%dT%H%M%S%fZ")
    )
    _require(
        run_id not in {"", ".", ".."} and Path(run_id).name == run_id,
        "Invalid RQ2 run_id.",
    )
    destination = (
        Path(output_directory) if output_directory
        else PROJECT_ROOT / protocol["output"]["root_template"].format(run_id=run_id)
    )
    existed = destination.exists()
    _require(not existed or resume, f"RQ2 destination already exists: {destination}")
    if not existed:
        destination.mkdir(parents=True)
    _emit_progress(
        progress_callback, "run_started", run_id=run_id,
        mode="resume" if existed else "new", batch_size=coalition_batch_size,
    )

    model_keys = tuple(protocol["scope"]["model_keys"])
    players = tuple(protocol["scope"]["ordered_players"])
    sampling = prepare_sampling_inputs(protocol, split_dir=split_dir)
    uncertainty = protocol["uncertainty"]
    checkpoint = uncertainty["checkpoint"]
    chunk_size = int(checkpoint["chunk_size"])
    chunk_count = int(checkpoint["chunk_count"])
    _require(
        checkpoint.get("enabled") is True
        and checkpoint.get("resume_existing_run") is True
        and checkpoint.get("safe_file_replace") is True
        and checkpoint.get("retain_completed_chunks") is True,
        "RQ2 checkpoint execution contract differs.",
    )
    bootstrap_indices = generate_bootstrap_indices(
        observation_count=len(sampling.explanation_X),
        replicate_count=int(uncertainty["bootstrap_replicates"]),
        seed=int(uncertainty["bootstrap_seed"]),
    )
    expected_shape = tuple(
        protocol["toy_fixtures"]["production_bootstrap_gate"]["expected_shape"]
    )
    _require(
        expected_shape
        == (int(uncertainty["bootstrap_replicates"]), len(sampling.explanation_X))
        and bootstrap_indices.shape == expected_shape
        and bootstrap_indices.dtype == np.int64
        and bootstrap_indices.min() >= 0
        and bootstrap_indices.max() < len(sampling.explanation_X)
        and chunk_size * chunk_count == len(bootstrap_indices),
        "Production bootstrap contract differs.",
    )

    sample_path = destination / "sample.csv"
    donors_path = destination / "donors.csv"
    bootstrap_indices_path = destination / "bootstrap_indices.npy"
    if existed:
        _csv_matches_expected(sample_path, sampling.sample_output, label="RQ2 sample")
        _csv_matches_expected(donors_path, sampling.donors_output, label="RQ2 donors")
        try:
            stored_indices = np.load(bootstrap_indices_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RQ2RunnerError("Cannot read the stored bootstrap index matrix.") from exc
        _require(
            stored_indices.dtype == np.int64
            and np.array_equal(stored_indices, bootstrap_indices),
            "Stored bootstrap indices differ from the locked matrix.",
        )
        _emit_progress(progress_callback, "resume_inputs_validated", run_id=run_id)
    else:
        _safe_write_csv(sample_path, sampling.sample_output)
        _safe_write_csv(donors_path, sampling.donors_output)
        _safe_write_npy(bootstrap_indices_path, bootstrap_indices)

    numerical = protocol["numerical_policy"]
    ranking = protocol["ranking_and_agreement"]
    normalized_by_model: dict[str, dict[str, np.ndarray]] = {}
    runtime_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        model_run_id = protocol["eligible_final_runs"][model_key]
        model_dir = destination / "models" / model_key
        status_path = model_dir / "status.json"
        stage = "model initialization"
        status: dict[str, Any] = {
            "run_id": run_id,
            "model_key": model_key,
            "model_run_id": model_run_id,
            "protocol_version": protocol["protocol_version"],
        }
        try:
            if model_dir.exists():
                _require(resume, f"Unexpected existing model directory for {model_key}.")
                _require(status_path.is_file(), f"Model status is missing for {model_key}.")
                status = _read_json(status_path, label=f"{model_key} status")
                identity = {
                    "run_id": run_id,
                    "model_key": model_key,
                    "model_run_id": model_run_id,
                    "protocol_version": protocol["protocol_version"],
                }
                _require(
                    {key: status.get(key) for key in identity} == identity,
                    f"Model status identity differs for {model_key}.",
                )
            else:
                model_dir.mkdir(parents=True)
                status = {
                    "run_id": run_id,
                    "model_key": model_key,
                    "model_run_id": model_run_id,
                    "protocol_version": protocol["protocol_version"],
                    "status": "IN_PROGRESS",
                    "stage": stage,
                    "completed_chunks": 0,
                }
                _safe_write_json(status_path, status)
            was_pass = status.get("status") == "PASS"

            _emit_progress(
                progress_callback, "model_started", model=model_key,
                previous_status=status.get("status", "UNKNOWN"),
            )
            cube_path = model_dir / "cube.npy"
            local_path = model_dir / "gsv_local.npy"
            importance_path = model_dir / "importance.csv"
            bootstrap_path = model_dir / "bootstrap.csv"

            if cube_path.exists():
                stage = "cached cube validation"
                try:
                    cube = np.load(cube_path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise RQ2RunnerError(f"Cannot read cached cube for {model_key}.") from exc
                _require(
                    cube.shape == (1 << len(players), len(sampling.explanation_X))
                    and cube.dtype == np.float64
                    and np.isfinite(cube).all()
                    and np.all((cube >= 0.0) & (cube <= 1.0)),
                    f"Cached cube is invalid for {model_key}.",
                )
                cube_seconds = float(status.get("cube_seconds", 0.0))
                _require(np.isfinite(cube_seconds) and cube_seconds >= 0.0,
                         f"Cached cube runtime is invalid for {model_key}.")
                _emit_progress(progress_callback, "cube_cache_validated", model=model_key)
            else:
                _require(
                    not local_path.exists()
                    and not importance_path.exists()
                    and not (model_dir / "bootstrap_checkpoints").exists(),
                    f"Model cache is incomplete for {model_key}; refusing resume.",
                )
                stage = "frozen predictor load"
                _emit_progress(progress_callback, "predictor_load_started", model=model_key)
                predictor = _model_step(
                    model_key, stage,
                    lambda: predictor_loader(model_key, artifact_root=frozen_artifact_root),
                )
                _require(
                    predictor.execution_device == protocol["execution"]["device"] == "cpu",
                    f"RQ2 predictor device differs for {model_key}.",
                )
                _require(predictor.run_id == model_run_id,
                         f"Frozen run_id differs for {model_key}.")
                stage = "stored-prediction smoke comparison"
                _model_step(
                    model_key, stage,
                    lambda: predictor.smoke_compare_stored(
                        sampling.explanation_rows.head(int(protocol["smoke"]["rows"])),
                        atol=float(protocol["smoke"]["atol"]),
                    ),
                )
                _emit_progress(progress_callback, "smoke_passed", model=model_key)
                stage = "coalition cube construction and inference"
                _emit_progress(progress_callback, "cube_started", model=model_key)
                cube, cube_timing = _model_step(
                    model_key, stage,
                    lambda: build_coalition_prediction_cube(
                        predictor, sampling.explanation_X, sampling.paired_donor_X,
                        players, coalition_batch_size=coalition_batch_size,
                    ),
                )
                cube_seconds = sum((
                    cube_timing.coalition_construction_seconds,
                    cube_timing.frozen_preprocessing_seconds,
                    cube_timing.frozen_inference_seconds,
                ))
                _safe_write_npy(cube_path, cube)
                _emit_progress(
                    progress_callback, "cube_passed", model=model_key,
                    seconds=f"{cube_seconds:.3f}", shape=f"{cube.shape[0]}x{cube.shape[1]}",
                )

            stage = "exact GSV and Shapley-Lorenz"
            method_started = time.perf_counter_ns()
            gsv = gsv_from_prediction_cube(
                cube, player_count=len(players),
                importance_epsilon=float(numerical["importance_epsilon"]),
            )
            sl = shapley_lorenz_from_prediction_cube(
                cube, player_count=len(players),
                importance_epsilon=float(numerical["importance_epsilon"]),
            )
            calculated_method_seconds = (time.perf_counter_ns() - method_started) / 1.0e9
            expected_local = np.asarray(gsv["local_contributions"], dtype=np.float64)
            if local_path.exists():
                try:
                    stored_local = np.load(local_path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise RQ2RunnerError(f"Cannot read cached GSV local values for {model_key}.") from exc
                _require(
                    stored_local.dtype == np.float64
                    and stored_local.shape == expected_local.shape
                    and np.array_equal(stored_local, expected_local),
                    f"Cached GSV local values differ for {model_key}.",
                )
            else:
                _safe_write_npy(local_path, expected_local)

            normalized_by_model[model_key] = {
                "gsv": np.asarray(gsv["normalized_importance"], dtype=np.float64),
                "shapley_lorenz": np.asarray(sl["normalized_importance"], dtype=np.float64),
            }
            importance_rows = _result_rows(
                analysis_run_id=run_id, model_key=model_key, model_run_id=model_run_id,
                ordered_players=players, method="gsv",
                importance=np.asarray(gsv["global_importance"]),
                normalized=normalized_by_model[model_key]["gsv"],
                signed=np.asarray(gsv["prediction_direction_mean"]),
                degenerate=bool(gsv["degenerate"]),
                tie_atol=float(numerical["rank_tie_atol"]),
            )
            importance_rows.extend(_result_rows(
                analysis_run_id=run_id, model_key=model_key, model_run_id=model_run_id,
                ordered_players=players, method="shapley_lorenz",
                importance=np.asarray(sl["global_importance"]),
                normalized=normalized_by_model[model_key]["shapley_lorenz"],
                signed=np.asarray(sl["signed_lz_contribution"]),
                degenerate=bool(sl["degenerate"]),
                tie_atol=float(numerical["rank_tie_atol"]),
            ))
            expected_importance = pd.DataFrame(importance_rows)
            if importance_path.exists():
                _result_csv_matches_expected(
                    importance_path, expected_importance, label=f"{model_key} importance"
                )
            else:
                _safe_write_csv(importance_path, expected_importance)
            method_seconds = float(status.get("method_seconds", calculated_method_seconds))
            _require(np.isfinite(method_seconds) and method_seconds >= 0.0,
                     f"Cached method runtime is invalid for {model_key}.")
            if not was_pass:
                status.update({
                    "status": "IN_PROGRESS",
                    "stage": "bootstrap",
                    "cube_seconds": cube_seconds,
                    "method_seconds": method_seconds,
                })
                _safe_write_json(status_path, status)
            _emit_progress(
                progress_callback, "exact_methods_passed", model=model_key,
                seconds=f"{method_seconds:.3f}",
            )

            point = agreement_metrics(
                normalized_by_model[model_key]["gsv"],
                normalized_by_model[model_key]["shapley_lorenz"],
                top_k=int(ranking["top_k"]),
                tie_atol=float(numerical["rank_tie_atol"]),
            )
            checkpoint_identity = _checkpoint_identity(
                protocol=protocol,
                run_id=run_id,
                model_key=model_key,
                model_run_id=model_run_id,
                player_count=len(players),
                observation_count=len(sampling.explanation_X),
                chunk_size=chunk_size,
            )

            def record_chunk_complete(chunk_id: int) -> None:
                status.update({
                    "status": "IN_PROGRESS",
                    "stage": "bootstrap",
                    "completed_chunks": chunk_id + 1,
                })
                _safe_write_json(status_path, status)
                if bootstrap_chunk_hook is not None:
                    bootstrap_chunk_hook(model_key, chunk_id)

            stage = "checkpointed cached bootstrap"
            bootstrap, bootstrap_seconds = _run_checkpointed_bootstrap(
                gsv_local_contributions=expected_local,
                coalition_prediction_cube=cube,
                bootstrap_indices=bootstrap_indices,
                checkpoint_directory=model_dir / "bootstrap_checkpoints",
                checkpoint_identity=checkpoint_identity,
                chunk_size=chunk_size,
                chunk_count=chunk_count,
                player_count=len(players),
                top_k=int(ranking["top_k"]),
                importance_epsilon=float(numerical["importance_epsilon"]),
                tie_atol=float(numerical["rank_tie_atol"]),
                minimum_valid_replicates=int(
                    uncertainty["undefined_metrics"]["minimum_valid_replicates"]
                ),
                progress_callback=progress_callback,
                chunk_complete_hook=None if was_pass else record_chunk_complete,
            )
            bootstrap_rows = _bootstrap_rows(
                run_id=run_id, model_key=model_key, model_run_id=model_run_id,
                point_estimate=point, bootstrap=bootstrap,
            )
            _require(
                all(row["ci_available"] for row in bootstrap_rows),
                f"Bootstrap valid-replicate gate failed for {model_key}.",
            )
            expected_bootstrap = pd.DataFrame(bootstrap_rows)
            if bootstrap_path.exists():
                _result_csv_matches_expected(
                    bootstrap_path, expected_bootstrap, label=f"{model_key} bootstrap"
                )
            else:
                _safe_write_csv(bootstrap_path, expected_bootstrap)

            runtime = {
                "run_id": run_id,
                "model_key": model_key,
                "cube_seconds": cube_seconds,
                "method_seconds": method_seconds,
                "bootstrap_seconds": bootstrap_seconds,
                "total_seconds": cube_seconds + method_seconds + bootstrap_seconds,
            }
            runtime_rows.append(runtime)
            if not was_pass:
                status.update({
                    "status": "PASS",
                    "stage": "complete",
                    "completed_chunks": chunk_count,
                    **{key: value for key, value in runtime.items() if key.endswith("_seconds")},
                })
                _safe_write_json(status_path, status)
            _safe_write_csv(destination / "runtime.csv", pd.DataFrame(runtime_rows))
            _emit_progress(
                progress_callback,
                "model_skipped_pass" if was_pass else "model_passed",
                model=model_key, total_seconds=f"{runtime['total_seconds']:.3f}",
            )
        except Exception as exc:
            failed = {
                **status,
                "run_id": run_id,
                "model_key": model_key,
                "model_run_id": model_run_id,
                "protocol_version": protocol["protocol_version"],
                "status": "FAILED",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _safe_write_json(status_path, failed)
            _emit_progress(
                progress_callback, "model_failed", model=model_key,
                stage=stage, error=type(exc).__name__,
            )
            if isinstance(exc, RQ2RunnerError) and f"model {model_key!r}" in str(exc):
                raise
            raise RQ2RunnerError(
                f"RQ2 model {model_key!r} failed during {stage}: {exc}"
            ) from exc

    stage = "cross-model stability"
    _safe_write_csv(
        destination / "cross_model_stability.csv",
        pd.DataFrame(_cross_model_rows(
            normalized_by_model,
            analysis_run_id=run_id,
            model_keys=model_keys,
            top_k=int(ranking["top_k"]),
            tie_atol=float(numerical["rank_tie_atol"]),
        )),
    )
    _safe_write_csv(destination / "runtime.csv", pd.DataFrame(runtime_rows))
    _emit_progress(progress_callback, "run_passed", run_id=run_id, models=len(model_keys))
    return RQ2RunResult(run_id, destination, "PASS")
