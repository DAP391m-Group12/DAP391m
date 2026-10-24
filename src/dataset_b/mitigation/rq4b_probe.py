"""RQ4b independent capability probe: synthetic correctness gate and real-probe pipeline.

Every probe setting is read from the approved RQ4b contract snapshot
(``rq4b_inputs.APPROVED_CONTRACT``), which ``rq4b_inputs.load_contract``
compares with the YAML in full. The probe input is exactly
``[margin, y]``; the group is only the prediction target.

The real frozen-Validation probe may be prepared here but is only fitted through
``run_real_probe_attempt`` with ``real_probe_authorized=True`` under a separately
authorized task. Every fit runs once inside a single-use attempt directory that
retains either evidence or an immutable failure receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
import time
import traceback
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.dataset_b.mitigation import rq4b_inputs as ri
from src.utils.path import PROJECT_ROOT


_CAPABILITY = ri.APPROVED_CONTRACT["capability_assessment"]
REAL_PROBE = _CAPABILITY["real_probe"]
FITTING = REAL_PROBE["fitting"]
COMPARATOR = REAL_PROBE["y_only_comparator"]
SPLIT = REAL_PROBE["split"]
SYNTHETIC = _CAPABILITY["synthetic_correctness_gate"]
FIXTURE = SYNTHETIC["fixture_source_and_generation"]
UNCERTAINTY = _CAPABILITY["probe_uncertainty"]

DEVICE = FITTING["device"]
INPUT_DIMENSIONS = 2
HIDDEN_UNITS = 32
OUTPUT_CLASSES = len(ri.G3_GROUPS)
SOURCE_SYNTHETIC = "synthetic_positive_control"
SOURCE_REAL = "real_frozen_validation_g3"
SPLIT_NAMES = ("train", "validation", "holdout")
ATTEMPT_ROOT = PROJECT_ROOT / "scratch" / "rq4b_implementation"
HEARTBEAT_BATCH_INTERVAL = 25
HEARTBEAT_SECONDS = 30.0


class ProbeProgress:
    """Durable live probe log; every record is flushed before training continues."""

    def __init__(self, root: Path, kind: str) -> None:
        self.root, self.kind = Path(root), kind
        self.events: list[dict[str, Any]] = []
        self.last_position: dict[str, Any] | None = None
        self._last_heartbeat = 0.0
        self.last_progress_utc: str | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        def observe() -> None:
            while not self._stop.wait(HEARTBEAT_SECONDS):
                try:
                    with self._lock:
                        position = self.last_position or {"phase": "probe_prepare", "epoch": None, "batch": None}
                        self.emit("observer_heartbeat", phase=position["phase"], epoch=position["epoch"], batch=position["batch"],
                                  optimizer_progress_claimed=False, unavailable_reason="observer alive; no optimizer-progress claim")
                except BaseException as exc:
                    self._error = exc
                    try:
                        print(f"PROBE LOGGING FAILURE: {exc!r}", file=sys.stderr, flush=True)
                    except BaseException:
                        pass
                    return
        self._thread = threading.Thread(target=observe, name="rq4b-probe-observer", daemon=True)
        self._thread.start()

    def stop(self, *, check_error: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if check_error and self._thread.is_alive():
                raise RQ4bProbeError("Probe logging observer did not stop; refusing publication.")
        if check_error and self._error is not None:
            raise self._error

    def emit(self, event: str, *, phase: str, epoch: int | None = None,
             batch: int | None = None, **values: Any) -> None:
        with self._lock:
            self._emit(event, phase=phase, epoch=epoch, batch=batch, **values)

    def _emit(self, event: str, *, phase: str, epoch: int | None = None,
              batch: int | None = None, **values: Any) -> None:
        if self._error is not None:
            raise self._error
        timestamp = datetime.now(timezone.utc).isoformat()
        if values.get("optimizer_progress_claimed") is True:
            self.last_progress_utc = timestamp
        record = {
            "timestamp_utc": timestamp, "last_progress_utc": self.last_progress_utc,
            "run_id": None, "attempt_id": self.root.name, "unit": self.kind,
            "adversary_lambda": None, "lambda": None, "model_seed": int(FITTING["random_state"]),
            "seed": int(FITTING["random_state"]), "run_id_unavailable_reason": "probe attempts are not production run stages",
            "phase": phase, "fold": None, "epoch": epoch, "batch": batch,
            "event": event, **values,
        }
        line = json.dumps(_jsonable(record), sort_keys=True) + "\n"
        for name in ("run.log", "training_history.jsonl"):
            with (self.root / name).open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        print(line.rstrip(), file=sys.stderr, flush=True)
        self.events.append(record)
        self.last_position = {"phase": phase, "fold": None, "epoch": epoch, "batch": batch, "event": event}

    def heartbeat(self, epoch: int, batch: int) -> None:
        now = time.monotonic()
        if batch == 1 or batch % HEARTBEAT_BATCH_INTERVAL == 0 or now - self._last_heartbeat >= HEARTBEAT_SECONDS:
            self._last_heartbeat = now
            self.emit(
                "heartbeat", phase="probe_fit", epoch=epoch, batch=batch,
                train_loss=None, validation_ce=None, learning_rate=None,
                gpu_memory_bytes=int(torch.cuda.memory_allocated(0)) if torch.cuda.is_available() else None,
                unavailable_reason="batch-start observation; no completed optimizer-step claim",
                optimizer_progress_claimed=False,
            )

    def finalize(self) -> None:
        target = self.root / "training_history.json"
        write_json_once(target, {"attempt_id": self.root.name, "events": self.events})


_CURRENT_PROGRESS: ProbeProgress | None = None


class RQ4bProbeError(RuntimeError):
    """Fail-closed capability-probe error."""


class RQ4bProbeNonFiniteError(RQ4bProbeError):
    """A probe loss, gradient, input or evaluation value was not finite."""


class RQ4bProbeAuthorizationError(RQ4bProbeError):
    """The real frozen-Validation probe was requested without authorization."""


class RQ4bProbeAttemptError(RQ4bProbeError):
    """A probe attempt directory is missing, reused or cannot record failure."""


# ---------------------------------------------------------------------------
# Probe data, synthetic fixture and stratified split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeData:
    """Probe rows: inputs are margin and true Y; ``group`` is the target only."""

    record_ids: tuple[str, ...]
    margin: np.ndarray
    y: np.ndarray
    group: np.ndarray
    source: str

    def __post_init__(self) -> None:
        n = len(self.record_ids)
        if len(set(self.record_ids)) != n or n == 0:
            raise RQ4bProbeError("Probe record IDs must be nonempty and unique.")
        if not (len(self.margin) == len(self.y) == len(self.group) == n):
            raise RQ4bProbeError("Probe columns are misaligned.")
        if not np.isfinite(np.asarray(self.margin, dtype=np.float64)).all():
            raise RQ4bProbeNonFiniteError("Probe margins must be finite.")
        if not np.isin(self.y, [0, 1]).all() or not np.isin(self.group, list(range(OUTPUT_CLASSES))).all():
            raise RQ4bProbeError("Probe labels must be binary and groups must be G3 codes 0..2.")
        if self.source not in (SOURCE_SYNTHETIC, SOURCE_REAL):
            raise RQ4bProbeError(f"Unknown probe source {self.source!r}.")

    def subset(self, positions: np.ndarray) -> "ProbeData":
        ids = np.asarray(self.record_ids, dtype=object)[positions]
        return ProbeData(tuple(ids.tolist()), self.margin[positions], self.y[positions], self.group[positions], self.source)

    @property
    def strata(self) -> np.ndarray:
        return self.group.astype(np.int64) * 2 + self.y.astype(np.int64)


def probe_features(margin: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The only probe inputs: [margin, true Y]; group never enters."""
    return np.stack([np.asarray(margin, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=1).astype(np.float32)


def make_synthetic_fixture() -> ProbeData:
    """Locked positive control: 6,000 rows, 1,000 per (G3 group, Y).

    Rows are ordered by G3 group then Y (0 before 1); one vector of
    N(0, 0.25) noise is drawn from ``default_rng(42)`` in that row order and
    added to the group's center.
    """
    rows_per_cell = FIXTURE["rows_per_group_and_y"]
    centers = np.asarray(FIXTURE["centers_in_g3_order"], dtype=np.float64)
    group = np.repeat(np.arange(OUTPUT_CLASSES, dtype=np.int64), 2 * rows_per_cell)
    y = np.tile(np.repeat(np.array([0, 1], dtype=np.int64), rows_per_cell), OUTPUT_CLASSES)
    rng = np.random.default_rng(FIXTURE["numpy_seed"])
    noise = rng.normal(0.0, FIXTURE["noise_standard_deviation"], size=len(group))
    margin = centers[group] + noise
    ids = tuple(f"rq4b_synthetic_{index:06d}" for index in range(1, len(group) + 1))
    data = ProbeData(ids, margin, y, group, SOURCE_SYNTHETIC)
    if len(group) != FIXTURE["rows"]:
        raise RQ4bProbeError("Synthetic fixture row count differs from the contract.")
    return data


def split_probe(data: ProbeData) -> dict[str, np.ndarray]:
    """Stratified (group, Y) 60/20/20 split with random_state 42.

    Stage 1 holds out 40% (stratified); stage 2 halves it into validation and
    holdout (stratified), both with the locked random_state.
    """
    fractions = SPLIT["fractions"]
    seed = SPLIT["random_state"] if data.source == SOURCE_REAL else FIXTURE["split_seed"]
    positions = np.arange(len(data.record_ids))
    rest_fraction = fractions["validation"] + fractions["holdout"]
    train, rest = train_test_split(positions, test_size=rest_fraction, stratify=data.strata, random_state=seed)
    validation, holdout = train_test_split(
        rest, test_size=fractions["holdout"] / rest_fraction, stratify=data.strata[rest], random_state=seed
    )
    split = {"train": np.sort(train), "validation": np.sort(validation), "holdout": np.sort(holdout)}
    _check_split(split, len(positions))
    return split


def _check_split(split: Mapping[str, np.ndarray], rows: int) -> None:
    counts = np.zeros(rows, dtype=np.int64)
    for name in SPLIT_NAMES:
        counts[split[name]] += 1
    if not np.all(counts == 1):
        raise RQ4bProbeError("Probe split parts must be disjoint and cover every row exactly once.")


def cell_counts(data: ProbeData) -> dict[str, int]:
    return {
        f"group{group}_y{label}": int(((data.group == group) & (data.y == label)).sum())
        for group in range(OUTPUT_CLASSES)
        for label in (0, 1)
    }


# ---------------------------------------------------------------------------
# Y-only comparator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YOnlyComparator:
    counts: np.ndarray  # (2, 3): n[A=a, Y=y]
    probabilities: np.ndarray  # (2, 3): P(A=a | Y=y)
    alpha: float

    def row_ce(self, y: np.ndarray, group: np.ndarray) -> np.ndarray:
        return -np.log(self.probabilities[np.asarray(y, dtype=np.int64), np.asarray(group, dtype=np.int64)])


def fit_y_only(train: ProbeData) -> YOnlyComparator:
    """Laplace-smoothed P(A|Y) = (n[A=a,Y] + alpha) / (n[Y] + 3 alpha), probe-train only."""
    alpha = float(COMPARATOR["alpha"])
    counts = np.zeros((2, OUTPUT_CLASSES), dtype=np.int64)
    np.add.at(counts, (train.y.astype(np.int64), train.group.astype(np.int64)), 1)
    probabilities = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + OUTPUT_CLASSES * alpha)
    return YOnlyComparator(counts=counts, probabilities=probabilities, alpha=alpha)


# ---------------------------------------------------------------------------
# Fixed probe architecture, AdamW and checkpoint selection
# ---------------------------------------------------------------------------


def require_cuda0() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RQ4bProbeError("RQ4b probe requires cuda:0; CPU fallback is prohibited.")
    probe = torch.ones(2, device=DEVICE)
    if probe.device != torch.device(DEVICE) or not bool(torch.isfinite(probe).all().item()):
        raise RQ4bProbeError("RQ4b probe cuda:0 finite-operation preflight failed.")
    return {"device": str(probe.device), "device_name": torch.cuda.get_device_name(0)}


def build_probe(seed: int) -> torch.nn.Module:
    """Locked 2 -> 32 ReLU -> 3 probe, initialized under an isolated RNG fork."""
    with torch.random.fork_rng(devices=[torch.device(DEVICE).index]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIMENSIONS, HIDDEN_UNITS),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_UNITS, OUTPUT_CLASSES),
        )
    return model.to(DEVICE)


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=FITTING["learning_rate"],
        betas=tuple(FITTING["betas"]),
        eps=FITTING["epsilon"],
        weight_decay=FITTING["weight_decay"],
        amsgrad=FITTING["amsgrad"],
        maximize=FITTING["maximize"],
        foreach=FITTING["foreach"],
        capturable=FITTING["capturable"],
        differentiable=FITTING["differentiable"],
        fused=FITTING["fused"],
    )


class StoppingTracker:
    """min_delta 0: only strictly lower CE improves; ties keep the earliest checkpoint."""

    def __init__(self, *, patience: int = FITTING["patience"], maximum_epochs: int = FITTING["maximum_epochs"]) -> None:
        self.patience = patience
        self.maximum_epochs = maximum_epochs
        self.best_value = math.inf
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.epochs_without_improvement = 0
        self.last_epoch = 0
        self.stop_reason: str | None = None

    def update(self, epoch: int, value: float, state: Mapping[str, torch.Tensor] | None = None) -> bool:
        if epoch != self.last_epoch + 1 or self.stop_reason is not None:
            raise RQ4bProbeError("Stopping tracker received an out-of-order or post-stop epoch.")
        if not math.isfinite(value):
            raise RQ4bProbeNonFiniteError("Probe-validation CE is not finite.")
        self.last_epoch = epoch
        improved = value < self.best_value - FITTING["min_delta"]
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.best_state = None if state is None else {key: tensor.detach().clone() for key, tensor in state.items()}
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.stop_reason = "patience_15_consecutive_non_improving_epochs"
        elif epoch >= self.maximum_epochs:
            self.stop_reason = "maximum_epochs_100"
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stop_reason is not None


def _batch_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted mean group CE for one batch (factored for negative-path tests)."""
    return F.cross_entropy(logits, target)


def row_ce(model: torch.nn.Module, data: ProbeData) -> np.ndarray:
    """Per-row group CE in float64, eval mode and no gradients."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.as_tensor(probe_features(data.margin, data.y), device=DEVICE)
            logits = model(x).to(torch.float64)
            losses = -torch.log_softmax(logits, dim=1).gather(1, torch.as_tensor(data.group, device=DEVICE).view(-1, 1)).squeeze(1)
    finally:
        model.train(was_training)
    values = losses.cpu().numpy()
    if not np.isfinite(values).all():
        raise RQ4bProbeNonFiniteError("Probe evaluation produced nonfinite CE.")
    return values


@dataclass
class ProbeFit:
    model: torch.nn.Module
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_ce: float
    stopped_epoch: int
    stop_reason: str
    rng_isolation: dict[str, bool] = field(default_factory=dict)


def fit_probe(train: ProbeData, validation: ProbeData) -> ProbeFit:
    """AdamW probe fit on probe-train; probe-validation selects the restored checkpoint."""
    require_cuda0()
    seed = int(FITTING["random_state"])
    batch_size = int(FITTING["batch_size"])
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(0)
    model = build_probe(seed)
    optimizer = make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.as_tensor(probe_features(train.margin, train.y), device=DEVICE)
    target = torch.as_tensor(train.group, dtype=torch.long, device=DEVICE)
    tracker = StoppingTracker()
    history: list[dict[str, Any]] = []
    fit_started = time.perf_counter()
    for epoch in range(1, int(FITTING["maximum_epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train.record_ids), generator=generator).to(DEVICE)
        loss_sum, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            if _CURRENT_PROGRESS is not None:
                _CURRENT_PROGRESS.heartbeat(epoch, batches + 1)
            rows = order[start : start + batch_size]
            loss = _batch_loss(model(x[rows]), target[rows])
            if not bool(torch.isfinite(loss).item()):
                raise RQ4bProbeNonFiniteError(f"Probe training loss is not finite at epoch {epoch}; attempt stopped.")
            optimizer.zero_grad()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item()):
                raise RQ4bProbeNonFiniteError(f"Probe gradients are not finite at epoch {epoch}; attempt stopped.")
            optimizer.step()
            loss_sum += float(loss.detach().item())
            batches += 1
        if _CURRENT_PROGRESS is not None:
            _CURRENT_PROGRESS.emit("phase_started", phase="probe_validation", epoch=epoch, batch=batches, optimizer_progress_claimed=False)
        validation_ce = float(row_ce(model, validation).mean())
        improved = tracker.update(epoch, validation_ce, model.state_dict())
        now = time.perf_counter()
        entry = {
            "epoch": epoch, "mean_train_batch_loss": loss_sum / batches, "batches": batches,
            "validation_ce": validation_ce, "improved": improved,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": now - epoch_started, "elapsed_seconds": now - fit_started,
            "gpu_memory_bytes": int(torch.cuda.memory_allocated(0)) if torch.cuda.is_available() else None,
            "status": "completed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        if _CURRENT_PROGRESS is not None:
            _CURRENT_PROGRESS.emit(
                "epoch_completed", phase="probe_fit", epoch=epoch, batch=batches,
                train_loss=entry["mean_train_batch_loss"], validation_ce=validation_ce,
                learning_rate=entry["learning_rate"], epoch_seconds=entry["epoch_seconds"],
                elapsed_seconds=entry["elapsed_seconds"], gpu_memory_bytes=entry["gpu_memory_bytes"],
                status="completed", unavailable_reason=None, optimizer_progress_claimed=True,
            )
        if tracker.should_stop:
            break
    if _CURRENT_PROGRESS is not None:
        _CURRENT_PROGRESS.emit("phase_started", phase="probe_checkpoint_restore", optimizer_progress_claimed=False)
    model.load_state_dict(tracker.best_state)
    restored = float(row_ce(model, validation).mean())
    if restored != tracker.best_value:
        raise RQ4bProbeError("Restored checkpoint does not reproduce the best probe-validation CE.")
    isolation = {
        "cpu_rng_state_unchanged": torch.equal(cpu_before, torch.get_rng_state()),
        "cuda_rng_state_unchanged": torch.equal(cuda_before, torch.cuda.get_rng_state(0)),
    }
    return ProbeFit(model, history, tracker.best_epoch, tracker.best_value, tracker.last_epoch, tracker.stop_reason, isolation)


# ---------------------------------------------------------------------------
# CE gain, paired stratified bootstrap and gate
# ---------------------------------------------------------------------------


def ce_gain(y_only_ce: np.ndarray, probe_ce: np.ndarray) -> float:
    """CE_Y-only - CE_probe in nats on the same rows."""
    return float(np.mean(y_only_ce) - np.mean(probe_ce))


def paired_stratified_bootstrap(
    y_only_ce: np.ndarray,
    probe_ce: np.ndarray,
    strata: np.ndarray,
    *,
    replicates: int = UNCERTAINTY["replicates"],
    seed: int = UNCERTAINTY["random_state"],
) -> dict[str, Any]:
    """Same resampled holdout rows for both CEs; within-stratum resampling; fixed fitted models."""
    y_only_ce = np.asarray(y_only_ce, dtype=np.float64)
    probe_ce = np.asarray(probe_ce, dtype=np.float64)
    strata = np.asarray(strata)
    if not (len(y_only_ce) == len(probe_ce) == len(strata)):
        raise RQ4bProbeError("Bootstrap inputs are misaligned.")
    rng = np.random.default_rng(seed)
    levels = np.unique(strata)
    members = [np.flatnonzero(strata == level) for level in levels]
    gains, y_only_means, probe_means, invalid = [], [], [], []
    for replicate in range(replicates):
        rows = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in members])
        y_mean, p_mean = float(y_only_ce[rows].mean()), float(probe_ce[rows].mean())
        gain = y_mean - p_mean
        y_only_means.append(y_mean)
        probe_means.append(p_mean)
        gains.append(gain)
        if not (math.isfinite(y_mean) and math.isfinite(p_mean) and math.isfinite(gain)):
            invalid.append({"replicate": replicate, "reason": "nonfinite_ce"})
    return summarize_bootstrap(gains, invalid, y_only_means=y_only_means, probe_means=probe_means, strata_sizes={int(l): int(len(m)) for l, m in zip(levels, members)})


def summarize_bootstrap(gains: Sequence[float], invalid: Sequence[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    """Percentile 95% CI with linear quantiles only when all 1,000 replicates are valid."""
    gains_array = np.asarray(gains, dtype=np.float64)
    required = int(UNCERTAINTY["minimum_valid_replicates"])
    valid = int(len(gains_array) - len(invalid))
    summary: dict[str, Any] = {
        "replicates": int(len(gains_array)),
        "valid_replicates": valid,
        "required_valid_replicates": required,
        "invalid_replicates": list(invalid),
        "quantiles": list(UNCERTAINTY["quantiles"]),
        "quantile_method": UNCERTAINTY["quantile_interpolation"],
        "gains": gains_array.tolist(),
        **extra,
    }
    if valid != required or len(gains_array) != required:
        summary.update(ci_lower=None, ci_upper=None, ci_evaluable=False, ci_reason="ci_not_evaluable_no_replacement_replicates")
        return summary
    lower, upper = np.quantile(gains_array, UNCERTAINTY["quantiles"], method=UNCERTAINTY["quantile_interpolation"])
    summary.update(ci_lower=float(lower), ci_upper=float(upper), ci_evaluable=True, ci_reason=None)
    return summary


def synthetic_gate_decision(gain: float, bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    """PASS iff point gain >= 0.05 nats AND lower 95% CI > 0 (unrounded)."""
    minimum = float(SYNTHETIC["ce_gain_minimum_nats"])
    gain_ok = gain >= minimum
    lower = bootstrap.get("ci_lower")
    ci_ok = bool(bootstrap.get("ci_evaluable")) and lower is not None and lower > 0.0
    return {
        "status": "PASS" if gain_ok and ci_ok else "FAIL",
        "point_gain_nats": gain,
        "point_gain_minimum_nats": minimum,
        "point_gain_condition_met": gain_ok,
        "ci_lower": lower,
        "ci_lower_strictly_positive": ci_ok,
        "ci_evaluable": bool(bootstrap.get("ci_evaluable")),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_probe_experiment(
    data: ProbeData, *, real_probe_authorized: bool = False, attempt: "ProbeAttempt | None" = None
) -> dict[str, Any]:
    """Split, Y-only comparator, probe fit and holdout gain with paired bootstrap.

    A real-source fit additionally needs an open real-probe ``ProbeAttempt`` so
    that any failure is recorded; use ``run_real_probe_attempt``.
    """
    if data.source == SOURCE_REAL:
        if real_probe_authorized is not True:
            raise RQ4bProbeAuthorizationError("Fitting the real frozen-Validation probe requires separate authorization.")
        if (
            not isinstance(attempt, ProbeAttempt)
            or attempt.kind != ATTEMPT_REAL
            or not attempt.root.is_dir()
            or _ACTIVE_ATTEMPTS.get(attempt.root) is not attempt
        ):
            raise RQ4bProbeAttemptError("The real probe may only be fitted inside an executing real-probe attempt.")
    ri.load_contract()  # the approved snapshot must match the YAML before any fit
    if _CURRENT_PROGRESS is not None:
        _CURRENT_PROGRESS.emit("phase_started", phase="probe_local_split", optimizer_progress_claimed=False)
    split = split_probe(data)
    parts = {name: data.subset(split[name]) for name in SPLIT_NAMES}
    comparator = fit_y_only(parts["train"])
    fit = fit_probe(parts["train"], parts["validation"])
    holdout = parts["holdout"]
    if _CURRENT_PROGRESS is not None:
        _CURRENT_PROGRESS.emit("phase_started", phase="probe_holdout", optimizer_progress_claimed=False)
    probe_ce = row_ce(fit.model, holdout)
    y_only_ce = comparator.row_ce(holdout.y, holdout.group)
    gain = ce_gain(y_only_ce, probe_ce)
    if _CURRENT_PROGRESS is not None:
        _CURRENT_PROGRESS.emit("phase_started", phase="probe_bootstrap", optimizer_progress_claimed=False)
    bootstrap = paired_stratified_bootstrap(y_only_ce, probe_ce, holdout.strata)
    return {
        "source": data.source,
        "rows": len(data.record_ids),
        "record_id_sha256": hashlib.sha256("\n".join(data.record_ids).encode()).hexdigest(),
        "cell_counts": cell_counts(data),
        "split_cell_counts": {name: cell_counts(parts[name]) for name in SPLIT_NAMES},
        "split_record_id_sha256": {name: hashlib.sha256("\n".join(parts[name].record_ids).encode()).hexdigest() for name in SPLIT_NAMES},
        "comparator": {"counts_train_y_by_group": comparator.counts.tolist(), "probabilities_y_by_group": comparator.probabilities.tolist(), "alpha": comparator.alpha},
        "probe_fit": {
            "best_epoch": fit.best_epoch,
            "best_validation_ce": fit.best_validation_ce,
            "stopped_epoch": fit.stopped_epoch,
            "stop_reason": fit.stop_reason,
            "history": fit.history,
            "rng_isolation": fit.rng_isolation,
        },
        "holdout": {"rows": len(holdout.record_ids), "ce_y_only": float(y_only_ce.mean()), "ce_probe": float(probe_ce.mean()), "ce_gain_nats": gain},
        "bootstrap": bootstrap,
        "settings": {"fitting": FITTING, "split": SPLIT, "comparator": COMPARATOR, "uncertainty": UNCERTAINTY,
                     "architecture": {"inputs": ["margin", "true_y"], "hidden": HIDDEN_UNITS, "activation": "relu", "outputs": OUTPUT_CLASSES}},
    }


def prepare_real_probe_population(access: ri.StageAccess) -> ProbeData:
    """Frozen RealMLP Validation predictions -> reconstructed margins -> derived race -> G3 rows.

    Reads only the declared frozen predictions and the requested raw rows; no inference.
    """
    if access.stage != "capability_probe":
        raise ri.RQ4bAccessError(f"Real probe preparation may not run in stage {access.stage!r}.")
    predictions = ri.load_frozen_probe_predictions(access)
    permitted = ri.stage_population_ids(access, probe_predictions=predictions)
    ids = predictions["record_id"].tolist()
    raw_rows = ri.read_derived_race_rows(access, ids, permitted_ids=permitted)
    race = ri.join_derived_race(ids, predictions["y_approved"].tolist(), raw_rows, access.contract)
    population = ri.build_real_probe_population(predictions, race, access.contract)
    return ProbeData(
        tuple(population["record_id"].tolist()),
        population["margin"].to_numpy(dtype=np.float64),
        population["y_approved"].to_numpy(dtype=np.int64),
        population["group_code"].to_numpy(dtype=np.int64),
        SOURCE_REAL,
    )


# ---------------------------------------------------------------------------
# Synthetic gate attempt (immutable evidence)
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


PARTIAL_SUFFIX = ".partial"


def _write_payload(target: Path, handle: Any, payload: bytes) -> None:
    """Write and flush the complete payload to the temporary file (factored for failure tests)."""
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def write_json_once(path: Path, payload: Any) -> None:
    """Publish JSON atomically and never overwrite.

    The payload is fully serialized first, written and fsynced to a unique
    ``.<name>.<uuid>.partial`` file, made read-only, and only then renamed to
    ``path``; ``os.rename`` refuses an existing destination on Windows and the
    destination is also checked beforehand. A failure before the rename leaves
    at most a ``.partial`` file, never a file under the final name.
    """
    path = Path(path)
    data = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}{PARTIAL_SUFFIX}")
    with partial.open("xb") as handle:
        _write_payload(path, handle, data)
    os.chmod(partial, stat.S_IREAD)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    os.rename(partial, path)


def _record_failure(attempt: "ProbeAttempt", data: "ProbeData | None", exc: BaseException) -> None:
    """Write the failure receipt once; a receipt-writing error never replaces ``exc``.

    The secondary error is attached to the original exception as
    ``rq4b_receipt_error`` plus a note, and the caller re-raises the original.
    """
    exc.rq4b_failure_recorded = True
    try:
        write_json_once(attempt.root / RECEIPT_FILE, _failure_receipt(attempt, data, exc))
    except BaseException as receipt_exc:  # no retry
        exc.rq4b_receipt_error = receipt_exc
        exc.add_note(f"RQ4b failure receipt could not be written: {type(receipt_exc).__name__}: {receipt_exc}")


def create_attempt_root(name: str) -> Path:
    attempt = (ATTEMPT_ROOT / name).resolve()
    if attempt.parent != ATTEMPT_ROOT.resolve():
        raise RQ4bProbeError("Attempt directory must be a direct child of scratch/rq4b_implementation.")
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


ATTEMPT_SYNTHETIC = "synthetic_gate"
ATTEMPT_REAL = "real_probe"
INVOCATION_FILE = "invocation.json"
RECEIPT_FILE = "FAILURE_RECEIPT.json"
EVIDENCE_FILES = {ATTEMPT_SYNTHETIC: "synthetic_gate_evidence.json", ATTEMPT_REAL: "real_probe_evidence.json"}


@dataclass(frozen=True)
class ProbeAttempt:
    """One single-use attempt directory that can record evidence or a failure receipt."""

    root: Path
    kind: str


def open_attempt(root: Path, kind: str) -> ProbeAttempt:
    """Accept only a new, empty, direct child of scratch/rq4b_implementation (no reuse)."""
    if kind not in EVIDENCE_FILES:
        raise RQ4bProbeAttemptError(f"Unknown probe attempt kind {kind!r}.")
    resolved = Path(root).resolve()
    if resolved.parent != ATTEMPT_ROOT.resolve() or not resolved.is_dir():
        raise RQ4bProbeAttemptError("Probe attempt must be an existing direct child of scratch/rq4b_implementation.")
    if any(resolved.iterdir()):
        raise RQ4bProbeAttemptError(f"Probe attempt {resolved} is not empty; attempts are never reused or overwritten.")
    return ProbeAttempt(root=resolved, kind=kind)


def _failure_receipt(attempt: ProbeAttempt, data: ProbeData | None, exc: BaseException) -> dict[str, Any]:
    last_progress = None
    history_path = attempt.root / "training_history.jsonl"
    if history_path.is_file():
        lines = history_path.read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                last_progress = json.loads(lines[-1])
            except json.JSONDecodeError:
                last_progress = {"unreadable_last_line": lines[-1]}
    return {
        "attempt_kind": attempt.kind,
        "source": None if data is None else data.source,
        "rows": None if data is None else len(data.record_ids),
        "record_id_sha256": None if data is None else hashlib.sha256("\n".join(data.record_ids).encode()).hexdigest(),
        "category": "nonfinite_loss_or_gradient" if isinstance(exc, RQ4bProbeNonFiniteError) else "attempt_failure",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_chain": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "last_position": None if last_progress is None else {
            key: last_progress.get(key) for key in ("phase", "fold", "epoch", "batch", "event")
        },
        "last_progress_utc": None if last_progress is None else last_progress.get("last_progress_utc"),
        "contract_action": FITTING["nonfinite_loss_or_gradient"],
        "automatic_retry": False,
        "bad_batch_skipped": False,
        "success_evidence_written": False,
        # Unpublished temporary files (never valid evidence) left by an interrupted write.
        "partial_files": sorted(path.name for path in attempt.root.glob(f"*{PARTIAL_SUFFIX}")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# Attempts currently inside _execute_attempt; a real-source fit is accepted only
# for these, so a directly constructed ProbeAttempt cannot bypass receipt recording.
_ACTIVE_ATTEMPTS: dict[Path, ProbeAttempt] = {}


def _execute_attempt(attempt: ProbeAttempt, data: ProbeData, *, real_probe_authorized: bool, finalize=None,
                     progress: ProbeProgress | None = None) -> dict[str, Any]:
    """Run the experiment exactly once; on any failure write one immutable receipt and re-raise it."""
    if attempt.root in _ACTIVE_ATTEMPTS:
        raise RQ4bProbeAttemptError("Probe attempt is already executing.")
    _ACTIVE_ATTEMPTS[attempt.root] = attempt
    try:
        return _execute_registered_attempt(attempt, data, real_probe_authorized=real_probe_authorized, finalize=finalize, progress=progress)
    finally:
        _ACTIVE_ATTEMPTS.pop(attempt.root, None)


def _execute_registered_attempt(attempt: ProbeAttempt, data: ProbeData, *, real_probe_authorized: bool, finalize=None,
                                progress: ProbeProgress | None = None) -> dict[str, Any]:
    global _CURRENT_PROGRESS
    started = time.perf_counter()
    supplied = progress is not None
    progress = progress or ProbeProgress(attempt.root, attempt.kind)
    try:
        if not supplied:
            progress.emit("attempt_started", phase="probe_prepare", optimizer_progress_claimed=False)
            progress.start()
        _CURRENT_PROGRESS = progress
        write_json_once(
            attempt.root / INVOCATION_FILE,
            {"argv": sys.argv, "attempt_kind": attempt.kind, "source": data.source, "started_utc": datetime.now(timezone.utc).isoformat(), "environment": _environment()},
        )
        evidence = run_probe_experiment(data, real_probe_authorized=real_probe_authorized, attempt=attempt)
        progress.emit("phase_started", phase="probe_lineage_verification", optimizer_progress_claimed=False)
        if finalize is not None:
            finalize(evidence)
        evidence["elapsed_seconds"] = time.perf_counter() - started
        if (attempt.root / RECEIPT_FILE).exists():
            raise RQ4bProbeAttemptError("A failure receipt exists; success evidence is never written after failure.")
        progress.stop()
        progress.emit("computation_completed", phase="probe_finalize", optimizer_progress_claimed=False)
        progress.finalize()
        evidence["logging"] = {"schema_version": 1, "files": {
            name: file_sha256(attempt.root / name) for name in ("run.log", "training_history.jsonl", "training_history.json")}}
        write_json_once(attempt.root / EVIDENCE_FILES[attempt.kind], evidence)
        return evidence
    except BaseException as exc:
        progress.stop(check_error=False)
        # Evidence exists under its final name only after a complete atomic publish,
        # i.e. only if the attempt succeeded; otherwise the failure is receipted.
        if not (attempt.root / EVIDENCE_FILES[attempt.kind]).exists():
            _record_failure(attempt, data, exc)
        raise  # the original exception object; no retry
    finally:
        progress.stop(check_error=False)
        _CURRENT_PROGRESS = None


def run_synthetic_gate(attempt_root: Path) -> dict[str, Any]:
    """Single locked synthetic attempt; results (pass or fail) are retained immutably."""
    attempt = open_attempt(attempt_root, ATTEMPT_SYNTHETIC)
    try:
        data = make_synthetic_fixture()
    except BaseException as exc:
        _record_failure(attempt, None, exc)
        raise  # the original exception object; no retry

    def add_gate(evidence: dict[str, Any]) -> None:
        evidence["gate"] = synthetic_gate_decision(evidence["holdout"]["ce_gain_nats"], evidence["bootstrap"])

    return _execute_attempt(attempt, data, real_probe_authorized=False, finalize=add_gate)


def run_real_probe_attempt(data: ProbeData, attempt_root: Path, *, real_probe_authorized: bool = False) -> dict[str, Any]:
    """Authorized real frozen-Validation probe inside a single-use attempt directory.

    Authorization and the attempt boundary are checked before anything is written or fitted.
    """
    if real_probe_authorized is not True:
        raise RQ4bProbeAuthorizationError("Fitting the real frozen-Validation probe requires separate authorization.")
    if data.source != SOURCE_REAL:
        raise RQ4bProbeError("The real-probe attempt requires real frozen-Validation probe data.")
    attempt = open_attempt(attempt_root, ATTEMPT_REAL)
    return _execute_attempt(attempt, data, real_probe_authorized=True)


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def real_probe_lineage(access: ri.StageAccess, data: ProbeData) -> dict[str, Any]:
    """Bind the diagnostic to its frozen inputs, approved method and producing code."""
    if access.stage != "capability_probe" or data.source != SOURCE_REAL:
        raise RQ4bProbeError("Real-probe lineage requires the capability population.")
    resources = {
        name: {"path": str(access.require_resource(name).resolve()), "sha256": file_sha256(access.require_resource(name))}
        for name in ("frozen_probe_validation_predictions", "derived_race_source")
    }
    payload = {"record_ids": data.record_ids, "margin": data.margin.tolist(),
               "y": data.y.tolist(), "group": data.group.tolist()}
    return {
        "schema_version": 1,
        "protocol_version": access.contract.protocol_version,
        "decision_revision": access.contract.decision_revision,
        "frozen_run_id": REAL_PROBE["frozen_run_id"],
        "resources": resources,
        "code_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in ("rq4b_probe.py", "rq4b_inputs.py")},
        "population_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    }


def run_real_probe_from_frozen(attempt_root: Path, *, real_probe_authorized: bool = False) -> dict[str, Any]:
    """User-operated real route; preparation failures also retain an immutable receipt."""
    if real_probe_authorized is not True:
        raise RQ4bProbeAuthorizationError("Real probe requires --confirm-real-probe.")
    attempt = open_attempt(attempt_root, ATTEMPT_REAL)
    data = None
    progress = ProbeProgress(attempt.root, attempt.kind)
    try:
        progress.emit("attempt_started", phase="probe_prepare", optimizer_progress_claimed=False)
        progress.start()
        access = ri.StageAccess("capability_probe", ri.load_contract())
        data = prepare_real_probe_population(access)
        lineage = real_probe_lineage(access, data)
        def finalize(evidence: dict[str, Any]) -> None:
            if real_probe_lineage(access, data) != lineage:
                raise RQ4bProbeError("Real-probe inputs or code changed during fitting.")
            evidence["lineage"] = lineage
            evidence["invocation_sha256"] = file_sha256(attempt.root / INVOCATION_FILE)
        return _execute_attempt(attempt, data, real_probe_authorized=True, finalize=finalize, progress=progress)
    except BaseException as exc:
        progress.stop(check_error=False)
        if not getattr(exc, "rq4b_failure_recorded", False):
            _record_failure(attempt, data, exc)
        raise
    finally:
        progress.stop(check_error=False)


def validate_probe_logs(root: Path, evidence: Mapping[str, Any]) -> None:
    """Check the sealed logging payload and its correspondence with observed epochs."""
    names = ("run.log", "training_history.jsonl", "training_history.json")
    binding = evidence.get("logging", {})
    if binding.get("schema_version") != 1 or set(binding.get("files", {})) != set(names):
        raise RQ4bProbeError("Required probe logging manifest is missing or invalid.")
    blobs = {name: (root / name).read_bytes() for name in names}
    if any(hashlib.sha256(blobs[name]).hexdigest() != binding["files"][name] for name in names):
        raise RQ4bProbeError("Probe log integrity differs from evidence.")
    if blobs["run.log"] != blobs["training_history.jsonl"]:
        raise RQ4bProbeError("Probe text and JSONL logs differ.")
    records = [json.loads(line) for line in blobs["training_history.jsonl"].splitlines()]
    summary = json.loads(blobs["training_history.json"])
    if (not records or summary != {"attempt_id": root.name, "events": records}
            or records[0]["event"] != "attempt_started" or records[-1]["event"] != "computation_completed"):
        raise RQ4bProbeError("Probe log summary/start/computation boundary differs.")
    last_progress = None
    for row in records:
        if row.get("attempt_id") != root.name or row.get("unit") != ATTEMPT_REAL:
            raise RQ4bProbeError("Probe log attempt identity differs.")
        if row.get("optimizer_progress_claimed") is True:
            if row["event"] != "epoch_completed":
                raise RQ4bProbeError("Unexpected probe optimizer-progress event.")
            last_progress = row["timestamp_utc"]
        if row.get("last_progress_utc") != last_progress:
            raise RQ4bProbeError("Probe observation and completed-progress timestamps are inconsistent.")
    epochs = [row for row in records if row["event"] == "epoch_completed"]
    history = evidence["probe_fit"]["history"]
    if len(epochs) != len(history):
        raise RQ4bProbeError("Probe live epoch count differs from evidence history.")
    mapping = {"epoch": "epoch", "batch": "batches", "train_loss": "mean_train_batch_loss",
               "validation_ce": "validation_ce", "learning_rate": "learning_rate",
               "epoch_seconds": "epoch_seconds", "elapsed_seconds": "elapsed_seconds", "gpu_memory_bytes": "gpu_memory_bytes"}
    for log, row in zip(epochs, history):
        if any(key not in row or log.get(log_key) != row[key] for log_key, key in mapping.items()):
            raise RQ4bProbeError("Probe live epoch metrics differ from evidence history.")


def load_real_probe_evidence(path: Path, contract: ri.RQ4bContract) -> dict[str, Any]:
    """Validate saved evidence without fitting or inference; reconstruct frozen population only.

    Legacy in-memory/synthetic attempts without frozen lineage are not production
    evidence. Source hashes establish integrity/lineage, not independent proof of a fit.
    """
    path = Path(path).resolve()
    root = path.parent
    if root.parent != ATTEMPT_ROOT.resolve() or path.name != EVIDENCE_FILES[ATTEMPT_REAL]:
        raise RQ4bProbeError("Real evidence must be real_probe_evidence.json in a direct probe attempt.")
    if (root / RECEIPT_FILE).exists():
        raise RQ4bProbeError("Real-probe attempt has a failure receipt; success evidence is rejected.")
    if list(root.glob(f"*{PARTIAL_SUFFIX}")):
        raise RQ4bProbeError("Real-probe attempt contains unpublished partial files.")
    evidence_bytes = path.read_bytes()
    invocation_bytes = (root / INVOCATION_FILE).read_bytes()
    evidence = json.loads(evidence_bytes)
    validate_probe_logs(root, evidence)
    invocation = json.loads(invocation_bytes)
    if invocation.get("attempt_kind") != ATTEMPT_REAL or invocation.get("source") != SOURCE_REAL:
        raise RQ4bProbeError("Real-probe invocation source/kind differs.")
    if evidence.get("invocation_sha256") != hashlib.sha256(invocation_bytes).hexdigest():
        raise RQ4bProbeError("Real-probe invocation digest differs.")
    access = ri.StageAccess("capability_probe", contract)
    data = prepare_real_probe_population(access)
    if evidence.get("source") != SOURCE_REAL or evidence.get("lineage") != real_probe_lineage(access, data):
        raise RQ4bProbeError("Real-probe frozen source, population or code lineage differs.")
    split = split_probe(data)
    parts = {name: data.subset(split[name]) for name in SPLIT_NAMES}
    expected = {
        "rows": len(data.record_ids),
        "record_id_sha256": hashlib.sha256("\n".join(data.record_ids).encode()).hexdigest(),
        "cell_counts": cell_counts(data),
        "split_cell_counts": {name: cell_counts(part) for name, part in parts.items()},
        "split_record_id_sha256": {name: hashlib.sha256("\n".join(part.record_ids).encode()).hexdigest() for name, part in parts.items()},
        "settings": {"fitting": FITTING, "split": SPLIT, "comparator": COMPARATOR, "uncertainty": UNCERTAINTY,
                     "architecture": {"inputs": ["margin", "true_y"], "hidden": HIDDEN_UNITS, "activation": "relu", "outputs": OUTPUT_CLASSES}},
    }
    if any(evidence.get(key) != _jsonable(value) for key, value in expected.items()):
        raise RQ4bProbeError("Real-probe population, split or locked settings differ.")
    comparator = fit_y_only(parts["train"])
    expected_comparator = {"counts_train_y_by_group": comparator.counts.tolist(),
                           "probabilities_y_by_group": comparator.probabilities.tolist(), "alpha": comparator.alpha}
    if evidence.get("comparator") != expected_comparator:
        raise RQ4bProbeError("Real-probe Y-only comparator differs from probe training rows.")
    fit = evidence["probe_fit"]
    tracker = StoppingTracker()
    for row in fit["history"]:
        if not math.isfinite(row["mean_train_batch_loss"]) or row["batches"] != math.ceil(len(parts["train"].record_ids) / FITTING["batch_size"]):
            raise RQ4bProbeError("Real-probe training history is invalid.")
        if tracker.update(row["epoch"], row["validation_ce"]) != row["improved"]:
            raise RQ4bProbeError("Real-probe checkpoint history differs.")
    if (tracker.stop_reason is None or fit["best_epoch"] != tracker.best_epoch
            or fit["best_validation_ce"] != tracker.best_value or fit["stopped_epoch"] != tracker.last_epoch
            or fit["stop_reason"] != tracker.stop_reason
            or fit["rng_isolation"] != {"cpu_rng_state_unchanged": True, "cuda_rng_state_unchanged": True}):
        raise RQ4bProbeError("Real-probe stopping/checkpoint or RNG evidence differs.")
    holdout, boot = evidence["holdout"], evidence["bootstrap"]
    if holdout["rows"] != len(parts["holdout"].record_ids):
        raise RQ4bProbeError("Real-probe holdout rows differ.")
    if not all(math.isfinite(holdout[key]) for key in ("ce_y_only", "ce_probe", "ce_gain_nats")):
        raise RQ4bProbeError("Real-probe holdout quantities must be finite.")
    if holdout["ce_gain_nats"] != holdout["ce_y_only"] - holdout["ce_probe"]:
        raise RQ4bProbeError("Real-probe CE gain arithmetic differs.")
    expected_y_ce = float(comparator.row_ce(parts["holdout"].y, parts["holdout"].group).mean())
    if holdout["ce_y_only"] != expected_y_ce or holdout["ce_probe"] < 0:
        raise RQ4bProbeError("Real-probe holdout CE differs from the comparator or is negative.")
    gains = np.asarray(boot["gains"], dtype=np.float64)
    y_means, p_means = np.asarray(boot["y_only_means"], dtype=np.float64), np.asarray(boot["probe_means"], dtype=np.float64)
    if gains.shape != (UNCERTAINTY["replicates"],) or y_means.shape != gains.shape or p_means.shape != gains.shape:
        raise RQ4bProbeError("Real-probe bootstrap payload length differs.")
    with np.errstate(invalid="ignore"):
        expected_gains = y_means - p_means
    if not np.array_equal(gains, expected_gains, equal_nan=True):
        raise RQ4bProbeError("Real-probe bootstrap gain arithmetic differs.")
    invalid = [{"replicate": int(index), "reason": "nonfinite_ce"}
               for index in np.flatnonzero(~(np.isfinite(y_means) & np.isfinite(p_means) & np.isfinite(gains)))]
    summary = summarize_bootstrap(gains, invalid)
    if any(boot.get(key) != value for key, value in summary.items() if key != "gains"):
        raise RQ4bProbeError("Real-probe bootstrap counts, quantiles or interval differ.")
    levels, counts = np.unique(parts["holdout"].strata, return_counts=True)
    if boot.get("strata_sizes") != {str(level): int(count) for level, count in zip(levels, counts)}:
        raise RQ4bProbeError("Real-probe bootstrap strata differ from holdout.")
    return {"status": "completed", "capability_status": "capability_inconclusive",
            "interpretation": "real probe reported descriptively; no synthetic cutoff applied or automatic capability promotion",
            "evidence_path": str(path), "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "evidence": evidence, "invocation": invocation}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one locked synthetic gate or explicitly acknowledged real frozen-Validation probe.")
    parser.add_argument("--attempt-name", required=True)
    parser.add_argument("--kind", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--confirm-real-probe", action="store_true")
    args = parser.parse_args(argv)
    if (args.kind == "real") != args.confirm_real_probe:
        parser.error("--kind real requires --confirm-real-probe; synthetic mode must not use it.")
    attempt = create_attempt_root(args.attempt_name)
    print(f"RQ4B_PROBE_ATTEMPT {attempt}", flush=True)
    if args.kind == "real":
        evidence = run_real_probe_from_frozen(attempt, real_probe_authorized=True)
        print(json.dumps({"status": "completed", "evidence_path": str(attempt / EVIDENCE_FILES[ATTEMPT_REAL]),
                          "holdout": evidence["holdout"], "bootstrap_ci": [evidence["bootstrap"]["ci_lower"], evidence["bootstrap"]["ci_upper"]]}), flush=True)
        return 0
    evidence = run_synthetic_gate(attempt)
    gate, holdout, boot = evidence["gate"], evidence["holdout"], evidence["bootstrap"]
    print(json.dumps({"gate": gate, "holdout": holdout, "ci": [boot["ci_lower"], boot["ci_upper"]], "valid": boot["valid_replicates"],
                      "best_epoch": evidence["probe_fit"]["best_epoch"], "stop": evidence["probe_fit"]["stop_reason"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
