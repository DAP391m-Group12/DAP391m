"""PlainMLP CUDA smoke gate and Train-only inner-stopping engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from src.dataset_b.models.plain_mlp import (
    EXPECTED_INPUT_WIDTH,
    LOCKED_DEVICE,
    LOCKED_DTYPE,
    LOCKED_HIDDEN_LAYERS,
    PlainMLPClassifier,
    _PlainMLPModule,
)


INNER_STOP_FRACTION = 0.10
INNER_STOP_RANDOM_STATE = 42
INNER_STOP_PATIENCE = 10
INNER_STOP_MIN_DELTA = 1.0e-5
INNER_STOP_MAX_EPOCHS = 100
SMOKE_EPOCHS = 2
BATCH_FALLBACK_SEQUENCE = (256, 128, 64)


class PlainMLPEngineError(RuntimeError):
    """Raised when the PlainMLP runtime or selector contract fails."""


@dataclass(frozen=True)
class PlainMLPInnerSplit:
    fit_positions: tuple[int, ...]
    stop_positions: tuple[int, ...]
    fit_record_ids: tuple[str, ...]
    stop_record_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlainMLPEpochRecord:
    epoch: int
    inner_stop_bce_loss: float
    improved: bool
    best_epoch_so_far: int
    epochs_without_improvement: int


@dataclass(frozen=True)
class PlainMLPSelectionResult:
    best_epoch: int
    best_inner_stop_bce_loss: float
    stopped_after_epoch: int
    stop_reason: str
    history: tuple[PlainMLPEpochRecord, ...]
    split: PlainMLPInnerSplit
    temporary_state_published: bool = False


@dataclass(frozen=True)
class PlainMLPSmokeResult:
    selected_batch_size: int
    attempts: tuple[dict[str, Any], ...]
    epochs: int = SMOKE_EPOCHS
    device: str = LOCKED_DEVICE
    dtype: str = "float64"
    cpu_fallback_used: bool = False


def plain_mlp_cuda_preflight() -> dict[str, Any]:
    """Run a read-only CUDA/float64 capability probe; never fall back to CPU."""
    if torch.version.cuda is None:
        raise PlainMLPEngineError("PlainMLP requires a CUDA-enabled PyTorch build.")
    if not torch.cuda.is_available():
        raise PlainMLPEngineError("PlainMLP requires CUDA; CPU fallback is forbidden.")
    if torch.cuda.device_count() <= 0:
        raise PlainMLPEngineError("PlainMLP CUDA device index 0 is unavailable.")
    device = torch.device(LOCKED_DEVICE)
    try:
        properties = torch.cuda.get_device_properties(0)
        probe = torch.asarray([1.25, 2.75], device=device, dtype=LOCKED_DTYPE)
        result = (probe * probe).sum()
    except (RuntimeError, AssertionError) as exc:
        raise PlainMLPEngineError("PlainMLP CUDA float64 preflight failed.") from exc
    if (
        probe.device != device
        or result.device != device
        or probe.dtype != LOCKED_DTYPE
        or result.dtype != LOCKED_DTYPE
        or not bool(torch.isfinite(result).item())
    ):
        raise PlainMLPEngineError("PlainMLP CUDA float64 preflight failed.")
    return {
        "status": "PASS",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": LOCKED_DEVICE,
        "device_index": 0,
        "gpu_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "float64_probe": "PASS",
        "cpu_fallback_allowed": False,
    }


def run_plain_mlp_cuda_smoke(
    X: csr_matrix,
    y: object,
    *,
    attempt_runner: Callable[[int], dict[str, Any]] | None = None,
    preflight_runner: Callable[[], dict[str, Any]] | None = None,
) -> PlainMLPSmokeResult:
    """Run an in-memory two-epoch smoke with locked CUDA OOM fallback."""
    PlainMLPClassifier._validate_X(X)
    PlainMLPClassifier._validated_labels(y, X.shape[0])
    preflight = (preflight_runner or plain_mlp_cuda_preflight)()
    if preflight.get("status") != "PASS" or preflight.get("device") != LOCKED_DEVICE:
        raise PlainMLPEngineError("PlainMLP CUDA preflight did not return a locked PASS.")
    runner = attempt_runner or (lambda batch_size: _execute_smoke_attempt(X, y, batch_size))
    attempts: list[dict[str, Any]] = []
    for attempt_number, batch_size in enumerate(BATCH_FALLBACK_SEQUENCE, start=1):
        try:
            result = runner(batch_size)
            _validate_smoke_attempt_result(result, batch_size)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "batch_size": batch_size,
                    "status": "PASS",
                }
            )
            return PlainMLPSmokeResult(
                selected_batch_size=batch_size,
                attempts=tuple(attempts),
            )
        except Exception as exc:
            if not _is_confirmed_cuda_oom(exc):
                raise PlainMLPEngineError(
                    f"PlainMLP smoke failed without confirmed CUDA OOM at batch {batch_size}."
                ) from exc
            attempts.append(
                {
                    "attempt": attempt_number,
                    "batch_size": batch_size,
                    "status": "CUDA_OOM",
                }
            )
            torch.cuda.empty_cache()
            if batch_size == BATCH_FALLBACK_SEQUENCE[-1]:
                raise PlainMLPEngineError(
                    "PlainMLP smoke exhausted the locked CUDA OOM fallback sequence."
                ) from exc
    raise AssertionError("Unreachable PlainMLP smoke fallback state.")


def make_plain_mlp_inner_split(
    source_record_ids: Sequence[object],
    y: object,
    *,
    outer_validation_record_ids: Iterable[object] = (),
    project_validation_record_ids: Iterable[object] = (),
    project_test_record_ids: Iterable[object] = (),
) -> PlainMLPInnerSplit:
    """Create the locked stratified 90/10 split inside one fit population."""
    record_ids = tuple(str(value) for value in source_record_ids)
    if not record_ids or len(set(record_ids)) != len(record_ids) or any(not value for value in record_ids):
        raise ValueError("PlainMLP inner split requires unique nonblank source record IDs.")
    labels = PlainMLPClassifier._validated_labels(y, len(record_ids))
    forbidden_groups = {
        "outer fold-validation": {str(value) for value in outer_validation_record_ids},
        "project Validation": {str(value) for value in project_validation_record_ids},
        "project Test": {str(value) for value in project_test_record_ids},
    }
    source_set = set(record_ids)
    for label, forbidden in forbidden_groups.items():
        overlap = source_set & forbidden
        if overlap:
            raise ValueError(f"PlainMLP source fit population overlaps {label} IDs.")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_STOP_FRACTION,
        random_state=INNER_STOP_RANDOM_STATE,
    )
    try:
        fit_positions, stop_positions = next(
            splitter.split(np.zeros(len(record_ids), dtype=np.int8), labels)
        )
    except ValueError as exc:
        raise ValueError("PlainMLP fit population cannot support the locked stratified inner split.") from exc
    fit_positions = np.sort(np.asarray(fit_positions, dtype=np.int64))
    stop_positions = np.sort(np.asarray(stop_positions, dtype=np.int64))
    if np.intersect1d(fit_positions, stop_positions).size:
        raise PlainMLPEngineError("PlainMLP inner-fit and inner-stop positions overlap.")
    combined = np.concatenate((fit_positions, stop_positions))
    if not np.array_equal(np.sort(combined), np.arange(len(record_ids), dtype=np.int64)):
        raise PlainMLPEngineError("PlainMLP inner split does not cover the source fit population.")
    return PlainMLPInnerSplit(
        fit_positions=tuple(int(value) for value in fit_positions),
        stop_positions=tuple(int(value) for value in stop_positions),
        fit_record_ids=tuple(record_ids[position] for position in fit_positions),
        stop_record_ids=tuple(record_ids[position] for position in stop_positions),
        source_record_ids=record_ids,
    )


def run_plain_mlp_inner_selector(
    X: csr_matrix,
    y: object,
    record_ids: Sequence[object],
    *,
    batch_size: int,
    outer_validation_record_ids: Iterable[object] = (),
    project_validation_record_ids: Iterable[object] = (),
    project_test_record_ids: Iterable[object] = (),
    epoch_loss_stream_factory: Callable[
        [csr_matrix, np.ndarray, csr_matrix, np.ndarray, int], Iterable[float]
    ]
    | None = None,
) -> PlainMLPSelectionResult:
    """Select best epoch solely from Train-derived inner-stop BCE loss."""
    matrix = PlainMLPClassifier._validate_X(X)
    labels = PlainMLPClassifier._validated_labels(y, matrix.shape[0])
    split = make_plain_mlp_inner_split(
        record_ids,
        labels,
        outer_validation_record_ids=outer_validation_record_ids,
        project_validation_record_ids=project_validation_record_ids,
        project_test_record_ids=project_test_record_ids,
    )
    if batch_size not in BATCH_FALLBACK_SEQUENCE:
        raise ValueError("PlainMLP selector batch size must be one of 256, 128 or 64.")
    fit_positions = np.asarray(split.fit_positions, dtype=np.int64)
    stop_positions = np.asarray(split.stop_positions, dtype=np.int64)
    factory = epoch_loss_stream_factory or _run_selector_epochs
    stream = factory(
        matrix[fit_positions],
        labels[fit_positions],
        matrix[stop_positions],
        labels[stop_positions],
        batch_size,
    )
    try:
        best_epoch, best_loss, history, stop_reason = _select_best_epoch(stream)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return PlainMLPSelectionResult(
        best_epoch=best_epoch,
        best_inner_stop_bce_loss=best_loss,
        stopped_after_epoch=history[-1].epoch,
        stop_reason=stop_reason,
        history=history,
        split=split,
    )


def make_plain_mlp_final_refit_estimator(
    locked_best_epoch: int,
    *,
    batch_size: int,
) -> PlainMLPClassifier:
    """Create fresh final-estimator state using only the locked epoch count."""
    if (
        isinstance(locked_best_epoch, bool)
        or not isinstance(locked_best_epoch, int)
        or not 1 <= locked_best_epoch <= INNER_STOP_MAX_EPOCHS
    ):
        raise ValueError("PlainMLP locked best_epoch must be an integer in [1, 100].")
    if batch_size not in BATCH_FALLBACK_SEQUENCE:
        raise ValueError("PlainMLP final-refit batch size must be one of 256, 128 or 64.")
    return PlainMLPClassifier(batch_size=batch_size, epochs=locked_best_epoch)


def _select_best_epoch(
    losses: Iterable[float],
) -> tuple[int, float, tuple[PlainMLPEpochRecord, ...], str]:
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    records: list[PlainMLPEpochRecord] = []
    stop_reason = "max_epochs"
    for epoch, raw_loss in enumerate(losses, start=1):
        if epoch > INNER_STOP_MAX_EPOCHS:
            break
        loss = float(raw_loss)
        if not np.isfinite(loss) or loss < 0:
            raise PlainMLPEngineError("PlainMLP inner-stop BCE loss must be finite and non-negative.")
        improved = best_epoch == 0 or loss < best_loss - INNER_STOP_MIN_DELTA
        if improved:
            best_epoch = epoch
            best_loss = loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        records.append(
            PlainMLPEpochRecord(
                epoch=epoch,
                inner_stop_bce_loss=loss,
                improved=improved,
                best_epoch_so_far=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
        )
        if epochs_without_improvement >= INNER_STOP_PATIENCE:
            stop_reason = "patience"
            break
    if not records or best_epoch == 0:
        raise PlainMLPEngineError("PlainMLP selector produced no valid inner-stop epoch loss.")
    return best_epoch, best_loss, tuple(records), stop_reason


def _execute_smoke_attempt(X: csr_matrix, y: object, batch_size: int) -> dict[str, Any]:
    estimator = PlainMLPClassifier(batch_size=batch_size, epochs=SMOKE_EPOCHS)
    estimator.fit(X, y)
    if len(estimator.training_history_) != SMOKE_EPOCHS:
        raise PlainMLPEngineError("PlainMLP smoke did not complete exactly two epochs.")
    if estimator.last_dtype_audit_ != {
        "dense_batch": "float64",
        "target": "float64",
        "logits": "float64",
        "loss": "float64",
    }:
        raise PlainMLPEngineError("PlainMLP smoke dtype audit failed.")
    return {
        "epochs": SMOKE_EPOCHS,
        "batch_size": batch_size,
        "device": LOCKED_DEVICE,
        "dtype": "float64",
    }


def _validate_smoke_attempt_result(result: dict[str, Any], batch_size: int) -> None:
    expected = {
        "epochs": SMOKE_EPOCHS,
        "batch_size": batch_size,
        "device": LOCKED_DEVICE,
        "dtype": "float64",
    }
    if result != expected:
        raise PlainMLPEngineError("PlainMLP two-epoch smoke result violates the locked contract.")


def _run_selector_epochs(
    X_fit: csr_matrix,
    y_fit: np.ndarray,
    X_stop: csr_matrix,
    y_stop: np.ndarray,
    batch_size: int,
) -> Iterator[float]:
    plain_mlp_cuda_preflight()
    device = torch.device(LOCKED_DEVICE)
    seed_control = PlainMLPClassifier(batch_size=batch_size, epochs=INNER_STOP_MAX_EPOCHS)
    seed_control._seed_runtime()
    model = _PlainMLPModule(
        EXPECTED_INPUT_WIDTH,
        LOCKED_HIDDEN_LAYERS,
        float(seed_control.dropout),
    ).to(device=device, dtype=LOCKED_DTYPE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seed_control.learning_rate),
        weight_decay=float(seed_control.weight_decay),
    )
    pos_weight = PlainMLPClassifier._fit_population_pos_weight(y_fit)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device, dtype=LOCKED_DTYPE)
    )
    rng = np.random.default_rng(seed_control.random_state)
    try:
        for _ in range(INNER_STOP_MAX_EPOCHS):
            model.train()
            order = rng.permutation(X_fit.shape[0])
            for start in range(0, X_fit.shape[0], batch_size):
                positions = order[start : start + batch_size]
                batch_X = _dense_cuda_batch(X_fit[positions], device)
                batch_y = torch.from_numpy(y_fit[positions]).to(
                    device=device, dtype=LOCKED_DTYPE
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                if logits.dtype != LOCKED_DTYPE or loss.dtype != LOCKED_DTYPE:
                    raise PlainMLPEngineError("PlainMLP selector training dtype changed.")
                loss.backward()
                optimizer.step()

            model.eval()
            weighted_loss = 0.0
            rows_seen = 0
            with torch.no_grad():
                for start in range(0, X_stop.shape[0], batch_size):
                    batch = X_stop[start : start + batch_size]
                    batch_y_np = y_stop[start : start + batch_size]
                    batch_X = _dense_cuda_batch(batch, device)
                    batch_y = torch.from_numpy(batch_y_np).to(
                        device=device, dtype=LOCKED_DTYPE
                    )
                    loss = criterion(model(batch_X), batch_y)
                    if loss.dtype != LOCKED_DTYPE or not bool(torch.isfinite(loss).item()):
                        raise PlainMLPEngineError("PlainMLP inner-stop BCE loss is invalid.")
                    rows = len(batch_y_np)
                    weighted_loss += float(loss.cpu()) * rows
                    rows_seen += rows
            yield weighted_loss / rows_seen
    finally:
        del criterion, optimizer, model
        torch.cuda.empty_cache()


def _dense_cuda_batch(matrix: csr_matrix, device: torch.device) -> torch.Tensor:
    dense = matrix.toarray().astype(np.float64, copy=False)
    tensor = torch.from_numpy(dense).to(device=device, dtype=LOCKED_DTYPE)
    if tensor.device != device or tensor.dtype != LOCKED_DTYPE:
        raise PlainMLPEngineError("PlainMLP selector batch dtype/device changed.")
    return tensor


def _is_confirmed_cuda_oom(exc: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message
