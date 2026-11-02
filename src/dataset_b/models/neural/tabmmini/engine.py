"""Canonical CUDA gate, smoke fallback, and pilot epoch decision for TabMmini."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from src.dataset_b.models.neural.tabmmini.estimator import (
    TABM_VERSION,
    TabMMiniModel,
    compute_pos_weight,
    mean_member_probabilities,
    member_wise_weighted_bce,
)
from src.dataset_b.models.neural.tabmmini.preprocessing import (
    TabMMiniNativeBatch,
    tabmmini_category_cardinalities,
)


LOCKED_DEVICE = "cuda:0"
SMOKE_EPOCHS = 2
BATCH_FALLBACK_SEQUENCE = (256, 128, 64)
PILOT_MAX_EPOCHS = 100
PILOT_TIE_ATOL = 1.0e-6


class TabMMiniRuntimeError(RuntimeError):
    """Raised when a locked runtime or pilot-decision gate fails."""


@dataclass(frozen=True)
class TabMMiniBatchLock:
    effective_batch_size: int
    attempts: tuple[dict[str, Any], ...]
    device: str = LOCKED_DEVICE
    smoke_epochs: int = SMOKE_EPOCHS
    cpu_fallback_used: bool = False

    def __post_init__(self) -> None:
        if self.effective_batch_size not in BATCH_FALLBACK_SEQUENCE:
            raise ValueError("TabMmini effective batch size must be 256, 128, or 64.")
        if self.device != LOCKED_DEVICE or self.cpu_fallback_used:
            raise ValueError("TabMmini batch lock must remain CUDA-only on cuda:0.")
        selected_index = BATCH_FALLBACK_SEQUENCE.index(self.effective_batch_size)
        if len(self.attempts) != selected_index + 1:
            raise ValueError("TabMmini batch lock attempts must cover the exact fallback prefix.")
        for index, attempt in enumerate(self.attempts):
            expected_status = "PASS" if index == selected_index else "CUDA_OOM"
            if (
                attempt.get("batch_size") != BATCH_FALLBACK_SEQUENCE[index]
                or attempt.get("status") != expected_status
            ):
                raise ValueError("TabMmini batch lock attempt evidence is invalid.")


@dataclass(frozen=True)
class TabMMiniEpochRecord:
    epoch: int
    inner_stop_weighted_bce_loss: float
    tied_with_best: bool
    best_epoch_so_far: int


@dataclass(frozen=True)
class TabMMiniEpochDecision:
    status: str
    fixed_epoch: int | None
    best_loss: float
    history: tuple[TabMMiniEpochRecord, ...]
    reason: str


def tabmmini_cuda_preflight() -> dict[str, Any]:
    """Verify pinned TabM and CUDA device 0 without allowing CPU fallback."""
    try:
        installed_tabm = version("tabm")
    except PackageNotFoundError as exc:
        raise TabMMiniRuntimeError(f"TabMmini requires tabm=={TABM_VERSION}.") from exc
    if installed_tabm != TABM_VERSION:
        raise TabMMiniRuntimeError(
            f"TabMmini requires tabm=={TABM_VERSION}; found {installed_tabm}."
        )
    try:
        from tabm import TabM
    except ImportError as exc:
        raise TabMMiniRuntimeError("Cannot import tabm.TabM.") from exc
    if not callable(getattr(TabM, "make", None)):
        raise TabMMiniRuntimeError("tabm.TabM.make is unavailable.")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise TabMMiniRuntimeError("TabMmini requires CUDA; CPU fallback is forbidden.")
    if torch.cuda.device_count() <= 0:
        raise TabMMiniRuntimeError("TabMmini CUDA device index 0 is unavailable.")
    device = torch.device(LOCKED_DEVICE)
    try:
        properties = torch.cuda.get_device_properties(0)
        probe = torch.tensor([1.25, 2.75], dtype=torch.float32, device=device)
        result = (probe * probe).sum()
    except (RuntimeError, AssertionError) as exc:
        raise TabMMiniRuntimeError("TabMmini CUDA float32 preflight failed.") from exc
    if probe.device != device or result.device != device or not torch.isfinite(result).item():
        raise TabMMiniRuntimeError("TabMmini CUDA float32 preflight failed.")
    return {
        "status": "PASS",
        "tabm_version": installed_tabm,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": LOCKED_DEVICE,
        "device_index": 0,
        "gpu_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "float32_probe": "PASS",
        "cpu_fallback_allowed": False,
    }


def run_tabmmini_cuda_smoke(
    *,
    attempt_runner: Callable[[int], dict[str, Any]] | None = None,
    preflight_runner: Callable[[], dict[str, Any]] | None = None,
) -> TabMMiniBatchLock:
    """Run a two-epoch synthetic smoke and lock one batch size for the whole run."""
    preflight = (preflight_runner or tabmmini_cuda_preflight)()
    if (
        preflight.get("status") != "PASS"
        or preflight.get("device") != LOCKED_DEVICE
        or preflight.get("cpu_fallback_allowed") is not False
    ):
        raise TabMMiniRuntimeError("TabMmini CUDA preflight did not return a locked PASS.")
    runner = attempt_runner or _execute_smoke_attempt
    attempts: list[dict[str, Any]] = []
    for attempt_number, batch_size in enumerate(BATCH_FALLBACK_SEQUENCE, start=1):
        try:
            result = runner(batch_size)
            _validate_smoke_result(result, batch_size)
            attempts.append(
                {"attempt": attempt_number, "batch_size": batch_size, "status": "PASS"}
            )
            return TabMMiniBatchLock(batch_size, tuple(attempts))
        except Exception as exc:
            if not _is_confirmed_cuda_oom(exc):
                raise TabMMiniRuntimeError(
                    f"TabMmini smoke failed without confirmed CUDA OOM at batch {batch_size}."
                ) from exc
            attempts.append(
                {"attempt": attempt_number, "batch_size": batch_size, "status": "CUDA_OOM"}
            )
            torch.cuda.empty_cache()
            if batch_size == BATCH_FALLBACK_SEQUENCE[-1]:
                raise TabMMiniRuntimeError(
                    "TabMmini smoke exhausted the locked CUDA OOM fallback sequence."
                ) from exc
    raise AssertionError("Unreachable TabMmini smoke state.")


def select_tabmmini_fixed_epoch(losses: Iterable[float]) -> TabMMiniEpochDecision:
    """Select the earliest minimum within tolerance, or return UNRESOLVED at 100."""
    history: list[TabMMiniEpochRecord] = []
    best_epoch = 0
    best_loss = float("inf")
    for epoch, raw_loss in enumerate(losses, start=1):
        if epoch > PILOT_MAX_EPOCHS:
            break
        loss = float(raw_loss)
        if not np.isfinite(loss) or loss < 0:
            raise TabMMiniRuntimeError(
                "TabMmini inner-stop weighted BCE must be finite and non-negative."
            )
        tied = best_epoch > 0 and abs(loss - best_loss) <= PILOT_TIE_ATOL
        if best_epoch == 0 or loss < best_loss - PILOT_TIE_ATOL:
            best_epoch = epoch
            best_loss = loss
            tied = False
        history.append(TabMMiniEpochRecord(epoch, loss, tied, best_epoch))
    if not history:
        raise TabMMiniRuntimeError("TabMmini pilot produced no epoch losses.")
    if best_epoch == PILOT_MAX_EPOCHS:
        return TabMMiniEpochDecision(
            status="UNRESOLVED",
            fixed_epoch=None,
            best_loss=best_loss,
            history=tuple(history),
            reason="best_epoch_is_100_and_curve_is_still_improving",
        )
    return TabMMiniEpochDecision(
        status="LOCKED",
        fixed_epoch=best_epoch,
        best_loss=best_loss,
        history=tuple(history),
        reason="minimum_inner_stop_weighted_bce_loss_earliest_tie",
    )


def _execute_smoke_attempt(batch_size: int) -> dict[str, Any]:
    device = torch.device(LOCKED_DEVICE)
    torch.manual_seed(42)
    model = TabMMiniModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    x_num = torch.randn(batch_size, 4, dtype=torch.float32, device=device)
    cardinalities = tabmmini_category_cardinalities()
    x_cat = torch.column_stack(
        [torch.randint(0, cardinality, (batch_size,), device=device) for cardinality in cardinalities]
    ).to(torch.int64)
    y = (torch.arange(batch_size, device=device) % 2).to(torch.float32)
    pos_weight = compute_pos_weight(y.cpu()).to(device)
    last_probability: torch.Tensor | None = None
    for _ in range(SMOKE_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_num, x_cat)
        loss = member_wise_weighted_bce(logits, y, pos_weight)
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise TabMMiniRuntimeError("TabMmini smoke produced non-finite gradients.")
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            last_probability = mean_member_probabilities(model(x_num, x_cat))
    assert last_probability is not None
    with TemporaryDirectory(prefix="tabmmini-smoke-") as temporary:
        bundle = Path(temporary) / "bundle.pt"
        model.save_inference_bundle(
            bundle,
            preprocessing_state={"smoke": True},
            effective_batch_size=batch_size,
            fixed_epoch=SMOKE_EPOCHS,
            run_id="tabmmini_cuda_smoke",
            runtime_versions={"tabm": TABM_VERSION, "torch": torch.__version__},
        )
        loaded, _ = TabMMiniModel.load_inference_bundle(bundle, map_location=device)
        loaded.to(device)
        with torch.inference_mode():
            reloaded_probability = loaded.predict_approved_probability(x_num, x_cat)
        if not torch.allclose(last_probability, reloaded_probability, rtol=0, atol=0):
            raise TabMMiniRuntimeError("TabMmini smoke serialization round trip changed predictions.")
    return {
        "epochs": SMOKE_EPOCHS,
        "batch_size": batch_size,
        "device": LOCKED_DEVICE,
        "dtype": "float32",
        "logits_shape": [batch_size, 32, 1],
        "probability_shape": [batch_size],
        "serialization_round_trip": "PASS",
    }


def _validate_smoke_result(result: dict[str, Any], batch_size: int) -> None:
    expected = {
        "epochs": SMOKE_EPOCHS,
        "batch_size": batch_size,
        "device": LOCKED_DEVICE,
        "dtype": "float32",
        "logits_shape": [batch_size, 32, 1],
        "probability_shape": [batch_size],
        "serialization_round_trip": "PASS",
    }
    if result != expected:
        raise TabMMiniRuntimeError("TabMmini two-epoch smoke result violates the contract.")


def _is_confirmed_cuda_oom(exc: Exception) -> bool:
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if out_of_memory_type and isinstance(exc, out_of_memory_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
