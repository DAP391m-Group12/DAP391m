"""Train-only pilot, CV, and full-Train orchestration for TabMmini."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.dataset_b.models.tabmmini import (
    TabMMiniModel,
    compute_pos_weight,
    mean_member_probabilities,
    member_wise_weighted_bce,
)
from src.dataset_b.preprocessing.tabmmini_native import (
    TabMMiniNativeBatch,
    make_tabmmini_native_preprocessing_pipeline,
)
from src.dataset_b.runtime.tabmmini_engine import (
    BATCH_FALLBACK_SEQUENCE,
    LOCKED_DEVICE,
    PILOT_MAX_EPOCHS,
    TabMMiniBatchLock,
    TabMMiniEpochDecision,
    TabMMiniRuntimeError,
    select_tabmmini_fixed_epoch,
)
from src.dataset_b.splits.tabmmini_pilot import (
    TabMMiniPilotSplit,
    make_tabmmini_pilot_split,
)


@dataclass(frozen=True)
class TabMMiniTrainPopulation:
    X: pd.DataFrame
    y: np.ndarray
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class TabMMiniPilotResult:
    run_id: str
    batch_size: int
    split: TabMMiniPilotSplit
    pos_weight: float
    n_approved: int
    n_denied: int
    decision: TabMMiniEpochDecision
    publication_directory: Path


PilotEpochRunner = Callable[
    [Any, Any, Any, np.ndarray, Any, np.ndarray, float, int, int], Iterable[float]
]
ProductionFitRunner = Callable[[Any, Any, Any, np.ndarray, float, int, int, bool], None]
PredictionRunner = Callable[[Any, Any, int], np.ndarray]
StateWriter = Callable[[Path, Any, Any, dict[str, Any]], None]


@dataclass(frozen=True)
class TabMMiniRuntimeDependencies:
    preprocessor_factory: Callable[[], Any]
    model_factory: Callable[[], Any]
    optimizer_factory: Callable[[Any], Any]
    pilot_epoch_runner: PilotEpochRunner
    production_fit_runner: ProductionFitRunner
    prediction_runner: PredictionRunner
    state_writer: StateWriter


def default_tabmmini_runtime_dependencies() -> TabMMiniRuntimeDependencies:
    """Return the locked CUDA implementation used by production entrypoints."""

    def model_factory() -> TabMMiniModel:
        return TabMMiniModel().to(torch.device(LOCKED_DEVICE))

    return TabMMiniRuntimeDependencies(
        preprocessor_factory=make_tabmmini_native_preprocessing_pipeline,
        model_factory=model_factory,
        optimizer_factory=lambda model: torch.optim.AdamW(
            model.parameters(), lr=0.002, weight_decay=0.0003
        ),
        pilot_epoch_runner=_default_pilot_epoch_runner,
        production_fit_runner=_default_production_fit_runner,
        prediction_runner=_default_prediction_runner,
        state_writer=_default_state_writer,
    )


def run_tabmmini_pilot(
    population: TabMMiniTrainPopulation,
    destination: str | Path,
    *,
    run_id: str,
    batch_lock: TabMMiniBatchLock,
    dependencies: TabMMiniRuntimeDependencies | None = None,
) -> TabMMiniPilotResult:
    """Run one Train-only pilot and atomically publish its epoch decision."""
    checked = _validate_population(population)
    deps = dependencies or default_tabmmini_runtime_dependencies()
    result_holder: dict[str, Any] = {}

    def writer(working: Path) -> dict[str, Any]:
        split = make_tabmmini_pilot_split(checked.record_ids, checked.y)
        fit_positions = np.asarray(split.fit_positions, dtype=np.int64)
        stop_positions = np.asarray(split.stop_positions, dtype=np.int64)
        preprocessor = deps.preprocessor_factory()
        fit_native = preprocessor.fit_transform(
            checked.X.iloc[fit_positions], checked.y[fit_positions]
        )
        stop_native = preprocessor.transform(checked.X.iloc[stop_positions])
        model = deps.model_factory()
        optimizer = deps.optimizer_factory(model)
        n_approved, n_denied, pos_weight = _class_weight(checked.y[fit_positions])
        losses = deps.pilot_epoch_runner(
            model,
            optimizer,
            fit_native,
            checked.y[fit_positions],
            stop_native,
            checked.y[stop_positions],
            pos_weight,
            batch_lock.effective_batch_size,
            PILOT_MAX_EPOCHS,
        )
        decision = select_tabmmini_fixed_epoch(losses)
        payload = {
            "run_id": run_id,
            "stage": "pilot",
            "batch_size": batch_lock.effective_batch_size,
            "pilot_record_ids": list(split.pilot_record_ids),
            "fit_record_ids": list(split.fit_record_ids),
            "stop_record_ids": list(split.stop_record_ids),
            "fit_rows": len(fit_positions),
            "stop_rows": len(stop_positions),
            "n_approved": n_approved,
            "n_denied": n_denied,
            "pos_weight_formula": "n_denied / n_approved",
            "pos_weight": pos_weight,
            "pilot_stop_fits": 0,
            "model_state_published": False,
            "optimizer_state_published": False,
            "decision": asdict(decision),
        }
        _write_json(working / "pilot_result.json", payload)
        result_holder.update(
            split=split,
            n_approved=n_approved,
            n_denied=n_denied,
            pos_weight=pos_weight,
            decision=decision,
        )
        return {"result_file": "pilot_result.json", "decision_status": decision.status}

    published = _publish_atomic_stage(destination, run_id=run_id, stage="pilot", writer=writer)
    return TabMMiniPilotResult(
        run_id=run_id,
        batch_size=batch_lock.effective_batch_size,
        publication_directory=published,
        **result_holder,
    )


def run_tabmmini_cv(
    population: TabMMiniTrainPopulation,
    destination: str | Path,
    *,
    run_id: str,
    batch_lock: TabMMiniBatchLock,
    pilot_decision: TabMMiniEpochDecision,
    dependencies: TabMMiniRuntimeDependencies | None = None,
) -> dict[str, Any]:
    """Run five fresh Train-only folds at one locked epoch and publish ordered OOF."""
    checked = _validate_population(population)
    fixed_epoch = _require_locked_epoch(pilot_decision)
    deps = dependencies or default_tabmmini_runtime_dependencies()
    result_holder: dict[str, Any] = {}

    def writer(working: Path) -> dict[str, Any]:
        oof = np.empty(len(checked.y), dtype=np.float32)
        coverage = np.zeros(len(checked.y), dtype=np.int8)
        fold_results: list[dict[str, Any]] = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_index, (fit_positions, validation_positions) in enumerate(
            splitter.split(checked.X, checked.y), start=1
        ):
            preprocessor = deps.preprocessor_factory()
            model = deps.model_factory()
            optimizer = deps.optimizer_factory(model)
            fit_native = preprocessor.fit_transform(
                checked.X.iloc[fit_positions], checked.y[fit_positions]
            )
            validation_native = preprocessor.transform(checked.X.iloc[validation_positions])
            n_approved, n_denied, pos_weight = _class_weight(checked.y[fit_positions])
            try:
                deps.production_fit_runner(
                    model,
                    optimizer,
                    fit_native,
                    checked.y[fit_positions],
                    pos_weight,
                    batch_lock.effective_batch_size,
                    fixed_epoch,
                    False,
                )
                probabilities = _validated_probabilities(
                    deps.prediction_runner(
                        model, validation_native, batch_lock.effective_batch_size
                    ),
                    len(validation_positions),
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    raise TabMMiniRuntimeError(
                        "TabMmini production CV OOM at the locked batch size; fallback is forbidden."
                    ) from exc
                raise
            if np.any(coverage[validation_positions]):
                raise TabMMiniRuntimeError("TabMmini OOF positions were predicted more than once.")
            oof[validation_positions] = probabilities
            coverage[validation_positions] += 1
            fold_directory = working / f"fold_{fold_index:02d}"
            fold_directory.mkdir()
            metadata = {
                "run_id": run_id,
                "fold": fold_index,
                "fit_rows": len(fit_positions),
                "validation_rows": len(validation_positions),
                "fixed_epoch": fixed_epoch,
                "early_stopping": False,
                "batch_size": batch_lock.effective_batch_size,
                "n_approved": n_approved,
                "n_denied": n_denied,
                "pos_weight": pos_weight,
            }
            deps.state_writer(fold_directory, model, preprocessor, metadata)
            _write_json(fold_directory / "result.json", metadata)
            fold_results.append(metadata)
        if not np.array_equal(coverage, np.ones(len(coverage), dtype=np.int8)):
            raise TabMMiniRuntimeError("TabMmini OOF coverage must be exactly once per Train row.")
        if len(set(checked.record_ids)) != len(checked.record_ids):
            raise TabMMiniRuntimeError("TabMmini OOF record IDs must be unique.")
        np.savez_compressed(
            working / "oof_predictions.npz",
            record_ids=np.asarray(checked.record_ids, dtype=str),
            probabilities=oof,
            coverage=coverage,
        )
        result = {
            "run_id": run_id,
            "stage": "cv",
            "folds": fold_results,
            "fixed_epoch": fixed_epoch,
            "batch_size": batch_lock.effective_batch_size,
            "early_stopping": False,
            "oof_rows": len(oof),
            "oof_coverage": "exactly_once",
            "record_id_order": "source_train_order",
            "fresh_state_counts": {"preprocessors": 5, "models": 5, "optimizers": 5},
        }
        _write_json(working / "cv_result.json", result)
        result_holder.update(result)
        return {
            "result_file": "cv_result.json",
            "prediction_file": "oof_predictions.npz",
        }

    published = _publish_atomic_stage(destination, run_id=run_id, stage="cv", writer=writer)
    return {**result_holder, "publication_directory": str(published)}


def run_tabmmini_full_train(
    population: TabMMiniTrainPopulation,
    destination: str | Path,
    *,
    run_id: str,
    batch_lock: TabMMiniBatchLock,
    pilot_decision: TabMMiniEpochDecision,
    dependencies: TabMMiniRuntimeDependencies | None = None,
) -> dict[str, Any]:
    """Fit one fresh full-Train state for exactly the pilot-locked epoch count."""
    checked = _validate_population(population)
    fixed_epoch = _require_locked_epoch(pilot_decision)
    deps = dependencies or default_tabmmini_runtime_dependencies()
    result_holder: dict[str, Any] = {}

    def writer(working: Path) -> dict[str, Any]:
        preprocessor = deps.preprocessor_factory()
        model = deps.model_factory()
        optimizer = deps.optimizer_factory(model)
        native = preprocessor.fit_transform(checked.X, checked.y)
        n_approved, n_denied, pos_weight = _class_weight(checked.y)
        try:
            deps.production_fit_runner(
                model,
                optimizer,
                native,
                checked.y,
                pos_weight,
                batch_lock.effective_batch_size,
                fixed_epoch,
                False,
            )
        except Exception as exc:
            if _is_cuda_oom(exc):
                raise TabMMiniRuntimeError(
                    "TabMmini full-Train OOM at the locked batch size; fallback is forbidden."
                ) from exc
            raise
        metadata = {
            "run_id": run_id,
            "stage": "full_train",
            "train_rows": len(checked.y),
            "fixed_epoch": fixed_epoch,
            "early_stopping": False,
            "batch_size": batch_lock.effective_batch_size,
            "n_approved": n_approved,
            "n_denied": n_denied,
            "pos_weight_formula": "n_denied / n_approved",
            "pos_weight": pos_weight,
            "fresh_state_counts": {"preprocessors": 1, "models": 1, "optimizers": 1},
            "resume_model_state": False,
            "resume_optimizer_state": False,
        }
        deps.state_writer(working / "model", model, preprocessor, metadata)
        _write_json(working / "full_train_result.json", metadata)
        result_holder.update(metadata)
        return {"result_file": "full_train_result.json", "model_directory": "model"}

    published = _publish_atomic_stage(
        destination, run_id=run_id, stage="full_train", writer=writer
    )
    return {**result_holder, "publication_directory": str(published)}


def _publish_atomic_stage(
    destination: str | Path,
    *,
    run_id: str,
    stage: str,
    writer: Callable[[Path], dict[str, Any]],
) -> Path:
    destination = Path(destination)
    working = destination.with_name(destination.name + ".working")
    if destination.exists():
        raise TabMMiniRuntimeError(
            f"TabMmini refuses to overwrite published {stage} destination: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if working.exists():
        failed = _next_failed_attempt(working)
        os.replace(working, failed)
    working.mkdir()
    details = writer(working)
    if not isinstance(details, dict):
        raise TabMMiniRuntimeError("TabMmini stage writer must return manifest details.")
    for relative in details.values():
        if isinstance(relative, str) and (relative.endswith(".json") or relative.endswith(".npz")):
            if not (working / relative).is_file():
                raise TabMMiniRuntimeError(f"TabMmini stage output is missing: {relative}")
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "status": "COMPLETE",
        "allow_resume_model_state": False,
        "allow_resume_optimizer_state": False,
        **details,
    }
    _write_json(working / "manifest.json", manifest)
    reopened = _read_json(working / "manifest.json")
    if reopened != manifest:
        raise TabMMiniRuntimeError("TabMmini stage manifest failed direct reopen validation.")
    os.replace(working, destination)
    if _read_json(destination / "manifest.json") != manifest:
        raise TabMMiniRuntimeError("TabMmini published manifest failed validation.")
    return destination


def _next_failed_attempt(working: Path) -> Path:
    for attempt in range(1, 1000):
        candidate = working.with_name(working.name + f".failed-{attempt:03d}")
        if not candidate.exists():
            return candidate
    raise TabMMiniRuntimeError("TabMmini failed-attempt namespace is exhausted.")


def _validate_population(population: TabMMiniTrainPopulation) -> TabMMiniTrainPopulation:
    if not isinstance(population.X, pd.DataFrame) or population.X.empty:
        raise ValueError("TabMmini Train population X must be a non-empty DataFrame.")
    labels = np.asarray(population.y)
    ids = tuple(str(value) for value in population.record_ids)
    if labels.ndim != 1 or len(labels) != len(population.X) or len(ids) != len(labels):
        raise ValueError("TabMmini Train X, labels, and record IDs must align.")
    if not np.isin(labels, (0, 1)).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("TabMmini Train population must contain both binary classes.")
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("TabMmini Train record IDs must be unique and nonblank.")
    return TabMMiniTrainPopulation(
        X=population.X,
        y=labels.astype(np.int64, copy=False),
        record_ids=ids,
    )


def _require_locked_epoch(decision: TabMMiniEpochDecision) -> int:
    if decision.status != "LOCKED" or decision.fixed_epoch is None:
        raise TabMMiniRuntimeError("TabMmini CV/full-Train requires a locked pilot epoch.")
    if not 1 <= decision.fixed_epoch < PILOT_MAX_EPOCHS:
        raise TabMMiniRuntimeError("TabMmini fixed epoch is outside the resolved range.")
    return decision.fixed_epoch


def _class_weight(labels: np.ndarray) -> tuple[int, int, float]:
    weight = float(compute_pos_weight(labels).item())
    n_approved = int(np.count_nonzero(labels == 1))
    n_denied = int(np.count_nonzero(labels == 0))
    return n_approved, n_denied, weight


def _validated_probabilities(values: object, rows: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float32)
    if probabilities.shape != (rows,):
        raise TabMMiniRuntimeError("TabMmini fold prediction must have shape [n].")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise TabMMiniRuntimeError("TabMmini fold probabilities must be finite in [0, 1].")
    return probabilities


def _is_cuda_oom(exc: Exception) -> bool:
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return bool(out_of_memory_type and isinstance(exc, out_of_memory_type)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TabMMiniRuntimeError(f"TabMmini JSON artifact must be an object: {path}")
    return value


def _default_pilot_epoch_runner(
    model: TabMMiniModel,
    optimizer: torch.optim.Optimizer,
    fit_native: TabMMiniNativeBatch,
    fit_y: np.ndarray,
    stop_native: TabMMiniNativeBatch,
    stop_y: np.ndarray,
    pos_weight: float,
    batch_size: int,
    max_epochs: int,
) -> Iterable[float]:
    for epoch in range(1, max_epochs + 1):
        _train_one_epoch(
            model, optimizer, fit_native, fit_y, pos_weight, batch_size, seed=42 + epoch
        )
        yield _evaluate_weighted_bce(
            model, stop_native, stop_y, pos_weight, batch_size
        )


def _default_production_fit_runner(
    model: TabMMiniModel,
    optimizer: torch.optim.Optimizer,
    native: TabMMiniNativeBatch,
    labels: np.ndarray,
    pos_weight: float,
    batch_size: int,
    epochs: int,
    early_stopping: bool,
) -> None:
    if early_stopping:
        raise TabMMiniRuntimeError("TabMmini production early stopping is forbidden.")
    for epoch in range(1, epochs + 1):
        _train_one_epoch(
            model, optimizer, native, labels, pos_weight, batch_size, seed=42 + epoch
        )


def _train_one_epoch(
    model: TabMMiniModel,
    optimizer: torch.optim.Optimizer,
    native: TabMMiniNativeBatch,
    labels: np.ndarray,
    pos_weight: float,
    batch_size: int,
    *,
    seed: int,
) -> None:
    if batch_size not in BATCH_FALLBACK_SEQUENCE:
        raise ValueError("TabMmini training must use the smoke-locked batch size.")
    device = torch.device(LOCKED_DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(labels), generator=generator)
    model.train()
    for start in range(0, len(labels), batch_size):
        positions = order[start : start + batch_size]
        x_num = native.x_num[positions].to(device)
        x_cat = native.x_cat[positions].to(device)
        y = torch.as_tensor(labels[positions.numpy()], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = member_wise_weighted_bce(model(x_num, x_cat), y, pos_weight)
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise TabMMiniRuntimeError("TabMmini training produced non-finite gradients.")
        optimizer.step()


def _evaluate_weighted_bce(
    model: TabMMiniModel,
    native: TabMMiniNativeBatch,
    labels: np.ndarray,
    pos_weight: float,
    batch_size: int,
) -> float:
    device = torch.device(LOCKED_DEVICE)
    total = 0.0
    rows = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            logits = model(native.x_num[start:stop].to(device), native.x_cat[start:stop].to(device))
            y = torch.as_tensor(labels[start:stop], dtype=torch.float32, device=device)
            loss = member_wise_weighted_bce(logits, y, pos_weight)
            total += float(loss.item()) * (stop - start)
            rows += stop - start
    return total / rows


def _default_prediction_runner(
    model: TabMMiniModel, native: TabMMiniNativeBatch, batch_size: int
) -> np.ndarray:
    device = torch.device(LOCKED_DEVICE)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(native.x_num), batch_size):
            stop = min(start + batch_size, len(native.x_num))
            values = mean_member_probabilities(
                model(native.x_num[start:stop].to(device), native.x_cat[start:stop].to(device))
            )
            chunks.append(values.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks)


def _default_state_writer(
    directory: Path, model: TabMMiniModel, preprocessor: Any, metadata: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        directory / "model_state.pt",
    )
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
