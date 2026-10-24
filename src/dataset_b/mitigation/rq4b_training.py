"""RQ4b production adversarial RealMLP training, CV epoch selection and refit.

Production-owned PyTabKit 1.7.3 extension. The accepted disposable spike
(``rq4b_spike.py``) is reference evidence only: nothing here imports it, uses
its two-epoch guard, or patches PyTabKit at runtime. Every extension point is a
subclass (module, algorithm interface, backend, sklearn wrapper, loader).

Entry points:

- ``fit_cv_fold`` / ``run_cv_epoch_selection``: Train-only five-fold curves,
  each fold executing all 256 epochs with fold-validation CE after every epoch.
- ``run_full_train_refit``: fresh full-Train fit executing exactly the selected
  epoch while the LR/WD/dropout schedule horizon stays 256.
- ``fit_smoke``: synthetic GPU smoke limited to two executed epochs; it is the
  only route that may execute fewer epochs than the production budget.

Production entry points require a ``ProductionContext`` whose mandatory
prerequisites are all exactly ``PASS`` and whose CUDA preflight passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover - environment dependent
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface, get_lignting_accel_and_devices
from pytabkit.models.data import data as _ptk_data
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.training.logging import Logger
from sklearn.model_selection import StratifiedKFold

from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.models.realmlp import PYTABKIT_VERSION, RealMLPClassifier, load_realmlp_v1_config
from src.dataset_b.preprocessing.categorical_encoder import NUMERIC_VALUE_COLUMNS
from src.utils.path import PROJECT_ROOT


DEVICE = "cuda:0"
DEVICE_INDEX = 0
BATCH_SIZE = 256
PREDICT_BATCH_SIZE = 1024
SCHEDULE_HORIZON_EPOCHS = 256
CV_FOLDS = 5
CV_RANDOM_STATE = 42
SMOKE_MAX_EPOCHS = 2
ADVERSARY_INPUTS = 2
ADVERSARY_HIDDEN = 32
ADVERSARY_OUTPUTS = 3
ADVERSARY_LR = 0.001
ADVERSARY_WEIGHT_DECAY = 0.0
EQUIVALENCE_RTOL = 1e-5
EQUIVALENCE_ATOL = 1e-6
# Predeclared tolerance for the autograd identity d(total) = d(pred) - lambda d(adv).
GRADIENT_IDENTITY_RTOL = 1e-5
GRADIENT_IDENTITY_ATOL = 1e-7
ROW_INDEX_KEY = "_rq4b_row_idx"
HEARTBEAT_BATCH_INTERVAL = 100
HEARTBEAT_SECONDS = 30.0


class DurableProgress:
    """Append-only fit evidence plus a replaceable live ledger.

    JSONL and text records are flushed and fsynced before returning. Heartbeats
    report callback observation only; they never claim that an optimizer step
    completed.
    """

    def __init__(self, root: Path, *, run_id: str, attempt_id: str, unit: str,
                 adversary_lambda: float, model_seed: int) -> None:
        self.root = Path(root)
        self.base = {
            "run_id": run_id, "attempt_id": attempt_id, "unit": unit,
            "adversary_lambda": float(adversary_lambda), "lambda": float(adversary_lambda),
            "model_seed": int(model_seed), "seed": int(model_seed),
        }
        self.events: list[dict[str, Any]] = []
        self.ledger: dict[str, Any] = {
            **self.base, "planned": {"cv_fold_fits": CV_FOLDS, "refit_fits": 1},
            "started": {"cv_fold_fits": 0, "refit_fits": 0},
            "completed": {"cv_fold_fits": 0, "refit_fits": 0},
            "partial": {"cv_fold_fits": 0, "refit_fits": 0},
            "unknown": False, "last_position": None, "last_progress_utc": None,
        }
        self._last_heartbeat = 0.0
        self._write_ledger()

    def _write_ledger(self) -> None:
        target = self.root / "fit_ledger.json"
        temporary = self.root / ".fit_ledger.json.tmp"
        data = json.dumps(self.ledger, indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def emit(self, event: str, *, phase: str, fold: int | None = None,
             epoch: int | None = None, batch: int | None = None, **values: Any) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        if values.get("optimizer_progress_claimed") is True or event == "phase_completed":
            self.ledger["last_progress_utc"] = timestamp
        record = {
            **self.base, "timestamp_utc": timestamp, "last_progress_utc": self.ledger["last_progress_utc"],
            "event": event, "phase": phase, "fold": fold, "epoch": epoch,
            "batch": batch, **values,
        }
        line = json.dumps(record, sort_keys=True, default=str)
        for path, rendered in ((self.root / "training_history.jsonl", line + "\n"),
                               (self.root / "run.log", line + "\n")):
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
        print(line, file=sys.stderr, flush=True)
        self.events.append(record)
        self.ledger["last_position"] = {"phase": phase, "fold": fold, "epoch": epoch, "batch": batch, "event": event}
        self._write_ledger()
        return record

    def phase_started(self, phase: str, fold: int | None = None) -> None:
        counter = "cv_fold_fits" if phase == "cv" else "refit_fits"
        self.ledger["started"][counter] += 1
        self.emit("phase_started", phase=phase, fold=fold, optimizer_progress_claimed=False)

    def phase_completed(self, phase: str, fold: int | None = None) -> None:
        counter = "cv_fold_fits" if phase == "cv" else "refit_fits"
        self.ledger["completed"][counter] += 1
        self.emit("phase_completed", phase=phase, fold=fold, optimizer_progress_claimed=False)

    def phase_failed(self, phase: str, fold: int | None = None) -> None:
        counter = "cv_fold_fits" if phase == "cv" else "refit_fits"
        self.ledger["partial"][counter] += 1
        self.emit("phase_failed", phase=phase, fold=fold, optimizer_progress_claimed=False)

    def heartbeat(self, *, phase: str, fold: int | None, epoch: int, batch: int,
                  gpu_memory_bytes: int | None) -> None:
        now = time.monotonic()
        if batch == 1 or batch % HEARTBEAT_BATCH_INTERVAL == 0 or now - self._last_heartbeat >= HEARTBEAT_SECONDS:
            self._last_heartbeat = now
            self.emit(
                "heartbeat", phase=phase, fold=fold, epoch=epoch, batch=batch,
                train_loss=None, fold_validation_ce=None, learning_rate=None,
                gpu_memory_bytes=gpu_memory_bytes,
                unavailable_reason="batch-start callback has no completed-batch loss or optimizer-step evidence",
                optimizer_progress_claimed=False,
            )

    def finalize(self) -> None:
        target = self.root / "training_history.json"
        payload = {"context": self.base, "ledger": self.ledger, "events": self.events}
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
SMOKE_OUTPUT_ROOT = PROJECT_ROOT / "scratch" / "rq4b_implementation"

EVIDENCE_OBSERVED = "observed_fit"
EVIDENCE_SMOKE = "synthetic_smoke"

# Installed PyTabKit sources accepted at C-spike: git blob SHA-1 of each internal
# file the extension relies on, equal to upstream tag v1.7.3 commit below
# (evidence scratch/rq4b_spike/r1_evidence_20261018_a1/g01_pin_pytabkit_source_revision.json).
PYTABKIT_UPSTREAM_COMMIT = "c126ea51187c5080b91f28d352481dbd3b2194b0"
PYTABKIT_LINEAGE_EVIDENCE = "scratch/rq4b_spike/r1_evidence_20261018_a1/g01_pin_pytabkit_source_revision.json"
PYTABKIT_PINNED_BLOBS = {
    "pytabkit/models/alg_interfaces/nn_interfaces.py": "015b3e2723496dd55a8158b82533217c093657e7",
    "pytabkit/models/data/data.py": "8201dd7afe9f279cdb56104335502a76d0e871f8",
    "pytabkit/models/optim/optimizers.py": "87fab3f1aacb2e5e71d6422cef8c321ea14a9f46",
    "pytabkit/models/sklearn/sklearn_base.py": "62ec2f50056def71382165ab0f98d7a91543f764",
    "pytabkit/models/sklearn/sklearn_interfaces.py": "9d3d9c7f65fb649da6bf462e68f25e46481f6a1c",
    "pytabkit/models/torch_utils.py": "df044a5903752d9aff7989b99a89b598ddc8b649",
    "pytabkit/models/training/coord.py": "7e2fede7515c417503c577bb23a0e0f177c3a855",
    "pytabkit/models/training/lightning_callbacks.py": "7cb9c7e86ff91fd3df531196ee095a7003ad5d1a",
    "pytabkit/models/training/lightning_modules.py": "efb89e438e04a93baf2ab9a987790a5f9a193fc7",
    "pytabkit/models/training/metrics.py": "38f7f4d1abe1d46984848cd09308e57f1d99990c",
    "pytabkit/models/training/nn_creator.py": "7b37c4565c3c64d2f0bb70af5f5526300b654260",
    "pytabkit/models/training/scheduling.py": "3d4d6fb8e2429b0a82641b13eafb5da7a7a7785d",
}


class RQ4bTrainingError(RuntimeError):
    """Base class for fail-closed RQ4b training errors."""


class RQ4bCudaUnavailableError(RQ4bTrainingError):
    """cuda:0 is unavailable or not used; CPU fallback is prohibited."""


class RQ4bPrerequisiteError(RQ4bTrainingError):
    """A mandatory prerequisite is not exactly PASS."""


class RQ4bBudgetError(RQ4bTrainingError):
    """An execution budget or schedule horizon differs from the locked policy."""


class RQ4bNonFiniteError(RQ4bTrainingError):
    """A loss, gradient, parameter or evaluation value was not finite."""


class RQ4bAlignmentError(RQ4bTrainingError):
    """Row/group metadata is misaligned with backend rows or shuffled batches."""


class RQ4bReloadError(RQ4bTrainingError):
    """A serialized model failed to reload or changed eval-mode probabilities."""


class RQ4bEpochSelectionError(RQ4bTrainingError):
    """CV curves are missing, malformed or nonfinite."""


# ---------------------------------------------------------------------------
# Prerequisites, lineage and CUDA preflight
# ---------------------------------------------------------------------------


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pytabkit_lineage() -> dict[str, Any]:
    """Compare installed internal PyTabKit files with the pinned upstream blobs."""
    import pytabkit

    site_packages = Path(pytabkit.__file__).resolve().parent.parent
    installed = version("pytabkit")
    files = {}
    for relative, pinned in PYTABKIT_PINNED_BLOBS.items():
        path = site_packages / relative
        blob = _git_blob_sha1(path.read_bytes()) if path.exists() else None
        files[relative] = {"installed_git_blob_sha1": blob, "pinned_git_blob_sha1": pinned, "match": blob == pinned}
    status = "PASS" if installed == PYTABKIT_VERSION and all(item["match"] for item in files.values()) else "FAIL"
    return {
        "status": status,
        "package_version": installed,
        "upstream_commit": PYTABKIT_UPSTREAM_COMMIT,
        "pinned_evidence": PYTABKIT_LINEAGE_EVIDENCE,
        "files": files,
    }


def prerequisite_statuses(config: Mapping[str, Any], *, lineage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Mandatory production prerequisites as recorded in the RQ4b YAML and environment."""

    def lookup(*path: str) -> Any:
        node: Any = config
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        return node

    lineage = pytabkit_lineage() if lineage is None else lineage
    return {
        "technical_spike_codex_acceptance": lookup("technical_spike", "codex_acceptance", "status"),
        "input_boundary_codex_acceptance": lookup("execution_governance", "t1_input_boundary_acceptance", "result"),
        "pytabkit_source_lineage": lineage.get("status"),
    }


def require_prerequisites(statuses: Mapping[str, Any]) -> dict[str, str]:
    """Every mandatory prerequisite must be the exact string ``PASS``."""
    required = ("technical_spike_codex_acceptance", "input_boundary_codex_acceptance", "pytabkit_source_lineage")
    failing = {name: statuses.get(name) for name in required if statuses.get(name) != "PASS" or type(statuses.get(name)) is not str}
    if failing:
        raise RQ4bPrerequisiteError(f"Mandatory RQ4b prerequisites are not PASS: {failing}")
    return {name: "PASS" for name in required}


def require_cuda0() -> dict[str, Any]:
    """Fail closed unless cuda:0 exists and performs a finite operation."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= DEVICE_INDEX:
        raise RQ4bCudaUnavailableError("RQ4b requires cuda:0; CPU fallback is prohibited.")
    probe = torch.ones(4, device=DEVICE) * 2.0
    if probe.device != torch.device(DEVICE) or not bool(torch.isfinite(probe).all().item()):
        raise RQ4bCudaUnavailableError("RQ4b cuda:0 finite-operation preflight failed.")
    return {"device": str(probe.device), "device_name": torch.cuda.get_device_name(DEVICE_INDEX), "finite_operation": True}


@dataclass(frozen=True)
class ProductionContext:
    """Validated contract plus PASS prerequisites; required by production entry points."""

    contract: ri.RQ4bContract
    prerequisites: Mapping[str, str]
    cuda: Mapping[str, Any]
    lineage: Mapping[str, Any] = field(repr=False)


def open_production_context(config_path: Path = ri.RQ4B_CONFIG_PATH) -> ProductionContext:
    contract = ri.load_contract(config_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    lineage = pytabkit_lineage()
    prerequisites = require_prerequisites(prerequisite_statuses(config, lineage=lineage))
    _require_runtime_contract(contract)
    cuda = require_cuda0()
    return ProductionContext(contract=contract, prerequisites=prerequisites, cuda=cuda, lineage=lineage)


def _require_runtime_contract(contract: ri.RQ4bContract) -> None:
    expected = (DEVICE, BATCH_SIZE, PREDICT_BATCH_SIZE, CV_FOLDS, CV_RANDOM_STATE, SCHEDULE_HORIZON_EPOCHS)
    found = (contract.device, contract.batch_size, contract.predict_batch_size, contract.cv_folds, contract.cv_random_state, contract.epochs_per_fold)
    if found != expected:
        raise RQ4bBudgetError(f"RQ4b runtime contract {found} differs from the training engine {expected}.")


# ---------------------------------------------------------------------------
# Fit specification and row metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitSpec:
    """One predictor fit: lambda, seed, execution budget and schedule horizon."""

    adversary_lambda: float
    model_seed: int
    execution_epochs: int
    schedule_horizon_epochs: int = SCHEDULE_HORIZON_EPOCHS
    module_kind: str = "adversarial"  # "adversarial" or "stock" (preserved-recipe reference)
    record_batches: bool = False
    record_gradient_diagnostics: bool = False
    alignment_group_feature_column: int | None = None  # synthetic verification fixtures only

    def __post_init__(self) -> None:
        if self.module_kind not in ("adversarial", "stock"):
            raise RQ4bTrainingError(f"Unknown RQ4b module kind {self.module_kind!r}.")
        if isinstance(self.adversary_lambda, bool) or not math.isfinite(float(self.adversary_lambda)) or self.adversary_lambda < 0:
            raise RQ4bTrainingError("RQ4b lambda must be finite and non-negative.")
        for name in ("model_seed", "execution_epochs", "schedule_horizon_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RQ4bTrainingError(f"RQ4b {name} must be an integer.")
        if not 1 <= self.execution_epochs <= self.schedule_horizon_epochs:
            raise RQ4bBudgetError("Execution epochs must lie in 1..schedule horizon.")


@dataclass(frozen=True)
class RowMetadata:
    """Row-aligned training metadata kept outside the predictor feature frame."""

    record_ids: tuple[str, ...]
    labels: np.ndarray
    group_codes: np.ndarray
    numeric_reference: np.ndarray  # float32 (rows, 4) NUMERIC_VALUE_COLUMNS in X order

    @classmethod
    def from_input(cls, inputs: ri.RQ4bPredictorInput) -> "RowMetadata":
        return cls(
            record_ids=tuple(inputs.record_ids),
            labels=np.asarray(inputs.y, dtype=np.int64),
            group_codes=np.asarray(inputs.group_codes, dtype=np.int64),
            numeric_reference=native_numeric_features(inputs.X),
        )


@dataclass(frozen=True)
class FoldValidation:
    """Held-out fold rows: transformed and predicted only, never fitted."""

    X: pd.DataFrame
    y: np.ndarray
    record_ids: tuple[str, ...]


def native_numeric_features(X: pd.DataFrame) -> np.ndarray:
    return X.loc[:, list(NUMERIC_VALUE_COLUMNS)].to_numpy(dtype=np.float32)


def stable_binary_ce_from_margin(margin: np.ndarray, y: np.ndarray) -> float:
    """Unweighted mean binary CE (Approved=1) from the two-class logit margin; no clipping."""
    margin = np.asarray(margin, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, margin) - labels * margin))


def approved_minus_denied_margin(logits: torch.Tensor) -> torch.Tensor:
    """Class-1 minus class-0 logit margin; RealMLP logits are (models, rows, 2)."""
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
        raise RQ4bTrainingError(f"Unexpected RealMLP score tensor shape {tuple(logits.shape)}.")
    return (logits[..., 1] - logits[..., 0]).reshape(-1)


def _state_digest(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().to("cpu").contiguous()
        digest.update(f"{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _snapshot(parameters: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _max_change(before: Sequence[torch.Tensor], parameters: Sequence[torch.Tensor]) -> float:
    return max(float((after.detach() - old).abs().max().item()) for old, after in zip(before, parameters))


def _all_finite(tensors: Sequence[torch.Tensor]) -> bool:
    flags = [torch.isfinite(tensor).all() for tensor in tensors if tensor is not None]
    return bool(torch.stack(flags).all().item()) if flags else True


def build_adversary(seed: int, device: str = DEVICE) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Locked 2 -> 32 ReLU -> 3 adversary built under an isolated RNG fork."""
    device_index = torch.device(device).index
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(device_index)
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        adversary = torch.nn.Sequential(
            torch.nn.Linear(ADVERSARY_INPUTS, ADVERSARY_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(ADVERSARY_HIDDEN, ADVERSARY_OUTPUTS),
        )
    adversary = adversary.to(device)
    isolation = {
        "cpu_rng_state_unchanged": torch.equal(cpu_before, torch.get_rng_state()),
        "cuda_rng_state_unchanged": torch.equal(cuda_before, torch.cuda.get_rng_state(device_index)),
        "initial_state_sha256": _state_digest(list(adversary.state_dict().values())),
    }
    if not (isolation["cpu_rng_state_unchanged"] and isolation["cuda_rng_state_unchanged"]):
        raise RQ4bTrainingError("Adversary construction changed predictor RNG state.")
    return adversary, isolation


# ---------------------------------------------------------------------------
# Epoch recorder: budget/horizon assertions, schedules, finite state, fold CE
# ---------------------------------------------------------------------------


def _batch_fingerprint(batch: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in batch if key != ROW_INDEX_KEY):
        tensor = batch[key].detach().to("cpu").contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _schedule_values(pl_module: pl.LightningModule) -> dict[str, dict[str, float]]:
    sched = pl_module.hp_manager.get_hyper_sched_values()
    return {name: {pattern: float(value) for pattern, value in sched[name].items()} for name in ("lr", "wd", "p_drop") if name in sched}


class EpochRecorder(Callback):
    """Read-only per-epoch recorder; also evaluates fold-validation CE in eval/no-grad."""

    def __init__(self, spec: FitSpec, fold_validation: FoldValidation | None = None, *,
                 progress: DurableProgress | None = None, phase: str = "fit", fold: int | None = None) -> None:
        super().__init__()
        self.spec = spec
        self._fold_validation = fold_validation
        self._converter = None
        self._val_loader = None
        self.epochs: list[dict[str, Any]] = []
        self.batch_fingerprints: list[str] = []
        self.batch_schedules: list[dict[str, Any]] = []
        self.fold_validation_ce: list[float] = []
        self.budget: dict[str, Any] | None = None
        self.completed_epochs = 0
        self._epoch_first_schedule: dict[str, Any] | None = None
        self._epoch_last_schedule: dict[str, Any] | None = None
        self._progress = progress
        self._phase = phase
        self._fold = fold
        self._fit_started = 0.0
        self._epoch_started = 0.0

    def bind_converter(self, converter: Any) -> None:
        """The backend's train-only fitted DataFrame -> DictDataset converter."""
        self._converter = converter

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._fit_started = time.perf_counter()
        self.budget = {
            "trainer_max_epochs": int(trainer.max_epochs),
            "module_schedule_horizon_epochs": int(pl_module.progress.max_epochs),
            "batch_size": int(pl_module.train_dl.batch_size),
            "drop_last": bool(pl_module.train_dl.drop_last),
            "shuffle": bool(pl_module.train_dl.shuffle),
            "batches_per_epoch": int(pl_module.train_dl.n_batches),
            "iterated_samples_per_epoch": int(pl_module.train_dl.get_num_iterated_samples()),
            "train_rows": int(pl_module.train_dl.n_samples),
            "predict_batch_size": int(pl_module.creator.config.get("predict_batch_size", -1)),
        }
        if self.budget["trainer_max_epochs"] != self.spec.execution_epochs:
            raise RQ4bBudgetError("Trainer epoch budget differs from the requested execution budget.")
        if self.budget["module_schedule_horizon_epochs"] != self.spec.schedule_horizon_epochs:
            raise RQ4bBudgetError("Module schedule horizon differs from the requested horizon.")
        if (self.budget["batch_size"], self.budget["drop_last"], self.budget["shuffle"]) != (BATCH_SIZE, True, True):
            raise RQ4bBudgetError("Native train loader must keep batch size 256, drop_last and shuffle.")
        if self.budget["predict_batch_size"] != PREDICT_BATCH_SIZE:
            raise RQ4bBudgetError("Prediction batch size must be 1024.")
        if self._fold_validation is not None:
            if self._converter is None:
                raise RQ4bTrainingError("Fold-validation recording requires the train-fitted converter.")
            ds = self._converter.transform(self._fold_validation.X).to(pl_module.creator.device_info)
            ds_x, _ = ds.split_xy()
            self._val_loader = pl_module.get_predict_dataloader(ds_x)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch >= self.spec.execution_epochs:
            raise RQ4bBudgetError("RQ4b attempted more epochs than the execution budget.")
        self._epoch_first_schedule = None
        self._epoch_started = time.perf_counter()

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        entry = {
            "epoch": int(trainer.current_epoch) + 1,
            "batch_idx": int(batch_idx),
            "fit_progress": float(pl_module.progress.get_fit_progress()),
            "sched": _schedule_values(pl_module),
        }
        if self._epoch_first_schedule is None:
            self._epoch_first_schedule = entry
        self._epoch_last_schedule = entry
        if self.spec.record_batches:
            self.batch_fingerprints.append(_batch_fingerprint(batch))
            self.batch_schedules.append(entry)
        if self._progress is not None:
            gpu_memory = int(torch.cuda.memory_allocated(DEVICE_INDEX)) if torch.cuda.is_available() else None
            self._progress.heartbeat(
                phase=self._phase, fold=self._fold, epoch=int(trainer.current_epoch) + 1,
                batch=int(batch_idx) + 1, gpu_memory_bytes=gpu_memory,
            )

    def _evaluate_fold_validation(self, pl_module: pl.LightningModule) -> tuple[float, bool]:
        model = pl_module.model
        parameters = list(model.parameters())
        before = _snapshot(parameters)
        was_training = model.training
        optimizer = pl_module.optimizers(use_pl_optimizer=False)
        model.eval()
        optimizer.eval()
        try:
            with torch.no_grad():
                logits = torch.cat([model(batch)["x_cont"] for batch in self._val_loader], dim=-2)
        finally:
            model.train(was_training)
            optimizer.train()
        unchanged = all(torch.equal(old, new.detach()) for old, new in zip(before, parameters))
        if not unchanged:
            raise RQ4bTrainingError("Fold-validation evaluation changed predictor parameters.")
        margin = approved_minus_denied_margin(logits).detach().to("cpu").numpy().astype(np.float64)
        if margin.shape != (len(self._fold_validation.y),):
            raise RQ4bAlignmentError("Fold-validation predictions do not cover every fold-validation row.")
        ce = stable_binary_ce_from_margin(margin, self._fold_validation.y)
        if not math.isfinite(ce):
            raise RQ4bNonFiniteError("Fold-validation CE is not finite.")
        return ce, unchanged

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        predictor_finite = _all_finite(list(pl_module.model.parameters()))
        adversary = getattr(pl_module, "rq4b_adversary", None)
        adversary_finite = _all_finite(list(adversary.parameters())) if adversary is not None else None
        if not predictor_finite or adversary_finite is False:
            raise RQ4bNonFiniteError("Predictor or adversary parameters became nonfinite.")
        summary: dict[str, Any] = {
            "epoch": int(trainer.current_epoch) + 1,
            "predictor_parameters_finite": predictor_finite,
            "adversary_parameters_finite": adversary_finite,
            "first_batch_schedule": self._epoch_first_schedule,
            "last_batch_schedule": self._epoch_last_schedule,
        }
        epoch_log = getattr(pl_module, "rq4b_epoch_counters", None)
        if callable(getattr(pl_module, "rq4b_close_epoch", None)):
            summary["losses"] = pl_module.rq4b_close_epoch()
        elif epoch_log is not None:
            summary["losses"] = dict(epoch_log)
        if self._val_loader is not None:
            ce, unchanged = self._evaluate_fold_validation(pl_module)
            self.fold_validation_ce.append(ce)
            summary.update(fold_validation_ce=ce, fold_validation_eval_left_parameters_unchanged=unchanged)
        losses = summary.get("losses", {})
        schedule = (self._epoch_last_schedule or {}).get("sched", {}).get("lr", {})
        learning_rate = next(iter(schedule.values()), None)
        now = time.perf_counter()
        summary.update(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            epoch_seconds=now - self._epoch_started,
            elapsed_seconds=now - self._fit_started,
            learning_rate=learning_rate,
            gpu_memory_bytes=int(torch.cuda.memory_allocated(DEVICE_INDEX)) if torch.cuda.is_available() else None,
            status="completed",
        )
        self.epochs.append(summary)
        self.completed_epochs += 1
        if self._progress is not None:
            self._progress.emit(
                "epoch_completed", phase=self._phase, fold=self._fold, epoch=summary["epoch"],
                batch=losses.get("batches"), train_loss=losses.get("mean_total_loss"),
                prediction_loss=losses.get("mean_prediction_loss"),
                fold_validation_ce=summary.get("fold_validation_ce"), learning_rate=learning_rate,
                epoch_seconds=summary["epoch_seconds"], elapsed_seconds=summary["elapsed_seconds"],
                gpu_memory_bytes=summary["gpu_memory_bytes"], status="completed",
                unavailable_reason=None if self._val_loader is not None else "fold validation is not applicable to full-Train refit",
                optimizer_progress_claimed=True,
            )

    def release(self) -> None:
        """Drop the converter, held-out data and loader after fit (nothing raw is kept)."""
        self._converter = None
        self._val_loader = None
        self._fold_validation = None

    def result(self) -> dict[str, Any]:
        return {
            "spec": _spec_dict(self.spec),
            "budget": self.budget,
            "completed_epochs": self.completed_epochs,
            "epochs": self.epochs,
            "fold_validation_ce": list(self.fold_validation_ce),
            "batch_fingerprints": list(self.batch_fingerprints),
            "batch_schedules": list(self.batch_schedules),
        }


def _spec_dict(spec: FitSpec) -> dict[str, Any]:
    return {
        "adversary_lambda": float(spec.adversary_lambda),
        "model_seed": spec.model_seed,
        "adversary_seed": spec.model_seed,
        "execution_epochs": spec.execution_epochs,
        "schedule_horizon_epochs": spec.schedule_horizon_epochs,
        "module_kind": spec.module_kind,
    }


# ---------------------------------------------------------------------------
# PyTabKit extension (subclasses only)
# ---------------------------------------------------------------------------


@dataclass
class _EngineSettings:
    spec: FitSpec
    metadata: RowMetadata
    recorder: EpochRecorder


class _RowIndexedTrainLoader(_ptk_data.ParallelDictDataLoader):
    """Native shuffled loader that also yields the gathered dataset row indices.

    Permutation and slicing are identical to ``ParallelDictDataLoader.__iter__``
    so RNG consumption and batch order are unchanged; the extra key is removed
    before predictor forward.
    """

    @classmethod
    def from_loader(cls, loader: _ptk_data.ParallelDictDataLoader) -> "_RowIndexedTrainLoader":
        if type(loader) is not _ptk_data.ParallelDictDataLoader or not loader.drop_last or not loader.shuffle:
            raise RQ4bTrainingError("Expected the native shuffled drop_last train loader.")
        wrapped = cls.__new__(cls)
        wrapped.__dict__.update(loader.__dict__)
        return wrapped

    def __iter__(self):
        if self.shuffle:
            perms = _ptk_data.batch_randperm(self.n_parallel, self.n_samples, device=self.ds.device)
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                idxs = self.idxs.gather(1, perms[:, start:stop])
                batches = self.ds.get_batch(idxs=idxs)
                out = {key: tensor.to(self.output_device) for key, tensor in batches.items()}
                out[ROW_INDEX_KEY] = idxs.to(self.output_device)
                yield out
        else:  # pragma: no cover - the train loader is always shuffled
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                idxs = self.idxs[:, start:stop]
                batches = self.ds.get_batch(idxs=idxs)
                out = {key: tensor.to(self.output_device) for key, tensor in batches.items()}
                out[ROW_INDEX_KEY] = idxs.to(self.output_device)
                yield out


class AdversarialTabNNModule(TabNNModule):
    """TabNNModule with the signed objective L_prediction - lambda * L_adversary."""

    def __init__(self, *, rq4b_settings: _EngineSettings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        spec = rq4b_settings.spec
        metadata = rq4b_settings.metadata
        self.rq4b_lambda = float(spec.adversary_lambda)
        self.rq4b_adversary_seed = int(spec.model_seed)
        self.rq4b_record_batches = bool(spec.record_batches)
        self.rq4b_record_gradients = bool(spec.record_gradient_diagnostics)
        self.rq4b_group_feature_column = spec.alignment_group_feature_column
        self._rq4b_group_codes_np = np.array(metadata.group_codes, dtype=np.int64, copy=True)
        self._rq4b_labels_np = np.array(metadata.labels, dtype=np.int64, copy=True)
        self._rq4b_features_np = np.array(metadata.numeric_reference, dtype=np.float32, copy=True)
        self.rq4b_log: dict[str, Any] = {
            "lambda": self.rq4b_lambda,
            "model_seed": int(spec.model_seed),
            "adversary_seed": self.rq4b_adversary_seed,
            "batches": [],
            "adversary_updates": 0,
            "adversary_skipped_empty_mask": 0,
            "adversary_bypassed_lambda_zero": 0,
            "predictor_updates": 0,
            "gradient_diagnostics": None,
            "devices": None,
            "alignment": {"dataset_rows_verified": 0, "batches_verified": 0, "batches_group_feature_verified": 0},
        }
        self._reset_epoch_counters()

    def _reset_epoch_counters(self) -> None:
        self.rq4b_epoch_counters = {
            "batches": 0,
            "prediction_loss_sum": 0.0,
            "adversary_loss_sum": 0.0,
            "adversary_loss_batches": 0,
            "total_loss_sum": 0.0,
            "roster_rows": 0,
            "adversary_updates": 0,
            "adversary_skipped_empty_mask": 0,
            "adversary_bypassed_lambda_zero": 0,
        }

    def rq4b_close_epoch(self) -> dict[str, Any]:
        counters = self.rq4b_epoch_counters
        batches = max(counters["batches"], 1)
        summary = {
            **counters,
            "mean_prediction_loss": counters["prediction_loss_sum"] / batches,
            "mean_total_loss": counters["total_loss_sum"] / batches,
            "mean_adversary_loss": counters["adversary_loss_sum"] / counters["adversary_loss_batches"]
            if counters["adversary_loss_batches"]
            else None,
        }
        self._reset_epoch_counters()
        return summary

    def compile_model(self, ds, idxs_list: list[SplitIdxs], interface_resources: InterfaceResources) -> None:
        super().compile_model(ds, idxs_list, interface_resources)
        device = self.creator.device_info
        if str(device) != DEVICE:
            raise RQ4bCudaUnavailableError(f"RQ4b predictor is on {device}, not {DEVICE}.")
        if (
            len(self._rq4b_group_codes_np) != ds.n_samples
            or len(self._rq4b_labels_np) != ds.n_samples
            or self._rq4b_features_np.shape != (ds.n_samples, len(NUMERIC_VALUE_COLUMNS))
        ):
            raise RQ4bAlignmentError("RQ4b metadata length differs from the backend dataset.")
        raw = ds.to(device)
        external = torch.as_tensor(self._rq4b_features_np, device=device)
        if raw.tensors["x_cont"].shape != external.shape or not torch.equal(raw.tensors["x_cont"], external):
            raise RQ4bAlignmentError("Backend dataset rows differ from the row-aligned numeric reference.")
        if not torch.equal(raw.tensors["y"][:, 0].cpu(), torch.as_tensor(self._rq4b_labels_np)):
            raise RQ4bAlignmentError("Backend dataset labels differ from the row-aligned labels.")
        if self.rq4b_group_feature_column is not None and not torch.equal(
            raw.tensors["x_cont"][:, self.rq4b_group_feature_column].to(torch.int64).cpu(),
            torch.as_tensor(self._rq4b_group_codes_np),
        ):
            raise RQ4bAlignmentError("Group metadata differs from the group-encoding verification feature.")
        self.rq4b_log["alignment"]["dataset_rows_verified"] = int(ds.n_samples)
        object.__setattr__(self, "_rq4b_raw_ds", raw)
        self.train_dl = _RowIndexedTrainLoader.from_loader(self.train_dl)
        # Plain attributes (not registered submodules/parameters): the predictor
        # module, its state_dict and its optimizer stay unchanged.
        object.__setattr__(self, "_rq4b_groups", torch.as_tensor(self._rq4b_group_codes_np, device=device))
        object.__setattr__(self, "_rq4b_labels", torch.as_tensor(self._rq4b_labels_np, device=device))
        adversary, isolation = build_adversary(self.rq4b_adversary_seed, str(device))
        object.__setattr__(self, "rq4b_adversary", adversary)
        object.__setattr__(
            self,
            "_rq4b_adversary_opt",
            torch.optim.Adam(adversary.parameters(), lr=ADVERSARY_LR, weight_decay=ADVERSARY_WEIGHT_DECAY),
        )
        self.rq4b_log["adversary_rng_isolation"] = isolation
        self.rq4b_log["adversary_optimizer"] = {"name": "adam", "lr": ADVERSARY_LR, "weight_decay": ADVERSARY_WEIGHT_DECAY}

    def manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Refuse nonfinite objectives before backward and nonfinite gradients before the step."""
        if not bool(torch.isfinite(loss).all().item()):
            raise RQ4bNonFiniteError("Predictor objective is not finite; backward and optimizer step refused.")
        super().manual_backward(loss, *args, **kwargs)
        if not _all_finite([p.grad for p in self.model.parameters()]):
            raise RQ4bNonFiniteError("Predictor gradients are not finite; optimizer step refused.")

    # Factored so negative-path tests can inject nonfinite losses.
    def _prediction_loss(self, output: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.criterion(output["x_cont"], output["y"]).sum()

    def _adversary_loss(self, margin: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        features = torch.stack((margin, labels.to(margin.dtype)), dim=1)
        return F.cross_entropy(self.rq4b_adversary(features), groups)

    def _align(self, batch: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if ROW_INDEX_KEY not in batch:
            raise RQ4bAlignmentError("Train batch lacks RQ4b row indices.")
        rows = batch[ROW_INDEX_KEY]
        core = {key: value for key, value in batch.items() if key != ROW_INDEX_KEY}
        batch_labels = core["y"]
        if rows.shape != batch_labels.shape[:-1] or rows.shape[0] != 1:
            raise RQ4bAlignmentError("Row-index and label batch shapes differ.")
        if not torch.equal(self._rq4b_labels[rows], batch_labels[..., 0]):
            raise RQ4bAlignmentError("Shuffled batch labels differ from aligned row metadata.")
        flat = rows.reshape(-1)
        with torch.no_grad():
            expected = self.creator.static_model(self._rq4b_raw_ds.get_sub_dataset(flat))
        if set(expected.tensors) != set(core) or any(not torch.equal(expected.tensors[key], core[key][0]) for key in core):
            raise RQ4bAlignmentError("Shuffled batch features differ from the indexed dataset rows.")
        self.rq4b_log["alignment"]["batches_verified"] += 1
        if self.rq4b_group_feature_column is not None:
            raw_x = self._rq4b_raw_ds.tensors["x_cont"][flat]
            if not torch.equal(raw_x[:, self.rq4b_group_feature_column].to(torch.int64), self._rq4b_groups[flat]):
                raise RQ4bAlignmentError("Batch group metadata differs from the group-encoding verification feature.")
            self.rq4b_log["alignment"]["batches_group_feature_verified"] += 1
        return core, flat

    def _record_devices(self, batch: Mapping[str, torch.Tensor]) -> None:
        devices = {
            **{f"batch_{key}": str(value.device) for key, value in batch.items()},
            "group_metadata": str(self._rq4b_groups.device),
            "predictor_parameters": sorted({str(p.device) for p in self.model.parameters()}),
            "adversary_parameters": sorted({str(p.device) for p in self.rq4b_adversary.parameters()}),
        }
        if any(value != DEVICE for value in devices.values() if isinstance(value, str)) or any(
            devices[key] != [DEVICE] for key in ("predictor_parameters", "adversary_parameters")
        ):
            raise RQ4bCudaUnavailableError(f"RQ4b training tensors are not all on {DEVICE}: {devices}")
        self.rq4b_log["devices"] = devices

    def training_step(self, batch, batch_idx):
        core, rows = self._align(batch)
        if self.rq4b_log["devices"] is None:
            self._record_devices(batch)
        groups = self._rq4b_groups[rows]
        labels = self._rq4b_labels[rows]
        mask = groups != ri.OUTSIDE_ROSTER_CODE
        n_roster = int(mask.sum().item())
        counters = self.rq4b_epoch_counters
        counters["batches"] += 1
        counters["roster_rows"] += n_roster
        record: dict[str, Any] | None = None
        if self.rq4b_record_batches:
            record = {
                "epoch": int(self.trainer.current_epoch) + 1,
                "batch_idx": int(batch_idx),
                "rows": rows.detach().cpu().tolist(),
                "groups": groups.detach().cpu().tolist(),
                "roster_rows": n_roster,
            }
        if self.rq4b_lambda == 0.0:
            # Preserved recipe: the unmodified PyTabKit step on the stripped batch.
            loss = TabNNModule.training_step(self, core, batch_idx)
            value = float(loss.detach().item())
            self.rq4b_log["adversary_bypassed_lambda_zero"] += 1
            self.rq4b_log["predictor_updates"] += 1
            counters["adversary_bypassed_lambda_zero"] += 1
            counters["prediction_loss_sum"] += value
            counters["total_loss_sum"] += value
            if record is not None:
                record.update(adversary="bypassed_lambda_zero", prediction_loss=value)
                self.rq4b_log["batches"].append(record)
            return loss

        output = self.model(core)
        opt = self.optimizers()
        logits = output["x_cont"]
        prediction_loss = self._prediction_loss(output)
        if not bool(torch.isfinite(prediction_loss).item()):
            raise RQ4bNonFiniteError("Prediction loss is not finite.")
        margin = approved_minus_denied_margin(logits)
        adversary_params = list(self.rq4b_adversary.parameters())
        # The first batch has lr=0 under coslog4 (t=0); diagnose from the second batch on.
        diagnostics_wanted = (
            self.rq4b_record_gradients and self.rq4b_log["gradient_diagnostics"] is None and self.rq4b_log["predictor_updates"] >= 1
        )
        if diagnostics_wanted:
            predictor_before = _snapshot(list(self.model.parameters()))
            adversary_before = _snapshot(adversary_params)
        adversary_loss = None
        if n_roster > 0:
            detached_loss = self._adversary_loss(margin.detach()[mask], labels[mask], groups[mask])
            if not bool(torch.isfinite(detached_loss).item()):
                raise RQ4bNonFiniteError("Adversary loss on detached scores is not finite.")
            self._rq4b_adversary_opt.zero_grad()
            # Plain backward: predictor callbacks (L1/L2, hyper updates) must not fire here.
            detached_loss.backward()
            if not _all_finite([p.grad for p in adversary_params]):
                raise RQ4bNonFiniteError("Adversary gradients are not finite; adversary step refused.")
            self._rq4b_adversary_opt.step()
            self.rq4b_log["adversary_updates"] += 1
            counters["adversary_updates"] += 1
            if diagnostics_wanted:
                predictor_after_adv = _snapshot(list(self.model.parameters()))
                adversary_after_adv = _snapshot(adversary_params)
                predictor_grads_after_adv = [p.grad is None or float(p.grad.abs().max()) == 0.0 for p in self.model.parameters()]
            for parameter in adversary_params:
                parameter.requires_grad_(False)
            try:
                adversary_loss = self._adversary_loss(margin[mask], labels[mask], groups[mask])
                if not bool(torch.isfinite(adversary_loss).item()):
                    raise RQ4bNonFiniteError("Adversary loss for the predictor update is not finite.")
                total = prediction_loss - self.rq4b_lambda * adversary_loss
                if not bool(torch.isfinite(total).item()):
                    raise RQ4bNonFiniteError("Signed predictor objective is not finite.")
                if diagnostics_wanted:
                    diag = self._gradient_diagnostics(logits, margin, mask, prediction_loss, adversary_loss, total)
                    adversary_grads_before_predictor = [p.grad.detach().clone() for p in adversary_params]
                self.manual_backward(total)
                opt.step(loss=total)
                opt.zero_grad()
            finally:
                for parameter in adversary_params:
                    parameter.requires_grad_(True)
            counters["adversary_loss_sum"] += float(adversary_loss.detach().item())
            counters["adversary_loss_batches"] += 1
            if record is not None:
                record.update(adversary="updated", adversary_loss_detached=float(detached_loss.item()), adversary_loss=float(adversary_loss.detach().item()))
        else:
            total = prediction_loss
            self.rq4b_log["adversary_skipped_empty_mask"] += 1
            counters["adversary_skipped_empty_mask"] += 1
            if record is not None:
                record.update(adversary="skipped_empty_mask")
            self.manual_backward(total)
            opt.step(loss=total)
            opt.zero_grad()
        self.rq4b_log["predictor_updates"] += 1
        if diagnostics_wanted and adversary_loss is not None:
            diag.update(
                diagnosed_predictor_update=self.rq4b_log["predictor_updates"],
                predictor_lr_at_diagnosed_batch=float(self.optimizers(use_pl_optimizer=False).get_hyper_values("lr", 0, use_hyper_factor=False)),
                predictor_unchanged_by_adversary_step=_max_change(predictor_before, predictor_after_adv) == 0.0,
                predictor_grads_empty_after_adversary_backward=all(predictor_grads_after_adv),
                adversary_max_change_by_adversary_step=_max_change(adversary_before, adversary_after_adv),
                adversary_unchanged_by_predictor_step=_max_change(adversary_after_adv, adversary_params) == 0.0,
                adversary_grads_unchanged_by_predictor_backward=all(
                    torch.equal(old, p.grad) for old, p in zip(adversary_grads_before_predictor, adversary_params)
                ),
                predictor_max_change_by_predictor_step=_max_change(predictor_after_adv, list(self.model.parameters())),
                adversary_params_in_predictor_optimizer=any(
                    p is q for group in opt.param_groups for p in group["params"] for q in adversary_params
                ),
            )
            self.rq4b_log["gradient_diagnostics"] = diag
        prediction_value = float(prediction_loss.detach().item())
        counters["prediction_loss_sum"] += prediction_value
        counters["total_loss_sum"] += float(total.detach().item())
        if record is not None:
            record.update(prediction_loss=prediction_value, total_loss=float(total.detach().item()))
            self.rq4b_log["batches"].append(record)
        self.progress.total_samples += core["y"].shape[-2]
        self.progress.epoch_float = self.progress.total_samples / self.train_dl.get_num_iterated_samples()
        return total

    def _gradient_diagnostics(self, logits, margin, mask, prediction_loss, adversary_loss, total) -> dict[str, Any]:
        lam = self.rq4b_lambda
        grad_pred = torch.autograd.grad(prediction_loss, logits, retain_graph=True)[0].reshape(-1, 2)
        grad_adv = torch.autograd.grad(adversary_loss, logits, retain_graph=True)[0].reshape(-1, 2)
        grad_total = torch.autograd.grad(total, logits, retain_graph=True)[0].reshape(-1, 2)
        grad_adv_margin = torch.autograd.grad(adversary_loss, margin, retain_graph=True)[0]
        outside = ~mask
        signed = grad_pred - lam * grad_adv
        additive = grad_pred + lam * grad_adv
        return {
            "score_tensor": "output['x_cont'] RealMLP logits",
            "logits_shape": list(logits.shape),
            "margin_shape": list(margin.shape),
            "adversary_input_shape": [int(mask.sum().item()), ADVERSARY_INPUTS],
            "adversary_output_shape": [int(mask.sum().item()), ADVERSARY_OUTPUTS],
            "batch_rows": int(mask.numel()),
            "roster_rows": int(mask.sum().item()),
            "outside_rows": int(outside.sum().item()),
            "lambda": lam,
            "signed_identity_holds": bool(torch.allclose(grad_total, signed, rtol=GRADIENT_IDENTITY_RTOL, atol=GRADIENT_IDENTITY_ATOL)),
            "signed_identity_max_abs_diff": float((grad_total - signed).abs().max().item()),
            "additive_variant_max_abs_diff": float((grad_total - additive).abs().max().item()),
            "adversary_grad_outside_roster_max_abs": float(grad_adv[outside].abs().max().item()) if outside.any() else None,
            "adversary_margin_grad_outside_roster_max_abs": float(grad_adv_margin[outside].abs().max().item()) if outside.any() else None,
            "adversary_grad_roster_nonzero_rows": int((grad_adv[mask].abs().sum(dim=1) > 0).sum().item()),
            "prediction_grad_outside_roster_nonzero_rows": int((grad_pred[outside].abs().sum(dim=1) > 0).sum().item()),
            "adversary_logit1_grad_nonzero_roster_rows": int((grad_adv[mask, 1] != 0).sum().item()),
            "total_minus_pred_opposes_adv_roster_rows": int(
                (torch.sign(grad_total[mask, 1] - grad_pred[mask, 1]) == -torch.sign(grad_adv[mask, 1]))
                .logical_and(grad_adv[mask, 1] != 0)
                .sum()
                .item()
            ),
        }

    def on_fit_end(self) -> None:
        super().on_fit_end()
        # Mirror PyTabKit: never pickle training data, metadata or optimizer state.
        for name in ("_rq4b_groups", "_rq4b_labels", "_rq4b_adversary_opt", "_rq4b_raw_ds"):
            if name in self.__dict__:
                object.__delattr__(self, name)
        self._rq4b_group_codes_np = None
        self._rq4b_labels_np = None
        self._rq4b_features_np = None
        self.rq4b_epoch_counters = None


class _AdversarialNNAlgInterface(NNAlgInterface):
    """Pinned NNAlgInterface.fit path with the RQ4b module, budget and horizon split."""

    def __init__(self, *, rq4b_settings: _EngineSettings, **config: Any) -> None:
        super().__init__(**config)
        self.rq4b_settings = rq4b_settings

    def fit(self, ds, idxs_list: list[SplitIdxs], interface_resources: InterfaceResources, logger: Logger, tmp_folders, name: str) -> None:
        if version("pytabkit") != PYTABKIT_VERSION:
            raise RQ4bTrainingError(f"RQ4b extension is locked to pytabkit=={PYTABKIT_VERSION}.")
        if self.config.get("random_seed_offset", 0) != 0 or self.config.get("same_seed_for_sub_splits", False):
            raise RQ4bTrainingError("RQ4b does not support seed-offset configuration.")
        if len(idxs_list) != 1 or idxs_list[0].train_idxs.shape[0] != 1:
            raise RQ4bTrainingError("RQ4b expects exactly one train split without internal CV.")
        settings = self.rq4b_settings
        spec = settings.spec
        log = logging.getLogger("lightning")
        log.propagate = False
        log.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`.")
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            if list(interface_resources.gpu_devices) != [DEVICE]:
                raise RQ4bCudaUnavailableError(f"RQ4b requires gpu_devices == ['{DEVICE}'].")
            self.device = interface_resources.gpu_devices[0]
            ds = ds.to(self.device)
            if self.config.get("n_epochs") != spec.execution_epochs or "stop_epoch" in self.config or self.fit_params is not None:
                raise RQ4bBudgetError("Backend epoch configuration differs from the RQ4b execution budget.")
            # Schedule horizon (module/creator n_epochs) is separated from the
            # trainer execution budget (max_epochs); schedules are never rescaled.
            module_kwargs = utils.join_dicts({"n_epochs": SCHEDULE_HORIZON_EPOCHS, "logger": logger}, self.config)
            module_kwargs["n_epochs"] = spec.schedule_horizon_epochs
            if spec.module_kind == "adversarial":
                self.model = AdversarialTabNNModule(rq4b_settings=settings, fit_params=None, **module_kwargs)
            else:
                self.model = TabNNModule(fit_params=None, **module_kwargs)
            self.model.compile_model(ds, idxs_list, interface_resources)
            accelerator, devices = get_lignting_accel_and_devices(self.device)
            max_time = None if interface_resources.time_in_seconds is None else timedelta(seconds=interface_resources.time_in_seconds)
            self.min_trainer_kwargs = dict(
                max_time=max_time,
                accelerator=accelerator,
                devices=devices,
                max_epochs=spec.execution_epochs,
                enable_checkpointing=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                enable_model_summary=False,
                log_every_n_steps=1,
            )
            callbacks = [*self.model.create_callbacks(), settings.recorder]
            trainer = pl.Trainer(**self.min_trainer_kwargs, callbacks=callbacks, logger=pl.loggers.logger.DummyLogger())
            trainer.fit(model=self.model, train_dataloaders=self.model.train_dl, val_dataloaders=self.model.val_dl)
            if settings.recorder.completed_epochs != spec.execution_epochs:
                raise RQ4bBudgetError(
                    f"RQ4b executed {settings.recorder.completed_epochs} epochs, expected {spec.execution_epochs}."
                )
            if hasattr(self.model, "fit_params"):
                self.fit_params = self.model.fit_params
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
            if self.model is not None and hasattr(self.model, "_trainer"):
                del self.model._trainer
            settings.recorder.release()
            # Drop training metadata and the live recorder from the pickled interface.
            self.rq4b_settings = None


class _AdversarialRealMLPBackend(RealMLP_TD_Classifier):
    rq4b_settings: _EngineSettings

    def _create_alg_interface(self, n_cv: int):
        if not hasattr(self, "rq4b_settings"):
            raise RQ4bTrainingError("RQ4b engine settings were not attached.")
        settings = self.rq4b_settings
        del self.rq4b_settings
        # x_converter_ was fitted on the training frame only, before this call.
        settings.recorder.bind_converter(self.x_converter_)
        return _AdversarialNNAlgInterface(rq4b_settings=settings, **self.get_config())


class AdversarialRealMLPClassifier(RealMLPClassifier):
    """Project wrapper routing fit through the RQ4b interface and module."""

    def attach_rq4b(self, settings: _EngineSettings) -> "AdversarialRealMLPClassifier":
        self._rq4b_settings = settings
        return self

    def fit(self, X, y, *, X_val=None, y_val=None):
        if X_val is not None or y_val is not None:
            raise RQ4bTrainingError("RQ4b prohibits the backend internal validation channel.")
        settings = getattr(self, "_rq4b_settings", None)
        if settings is None:
            raise RQ4bTrainingError("RQ4b engine settings were not attached.")
        if len(settings.metadata.group_codes) != len(X) or len(settings.metadata.labels) != len(X):
            raise RQ4bAlignmentError("RQ4b metadata length differs from X.")
        try:
            result = super().fit(X, y)
        finally:
            self.__dict__.pop("_rq4b_settings", None)
        module = self.backend_.alg_interface_.model
        self.rq4b_log_ = copy.deepcopy(getattr(module, "rq4b_log", None))
        self.rq4b_history_ = settings.recorder.result()
        return result

    def _validate_requested_device(self) -> None:
        if self.device != DEVICE:
            raise RQ4bCudaUnavailableError(f"RQ4b requires device={DEVICE}.")
        require_cuda0()

    def _make_backend(self):
        self._backend_class()
        backend = _AdversarialRealMLPBackend(**self.backend_parameters())
        backend.rq4b_settings = self._rq4b_settings
        return backend


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def estimator_parameters(model_seed: int, execution_epochs: int) -> dict[str, Any]:
    """Locked realmlp_v1 parameters with only seed, cuda:0 and execution budget overridden."""
    payload = load_realmlp_v1_config()
    params = {**payload["estimator_parameters"], **payload["project_policy"]}
    params.update(device=DEVICE, random_state=int(model_seed), n_epochs=int(execution_epochs))
    if params["batch_size"] != BATCH_SIZE or params["predict_batch_size"] != PREDICT_BATCH_SIZE:
        raise RQ4bBudgetError("RealMLP batch sizes differ from the RQ4b runtime contract.")
    return params


def input_digest(X: pd.DataFrame, metadata: RowMetadata) -> dict[str, str]:
    return {
        "X": hashlib.sha256(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()).hexdigest(),
        "labels": hashlib.sha256(np.ascontiguousarray(metadata.labels, dtype=np.int64).tobytes()).hexdigest(),
        "record_ids": hashlib.sha256("\n".join(metadata.record_ids).encode()).hexdigest(),
        "group_codes": hashlib.sha256(np.ascontiguousarray(metadata.group_codes, dtype=np.int64).tobytes()).hexdigest(),
    }


@dataclass
class FitResult:
    estimator: AdversarialRealMLPClassifier
    spec: FitSpec
    history: dict[str, Any]
    log: dict[str, Any] | None
    digest: dict[str, str]
    evidence_kind: str


def _fit_engine(
    inputs: ri.RQ4bPredictorInput,
    spec: FitSpec,
    *,
    evidence_kind: str,
    fold_validation: FoldValidation | None = None,
    metadata: RowMetadata | None = None,
    progress: DurableProgress | None = None,
    phase: str = "fit",
    fold_index: int | None = None,
) -> FitResult:
    ri.validate_native_predictor_frame(inputs.X)
    metadata = RowMetadata.from_input(inputs) if metadata is None else metadata
    if metadata.record_ids != tuple(inputs.record_ids) or not np.array_equal(metadata.labels, np.asarray(inputs.y, dtype=np.int64)):
        raise RQ4bAlignmentError("Row metadata identifiers or labels differ from the predictor input.")
    if fold_validation is not None:
        ri.validate_native_predictor_frame(fold_validation.X)
        if set(fold_validation.record_ids) & set(metadata.record_ids):
            raise RQ4bAlignmentError("Fold-validation rows overlap the fitted rows.")
        if len(fold_validation.y) != len(fold_validation.X) or len(fold_validation.record_ids) != len(fold_validation.X):
            raise RQ4bAlignmentError("Fold-validation features, labels and IDs are misaligned.")
    require_cuda0()
    recorder = EpochRecorder(spec, fold_validation, progress=progress, phase=phase, fold=fold_index)
    settings = _EngineSettings(spec=spec, metadata=metadata, recorder=recorder)
    estimator = AdversarialRealMLPClassifier(**estimator_parameters(spec.model_seed, spec.execution_epochs)).attach_rq4b(settings)
    digest = input_digest(inputs.X, metadata)
    estimator.fit(inputs.X, np.asarray(inputs.y, dtype=np.int64))
    history = estimator.rq4b_history_
    if history["completed_epochs"] != spec.execution_epochs:
        raise RQ4bBudgetError("Completed epochs differ from the execution budget.")
    return FitResult(estimator, spec, history, estimator.rq4b_log_, digest, evidence_kind)


def fit_smoke(
    inputs: ri.RQ4bPredictorInput,
    *,
    adversary_lambda: float,
    model_seed: int,
    execution_epochs: int = SMOKE_MAX_EPOCHS,
    schedule_horizon_epochs: int = SCHEDULE_HORIZON_EPOCHS,
    module_kind: str = "adversarial",
    fold_validation: FoldValidation | None = None,
    metadata: RowMetadata | None = None,
    record_batches: bool = True,
    record_gradient_diagnostics: bool = False,
    alignment_group_feature_column: int | None = None,
) -> FitResult:
    """Synthetic GPU smoke; the only route allowed to execute at most two epochs."""
    if execution_epochs > SMOKE_MAX_EPOCHS:
        raise RQ4bBudgetError("Smoke fits are limited to two executed epochs.")
    spec = FitSpec(
        adversary_lambda=adversary_lambda,
        model_seed=model_seed,
        execution_epochs=execution_epochs,
        schedule_horizon_epochs=schedule_horizon_epochs,
        module_kind=module_kind,
        record_batches=record_batches,
        record_gradient_diagnostics=record_gradient_diagnostics,
        alignment_group_feature_column=alignment_group_feature_column,
    )
    return _fit_engine(inputs, spec, evidence_kind=EVIDENCE_SMOKE, fold_validation=fold_validation, metadata=metadata)


def fit_preserved_recipe_smoke(inputs: ri.RQ4bPredictorInput, *, model_seed: int, execution_epochs: int = SMOKE_MAX_EPOCHS) -> RealMLPClassifier:
    """Unmodified project wrapper and backend (schedule horizon == execution), for lambda=0 equivalence."""
    if execution_epochs > SMOKE_MAX_EPOCHS:
        raise RQ4bBudgetError("Smoke fits are limited to two executed epochs.")
    require_cuda0()
    estimator = RealMLPClassifier(**estimator_parameters(model_seed, execution_epochs))
    estimator.fit(inputs.X, np.asarray(inputs.y, dtype=np.int64))
    return estimator


def _require_production_spec(context: ProductionContext, adversary_lambda: float, model_seed: int) -> None:
    if not isinstance(context, ProductionContext):
        raise RQ4bPrerequisiteError("Production fits require a ProductionContext.")
    require_prerequisites(context.prerequisites)
    _require_runtime_contract(context.contract)
    if float(adversary_lambda) not in context.contract.lambda_grid:
        raise RQ4bTrainingError(f"Lambda {adversary_lambda} is not in the locked grid.")
    if model_seed not in context.contract.final_seeds:
        raise RQ4bTrainingError(f"Seed {model_seed} is not a locked RQ4b seed.")


@dataclass
class FoldFitResult:
    fold_index: int
    fold_validation_ce: list[float]
    executed_epochs: int
    schedule_horizon_epochs: int
    evidence_kind: str
    history: dict[str, Any] = field(repr=False, default_factory=dict)
    log: dict[str, Any] | None = field(repr=False, default=None)


def fit_cv_fold(
    context: ProductionContext,
    fold_train: ri.RQ4bPredictorInput,
    fold_validation: FoldValidation,
    *,
    adversary_lambda: float,
    model_seed: int,
    fold_index: int,
    progress: DurableProgress | None = None,
) -> FoldFitResult:
    """Production fold fit: all 256 epochs, horizon 256, fold CE after every epoch."""
    _require_production_spec(context, adversary_lambda, model_seed)
    spec = FitSpec(
        adversary_lambda=float(adversary_lambda),
        model_seed=model_seed,
        execution_epochs=SCHEDULE_HORIZON_EPOCHS,
        schedule_horizon_epochs=SCHEDULE_HORIZON_EPOCHS,
    )
    result = _fit_engine(
        fold_train, spec, evidence_kind=EVIDENCE_OBSERVED, fold_validation=fold_validation,
        progress=progress, phase="cv", fold_index=fold_index,
    )
    return FoldFitResult(
        fold_index=fold_index,
        fold_validation_ce=list(result.history["fold_validation_ce"]),
        executed_epochs=result.history["completed_epochs"],
        schedule_horizon_epochs=result.history["budget"]["module_schedule_horizon_epochs"],
        evidence_kind=EVIDENCE_OBSERVED,
        history={key: value for key, value in result.history.items() if key not in ("batch_fingerprints", "batch_schedules")},
        log=result.log,
    )


# ---------------------------------------------------------------------------
# Train-only folds, epoch selection and orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainPopulation:
    """Quality-gated raw Train rows with aligned labels, IDs and derived race."""

    X_raw: pd.DataFrame
    y: np.ndarray
    record_ids: tuple[str, ...]
    derived_race: np.ndarray

    def __post_init__(self) -> None:
        ri.validate_raw_predictor_frame(self.X_raw)
        ids = ri.validate_record_ids(self.record_ids, "Train population")
        labels = ri.validate_binary_labels(self.y, "Train population", rows=len(self.X_raw))
        if len(ids) != len(self.X_raw) or len(self.derived_race) != len(self.X_raw) or len(labels) != len(ids):
            raise RQ4bAlignmentError("Train population columns are misaligned.")


def make_cv_folds(record_ids: Sequence[str], y: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Locked Train-only StratifiedKFold(5, shuffle=True, random_state=42) positions."""
    labels = np.asarray(y, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = [(np.asarray(train), np.asarray(val)) for train, val in splitter.split(np.zeros(len(labels)), labels)]
    _check_fold_partition(folds, len(labels))
    return folds


def _check_fold_partition(folds: Sequence[tuple[np.ndarray, np.ndarray]], rows: int) -> None:
    if len(folds) != CV_FOLDS:
        raise RQ4bEpochSelectionError("CV must have exactly five folds.")
    counts = np.zeros(rows, dtype=np.int64)
    for train, val in folds:
        if np.intersect1d(train, val).size or len(train) + len(val) != rows:
            raise RQ4bEpochSelectionError("Fold-train and fold-validation must partition Train.")
        counts[val] += 1
    if not np.all(counts == 1):
        raise RQ4bEpochSelectionError("Every Train row must be in exactly one fold-validation set.")


def fold_membership_digest(record_ids: Sequence[str], folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    ids = np.asarray(record_ids, dtype=object)
    digest = hashlib.sha256()
    for index, (_, val) in enumerate(folds, start=1):
        digest.update(f"fold{index}:".encode() + "\n".join(ids[np.sort(val)]).encode() + b"|")
    return digest.hexdigest()


def select_epoch(curves: np.ndarray | Sequence[Sequence[float]]) -> dict[str, Any]:
    """Equal-weight fold mean CE; minimum unrounded mean; exact ties choose the earliest epoch."""
    try:
        matrix = np.asarray(curves, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RQ4bEpochSelectionError("CV curves must be numeric.") from exc
    if matrix.shape != (CV_FOLDS, SCHEDULE_HORIZON_EPOCHS):
        raise RQ4bEpochSelectionError(f"CV curves must have shape ({CV_FOLDS}, {SCHEDULE_HORIZON_EPOCHS}); got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise RQ4bEpochSelectionError("CV curves contain missing or nonfinite values.")
    mean_curve = matrix.mean(axis=0)
    minimum = float(mean_curve.min())
    tied = np.flatnonzero(mean_curve == minimum)
    selected = int(tied[0]) + 1
    return {
        "selected_epoch": selected,
        "minimum_mean_ce": minimum,
        "tied_epochs": (tied + 1).tolist(),
        "epoch_cap_hit": selected == SCHEDULE_HORIZON_EPOCHS,
        "finite_values": int(matrix.size),
        "mean_curve": mean_curve.tolist(),
    }


@dataclass(frozen=True)
class EpochSelectionRecord:
    adversary_lambda: float
    model_seed: int
    curves: tuple[tuple[float, ...], ...]
    selection: Mapping[str, Any]
    fold_membership_sha256: str
    evidence_kind: str
    fold_summaries: tuple[Mapping[str, Any], ...] = field(repr=False, default=())
    fold_outputs: tuple[Mapping[str, Any], ...] = field(repr=False, default=())

    @property
    def selected_epoch(self) -> int:
        return int(self.selection["selected_epoch"])


FoldFitter = Callable[..., FoldFitResult]


def _prepare_fold(access: ri.StageAccess, population: TrainPopulation, train: np.ndarray, val: np.ndarray, contract: ri.RQ4bContract):
    preprocessor = ri.make_fit_population_preprocessor(access)
    X_train = population.X_raw.iloc[train].reset_index(drop=True)
    X_val = population.X_raw.iloc[val].reset_index(drop=True)
    native_train = preprocessor.fit_transform(X_train)
    native_val = preprocessor.transform(X_val)  # transform only: no state is fitted on fold-validation
    ids = np.asarray(population.record_ids, dtype=object)
    labels = np.asarray(population.y, dtype=np.int64)
    fold_train = ri.build_predictor_input(native_train, labels[train], ids[train].tolist(), population.derived_race[train].tolist(), contract)
    fold_validation = FoldValidation(native_val, labels[val], tuple(ids[val].tolist()))
    return preprocessor, fold_train, fold_validation


def run_cv_epoch_selection(
    context: ProductionContext,
    access: ri.StageAccess,
    population: TrainPopulation,
    *,
    adversary_lambda: float,
    model_seed: int,
    fold_fitter: FoldFitter | None = None,
    progress: DurableProgress | None = None,
) -> EpochSelectionRecord:
    """Five fresh fold fits (fresh fold-local preprocessing each), 5 x 256 curves, exact selection.

    ``fold_fitter`` defaults to ``fit_cv_fold``. A replacement must return all
    256 values per fold; its ``evidence_kind`` is carried into the record so
    test-double curves can never be mistaken for observed production curves.
    """
    if access.stage != "cv_epoch_selection":
        raise ri.RQ4bAccessError(f"CV epoch selection may not run in stage {access.stage!r}.")
    _require_production_spec(context, adversary_lambda, model_seed)
    fitter = fit_cv_fold if fold_fitter is None else fold_fitter
    folds = make_cv_folds(population.record_ids, population.y)
    curves, summaries, outputs, kinds = [], [], [], set()
    for index, (train, val) in enumerate(folds, start=1):
        preprocessor, fold_train, fold_validation = _prepare_fold(access, population, train, val, context.contract)
        if progress is not None:
            progress.phase_started("cv", index)
        try:
            kwargs = dict(adversary_lambda=float(adversary_lambda), model_seed=model_seed, fold_index=index)
            if fold_fitter is None:
                kwargs["progress"] = progress
            result = fitter(context, fold_train, fold_validation, **kwargs)
        except BaseException:
            if progress is not None:
                progress.phase_failed("cv", index)
            raise
        if result.fold_index != index or result.executed_epochs != SCHEDULE_HORIZON_EPOCHS or result.schedule_horizon_epochs != SCHEDULE_HORIZON_EPOCHS:
            raise RQ4bBudgetError(f"Fold {index} did not execute the full 256-epoch production budget.")
        if len(result.fold_validation_ce) != SCHEDULE_HORIZON_EPOCHS:
            raise RQ4bEpochSelectionError(f"Fold {index} returned {len(result.fold_validation_ce)} CE values, expected 256.")
        curves.append(tuple(float(value) for value in result.fold_validation_ce))
        kinds.add(result.evidence_kind)
        outputs.append({
            "fold_index": index, "fold_validation_ce": list(result.fold_validation_ce),
            "history": result.history, "log": result.log, "evidence_kind": result.evidence_kind,
        })
        summaries.append(
            {
                "fold_index": index,
                "fold_train_rows": int(len(train)),
                "fold_validation_rows": int(len(val)),
                "preprocessor_fit_rows": int(sum(preprocessor.named_steps["native"].fold_train_category_counts_["loan_type"].values())),
                "evidence_kind": result.evidence_kind,
            }
        )
        if progress is not None:
            progress.phase_completed("cv", index)
    if len(kinds) != 1:
        raise RQ4bEpochSelectionError(f"Fold evidence kinds differ: {sorted(kinds)}.")
    selection = select_epoch(curves)
    linked_outputs = tuple({**output, "selected_epoch": selection["selected_epoch"],
                            "curve_matches_history": output["history"].get("fold_validation_ce") == output["fold_validation_ce"]}
                           for output in outputs)
    return EpochSelectionRecord(
        adversary_lambda=float(adversary_lambda),
        model_seed=model_seed,
        curves=tuple(curves),
        selection=selection,
        fold_membership_sha256=fold_membership_digest(population.record_ids, folds),
        evidence_kind=kinds.pop(),
        fold_summaries=tuple(summaries),
        fold_outputs=linked_outputs,
    )


@dataclass
class RefitResult:
    estimator: Any
    preprocessor: Any
    selected_epoch: int
    epoch_cap_hit: bool
    schedule_horizon_epochs: int
    evidence_kind: str
    history: dict[str, Any] = field(repr=False, default_factory=dict)
    log: dict[str, Any] | None = field(repr=False, default=None)


RefitFitter = Callable[..., FitResult]


def _validate_selection_record(record: EpochSelectionRecord, adversary_lambda: float, model_seed: int) -> dict[str, Any]:
    if record.adversary_lambda != float(adversary_lambda) or record.model_seed != model_seed:
        raise RQ4bEpochSelectionError("Selection record belongs to another lambda/seed.")
    recomputed = select_epoch(record.curves)
    if recomputed["selected_epoch"] != record.selected_epoch or recomputed["epoch_cap_hit"] != record.selection.get("epoch_cap_hit"):
        raise RQ4bEpochSelectionError("Selection record is inconsistent with its curves.")
    return recomputed


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def require_population_lineage(record: EpochSelectionRecord, population: TrainPopulation) -> str:
    """The selection's fold-membership digest must equal the locked folds of this population."""
    stored = record.fold_membership_sha256
    if not isinstance(stored, str) or _SHA256_PATTERN.fullmatch(stored) is None:
        raise RQ4bEpochSelectionError("Selection record lacks a valid fold-membership SHA-256 digest.")
    current = fold_membership_digest(population.record_ids, make_cv_folds(population.record_ids, population.y))
    if stored != current:
        raise RQ4bEpochSelectionError(
            "Selection record fold membership differs from the refit Train population under the locked fold policy."
        )
    return current


def run_full_train_refit(
    context: ProductionContext,
    access: ri.StageAccess,
    population: TrainPopulation,
    selection_record: EpochSelectionRecord,
    *,
    adversary_lambda: float,
    model_seed: int,
    fitter: RefitFitter | None = None,
    progress: DurableProgress | None = None,
) -> RefitResult:
    """Fresh preprocessing/predictor/adversary/optimizers/backend; exactly the selected epoch, horizon 256."""
    if access.stage != "full_train_refit":
        raise ri.RQ4bAccessError(f"Full-Train refit may not run in stage {access.stage!r}.")
    _require_production_spec(context, adversary_lambda, model_seed)
    selection = _validate_selection_record(selection_record, adversary_lambda, model_seed)
    if fitter is None and selection_record.evidence_kind != EVIDENCE_OBSERVED:
        raise RQ4bEpochSelectionError("Production refit requires an observed (not test-double) selection record.")
    require_population_lineage(selection_record, population)
    preprocessor = ri.make_fit_population_preprocessor(access)
    native = preprocessor.fit_transform(population.X_raw.reset_index(drop=True))
    inputs = ri.build_predictor_input(native, population.y, list(population.record_ids), list(population.derived_race), context.contract)
    spec = FitSpec(
        adversary_lambda=float(adversary_lambda),
        model_seed=model_seed,
        execution_epochs=int(selection["selected_epoch"]),
        schedule_horizon_epochs=SCHEDULE_HORIZON_EPOCHS,
    )
    if progress is not None:
        progress.phase_started("refit")
    try:
        if fitter is None:
            result = _fit_engine(inputs, spec, evidence_kind=EVIDENCE_OBSERVED, progress=progress, phase="refit")
        else:
            result = fitter(inputs, spec)
    except BaseException:
        if progress is not None:
            progress.phase_failed("refit")
        raise
    if result.history["completed_epochs"] != spec.execution_epochs or result.history["budget"]["module_schedule_horizon_epochs"] != SCHEDULE_HORIZON_EPOCHS:
        raise RQ4bBudgetError("Refit budget or schedule horizon differs from the selected policy.")
    if progress is not None:
        progress.phase_completed("refit")
    return RefitResult(
        estimator=result.estimator,
        preprocessor=preprocessor,
        selected_epoch=spec.execution_epochs,
        epoch_cap_hit=bool(selection["epoch_cap_hit"]),
        schedule_horizon_epochs=SCHEDULE_HORIZON_EPOCHS,
        evidence_kind=result.evidence_kind,
        history={key: value for key, value in result.history.items() if key not in ("batch_fingerprints", "batch_schedules")},
        log=result.log,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


_TRAINING_ONLY_ATTRIBUTES = ("_rq4b_groups", "_rq4b_labels", "_rq4b_adversary_opt", "_rq4b_raw_ds")
_TRAINING_ONLY_ARRAYS = ("_rq4b_group_codes_np", "_rq4b_labels_np", "_rq4b_features_np")


def _batch_record_logs(estimator: RealMLPClassifier) -> list[dict[str, Any]]:
    module = estimator.backend_.alg_interface_.model
    return [log for log in (getattr(estimator, "rq4b_log_", None), getattr(module, "rq4b_log", None)) if isinstance(log, dict)]


def strip_batch_records(estimator: RealMLPClassifier) -> int:
    """Remove smoke per-batch row/group records (training metadata) before pickling."""
    removed = 0
    for log in _batch_record_logs(estimator):
        removed += len(log.get("batches", []))
        log["batches"] = []
    return removed


def training_metadata_present(estimator: RealMLPClassifier) -> bool:
    module = estimator.backend_.alg_interface_.model
    interface = estimator.backend_.alg_interface_
    return (
        any(name in module.__dict__ for name in _TRAINING_ONLY_ATTRIBUTES)
        or any(getattr(module, name, None) is not None for name in _TRAINING_ONLY_ARRAYS)
        or getattr(interface, "rq4b_settings", None) is not None
        or "_rq4b_settings" in estimator.__dict__
        or hasattr(estimator.backend_, "rq4b_settings")
        or any(log.get("batches") for log in _batch_record_logs(estimator))
    )


def save_and_verify_reload(
    estimator: RealMLPClassifier,
    X: pd.DataFrame,
    path: str | Path,
    *,
    contract: ri.RQ4bContract | None = None,
    loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """joblib round trip; eval-mode probabilities must be exactly equal; no raw metadata pickled.

    Paths must be new and lie under the RQ4b artifact namespace (production) or
    under scratch/rq4b_implementation (smoke); baseline paths are refused.
    """
    import joblib

    path = Path(path).resolve()
    if contract is not None:
        ri.require_rq4b_output_path(path, contract)
    elif SMOKE_OUTPUT_ROOT.resolve() not in path.parents:
        raise RQ4bReloadError(f"Smoke model path {path} is outside {SMOKE_OUTPUT_ROOT}.")
    if path.exists():
        raise RQ4bReloadError(f"Refusing to overwrite serialized RQ4b model {path}.")
    strip_batch_records(estimator)
    if training_metadata_present(estimator):
        raise RQ4bReloadError("Estimator still holds raw training metadata.")
    before = estimator.predict_proba(X)
    joblib.dump(estimator, path)
    try:
        reloaded = (loader or joblib.load)(path)
        after = reloaded.predict_proba(X)
    except Exception as exc:
        raise RQ4bReloadError(f"Serialized RQ4b model failed to reload: {type(exc).__name__}: {exc}") from exc
    if before.shape != after.shape or not np.array_equal(before, after):
        raise RQ4bReloadError("Reloaded RQ4b model changed eval-mode probabilities.")
    module = reloaded.backend_.alg_interface_.model
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": int(len(after)),
        "probabilities_exactly_equal": True,
        "training_metadata_pickled": training_metadata_present(reloaded),
        "adversary_present_after_reload": hasattr(module, "rq4b_adversary"),
        "predictor_training_mode_after_reload": bool(module.model.training),
        "interface_device_after_reload": str(reloaded.backend_.alg_interface_.device),
    }
