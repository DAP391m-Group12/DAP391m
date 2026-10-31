"""Canonical Version-locked PyTabKit history integration for Dataset B RealMLP."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.alg_interfaces.nn_interfaces import (
    NNAlgInterface,
    get_lignting_accel_and_devices,
)
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.training.logging import Logger

from src.dataset_b.models.neural.realmlp.estimator import PYTABKIT_VERSION, RealMLPClassifier


HISTORY_METHOD = "pytorch_lightning_callback_backend_logger"
_VAL_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+val\s+"
    r"(?P<metric>[A-Za-z0-9_\-]+)\s*=\s*(?P<value>[-+0-9.eE]+)"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RealMLPHistoryRecorder:
    """Collect and durably append one truthful record per completed epoch."""

    def __init__(
        self,
        path: Path,
        *,
        device_index: int,
        expected_epochs: int,
        stage: str = "gpu_smoke",
    ) -> None:
        if not stage or any(character.isspace() for character in stage):
            raise ValueError("RealMLP history stage must be a non-empty token.")
        self.path = Path(path)
        self.device_index = device_index
        self.expected_epochs = expected_epochs
        self.stage = stage
        self.records: list[dict[str, Any]] = []
        self._fit_started_at: float | None = None
        self._epoch_started_at: float | None = None
        self._train_loss_total = 0.0
        self._train_sample_count = 0
        self._train_loss: float | None = None
        self._learning_rate: float | None = None
        self._validation_losses: dict[int, float] = {}

    def start_fit(self) -> None:
        if self.path.exists():
            raise RuntimeError(f"RealMLP live history already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fit_started_at = time.perf_counter()

    def start_epoch(self) -> None:
        if self._fit_started_at is None:
            raise RuntimeError("RealMLP history fit timer was not started.")
        self._epoch_started_at = time.perf_counter()
        self._train_loss_total = 0.0
        self._train_sample_count = 0
        self._train_loss = None
        self._learning_rate = None
        torch.cuda.reset_peak_memory_stats(self.device_index)

    def add_train_batch(self, loss: float, sample_count: int) -> None:
        if not math.isfinite(loss) or sample_count <= 0:
            raise RuntimeError("RealMLP produced an invalid training batch loss.")
        self._train_loss_total += loss * sample_count
        self._train_sample_count += sample_count

    def finish_training(self, learning_rate: float) -> None:
        if self._train_sample_count <= 0:
            raise RuntimeError("RealMLP history observed no training batches.")
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise RuntimeError("RealMLP produced an invalid learning rate.")
        self._train_loss = self._train_loss_total / self._train_sample_count
        self._learning_rate = learning_rate

    def record_validation(self, epoch: int, value: float) -> None:
        if epoch <= 0 or not math.isfinite(value):
            raise RuntimeError("RealMLP produced an invalid validation loss.")
        self._validation_losses[epoch] = value

    def complete_epoch(
        self,
        epoch: int,
        best_epoch: int | None,
        *,
        inner_validation_required: bool = True,
    ) -> dict[str, Any]:
        if self._fit_started_at is None or self._epoch_started_at is None:
            raise RuntimeError("RealMLP history epoch timer was not started.")
        if self._train_loss is None or self._learning_rate is None:
            raise RuntimeError("RealMLP history is missing completed training metrics.")
        if inner_validation_required and epoch not in self._validation_losses:
            raise RuntimeError("RealMLP backend logger did not publish validation loss.")
        torch.cuda.synchronize(self.device_index)
        completed_at = time.perf_counter()
        record = {
            "epoch": epoch,
            "train_loss": self._train_loss,
            "inner_val_loss": (
                self._validation_losses.pop(epoch)
                if inner_validation_required
                else None
            ),
            "learning_rate": self._learning_rate,
            "epoch_time_seconds": completed_at - self._epoch_started_at,
            "elapsed_time_seconds": completed_at - self._fit_started_at,
            "gpu_memory_used_mb": torch.cuda.max_memory_allocated(self.device_index)
            / (1024 * 1024),
            "best_epoch_so_far": best_epoch,
            "status": "COMPLETED",
            "timestamp": _utc_now(),
        }
        if not inner_validation_required:
            record.update(
                inner_val_status="NOT_APPLICABLE_FIXED_EPOCH",
                best_epoch_status="NOT_APPLICABLE_FIXED_EPOCH",
            )
        if epoch != len(self.records) + 1 or epoch > self.expected_epochs:
            raise RuntimeError("RealMLP history epoch sequence is invalid.")
        numeric_fields = (
            "train_loss",
            "learning_rate",
            "epoch_time_seconds",
            "elapsed_time_seconds",
            "gpu_memory_used_mb",
        )
        if any(not math.isfinite(float(record[field])) for field in numeric_fields):
            raise RuntimeError("RealMLP history contains a non-finite metric.")
        if inner_validation_required and not math.isfinite(
            float(record["inner_val_loss"])
        ):
            raise RuntimeError("RealMLP history contains a non-finite validation loss.")
        self.records.append(record)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            "REALMLP_HISTORY "
            + json.dumps(
                {"model": "realmlp", "stage": self.stage, **record},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return record


class _HistoryLogger(Logger):
    def __init__(self, delegate: Logger, recorder: RealMLPHistoryRecorder) -> None:
        super().__init__(delegate.get_verbosity_level())
        self.delegate = delegate
        self.recorder = recorder

    def log(self, verbosity: int, content: str) -> None:
        self._record(content)
        self.delegate.log(verbosity, content)

    def force_log(self, content: str) -> None:
        self._record(content)
        self.delegate.force_log(content)

    def _record(self, content: str) -> None:
        for match in _VAL_PATTERN.finditer(content):
            self.recorder.record_validation(
                int(match.group("epoch")), float(match.group("value"))
            )


class _RealMLPHistoryCallback(Callback):
    def __init__(
        self,
        recorder: RealMLPHistoryRecorder,
        *,
        inner_validation_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.inner_validation_enabled = inner_validation_enabled

    @staticmethod
    def _learning_rate(pl_module: pl.LightningModule) -> float:
        optimizer = pl_module.optimizers(use_pl_optimizer=False)
        if hasattr(optimizer, "get_hyper_values"):
            return float(
                optimizer.get_hyper_values("lr", 0, use_hyper_factor=False)
            )
        return float(optimizer.param_groups[0]["lr"])

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.recorder.start_fit()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        torch.cuda.synchronize(self.recorder.device_index)
        self.recorder.start_epoch()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise RuntimeError("RealMLP callback expected one scalar training loss.")
        sample_count = int(batch["y"].shape[-2])
        self.recorder.add_train_batch(float(loss.detach().item()), sample_count)

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.inner_validation_enabled:
            self.recorder.finish_training(self._learning_rate(pl_module))

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.inner_validation_enabled:
            return
        epoch = int(pl_module.progress.epoch)
        metric_name = pl_module.val_metric_names[0]
        best_epoch = int(pl_module.best_mean_val_epochs[metric_name][0])
        self.recorder.complete_epoch(epoch, best_epoch)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.inner_validation_enabled:
            return
        self.recorder.finish_training(self._learning_rate(pl_module))
        self.recorder.complete_epoch(
            int(trainer.current_epoch) + 1,
            None,
            inner_validation_required=False,
        )


class _HistoryNNAlgInterface(NNAlgInterface):
    """PyTabKit 1.7.3 interface with an additional public Lightning callback."""

    def __init__(self, *, history_recorder: RealMLPHistoryRecorder, **config: Any) -> None:
        super().__init__(**config)
        self.history_recorder = history_recorder

    def fit(
        self,
        ds,
        idxs_list: list[SplitIdxs],
        interface_resources: InterfaceResources,
        logger: Logger,
        tmp_folders: list[Path | None],
        name: str,
    ) -> None:
        if PYTABKIT_VERSION != "1.7.3":
            raise RuntimeError("RealMLP history adapter is locked to pytabkit==1.7.3.")
        if not np.all(
            [
                idxs_list[index].train_idxs.shape[0]
                == idxs_list[0].train_idxs.shape[0]
                for index in range(len(idxs_list))
            ]
        ):
            raise RuntimeError("RealMLP history adapter received incompatible splits.")

        log = logging.getLogger("lightning")
        log.propagate = False
        log.setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message="You defined a `validation_step` but have no `val_dataloader`.",
        )

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            gpu_devices = interface_resources.gpu_devices
            self.device = gpu_devices[0] if gpu_devices else "cpu"
            ds = ds.to(self.device)
            fit_params = self.fit_params
            if fit_params is None and "stop_epoch" in self.config:
                fit_params = [dict(stop_epoch=self.config["stop_epoch"])] * len(
                    idxs_list
                )

            n_epochs = self.config.get("n_epochs", 256)
            history_logger = _HistoryLogger(logger, self.history_recorder)
            self.model = TabNNModule(
                **utils.join_dicts(
                    {"n_epochs": 256, "logger": history_logger}, self.config
                ),
                fit_params=fit_params,
            )
            self.model.compile_model(ds, idxs_list, interface_resources)
            accelerator, devices = get_lignting_accel_and_devices(self.device)
            max_time = (
                None
                if interface_resources.time_in_seconds is None
                else timedelta(seconds=interface_resources.time_in_seconds)
            )
            self.min_trainer_kwargs = {
                "max_time": max_time,
                "accelerator": accelerator,
                "devices": devices,
                "max_epochs": n_epochs,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "enable_model_summary": False,
                "log_every_n_steps": 1,
            }
            callbacks = [
                *self.model.create_callbacks(),
                _RealMLPHistoryCallback(
                    self.history_recorder,
                    inner_validation_enabled=self.model.val_dl is not None,
                ),
            ]
            trainer = pl.Trainer(
                **self.min_trainer_kwargs,
                callbacks=callbacks,
                logger=pl.loggers.logger.DummyLogger(),
            )
            trainer.fit(
                model=self.model,
                train_dataloaders=self.model.train_dl,
                val_dataloaders=self.model.val_dl,
            )
            if hasattr(self.model, "fit_params"):
                self.fit_params = self.model.fit_params
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
            if self.model is not None and hasattr(self.model, "_trainer"):
                del self.model._trainer


class _HistoryRealMLPBackend(RealMLP_TD_Classifier):
    history_recorder: RealMLPHistoryRecorder
    history_use_best_epoch: bool

    def _create_alg_interface(self, n_cv: int):
        if not hasattr(self, "history_recorder") or not hasattr(
            self, "history_use_best_epoch"
        ):
            raise RuntimeError("RealMLP history recorder was not attached.")
        config = self.get_config()
        config["use_best_epoch"] = self.history_use_best_epoch
        return _HistoryNNAlgInterface(
            history_recorder=self.history_recorder,
            **config,
        )


class HistoryEnabledRealMLPClassifier(RealMLPClassifier):
    """Dataset B wrapper using the pinned backend history adapter."""

    def attach_history_recorder(
        self, recorder: RealMLPHistoryRecorder
    ) -> "HistoryEnabledRealMLPClassifier":
        self._history_recorder = recorder
        return self

    def _make_backend(self):
        if not hasattr(self, "_history_recorder"):
            raise RuntimeError("RealMLP history recorder was not attached.")
        # Reuse the project wrapper's package-version and public-import gate.
        self._backend_class()
        backend = _HistoryRealMLPBackend(**self.backend_parameters())
        backend.history_recorder = self._history_recorder
        backend.history_use_best_epoch = self.use_best_epoch
        return backend
