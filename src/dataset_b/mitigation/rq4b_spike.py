"""Disposable RQ4b internal-extension technical spike (PyTabKit 1.7.3 RealMLP).

This module is spike evidence only. It is not an RQ4b runner, it does not
select lambda, epochs or thresholds, and it never reads project Train,
Validation or Test data. Every fit uses synthetic native-schema fixtures and
at most two executed epochs.

Extension path (pinned PyTabKit internals):
``RealMLP_TD_Classifier._create_alg_interface`` -> ``NNAlgInterface.fit`` ->
``TabNNModule.training_step``. The predictor optimizer, Lightning callbacks
(``HyperparamCallback``, ``L1L2RegCallback``), hyperparameter schedules and
the native ``drop_last=True`` loader are reused unchanged. The adversary is
held outside the predictor module registry and outside the predictor
optimizer, and its update uses plain ``Tensor.backward`` so predictor
callbacks fire exactly once per batch.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import math
import os
import platform
import stat
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
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
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.alg_interfaces.nn_interfaces import (
    NNAlgInterface,
    get_lignting_accel_and_devices,
)
from pytabkit.models.data import data as _ptk_data
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.training.logging import Logger

from src.dataset_b.models.realmlp import (
    PYTABKIT_VERSION,
    RealMLPClassifier,
    load_realmlp_v1_config,
)
from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


RQ4B_CONFIG_PATH = CONFIG_DIR / "dataset_b" / "rq4b_adversarial.yaml"
RQ4B_PROTOCOL_VERSION = "rq4b_adversarial_realmlp_v2_cv_epoch_selection"
DISPOSABLE_ROOT = PROJECT_ROOT / "scratch" / "rq4b_spike"

PYTABKIT_UPSTREAM_REPOSITORY = "https://github.com/dholzmueller/pytabkit"
PYTABKIT_UPSTREAM_TAG = "v1.7.3"
PYTABKIT_UPSTREAM_COMMIT = "c126ea51187c5080b91f28d352481dbd3b2194b0"
PYTABKIT_INTERNAL_FILES = (
    "pytabkit/models/sklearn/sklearn_base.py",
    "pytabkit/models/sklearn/sklearn_interfaces.py",
    "pytabkit/models/alg_interfaces/nn_interfaces.py",
    "pytabkit/models/training/lightning_modules.py",
    "pytabkit/models/training/lightning_callbacks.py",
    "pytabkit/models/training/nn_creator.py",
    "pytabkit/models/training/coord.py",
    "pytabkit/models/training/scheduling.py",
    "pytabkit/models/training/metrics.py",
    "pytabkit/models/optim/optimizers.py",
    "pytabkit/models/data/data.py",
    "pytabkit/models/torch_utils.py",
)

SPIKE_DEVICE = "cuda:0"
SPIKE_DEVICE_INDEX = 0
SPIKE_BATCH_SIZE = 256
SPIKE_PREDICT_BATCH_SIZE = 1024
MAX_SMOKE_EPOCHS = 2
PRODUCTION_EPOCH_HORIZON = 256
EPOCH_SELECTION_FOLDS = 5
G3_GROUPS = ("White", "Black or African American", "Asian")
OUTSIDE_G3_GROUPS = (
    "American Indian or Alaska Native",
    "2 or more minority races",
    "Joint",
    "Native Hawaiian or Other Pacific Islander",
    "Free Form Text Only",
    "Race Not Available",
)
OUTSIDE_ROSTER_CODE = -1
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


class RQ4bSpikeError(RuntimeError):
    """Base class for fail-closed spike errors."""


class RQ4bCudaUnavailableError(RQ4bSpikeError):
    """CUDA cuda:0 is unavailable; CPU fallback is prohibited."""


class RQ4bNonFiniteError(RQ4bSpikeError):
    """A prediction, adversary or total loss was not finite."""


class RQ4bAlignmentError(RQ4bSpikeError):
    """Row/group metadata is misaligned with the shuffled loader batch."""


class RQ4bReloadError(RQ4bSpikeError):
    """Serialized model failed to reload or changed eval-mode probabilities."""


class RQ4bEpochSelectionError(ValueError):
    """Synthetic epoch-selection curves are missing, malformed or nonfinite."""


class RQ4bContractError(RQ4bSpikeError):
    """The locked RQ4b v2 spike contract differs from this implementation."""


# ---------------------------------------------------------------------------
# Contract, provenance and preflight
# ---------------------------------------------------------------------------


def load_rq4b_spike_contract(config_path: str | Path = RQ4B_CONFIG_PATH) -> dict[str, Any]:
    """Read the locked v2 contract and assert the spike-relevant fields."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    expected = {
        ("protocol_version",): RQ4B_PROTOCOL_VERSION,
        ("technical_spike", "maximum_epochs_per_smoke_fit"): MAX_SMOKE_EPOCHS,
        ("technical_spike", "batch_size"): SPIKE_BATCH_SIZE,
        ("technical_spike", "architecture"): {
            "hidden_sizes": [ADVERSARY_HIDDEN],
            "activation": "relu",
            "output_classes": ADVERSARY_OUTPUTS,
            "dropout": 0.0,
        },
        ("technical_spike", "baseline_equivalence"): {
            "rtol": EQUIVALENCE_RTOL,
            "atol": EQUIVALENCE_ATOL,
        },
        ("technical_spike", "same_seed_smoke_repetitions_minimum"): 2,
        ("runtime", "device"): SPIKE_DEVICE,
        ("runtime", "batch_size"): SPIKE_BATCH_SIZE,
        ("runtime", "predict_batch_size"): SPIKE_PREDICT_BATCH_SIZE,
        ("runtime", "allow_silent_cpu_fallback"): False,
        ("groups", "g3"): list(G3_GROUPS),
        ("realmlp_recipe", "adversary", "optimizer"): {
            "name": "adam",
            "learning_rate": ADVERSARY_LR,
            "weight_decay": ADVERSARY_WEIGHT_DECAY,
        },
        ("realmlp_recipe", "adversary", "predictor_total_loss"): "L_prediction - lambda * L_adversary",
        ("realmlp_recipe", "adversary", "input_preference"): "approved_minus_denied_logit_margin",
        ("realmlp_recipe", "preserve_native_loader_drop_last"): True,
        ("epoch_selection", "epochs_per_fold"): PRODUCTION_EPOCH_HORIZON,
        ("epoch_selection", "splitter", "folds"): EPOCH_SELECTION_FOLDS,
        ("epoch_selection", "exact_tie_break"): "earliest_epoch",
        ("epoch_selection", "required_finite_fold_epoch_values"): 1280,
    }
    conflicts = []
    for path, value in expected.items():
        node: Any = payload
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node != value:
            conflicts.append({"field": ".".join(path), "expected": value, "found": node})
    if conflicts:
        raise RQ4bContractError(f"RQ4b v2 spike contract conflicts: {conflicts}")
    return payload


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pytabkit_installed_lineage(
    upstream_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hash installed PyTabKit sources and compare git blobs to the pinned commit.

    ``upstream_dir`` must hold ``commit.json`` and ``tree.json`` from the GitHub
    API for ``PYTABKIT_UPSTREAM_COMMIT``. Without it the lineage gate is
    reported unresolved; version and hashes alone are not lineage.
    """
    import pytabkit

    package_dir = Path(pytabkit.__file__).resolve().parent
    site_packages = package_dir.parent
    dist_info = site_packages / f"pytabkit-{PYTABKIT_VERSION}.dist-info"
    record_rows = list(csv.reader((dist_info / "RECORD").open(encoding="utf-8")))
    record_hashes = {row[0]: row[1] for row in record_rows if len(row) >= 2}
    direct_url = dist_info / "direct_url.json"
    files: dict[str, dict[str, Any]] = {}
    record_mismatches = []
    for relative in sorted(path for path in record_hashes if path.startswith("pytabkit/") and path.endswith(".py")):
        content = (site_packages / relative).read_bytes()
        sha256 = hashlib.sha256(content).digest()
        record_value = "sha256=" + base64.urlsafe_b64encode(sha256).rstrip(b"=").decode()
        if record_hashes[relative] != record_value:
            record_mismatches.append(relative)
        files[relative] = {
            "sha256": sha256.hex(),
            "git_blob_sha1": _git_blob_sha1(content),
            "record_hash_matches": record_hashes[relative] == record_value,
        }
    result: dict[str, Any] = {
        "package_version": version("pytabkit"),
        "package_dir": str(package_dir),
        "installer": (dist_info / "INSTALLER").read_text(encoding="utf-8").strip(),
        "wheel_metadata": (dist_info / "WHEEL").read_text(encoding="utf-8").strip().splitlines(),
        "direct_url_json_present": direct_url.exists(),
        "python_source_file_count": len(files),
        "record_hash_mismatches": record_mismatches,
        "internal_files": {path: files[path] for path in PYTABKIT_INTERNAL_FILES},
        "upstream_repository": PYTABKIT_UPSTREAM_REPOSITORY,
        "upstream_tag": PYTABKIT_UPSTREAM_TAG,
        "upstream_commit": PYTABKIT_UPSTREAM_COMMIT,
        "lineage_status": "UNRESOLVED",
        "lineage_reason": "upstream tree evidence not supplied",
    }
    if upstream_dir is None:
        return result
    upstream = Path(upstream_dir)
    commit = json.loads((upstream / "commit.json").read_text(encoding="utf-8"))
    tree = json.loads((upstream / "tree.json").read_text(encoding="utf-8"))
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    blob_mismatch = [path for path, info in files.items() if blobs.get(path) != info["git_blob_sha1"]]
    tag_ref = (upstream / "ls_remote_tags.txt").read_text(encoding="utf-8")
    tag_line = f"{PYTABKIT_UPSTREAM_COMMIT}\trefs/tags/{PYTABKIT_UPSTREAM_TAG}"
    checks = {
        "tag_resolves_to_commit": tag_line in tag_ref.splitlines(),
        "commit_sha_matches": commit.get("sha") == PYTABKIT_UPSTREAM_COMMIT,
        "tree_sha_matches_commit_tree": commit.get("commit", {}).get("tree", {}).get("sha") == tree.get("sha"),
        "tree_not_truncated": tree.get("truncated") is False,
        "all_installed_python_blobs_match": not blob_mismatch and bool(files),
        "record_hashes_match_installed_files": not record_mismatches,
    }
    result.update(
        upstream_checks=checks,
        upstream_tree_sha=tree.get("sha"),
        upstream_commit_date=commit.get("commit", {}).get("committer", {}).get("date"),
        blob_mismatches=blob_mismatch,
        lineage_status="PASS" if all(checks.values()) else "FAIL",
        lineage_reason="installed python blobs equal the pinned upstream tag commit"
        if all(checks.values())
        else "upstream comparison failed",
    )
    return result


def collect_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "pytabkit", "pytorch-lightning", "joblib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(SPIKE_DEVICE_INDEX)
        if torch.cuda.is_available()
        else None,
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def require_cuda0() -> dict[str, Any]:
    """Fail closed unless cuda:0 exists and performs a finite operation."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= SPIKE_DEVICE_INDEX:
        raise RQ4bCudaUnavailableError("RQ4b spike requires cuda:0; CPU fallback is prohibited.")
    probe = torch.ones(4, device=SPIKE_DEVICE) * 2.0
    if probe.device != torch.device(SPIKE_DEVICE) or not bool(torch.isfinite(probe).all().item()):
        raise RQ4bCudaUnavailableError("RQ4b spike cuda:0 finite-operation preflight failed.")
    return {"device": str(probe.device), "finite_operation": True}


# ---------------------------------------------------------------------------
# Synthetic native-schema fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticFixture:
    X: pd.DataFrame
    y: np.ndarray
    record_ids: np.ndarray
    derived_race: np.ndarray
    group_codes: np.ndarray


def g3_codes(derived_race: Sequence[str]) -> np.ndarray:
    lookup = {group: code for code, group in enumerate(G3_GROUPS)}
    return np.asarray([lookup.get(value, OUTSIDE_ROSTER_CODE) for value in derived_race], dtype=np.int64)


def make_synthetic_fixture(n_rows: int = 1100, seed: int = 20261018, *, g3_fraction: float = 0.8) -> SyntheticFixture:
    """Build a synthetic 19-column native frame; no project data is read."""
    if n_rows < 2 * SPIKE_BATCH_SIZE:
        raise ValueError("Synthetic fixture needs at least two full 256-row batches.")
    rng = np.random.default_rng(seed)
    frame: dict[str, Any] = {}
    numeric = rng.normal(size=(n_rows, len(NUMERIC_VALUE_COLUMNS))).astype(np.float32)
    for column in TWO_CHANNEL_COLUMNS:
        if column in NUMERIC_VALUE_COLUMNS:
            frame[column] = numeric[:, NUMERIC_VALUE_COLUMNS.index(column)]
        else:
            categories = list(CATEGORICAL_SCHEMA[column])
            codes = rng.integers(0, len(categories), size=n_rows)
            frame[column] = pd.Categorical.from_codes(codes, categories=categories)
    X = pd.DataFrame(frame, columns=list(TWO_CHANNEL_COLUMNS))
    score = numeric[:, 0] - 0.7 * numeric[:, 1] + 0.3 * rng.normal(size=n_rows)
    y = (score > np.quantile(score, 0.35)).astype(np.int64)
    in_g3 = rng.random(n_rows) < g3_fraction
    race = np.where(
        in_g3,
        np.asarray(G3_GROUPS, dtype=object)[rng.integers(0, len(G3_GROUPS), size=n_rows)],
        np.asarray(OUTSIDE_G3_GROUPS, dtype=object)[rng.integers(0, len(OUTSIDE_G3_GROUPS), size=n_rows)],
    )
    race[: len(G3_GROUPS)] = G3_GROUPS
    race[len(G3_GROUPS)] = OUTSIDE_G3_GROUPS[0]
    record_ids = np.asarray([f"SYN-{index:06d}" for index in range(n_rows)], dtype=object)
    fixture = SyntheticFixture(X, y, record_ids, race.astype(object), g3_codes(race))
    validate_fixture(fixture)
    return fixture


def native_numeric_features(X: pd.DataFrame) -> np.ndarray:
    """External row-aligned numeric value channels, in backend x_cont order."""
    return X.loc[:, list(NUMERIC_VALUE_COLUMNS)].to_numpy(dtype=np.float32)


ALIGNMENT_ID_COLUMN = 0  # income_value encodes the synthetic record number
ALIGNMENT_GROUP_COLUMN = 1  # loan_amount_value encodes the G3 code (-1 outside G3)


def make_alignment_fixture(n_rows: int = 1100, seed: int = 20261019) -> SyntheticFixture:
    """Fixture whose features encode record identity and group, so same-outcome
    rows with different identities/groups are distinguishable in the batch."""
    base = make_synthetic_fixture(n_rows, seed)
    X = base.X.copy()
    X[NUMERIC_VALUE_COLUMNS[ALIGNMENT_ID_COLUMN]] = np.arange(n_rows, dtype=np.float32)
    X[NUMERIC_VALUE_COLUMNS[ALIGNMENT_GROUP_COLUMN]] = base.group_codes.astype(np.float32)
    fixture = SyntheticFixture(X, base.y, base.record_ids, base.derived_race, base.group_codes)
    validate_fixture(fixture)
    return fixture


def label_preserving_group_swap(fixture: SyntheticFixture) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Swap metadata between the first same-outcome row pair that differs in group."""
    for label in (0, 1):
        rows = np.flatnonzero(fixture.y == label)
        a = rows[0]
        others = rows[fixture.group_codes[rows] != fixture.group_codes[a]]
        if len(others):
            b = int(others[0])
            perm = np.arange(len(fixture.y))
            perm[[a, b]] = perm[[b, a]]
            return fixture.group_codes[perm], perm, {
                "rows": [int(a), b],
                "label": label,
                "groups_before": [int(fixture.group_codes[a]), int(fixture.group_codes[b])],
                "record_ids": [str(fixture.record_ids[a]), str(fixture.record_ids[b])],
            }
    raise ValueError("No same-outcome pair with different groups.")


def validate_fixture(fixture: SyntheticFixture) -> dict[str, Any]:
    n_rows = len(fixture.X)
    if not (len(fixture.y) == len(fixture.record_ids) == len(fixture.group_codes) == n_rows):
        raise ValueError("Synthetic fixture arrays are misaligned.")
    if len(set(fixture.record_ids.tolist())) != n_rows:
        raise ValueError("Synthetic fixture record IDs are not unique.")
    if set(fixture.y.tolist()) != {0, 1}:
        raise ValueError("Synthetic fixture requires both outcomes.")
    present = set(fixture.group_codes.tolist())
    if not {0, 1, 2}.issubset(present) or OUTSIDE_ROSTER_CODE not in present:
        raise ValueError("Synthetic fixture requires all G3 groups and rows outside G3.")
    RealMLPClassifier._validate_X(fixture.X)
    return {
        "rows": n_rows,
        "columns": list(fixture.X.columns),
        "approved": int(fixture.y.sum()),
        "denied": int((fixture.y == 0).sum()),
        "group_code_counts": {str(code): int((fixture.group_codes == code).sum()) for code in sorted(present)},
        "record_id_sha256": hashlib.sha256("\n".join(fixture.record_ids.tolist()).encode()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Extension internals
# ---------------------------------------------------------------------------


@dataclass
class RQ4bSpikeSettings:
    """Per-fit spike settings; data arrays are aligned to the X row order."""

    adversary_lambda: float
    group_codes: np.ndarray
    alignment_labels: np.ndarray
    adversary_seed: int
    alignment_features: np.ndarray | None = None  # float32 (rows, 4) NUMERIC_VALUE_COLUMNS in X order
    group_feature_column: int | None = None  # alignment fixtures: raw x_cont column encoding the group code
    log_raw_batch_features: bool = False
    module_kind: str = "adversarial"  # "adversarial" or "stock_observed"
    schedule_horizon_epochs: int | None = None
    native_stop_epoch: int | None = None
    record_gradient_diagnostics: bool = False
    max_executed_epochs: int = MAX_SMOKE_EPOCHS
    observer: "SpikeObserver | None" = None

    def __post_init__(self) -> None:
        if self.module_kind not in ("adversarial", "stock_observed"):
            raise ValueError("Unknown RQ4b spike module kind.")
        if not math.isfinite(self.adversary_lambda) or self.adversary_lambda < 0:
            raise ValueError("RQ4b spike lambda must be finite and non-negative.")
        if self.max_executed_epochs > MAX_SMOKE_EPOCHS:
            raise ValueError("RQ4b spike smoke fits are limited to two executed epochs.")


def _batch_fingerprint(batch: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in batch if key != ROW_INDEX_KEY):
        tensor = batch[key].detach().to("cpu").contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SpikeObserver(Callback):
    """Read-only per-batch recorder and two-epoch execution guard."""

    def __init__(self, max_executed_epochs: int = MAX_SMOKE_EPOCHS) -> None:
        super().__init__()
        self.max_executed_epochs = max_executed_epochs
        self.batch_fingerprints: list[str] = []
        self.hypers: list[dict[str, Any]] = []
        self.completed_epochs = 0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch >= self.max_executed_epochs:
            raise RQ4bSpikeError("RQ4b spike attempted more than two executed epochs.")

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        self.batch_fingerprints.append(_batch_fingerprint(batch))
        sched = pl_module.hp_manager.get_hyper_sched_values()
        self.hypers.append(
            {
                "epoch": int(trainer.current_epoch) + 1,
                "batch_idx": int(batch_idx),
                "fit_progress": float(pl_module.progress.get_fit_progress()),
                "module_max_epochs": int(pl_module.progress.max_epochs),
                "sched": {
                    name: {pattern: float(value) for pattern, value in sched[name].items()}
                    for name in ("lr", "wd", "p_drop")
                    if name in sched
                },
            }
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.completed_epochs += 1


class _RowIndexedTrainLoader(_ptk_data.ParallelDictDataLoader):
    """Native shuffled loader that also yields gathered dataset row indices.

    The permutation call and batch slicing are identical to
    ``ParallelDictDataLoader.__iter__`` so RNG consumption and batch order are
    unchanged; the extra key is removed before predictor forward.
    """

    @classmethod
    def from_loader(cls, loader: _ptk_data.ParallelDictDataLoader) -> "_RowIndexedTrainLoader":
        if type(loader) is not _ptk_data.ParallelDictDataLoader or not loader.drop_last or not loader.shuffle:
            raise RQ4bSpikeError("Expected the native shuffled drop_last train loader.")
        wrapped = cls.__new__(cls)
        wrapped.__dict__.update(loader.__dict__)
        return wrapped

    def __iter__(self):
        if self.shuffle:
            perms = _ptk_data.batch_randperm(self.n_parallel, self.n_samples, device=self.ds.device)
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                idxs = self.idxs.gather(1, perms[:, start:stop])
                batches = self.ds.get_batch(idxs=idxs)
                out = {key: t.to(self.output_device) for key, t in batches.items()}
                out[ROW_INDEX_KEY] = idxs.to(self.output_device)
                yield out
        else:
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                idxs = self.idxs[:, start:stop]
                batches = self.ds.get_batch(idxs=idxs)
                out = {key: t.to(self.output_device) for key, t in batches.items()}
                out[ROW_INDEX_KEY] = idxs.to(self.output_device)
                yield out


def build_spike_adversary(seed: int, device: str) -> tuple[torch.nn.Module, dict[str, bool]]:
    """Spike-only 2->32 ReLU->3 adversary built under an isolated RNG fork."""
    device_index = torch.device(device).index
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(device_index)
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        adversary = torch.nn.Sequential(
            torch.nn.Linear(2, ADVERSARY_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(ADVERSARY_HIDDEN, ADVERSARY_OUTPUTS),
        )
    adversary = adversary.to(device)
    isolation = {
        "cpu_rng_state_unchanged": torch.equal(cpu_before, torch.get_rng_state()),
        "cuda_rng_state_unchanged": torch.equal(cuda_before, torch.cuda.get_rng_state(device_index)),
    }
    if not all(isolation.values()):
        raise RQ4bSpikeError("Adversary construction changed predictor RNG state.")
    return adversary, isolation


def approved_minus_denied_margin(logits: torch.Tensor) -> torch.Tensor:
    """Return the class-1 minus class-0 logit margin; logits are (models, rows, 2)."""
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
        raise RQ4bSpikeError(f"Unexpected RealMLP score tensor shape {tuple(logits.shape)}.")
    return (logits[..., 1] - logits[..., 0]).reshape(-1)


def _snapshot(parameters: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _max_change(before: Sequence[torch.Tensor], parameters: Sequence[torch.Tensor]) -> float:
    return max(float((after.detach() - old).abs().max().item()) for old, after in zip(before, parameters))


class AdversarialTabNNModule(TabNNModule):
    """TabNNModule with a signed adversarial predictor objective."""

    def __init__(self, *, rq4b_settings: RQ4bSpikeSettings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rq4b_lambda = float(rq4b_settings.adversary_lambda)
        self.rq4b_adversary_seed = int(rq4b_settings.adversary_seed)
        self.rq4b_record_gradients = bool(rq4b_settings.record_gradient_diagnostics)
        self._rq4b_group_codes_np = np.asarray(rq4b_settings.group_codes, dtype=np.int64)
        self._rq4b_labels_np = np.asarray(rq4b_settings.alignment_labels, dtype=np.int64)
        if rq4b_settings.alignment_features is None:
            raise RQ4bAlignmentError("RQ4b spike requires external row-aligned feature metadata.")
        self._rq4b_features_np = np.asarray(rq4b_settings.alignment_features, dtype=np.float32)
        self.rq4b_group_feature_column = rq4b_settings.group_feature_column
        self.rq4b_log_raw_features = bool(rq4b_settings.log_raw_batch_features)
        self.rq4b_log: dict[str, Any] = {
            "lambda": self.rq4b_lambda,
            "adversary_seed": self.rq4b_adversary_seed,
            "batches": [],
            "adversary_updates": 0,
            "adversary_skipped_empty_mask": 0,
            "adversary_bypassed_lambda_zero": 0,
            "gradient_diagnostics": None,
            "devices": None,
            "alignment": {"dataset_rows_verified": 0, "batches_feature_verified": 0, "batches_group_feature_verified": 0},
        }

    def compile_model(self, ds, idxs_list: list[SplitIdxs], interface_resources: InterfaceResources) -> None:
        super().compile_model(ds, idxs_list, interface_resources)
        device = self.creator.device_info
        if str(device) != SPIKE_DEVICE:
            raise RQ4bCudaUnavailableError(f"RQ4b spike predictor is on {device}, not {SPIKE_DEVICE}.")
        if (
            len(self._rq4b_group_codes_np) != ds.n_samples
            or len(self._rq4b_labels_np) != ds.n_samples
            or self._rq4b_features_np.shape != (ds.n_samples, len(NUMERIC_VALUE_COLUMNS))
        ):
            raise RQ4bAlignmentError("RQ4b metadata length differs from the backend dataset.")
        raw = ds.to(device)
        # Backend dataset row i must carry exactly the numeric features and label of X row i.
        external = torch.as_tensor(self._rq4b_features_np, device=device)
        if raw.tensors["x_cont"].shape != external.shape or not torch.equal(raw.tensors["x_cont"], external):
            raise RQ4bAlignmentError("Backend dataset rows differ from external row-aligned features.")
        if not torch.equal(raw.tensors["y"][:, 0].cpu(), torch.as_tensor(self._rq4b_labels_np)):
            raise RQ4bAlignmentError("Backend dataset labels differ from external row-aligned labels.")
        if self.rq4b_group_feature_column is not None and not torch.equal(
            raw.tensors["x_cont"][:, self.rq4b_group_feature_column].to(torch.int64).cpu(),
            torch.as_tensor(self._rq4b_group_codes_np),
        ):
            raise RQ4bAlignmentError("Group metadata differs from the group-encoding feature.")
        self.rq4b_log["alignment"]["dataset_rows_verified"] = int(ds.n_samples)
        object.__setattr__(self, "_rq4b_raw_ds", raw)
        object.__setattr__(self, "_rq4b_last_raw_x", None)
        self.train_dl = _RowIndexedTrainLoader.from_loader(self.train_dl)
        # Kept as plain attributes (not registered submodules/parameters) so the
        # predictor module, its state_dict and its optimizer are unchanged.
        object.__setattr__(self, "_rq4b_groups", torch.as_tensor(self._rq4b_group_codes_np, device=device))
        object.__setattr__(self, "_rq4b_labels", torch.as_tensor(self._rq4b_labels_np, device=device))
        adversary, isolation = build_spike_adversary(self.rq4b_adversary_seed, str(device))
        object.__setattr__(self, "rq4b_adversary", adversary)
        object.__setattr__(
            self,
            "_rq4b_adversary_opt",
            torch.optim.Adam(adversary.parameters(), lr=ADVERSARY_LR, weight_decay=ADVERSARY_WEIGHT_DECAY),
        )
        self.rq4b_log["adversary_rng_isolation"] = isolation
        self.rq4b_log["adversary_constructed"] = True

    def manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Refuse nonfinite objectives before any backward pass or optimizer step."""
        if not bool(torch.isfinite(loss).all().item()):
            raise RQ4bNonFiniteError("Predictor objective is not finite; backward and optimizer step refused.")
        super().manual_backward(loss, *args, **kwargs)

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
        # The predictor batch must be exactly the static transform of the indexed dataset rows.
        with torch.no_grad():
            expected = self.creator.static_model(self._rq4b_raw_ds.get_sub_dataset(flat))
        if set(expected.tensors) != set(core) or any(
            not torch.equal(expected.tensors[key], core[key][0]) for key in core
        ):
            raise RQ4bAlignmentError("Shuffled batch features differ from the indexed dataset rows.")
        self.rq4b_log["alignment"]["batches_feature_verified"] += 1
        raw_x = self._rq4b_raw_ds.tensors["x_cont"][flat]
        if self.rq4b_group_feature_column is not None:
            if not torch.equal(raw_x[:, self.rq4b_group_feature_column].to(torch.int64), self._rq4b_groups[flat]):
                raise RQ4bAlignmentError("Batch group metadata differs from the group-encoding feature.")
            self.rq4b_log["alignment"]["batches_group_feature_verified"] += 1
        object.__setattr__(self, "_rq4b_last_raw_x", raw_x if self.rq4b_log_raw_features else None)
        return core, flat

    def training_step(self, batch, batch_idx):
        core, rows = self._align(batch)
        groups = self._rq4b_groups[rows]
        labels = self._rq4b_labels[rows]
        mask = groups != OUTSIDE_ROSTER_CODE
        n_roster = int(mask.sum().item())
        if self.rq4b_log["devices"] is None:
            self.rq4b_log["devices"] = {
                **{f"batch_{key}": str(value.device) for key, value in batch.items()},
                "group_metadata": str(self._rq4b_groups.device),
                "predictor_parameters": sorted({str(p.device) for p in self.model.parameters()}),
                "adversary_parameters": sorted({str(p.device) for p in self.rq4b_adversary.parameters()}),
                "train_loader": {
                    "batch_size": int(self.train_dl.batch_size),
                    "drop_last": bool(self.train_dl.drop_last),
                    "shuffle": bool(self.train_dl.shuffle),
                    "batches_per_epoch": int(self.train_dl.n_batches),
                    "iterated_samples_per_epoch": int(self.train_dl.get_num_iterated_samples()),
                },
            }
        record: dict[str, Any] = {
            "epoch": int(self.trainer.current_epoch) + 1,
            "batch_idx": int(batch_idx),
            "rows": rows.detach().cpu().tolist(),
            "groups": groups.detach().cpu().tolist(),
            "roster_rows": n_roster,
        }
        if self._rq4b_last_raw_x is not None:
            record["raw_x_cont"] = self._rq4b_last_raw_x.detach().cpu().tolist()
        if self.rq4b_lambda == 0.0:
            # Preserved recipe: the unmodified PyTabKit step on the stripped batch.
            # A nonfinite loss is rejected in manual_backward, before backward/step.
            loss = TabNNModule.training_step(self, core, batch_idx)
            self.rq4b_log["adversary_bypassed_lambda_zero"] += 1
            record.update(adversary="bypassed_lambda_zero", prediction_loss=float(loss.detach().item()))
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
        # Batch 1 has lr=0 under coslog4 (t=0), so diagnose from the second batch on.
        diagnostics_wanted = (
            self.rq4b_record_gradients
            and self.rq4b_log["gradient_diagnostics"] is None
            and len(self.rq4b_log["batches"]) >= 1
        )
        if diagnostics_wanted:
            predictor_before = _snapshot(list(self.model.parameters()))
            adversary_before = _snapshot(adversary_params)
        if n_roster > 0:
            detached_loss = self._adversary_loss(margin.detach()[mask], labels[mask], groups[mask])
            if not bool(torch.isfinite(detached_loss).item()):
                raise RQ4bNonFiniteError("Adversary loss on detached scores is not finite.")
            self._rq4b_adversary_opt.zero_grad()
            # Plain backward: predictor callbacks (L1/L2, hyper updates) must not fire here.
            detached_loss.backward()
            self._rq4b_adversary_opt.step()
            self.rq4b_log["adversary_updates"] += 1
            if diagnostics_wanted:
                predictor_after_adv = _snapshot(list(self.model.parameters()))
                adversary_after_adv = _snapshot(adversary_params)
                predictor_grads_after_adv = [p.grad is None or float(p.grad.abs().max()) == 0.0 for p in self.model.parameters()]
            for parameter in adversary_params:
                parameter.requires_grad_(False)
            adversary_loss = self._adversary_loss(margin[mask], labels[mask], groups[mask])
            if not bool(torch.isfinite(adversary_loss).item()):
                for parameter in adversary_params:
                    parameter.requires_grad_(True)
                raise RQ4bNonFiniteError("Adversary loss for the predictor update is not finite.")
            total = prediction_loss - self.rq4b_lambda * adversary_loss
            record.update(adversary="updated", adversary_loss_detached=float(detached_loss.item()), adversary_loss=float(adversary_loss.detach().item()))
        else:
            adversary_loss = None
            total = prediction_loss
            self.rq4b_log["adversary_skipped_empty_mask"] += 1
            record.update(adversary="skipped_empty_mask")
        try:
            if not bool(torch.isfinite(total).item()):
                raise RQ4bNonFiniteError("Signed predictor objective is not finite.")
            if diagnostics_wanted and adversary_loss is not None:
                diag = self._gradient_diagnostics(logits, margin, mask, prediction_loss, adversary_loss, total)
                adversary_grads_before_predictor = [p.grad.detach().clone() for p in adversary_params]
            self.manual_backward(total)
            opt.step(loss=total)
            opt.zero_grad()
        finally:
            for parameter in adversary_params:
                parameter.requires_grad_(True)
        if diagnostics_wanted and adversary_loss is not None:
            diag.update(
                diagnosed_batch_number=len(self.rq4b_log["batches"]) + 1,
                predictor_lr_at_diagnosed_batch=float(
                    self.optimizers(use_pl_optimizer=False).get_hyper_values("lr", 0, use_hyper_factor=False)
                ),
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
        record.update(prediction_loss=float(prediction_loss.detach().item()), total_loss=float(total.detach().item()))
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
            "adversary_input_shape": [int(mask.sum().item()), 2],
            "adversary_output_shape": [int(mask.sum().item()), ADVERSARY_OUTPUTS],
            "batch_rows": int(mask.numel()),
            "roster_rows": int(mask.sum().item()),
            "outside_rows": int(outside.sum().item()),
            "lambda": lam,
            "signed_identity_holds": bool(
                torch.allclose(grad_total, signed, rtol=GRADIENT_IDENTITY_RTOL, atol=GRADIENT_IDENTITY_ATOL)
            ),
            "signed_identity_max_abs_diff": float((grad_total - signed).abs().max().item()),
            "additive_variant_max_abs_diff": float((grad_total - additive).abs().max().item()),
            "adversary_grad_outside_g3_max_abs": float(grad_adv[outside].abs().max().item()) if outside.any() else None,
            "adversary_margin_grad_outside_g3_max_abs": float(grad_adv_margin[outside].abs().max().item())
            if outside.any()
            else None,
            "adversary_grad_g3_nonzero_rows": int((grad_adv[mask].abs().sum(dim=1) > 0).sum().item()),
            "prediction_grad_outside_g3_nonzero_rows": int((grad_pred[outside].abs().sum(dim=1) > 0).sum().item()),
            "prediction_grad_g3_nonzero_rows": int((grad_pred[mask].abs().sum(dim=1) > 0).sum().item()),
            # Rows where the adversarial part of d(total)/d(logit_1) opposes d(L_adv)/d(logit_1).
            "adversary_logit1_grad_nonzero_g3_rows": int((grad_adv[mask, 1] != 0).sum().item()),
            "total_minus_pred_opposes_adv_g3_rows": int(
                (torch.sign(grad_total[mask, 1] - grad_pred[mask, 1]) == -torch.sign(grad_adv[mask, 1]))
                .logical_and(grad_adv[mask, 1] != 0)
                .sum()
                .item()
            ),
        }

    def on_fit_end(self) -> None:
        super().on_fit_end()
        # Mirror PyTabKit: do not pickle training data or optimizer state.
        for name in ("_rq4b_groups", "_rq4b_labels", "_rq4b_adversary_opt", "_rq4b_raw_ds", "_rq4b_last_raw_x"):
            if name in self.__dict__:
                object.__delattr__(self, name)
        self._rq4b_group_codes_np = None
        self._rq4b_labels_np = None
        self._rq4b_features_np = None


class _SpikeNNAlgInterface(NNAlgInterface):
    """Pinned NNAlgInterface.fit path with a selectable TabNNModule class."""

    def __init__(self, *, rq4b_settings: RQ4bSpikeSettings, **config: Any) -> None:
        super().__init__(**config)
        self.rq4b_settings = rq4b_settings

    def fit(self, ds, idxs_list: list[SplitIdxs], interface_resources: InterfaceResources, logger: Logger, tmp_folders, name: str) -> None:
        if version("pytabkit") != "1.7.3":
            raise RuntimeError("RQ4b spike adapter is locked to pytabkit==1.7.3.")
        if self.config.get("random_seed_offset", 0) != 0 or self.config.get("same_seed_for_sub_splits", False):
            raise RQ4bSpikeError("RQ4b spike does not support seed-offset configuration.")
        if not np.all([idxs.train_idxs.shape[0] == idxs_list[0].train_idxs.shape[0] for idxs in idxs_list]):
            raise RQ4bSpikeError("RQ4b spike received incompatible splits.")
        settings = self.rq4b_settings
        log = logging.getLogger("lightning")
        log.propagate = False
        log.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`.")
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            gpu_devices = interface_resources.gpu_devices
            if list(gpu_devices) != [SPIKE_DEVICE]:
                raise RQ4bCudaUnavailableError(f"RQ4b spike requires gpu_devices == ['{SPIKE_DEVICE}'].")
            self.device = gpu_devices[0]
            ds = ds.to(self.device)
            fit_params = self.fit_params
            if fit_params is None and "stop_epoch" in self.config:
                fit_params = [dict(stop_epoch=self.config["stop_epoch"])] * len(idxs_list)
            n_epochs = self.config.get("n_epochs", 256)
            module_kwargs = utils.join_dicts({"n_epochs": 256, "logger": logger}, self.config)
            if settings.schedule_horizon_epochs is not None:
                module_kwargs["n_epochs"] = settings.schedule_horizon_epochs
            if settings.module_kind == "adversarial":
                self.model = AdversarialTabNNModule(rq4b_settings=settings, fit_params=fit_params, **module_kwargs)
            else:
                self.model = TabNNModule(fit_params=fit_params, **module_kwargs)
            self.model.compile_model(ds, idxs_list, interface_resources)
            accelerator, devices = get_lignting_accel_and_devices(self.device)
            max_time = None if interface_resources.time_in_seconds is None else timedelta(seconds=interface_resources.time_in_seconds)
            self.min_trainer_kwargs = dict(
                max_time=max_time,
                accelerator=accelerator,
                devices=devices,
                max_epochs=n_epochs,
                enable_checkpointing=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                enable_model_summary=False,
                log_every_n_steps=1,
            )
            callbacks = list(self.model.create_callbacks())
            if settings.observer is not None:
                callbacks.append(settings.observer)
            trainer = pl.Trainer(**self.min_trainer_kwargs, callbacks=callbacks, logger=pl.loggers.logger.DummyLogger())
            trainer.fit(model=self.model, train_dataloaders=self.model.train_dl, val_dataloaders=self.model.val_dl)
            if hasattr(self.model, "fit_params"):
                self.fit_params = self.model.fit_params
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
            if self.model is not None and hasattr(self.model, "_trainer"):
                del self.model._trainer
            # Drop training metadata and the live observer from the pickled interface.
            self.rq4b_settings = None


class _SpikeRealMLPBackend(RealMLP_TD_Classifier):
    rq4b_settings: RQ4bSpikeSettings

    def _create_alg_interface(self, n_cv: int):
        if not hasattr(self, "rq4b_settings"):
            raise RQ4bSpikeError("RQ4b spike settings were not attached.")
        config = self.get_config()
        if self.rq4b_settings.native_stop_epoch is not None:
            config["stop_epoch"] = int(self.rq4b_settings.native_stop_epoch)
        settings = self.rq4b_settings
        del self.rq4b_settings
        return _SpikeNNAlgInterface(rq4b_settings=settings, **config)


class SpikeAdversarialRealMLPClassifier(RealMLPClassifier):
    """Project wrapper routing fit through the spike interface/module."""

    def attach_rq4b(self, settings: RQ4bSpikeSettings) -> "SpikeAdversarialRealMLPClassifier":
        self._rq4b_settings = settings
        return self

    def fit(self, X, y, *, X_val=None, y_val=None):
        if X_val is not None or y_val is not None:
            raise RQ4bSpikeError("RQ4b spike prohibits an internal validation channel.")
        settings = getattr(self, "_rq4b_settings", None)
        if settings is None:
            raise RQ4bSpikeError("RQ4b spike settings were not attached.")
        if len(settings.group_codes) != len(X) or len(settings.alignment_labels) != len(X):
            raise RQ4bAlignmentError("RQ4b metadata length differs from X.")
        result = super().fit(X, y)
        module = self.backend_.alg_interface_.model
        self.rq4b_log_ = getattr(module, "rq4b_log", None)
        del self._rq4b_settings
        return result

    def _validate_requested_device(self) -> None:
        if self.device != SPIKE_DEVICE:
            raise RQ4bCudaUnavailableError(f"RQ4b spike requires device={SPIKE_DEVICE}.")
        require_cuda0()

    def _make_backend(self):
        self._backend_class()
        backend = _SpikeRealMLPBackend(**self.backend_parameters())
        backend.rq4b_settings = self._rq4b_settings
        return backend


# ---------------------------------------------------------------------------
# Spike fits and comparisons
# ---------------------------------------------------------------------------


def spike_estimator_parameters(seed: int, n_epochs: int) -> dict[str, Any]:
    """Locked realmlp_v1 parameters with only the project's fit-time overrides."""
    payload = load_realmlp_v1_config()
    params = {**payload["estimator_parameters"], **payload["project_policy"]}
    params.update(device=SPIKE_DEVICE, random_state=int(seed), n_epochs=int(n_epochs))
    if params["batch_size"] != SPIKE_BATCH_SIZE or params["predict_batch_size"] != SPIKE_PREDICT_BATCH_SIZE:
        raise RQ4bContractError("RealMLP batch sizes differ from the RQ4b runtime contract.")
    return params


def input_digest(X: pd.DataFrame, y: np.ndarray, record_ids: np.ndarray, group_codes: np.ndarray) -> dict[str, str]:
    """SHA-256 of the exact inputs handed to one fit, for exact-equality checks."""
    return {
        "X": hashlib.sha256(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()).hexdigest(),
        "X_dtypes": hashlib.sha256(repr([(c, str(X[c].dtype)) for c in X.columns]).encode()).hexdigest(),
        "labels": hashlib.sha256(np.ascontiguousarray(y, dtype=np.int64).tobytes()).hexdigest(),
        "record_ids": hashlib.sha256("\n".join(map(str, record_ids)).encode()).hexdigest(),
        "masks": hashlib.sha256(np.ascontiguousarray(group_codes, dtype=np.int64).tobytes()).hexdigest(),
    }


def fit_stock_realmlp(fixture: SyntheticFixture, *, seed: int, n_epochs: int = MAX_SMOKE_EPOCHS) -> RealMLPClassifier:
    """Fresh preserved-recipe smoke through the unmodified project wrapper/backend."""
    if n_epochs > MAX_SMOKE_EPOCHS:
        raise RQ4bSpikeError("Stock smoke is limited to two epochs.")
    require_cuda0()
    estimator = RealMLPClassifier(**spike_estimator_parameters(seed, n_epochs))
    digest = input_digest(fixture.X, fixture.y, fixture.record_ids, fixture.group_codes)
    estimator.fit(fixture.X, fixture.y)
    estimator.rq4b_input_digest_ = digest
    return estimator


def fit_spike_realmlp(
    fixture: SyntheticFixture,
    *,
    seed: int,
    adversary_lambda: float,
    module_kind: str = "adversarial",
    n_epochs: int = MAX_SMOKE_EPOCHS,
    schedule_horizon_epochs: int | None = None,
    native_stop_epoch: int | None = None,
    record_gradient_diagnostics: bool = False,
    group_codes: np.ndarray | None = None,
    alignment_labels: np.ndarray | None = None,
    alignment_features: np.ndarray | None = None,
    group_feature_column: int | None = None,
    log_raw_batch_features: bool = False,
    adversary_seed: int | None = None,
) -> tuple[SpikeAdversarialRealMLPClassifier, SpikeObserver]:
    observer = SpikeObserver()
    settings = RQ4bSpikeSettings(
        adversary_lambda=adversary_lambda,
        group_codes=fixture.group_codes if group_codes is None else group_codes,
        alignment_labels=fixture.y if alignment_labels is None else alignment_labels,
        alignment_features=native_numeric_features(fixture.X) if alignment_features is None else alignment_features,
        group_feature_column=group_feature_column,
        log_raw_batch_features=log_raw_batch_features,
        adversary_seed=seed if adversary_seed is None else adversary_seed,
        module_kind=module_kind,
        schedule_horizon_epochs=schedule_horizon_epochs,
        native_stop_epoch=native_stop_epoch,
        record_gradient_diagnostics=record_gradient_diagnostics,
        observer=observer,
    )
    if native_stop_epoch is None and n_epochs > MAX_SMOKE_EPOCHS:
        raise RQ4bSpikeError("Spike smoke is limited to two epochs.")
    estimator = SpikeAdversarialRealMLPClassifier(**spike_estimator_parameters(seed, n_epochs)).attach_rq4b(settings)
    digest = input_digest(fixture.X, fixture.y, fixture.record_ids, settings.group_codes)
    estimator.fit(fixture.X, fixture.y)
    estimator.rq4b_input_digest_ = digest
    if observer.completed_epochs > MAX_SMOKE_EPOCHS:
        raise RQ4bSpikeError("Spike smoke executed more than two epochs.")
    return estimator, observer


def predictor_tensors(estimator: RealMLPClassifier) -> dict[str, np.ndarray]:
    layer = estimator.backend_.alg_interface_.model.model
    tensors = {f"param:{name}": p.detach().cpu().numpy().copy() for name, p in layer.named_parameters()}
    tensors.update({f"buffer:{name}": b.detach().cpu().numpy().copy() for name, b in layer.named_buffers()})
    return tensors


def predictor_devices(estimator: RealMLPClassifier) -> dict[str, Any]:
    """Post-fit placement; Lightning teardown returns the module to CPU after fit/predict."""
    interface = estimator.backend_.alg_interface_
    return {
        "interface_device": str(interface.device),
        "trainer_accelerator": str(interface.min_trainer_kwargs.get("accelerator")),
        "trainer_devices": _jsonable(interface.min_trainer_kwargs.get("devices")),
        "predictor_parameter_devices_at_rest": sorted({str(p.device) for p in interface.model.model.parameters()}),
    }


def tolerance_utilization(reference: np.ndarray, candidate: np.ndarray, *, rtol: float, atol: float) -> np.ndarray:
    """Elementwise |candidate - reference| / (atol + rtol * |reference|); <= 1 passes np.allclose."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    return np.abs(b - a) / (atol + rtol * np.abs(a))


def _utilization_summary(utilizations: Sequence[tuple[str, np.ndarray]]) -> dict[str, Any]:
    flat = np.concatenate([u.reshape(-1) for _, u in utilizations]) if utilizations else np.zeros(0)
    if flat.size == 0:
        return {"elements": 0}
    worst_key, worst_index, worst = None, None, -1.0
    for key, u in utilizations:
        if u.size and float(u.max()) > worst:
            worst = float(u.max())
            worst_key = key
            worst_index = [int(i) for i in np.unravel_index(int(np.argmax(u)), u.shape)]
    return {
        "elements": int(flat.size),
        "max_utilization": worst,
        "margin_to_tolerance": 1.0 - worst,
        "worst_tensor": worst_key,
        "worst_index": worst_index,
        "p99_utilization": float(np.quantile(flat, 0.99)),
        "mean_utilization": float(flat.mean()),
        "elements_over_0_5": int((flat > 0.5).sum()),
        "elements_over_0_9": int((flat > 0.9).sum()),
        "elements_over_1_0": int((flat > 1.0).sum()),
        "nonzero_diff_elements": int((flat > 0).sum()),
    }


def compare_tensors(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], *, rtol: float, atol: float) -> dict[str, Any]:
    if list(reference) != list(candidate):
        return {"keys_equal": False, "within_tolerance": False, "max_abs_diff": None}
    diffs, ok, exact, utilizations = [], True, True, []
    for key in reference:
        a, b = reference[key], candidate[key]
        if a.shape != b.shape:
            return {"keys_equal": True, "shapes_equal": False, "within_tolerance": False, "max_abs_diff": None}
        diffs.append(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0)
        ok = ok and bool(np.allclose(b, a, rtol=rtol, atol=atol))
        exact = exact and bool(np.array_equal(a, b))
        utilizations.append((key, tolerance_utilization(a, b, rtol=rtol, atol=atol)))
    return {
        "keys_equal": True,
        "shapes_equal": True,
        "tensor_count": len(reference),
        "within_tolerance": ok,
        "bitwise_equal": exact,
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "tolerance_utilization": _utilization_summary(utilizations),
        "rtol": rtol,
        "atol": atol,
    }


def compare_probabilities(reference: np.ndarray, candidate: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    same_shape = reference.shape == candidate.shape
    return {
        "shape": list(candidate.shape),
        "within_tolerance": bool(same_shape and np.allclose(candidate, reference, rtol=rtol, atol=atol)),
        "bitwise_equal": bool(np.array_equal(reference, candidate)),
        "max_abs_diff": float(np.max(np.abs(reference - candidate))) if same_shape else None,
        "tolerance_utilization": _utilization_summary(
            [("probabilities", tolerance_utilization(reference, candidate, rtol=rtol, atol=atol))]
        )
        if same_shape
        else None,
        "rtol": rtol,
        "atol": atol,
    }


def serialize_and_verify_reload(
    estimator: RealMLPClassifier,
    X: pd.DataFrame,
    path: str | Path,
    *,
    loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """joblib round trip; eval-mode probabilities must be exactly equal."""
    import joblib

    path = Path(path)
    if path.exists():
        raise RQ4bReloadError(f"Refusing to overwrite serialized spike model {path}.")
    before = estimator.predict_proba(X)
    joblib.dump(estimator, path)
    try:
        reloaded = (loader or joblib.load)(path)
        after = reloaded.predict_proba(X)
    except Exception as exc:
        raise RQ4bReloadError(f"Serialized spike model failed to reload: {type(exc).__name__}: {exc}") from exc
    if before.shape != after.shape or not np.array_equal(before, after):
        raise RQ4bReloadError("Reloaded spike model changed eval-mode probabilities.")
    module = reloaded.backend_.alg_interface_.model
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "probabilities_exactly_equal": True,
        "rows": int(len(after)),
        "predictor_training_mode_after_reload": bool(module.model.training),
        "adversary_present_after_reload": hasattr(module, "rq4b_adversary"),
        "training_metadata_pickled": any(
            name in module.__dict__ for name in ("_rq4b_groups", "_rq4b_labels", "_rq4b_adversary_opt", "_rq4b_raw_ds")
        )
        or any(getattr(module, name, None) is not None for name in ("_rq4b_group_codes_np", "_rq4b_labels_np", "_rq4b_features_np")),
        "predictor_devices_after_reload": predictor_devices(reloaded),
    }


# ---------------------------------------------------------------------------
# Synthetic epoch selection and CE helper
# ---------------------------------------------------------------------------


def stable_binary_ce_from_margin(margin: np.ndarray, y: np.ndarray) -> float:
    """Unweighted mean binary CE (Approved=1) from the two-class logit margin."""
    margin = np.asarray(margin, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, margin) - y * margin))


def select_epoch(curves: Mapping[int, Sequence[float]] | np.ndarray, *, epochs: int = PRODUCTION_EPOCH_HORIZON, folds: int = EPOCH_SELECTION_FOLDS) -> dict[str, Any]:
    """Equal-weight fold mean CE; minimum unrounded mean; earliest exact tie."""
    if isinstance(curves, Mapping):
        if sorted(curves) != list(range(1, folds + 1)):
            raise RQ4bEpochSelectionError(f"Epoch selection requires folds 1..{folds}; got {sorted(curves)}.")
        rows = []
        for fold in range(1, folds + 1):
            values = list(curves[fold])
            if len(values) != epochs or any(value is None for value in values):
                raise RQ4bEpochSelectionError(f"Fold {fold} must contain {epochs} present values.")
            rows.append(values)
        matrix = np.asarray(rows, dtype=np.float64)
    else:
        matrix = np.asarray(curves, dtype=np.float64)
    if matrix.shape != (folds, epochs):
        raise RQ4bEpochSelectionError(f"Epoch-selection curves must have shape ({folds}, {epochs}).")
    if not np.isfinite(matrix).all():
        raise RQ4bEpochSelectionError("Epoch-selection curves contain missing or nonfinite values.")
    mean_curve = matrix.mean(axis=0)
    minimum = float(mean_curve.min())
    tied = np.flatnonzero(mean_curve == minimum)
    selected = int(tied[0]) + 1
    return {
        "selected_epoch": selected,
        "minimum_mean_ce": minimum,
        "tied_epochs": (tied + 1).tolist(),
        "epoch_cap_hit": selected == epochs,
        "finite_values": int(matrix.size),
        "mean_curve": mean_curve.tolist(),
    }


# ---------------------------------------------------------------------------
# Evidence runner
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_once(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_failure_receipt(attempt_root: Path, *, gate: str, category: str, message: str) -> Path:
    """Write an immutable (exclusive-create, read-only) failure receipt."""
    if category not in ("environment_failure", "implementation_defect", "unsupported_design"):
        raise ValueError("Unknown failure category.")
    path = Path(attempt_root) / "FAILURE_RECEIPT.json"
    write_json_once(path, {"gate": gate, "category": category, "message": message, "timestamp": _utc_now()})
    os.chmod(path, stat.S_IREAD)
    return path


def create_attempt_root(name: str | None = None, *, root: Path = DISPOSABLE_ROOT) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = Path(root) / (name or f"attempt_{stamp}")
    if attempt.resolve().parent != Path(root).resolve():
        raise RQ4bSpikeError("Attempt directory must be a direct child of the disposable root.")
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _fit_summary(estimator: SpikeAdversarialRealMLPClassifier, observer: SpikeObserver) -> dict[str, Any]:
    log = estimator.rq4b_log_
    summary = {
        "completed_epochs": observer.completed_epochs,
        "batches": len(observer.batch_fingerprints),
        "post_fit_placement": predictor_devices(estimator),
        "predict_batch_size": int(estimator.backend_.alg_interface_.model.creator.config["predict_batch_size"]),
    }
    if log is not None:
        summary.update(
            {key: log[key] for key in ("lambda", "adversary_seed", "adversary_updates", "adversary_skipped_empty_mask", "adversary_bypassed_lambda_zero", "devices")},
            adversary_rng_isolation=log.get("adversary_rng_isolation"),
        )
    return summary


GATE_STATUSES = ("PASS", "FAIL", "UNRESOLVED", "NOT_RUN")


def run_spike(attempt_root: Path, *, upstream_dir: Path | None, seed: int = 42, diagnostic_lambda: float = 0.1) -> dict[str, Any]:
    """Run all gates; any terminating error leaves an immutable receipt naming the active gate."""
    attempt_root = Path(attempt_root)
    state: dict[str, Any] = {"gate": "setup", "gates": {}}
    try:
        return _run_spike_gates(attempt_root, state, upstream_dir=upstream_dir, seed=seed, diagnostic_lambda=diagnostic_lambda)
    except Exception as exc:
        if not (attempt_root / "FAILURE_RECEIPT.json").exists():
            category = "environment_failure" if isinstance(exc, RQ4bCudaUnavailableError) else "implementation_defect"
            state["gates"][state["gate"]] = {"status": "FAIL", "mandatory": True, "error": f"{type(exc).__name__}: {exc}"}
            if not (attempt_root / "gates.json").exists():
                write_json_once(attempt_root / "gates.json", state["gates"])
            write_failure_receipt(attempt_root, gate=state["gate"], category=category, message=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, RQ4bSpikeError):
            raise
        raise RQ4bSpikeError(f"Spike attempt terminated at {state['gate']}: {type(exc).__name__}: {exc}") from exc


def _run_spike_gates(attempt_root: Path, state: dict[str, Any], *, upstream_dir: Path | None, seed: int, diagnostic_lambda: float) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = state["gates"]

    def stage(gate: str) -> None:
        state["gate"] = gate

    def record(gate: str, status: str, evidence: Any, *, mandatory: bool = True, category: str = "implementation_defect") -> None:
        """Mandatory gates proceed only on PASS; unknown statuses always terminate."""
        state["gate"] = gate
        gates[gate] = {"status": status, "mandatory": mandatory}
        write_json_once(attempt_root / f"{gate}.json", {"gate": gate, "status": status, "evidence": evidence, "timestamp": _utc_now()})
        if status not in GATE_STATUSES:
            write_json_once(attempt_root / "gates.json", gates)
            write_failure_receipt(attempt_root, gate=gate, category="implementation_defect", message=f"Unexpected gate status {status!r}.")
            raise RQ4bSpikeError(f"Unexpected status {status!r} for gate {gate}")
        if mandatory and status != "PASS":
            write_json_once(attempt_root / "gates.json", gates)
            write_failure_receipt(
                attempt_root, gate=gate, category=category,
                message=f"Mandatory gate status {status}: " + json.dumps(_jsonable(evidence))[:4000],
            )
            raise RQ4bSpikeError(f"Mandatory spike gate {gate} is {status}; attempt terminated.")

    write_json_once(attempt_root / "invocation.json", {"argv": sys.argv, "cwd": os.getcwd(), "started": _utc_now(), "seed": seed, "diagnostic_lambda": diagnostic_lambda})
    write_json_once(attempt_root / "environment.json", collect_environment())

    stage("g00_contract")
    contract = load_rq4b_spike_contract()
    record("g00_contract", "PASS", {"protocol_version": contract["protocol_version"], "status": contract["status"], "technical_spike": contract["technical_spike"], "runtime": contract["runtime"]})

    stage("g01_pin_pytabkit_source_revision")
    lineage = pytabkit_installed_lineage(upstream_dir)
    record("g01_pin_pytabkit_source_revision", lineage["lineage_status"], lineage, category="environment_failure")

    stage("g02_cuda_preflight")
    try:
        cuda = require_cuda0()
    except RQ4bCudaUnavailableError as exc:
        record("g02_cuda_preflight", "FAIL", {"error": str(exc)}, category="environment_failure")
    record("g02_cuda_preflight", "PASS", cuda)

    fixture = make_synthetic_fixture()
    fixture_info = validate_fixture(fixture)
    write_json_once(attempt_root / "fixture.json", fixture_info)
    config = spike_estimator_parameters(seed, MAX_SMOKE_EPOCHS)
    write_json_once(attempt_root / "configuration.json", {"estimator_parameters": config, "seed": seed, "adversary_seed": seed, "diagnostic_lambda": diagnostic_lambda})

    stage("g03_training_step_and_score_tensor")
    # Gradient diagnostics on a lambda>0 batch with lr > 0.
    diag_model, diag_observer = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=diagnostic_lambda, record_gradient_diagnostics=True)
    diag = diag_model.rq4b_log_["gradient_diagnostics"]
    grad_ok = bool(
        diag
        and diag["logits_shape"] == [1, SPIKE_BATCH_SIZE, 2]
        and diag["margin_shape"] == [SPIKE_BATCH_SIZE]
        and diag["signed_identity_holds"]
        and diag["additive_variant_max_abs_diff"] > 0
        and diag["adversary_logit1_grad_nonzero_g3_rows"] > 0
        and diag["total_minus_pred_opposes_adv_g3_rows"] == diag["adversary_logit1_grad_nonzero_g3_rows"]
        and diag["adversary_grad_outside_g3_max_abs"] == 0.0
        and diag["adversary_margin_grad_outside_g3_max_abs"] == 0.0
        and diag["adversary_grad_g3_nonzero_rows"] > 0
        and diag["prediction_grad_outside_g3_nonzero_rows"] == diag["outside_rows"]
        and diag["predictor_unchanged_by_adversary_step"]
        and diag["predictor_grads_empty_after_adversary_backward"]
        and diag["adversary_max_change_by_adversary_step"] > 0
        and diag["adversary_unchanged_by_predictor_step"]
        and diag["adversary_grads_unchanged_by_predictor_backward"]
        and diag["predictor_max_change_by_predictor_step"] > 0
        and not diag["adversary_params_in_predictor_optimizer"]
    )
    devices = diag_model.rq4b_log_["devices"]
    device_ok = (
        all(value == SPIKE_DEVICE for value in devices.values() if isinstance(value, str))
        and all(devices[key] == [SPIKE_DEVICE] for key in ("predictor_parameters", "adversary_parameters"))
        and devices["train_loader"]["batch_size"] == SPIKE_BATCH_SIZE
        and devices["train_loader"]["drop_last"] is True
        and _fit_summary(diag_model, diag_observer)["predict_batch_size"] == SPIKE_PREDICT_BATCH_SIZE
    )
    record("g03_training_step_and_score_tensor", "PASS" if grad_ok and device_ok else "FAIL", {"gradient_diagnostics": diag, "fit": _fit_summary(diag_model, diag_observer)})

    stage("g04_lambda_zero_equivalence")
    # Lambda=0 equivalence: plain stock, observed stock, extension.
    stock = fit_stock_realmlp(fixture, seed=seed)
    observed, observed_obs = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=0.0, module_kind="stock_observed")
    extension, extension_obs = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=0.0)
    p_stock, p_observed, p_ext = (m.predict_proba(fixture.X) for m in (stock, observed, extension))
    batch_rows = [batch["rows"] for batch in extension.rq4b_log_["batches"]]
    n_batches = len(fixture.X) // SPIKE_BATCH_SIZE
    digests = [m.rq4b_input_digest_ for m in (stock, observed, extension)]
    exact = {
        "record_ids_equal": len({d["record_ids"] for d in digests}) == 1,
        "inputs_X_equal": len({(d["X"], d["X_dtypes"]) for d in digests}) == 1,
        "labels_equal": len({d["labels"] for d in digests}) == 1,
        "masks_equal": len({d["masks"] for d in digests}) == 1,
        "batch_order_fingerprints_equal": observed_obs.batch_fingerprints == extension_obs.batch_fingerprints,
        "batch_count": len(extension_obs.batch_fingerprints),
        "configuration_equal": stock.backend_parameters() == extension.backend_parameters() == observed.backend_parameters(),
        "backend_get_params_equal": stock.backend_.get_params() == extension.backend_.get_params(),
        "batches_per_epoch": n_batches,
        "batch_rows_unique_within_epoch": all(
            len(set(sum(batch_rows[i : i + n_batches], []))) == n_batches * SPIKE_BATCH_SIZE
            for i in range(0, len(batch_rows), n_batches)
        ),
        "rows_dropped_per_epoch_drop_last": len(fixture.X) - n_batches * SPIKE_BATCH_SIZE,
    }
    exact["input_digests"] = digests[0]
    exact["record_ids_of_first_batch"] = fixture.record_ids[batch_rows[0][:5]].tolist()
    neutrality = {
        "params": compare_tensors(predictor_tensors(stock), predictor_tensors(observed), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "probabilities": compare_probabilities(p_stock, p_observed, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
    }
    equivalence = {
        "params_vs_stock": compare_tensors(predictor_tensors(stock), predictor_tensors(extension), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "probabilities_vs_stock": compare_probabilities(p_stock, p_ext, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "params_vs_observed_stock": compare_tensors(predictor_tensors(observed), predictor_tensors(extension), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "probabilities_vs_observed_stock": compare_probabilities(p_observed, p_ext, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
    }
    eq_ok = (
        all(value for key, value in exact.items() if isinstance(value, bool))
        and all(item["within_tolerance"] for item in equivalence.values())
        and extension.rq4b_log_["adversary_updates"] == 0
        and all(extension.rq4b_log_["adversary_rng_isolation"].values())
    )
    record(
        "g04_lambda_zero_equivalence",
        "PASS" if eq_ok else "FAIL",
        {"exact": exact, "equivalence": equivalence, "observer_neutrality": neutrality, "extension": _fit_summary(extension, extension_obs)},
    )

    stage("g05_seed_plumbing_and_repeatability")
    # Same-seed repeatability (lambda=0 and lambda>0), predeclared baseline tolerance.
    repeat_zero, _ = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=0.0)
    repeat_a, obs_a = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=diagnostic_lambda)
    repeat_b, obs_b = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=diagnostic_lambda)
    pa, pb = repeat_a.predict_proba(fixture.X), repeat_b.predict_proba(fixture.X)
    repeat = {
        "lambda_zero_params": compare_tensors(predictor_tensors(extension), predictor_tensors(repeat_zero), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "lambda_zero_probabilities": compare_probabilities(p_ext, repeat_zero.predict_proba(fixture.X), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "lambda_positive_params": compare_tensors(predictor_tensors(repeat_a), predictor_tensors(repeat_b), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "lambda_positive_probabilities": compare_probabilities(pa, pb, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "lambda_positive_batch_order_equal": obs_a.batch_fingerprints == obs_b.batch_fingerprints,
        "lambda_positive_adversary_updates": [repeat_a.rq4b_log_["adversary_updates"], repeat_b.rq4b_log_["adversary_updates"]],
        "lambda_positive_vs_lambda_zero_probability_max_abs_diff": float(np.max(np.abs(pa - p_ext))),
        "scope": "two-epoch synthetic smoke only; not full-run determinism",
    }
    rep_ok = all(item["within_tolerance"] for key, item in repeat.items() if isinstance(item, dict)) and repeat["lambda_positive_batch_order_equal"]
    record("g05_seed_plumbing_and_repeatability", "PASS" if rep_ok else "FAIL", repeat)

    stage("g06_serialization_and_reload")
    # Serialization/reload (lambda>0 extension and lambda=0 extension).
    reload_evidence = {
        "lambda_positive": serialize_and_verify_reload(repeat_a, fixture.X, attempt_root / "reload_lambda_positive.joblib"),
        "lambda_zero": serialize_and_verify_reload(extension, fixture.X, attempt_root / "reload_lambda_zero.joblib"),
    }
    reload_ok = all(
        not item["training_metadata_pickled"] and item["predictor_devices_after_reload"]["interface_device"] == SPIKE_DEVICE
        for item in reload_evidence.values()
    )
    record("g06_serialization_and_reload", "PASS" if reload_ok else "FAIL", reload_evidence)

    stage("g07_empty_mask")
    # Empty roster mask: every row outside G3.
    empty_codes = np.full(len(fixture.X), OUTSIDE_ROSTER_CODE, dtype=np.int64)
    empty_model, empty_obs = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=diagnostic_lambda, group_codes=empty_codes)
    empty_log = empty_model.rq4b_log_
    empty_change = compare_tensors(predictor_tensors(stock), predictor_tensors(empty_model), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL)
    empty_ok = empty_log["adversary_updates"] == 0 and empty_log["adversary_skipped_empty_mask"] == len(empty_obs.batch_fingerprints) > 0
    record("g07_empty_mask", "PASS" if empty_ok else "FAIL", {"log": _fit_summary(empty_model, empty_obs), "predictor_vs_stock_lambda_zero": empty_change})

    stage("g08_synthetic_epoch_selection")
    # Synthetic epoch selection.
    rng = np.random.default_rng(seed)
    base = 0.7 - 0.2 * np.log1p(np.arange(PRODUCTION_EPOCH_HORIZON)) / np.log(PRODUCTION_EPOCH_HORIZON) + 0.0005 * (np.arange(PRODUCTION_EPOCH_HORIZON) - 150).clip(0)
    curves = base + rng.normal(scale=0.002, size=(EPOCH_SELECTION_FOLDS, PRODUCTION_EPOCH_HORIZON))
    selection = select_epoch(curves)
    tie = np.ones((5, 256))
    tie[:, [9, 19]] = 0.5
    cap = np.tile(np.linspace(1.0, 0.5, 256), (5, 1))
    unequal = np.ones((5, 256))
    unequal[0, 3] = 0.0  # equal fold weighting moves mean by 0.2
    unequal[1:, 5] = 0.76
    rejections = {}
    for label, bad in {"nan": np.where(np.eye(5, 256, dtype=bool), np.nan, curves), "short": curves[:, :255], "four_folds": curves[:4]}.items():
        try:
            select_epoch(bad)
            rejections[label] = "accepted"
        except RQ4bEpochSelectionError:
            rejections[label] = "rejected"
    selector = {
        "synthetic": {key: selection[key] for key in ("selected_epoch", "minimum_mean_ce", "epoch_cap_hit", "finite_values")},
        "earliest_tie": select_epoch(tie)["selected_epoch"],
        "cap": select_epoch(cap)["epoch_cap_hit"],
        "equal_weighting_selected": select_epoch(unequal)["selected_epoch"],
        "rejections": rejections,
    }
    sel_ok = selector["earliest_tie"] == 10 and selector["cap"] is True and selector["equal_weighting_selected"] == 4 and set(rejections.values()) == {"rejected"}
    record("g08_synthetic_epoch_selection", "PASS" if sel_ok else "FAIL", selector)

    stage("g09_schedule_horizon_feasibility")
    # Schedule-horizon feasibility probe (two executed epochs each).
    horizon_2, obs_h2 = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=0.0, module_kind="stock_observed")
    native_stop, obs_ns = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=0.0, module_kind="stock_observed", n_epochs=PRODUCTION_EPOCH_HORIZON, native_stop_epoch=MAX_SMOKE_EPOCHS)
    split_budget, obs_sb = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=0.0, module_kind="stock_observed", schedule_horizon_epochs=PRODUCTION_EPOCH_HORIZON)
    split_adv, obs_sa = fit_spike_realmlp(fixture, seed=seed, adversary_lambda=diagnostic_lambda, schedule_horizon_epochs=PRODUCTION_EPOCH_HORIZON)

    def _last(obs: SpikeObserver) -> dict[str, Any]:
        return obs.hypers[-1]

    probe = {
        "horizon_2_n_epochs_2": {"completed_epochs": obs_h2.completed_epochs, "last_batch": _last(obs_h2)},
        "native_stop_epoch_2_n_epochs_256": {"completed_epochs": obs_ns.completed_epochs, "last_batch": _last(obs_ns)},
        "trainer_budget_2_module_horizon_256": {"completed_epochs": obs_sb.completed_epochs, "last_batch": _last(obs_sb)},
        "adversarial_trainer_budget_2_module_horizon_256": {"completed_epochs": obs_sa.completed_epochs, "last_batch": _last(obs_sa)},
        "native_stop_vs_split_budget_hypers_equal": obs_ns.hypers == obs_sb.hypers,
        "native_stop_vs_split_budget_params": compare_tensors(predictor_tensors(native_stop), predictor_tensors(split_budget), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "horizon_2_vs_256_hypers_equal": obs_h2.hypers == obs_sb.hypers,
        "horizon_2_vs_256_params": compare_tensors(predictor_tensors(horizon_2), predictor_tensors(split_budget), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL),
        "fit_progress_matches_256_horizon": all(
            abs(item["fit_progress"] * PRODUCTION_EPOCH_HORIZON - (item["epoch"] - 1 + item["batch_idx"] / n_batches)) < 1e-9
            for item in obs_sb.hypers
        ),
        "limitations": [
            "Only two executed epochs were probed; behaviour at selected epochs 3..256 is not executed.",
            "native stop_epoch uses StopAtEpochsCallback (refit path) whose final restore_all is a no-op without a saved checkpoint.",
            "Existing project _fit_model sets n_epochs=epochs, which rescales lr/wd/p_drop schedules to the executed budget.",
            "Schedule policy choice remains a user decision; this probe is feasibility evidence only.",
        ],
    }
    probe_ok = (
        all(item["completed_epochs"] == MAX_SMOKE_EPOCHS for key, item in probe.items() if isinstance(item, dict) and "completed_epochs" in item)
        and probe["native_stop_vs_split_budget_hypers_equal"]
        and not probe["horizon_2_vs_256_hypers_equal"]
        and probe["fit_progress_matches_256_horizon"]
    )
    record("g09_schedule_horizon_feasibility", "PASS" if probe_ok else "FAIL", probe, mandatory=False)

    stage("g11_row_feature_group_alignment")
    align = make_alignment_fixture()
    align_model, align_obs = fit_spike_realmlp(
        align, seed=seed, adversary_lambda=diagnostic_lambda,
        group_feature_column=ALIGNMENT_GROUP_COLUMN, log_raw_batch_features=True,
    )
    mismatches = {"record_id_vs_feature": 0, "group_vs_feature": 0, "group_vs_external_metadata": 0, "label_vs_external": 0}
    same_outcome_distinct_group_pairs = 0
    for batch in align_model.rq4b_log_["batches"]:
        raw = np.asarray(batch["raw_x_cont"], dtype=np.float32)
        rows = np.asarray(batch["rows"], dtype=np.int64)
        mismatches["record_id_vs_feature"] += int(
            (align.record_ids[rows] != np.asarray([f"SYN-{int(v):06d}" for v in raw[:, ALIGNMENT_ID_COLUMN]], dtype=object)).sum()
        )
        mismatches["group_vs_feature"] += int((raw[:, ALIGNMENT_GROUP_COLUMN].astype(np.int64) != np.asarray(batch["groups"])).sum())
        mismatches["group_vs_external_metadata"] += int((align.group_codes[rows] != np.asarray(batch["groups"])).sum())
        labels = align.y[rows]
        groups = np.asarray(batch["groups"])
        for label in (0, 1):
            same = groups[labels == label]
            same_outcome_distinct_group_pairs += int(len(same) * (len(same) - 1) // 2 - sum(
                int(c) * (int(c) - 1) // 2 for c in np.unique(same, return_counts=True)[1]
            ))
    swapped_groups, perm, swap_info = label_preserving_group_swap(align)
    corruption = {}
    for label, kwargs in {
        "label_preserving_group_swap": {"group_codes": swapped_groups},
        "label_preserving_metadata_row_swap": {
            "group_codes": align.group_codes[perm],
            "alignment_features": native_numeric_features(align.X)[perm],
        },
    }.items():
        try:
            fit_spike_realmlp(align, seed=seed, adversary_lambda=diagnostic_lambda, group_feature_column=ALIGNMENT_GROUP_COLUMN, **kwargs)
            corruption[label] = "accepted"
        except RQ4bAlignmentError as exc:
            corruption[label] = f"rejected: {exc}"
    alignment_log = align_model.rq4b_log_["alignment"]
    align_ok = (
        not any(mismatches.values())
        and same_outcome_distinct_group_pairs > 0
        and alignment_log["dataset_rows_verified"] == len(align.X)
        and alignment_log["batches_feature_verified"] == alignment_log["batches_group_feature_verified"] == len(align_obs.batch_fingerprints) > 0
        and all(value.startswith("rejected") for value in corruption.values())
        and np.array_equal(align.y[perm], align.y)
    )
    record("g11_row_feature_group_alignment", "PASS" if align_ok else "FAIL", {
        "mismatches": mismatches,
        "same_outcome_distinct_group_pairs_in_batches": same_outcome_distinct_group_pairs,
        "in_step_alignment_log": alignment_log,
        "corruption_swap": swap_info,
        "corruption_results": corruption,
        "fit": _fit_summary(align_model, align_obs),
    })

    record("g10_cuda_fail_closed_and_nonfinite_paths", "NOT_RUN", {"covered_by": "tests/test_dataset_b_rq4b_spike.py"}, mandatory=False)
    gates["naming_conclusion"] = {
        "status": "PRELIMINARY_ONLY",
        "note": "Implementer results only; adversarial RealMLP naming requires independent Codex acceptance.",
    }
    write_json_once(attempt_root / "gates.json", gates)
    return gates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the disposable RQ4b internal-extension spike.")
    parser.add_argument("--attempt-name", default=None)
    parser.add_argument("--upstream-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    attempt_root = create_attempt_root(args.attempt_name)
    print(f"RQ4B_SPIKE_ATTEMPT {attempt_root}", flush=True)
    started = time.perf_counter()
    try:
        gates = run_spike(attempt_root, upstream_dir=args.upstream_dir, seed=args.seed)
    except RQ4bSpikeError as exc:
        print(f"RQ4B_SPIKE_FAILED {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(json.dumps({"gates": gates, "elapsed_seconds": time.perf_counter() - started}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
