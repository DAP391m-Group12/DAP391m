"""RQ4b-3 staged runner, artifact layout and package validation.

Layout (one run per unique ``run_id`` under the declared RQ4b namespace)::

    <namespace>/<run_id>/
        run_manifest.json                  run identity, mode, contract/code lineage
        stages/<stage_key>/receipt.json    PASS receipt: inputs, outputs (SHA-256), counts
        stages/<stage_key>/...             stage payload files
        failures/<stage_key>.<uuid>/       retained failed attempts + FAILURE_RECEIPT.json

A stage is written into ``stages/.tmp_<key>_<uuid>`` and published by a single
directory rename only after its receipt is written; a published stage is
never overwritten or resumed. Every stage is one explicit unit of work (no
command runs the whole grid). Stage prerequisites fail closed: Test is opened
only by ``test_evaluation`` after the selection lock and every final-arm
threshold lock are published and verified.

Runs are either ``production`` (root must be the declared RQ4b namespace,
``ProductionBackend`` only, explicit CLI confirmation) or ``synthetic_test``
(root under scratch/rq4b_implementation). Receipts carry the mode, and package
validation of a production run rejects any synthetic receipt.

Gate C (``--stage gate_c_aggregation``) is evaluation-only: it verifies the
exact pins of the three immutable source runs and their nine units, evaluates
the saved Validation predictions and publishes one aggregation run atomically.
It constructs no backend, so it cannot fit, infer or open Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.dataset_b.mitigation import rq4b_evaluation as ev
from src.dataset_b.mitigation import rq4b_inputs as ri
from src.utils.path import PROJECT_ROOT


RUN_ID_PATTERN = re.compile(r"^rq4b3_adv_\d{8}_\d{2}$")
GATE_C_RUN_ID_PATTERN = re.compile(r"^rq4b3_gatec_\d{8}_\d{2}$")
TEST_RUN_ID_PATTERN = re.compile(r"^rq4b3_test_\d{8}_\d{2}$")
MODE_PRODUCTION = "production"
MODE_SYNTHETIC = "synthetic_test"
SYNTHETIC_ROOT = PROJECT_ROOT / "scratch" / "rq4b_implementation"
RECEIPT_FILE = "receipt.json"
FAILURE_FILE = "FAILURE_RECEIPT.json"
MANIFEST_FILE = "run_manifest.json"
REQUIRED_ACCEPTANCE = (
    "t1_input_boundary_acceptance",
    "t2_training_engine_acceptance",
    "t3_capability_probe_acceptance",
    "t4_evaluation_logic_acceptance",
    "t5_runner_acceptance",
)
CODE_FILES = ("rq4b_inputs.py", "rq4b_training.py", "rq4b_probe.py", "rq4b_evaluation.py", "rq4b_runner.py")
PREDICTION_COLUMNS = ("record_id", "y_approved", "p_approved", "derived_race", "applicant_sex_group")
_BUDGET = ri.APPROVED_CONTRACT["fit_budget"]
CV_FIT_BUDGET = int(_BUDGET["rq4b_3_maximum_cv_fold_fits_excluding_spike_and_retries"])
REFIT_BUDGET = int(_BUDGET["rq4b_3_maximum_full_train_refits_excluding_spike_and_retries"])
CV_FOLDS = int(ri.APPROVED_CONTRACT["epoch_selection"]["splitter"]["folds"])


class RQ4bRunnerError(RuntimeError):
    """Fail-closed runner error."""


class RQ4bStageOrderError(RQ4bRunnerError):
    """A stage was requested before its prerequisites were published and verified."""


class RQ4bPackageError(RQ4bRunnerError):
    """A published package file is missing, corrupt, unexpected or inconsistent."""


def _require(condition: bool, message: str, error: type[RQ4bRunnerError] = RQ4bRunnerError) -> None:
    if not condition:
        raise error(message)


# ---------------------------------------------------------------------------
# Hashing and JSON
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(values: Sequence[str]) -> str:
    """Same convention as the RQ1 bootstrap manifest: each ID followed by a newline."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return jsonable({name: getattr(value, name) for name in value.__dataclass_fields__})
    return value


def write_json_new(path: Path, payload: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RQ4bPackageError(f"Cannot read JSON {path}: {exc}") from exc


def code_lineage() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in CODE_FILES}


def lam_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def unit_key(prefix: str, adversary_lambda: float, seed: int) -> str:
    return f"{prefix}_l{lam_key(adversary_lambda)}_s{int(seed)}"


# ---------------------------------------------------------------------------
# Run layout and atomic, non-overwriting stage publication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_id: str
    mode: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def stages(self) -> Path:
        return self.run_dir / "stages"

    @property
    def failures(self) -> Path:
        return self.run_dir / "failures"

    def stage_dir(self, key: str) -> Path:
        _require(re.fullmatch(r"[a-z0-9_]+", key) is not None, f"Invalid stage key {key!r}.")
        return self.stages / key


def _check_root(root: Path, mode: str, contract: ri.RQ4bContract) -> Path:
    resolved = Path(root).resolve()
    if mode == MODE_PRODUCTION:
        _require(resolved == contract.artifact_namespace.resolve(), "Production runs must use the declared RQ4b artifact namespace.")
    elif mode == MODE_SYNTHETIC:
        _require(SYNTHETIC_ROOT.resolve() in resolved.parents, "Synthetic runs must live under scratch/rq4b_implementation.")
    else:
        raise RQ4bRunnerError(f"Unknown run mode {mode!r}.")
    return resolved


def execution_lineage(contract: ri.RQ4bContract) -> dict[str, Any]:
    """Locked continuation identity; source runs are immutable evidence only."""
    return {
        "role": contract.production_run_role,
        "track_a_source_run_id": contract.track_a_source_run_id,
        "track_b_seed43_source_run_id": contract.track_b_child_run_id,
        "seed44_continuation_run_id": contract.production_run_id,
        "source_runs_are_immutable": True,
        "fitted_state_reuse": "prohibited",
        "migration_policy": "no_migration_fresh_seed44_fit_state",
        "source_run_evidence": contract.source_run_evidence,
        "cross_run_aggregation_status": "blocked_until_gate_c_lineage_validation",
    }


def verify_source_run_evidence(root: Path, contract: ri.RQ4bContract) -> None:
    """Fail closed unless every pinned source manifest and receipt is unchanged."""
    for label, evidence in contract.source_run_evidence.items():
        run_dir = Path(root) / evidence["run_id"]
        manifest = run_dir / MANIFEST_FILE
        _require(manifest.is_file(), f"Pinned {label} source manifest is missing.", RQ4bPackageError)
        _require(
            sha256_file(manifest) == evidence["manifest_sha256"],
            f"Pinned {label} source manifest digest differs.",
            RQ4bPackageError,
        )
        for digest_field in ("receipt_sha256", "failure_receipt_sha256"):
            for relative, expected in evidence.get(digest_field, {}).items():
                receipt = run_dir / relative
                _require(receipt.is_file(), f"Pinned {label} source receipt is missing: {relative}.", RQ4bPackageError)
                _require(
                    sha256_file(receipt) == expected,
                    f"Pinned {label} source receipt digest differs: {relative}.",
                    RQ4bPackageError,
                )


def create_run(root: Path, run_id: str, mode: str, contract: ri.RQ4bContract, *, config_path: Path = ri.RQ4B_CONFIG_PATH) -> RunLayout:
    _require(isinstance(run_id, str) and RUN_ID_PATTERN.fullmatch(run_id) is not None, "run_id must match rq4b3_adv_<YYYYMMDD>_<NN>.")
    if mode == MODE_PRODUCTION:
        _require(run_id == contract.production_run_id,
                 f"Production seed-44 work must use the approved continuation run {contract.production_run_id}; source runs are immutable.")
        verify_source_run_evidence(root, contract)
    layout = RunLayout(_check_root(root, mode, contract), run_id, mode)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.run_dir.mkdir(exist_ok=False)  # unique run identity; never reused
    layout.stages.mkdir()
    layout.failures.mkdir()
    write_json_new(layout.run_dir / MANIFEST_FILE, {
        "run_id": run_id,
        "mode": mode,
        "configuration": "RQ4b-3",
        "protocol_version": contract.protocol_version,
        "decision_revision": contract.decision_revision,
        "execution_lineage": execution_lineage(contract),
        "config_sha256": sha256_file(config_path),
        "code_sha256": code_lineage(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    return layout


def open_run(root: Path, run_id: str, mode: str, contract: ri.RQ4bContract) -> RunLayout:
    if mode == MODE_PRODUCTION:
        _require(run_id == contract.production_run_id,
                 f"Production seed-44 work must use the approved continuation run {contract.production_run_id}; source runs are immutable.",
                 RQ4bPackageError)
        verify_source_run_evidence(root, contract)
    layout = RunLayout(_check_root(root, mode, contract), run_id, mode)
    manifest = read_json(layout.run_dir / MANIFEST_FILE)
    _require(manifest.get("run_id") == run_id and manifest.get("mode") == mode, "Run manifest identity or mode differs.", RQ4bPackageError)
    _require(manifest.get("protocol_version") == contract.protocol_version, "Run manifest protocol differs from the active contract.", RQ4bPackageError)
    _require(manifest.get("decision_revision") == contract.decision_revision, "Run manifest decision revision differs from the active contract.", RQ4bPackageError)
    _require(manifest.get("execution_lineage") == execution_lineage(contract), "Run manifest execution lineage differs from the approved Track A/Track B relationship.", RQ4bPackageError)
    _require(manifest.get("config_sha256") == sha256_file(ri.RQ4B_CONFIG_PATH), "RQ4b config changed since the run was created; runs are never resumed across contract changes.", RQ4bPackageError)
    _require(manifest.get("code_sha256") == code_lineage(), "RQ4b code changed since the run was created; runs are never resumed across code changes.", RQ4bPackageError)
    return layout


class StageWriter:
    """Write one stage unit into a private temporary directory, then publish it atomically."""

    def __init__(self, layout: RunLayout, key: str, *, stage: str, unit: Mapping[str, Any], inputs: Mapping[str, str],
                 allow_new_attempt_after_failure: bool = False) -> None:
        self.layout, self.key, self.stage, self.unit, self.inputs = layout, key, stage, dict(unit), dict(inputs)
        _require(not layout.stage_dir(key).exists(), f"Stage {key} is already published; it is never overwritten or resumed.", RQ4bStageOrderError)
        prior = sorted(p.name for p in layout.failures.glob(f"{key}.*"))
        _require(not prior or allow_new_attempt_after_failure is True,
                 f"Stage {key} has failed attempts {prior}; a new attempt must be requested explicitly.", RQ4bStageOrderError)
        self.prior_failures = prior
        self.temp = layout.stages / f".tmp_{key}_{uuid.uuid4().hex}"
        self.counts: dict[str, Any] = {"cv_fold_fits": 0, "refit_fits": 0, "probe_fits": 0, "predictions": {}}
        self.started = datetime.now(timezone.utc).isoformat()
        self.attempt_id = self.temp.name

    def __enter__(self) -> "StageWriter":
        self.temp.mkdir(exist_ok=False)
        if self.stage.endswith("_fit"):
            self.progress_event("stage_started", phase=self.stage, optimizer_progress_claimed=False)
        return self

    def progress_event(self, event: str, *, phase: str, **values: Any) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp_utc": timestamp, "last_progress_utc": timestamp,
            "run_id": self.layout.run_id, "attempt_id": self.attempt_id,
            "unit": self.key, "adversary_lambda": self.unit.get("adversary_lambda"),
            "lambda": self.unit.get("adversary_lambda"), "model_seed": self.unit.get("model_seed"),
            "seed": self.unit.get("model_seed"), "phase": phase,
            "fold": values.pop("fold", None), "epoch": values.pop("epoch", None),
            "batch": values.pop("batch", None), "event": event, **values,
        }
        line = json.dumps(jsonable(record), sort_keys=True) + "\n"
        for name in ("run.log", "training_history.jsonl"):
            with (self.temp / name).open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        print(line.rstrip(), file=sys.stderr, flush=True)

    def _sync_fit_ledger(self, directory: Path | None = None) -> None:
        ledger_path = (self.temp if directory is None else directory) / "fit_ledger.json"
        if not ledger_path.is_file():
            return
        ledger = read_json(ledger_path)
        self.counts["fit_accounting"] = {
            "planned": ledger.get("planned"), "started": ledger.get("started"),
            "completed": ledger.get("completed"), "partial": ledger.get("partial"),
            "unknown": ledger.get("unknown", False), "last_position": ledger.get("last_position"),
            "last_progress_utc": ledger.get("last_progress_utc"), "source": "fit_ledger.json",
        }

    def path(self, name: str) -> Path:
        _require("/" not in name and "\\" not in name and name not in (RECEIPT_FILE, FAILURE_FILE) and not name.startswith("."), f"Invalid output name {name!r}.")
        return self.temp / name

    def write_json(self, name: str, payload: Any) -> None:
        write_json_new(self.path(name), payload)

    def write_predictions(self, name: str, predictions: ev.PartitionPredictions) -> None:
        frame = pd.DataFrame({
            "record_id": list(predictions.record_ids), "y_approved": np.asarray(predictions.y, dtype=np.int8),
            "p_approved": np.asarray(predictions.p_approved, dtype=np.float64), "derived_race": np.asarray(predictions.race).astype(str),
            "applicant_sex_group": np.asarray(predictions.sex).astype(str),
        })
        with self.path(name).open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")

    def _publish(self) -> None:
        self._sync_fit_ledger()
        outputs = {p.name: sha256_file(p) for p in sorted(self.temp.iterdir()) if p.is_file()}
        receipt = {
            "run_id": self.layout.run_id, "mode": self.layout.mode, "stage": self.stage, "stage_key": self.key, "unit": self.unit,
            "status": "PASS", "inputs": self.inputs, "outputs": outputs, "counts": self.counts, "prior_failed_attempts": self.prior_failures,
            "code_sha256": code_lineage(), "started_utc": self.started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_new(self.temp / RECEIPT_FILE, receipt)
        _require(not self.layout.stage_dir(self.key).exists(), f"Stage {self.key} appeared during execution; refusing to overwrite.", RQ4bStageOrderError)
        os.rename(self.temp, self.layout.stage_dir(self.key))

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            try:
                self._publish()
                return False
            except BaseException as publish_exc:
                exc = publish_exc
                self._retain_failure(exc)
                raise
        self._retain_failure(exc)
        return False  # the original exception propagates unchanged

    def _retain_failure(self, exc: BaseException) -> None:
        target = self.layout.failures / f"{self.key}.{uuid.uuid4().hex}"
        try:
            if "fit_accounting" not in self.counts and self.stage.endswith("_fit"):
                self.counts["fit_accounting"] = {
                    "planned": {"cv_fold_fits": CV_FOLDS, "refit_fits": 1},
                    "started": None, "completed": None, "partial": None, "unknown": True,
                    "last_position": None, "last_progress_utc": self.started,
                    "source": "backend did not return and no readable fit ledger was available",
                }
            if self.temp.exists():
                os.rename(self.temp, target)
            else:
                target.mkdir()
            self._sync_fit_ledger(target)
            write_json_new(target / FAILURE_FILE, {
                "run_id": self.layout.run_id, "mode": self.layout.mode, "stage": self.stage, "stage_key": self.key, "unit": self.unit,
                "execution_status": "failed", "exception_type": type(exc).__name__, "exception_message": str(exc),
                "exception_chain": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "last_position": self.counts.get("fit_accounting", {}).get("last_position"),
                "last_progress_utc": self.counts.get("fit_accounting", {}).get("last_progress_utc"),
                "partial_outputs": sorted(p.name for p in target.iterdir() if p.name != FAILURE_FILE),
                "counts": self.counts, "automatic_retry": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
        except BaseException as receipt_exc:  # never mask the original failure
            exc.rq4b_failure_receipt_error = receipt_exc
            exc.add_note(f"RQ4b stage failure receipt could not be written: {type(receipt_exc).__name__}: {receipt_exc}")


def read_stage(layout: RunLayout, key: str) -> dict[str, Any]:
    """Verify a published stage: receipt present, mode/run match, every output present with matching SHA-256, no extras."""
    directory = layout.stage_dir(key)
    _require(directory.is_dir(), f"Stage {key} is not published.", RQ4bStageOrderError)
    receipt = read_json(directory / RECEIPT_FILE)
    _require(receipt.get("status") == "PASS" and receipt.get("stage_key") == key, f"Stage {key} receipt is not a PASS receipt for this key.", RQ4bPackageError)
    _require(receipt.get("run_id") == layout.run_id and receipt.get("mode") == layout.mode, f"Stage {key} belongs to another run or mode.", RQ4bPackageError)
    present = {p.name for p in directory.iterdir()} - {RECEIPT_FILE}
    expected = set(receipt["outputs"])
    _require(present == expected, f"Stage {key} files differ from its receipt: missing={sorted(expected - present)}, extra={sorted(present - expected)}.", RQ4bPackageError)
    for name, digest in receipt["outputs"].items():
        _require(sha256_file(directory / name) == digest, f"Stage {key} output {name} is corrupt (SHA-256 mismatch).", RQ4bPackageError)
    return receipt


def receipt_sha(layout: RunLayout, key: str) -> str:
    read_stage(layout, key)
    return sha256_file(layout.stage_dir(key) / RECEIPT_FILE)


def read_predictions(path: Path) -> ev.PartitionPredictions:
    frame = pd.read_csv(path, dtype={"record_id": str, "derived_race": str, "applicant_sex_group": str}, keep_default_na=False, float_precision="round_trip")
    _require(tuple(frame.columns) == PREDICTION_COLUMNS, f"Prediction file {path.name} has an unexpected schema.", RQ4bPackageError)
    return ev.PartitionPredictions(
        tuple(frame["record_id"]), frame["y_approved"].to_numpy(dtype=np.int8), frame["p_approved"].to_numpy(dtype=np.float64),
        frame["derived_race"].to_numpy(dtype=object), frame["applicant_sex_group"].to_numpy(dtype=object),
    )


# ---------------------------------------------------------------------------
# Gate C: evaluation-only cross-run aggregation (no backend, fit, inference or Test)
# ---------------------------------------------------------------------------


GATE_C_STAGE = "gate_c_aggregation"
GATE_C_SELECTION_FILE = "gate_c_selection.json"
GATE_C_LAYOUT_MODES = {MODE_PRODUCTION: "evaluation_aggregation", MODE_SYNTHETIC: "evaluation_aggregation_synthetic_test"}
GATE_C_TEST_STATUS = "blocked_pending_separate_methodological_gate"
SOURCE_SUPPORT_STAGES = ("preflight", "capability")
_DEVELOPMENT_STAGE = re.compile(r"^(dev|devval)_l([0-9p]+)_s(\d+)$")


def _gate_c_source_mode(source_root: Path, aggregation_root: Path, contract: ri.RQ4bContract) -> str:
    """Production reads only the declared RQ4b namespace; synthetic sources and output stay under scratch."""
    source, target = Path(source_root).resolve(), Path(aggregation_root).resolve()
    if target == contract.aggregation_namespace.resolve():
        _require(source == contract.artifact_namespace.resolve(), "Production Gate C aggregation must read the declared RQ4b artifact namespace.")
        return MODE_PRODUCTION
    synthetic = SYNTHETIC_ROOT.resolve()
    _require(synthetic in target.parents and synthetic in source.parents, "Gate C aggregation root is not authorized.")
    return MODE_SYNTHETIC


def gate_c_source_seeds(contract: ri.RQ4bContract) -> dict[str, int]:
    """Pinned source run -> the one development seed it owns (42, 43, 44 in order)."""
    expected = (contract.track_a_source_run_id, contract.track_b_child_run_id, contract.production_run_id)
    _require(tuple(contract.aggregation_source_run_ids) == expected, "Gate C source run ids differ from the pinned three-run lineage.")
    _require(tuple(contract.aggregation_source_pins) == expected, "Gate C source pins must cover exactly the three source runs in order.")
    _require(len(contract.development_seeds) == len(expected), "Gate C requires one source run per development seed.")
    return dict(zip(expected, contract.development_seeds))


def verify_gate_c_source_pins(source_root: Path, contract: ri.RQ4bContract, *, source_mode: str = MODE_PRODUCTION) -> dict[str, RunLayout]:
    """Exact pinning before any source output is read: run identity, manifest digest,
    the nine-unit roster (no missing, extra or duplicate unit) and every pinned receipt digest."""
    seeds = gate_c_source_seeds(contract)
    layouts: dict[str, RunLayout] = {}
    found: dict[tuple[float, int], list[str]] = {}
    for run_id in seeds:
        pin = contract.aggregation_source_pins[run_id]
        layout = RunLayout(Path(source_root).resolve(), run_id, source_mode)
        manifest_path = layout.run_dir / MANIFEST_FILE
        _require(layout.run_dir.is_dir() and manifest_path.is_file(), f"Gate C source run {run_id} is missing.", RQ4bPackageError)
        _require(sha256_file(manifest_path) == pin["manifest_sha256"], f"Gate C source manifest digest differs for {run_id}.", RQ4bPackageError)
        manifest = read_json(manifest_path)
        _require(manifest.get("run_id") == run_id and manifest.get("mode") == source_mode,
                 f"Gate C source run identity or mode differs for {run_id}.", RQ4bPackageError)
        published = sorted(p.name for p in layout.stages.iterdir() if p.is_dir() and not p.name.startswith(".")) if layout.stages.is_dir() else []
        units: dict[str, set[tuple[float, int]]] = {"dev": set(), "devval": set()}
        for key in published:
            if key in SOURCE_SUPPORT_STAGES:
                continue
            match = _DEVELOPMENT_STAGE.fullmatch(key)
            _require(match is not None, f"Gate C source run {run_id} has an unexpected stage {key}.", RQ4bPackageError)
            units[match.group(1)].add((float(match.group(2).replace("p", ".")), int(match.group(3))))
        _require(units["dev"] == units["devval"], f"Gate C source run {run_id} has unpaired development/development-validation stages.", RQ4bPackageError)
        for unit in units["dev"]:
            found.setdefault(unit, []).append(run_id)
        layouts[run_id] = layout
    duplicates = sorted(unit for unit, runs in found.items() if len(runs) > 1)
    _require(not duplicates, f"Duplicate Gate C development units across source runs: {duplicates}.", RQ4bPackageError)
    owner = {unit: run_id for run_id, seed in seeds.items() for unit in ev.DEVELOPMENT_UNITS if unit[1] == seed}
    missing = sorted(set(owner) - set(found))
    _require(not missing, f"Missing Gate C development units: {missing}.", RQ4bPackageError)
    extra = sorted(unit for unit, runs in found.items() if owner.get(unit) != runs[0])
    _require(not extra, f"Unexpected Gate C development units: {extra}.", RQ4bPackageError)
    for run_id, layout in layouts.items():
        receipts = contract.aggregation_source_pins[run_id]["receipt_sha256"]
        expected = {f"stages/{unit_key(prefix, lam, seed)}/{RECEIPT_FILE}" for (lam, seed), run in owner.items() if run == run_id for prefix in ("dev", "devval")}
        _require(set(receipts) == expected, f"Gate C receipt pins for {run_id} do not cover exactly its development units.", RQ4bPackageError)
        for relative, digest in receipts.items():
            path = layout.run_dir / relative
            _require(path.is_file() and sha256_file(path) == digest, f"Gate C source receipt digest differs: {run_id}/{relative}.", RQ4bPackageError)
    return layouts


def _load_gate_c_units(layouts: Mapping[str, RunLayout], contract: ri.RQ4bContract) -> dict[str, Any]:
    """Verify each pinned unit's outputs, accounting and common ordered Validation population; evaluate saved predictions only."""
    seeds = gate_c_source_seeds(contract)
    run_for_seed = {seed: run_id for run_id, seed in seeds.items()}
    evaluations, source_receipts, stability, reference = [], {}, [], None
    cv_fits = refits = 0
    for lam, seed in ev.DEVELOPMENT_UNITS:
        run_id = run_for_seed[seed]
        source = layouts[run_id]
        pins = contract.aggregation_source_pins[run_id]["receipt_sha256"]
        fit_key, val_key = unit_key("dev", lam, seed), unit_key("devval", lam, seed)
        fit, val = read_stage(source, fit_key), read_stage(source, val_key)  # every declared output re-hashed
        unit = {"adversary_lambda": lam, "model_seed": seed}
        _require(fit["unit"] == unit and val["unit"] == unit, f"Gate C source receipt unit identity differs for {fit_key}.", RQ4bPackageError)
        _require(fit["counts"].get("cv_fold_fits") == CV_FOLDS and fit["counts"].get("refit_fits") == 1, f"Gate C source fit accounting is incomplete for {fit_key}.", RQ4bPackageError)
        fit_sha, val_sha = pins[f"stages/{fit_key}/{RECEIPT_FILE}"], pins[f"stages/{val_key}/{RECEIPT_FILE}"]
        _require(val.get("inputs") == {fit_key: fit_sha}, f"Gate C {val_key} does not trace to the pinned {fit_key} receipt.", RQ4bPackageError)
        predictions = read_predictions(source.stage_dir(val_key) / "validation_predictions.csv")
        ids = predictions.record_ids
        if reference is None:
            reference = ids
        _require(len(ids) == len(reference) and set(ids) == set(reference), f"Gate C Validation population differs for {val_key}.", RQ4bPackageError)
        _require(ids == reference, f"Gate C Validation record ID ordering differs for {val_key}.", RQ4bPackageError)
        digest = sha256_ids(ids)
        _require(digest == contract.aggregation_validation_record_id_sha256 and len(ids) == contract.aggregation_validation_rows,
                 f"Gate C Validation population for {val_key} differs from the pinned ordered population.", RQ4bPackageError)
        recorded = val["counts"].get("predictions", {}).get("validation_predictions.csv", {})
        _require(recorded == {"artifact": fit_key, "partition": "validation", "record_id_sha256": digest, "rows": len(ids)},
                 f"Gate C {val_key} receipt prediction record differs from its saved Validation predictions.", RQ4bPackageError)
        evaluation = ev.evaluate_candidate(predictions, adversary_lambda=lam, model_seed=seed)
        stored = read_json(source.stage_dir(val_key) / "threshold.json")["threshold"]
        _require(evaluation.threshold.threshold == stored, f"Gate C recomputed Validation threshold differs from the stored threshold for {val_key}.", RQ4bPackageError)
        evaluations.append(evaluation)
        source_receipts[f"{run_id}:{fit_key}"], source_receipts[f"{run_id}:{val_key}"] = fit_sha, val_sha
        cv_fits += fit["counts"]["cv_fold_fits"]
        refits += fit["counts"]["refit_fits"]
        stability.append({"adversary_lambda": lam, "model_seed": seed, "status": "training-unstable beyond selected epoch" if lam > 0 else "not_observed_for_lambda_zero"})
    _require((cv_fits, refits) == (CV_FIT_BUDGET, REFIT_BUDGET), f"Gate C source accounting must be exactly {CV_FIT_BUDGET} CV fits and {REFIT_BUDGET} refits.", RQ4bPackageError)
    return {"evaluations": evaluations, "source_receipts": source_receipts, "training_stability": stability,
            "source_fit_accounting": {"cv_fold_fits": cv_fits, "refit_fits": refits}, "validation_record_id_sha256": sha256_ids(reference)}


def _publish_gate_c(layout: RunLayout, manifest: Mapping[str, Any], payload: Mapping[str, Any], inputs: Mapping[str, str], counts: Mapping[str, Any]) -> None:
    """Build the complete run privately and publish it by one directory rename; nothing partial survives a failure."""
    layout.root.mkdir(parents=True, exist_ok=True)
    temp = layout.root / f".tmp_{layout.run_id}_{uuid.uuid4().hex}"
    started = datetime.now(timezone.utc).isoformat()
    try:
        stage = temp / "stages" / GATE_C_STAGE
        stage.mkdir(parents=True)
        (temp / "failures").mkdir()
        write_json_new(temp / MANIFEST_FILE, manifest)
        write_json_new(stage / GATE_C_SELECTION_FILE, payload)
        write_json_new(stage / RECEIPT_FILE, {
            "run_id": layout.run_id, "mode": layout.mode, "stage": GATE_C_STAGE, "stage_key": GATE_C_STAGE, "unit": {},
            "status": "PASS", "inputs": dict(inputs), "outputs": {GATE_C_SELECTION_FILE: sha256_file(stage / GATE_C_SELECTION_FILE)},
            "counts": dict(counts), "prior_failed_attempts": [], "code_sha256": manifest["code_sha256"],
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        _require(not layout.run_dir.exists(), f"Gate C aggregation run {layout.run_id} appeared during aggregation; refusing to overwrite.", RQ4bStageOrderError)
        os.rename(temp, layout.run_dir)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def aggregate_gate_c_sources(source_root: Path, aggregation_root: Path, contract: ri.RQ4bContract) -> RunLayout:
    """Read immutable completed development outputs and publish evaluation-only Gate C evidence.

    This deliberately has no backend parameter: it cannot fit or infer, and it
    never opens Test.  Every source pin, unit and Validation population is
    verified before anything is created; only then is the run published atomically.
    """
    _require(GATE_C_RUN_ID_PATTERN.fullmatch(contract.aggregation_run_id) is not None, "Gate C aggregation run id is invalid.")
    _require(contract.selection_policy_status == ev.SELECTION_POLICY_STATUS == "approved_evaluation_only_three_seed_validation_selection",
             "Gate C requires the locked evaluation-only three-seed Validation policy.", RQ4bStageOrderError)
    source_mode = _gate_c_source_mode(source_root, aggregation_root, contract)
    layout = RunLayout(Path(aggregation_root).resolve(), contract.aggregation_run_id, GATE_C_LAYOUT_MODES[source_mode])
    _require(not layout.run_dir.exists(), f"Gate C aggregation run {layout.run_id} is already published; it is never overwritten or resumed.", RQ4bStageOrderError)
    source_layouts = verify_gate_c_source_pins(source_root, contract, source_mode=source_mode)
    units = _load_gate_c_units(source_layouts, contract)
    selection = ev.select_lambda(units["evaluations"])
    payload = {**selection, "validation_record_id_sha256": units["validation_record_id_sha256"], "validation_rows": contract.aggregation_validation_rows,
               "source_receipts": units["source_receipts"], "source_fit_accounting": units["source_fit_accounting"],
               "training_stability": units["training_stability"], "test_status": GATE_C_TEST_STATUS}
    manifest = {
        "run_id": layout.run_id, "mode": layout.mode, "source_mode": source_mode, "configuration": "RQ4b-3",
        "protocol_version": contract.aggregation_protocol_version, "decision_revision": contract.decision_revision,
        "source_run_ids": list(contract.aggregation_source_run_ids), "source_pins": contract.aggregation_source_pins,
        "validation_rows": contract.aggregation_validation_rows, "validation_record_id_sha256": contract.aggregation_validation_record_id_sha256,
        "evaluation_only": True, "prohibit_fit": True, "prohibit_inference": True, "test_status": GATE_C_TEST_STATUS,
        "config_sha256": sha256_file(ri.RQ4B_CONFIG_PATH), "code_sha256": code_lineage(), "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    counts = {"cv_fold_fits": 0, "refit_fits": 0, "probe_fits": 0, "predictions": {}, "source_fit_accounting": units["source_fit_accounting"]}
    _publish_gate_c(layout, manifest, payload, units["source_receipts"], counts)
    read_stage(layout, GATE_C_STAGE)
    return layout


# ---------------------------------------------------------------------------
# Cross-run Test evaluation (nine pinned artifacts; no fitting)
# ---------------------------------------------------------------------------


TEST_STAGE = "test_evaluation"
PACKAGE_STAGE = "package_validation"
TEST_LAYOUT_MODES = {
    MODE_PRODUCTION: "cross_run_test_evaluation",
    MODE_SYNTHETIC: "cross_run_test_evaluation_synthetic_test",
}
TEST_OUTPUTS = (
    "arm_metrics.csv", "group_metrics.csv", "matched_deltas.csv", "decision_changes.csv",
    "bootstrap_replicates.csv", "bootstrap_summary.csv", "across_seed_summary.csv",
    "generalization_gaps.csv", "execution_status.json",
)


@dataclass(frozen=True)
class TestArmSource:
    adversary_lambda: float
    seed: int
    role: str
    provenance: str
    run_id: str
    fit_key: str
    validation_key: str
    fit_receipt_sha256: str
    validation_receipt_sha256: str
    estimator_sha256: str
    preprocessor_sha256: str
    threshold: float
    fit_dir: Path

    @property
    def artifact_id(self) -> str:
        return f"{self.run_id}:{self.fit_key}"

    @property
    def prediction_name(self) -> str:
        return f"test_predictions_l{lam_key(self.adversary_lambda)}_s{self.seed}.csv"


def _test_mode(source_root: Path, gate_c_root: Path, test_root: Path, contract: ri.RQ4bContract) -> str:
    source, gate, target = map(lambda value: Path(value).resolve(), (source_root, gate_c_root, test_root))
    if target == contract.test_namespace.resolve():
        _require(source == contract.artifact_namespace.resolve(), "Production Test must read the declared RQ4b source namespace.")
        _require(gate == contract.aggregation_namespace.resolve(), "Production Test must read the declared Gate C namespace.")
        return MODE_PRODUCTION
    synthetic = SYNTHETIC_ROOT.resolve()
    _require(all(synthetic in value.parents for value in (source, gate, target)), "Cross-run Test roots are not authorized.")
    return MODE_SYNTHETIC


def _verify_gate_c_for_test(gate_c_root: Path, contract: ri.RQ4bContract, source_mode: str) -> dict[str, str]:
    mode = GATE_C_LAYOUT_MODES[source_mode]
    layout = RunLayout(Path(gate_c_root).resolve(), contract.aggregation_run_id, mode)
    manifest_path = layout.run_dir / MANIFEST_FILE
    receipt_path = layout.stage_dir(GATE_C_STAGE) / RECEIPT_FILE
    selection_path = layout.stage_dir(GATE_C_STAGE) / GATE_C_SELECTION_FILE
    _require(manifest_path.is_file(), "Pinned Gate C manifest is missing.", RQ4bPackageError)
    _require(sha256_file(manifest_path) == contract.test_gate_c_manifest_sha256, "Pinned Gate C manifest digest differs.", RQ4bPackageError)
    _require(receipt_path.is_file() and sha256_file(receipt_path) == contract.test_gate_c_receipt_sha256,
             "Pinned Gate C receipt digest differs.", RQ4bPackageError)
    receipt = read_stage(layout, GATE_C_STAGE)
    _require(selection_path.is_file() and sha256_file(selection_path) == contract.test_gate_c_selection_sha256,
             "Pinned Gate C selection output digest differs.", RQ4bPackageError)
    _require(receipt["outputs"].get(GATE_C_SELECTION_FILE) == contract.test_gate_c_selection_sha256,
             "Gate C receipt does not pin the locked selection output.", RQ4bPackageError)
    selection = read_json(selection_path)
    if source_mode == MODE_PRODUCTION:
        _require(selection.get("selected_lambda") == 0.0, "Gate C selected lambda differs from the stored result.", RQ4bPackageError)
    return {
        "manifest": contract.test_gate_c_manifest_sha256,
        "receipt": contract.test_gate_c_receipt_sha256,
        "selection": contract.test_gate_c_selection_sha256,
    }


def _test_arm_roles(contract: ri.RQ4bContract) -> dict[tuple[float, int], tuple[str, str]]:
    roles: dict[tuple[float, int], tuple[str, str]] = {}
    for arm in contract.test_evaluation_arms:
        for seed in arm["seeds"]:
            key = (float(arm["adversary_lambda"]), int(seed))
            _require(key not in roles, f"Duplicate Test arm {key}.", RQ4bPackageError)
            roles[key] = (str(arm["role"]), str(arm["provenance"]))
    _require(tuple(roles) == contract.test_evaluation_units, "Test arm roster differs from the locked contract.", RQ4bPackageError)
    return roles


def _resolve_test_arms(source_root: Path, contract: ri.RQ4bContract, source_mode: str) -> list[TestArmSource]:
    layouts = verify_gate_c_source_pins(source_root, contract, source_mode=source_mode)
    _require(contract.test_source_pins == {
        run_id: {"manifest_sha256": pin["manifest_sha256"], "evaluated_arm_receipt_sha256": pin["receipt_sha256"]}
        for run_id, pin in contract.aggregation_source_pins.items()
    }, "Test source pins differ from the Gate C source pins.", RQ4bPackageError)
    run_for_seed = {seed: run_id for run_id, seed in gate_c_source_seeds(contract).items()}
    roles = _test_arm_roles(contract)
    arms: list[TestArmSource] = []
    for lam, seed in contract.test_evaluation_units:
        run_id, fit_key, val_key = run_for_seed[seed], unit_key("dev", lam, seed), unit_key("devval", lam, seed)
        layout = layouts[run_id]
        fit, validation = read_stage(layout, fit_key), read_stage(layout, val_key)
        pins = contract.test_source_pins[run_id]["evaluated_arm_receipt_sha256"]
        fit_relative, val_relative = f"stages/{fit_key}/{RECEIPT_FILE}", f"stages/{val_key}/{RECEIPT_FILE}"
        _require(sha256_file(layout.stage_dir(fit_key) / RECEIPT_FILE) == pins[fit_relative],
                 f"Pinned model receipt differs for {fit_key}.", RQ4bPackageError)
        _require(sha256_file(layout.stage_dir(val_key) / RECEIPT_FILE) == pins[val_relative],
                 f"Pinned Validation receipt differs for {val_key}.", RQ4bPackageError)
        _require(validation.get("inputs") == {fit_key: pins[fit_relative]},
                 f"Validation receipt {val_key} does not trace to its pinned model receipt.", RQ4bPackageError)
        for model_name in ("estimator.joblib", "preprocessor.joblib"):
            _require(model_name in fit["outputs"], f"Pinned model output {model_name} is missing for {fit_key}.", RQ4bPackageError)
        threshold_path = layout.stage_dir(val_key) / "threshold.json"
        _require("threshold.json" in validation["outputs"], f"Locked threshold is missing for {val_key}.", RQ4bPackageError)
        threshold = read_json(threshold_path).get("threshold")
        _require(isinstance(threshold, (int, float)) and 0.0 <= float(threshold) <= 1.0,
                 f"Locked threshold is invalid for {val_key}.", RQ4bPackageError)
        role, provenance = roles[(lam, seed)]
        arms.append(TestArmSource(lam, seed, role, provenance, run_id, fit_key, val_key,
                                  pins[fit_relative], pins[val_relative], fit["outputs"]["estimator.joblib"],
                                  fit["outputs"]["preprocessor.joblib"], float(threshold), layout.stage_dir(fit_key)))
    _require(len(arms) == 9, "Cross-run Test requires exactly nine arms.", RQ4bPackageError)
    return arms


def _load_test_indices(contract: ri.RQ4bContract, predictions: ev.PartitionPredictions) -> np.ndarray:
    path = contract.test_bootstrap_path
    _require(path.is_file() and sha256_file(path) == contract.test_bootstrap_sha256,
             "Locked Test bootstrap digest differs.", RQ4bPackageError)
    manifest_path = path.parent / "rq1_test_bootstrap_manifest.json"
    _require(manifest_path.is_file(), "Locked Test bootstrap manifest is missing.", RQ4bPackageError)
    manifest = read_json(manifest_path)
    _require(manifest.get("indices_sha256") == contract.test_bootstrap_sha256,
             "Bootstrap manifest digest differs from the Test contract.", RQ4bPackageError)
    _require(manifest.get("test_record_id_sha256") == sha256_ids(predictions.record_ids)
             and manifest.get("test_rows") == len(predictions.record_ids),
             "Ordered Test IDs differ from the locked bootstrap manifest.", RQ4bPackageError)
    with np.load(path, allow_pickle=False) as payload:
        _require(payload.files == [contract.test_bootstrap_array], "Locked bootstrap artifact has an unexpected array roster.", RQ4bPackageError)
        indices = payload[contract.test_bootstrap_array]
    _require(indices.shape[0] == contract.test_bootstrap_replicates, "Locked bootstrap replicate count differs.", RQ4bPackageError)
    return ev.validate_locked_indices(indices, predictions.y)


def _metric_value(estimate: Any) -> float | None:
    return float(estimate.value) if estimate.valid else None


def _group_rows(predictions: ev.PartitionPredictions, threshold: float, seed: int, lam: float) -> list[dict[str, Any]]:
    decisions = ev.decisions_at(predictions.p_approved, threshold)
    race = np.asarray(predictions.race).astype(str)
    rows: list[dict[str, Any]] = []
    for roster_name, roster in (("g3", ev.G3), ("g6", ev.G6)):
        metrics = ev.rq4a_metrics.compute_roster_group_metrics(predictions.y, predictions.p_approved, decisions, race, roster, rules=ev.RULES)
        for group in roster:
            mask = race == group
            tpr, fpr = metrics[group]["tpr_approved"], metrics[group]["fpr_approved"]
            reasons = sorted({item.status for item in (tpr, fpr) if not item.is_defined})
            rows.append({
                "seed": seed, "adversary_lambda": lam, "roster": roster_name, "group": group,
                "row_count": int(mask.sum()), "actual_approved_count": int(np.sum(predictions.y[mask] == 1)),
                "actual_denied_count": int(np.sum(predictions.y[mask] == 0)),
                "predicted_approved_count": int(np.sum(decisions[mask] == 1)), "predicted_denied_count": int(np.sum(decisions[mask] == 0)),
                "tpr_approved": tpr.value, "fpr_approved": fpr.value,
                "tpr_valid": bool(tpr.is_defined), "fpr_valid": bool(fpr.is_defined),
                "invalid_reason": None if not reasons else "|".join(reasons),
            })
    return rows


def _decision_rows(control: ev.PartitionPredictions, comparator: ev.PartitionPredictions, c_threshold: float,
                   a_threshold: float, seed: int, comparator_lambda: float) -> list[dict[str, Any]]:
    c_decisions, a_decisions = ev.decisions_at(control.p_approved, c_threshold), ev.decisions_at(comparator.p_approved, a_threshold)
    race = np.asarray(control.race).astype(str)
    rows: list[dict[str, Any]] = []
    scopes = [("overall", "__all__", np.ones(len(race), dtype=bool))]
    scopes += [("g3", group, race == group) for group in ev.G3]
    scopes += [("g6", group, race == group) for group in ev.G6]
    for roster, group, mask in scopes:
        changed = c_decisions[mask] != a_decisions[mask]
        n = int(mask.sum())
        rows.append({
            "seed": seed, "comparator_lambda": comparator_lambda, "control_lambda": 0.0,
            "roster": roster, "group": group, "row_count": n,
            "unchanged_count": int(n - changed.sum()), "changed_count": int(changed.sum()),
            "denied_to_approved_count": int(np.sum((c_decisions[mask] == 0) & (a_decisions[mask] == 1))),
            "approved_to_denied_count": int(np.sum((c_decisions[mask] == 1) & (a_decisions[mask] == 0))),
            "change_rate": float(changed.mean()) if n else None,
        })
    return rows


def _summary_row(scope: str, lam: float | None, comparator: float | None, quantity: str, values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "scope": scope, "adversary_lambda": lam, "comparator_lambda": comparator,
        "control_lambda": 0.0 if comparator is not None else None, "quantity": quantity,
        "seed_count": len(array), "mean": float(array.mean()), "standard_deviation": float(array.std(ddof=1)),
        "ddof": 1, "min": float(array.min()), "max": float(array.max()), "range": float(array.max() - array.min()),
    }


def _compute_test_tables(arms: Sequence[TestArmSource], predictions: Mapping[str, ev.PartitionPredictions],
                         validation_predictions: Mapping[str, ev.PartitionPredictions], indices: np.ndarray) -> dict[str, Any]:
    by_unit = {(arm.adversary_lambda, arm.seed): arm for arm in arms}
    point: dict[tuple[float, int], dict[str, Any]] = {}
    arm_rows, group_rows, gap_rows = [], [], []
    for arm in arms:
        pred, validation = predictions[arm.artifact_id], validation_predictions[arm.artifact_id]
        metrics, validation_core = ev.arm_point_metrics(pred, arm.threshold), ev.core_quantities(validation, arm.threshold)
        point[(arm.adversary_lambda, arm.seed)] = metrics
        arm_rows.append({
            "seed": arm.seed, "adversary_lambda": arm.adversary_lambda, "role": arm.role,
            "request_provenance": arm.provenance, "source_run_id": arm.run_id,
            "model_receipt_sha256": arm.fit_receipt_sha256, "validation_receipt_sha256": arm.validation_receipt_sha256,
            "threshold": arm.threshold, "test_rows": len(pred.record_ids), "test_record_id_sha256": sha256_ids(pred.record_ids),
            "balanced_accuracy": metrics["ba"], "eod3": metrics["eod3"]["value"], "eod6_audit": metrics["eod6"]["value"],
        })
        group_rows.extend(_group_rows(pred, arm.threshold, arm.seed, arm.adversary_lambda))
        test_core = ev.core_quantities(pred, arm.threshold)
        for quantity, key in (("balanced_accuracy", "ba"), ("eod3", "eod3"), ("eod6", "eod6")):
            test_value = test_core[key] if key == "ba" else _metric_value(test_core[key])
            validation_value = validation_core[key] if key == "ba" else _metric_value(validation_core[key])
            gap_rows.append({"seed": arm.seed, "adversary_lambda": arm.adversary_lambda, "quantity": quantity,
                             "validation_value": validation_value, "test_value": test_value,
                             "test_minus_validation": None if test_value is None or validation_value is None else float(test_value - validation_value)})
    delta_rows, decision_rows, replicate_rows, bootstrap_rows = [], [], [], []
    for seed, comparator_lambda, control_lambda in ri.TEST_MATCHED_COMPARISONS:
        control_arm, comparator_arm = by_unit[(control_lambda, seed)], by_unit[(comparator_lambda, seed)]
        control, comparator = predictions[control_arm.artifact_id], predictions[comparator_arm.artifact_id]
        deltas = ev.matched_deltas(ev.core_quantities(control, control_arm.threshold), ev.core_quantities(comparator, comparator_arm.threshold))
        delta_rows.append({"seed": seed, "comparator_lambda": comparator_lambda, "control_lambda": control_lambda,
                           "delta_balanced_accuracy": deltas["delta_ba"], "delta_eod3": deltas["delta_eod3"], "delta_eod6": deltas["delta_eod6"]})
        decision_rows.extend(_decision_rows(control, comparator, control_arm.threshold, comparator_arm.threshold, seed, comparator_lambda))
        bootstrap = ev.paired_bootstrap(control, comparator, control_threshold=control_arm.threshold,
                                        adversarial_threshold=comparator_arm.threshold, indices=indices)
        for replicate_id in range(len(indices)):
            values = {name: bootstrap[name]["values"][replicate_id] for name in ev.DELTA_QUANTITIES}
            reasons = [f"{name}:{next((item['reason'] for item in bootstrap[name]['invalid_replicates'] if item['replicate'] == replicate_id), '')}"
                       for name, value in values.items() if value is None]
            replicate_rows.append({"seed": seed, "comparator_lambda": comparator_lambda, "control_lambda": control_lambda,
                                   "replicate_id": replicate_id, "sample_size": len(control.y),
                                   "delta_balanced_accuracy": values["delta_ba"], "delta_eod3": values["delta_eod3"], "delta_eod6": values["delta_eod6"],
                                   "balanced_accuracy_valid": values["delta_ba"] is not None, "eod3_valid": values["delta_eod3"] is not None,
                                   "eod6_valid": values["delta_eod6"] is not None, "invalid_reasons": None if not reasons else "|".join(reasons)})
        for name, output_name in (("delta_ba", "delta_balanced_accuracy"), ("delta_eod3", "delta_eod3"), ("delta_eod6", "delta_eod6")):
            summary = bootstrap[name]
            bootstrap_rows.append({"seed": seed, "comparator_lambda": comparator_lambda, "control_lambda": control_lambda,
                                   "quantity": output_name, "valid_replicates": summary["valid_replicates"],
                                   "invalid_replicates": len(summary["invalid_replicates"]), "confidence_level": 0.95,
                                   "lower": summary["lower"], "upper": summary["upper"], "quantile_method": summary["quantile_method"]})
    across_rows = []
    for lam in (0.0, 0.01):
        for quantity, field in (("balanced_accuracy", "ba"), ("eod3", "eod3"), ("eod6_audit", "eod6")):
            values = [point[(lam, seed)][field] if field == "ba" else point[(lam, seed)][field]["value"] for seed in (42, 43, 44)]
            if all(value is not None for value in values):
                across_rows.append(_summary_row("per_arm", lam, None, quantity, values))
    for quantity in ("delta_balanced_accuracy", "delta_eod3", "delta_eod6"):
        values = [row[quantity] for row in delta_rows if row["comparator_lambda"] == 0.01]
        if len(values) == 3 and all(value is not None for value in values):
            across_rows.append(_summary_row("matched_delta", None, 0.01, quantity, values))
    return {
        "arm_metrics.csv": arm_rows, "group_metrics.csv": group_rows, "matched_deltas.csv": delta_rows,
        "decision_changes.csv": decision_rows, "bootstrap_replicates.csv": replicate_rows,
        "bootstrap_summary.csv": bootstrap_rows, "across_seed_summary.csv": across_rows,
        "generalization_gaps.csv": gap_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")


def _csv_content(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return pd.DataFrame(list(rows), columns=list(columns)).to_csv(index=False, float_format="%.17g")


def _arm_manifest(arm: TestArmSource) -> dict[str, Any]:
    """Portable arm lineage; local source paths are deliberately excluded."""
    return {name: jsonable(getattr(arm, name)) for name in arm.__dataclass_fields__ if name != "fit_dir"}


def _publish_test_run(layout: RunLayout, manifest: Mapping[str, Any], inputs: Mapping[str, str], arms: Sequence[TestArmSource],
                      predictions: Mapping[str, ev.PartitionPredictions], tables: Mapping[str, Any]) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    temp = layout.root / f".tmp_{layout.run_id}_{uuid.uuid4().hex}"
    started = datetime.now(timezone.utc).isoformat()
    try:
        stage = temp / "stages" / TEST_STAGE
        stage.mkdir(parents=True)
        (temp / "failures").mkdir()
        write_json_new(temp / MANIFEST_FILE, manifest)
        for arm in arms:
            frame = pd.DataFrame({"record_id": list(predictions[arm.artifact_id].record_ids),
                                  "y_approved": predictions[arm.artifact_id].y,
                                  "p_approved": predictions[arm.artifact_id].p_approved,
                                  "derived_race": predictions[arm.artifact_id].race,
                                  "applicant_sex_group": predictions[arm.artifact_id].sex})
            with (stage / arm.prediction_name).open("x", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False, float_format="%.17g")
        for name, rows in tables.items():
            columns = manifest["output_schema"][f"stages/{TEST_STAGE}/{name}"]
            _write_csv(stage / name, rows, columns)
        status = {"execution_status": "completed", "reason_codes": ["none"],
                  "test_rows": len(next(iter(predictions.values())).record_ids), "inference_count": len(predictions),
                  "fit_count": 0, "refit_count": 0, "cv_fit_count": 0}
        write_json_new(stage / "execution_status.json", status)
        outputs = {path.name: sha256_file(path) for path in sorted(stage.iterdir())}
        prediction_counts = {arm.prediction_name: {"artifact": arm.artifact_id, "partition": "test",
                             "rows": len(predictions[arm.artifact_id].record_ids),
                             "record_id_sha256": sha256_ids(predictions[arm.artifact_id].record_ids)} for arm in arms}
        write_json_new(stage / RECEIPT_FILE, {"run_id": layout.run_id, "mode": layout.mode, "stage": TEST_STAGE,
                       "stage_key": TEST_STAGE, "unit": {}, "status": "PASS", "inputs": dict(inputs), "outputs": outputs,
                       "counts": {"cv_fold_fits": 0, "refit_fits": 0, "probe_fits": 0, "predictions": prediction_counts,
                                  "source_fit_accounting": {"cv_fold_fits": 45, "refit_fits": 9}},
                       "prior_failed_attempts": [], "code_sha256": manifest["code_sha256"],
                       "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat()})
        _require(not layout.run_dir.exists(), f"Test run {layout.run_id} appeared during evaluation; refusing to overwrite.", RQ4bStageOrderError)
        os.rename(temp, layout.run_dir)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def evaluate_cross_run_test(source_root: Path, gate_c_root: Path, test_root: Path, backend: Any,
                            contract: ri.RQ4bContract) -> RunLayout:
    """Verify all lineage, infer each unique pinned artifact once, and atomically publish a separate Test run."""
    _require(TEST_RUN_ID_PATTERN.fullmatch(contract.test_run_id) is not None, "Test run id is invalid.")
    source_mode = _test_mode(source_root, gate_c_root, test_root, contract)
    layout = RunLayout(Path(test_root).resolve(), contract.test_run_id, TEST_LAYOUT_MODES[source_mode])
    _require(not layout.run_dir.exists(), f"Test run {layout.run_id} is already published; it is never overwritten or resumed.", RQ4bStageOrderError)
    gate_pins = _verify_gate_c_for_test(gate_c_root, contract, source_mode)
    arms = _resolve_test_arms(source_root, contract, source_mode)
    _require(contract.test_bootstrap_path.is_file() and sha256_file(contract.test_bootstrap_path) == contract.test_bootstrap_sha256,
             "Locked Test bootstrap digest differs.", RQ4bPackageError)
    predictions: dict[str, ev.PartitionPredictions] = {}
    for arm in arms:
        if arm.artifact_id not in predictions:
            predictions[arm.artifact_id] = backend.predict(arm.fit_dir, "test")
    _require(len(predictions) <= 9, "Test inference count exceeds the locked maximum.", RQ4bPackageError)
    reference = next(iter(predictions.values()))
    _require(all(item.record_ids == reference.record_ids for item in predictions.values()),
             "Test predictions cover different ordered Test IDs.", RQ4bPackageError)
    indices = _load_test_indices(contract, reference)
    validation = {arm.artifact_id: read_predictions(arm.fit_dir.parent / arm.validation_key / "validation_predictions.csv") for arm in arms}
    tables = _compute_test_tables(arms, predictions, validation, indices)
    output_schema = {key: list(value) for key, value in contract.test_required_output_files.items()}
    manifest = {"run_id": layout.run_id, "protocol_version": contract.test_protocol_version, "mode": layout.mode,
                "configuration": "RQ4b-3", "config_sha256": sha256_file(ri.RQ4B_CONFIG_PATH), "code_sha256": code_lineage(),
                "gate_c_pins": gate_pins, "source_pins": contract.test_source_pins,
                "bootstrap_index_pin": {"path": str(contract.test_bootstrap_path), "sha256": contract.test_bootstrap_sha256,
                                        "array": contract.test_bootstrap_array, "replicates": contract.test_bootstrap_replicates},
                "evaluation_arms": [_arm_manifest(arm) for arm in arms], "output_schema": output_schema,
                "prohibit_fit": True, "prohibit_refit": True, "prohibit_cv": True,
                "research_interpretation_owner": "user", "created_utc": datetime.now(timezone.utc).isoformat()}
    inputs = {"gate_c_manifest": gate_pins["manifest"], "gate_c_receipt": gate_pins["receipt"],
              "gate_c_selection": gate_pins["selection"], "bootstrap_indices": contract.test_bootstrap_sha256}
    inputs.update({f"{arm.run_id}:{arm.fit_key}": arm.fit_receipt_sha256 for arm in arms})
    inputs.update({f"{arm.run_id}:{arm.validation_key}": arm.validation_receipt_sha256 for arm in arms})
    _publish_test_run(layout, manifest, inputs, arms, predictions, tables)
    read_stage(layout, TEST_STAGE)
    return layout


def validate_cross_run_test_package(source_root: Path, gate_c_root: Path, test_root: Path,
                                    contract: ri.RQ4bContract) -> dict[str, Any]:
    """Recompute every Test table from saved predictions; never constructs a backend or performs inference."""
    source_mode = _test_mode(source_root, gate_c_root, test_root, contract)
    layout = RunLayout(Path(test_root).resolve(), contract.test_run_id, TEST_LAYOUT_MODES[source_mode])
    manifest = read_json(layout.run_dir / MANIFEST_FILE)
    _require(manifest.get("run_id") == contract.test_run_id and manifest.get("mode") == layout.mode,
             "Test manifest identity or mode differs.", RQ4bPackageError)
    _require(manifest.get("code_sha256") == code_lineage(), "Test run code lineage differs.", RQ4bPackageError)
    _require(manifest.get("protocol_version") == contract.test_protocol_version
             and manifest.get("config_sha256") == sha256_file(ri.RQ4B_CONFIG_PATH),
             "Test manifest protocol or config lineage differs.", RQ4bPackageError)
    expected_schema = {key: list(value) for key, value in contract.test_required_output_files.items()}
    _require(manifest.get("output_schema") == expected_schema
             and manifest.get("prohibit_fit") is True and manifest.get("prohibit_refit") is True
             and manifest.get("prohibit_cv") is True and manifest.get("research_interpretation_owner") == "user",
             "Test manifest output or interpretation contract differs.", RQ4bPackageError)
    gate_pins = _verify_gate_c_for_test(gate_c_root, contract, source_mode)
    _require(manifest.get("gate_c_pins") == gate_pins and manifest.get("source_pins") == jsonable(contract.test_source_pins),
             "Test manifest lineage pins differ.", RQ4bPackageError)
    arms = _resolve_test_arms(source_root, contract, source_mode)
    _require(manifest.get("evaluation_arms") == [_arm_manifest(arm) for arm in arms],
             "Test manifest arm, model-hash or threshold lineage differs.", RQ4bPackageError)
    expected_bootstrap = {"path": str(contract.test_bootstrap_path), "sha256": contract.test_bootstrap_sha256,
                          "array": contract.test_bootstrap_array, "replicates": contract.test_bootstrap_replicates}
    recorded_bootstrap = manifest.get("bootstrap_index_pin", {})
    comparable_fields = ("sha256", "array", "replicates")
    _require(all(recorded_bootstrap.get(name) == expected_bootstrap[name] for name in comparable_fields)
             and (source_mode != MODE_PRODUCTION or recorded_bootstrap.get("path") == expected_bootstrap["path"]),
             "Test manifest bootstrap pin differs.", RQ4bPackageError)
    receipt = read_stage(layout, TEST_STAGE)
    expected_outputs = {*TEST_OUTPUTS, *(arm.prediction_name for arm in arms)}
    _require(set(receipt.get("outputs", {})) == expected_outputs,
             "Test receipt output roster differs from the locked package.", RQ4bPackageError)
    _require(receipt.get("code_sha256") == manifest["code_sha256"],
             "Test receipt code lineage differs from the manifest.", RQ4bPackageError)
    expected_inputs = {"gate_c_manifest": gate_pins["manifest"], "gate_c_receipt": gate_pins["receipt"],
                       "gate_c_selection": gate_pins["selection"], "bootstrap_indices": contract.test_bootstrap_sha256}
    expected_inputs.update({f"{arm.run_id}:{arm.fit_key}": arm.fit_receipt_sha256 for arm in arms})
    expected_inputs.update({f"{arm.run_id}:{arm.validation_key}": arm.validation_receipt_sha256 for arm in arms})
    _require(receipt.get("inputs") == expected_inputs, "Test receipt lineage inputs differ.", RQ4bPackageError)
    _require(receipt["counts"]["cv_fold_fits"] == receipt["counts"]["refit_fits"] == 0,
             "Test run contains local fit accounting.", RQ4bPackageError)
    _require(receipt["counts"].get("source_fit_accounting") == {"cv_fold_fits": 45, "refit_fits": 9},
             "Test source accounting differs from 45 CV fits and 9 refits.", RQ4bPackageError)
    stage = layout.stage_dir(TEST_STAGE)
    saved = {arm.artifact_id: read_predictions(stage / arm.prediction_name) for arm in arms}
    reference = next(iter(saved.values()))
    _require(all(item.record_ids == reference.record_ids for item in saved.values()), "Saved Test predictions have different ordered IDs.", RQ4bPackageError)
    for arm in arms:
        recorded = receipt["counts"]["predictions"].get(arm.prediction_name)
        _require(recorded == {"artifact": arm.artifact_id, "partition": "test", "rows": len(saved[arm.artifact_id].record_ids),
                              "record_id_sha256": sha256_ids(saved[arm.artifact_id].record_ids)},
                 f"Test receipt prediction record differs for {arm.prediction_name}.", RQ4bPackageError)
    indices = _load_test_indices(contract, reference)
    validation = {arm.artifact_id: read_predictions(arm.fit_dir.parent / arm.validation_key / "validation_predictions.csv") for arm in arms}
    expected = _compute_test_tables(arms, saved, validation, indices)
    for name, rows in expected.items():
        columns = list(contract.test_required_output_files[f"stages/{TEST_STAGE}/{name}"])
        _require((stage / name).read_bytes() == _csv_content(rows, columns).encode("utf-8"),
                 f"Saved {name} does not match recomputation.", RQ4bPackageError)
    expected_status = {"execution_status": "completed", "reason_codes": ["none"], "test_rows": len(reference.record_ids),
                       "inference_count": len(saved), "fit_count": 0, "refit_count": 0, "cv_fit_count": 0}
    _require(read_json(stage / "execution_status.json") == expected_status,
             "Saved execution_status.json does not match recomputation.", RQ4bPackageError)
    inputs = {TEST_STAGE: sha256_file(stage / RECEIPT_FILE)}
    with StageWriter(layout, PACKAGE_STAGE, stage=PACKAGE_STAGE, unit={}, inputs=inputs) as writer:
        writer.counts["source_fit_accounting"] = {"cv_fold_fits": 45, "refit_fits": 9}
        writer.write_json("package_report.json", {"execution_status": "completed", "local_fit_counts": {"cv_fold_fits": 0, "refit_fits": 0},
                          "source_fit_accounting": {"cv_fold_fits": 45, "refit_fits": 9}, "inference_performed": False,
                          "recomputed_from_saved_predictions": True, "verified_outputs": list(TEST_OUTPUTS)})
    return read_stage(layout, PACKAGE_STAGE)


def generalization_gap(test: ev.PartitionPredictions, validation: ev.PartitionPredictions, threshold: float) -> dict[str, Any]:
    """fairness_generalization_gap = Test EOD - complete-Validation EOD at the same locked threshold.

    Undefined EODs stay None with their reason; the Validation EOD is the
    development estimate from the same partition that calibrated the threshold.
    """
    test_core, validation_core = ev.core_quantities(test, threshold), ev.core_quantities(validation, threshold)
    out: dict[str, Any] = {"threshold": float(threshold), "development_partition": "validation",
                           "same_partition_selection_calibration_limitation": True}
    for name in ("eod3", "eod6"):
        t, v = test_core[name], validation_core[name]
        value = ev.fairness_generalization_gap(t.value if t.valid else None, v.value if v.valid else None)
        reasons = {part: {"reason": est.reason_code, "invalid_groups": list(est.invalid_groups)}
                   for part, est in (("test", t), ("validation", v)) if not est.valid}
        out[name] = {"value": value, "test_eod": t.value if t.valid else None, "validation_eod": v.value if v.valid else None,
                     "undefined_reasons": reasons}
    return out


def status_from_assessment(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Completed-run status fields derived only from the verified assessment (independent fields)."""
    return {
        "execution_status": "completed",
        "stability_status": assessment["stability_status"],
        "primary_criterion_per_seed": assessment["per_seed_point_criterion"],
        "strong_support": assessment["strong_support"]["status"],
        "fields_are_independent": True,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RQ4bRunner:
    """Explicit single-unit stages; each verifies its published prerequisites first."""

    def __init__(self, layout: RunLayout, backend: Any, contract: ri.RQ4bContract) -> None:
        if layout.mode == MODE_PRODUCTION:
            _require(isinstance(backend, ProductionBackend), "Production runs require the ProductionBackend.")
        else:
            _require(not isinstance(backend, ProductionBackend), "Synthetic runs may not use the ProductionBackend.")
        self.layout, self.backend, self.contract = layout, backend, contract

    # -- helpers --------------------------------------------------------------

    def _inputs(self, keys: Iterable[str]) -> dict[str, str]:
        return {key: receipt_sha(self.layout, key) for key in keys}

    def _published_keys(self) -> list[str]:
        return sorted(p.name for p in self.layout.stages.iterdir() if p.is_dir() and not p.name.startswith("."))

    def _prediction_ledger(self) -> set[tuple[str, str]]:
        ledger = set()
        for key in self._published_keys():
            for item in read_stage(self.layout, key)["counts"].get("predictions", {}).values():
                ledger.add((item["artifact"], item["partition"]))
        return ledger

    def _predict_once(self, writer: StageWriter, artifact_key: str, partition: str, name: str) -> ev.PartitionPredictions:
        """Inference guard: each (artifact, partition) is predicted at most once per run."""
        _require((artifact_key, partition) not in self._prediction_ledger(), f"{artifact_key} was already predicted on {partition}.", RQ4bStageOrderError)
        _require(all(v["artifact"] != artifact_key or v["partition"] != partition for v in writer.counts["predictions"].values()),
                 f"{artifact_key} would be predicted twice on {partition}.", RQ4bStageOrderError)
        predictions = self.backend.predict(self.layout.stage_dir(artifact_key), partition)
        _require(isinstance(predictions, ev.PartitionPredictions), "Backend predictions must be PartitionPredictions.")
        writer.counts["predictions"][name] = {"artifact": artifact_key, "partition": partition, "rows": len(predictions.record_ids),
                                              "record_id_sha256": sha256_ids(predictions.record_ids)}
        writer.write_predictions(name, predictions)
        return predictions

    # -- stages ---------------------------------------------------------------

    def preflight(self) -> dict[str, Any]:
        with StageWriter(self.layout, "preflight", stage="preflight", unit={}, inputs={}) as writer:
            statuses = self.backend.preflight()
            if self.layout.mode == MODE_PRODUCTION:
                statuses["gate_a_technical_readiness"] = (
                    "PASS" if ri.APPROVED_CONTRACT["execution_gates"]["gate_a_technical_readiness"]["status"] == "PASS" else "FAIL"
                )
                statuses["track_b_child_execution_lineage"] = (
                    "PASS" if self.layout.run_id == self.contract.production_run_id else "FAIL"
                )
            failing = {k: v for k, v in statuses.items() if v != "PASS"}
            _require(not failing, f"Preflight prerequisites are not PASS: {failing}")
            writer.write_json("preflight.json", {"statuses": statuses, "backend": self.backend.kind})
        return read_stage(self.layout, "preflight")

    def capability(self) -> dict[str, Any]:
        inputs = self._inputs(["preflight"])
        with StageWriter(self.layout, "capability", stage="capability", unit={}, inputs=inputs) as writer:
            evidence = self.backend.capability_evidence()
            _require(evidence.get("synthetic_gate_status") == "PASS", "Synthetic capability correctness gate is not an accepted PASS.")
            real = evidence.get("real_probe", {"status": "not_run_requires_separate_authorization"})
            writer.write_json("capability.json", {
                "synthetic_gate": evidence, "real_probe": real,
                "capability_status": real.get("capability_status", "capability_inconclusive"),
                "note": "capability is reported separately from execution and scientific support",
            })
        return read_stage(self.layout, "capability")

    def development(self, adversary_lambda: float, seed: int, *, allow_new_attempt_after_failure: bool = False) -> dict[str, Any]:
        lam = float(adversary_lambda)
        _require((lam, int(seed)) in ev.ADDITIONAL_DEVELOPMENT_UNITS,
                 "Track B may fit only the four approved additional development units; retained seed-42 units must not be refit.",
                 RQ4bStageOrderError)
        if self.layout.mode == MODE_PRODUCTION:
            _require((lam, int(seed)) in self.contract.authorized_production_units,
                     "This production assignment authorizes only the two seed-44 continuation units (lambda 0 and 0.01).",
                     RQ4bStageOrderError)
        prerequisites = ["preflight", "capability"]
        if self.layout.mode == MODE_PRODUCTION and (lam, int(seed)) == (0.01, 44):
            self._verify_seed44_control_artifacts()
            prerequisites += [unit_key("dev", 0.0, 44), unit_key("devval", 0.0, 44)]
        return self._fit_unit("dev", lam, seed, prerequisites, allow_new_attempt_after_failure)

    def _verify_seed44_control_artifacts(self) -> None:
        """Require an intact, complete lambda-zero seed-44 unit before lambda 0.01."""
        fit = read_stage(self.layout, unit_key("dev", 0.0, 44))
        validation = read_stage(self.layout, unit_key("devval", 0.0, 44))
        _require(
            fit.get("counts", {}).get("cv_fold_fits") == CV_FOLDS
            and fit.get("counts", {}).get("refit_fits") == 1,
            "Seed-44 lambda-zero fit accounting is incomplete.",
            RQ4bPackageError,
        )
        validation_predictions = validation.get("counts", {}).get("predictions", {}).values()
        _require(
            any(
                item.get("partition") == "validation"
                and item.get("rows") == self.contract.split_rows["validation"]
                for item in validation_predictions
            ),
            "Seed-44 lambda-zero Validation prediction count is incomplete.",
            RQ4bPackageError,
        )

    def _fit_unit(self, prefix: str, lam: float, seed: int, prerequisites: list[str], allow_retry: bool) -> dict[str, Any]:
        key = unit_key(prefix, lam, seed)
        inputs = self._inputs(prerequisites)
        with StageWriter(self.layout, key, stage=f"{prefix}_fit", unit={"adversary_lambda": lam, "model_seed": seed},
                         inputs=inputs, allow_new_attempt_after_failure=allow_retry) as writer:
            writer.counts["fit_accounting"] = {
                "planned": {"cv_fold_fits": CV_FOLDS, "refit_fits": 1},
                "started": None, "completed": None, "partial": None, "unknown": True,
                "last_position": {"phase": "backend_dispatch", "fold": None, "epoch": None, "batch": None},
                "last_progress_utc": datetime.now(timezone.utc).isoformat(),
                "source": "backend dispatch recorded; fit ledger not yet returned",
            }
            writer.progress_event("backend_dispatch", phase="backend_dispatch", optimizer_progress_claimed=False)
            result = self.backend.fit_unit(lam, seed, writer.temp)
            _require(result["cv_fold_fits"] == CV_FOLDS and result["refit_fits"] == 1, "A fit unit must run exactly five CV folds and one refit.")
            writer.counts["cv_fold_fits"] = result["cv_fold_fits"]
            writer.counts["refit_fits"] = result["refit_fits"]
            writer._sync_fit_ledger()
            if writer.counts["fit_accounting"].get("source") != "fit_ledger.json":
                writer.counts["fit_accounting"] = {
                    "planned": {"cv_fold_fits": CV_FOLDS, "refit_fits": 1},
                    "started": {"cv_fold_fits": result["cv_fold_fits"], "refit_fits": result["refit_fits"]},
                    "completed": {"cv_fold_fits": result["cv_fold_fits"], "refit_fits": result["refit_fits"]},
                    "partial": {"cv_fold_fits": 0, "refit_fits": 0}, "unknown": False,
                    "last_position": {"phase": "backend_complete", "fold": None, "epoch": None, "batch": None},
                    "last_progress_utc": datetime.now(timezone.utc).isoformat(), "source": "validated backend result",
                }
            curves = np.asarray(result["curves"], dtype=np.float64)
            _require(curves.shape == (CV_FOLDS, ri.EPOCHS_PER_FOLD) and np.isfinite(curves).all(), "CV curves must be 5 x 256 finite values.")
            writer.write_json("cv_curves.json", {"curves": curves, "evidence_kind": result["evidence_kind"]})
            writer.write_json("selection_record.json", result["selection_record"])
            writer.write_json("cv_fold_outputs.json", result.get("cv_fold_outputs", []))
            writer.write_json("diagnostics.json", result.get("diagnostics", {}))
        return read_stage(self.layout, key)

    def _predict_validation_unit(self, prefix: str, lam: float, seed: int) -> dict[str, Any]:
        fit_key = unit_key(prefix, lam, seed)
        key = unit_key(f"{prefix}val", lam, seed)
        inputs = self._inputs([fit_key])
        with StageWriter(self.layout, key, stage=f"{prefix}_validation_prediction", unit={"adversary_lambda": lam, "model_seed": seed}, inputs=inputs) as writer:
            predictions = self._predict_once(writer, fit_key, "validation", "validation_predictions.csv")
            selection = ev.calibrate_threshold(predictions.y, predictions.p_approved)
            writer.write_json("threshold.json", {"threshold": selection.threshold, "balanced_accuracy": selection.balanced_accuracy,
                                                 "tied_thresholds": selection.tied_thresholds, "candidate_count": selection.candidate_count,
                                                 "source_prediction": "validation_predictions.csv"})
        return read_stage(self.layout, key)

    def development_validation(self, adversary_lambda: float, seed: int) -> dict[str, Any]:
        lam = float(adversary_lambda)
        _require((lam, int(seed)) in ev.ADDITIONAL_DEVELOPMENT_UNITS,
                 "Track B validation may use only the four approved additional development units.",
                 RQ4bStageOrderError)
        if self.layout.mode == MODE_PRODUCTION:
            _require((lam, int(seed)) in self.contract.authorized_production_units,
                     "This production assignment authorizes validation only for the two seed-44 continuation units.",
                     RQ4bStageOrderError)
        return self._predict_validation_unit("dev", lam, seed)

    def _artifact_ref(self, prefix: str, lam: float, seed: int) -> ev.ArtifactRef:
        fit_key, val_key = unit_key(prefix, lam, seed), unit_key(f"{prefix}val", lam, seed)
        fit = read_stage(self.layout, fit_key)
        threshold = read_json(self.layout.stage_dir(val_key) / "threshold.json")["threshold"]
        model_digest = hashlib.sha256(json.dumps(fit["outputs"], sort_keys=True).encode()).hexdigest()
        return ev.ArtifactRef(lam, seed, fit_key, model_digest, fit["outputs"]["selection_record.json"], float(threshold))

    def validation_selection(self) -> dict[str, Any]:
        _require(False, "In-run selection is inactive under the nine-unit contract: the units span three immutable source runs, "
                 "so selection runs only through the evaluation-only gate_c_aggregation stage.", RQ4bStageOrderError)
        _require(ev.SELECTION_POLICY_STATUS == "approved_evaluation_only_three_seed_validation_selection",
                 "Gate C selection requires the locked evaluation-only three-seed Validation policy.",
                 RQ4bStageOrderError)
        units = list(ev.DEVELOPMENT_UNITS)
        keys = [unit_key("devval", lam, seed) for lam, seed in units]
        inputs = self._inputs(keys)
        with StageWriter(self.layout, "selection", stage="validation_selection", unit={}, inputs=inputs) as writer:
            evaluations, references = [], {}
            for lam, seed in units:
                predictions = read_predictions(self.layout.stage_dir(unit_key("devval", lam, seed)) / "validation_predictions.csv")
                evaluation = ev.evaluate_candidate(predictions, adversary_lambda=lam, model_seed=seed)
                stored = read_json(self.layout.stage_dir(unit_key("devval", lam, seed)) / "threshold.json")["threshold"]
                _require(evaluation.threshold.threshold == stored, "Recomputed Validation threshold differs from the stored threshold.", RQ4bPackageError)
                evaluations.append(evaluation)
                references[(lam, seed)] = self._artifact_ref("dev", lam, seed)
            selection = ev.select_lambda(evaluations)
            _require(selection["status"] == "selected", f"Lambda selection is not evaluable: {selection.get('reason')}")
            plan = ev.plan_final_arms(selection["selected_lambda"], references)
            writer.write_json("candidates.json", selection)
            writer.write_json("selection_lock.json", {
                "selected_lambda": selection["selected_lambda"], "lambda_zero_selected": selection["lambda_zero_selected"],
                "development_artifacts": [references[cell] for cell in units],
                "final_arm_plan": {str(seed): {role: jsonable(arm) for role, arm in arms.items()} for seed, arms in plan.items()},
                "required_new_fits": ev.required_new_fits(plan),
            })
        return read_stage(self.layout, "selection")

    def _lock(self) -> dict[str, Any]:
        read_stage(self.layout, "selection")
        return read_json(self.layout.stage_dir("selection") / "selection_lock.json")

    def final_fit(self, adversary_lambda: float, seed: int, *, allow_new_attempt_after_failure: bool = False) -> dict[str, Any]:
        raise RQ4bStageOrderError("final_fit is inactive under the amended nine-unit contract; seeds 45-46 and additional post-selection fits are not authorized.")

    def final_validation(self, adversary_lambda: float, seed: int) -> dict[str, Any]:
        raise RQ4bStageOrderError("final_validation is inactive under the amended nine-unit contract.")

    def _final_arms(self) -> dict[int, dict[str, tuple[str, float]]]:
        """Per final seed: (artifact fit key, locked threshold) for control and adversarial arms."""
        lock = self._lock()
        dev = {(float(r["adversary_lambda"]), int(r["model_seed"])): r for r in lock["development_artifacts"]}
        arms: dict[int, dict[str, tuple[str, float]]] = {}
        for seed in ev.FINAL_SEEDS:
            plan = lock["final_arm_plan"][str(seed)]
            resolved: dict[str, tuple[str, float]] = {}
            for role, lam in (("control", 0.0), ("adversarial", float(lock["selected_lambda"]))):
                arm = plan[role]
                if arm["source"] == "same_as_control":
                    resolved[role] = resolved["control"]
                elif arm["source"] == "reuse_development":
                    ref = ev.ArtifactRef(**{k: arm["artifact"][k] for k in ev.ArtifactRef.__dataclass_fields__})
                    current = self._artifact_ref("dev", lam, seed)
                    ev.validate_reuse_identity(current, ref)
                    _require(dev[(lam, seed)]["artifact_id"] == ref.artifact_id, "Reuse plan and development registry differ.", RQ4bPackageError)
                    resolved[role] = (ref.artifact_id, ref.threshold)
                else:
                    val_key = unit_key("finalval", lam, seed)
                    read_stage(self.layout, unit_key("final", lam, seed))  # model files re-hashed before any Test use
                    read_stage(self.layout, val_key)
                    threshold = read_json(self.layout.stage_dir(val_key) / "threshold.json")["threshold"]
                    resolved[role] = (unit_key("final", lam, seed), float(threshold))
            arms[seed] = resolved
        return arms

    def test_evaluation(self) -> dict[str, Any]:
        _require(False, "The local-run Test route is retired; use evaluate_cross_run_test with pinned Gate C and source runs.",
                 RQ4bStageOrderError)
        lock = self._lock()
        required = [unit_key("finalval", lam, seed) for lam, seed in lock["required_new_fits"]]
        missing = [key for key in required if not self.layout.stage_dir(key).is_dir()]
        _require(not missing, f"Test stays locked: final-arm threshold locks are missing: {missing}.", RQ4bStageOrderError)
        arms = self._final_arms()  # verifies every reused/final artifact and threshold before Test is opened
        inputs = self._inputs(["selection", *required])
        with StageWriter(self.layout, "test_evaluation", stage="test_evaluation", unit={}, inputs=inputs) as writer:
            predictions: dict[str, ev.PartitionPredictions] = {}
            for seed in ev.FINAL_SEEDS:
                for role in ("control", "adversarial"):
                    artifact = arms[seed][role][0]
                    if artifact not in predictions:  # a deduplicated lambda=0 arm is predicted once
                        predictions[artifact] = self._predict_once(writer, artifact, "test", f"test_predictions_{artifact}.csv")
            reference = next(iter(predictions.values()))
            _require(all(p.record_ids == reference.record_ids for p in predictions.values()), "Test predictions of different artifacts cover different ordered record IDs.")
            indices = self.backend.load_test_indices(reference.y, reference.record_ids)
            ev.validate_locked_indices(indices, reference.y)
            per_seed = {}
            for seed in ev.FINAL_SEEDS:
                per_seed[seed] = self._seed_result(seed, arms, predictions, indices)
                writer.write_json(f"seed_{seed}.json", per_seed[seed])
            assessment = ev.assess_final_seeds(per_seed)
            writer.write_json("assessment.json", assessment)
            writer.write_json("status.json", status_from_assessment(assessment))
        return read_stage(self.layout, "test_evaluation")

    def _validation_predictions(self, fit_key: str) -> ev.PartitionPredictions:
        """Saved complete-Validation predictions of a fit artifact (verified; no new inference)."""
        prefix, rest = fit_key.split("_", 1)
        val_key = f"{prefix}val_{rest}"
        read_stage(self.layout, val_key)
        return read_predictions(self.layout.stage_dir(val_key) / "validation_predictions.csv")

    def _seed_result(self, seed: int, arms: Mapping[int, Mapping[str, tuple[str, float]]],
                     test_predictions: Mapping[str, ev.PartitionPredictions], indices: np.ndarray) -> dict[str, Any]:
        """Per-seed Test evaluation, lineage and fairness generalization gap from locked artifacts and thresholds."""
        (c_key, c_threshold), (a_key, a_threshold) = arms[seed]["control"], arms[seed]["adversarial"]
        control, adversarial = test_predictions[c_key], test_predictions[a_key]
        result = ev.evaluate_seed(control, adversarial, control_threshold=c_threshold, adversarial_threshold=a_threshold, indices=indices)
        result["lineage"] = {
            "control_artifact": c_key, "adversarial_artifact": a_key, "control_threshold": c_threshold, "adversarial_threshold": a_threshold,
            "control_test_predictions": f"test_predictions_{c_key}.csv", "adversarial_test_predictions": f"test_predictions_{a_key}.csv",
            "test_record_id_sha256": sha256_ids(control.record_ids), "adversarial_test_record_id_sha256": sha256_ids(adversarial.record_ids),
            "deduplicated_lambda_zero": c_key == a_key,
        }
        result["fairness_generalization_gap"] = {
            role: generalization_gap(test_predictions[key], self._validation_predictions(key), threshold)
            for role, (key, threshold) in arms[seed].items()
        }
        return result

    def package_validation(self) -> dict[str, Any]:
        _require(False, "The local-run package route is retired; use validate_cross_run_test_package.", RQ4bStageOrderError)
        receipts = {key: read_stage(self.layout, key) for key in self._published_keys()}
        _require("test_evaluation" in receipts, "Package validation requires a published test_evaluation stage.", RQ4bStageOrderError)
        inputs = {key: sha256_file(self.layout.stage_dir(key) / RECEIPT_FILE) for key in receipts}
        with StageWriter(self.layout, "package_validation", stage="package_validation", unit={}, inputs=inputs) as writer:
            manifest = read_json(self.layout.run_dir / MANIFEST_FILE)
            for key, receipt in receipts.items():
                _require(receipt["mode"] == manifest["mode"] == self.layout.mode, f"{key} mode differs from the run mode.", RQ4bPackageError)
                _require(receipt["code_sha256"] == manifest["code_sha256"], f"{key} was produced by different code.", RQ4bPackageError)
                for upstream, digest in receipt["inputs"].items():
                    _require(sha256_file(self.layout.stage_dir(upstream) / RECEIPT_FILE) == digest, f"{key} input {upstream} changed after use.", RQ4bPackageError)
            cv_fits = sum(r["counts"]["cv_fold_fits"] for r in receipts.values())
            refits = sum(r["counts"]["refit_fits"] for r in receipts.values())
            _require(cv_fits <= CV_FIT_BUDGET and refits <= REFIT_BUDGET, f"Fit counts exceed the declared budget: cv={cv_fits}, refit={refits}.", RQ4bPackageError)
            lock = self._lock()
            expected_units = len(ev.DEVELOPMENT_UNITS)
            _require(refits == expected_units and cv_fits == expected_units * CV_FOLDS, "Fit counts differ from the published units.", RQ4bPackageError)
            self._verify_report_traces()
            writer.write_json("package_report.json", {
                "fit_counts": {"cv_fold_fits": cv_fits, "refit_fits": refits, "declared_cv_fold_fit_budget": CV_FIT_BUDGET,
                               "declared_refit_budget": REFIT_BUDGET, "development_units": len(ev.DEVELOPMENT_UNITS),
                               "retained_development_units": len(ev.RETAINED_DEVELOPMENT_UNITS),
                               "additional_development_units": len(ev.ADDITIONAL_DEVELOPMENT_UNITS),
                               "final_new_units": 0, "excludes": "capability probe fits, spike, retries and failed attempts"},
                "failed_attempts_retained": sorted(p.name for p in self.layout.failures.iterdir()),
                "stages_verified": sorted(receipts),
                "report_values_trace_to_saved_inputs": True,
                "traced_outputs": ["seed_*.json (metrics, bootstrap, criteria, lineage, fairness_generalization_gap)", "assessment.json", "status.json"],
            })
        return read_stage(self.layout, "package_validation")

    def _verify_report_traces(self) -> None:
        """Recompute per-seed results and the assessment from saved predictions, thresholds and indices."""
        directory = self.layout.stage_dir("test_evaluation")
        arms = self._final_arms()
        saved = {p.name[len("test_predictions_"):-4]: read_predictions(p) for p in directory.glob("test_predictions_*.csv")}
        reference = next(iter(saved.values()))
        indices = self.backend.load_test_indices(reference.y, reference.record_ids)
        expected_files = {f"test_predictions_{arms[s][role][0]}.csv" for s in ev.FINAL_SEEDS for role in ("control", "adversarial")}
        _require({f"test_predictions_{k}.csv" for k in saved} == expected_files, "Saved Test prediction files differ from the locked arms.", RQ4bPackageError)
        # Every saved Test file must carry the exact record digest recorded at prediction time and the
        # same ordered Test IDs as the reference that load_test_indices checks against the locked manifest.
        recorded = read_stage(self.layout, "test_evaluation")["counts"]["predictions"]
        reference_digest = sha256_ids(reference.record_ids)
        for artifact, predictions in saved.items():
            name = f"test_predictions_{artifact}.csv"
            digest = sha256_ids(predictions.record_ids)
            _require(recorded.get(name, {}).get("record_id_sha256") == digest and recorded[name]["artifact"] == artifact and recorded[name]["partition"] == "test",
                     f"{name} record IDs differ from those recorded at Test prediction time.", RQ4bPackageError)
            _require(digest == reference_digest, f"{name} covers different ordered Test IDs from the locked reference.", RQ4bPackageError)
        per_seed = {}
        for seed in ev.FINAL_SEEDS:
            recomputed = self._seed_result(seed, arms, saved, indices)  # metrics, lineage and generalization gap
            stored = read_json(directory / f"seed_{seed}.json")
            for part in ("lineage", "fairness_generalization_gap"):
                _require(jsonable(recomputed[part]) == stored.get(part), f"Seed {seed} {part} does not trace to the locks and saved inputs.", RQ4bPackageError)
            _require(jsonable(recomputed) == stored, f"Seed {seed} results do not trace to the saved inputs.", RQ4bPackageError)
            per_seed[seed] = recomputed
        assessment = ev.assess_final_seeds(per_seed)
        _require(jsonable(assessment) == read_json(directory / "assessment.json"), "Assessment does not trace to saved inputs.", RQ4bPackageError)
        _require(jsonable(status_from_assessment(assessment)) == read_json(directory / "status.json"),
                 "status.json does not trace to the verified assessment.", RQ4bPackageError)


# ---------------------------------------------------------------------------
# Production backend (real data; exercised only by an explicitly launched production run)
# ---------------------------------------------------------------------------


class ProductionBackend:
    """Real T1-T4 glue. Never used by tests; production runs need explicit user launch."""

    kind = MODE_PRODUCTION

    def __init__(self, contract: ri.RQ4bContract, *, config_path: Path = ri.RQ4B_CONFIG_PATH,
                 real_probe_evidence: Path | None = None) -> None:
        self.contract = contract
        self.config_path = config_path
        self.real_probe_evidence = real_probe_evidence
        self._context = None

    def _training(self):
        from src.dataset_b.mitigation import rq4b_training as rt

        return rt

    def context(self):
        if self._context is None:
            self._context = self._training().open_production_context(self.config_path)
        return self._context

    def preflight(self) -> dict[str, str]:
        config = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8"))
        governance = config.get("execution_governance", {})
        statuses = {name: (governance.get(name) or {}).get("result") for name in REQUIRED_ACCEPTANCE}
        context = self.context()
        statuses.update({f"prerequisite_{name}": value for name, value in context.prerequisites.items()})
        statuses["cuda0"] = "PASS" if context.cuda.get("device") == "cuda:0" else "FAIL"
        indices = self.contract.resources["locked_test_bootstrap_indices"]
        statuses["rq1_locked_bootstrap_present"] = "PASS" if indices.is_file() and (indices.parent / "rq1_test_bootstrap_manifest.json").is_file() else "FAIL"
        return statuses

    def capability_evidence(self) -> dict[str, Any]:
        config = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8"))
        record = config["execution_governance"]["t3_capability_probe_acceptance"]
        gate = record["synthetic_gate"]
        path = PROJECT_ROOT / gate["evidence_path"]
        real = {"status": "not_run_requires_separate_authorization", "capability_status": "capability_inconclusive"}
        if self.real_probe_evidence is not None:
            from src.dataset_b.mitigation import rq4b_probe as rp

            real = rp.load_real_probe_evidence(self.real_probe_evidence, self.contract)
        return {"synthetic_gate_status": gate["result"] if record.get("result") == "PASS" else "FAIL",
                "evidence_path": gate["evidence_path"], "evidence_sha256": sha256_file(path),
                "real_probe": real}

    def _population(self, access: ri.StageAccess):
        rt = self._training()
        gated = ri.load_partition(access, "train")
        ids = gated.record_ids.astype(str).tolist()
        y = gated.y.to_numpy(dtype=np.int64)
        raw = ri.read_derived_race_rows(access, ids, permitted_ids=ri.stage_population_ids(access), reconciliation_fields=("lei", "applicant_sex"))
        reconciliation = pd.DataFrame({"lei": gated.audit["lei"].astype(str).to_numpy(), "applicant_sex": gated.audit["applicant_sex"].astype(str).to_numpy()})
        race = ri.join_derived_race(ids, y, raw, self.contract, reconciliation=reconciliation)
        return rt.TrainPopulation(gated.X.reset_index(drop=True), y, tuple(ids), race)

    def fit_unit(self, adversary_lambda: float, seed: int, out_dir: Path) -> dict[str, Any]:
        import joblib

        rt = self._training()
        attempt_id = out_dir.name
        match = re.fullmatch(r"\.tmp_(.+)_[0-9a-f]{32}", attempt_id)
        unit = match.group(1) if match is not None else attempt_id
        progress = rt.DurableProgress(
            out_dir, run_id=out_dir.parent.parent.name, attempt_id=attempt_id,
            unit=unit, adversary_lambda=adversary_lambda, model_seed=seed,
        )
        context = self.context()
        cv_access = ri.StageAccess("cv_epoch_selection", self.contract)
        record = rt.run_cv_epoch_selection(
            context, cv_access, self._population(cv_access), adversary_lambda=adversary_lambda,
            model_seed=seed, progress=progress,
        )
        refit_access = ri.StageAccess("full_train_refit", self.contract)
        population = self._population(refit_access)
        refit = rt.run_full_train_refit(
            context, refit_access, population, record, adversary_lambda=adversary_lambda,
            model_seed=seed, progress=progress,
        )
        joblib.dump(refit.preprocessor, out_dir / "preprocessor.joblib")
        sample = refit.preprocessor.transform(population.X_raw.iloc[:2048].reset_index(drop=True))
        reload = rt.save_and_verify_reload(refit.estimator, sample, out_dir / "estimator.joblib", contract=self.contract)
        progress.emit("unit_completed", phase="serialization", optimizer_progress_claimed=False)
        progress.finalize()
        return {
            "cv_fold_fits": CV_FOLDS, "refit_fits": 1, "curves": record.curves, "evidence_kind": record.evidence_kind,
            "selection_record": {"adversary_lambda": record.adversary_lambda, "model_seed": record.model_seed, "selection": record.selection,
                                 "fold_membership_sha256": record.fold_membership_sha256, "fold_summaries": record.fold_summaries,
                                 "evidence_kind": record.evidence_kind},
            "cv_fold_outputs": record.fold_outputs,
            "diagnostics": {"refit_history": refit.history, "refit_log": refit.log, "reload": reload, "epoch_cap_hit": refit.epoch_cap_hit},
        }

    def predict(self, fit_dir: Path, partition: str) -> ev.PartitionPredictions:
        import joblib

        stage = {"validation": "validation_calibration_selection", "test": "test_evaluation"}[partition]
        access = ri.StageAccess(stage, self.contract)
        gated = ri.load_partition(access, partition)
        ids = gated.record_ids.astype(str).tolist()
        y = gated.y.to_numpy(dtype=np.int64)
        preprocessor = joblib.load(fit_dir / "preprocessor.joblib")
        estimator = joblib.load(fit_dir / "estimator.joblib")
        p = estimator.predict_proba(preprocessor.transform(gated.X.reset_index(drop=True)))[:, 1]
        raw = ri.read_derived_race_rows(access, ids, permitted_ids=ri.stage_population_ids(access), reconciliation_fields=("lei", "applicant_sex"))
        reconciliation = pd.DataFrame({"lei": gated.audit["lei"].astype(str).to_numpy(), "applicant_sex": gated.audit["applicant_sex"].astype(str).to_numpy()})
        race = ri.join_derived_race(ids, y, raw, self.contract, reconciliation=reconciliation)
        rq3 = yaml.safe_load((PROJECT_ROOT / "config" / "dataset_b" / "rq3_protocol.yaml").read_text(encoding="utf-8"))
        sex = ev.normalize_rq3_sex_codes(gated.audit["applicant_sex"].astype(str).tolist(), rq3)
        return ev.PartitionPredictions(tuple(ids), y.astype(np.int8), np.asarray(p, dtype=np.float64), race, sex)

    def load_test_indices(self, y: np.ndarray, record_ids: Sequence[str]) -> np.ndarray:
        access = ri.StageAccess("test_evaluation", self.contract)
        path = access.require_resource("locked_test_bootstrap_indices")
        manifest = read_json(path.parent / "rq1_test_bootstrap_manifest.json")
        _require(sha256_file(path) == manifest.get("indices_sha256"), "Locked Test bootstrap index file hash differs from its manifest.")
        _require(manifest.get("test_record_id_sha256") == sha256_ids(record_ids), "Test record order differs from the locked bootstrap manifest.")
        _require(manifest.get("test_rows") == len(record_ids), "Test row count differs from the locked bootstrap manifest.")
        with np.load(path, allow_pickle=False) as payload:
            _require(payload.files == ["indices"], "Locked bootstrap artifact must contain only 'indices'.")
            indices = payload["indices"]
        return ev.validate_locked_indices(indices, y)


# ---------------------------------------------------------------------------
# CLI: production only, one explicit unit per command
# ---------------------------------------------------------------------------


STAGE_COMMANDS = ("create", "preflight", "capability", "development", "development_validation", "selection",
                  "final_fit", "final_validation", "test_evaluation", "package_validation", GATE_C_STAGE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RQ4b-3 production runner: one explicit stage unit per command.")
    parser.add_argument("--stage", required=True, choices=STAGE_COMMANDS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lambda", dest="adversary_lambda", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--confirm-production", action="store_true", required=True,
                        help="Required acknowledgement that this launches real RQ4b production work.")
    parser.add_argument("--new-attempt-after-failure", action="store_true")
    parser.add_argument("--real-probe-evidence", type=Path,
                        help="Validated real_probe_evidence.json to ingest during capability only; never fits a probe.")
    args = parser.parse_args(argv)
    _require(args.real_probe_evidence is None or args.stage == "capability", "--real-probe-evidence is only valid for capability.")
    contract = ri.load_contract()
    root = contract.artifact_namespace
    if args.stage == GATE_C_STAGE:
        # Evaluation-only: no backend is constructed, so nothing can fit, infer or open Test.
        _require(args.adversary_lambda is None and args.seed is None and not args.new_attempt_after_failure,
                 f"--stage {GATE_C_STAGE} takes no --lambda/--seed/--new-attempt-after-failure.")
        _require(args.run_id == contract.aggregation_run_id, f"Gate C aggregation must use the pinned run id {contract.aggregation_run_id}.")
        layout = aggregate_gate_c_sources(root, contract.aggregation_namespace, contract)
        receipt = read_stage(layout, GATE_C_STAGE)
        print(json.dumps({"run_dir": str(layout.run_dir), "stage": receipt["stage_key"], "status": receipt["status"], "counts": receipt["counts"]}, indent=2))
        return 0
    if args.stage in (TEST_STAGE, PACKAGE_STAGE):
        _require(args.adversary_lambda is None and args.seed is None and not args.new_attempt_after_failure,
                 f"--stage {args.stage} takes no --lambda/--seed/--new-attempt-after-failure.")
        _require(args.run_id == contract.test_run_id, f"Cross-run Test must use the pinned run id {contract.test_run_id}.")
        if args.stage == TEST_STAGE:
            layout = evaluate_cross_run_test(root, contract.aggregation_namespace, contract.test_namespace,
                                             ProductionBackend(contract), contract)
            receipt = read_stage(layout, TEST_STAGE)
        else:
            receipt = validate_cross_run_test_package(root, contract.aggregation_namespace, contract.test_namespace, contract)
        print(json.dumps({"stage": receipt["stage_key"], "status": receipt["status"], "counts": receipt["counts"]}, indent=2))
        return 0
    if args.stage == "create":
        layout = create_run(root, args.run_id, MODE_PRODUCTION, contract)
        print(json.dumps({"run_dir": str(layout.run_dir)}))
        return 0
    backend = ProductionBackend(contract) if args.real_probe_evidence is None else ProductionBackend(contract, real_probe_evidence=args.real_probe_evidence)
    runner = RQ4bRunner(open_run(root, args.run_id, MODE_PRODUCTION, contract), backend, contract)
    unit_stages = {"development", "development_validation", "final_fit", "final_validation"}
    if args.stage in unit_stages:
        _require(args.adversary_lambda is not None and args.seed is not None, f"--stage {args.stage} requires --lambda and --seed.")
    else:
        _require(args.adversary_lambda is None and args.seed is None, f"--stage {args.stage} takes no --lambda/--seed.")
    calls: dict[str, Callable[[], Any]] = {
        "preflight": runner.preflight,
        "capability": runner.capability,
        "development": lambda: runner.development(args.adversary_lambda, args.seed, allow_new_attempt_after_failure=args.new_attempt_after_failure),
        "development_validation": lambda: runner.development_validation(args.adversary_lambda, args.seed),
        "selection": runner.validation_selection,
        "final_fit": lambda: runner.final_fit(args.adversary_lambda, args.seed, allow_new_attempt_after_failure=args.new_attempt_after_failure),
        "final_validation": lambda: runner.final_validation(args.adversary_lambda, args.seed),
        "test_evaluation": runner.test_evaluation,
        "package_validation": runner.package_validation,
    }
    receipt = calls[args.stage]()
    print(json.dumps({"stage": receipt["stage_key"], "status": receipt["status"], "counts": receipt["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
