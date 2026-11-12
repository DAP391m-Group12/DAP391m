"""Independent, new-only C17 orchestration. CLI is deliberately dry-run only.

No heavy imports, data access, directory creation or fitting at import time.
Execution APIs require explicit, source-bound gates from a later authorized task.
They never use the old runner, selection records, or original estimators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config/dataset_b/rq4b_c17_independent.yaml"
RUN_ROOT = ROOT / "artifacts/dataset_b/rq4/rq4b_c17/rq4b_c17_20261111_02"
SCRATCH = ROOT / "scratch/rq4b_c17_independent"
BASELINE_DIGEST = "52465d2650e12bb40c7161db72095d0a5d8ee46c67e5dfa87f58dbed8d01b93f"
POLICY = "FIXED_BUDGET_SENSITIVITY"
SEEDS = (42, 43, 44)
EPOCHS, HORIZON, BATCHES, UPDATES = 17, 256, 801, 13617
_WRITE_ROOTS = None  # Process-wide so Python worker threads are fenced too.
_HOOK_INSTALLED = False


class C17Error(RuntimeError):
    """Fail closed; never repair, overwrite, retry or change protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C17Error(message)


def canonical(path: Path) -> Path:
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Absolute path without traversal required.")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in path.parts:
        if part != path.anchor:
            require(part == part.rstrip(" .") and part.split(".")[0].upper() not in reserved,
                    "Ambiguous Windows path component refused.")
    for ancestor in (path, *path.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not (
            getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
        ), f"Reparse/symlink path refused: {ancestor}")
    return path.resolve()


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with canonical(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(canonical(path).read_text(encoding="utf-8"))


def inventory_digest(inventory: Mapping[str, str]) -> str:
    payload = "".join(f"{key}\t{inventory[key]}\n" for key in sorted(inventory))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_baseline(inventory: Mapping[str, str]) -> None:
    require(len(inventory) == 137 and inventory_digest(inventory) == BASELINE_DIGEST,
            "Not the frozen 137-file Gate A baseline.")
    for name, expected in inventory.items():
        path = canonical(ROOT / name)
        require(within(path, ROOT), "Baseline path escapes repository.")
        require(sha256(path) == expected, f"Protected checksum changed: {name}")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    import yaml
    require(canonical(path) == canonical(CONFIG), "Only the approved C17 config is supported.")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "experiment_type": "posthoc_epoch_matched_sensitivity",
        "epoch_policy": POLICY, "execution_epochs": EPOCHS, "schedule_horizon_epochs": HORIZON,
        "adversary_lambda": 0, "seeds": list(SEEDS), "batch_size": 256, "device": "cuda:0",
        "split_rows": {"train": 205310, "validation": 43995, "test": 43996},
        "cv_fits_per_seed": 0, "full_train_fits_per_seed": 1,
        "recipe": "config/dataset_b/realmlp_v1.yaml",
        "input_contract": "config/dataset_b/rq4b_adversarial.yaml",
        "run_root": RUN_ROOT.relative_to(ROOT).as_posix(),
        "scratch_root": SCRATCH.relative_to(ROOT).as_posix(),
        "reference_manifest": "artifacts/dataset_b/rq4/rq4b/test_evaluation/rq4b3_test_20261024_01/run_manifest.json",
        "reference_manifest_sha256": "d05fc2980986452428b5fc656aaba89c8bad698923cab7da67f5ff241e7595a4",
        "bootstrap_sha256": "2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1",
        "bootstrap_replicates": 1000, "minimum_valid_replicates_per_quantity": 1000,
        "quantile_method": "linear", "quantities": ["ba", "eod3", "eod6"],
        "contrasts": ["A17-C17", "C17-C_long"],
        "threshold": {"start": .001, "stop": .999, "step": .001, "count": 999,
                      "tie_break": ["closest_to_0.5", "lower"]},
        "replay": {"atol": 1e-6, "rtol": 1e-5, "purpose": "diagnostic_alert_not_weight_equivalence"},
        "acceptance_policy": "independent_fixed_epoch_control_v1",
        "original_results": "read_only_no_recomputation",
    }
    # JSON comparison is type-sensitive (e.g. true cannot replace epoch 1).
    require(json.dumps(config, sort_keys=True) == json.dumps(expected, sort_keys=True), "C17 configuration drift.")
    return config


def arm_id(seed: int, arm: str = "C17") -> str:
    require(type(seed) is int and seed in SEEDS, "Unapproved seed.")
    require(arm in ("C17", "A17", "C_long"), "Unapproved arm.")
    if arm == "C17":
        return f"C17-{POLICY}-E17-H256-L0-S{seed}"
    return f"{arm}-ACCEPTED-S{seed}"


def dry_run() -> dict[str, Any]:
    config = load_config()
    canonical(RUN_ROOT)
    require(not RUN_ROOT.exists(), "Production run already exists; no implicit resume.")
    return {"experiment_type": config["experiment_type"], "execution_authorized": False,
            "run_root": str(RUN_ROOT), "cv_fits": 0, "full_train_fits": 3,
            "units": [{"arm_id": arm_id(seed), "seed": seed, "lambda": 0,
                       "execution_epochs": EPOCHS, "schedule_horizon_epochs": HORIZON} for seed in SEEDS],
            "original_fits": 0, "original_inference": 0, "original_recomputation": False,
            "contrasts": config["contrasts"]}


class OutputGuard:
    """Only exact production root or isolated descendants of approved scratch."""

    def __init__(self, root: Path):
        self.root = canonical(root)
        require(self.root == canonical(RUN_ROOT) or (
            within(self.root, canonical(SCRATCH)) and self.root != canonical(SCRATCH)
        ), "Unapproved output root.")

    def path(self, relative: str, *, new: bool = False) -> Path:
        part = Path(relative)
        require(relative and not part.is_absolute() and ".." not in part.parts
                and ":" not in relative, "Invalid output name.")
        target = canonical(self.root / part)
        require(within(target, self.root) and target != self.root, "Output escapes root.")
        require(not new or not target.exists(), f"Refusing overwrite: {target}")
        return target

    def create(self) -> None:
        canonical(self.root)
        require(not self.root.exists(), "Existing root; resume/refit refused.")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(exist_ok=False)

    def mkdir(self, relative: str) -> Path:
        path = self.path(relative, new=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        return path

    def write_json(self, relative: str, value: Any) -> None:
        path = self.path(relative, new=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def is_os_null_device(path: Any) -> bool:
    """Recognize only the literal Windows os.devnull spelling (case insensitive).

    This exception belongs only to audited opens, not output-path validation.
    Absolute paths, NT aliases, extensions and trailing characters stay denied.
    """
    return (os.name == "nt" and isinstance(path, (str, bytes, os.PathLike))
            and os.fsdecode(path).casefold() == os.devnull.casefold())


def _audit_write(event: str, args: tuple) -> None:
    roots = _WRITE_ROOTS
    if roots is None:
        return
    paths = ()
    if event == "open":
        path, mode, flags = args
        if isinstance(path, int):
            return
        if is_os_null_device(path):
            return  # subprocess.DEVNULL is an OS sink, never an artifact file.
        if (mode and any(char in mode for char in "wax+")) or flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        ):
            paths = (path,)
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"):
        paths = (args[0],)
    elif event in ("os.rename", "os.link", "os.symlink"):
        paths = args[:2]
    for path in paths:
        resolved = canonical(Path(os.fsdecode(path)).absolute())
        require(any(within(resolved, root) for root in roots), f"Unapproved write: {resolved}")


class WriteFence:
    """Python audit fence; native/driver cache containment still needs runtime preflight."""

    def __init__(self, *roots: Path):
        self.roots = tuple(OutputGuard(root).root if root != SCRATCH else canonical(SCRATCH) for root in roots)

    def __enter__(self):
        global _HOOK_INSTALLED, _WRITE_ROOTS
        require(_WRITE_ROOTS is None, "Concurrent/nested execution fences are unsupported.")
        if not _HOOK_INSTALLED:
            sys.addaudithook(_audit_write)
            _HOOK_INSTALLED = True
        _WRITE_ROOTS = self.roots
        return self

    def __exit__(self, *_):
        global _WRITE_ROOTS
        _WRITE_ROOTS = None


@dataclass(frozen=True)
class ExecutionGates:
    """Later task supplies actual gate evidence; Task 02 CLI cannot create it."""
    approved_stages: frozenset[str]
    preflight_status: str
    baseline_digest: str
    module_sha256: str
    config_sha256: str

    def require_stage(self, stage: str) -> None:
        require(stage in self.approved_stages and self.preflight_status == "PASS", "Stage not authorized/preflight not PASS.")
        require(self.baseline_digest == BASELINE_DIGEST, "Gate baseline differs.")
        require(self.module_sha256 == sha256(Path(__file__)) and self.config_sha256 == sha256(CONFIG), "Gate source/config drift.")


def require_runtime_directories() -> None:
    """Launcher must set these before Python/core imports, not after CUDA init."""
    for variable in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "TORCH_HOME", "CUDA_CACHE_PATH"):
        require(variable in os.environ and within(canonical(Path(os.environ[variable])), canonical(SCRATCH)),
                f"Uncontained runtime directory: {variable}")
    require(sys.dont_write_bytecode, "Launch with -B before importing core.")


class FixedBudgetProgress:
    """New durable adapter. No old DurableProgress constructor or ledger edits."""

    def __init__(self, guard: OutputGuard, seed: int):
        self.guard, self.seed, self.unit = guard, seed, arm_id(seed)
        guard.mkdir(f"units/{self.unit}")
        self.ledger = {"arm_id": self.unit, "epoch_policy": POLICY,
                       "planned": {"cv_fold_fits": 0, "refit_fits": 1},
                       "started": {"cv_fold_fits": 0, "refit_fits": 0},
                       "completed": {"cv_fold_fits": 0, "refit_fits": 0},
                       "partial": {"cv_fold_fits": 0, "refit_fits": 0},
                       "status": "new", "completed_epochs": 0, "last_position": None}
        self._last_heartbeat = 0.0
        self._save()

    def _save(self) -> None:
        name = f"units/{self.unit}/fit_ledger.json"
        temporary = self.guard.path(name + ".tmp", new=True)
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(self.ledger, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.guard.path(name))

    def start(self) -> None:
        require(self.ledger["status"] == "new", "Duplicate fit/resume refused.")
        self.ledger["started"]["refit_fits"] = 1
        self.ledger["status"] = "running"
        self._save()

    def emit(self, event: str, *, phase: str, fold=None, epoch=None, batch=None, **values) -> None:
        require(phase == "refit" and fold is None and self.ledger["status"] == "running", "Unexpected fit phase/state.")
        require(type(epoch) is int and 1 <= epoch <= EPOCHS and type(batch) is int and 1 <= batch <= BATCHES,
                "Unexpected epoch/batch.")
        require(event in ("heartbeat", "epoch_completed"), "Unexpected engine event.")
        if event == "epoch_completed":
            require(epoch == self.ledger["completed_epochs"] + 1 and batch == BATCHES, "Epoch count/order mismatch.")
            require(values.get("optimizer_progress_claimed") is True, "Missing optimizer progress evidence.")
            self.ledger["completed_epochs"] = epoch
        self.ledger["last_position"] = {"phase": phase, "epoch": epoch, "batch": batch, "event": event}
        record = {"arm_id": self.unit, "event": event, "phase": phase, "epoch": epoch, "batch": batch, **values}
        for filename in ("training_history.jsonl", "run.log"):
            with self.guard.path(f"units/{self.unit}/{filename}").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self._save()

    def heartbeat(self, *, phase: str, fold, epoch: int, batch: int, gpu_memory_bytes) -> None:
        require(phase == "refit" and fold is None and self.ledger["status"] == "running", "Unexpected heartbeat phase.")
        require(type(epoch) is int and 1 <= epoch <= EPOCHS and type(batch) is int and 1 <= batch <= BATCHES,
                "Unexpected heartbeat position.")
        now = time.monotonic()
        if batch == 1 or batch % 100 == 0 or now - self._last_heartbeat >= 30:
            self.emit("heartbeat", phase=phase, fold=fold, epoch=epoch, batch=batch,
                      gpu_memory_bytes=gpu_memory_bytes, optimizer_progress_claimed=False)
            self._last_heartbeat = now

    def finish(self) -> None:
        require(self.ledger["status"] == "running" and self.ledger["completed_epochs"] == EPOCHS, "Incomplete fit.")
        self.ledger["completed"]["refit_fits"] = 1
        self.ledger["status"] = "completed"
        self._save()

    def fail(self, exc: BaseException) -> None:
        self.ledger["partial"]["refit_fits"] = 1
        self.ledger["status"] = "failed"
        self.ledger["error"] = {"type": type(exc).__name__, "message": str(exc), "automatic_retry": False}
        self._save()


def validate_fit(result: Any, seed: int) -> None:
    history, log = result.history, result.log
    require(history["spec"] == {"adversary_lambda": 0.0, "model_seed": seed, "adversary_seed": seed,
            "execution_epochs": EPOCHS, "schedule_horizon_epochs": HORIZON, "module_kind": "adversarial"},
            "Executed specification differs from the approved C17 unit.")
    expected = {"trainer_max_epochs": EPOCHS, "module_schedule_horizon_epochs": HORIZON,
                "batch_size": 256, "batches_per_epoch": BATCHES, "iterated_samples_per_epoch": 205056,
                "train_rows": 205310, "drop_last": True, "shuffle": True, "predict_batch_size": 1024}
    require(history["completed_epochs"] == EPOCHS and history["budget"] == expected, "Fit budget differs.")
    require(len(history["epochs"]) == EPOCHS and not history["fold_validation_ce"], "Unexpected epochs/CV.")
    for epoch, item in enumerate(history["epochs"], 1):
        losses = item["losses"]
        require(item["epoch"] == epoch and losses["batches"] == BATCHES
                and losses["adversary_updates"] == 0 and losses["adversary_bypassed_lambda_zero"] == BATCHES,
                "Predictor/adversary step counts differ.")
        require(math.isfinite(losses["mean_prediction_loss"]), "Nonfinite training CE.")
    require(log["predictor_updates"] == UPDATES and log["adversary_updates"] == 0, "Update totals differ.")
    isolation = log["adversary_rng_isolation"]
    require(isolation["cpu_rng_state_unchanged"] is True and isolation["cuda_rng_state_unchanged"] is True, "RNG isolation failed.")


def replay_diagnostic(history: Mapping, historical: Mapping) -> dict[str, Any]:
    require(len(history["epochs"]) == EPOCHS and len(historical["epochs"]) >= EPOCHS, "Missing replay prefix.")
    differences = []
    for actual, old in zip(history["epochs"], historical["epochs"][:EPOCHS]):
        for key in ("epoch", "first_batch_schedule", "last_batch_schedule"):
            require(actual[key] == old[key], f"Replay schedule/position mismatch: {key}")
        require(actual["losses"]["batches"] == old["losses"]["batches"] == BATCHES, "Replay step count mismatch.")
        value, reference = actual["losses"]["mean_prediction_loss"], old["losses"]["mean_prediction_loss"]
        require(math.isfinite(value) and math.isfinite(reference), "Nonfinite replay CE.")
        error, tolerance = abs(value - reference), 1e-6 + 1e-5 * abs(reference)
        differences.append({"epoch": actual["epoch"], "absolute_difference": error,
                            "tolerance": tolerance, "within_diagnostic_tolerance": error <= tolerance})
    return {"status": "PASS" if all(row["within_diagnostic_tolerance"] for row in differences) else "BLOCK",
            "purpose": "diagnostic_alert_not_weight_equivalence", "epochs": differences,
            "historical_batch_order_proven": False, "weight_equivalence_proven": False}


def serialize_reload(estimator: Any, sample: Any, guard: OutputGuard, relative: str, *, training=None, codec=None) -> dict:
    import numpy as np
    if training is None:
        from src.dataset_b.mitigation import rq4b_training as training
    if codec is None:
        import joblib as codec
    training.strip_batch_records(estimator)
    require(not training.training_metadata_present(estimator), "Training metadata before serialization.")
    # Lightning predict restores each module's incoming mode. Establish eval
    # before the round trip so prediction and the saved state share that mode.
    estimator.backend_.alg_interface_.model.eval()
    before = estimator.predict_proba(sample)
    require(before.shape == (len(sample), 2) and np.isfinite(before).all()
            and ((before >= 0) & (before <= 1)).all()
            and np.allclose(before.sum(axis=1), 1, rtol=0, atol=1e-6), "Invalid reload sample probabilities.")
    path = guard.path(relative, new=True)
    with path.open("xb") as handle:
        codec.dump(estimator, handle)
    digest = sha256(path)
    with path.open("rb") as handle:
        restored = codec.load(handle)
    after = restored.predict_proba(sample)
    require(sha256(path) == digest and before.shape == after.shape and np.array_equal(before, after), "Reload probabilities changed.")
    require(not training.training_metadata_present(restored), "Training metadata after reload.")
    module = restored.backend_.alg_interface_.model
    require(not module.model.training and str(restored.backend_.alg_interface_.device) == "cuda:0", "Reload mode/device differs.")
    return {"sha256": digest, "probabilities_exactly_equal": True, "training_metadata_pickled": False,
            "predictor_training_mode_after_reload": False, "interface_device_after_reload": "cuda:0"}


class CoreBackend:
    """Lazy bridge to immutable core; never loads accepted estimators."""

    def __init__(self):
        from src.dataset_b.mitigation import rq4b_inputs as ri, rq4b_training as rt, rq4b_evaluation as ev
        self.ri, self.rt, self.ev = ri, rt, ev
        self.contract = ri.load_contract()

    def provenance(self) -> dict:
        from importlib.metadata import version
        from importlib.util import find_spec
        lineage = self.rt.pytabkit_lineage()
        require(lineage["status"] == "PASS", "Backend lineage failed.")
        installed = Path(find_spec("pytabkit").origin).parent.parent
        backend_hashes = {name: sha256(installed / name) for name in lineage["files"]}
        backend_hashes.update({name: sha256(installed / name) for name in ("torch/optim/adam.py", "torch/version.py")})
        return {"pytabkit_lineage": lineage,
                "backend_sha256": backend_hashes, "python": sys.version,
                "versions": {name: version(name) for name in ("torch", "pytabkit", "numpy", "pandas", "pytorch-lightning")},
                "source_sha256": {name: sha256(ROOT / name) for name in (
                    "src/dataset_b/mitigation/rq4b_training.py", "src/dataset_b/mitigation/rq4b_inputs.py",
                    "src/dataset_b/mitigation/rq4b_evaluation.py", "src/dataset_b/models/neural/realmlp/estimator.py",
                    "src/dataset_b/models/neural/realmlp/preprocessing.py", "config/dataset_b/realmlp_v1.yaml")}}

    def historical(self, seed: int, config: Mapping) -> dict:
        manifest_path = ROOT / config["reference_manifest"]
        require(sha256(manifest_path) == config["reference_manifest_sha256"], "Reference manifest changed.")
        ref = next(arm for arm in read_json(manifest_path)["evaluation_arms"]
                   if arm["seed"] == seed and arm["adversary_lambda"] == 0)
        path = ROOT / "artifacts/dataset_b/rq4/rq4b" / ref["run_id"] / "stages" / ref["fit_key"] / "diagnostics.json"
        return read_json(path)["refit_history"]

    def population(self, partition: str):
        import numpy as np
        import pandas as pd
        stage = {"train": "full_train_refit", "validation": "validation_calibration_selection", "test": "test_evaluation"}[partition]
        access = self.ri.StageAccess(stage, self.contract)
        gated = self.ri.load_partition(access, partition)
        ids, y = gated.record_ids.astype(str).tolist(), gated.y.to_numpy(dtype=np.int64)
        raw = self.ri.read_derived_race_rows(access, ids, permitted_ids=self.ri.stage_population_ids(access),
                                           reconciliation_fields=("lei", "applicant_sex"))
        reconciliation = pd.DataFrame({key: gated.audit[key].astype(str).to_numpy() for key in ("lei", "applicant_sex")})
        race = self.ri.join_derived_race(ids, y, raw, self.contract, reconciliation=reconciliation)
        return gated, ids, y, race, access

    def fit(self, seed: int, progress: FixedBudgetProgress):
        context = self.rt.open_production_context()
        self.rt._require_production_spec(context, 0.0, seed)
        gated, ids, y, race, access = self.population("train")
        preprocessor = self.ri.make_fit_population_preprocessor(access)
        native = preprocessor.fit_transform(gated.X.reset_index(drop=True))
        inputs = self.ri.build_predictor_input(native, y, ids, race, self.contract)
        spec = self.rt.FitSpec(adversary_lambda=0.0, model_seed=seed, execution_epochs=EPOCHS,
                               schedule_horizon_epochs=HORIZON, module_kind="adversarial")
        result = self.rt._fit_engine(inputs, spec, evidence_kind=self.rt.EVIDENCE_OBSERVED,
                                     progress=progress, phase="refit")
        return result, preprocessor, native.iloc[:2048].reset_index(drop=True)

    def predict(self, estimator, preprocessor, partition: str):
        import numpy as np
        import yaml
        gated, ids, y, race, _ = self.population(partition)
        probabilities = estimator.predict_proba(preprocessor.transform(gated.X.reset_index(drop=True)))[:, 1]
        rq3 = yaml.safe_load((ROOT / "config/dataset_b/rq3_protocol.yaml").read_text(encoding="utf-8"))
        sex = self.ev.normalize_rq3_sex_codes(gated.audit["applicant_sex"].astype(str).tolist(), rq3)
        return self.ev.PartitionPredictions(tuple(ids), y.astype(np.int8), np.asarray(probabilities, dtype=np.float64), race, sex)


def validate_bootstrap_policy(config: Mapping, helper: Any, old_config: Mapping) -> dict:
    require(helper.REPLICATES == config["bootstrap_replicates"] == 1000
            and helper.UNCERTAINTY["within_seed"]["minimum_valid_replicates_per_quantity"] == 1000
            and old_config["uncertainty"]["within_seed"]["minimum_valid_replicates_per_quantity"] == 1000,
            "Bootstrap contract/helper validity mismatch.")
    require(config["minimum_valid_replicates_per_quantity"] == 1000 and helper.QUANTILE_METHOD == "linear",
            "Bootstrap validity drift.")
    return {"minimum_valid_replicates_per_quantity": 1000, "replicates": 1000,
            "invalid_policy": "CI_not_evaluable_no_top_up", "quantile_method": "linear"}


def evaluate_contrasts(seed: int, c17: Any, a17: Any, c_long: Any, thresholds: Mapping[str, float],
                       indices: Any, *, gates: ExecutionGates, helper=None) -> dict:
    gates.require_stage("evaluation")
    if helper is None:
        from src.dataset_b.mitigation import rq4b_evaluation as helper
    import yaml
    validate_bootstrap_policy(load_config(), helper, yaml.safe_load((ROOT / "config/dataset_b/rq4b_adversarial.yaml").read_text()))
    require(set(thresholds) == {"C17", "A17", "C_long"}, "Threshold arm roster differs.")
    require(all(math.isfinite(value) and .001 <= value <= .999
                and abs(value * 1000 - round(value * 1000)) < 1e-9 for value in thresholds.values()),
            "Threshold outside the locked grid.")
    helper.require_paired(c17, a17)
    helper.require_paired(c17, c_long)
    output = {}
    for name, control, comparator, c_name, a_name in (
        ("A17-C17", c17, a17, "C17", "A17"), ("C17-C_long", c_long, c17, "C_long", "C17")
    ):
        helper.require_paired(control, comparator)
        output[name] = {"control_arm_id": arm_id(seed, c_name), "comparator_arm_id": arm_id(seed, a_name),
                        "point_deltas": helper.matched_deltas(helper.core_quantities(control, thresholds[c_name]),
                                                              helper.core_quantities(comparator, thresholds[a_name])),
                        "bootstrap": helper.paired_bootstrap(control, comparator,
                            control_threshold=thresholds[c_name], adversarial_threshold=thresholds[a_name], indices=indices)}
    return output


def summarize_new_contrasts(per_seed: Mapping[int, Mapping]) -> dict:
    """Descriptive mean/sample SD only; never pool bootstrap replicates."""
    from statistics import mean, stdev
    require(set(per_seed) == set(SEEDS), "Exactly three seeds are required.")
    output = {}
    for contrast in ("A17-C17", "C17-C_long"):
        for quantity in ("delta_ba", "delta_eod3", "delta_eod6"):
            values = [per_seed[seed][contrast]["point_deltas"][quantity] for seed in SEEDS]
            output[contrast + ":" + quantity] = {"values": values, "ddof": 1,
                "mean": None if any(value is None for value in values) else mean(values),
                "sample_sd": None if any(value is None for value in values) else stdev(values),
                "pooled_bootstrap": False, "across_seed_significance_claim": False}
    return output


def _prediction_csv(predictions) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("record_id", "y_approved", "p_approved", "derived_race", "applicant_sex_group"))
    writer.writerows(zip(predictions.record_ids, predictions.y, predictions.p_approved, predictions.race, predictions.sex))
    return output.getvalue()


def _read_predictions(path: Path, helper=None):
    import numpy as np
    if helper is None:
        from src.dataset_b.mitigation import rq4b_evaluation as helper
    with canonical(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["record_id", "y_approved", "p_approved", "derived_race", "applicant_sex_group"],
                "Prediction schema differs.")
        rows = list(reader)
    require(all(row["y_approved"] in ("0", "1") for row in rows), "Invalid labels.")
    return helper.PartitionPredictions(tuple(row["record_id"] for row in rows),
        np.asarray([int(row["y_approved"]) for row in rows], dtype=np.int8),
        np.asarray([float(row["p_approved"]) for row in rows], dtype=np.float64),
        np.asarray([row["derived_race"] for row in rows]), np.asarray([row["applicant_sex_group"] for row in rows]))


def open_c17_run(guard: OutputGuard, gates: ExecutionGates) -> dict:
    provenance = read_json(guard.path("provenance.json"))
    require(provenance["module_sha256"] == gates.module_sha256
            and provenance["config_sha256"] == gates.config_sha256, "Run code/config differs; no cross-code continuation.")
    verify_baseline(provenance["protected_inventory"])
    return provenance


def predict_c17(guard: OutputGuard, seed: int, partition: str, *, gates: ExecutionGates, backend=None) -> None:
    """New C17 inference only; completed models are read, never resumed for training."""
    require(partition in ("validation", "test"), "Only Validation/Test prediction is supported.")
    gates.require_stage(partition + "_inference")
    require_runtime_directories()
    provenance = open_c17_run(guard, gates)
    prefix = f"units/{arm_id(seed)}"
    require(read_json(guard.path(prefix + "/fit_ledger.json"))["status"] == "completed", "Fit is not completed.")
    if partition == "test":
        validation = read_json(guard.path(prefix + "/validation_receipt.json"))
        threshold_file = guard.path(prefix + "/threshold.json")
        require(sha256(threshold_file) == validation["threshold_sha256"], "Validation threshold changed.")
        require(sha256(guard.path(prefix + "/validation_predictions.csv")) == validation["predictions_sha256"],
                "Validation scores changed.")
    with WriteFence(guard.root, SCRATCH):
        # The claim remains on failure and prevents automatic reinference.
        guard.write_json(prefix + f"/{partition}_started.json", {"seed": seed, "partition": partition})
        try:
            import joblib
            from src.dataset_b.mitigation import rq4b_evaluation as ev
            reload = read_json(guard.path(prefix + "/reload.json"))
            model = guard.path(prefix + "/estimator.joblib")
            preprocessor_path = guard.path(prefix + "/preprocessor.joblib")
            require(sha256(model) == reload["sha256"] and sha256(preprocessor_path) == reload["preprocessor_sha256"], "C17 model digest differs.")
            require(guard.root != canonical(RUN_ROOT) or backend is None, "Mock backend forbidden in production.")
            backend = CoreBackend() if backend is None else backend
            predictions = backend.predict(joblib.load(model), joblib.load(preprocessor_path), partition)
            require(guard.root != canonical(RUN_ROOT) or len(predictions.record_ids) == load_config()["split_rows"][partition],
                    "Prediction row count differs.")
            filename = prefix + f"/{partition}_predictions.csv"
            with guard.path(filename, new=True).open("x", encoding="utf-8", newline="") as handle:
                handle.write(_prediction_csv(predictions))
            receipt = {"arm_id": arm_id(seed), "partition": partition, "rows": len(predictions.record_ids),
                       "predictions_sha256": sha256(guard.path(filename)), "model_sha256": reload["sha256"]}
            if partition == "validation":
                threshold = ev.calibrate_threshold(predictions.y, predictions.p_approved)
                guard.write_json(prefix + "/threshold.json", {"threshold": threshold.threshold,
                    "balanced_accuracy": threshold.balanced_accuracy, "candidate_count": threshold.candidate_count,
                    "tied_thresholds": list(threshold.tied_thresholds), "source_partition": "validation",
                    "source_predictions_sha256": receipt["predictions_sha256"]})
                receipt["threshold_sha256"] = sha256(guard.path(prefix + "/threshold.json"))
            else:
                receipt["threshold_sha256"] = validation["threshold_sha256"]
            guard.write_json(prefix + f"/{partition}_receipt.json", receipt)
        finally:
            verify_baseline(provenance["protected_inventory"])


def load_reference(seed: int, arm_name: str, config: Mapping):
    """Read saved scores/threshold only. No joblib or original model inference."""
    require(arm_name in ("A17", "C_long"), "Not an accepted reference arm.")
    arm_id(seed, arm_name)
    manifest_path = ROOT / config["reference_manifest"]
    require(sha256(manifest_path) == config["reference_manifest_sha256"], "Reference manifest changed.")
    manifest = read_json(manifest_path)
    lam = .01 if arm_name == "A17" else 0
    ref = next(arm for arm in manifest["evaluation_arms"] if arm["seed"] == seed and arm["adversary_lambda"] == lam)
    source = ROOT / "artifacts/dataset_b/rq4/rq4b" / ref["run_id"] / "stages"
    validation_receipt = source / ref["validation_key"] / "receipt.json"
    require(sha256(validation_receipt) == ref["validation_receipt_sha256"], "Reference Validation receipt changed.")
    threshold_path = validation_receipt.parent / "threshold.json"
    require(sha256(threshold_path) == read_json(validation_receipt)["outputs"]["threshold.json"], "Reference threshold changed.")
    threshold = read_json(threshold_path)["threshold"]
    require(threshold == ref["threshold"], "Reference manifest/threshold mismatch.")
    fit_receipt = source / ref["fit_key"] / "receipt.json"
    require(sha256(fit_receipt) == ref["fit_receipt_sha256"], "Reference fit receipt changed.")
    selection_path = fit_receipt.parent / "selection_record.json"
    require(sha256(selection_path) == read_json(fit_receipt)["outputs"]["selection_record.json"], "Reference selection changed.")
    expected_epoch = 17 if arm_name == "A17" else {42: 120, 43: 118, 44: 120}[seed]
    require(read_json(selection_path)["selection"]["selected_epoch"] == expected_epoch, "Reference epoch differs.")
    lam_key = "l0p01" if arm_name == "A17" else "l0"
    stage = manifest_path.parent / "stages/test_evaluation"
    name = f"test_predictions_{lam_key}_s{seed}.csv"
    receipt = read_json(stage / "receipt.json")
    require(sha256(stage / "receipt.json") == "2f7c1ae8b31d8df82b9342182cb40fb9b3c18aa783a31ff53eeeaf54e64632c4",
            "Accepted Test receipt changed.")
    require(sha256(stage / name) == receipt["outputs"][name], "Reference scores changed.")
    return _read_predictions(stage / name), threshold


def evaluate_saved(guard: OutputGuard, seed: int, *, gates: ExecutionGates) -> None:
    gates.require_stage("evaluation")
    require_runtime_directories()
    provenance = open_c17_run(guard, gates)
    config = load_config()
    prefix = f"units/{arm_id(seed)}"
    with WriteFence(guard.root, SCRATCH):
        guard.write_json(prefix + "/evaluation_started.json", {"seed": seed, "contrasts": config["contrasts"]})
        try:
            import numpy as np
            c17_path = guard.path(prefix + "/test_predictions.csv")
            receipt = read_json(guard.path(prefix + "/test_receipt.json"))
            require(sha256(c17_path) == receipt["predictions_sha256"], "C17 scores changed.")
            c17 = _read_predictions(c17_path)
            a17, a_threshold = load_reference(seed, "A17", config)
            c_long, c_threshold = load_reference(seed, "C_long", config)
            threshold = read_json(guard.path(prefix + "/threshold.json"))
            validation_receipt = read_json(guard.path(prefix + "/validation_receipt.json"))
            require(threshold["source_predictions_sha256"] == validation_receipt["predictions_sha256"]
                    and sha256(guard.path(prefix + "/threshold.json")) == validation_receipt["threshold_sha256"] == receipt["threshold_sha256"],
                    "C17 threshold lineage differs.")
            bootstrap = ROOT / "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"
            manifest = read_json(bootstrap.parent / "rq1_test_bootstrap_manifest.json")
            ids_digest = hashlib.sha256("".join(value + "\n" for value in c17.record_ids).encode()).hexdigest()
            require(sha256(bootstrap) == config["bootstrap_sha256"] == manifest["indices_sha256"]
                    and manifest["test_rows"] == len(c17.record_ids) and manifest["test_record_id_sha256"] == ids_digest,
                    "Locked bootstrap/ordered IDs differ.")
            with np.load(bootstrap, allow_pickle=False) as payload:
                require(payload.files == ["indices"], "Bootstrap array roster differs.")
                indices = payload["indices"]
            output = evaluate_contrasts(seed, c17, a17, c_long,
                {"C17": threshold["threshold"], "A17": a_threshold, "C_long": c_threshold}, indices, gates=gates)
            guard.write_json(prefix + "/new_contrasts.json", output)
        finally:
            verify_baseline(provenance["protected_inventory"])


def fit_controls(guard: OutputGuard, inventory: Mapping[str, str], *, gates: ExecutionGates, backend=None) -> None:
    """Exactly three fresh units. No retry, CV, prediction or evaluation dispatch."""
    gates.require_stage("training")
    config = load_config()
    verify_baseline(inventory)
    require(guard.root != canonical(RUN_ROOT) or backend is None, "Mock backend forbidden in production.")
    require(not guard.root.exists(), "Existing run; no resume.")
    require_runtime_directories()
    guard.create()
    with WriteFence(guard.root, SCRATCH):
        guard.write_json("provenance.json", {"config": config, "module_sha256": sha256(Path(__file__)),
            "config_sha256": sha256(CONFIG), "protected_inventory": dict(inventory),
            "protected_inventory_digest": BASELINE_DIGEST, "execution_status": "started",
            "derived_evidence": "NEW", "original_results": "read_only_reference",
            "backend_lineage": "recorded_per_unit_before_fit"})
        try:
            backend = CoreBackend() if backend is None else backend
            guard.write_json("backend_provenance.json", backend.provenance())
            for seed in SEEDS:
                verify_baseline(inventory)
                progress = FixedBudgetProgress(guard, seed)
                progress.start()
                try:
                    result, preprocessor, sample = backend.fit(seed, progress)
                    verify_baseline(inventory)
                    validate_fit(result, seed)
                    prefix = f"units/{arm_id(seed)}"
                    guard.write_json(prefix + "/history.json", result.history)
                    guard.write_json(prefix + "/diagnostics.json", result.log)
                    guard.write_json(prefix + "/input_digest.json", result.digest)
                    replay = replay_diagnostic(result.history, backend.historical(seed, config))
                    guard.write_json(prefix + "/replay_diagnostic.json", replay)
                    # Historical CE is a diagnostic, not acceptance of this fresh control.
                    # replay_diagnostic still enforces schedule/position/batch invariants.
                    reload = serialize_reload(result.estimator, sample, guard, prefix + "/estimator.joblib", training=backend.rt)
                    import joblib
                    with guard.path(prefix + "/preprocessor.joblib", new=True).open("xb") as handle:
                        joblib.dump(preprocessor, handle)
                    reload["preprocessor_sha256"] = sha256(guard.path(prefix + "/preprocessor.joblib"))
                    guard.write_json(prefix + "/reload.json", reload)
                    guard.write_json(prefix + "/acceptance.json", {
                        "policy": config["acceptance_policy"], "status": "PASS",
                        "fixed_budget_and_recipe_checks": "PASS", "checkpoint_reload": "PASS",
                        "replay_status": replay["status"], "ce_replay_required": False,
                        "historical_weight_equivalence_proven": False})
                    progress.finish()
                except BaseException as exc:
                    try:
                        progress.fail(exc)
                    except BaseException as secondary:
                        exc.add_note(f"Failure receipt also failed: {secondary}")
                    raise
            guard.write_json("fit_complete.json", {"cv_fold_fits": 0, "refit_fits": 3, "seeds": list(SEEDS)})
        finally:
            verify_baseline(inventory)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(dry_run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
