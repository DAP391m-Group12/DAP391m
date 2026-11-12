"""Task 02: mocks/synthetic arrays only. Never invoke a fit or real bootstrap.

Run with -B, plugin autoload/cacheprovider disabled and basetemp under the
approved C17 scratch root. No original test fixtures are imported.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.dataset_b.mitigation import rq4b_c17_independent as c17
from src.dataset_b.mitigation import rq4b_evaluation as ev


@pytest.fixture
def guard(tmp_path):
    assert c17.within(tmp_path.resolve(), c17.SCRATCH.resolve())
    return c17.OutputGuard(tmp_path / "run")


@pytest.fixture
def gates():
    return c17.ExecutionGates(frozenset({"training", "validation_inference", "test_inference", "evaluation"}),
                             "PASS", c17.BASELINE_DIGEST, c17.sha256(Path(c17.__file__)), c17.sha256(c17.CONFIG))


@pytest.fixture
def baseline():
    return c17.read_json(c17.SCRATCH / "gate_a_baseline.json")


def fake_history(seed=42):
    return {"spec": {"adversary_lambda": 0.0, "model_seed": seed, "adversary_seed": seed,
                     "execution_epochs": 17, "schedule_horizon_epochs": 256, "module_kind": "adversarial"},
        "completed_epochs": 17, "fold_validation_ce": [], "budget": {
        "trainer_max_epochs": 17, "module_schedule_horizon_epochs": 256, "batch_size": 256,
        "batches_per_epoch": 801, "iterated_samples_per_epoch": 205056, "train_rows": 205310,
        "drop_last": True, "shuffle": True, "predict_batch_size": 1024},
        "epochs": [{"epoch": epoch,
            "first_batch_schedule": {"epoch": epoch, "batch_idx": 0, "fit_progress": (epoch-1)/256,
                                     "sched": {"lr": {"": 0.1}, "wd": {"": 1.0}, "p_drop": {"": 1.0}}},
            "last_batch_schedule": {"epoch": epoch, "batch_idx": 800},
            "losses": {"batches": 801, "adversary_updates": 0, "adversary_bypassed_lambda_zero": 801,
                       "mean_prediction_loss": .4}} for epoch in range(1, 18)]}


class FakeModule:
    def __init__(self):
        self.model = NS(training=False)
    def eval(self):
        self.model.training = False
        return self


class FakeEstimator:
    def __init__(self):
        self.backend_ = NS(alg_interface_=NS(device="cuda:0", model=FakeModule()))
    def predict_proba(self, sample):
        return np.tile([.3, .7], (len(sample), 1))


class FakePreprocessor:
    def transform(self, frame):
        return frame


class FakeTraining:
    @staticmethod
    def strip_batch_records(estimator):
        return 0
    @staticmethod
    def training_metadata_present(estimator):
        return False


class FakeBackend:
    rt = FakeTraining
    def __init__(self, failure=None):
        self.seeds = []
        self.failure = failure
    def provenance(self):
        return {"evidence_kind": "mock_synthetic", "no_real_backend_imported": True}
    def historical(self, seed, config):
        return fake_history(seed)
    def fit(self, seed, progress):
        self.seeds.append(seed)
        for epoch in range(1, 18):
            progress.heartbeat(phase="refit", fold=None, epoch=epoch, batch=1, gpu_memory_bytes=None)
            if seed == self.failure and epoch == 2:
                raise FloatingPointError("synthetic interruption")
            progress.emit("epoch_completed", phase="refit", epoch=epoch, batch=801,
                          prediction_loss=.4, optimizer_progress_claimed=True)
        result = NS(history=fake_history(seed), log={"predictor_updates": 13617, "adversary_updates": 0,
                    "adversary_rng_isolation": {"cpu_rng_state_unchanged": True, "cuda_rng_state_unchanged": True}},
                    digest={"record_ids": "synthetic"}, estimator=FakeEstimator())
        return result, FakePreprocessor(), np.zeros((4, 19))


def test_plan_exact_three_fits_no_artifact_initialization(monkeypatch):
    # Simulate absence without changing the locked config or touching the run.
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == c17.RUN_ROOT else exists(path))
    plan = c17.dry_run()
    assert len(plan["units"]) == plan["full_train_fits"] == 3
    assert [row["seed"] for row in plan["units"]] == [42, 43, 44]
    assert all((row["lambda"], row["execution_epochs"], row["schedule_horizon_epochs"]) == (0, 17, 256) for row in plan["units"])
    assert plan["cv_fits"] == plan["original_fits"] == plan["original_inference"] == 0
    assert not plan["execution_authorized"] and not c17.RUN_ROOT.exists()
    assert len({c17.arm_id(seed, arm) for seed in c17.SEEDS for arm in ("C17", "A17", "C_long")}) == 9


@pytest.mark.parametrize("seed", [True, 41, 45, "42", 42.0])
def test_wrong_seed_refused(seed):
    with pytest.raises(c17.C17Error):
        c17.arm_id(seed)


def test_config_is_closed_and_recipe_is_not_written():
    config = c17.load_config()
    assert config["epoch_policy"] == c17.POLICY
    assert config["quantities"] == ["ba", "eod3", "eod6"]
    with pytest.raises(c17.C17Error):
        c17.load_config(c17.ROOT / config["recipe"])


def test_protected_baseline_and_private_engine_source(baseline):
    c17.verify_baseline(baseline)
    assert len(baseline) == 137
    tree = ast.parse((c17.ROOT / "src/dataset_b/mitigation/rq4b_training.py").read_text())
    engine = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_engine")
    assert "selection_record" not in {arg.arg for arg in engine.args.args}
    source = ast.get_source_segment((c17.ROOT / "src/dataset_b/mitigation/rq4b_training.py").read_text(), engine)
    assert "AdversarialRealMLPClassifier(" in source and "estimator.fit(" in source
    training = (c17.ROOT / "src/dataset_b/mitigation/rq4b_training.py").read_text()
    assert "if self.rq4b_lambda == 0.0:" in training and "TabNNModule.training_step(self, core, batch_idx)" in training
    assert "with torch.random.fork_rng(devices=[device_index])" in training


def test_baseline_forgery_refused(baseline):
    baseline["AGENTS.md"] = "0" * 64
    with pytest.raises(c17.C17Error, match="frozen"):
        c17.verify_baseline(baseline)


def test_full_orchestration_three_mock_units(guard, gates, baseline):
    backend = FakeBackend()
    c17.fit_controls(guard, baseline, gates=gates, backend=backend)
    assert backend.seeds == [42, 43, 44]
    assert c17.read_json(guard.path("fit_complete.json"))["refit_fits"] == 3
    for seed in c17.SEEDS:
        ledger = c17.read_json(guard.path(f"units/{c17.arm_id(seed)}/fit_ledger.json"))
        assert ledger["planned"] == ledger["started"] == ledger["completed"] == {"cv_fold_fits": 0, "refit_fits": 1}
        assert ledger["status"] == "completed" and ledger["completed_epochs"] == 17
    with pytest.raises(c17.C17Error, match="Existing"):
        c17.fit_controls(guard, baseline, gates=gates, backend=backend)
    assert backend.seeds == [42, 43, 44]
    c17.verify_baseline(baseline)


def test_partial_failure_retained_no_retry(guard, gates, baseline):
    backend = FakeBackend(failure=43)
    with pytest.raises(FloatingPointError, match="synthetic"):
        c17.fit_controls(guard, baseline, gates=gates, backend=backend)
    assert backend.seeds == [42, 43] and not guard.path("fit_complete.json").exists()
    ledger = c17.read_json(guard.path(f"units/{c17.arm_id(43)}/fit_ledger.json"))
    assert ledger["status"] == "failed" and ledger["partial"]["refit_fits"] == 1
    assert ledger["completed"]["refit_fits"] == 0 and ledger["error"]["automatic_retry"] is False


def test_ce_mismatch_is_recorded_without_blocking_checkpoint(guard, gates, baseline):
    class DifferentCE(FakeBackend):
        def historical(self, seed, config):
            history = fake_history(seed)
            for item in history["epochs"]:
                item["losses"]["mean_prediction_loss"] = .5
            return history
    backend = DifferentCE()
    c17.fit_controls(guard, baseline, gates=gates, backend=backend)
    assert backend.seeds == [42, 43, 44]
    for seed in c17.SEEDS:
        prefix = f"units/{c17.arm_id(seed)}"
        assert c17.read_json(guard.path(prefix + "/replay_diagnostic.json"))["status"] == "BLOCK"
        acceptance = c17.read_json(guard.path(prefix + "/acceptance.json"))
        assert acceptance["status"] == "PASS" and acceptance["ce_replay_required"] is False
        assert guard.path(prefix + "/estimator.joblib").is_file()
        assert c17.read_json(guard.path(prefix + "/reload.json"))["probabilities_exactly_equal"]


def test_schedule_mismatch_still_blocks_fit(guard, gates, baseline):
    class DifferentSchedule(FakeBackend):
        def historical(self, seed, config):
            history = fake_history(seed)
            history["epochs"][0]["first_batch_schedule"]["fit_progress"] = .1
            return history
    backend = DifferentSchedule()
    with pytest.raises(c17.C17Error, match="schedule"):
        c17.fit_controls(guard, baseline, gates=gates, backend=backend)
    assert backend.seeds == [42]
    assert not guard.path("fit_complete.json").exists()


def test_no_gates_no_run(guard, baseline):
    gate = c17.ExecutionGates(frozenset(), "NOT_RUN", c17.BASELINE_DIGEST, "", "")
    with pytest.raises(c17.C17Error, match="authorized"):
        c17.fit_controls(guard, baseline, gates=gate, backend=FakeBackend())
    assert not guard.root.exists()


def test_gate_code_drift_refused(gates):
    wrong = c17.ExecutionGates(gates.approved_stages, "PASS", c17.BASELINE_DIGEST, "0"*64, gates.config_sha256)
    with pytest.raises(c17.C17Error, match="drift"):
        wrong.require_stage("training")


def test_progress_wrong_phase_duplicate_extra_fit(guard):
    guard.create()
    progress = c17.FixedBudgetProgress(guard, 42)
    progress.start()
    for args in ({"phase": "cv", "fold": 1, "epoch": 1, "batch": 1},
                 {"phase": "refit", "fold": None, "epoch": 18, "batch": 1},
                 {"phase": "refit", "fold": None, "epoch": 1, "batch": 802}):
        with pytest.raises(c17.C17Error):
            progress.heartbeat(**args, gpu_memory_bytes=None)
    with pytest.raises(c17.C17Error):
        progress.start()
    with pytest.raises(c17.C17Error):
        progress.finish()
    with pytest.raises(c17.C17Error):
        c17.FixedBudgetProgress(guard, 42)


@pytest.mark.parametrize("mutation", ["epochs", "batches", "adversary", "rng", "updates", "seed", "lambda", "nonfinite"])
def test_bad_fit_evidence_refused(mutation):
    result = NS(history=fake_history(), log={"predictor_updates": 13617, "adversary_updates": 0,
        "adversary_rng_isolation": {"cpu_rng_state_unchanged": True, "cuda_rng_state_unchanged": True}})
    if mutation == "epochs": result.history["completed_epochs"] = 16
    if mutation == "batches": result.history["epochs"][0]["losses"]["batches"] = 800
    if mutation == "adversary": result.log["adversary_updates"] = 1
    if mutation == "rng": result.log["adversary_rng_isolation"]["cuda_rng_state_unchanged"] = False
    if mutation == "updates": result.log["predictor_updates"] = 13616
    if mutation == "seed": result.history["spec"]["model_seed"] = 43
    if mutation == "lambda": result.history["spec"]["adversary_lambda"] = .01
    if mutation == "nonfinite": result.history["epochs"][0]["losses"]["mean_prediction_loss"] = float("nan")
    with pytest.raises(c17.C17Error): c17.validate_fit(result, 42)


def test_replay_alert_not_equivalence():
    actual, old = fake_history(), fake_history()
    assert c17.replay_diagnostic(actual, old)["status"] == "PASS"
    actual["epochs"][0]["losses"]["mean_prediction_loss"] += .0001
    report = c17.replay_diagnostic(actual, old)
    assert report["status"] == "BLOCK" and not report["weight_equivalence_proven"]
    actual["epochs"][0]["first_batch_schedule"]["fit_progress"] = .1
    with pytest.raises(c17.C17Error, match="schedule"):
        c17.replay_diagnostic(actual, old)


def test_core_bridge_keeps_recipe_rng_spec_and_train_only():
    calls = []
    frame = NS(reset_index=lambda **kwargs: NS())
    class Native:
        @property
        def iloc(self): return self
        def __getitem__(self, key): return self
        def reset_index(self, **kwargs): return self
    class Prep:
        def fit_transform(self, X): calls.append(("fit_preprocessing", X)); return Native()
    prep = Prep()
    def engine(inputs, spec, **kwargs):
        calls.append(("engine", inputs, spec, kwargs)); return "fake_result"
    rt = NS(open_production_context=lambda: "context", _require_production_spec=lambda *args: calls.append(("require", args)),
            FitSpec=lambda **kwargs: kwargs, _fit_engine=engine, EVIDENCE_OBSERVED="observed_fit")
    ri = NS(make_fit_population_preprocessor=lambda access: prep,
            build_predictor_input=lambda *args: ("native_inputs", args))
    backend = object.__new__(c17.CoreBackend)
    backend.rt, backend.ri, backend.contract = rt, ri, "locked_contract"
    backend.population = lambda partition: (calls.append(("population", partition)) or (NS(X=frame), ["a"], [1], ["White"], "train_access"))
    result, fitted, sample = backend.fit(42, "progress")
    assert result == "fake_result" and fitted is prep
    assert [call for call in calls if call[0] == "population"] == [("population", "train")]
    call = next(call for call in calls if call[0] == "engine")
    assert call[2] == {"adversary_lambda": 0.0, "model_seed": 42, "execution_epochs": 17,
                       "schedule_horizon_epochs": 256, "module_kind": "adversarial"}
    assert call[3] == {"evidence_kind": "observed_fit", "progress": "progress", "phase": "refit"}
    assert not any("selection" in str(call) for call in calls)


@pytest.mark.parametrize("relative", ["../outside.txt", "a/../../outside.txt", "C:/outside.txt", "/outside.txt", "", "foo. ", "NUL.txt"])
def test_traversal_absolute_denied(guard, relative):
    with pytest.raises(c17.C17Error): guard.path(relative)


def test_old_root_and_case_alias_denied():
    for old in (c17.ROOT / "artifacts/dataset_b/rq4/rq4b", c17.ROOT / "data", c17.ROOT):
        with pytest.raises(c17.C17Error): c17.OutputGuard(old)
        with pytest.raises(c17.C17Error): c17.OutputGuard(Path(str(old).swapcase()))


def test_no_overwrite_and_write_fence(guard):
    guard.create()
    with c17.WriteFence(guard.root, c17.SCRATCH):
        guard.write_json("new.json", {"x": 1})
        with pytest.raises(c17.C17Error): guard.write_json("new.json", {"x": 2})
        with pytest.raises(c17.C17Error):
            (c17.ROOT / "c17_unapproved_write.txt").write_text("forbidden")
    assert not (c17.ROOT / "c17_unapproved_write.txt").exists()
    assert c17.read_json(guard.path("new.json")) == {"x": 1}


def test_reparse_path_refused_synthetic_stat(guard, monkeypatch):
    path = guard.root
    real = Path.lstat
    def fake_lstat(self):
        if self == path: return NS(st_mode=0o040755, st_file_attributes=1024)
        return real(self)
    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(c17.C17Error, match="Reparse"):
        c17.canonical(path)


def test_reload_exact_and_metadata_checks(guard):
    guard.create()
    report = c17.serialize_reload(FakeEstimator(), np.zeros((3, 19)), guard, "model.joblib", training=FakeTraining)
    assert report["probabilities_exactly_equal"] and not report["training_metadata_pickled"]
    with pytest.raises(c17.C17Error, match="overwrite"):
        c17.serialize_reload(FakeEstimator(), np.zeros((3, 19)), guard, "model.joblib", training=FakeTraining)
    bad_training = NS(strip_batch_records=lambda est: 0, training_metadata_present=lambda est: True)
    with pytest.raises(c17.C17Error, match="metadata"):
        c17.serialize_reload(FakeEstimator(), np.zeros((3, 19)), guard, "bad.joblib", training=bad_training)
    assert not guard.path("bad.joblib").exists()


class ModeRestoringEstimator(FakeEstimator):
    """Synthetic predictor using Lightning's actual per-module mode restoration."""
    def __init__(self):
        import torch
        super().__init__()
        module = torch.nn.Module()
        module.model = torch.nn.Sequential(torch.nn.Linear(19, 2),
                                          torch.nn.BatchNorm1d(2), torch.nn.Dropout(.5))
        self.backend_.alg_interface_.model = module

    def predict_proba(self, sample):
        import torch
        from pytorch_lightning.utilities.model_helpers import _ModuleMode
        module = self.backend_.alg_interface_.model
        mode = _ModuleMode()
        mode.capture(module)
        module.model.eval()  # TabNNModule.on_predict_model_eval / predict_step
        try:
            with torch.no_grad():
                return module.model(torch.as_tensor(sample, dtype=torch.float32)).softmax(-1).numpy()
        finally:
            mode.restore(module)


def test_training_mode_round_trip_regression(guard):
    import joblib
    import torch
    estimator = ModeRestoringEstimator()
    module = estimator.backend_.alg_interface_.model
    assert all(child.training for child in module.modules())
    sample = np.zeros((3, 19))
    reference = estimator.predict_proba(sample)
    assert all(child.training for child in module.modules())
    state = {name: value.clone() for name, value in module.state_dict().items()}
    devices = {name: value.device for name, value in module.state_dict().items()}
    guard.create()
    report = c17.serialize_reload(estimator, sample, guard, "mode.joblib", training=FakeTraining)
    restored = joblib.load(guard.path("mode.joblib"))
    assert report["probabilities_exactly_equal"]
    assert np.array_equal(reference, restored.predict_proba(sample))
    for current in (module, restored.backend_.alg_interface_.model):
        assert all(not child.training for child in current.modules())
        assert all(torch.equal(state[name], value) for name, value in current.state_dict().items())
        assert {name: value.device for name, value in current.state_dict().items()} == devices
    assert restored.backend_.alg_interface_.device == "cuda:0"


@pytest.mark.parametrize("invalid", ["mode", "device", "metadata", "digest"])
def test_mode_fix_preserves_reload_integrity_checks(guard, invalid):
    import joblib
    class CorruptCodec:
        dump = staticmethod(joblib.dump)
        @staticmethod
        def load(handle):
            restored = joblib.load(handle)
            if invalid == "mode":
                restored.backend_.alg_interface_.model.model.training = True
            elif invalid == "device":
                restored.backend_.alg_interface_.device = "cpu"
            elif invalid == "metadata":
                restored.invalid_metadata = True
            else:
                with Path(handle.name).open("ab") as output:
                    output.write(b"tampered")
            return restored
    training = NS(strip_batch_records=lambda est: 0,
                  training_metadata_present=lambda est: getattr(est, "invalid_metadata", False))
    guard.create()
    expected = {"mode": "mode/device", "device": "mode/device",
                "metadata": "metadata after", "digest": "probabilities changed"}
    with pytest.raises(c17.C17Error, match=expected[invalid]):
        c17.serialize_reload(FakeEstimator(), np.zeros((3, 19)), guard, "invalid.joblib",
                             training=training, codec=CorruptCodec)


def test_threshold_grid_ties_using_synthetic_arrays():
    assert len(ev.threshold_candidates()) == 999
    assert ev.resolve_threshold_tie([.499, .501]) == .499
    selection = ev.calibrate_threshold(np.asarray([0, 1]), np.asarray([0., 1.]))
    assert selection.threshold == .5 and selection.candidate_count == 999


def partition(ids=("a", "b")):
    return ev.PartitionPredictions(tuple(ids), np.asarray([0, 1]), np.asarray([.2, .8]),
                                   np.asarray(["White", "White"]), np.asarray(["Male", "Female"]))


def test_contrasts_pairing_sign_and_indices_mock_only(gates):
    calls = []
    def fake_bootstrap(control, adversarial, **kwargs):
        calls.append((control, adversarial, kwargs)); return {"mock": True}
    def core(pred, threshold): return {"ba": threshold}
    helper = NS(REPLICATES=1000, QUANTILE_METHOD="linear", UNCERTAINTY=ev.UNCERTAINTY,
        require_paired=ev.require_paired, core_quantities=core,
        matched_deltas=lambda c, a: {"delta_ba": a["ba"]-c["ba"]}, paired_bootstrap=fake_bootstrap)
    c17_pred, a17, long = partition(), partition(), partition()
    indices = object()  # No real bootstrap generation or computation.
    result = c17.evaluate_contrasts(42, c17_pred, a17, long, {"C17": .5, "A17": .6, "C_long": .7}, indices,
                                    gates=gates, helper=helper)
    assert set(result) == {"A17-C17", "C17-C_long"} and len(calls) == 2
    assert calls[0][:2] == (c17_pred, a17) and calls[1][:2] == (long, c17_pred)
    assert all(call[2]["indices"] is indices for call in calls)
    assert result["A17-C17"]["point_deltas"]["delta_ba"] > 0
    assert result["C17-C_long"]["point_deltas"]["delta_ba"] < 0
    with pytest.raises(ev.RQ4bEvaluationError):
        c17.evaluate_contrasts(42, c17_pred, partition(("b", "a")), long,
            {"C17": .5, "A17": .6, "C_long": .7}, indices, gates=gates, helper=helper)


def test_bootstrap_minimum_is_1000_per_quantity():
    import yaml
    old = yaml.safe_load((c17.ROOT / "config/dataset_b/rq4b_adversarial.yaml").read_text())
    evidence = c17.validate_bootstrap_policy(c17.load_config(), ev, old)
    assert evidence["minimum_valid_replicates_per_quantity"] == 1000
    old["uncertainty"]["within_seed"]["minimum_valid_replicates_per_quantity"] = 999
    with pytest.raises(c17.C17Error): c17.validate_bootstrap_policy(c17.load_config(), ev, old)


def test_cli_has_no_execution_switch():
    with pytest.raises(SystemExit): c17.main(["--train"])
    with pytest.raises(SystemExit): c17.main([])


def test_write_fence_covers_python_worker_threads(guard):
    import threading
    errors = []
    def worker():
        try:
            (c17.ROOT / "c17_forbidden_thread_write.txt").write_text("forbidden")
        except c17.C17Error as exc:
            errors.append(str(exc))
    guard.create()
    with c17.WriteFence(guard.root, c17.SCRATCH):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert len(errors) == 1 and not (c17.ROOT / "c17_forbidden_thread_write.txt").exists()


def test_uncontained_runtime_temp_refused(guard, gates, baseline, monkeypatch):
    monkeypatch.setenv("TMP", str(c17.ROOT))
    with pytest.raises(c17.C17Error, match="Uncontained"):
        c17.fit_controls(guard, baseline, gates=gates, backend=FakeBackend())
    assert not guard.root.exists()


def test_new_prediction_interfaces_use_only_saved_c17(guard, gates, baseline):
    class PredictBackend(FakeBackend):
        def __init__(self): super().__init__(); self.partitions = []
        def historical(self, seed, config):
            history = fake_history(seed)
            history["epochs"][0]["losses"]["mean_prediction_loss"] = .5
            return history
        def predict(self, estimator, preprocessor, which):
            assert isinstance(estimator, FakeEstimator) and isinstance(preprocessor, FakePreprocessor)
            self.partitions.append(which)
            return partition()
    backend = PredictBackend()
    c17.fit_controls(guard, baseline, gates=gates, backend=backend)
    with pytest.raises(FileNotFoundError):
        c17.predict_c17(guard, 42, "test", gates=gates, backend=backend)
    assert backend.partitions == []
    c17.predict_c17(guard, 42, "validation", gates=gates, backend=backend)
    c17.predict_c17(guard, 42, "test", gates=gates, backend=backend)
    assert backend.partitions == ["validation", "test"]
    with pytest.raises(c17.C17Error, match="overwrite"):
        c17.predict_c17(guard, 42, "validation", gates=gates, backend=backend)
    assert backend.partitions == ["validation", "test"]
    prefix = f"units/{c17.arm_id(42)}"
    receipt = c17.read_json(guard.path(prefix + "/test_receipt.json"))
    assert receipt["threshold_sha256"] == c17.sha256(guard.path(prefix + "/threshold.json"))


def test_serialization_changed_probabilities_refused(guard):
    import joblib
    class BadCodec:
        dump = staticmethod(joblib.dump)
        @staticmethod
        def load(handle):
            estimator = joblib.load(handle)
            estimator.predict_proba = lambda X: np.tile([.4, .6], (len(X), 1))
            return estimator
    guard.create()
    with pytest.raises(c17.C17Error, match="probabilities changed"):
        c17.serialize_reload(FakeEstimator(), np.zeros((3, 19)), guard, "bad_reload.joblib", training=FakeTraining, codec=BadCodec)
    assert guard.path("bad_reload.joblib").exists()  # Preserve failed output; no silent cleanup.


def test_three_seed_descriptive_summary_no_pooling():
    data = {seed: {name: {"point_deltas": {quantity: value for quantity in ("delta_ba", "delta_eod3", "delta_eod6")}}
                    for name in ("A17-C17", "C17-C_long")} for seed, value in zip(c17.SEEDS, (.1, .2, .3))}
    summary = c17.summarize_new_contrasts(data)
    assert summary["A17-C17:delta_ba"]["sample_sd"] == pytest.approx(.1)
    assert not summary["A17-C17:delta_ba"]["pooled_bootstrap"]
    data[43]["A17-C17"]["point_deltas"]["delta_ba"] = None
    assert c17.summarize_new_contrasts(data)["A17-C17:delta_ba"]["mean"] is None
    with pytest.raises(c17.C17Error): c17.summarize_new_contrasts({42: data[42]})


def test_no_old_runner_or_selection_entrypoints_in_new_module():
    tree = ast.parse(Path(c17.__file__).read_text())
    forbidden = {"run_full_train_refit", "run_cv_epoch_selection", "select_epoch", "_compute_test_tables", "create_run", "open_run", "fit_unit", "save_and_verify_reload"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden


def test_literal_os_null_device_open_only(guard):
    import os
    if os.name != "nt":
        pytest.skip("Windows null-device regression")
    guard.create()
    assert c17.is_os_null_device(os.devnull)
    assert c17.is_os_null_device(os.devnull.upper())
    assert c17.is_os_null_device(os.fsencode(os.devnull))
    for alias in ("NUL.txt", "nul ", "nul.", "./nul", str(guard.root / "nul"), r"\\.\NUL"):
        assert not c17.is_os_null_device(alias)
    with c17.WriteFence(guard.root, c17.SCRATCH):
        fd = os.open(os.devnull, os.O_RDWR)
        try:
            os.write(fd, b"null-device-only")
        finally:
            os.close(fd)
        for name in ("NUL", "nul", "NUL.txt", "nul "):
            with pytest.raises(c17.C17Error):
                guard.path(name, new=True)
        with pytest.raises(c17.C17Error):
            guard.path("../outside.txt", new=True)
        with pytest.raises(c17.C17Error):
            (c17.ROOT / "c17_unapproved_null_fix.txt").write_text("forbidden")
    assert not (c17.ROOT / "c17_unapproved_null_fix.txt").exists()


def test_cold_import_with_fence_before_pandas(guard):
    """A new interpreter imports pandas only after installing the audit hook.

    Explicit DEVNULL use also covers hosts where platform.machine() needs no
    subprocess fallback. No model/CUDA calls or fitting occur in this child.
    """
    import os
    import subprocess
    import sys
    script = r'''
import json, os, subprocess, sys
from pathlib import Path
from src.dataset_b.mitigation import rq4b_c17_independent as m
assert 'pandas' not in sys.modules
assert sys.dont_write_bytecode
root=Path(sys.argv[1])
def collector(event,args):
    # Match the failed Task 03 collector, sharing the narrow device exception.
    if event!='open' or isinstance(args[0],int) or m.is_os_null_device(args[0]):
        return
    path,mode,flags=args
    if (mode and any(ch in mode for ch in 'wax+')) or flags & (
        os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
        resolved=m.canonical(Path(os.fsdecode(path)).absolute())
        m.require(m.within(resolved,root),'Collector write outside scratch')
sys.addaudithook(collector)
with m.WriteFence(root):
    # Match the raw open used by subprocess._get_devnull on Windows.
    fd=os.open(os.devnull,os.O_RDWR)
    os.close(fd)
    import pandas, numpy
    child=subprocess.run([sys.executable,'-B','-c','print("devnull-ok")'],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        check=True,text=True,timeout=30)
    assert child.stdout.strip()=='devnull-ok'
    for name in ('NUL','NUL.txt','../outside.txt'):
        try:
            m.OutputGuard(root).path(name,new=True)
        except m.C17Error:
            pass
        else:
            raise AssertionError('Output guard weakened: '+name)
print(json.dumps({'pandas':pandas.__version__,'numpy':numpy.__version__,
    'cold_import_under_fence':True,'os_null_only':True,'training':False,'inference':False}))
'''
    guard.create()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-B", "-c", script, str(guard.root)],
        cwd=c17.ROOT, env=environment, check=True, capture_output=True, text=True, timeout=60)
    evidence = json.loads(result.stdout)
    assert evidence["cold_import_under_fence"] and evidence["os_null_only"]
    assert not evidence["training"] and not evidence["inference"]
