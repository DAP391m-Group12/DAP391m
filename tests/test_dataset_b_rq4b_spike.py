"""Focused tests for the disposable RQ4b internal-extension spike.

GPU tests fail (never skip) when cuda:0 is unavailable: the spike forbids CPU
fallback. Every smoke fit uses synthetic data and at most two epochs. Files are
written only beneath pytest's ``tmp_path``.
"""

from __future__ import annotations

import os
import stat
import warnings

import numpy as np
import pytest
import torch

from src.dataset_b.mitigation import rq4b_spike as spike


warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
LAMBDA = 0.1


@pytest.fixture(scope="module")
def fixture_data() -> spike.SyntheticFixture:
    return spike.make_synthetic_fixture()


@pytest.fixture(scope="module")
def cuda0() -> None:
    if not torch.cuda.is_available():
        pytest.fail("RQ4b spike tests require cuda:0; CPU fallback is prohibited.")
    spike.require_cuda0()


@pytest.fixture(scope="module")
def lambda_zero_runs(cuda0, fixture_data):
    stock = spike.fit_stock_realmlp(fixture_data, seed=SEED)
    observed, observed_obs = spike.fit_spike_realmlp(
        fixture_data, seed=SEED, adversary_lambda=0.0, module_kind="stock_observed"
    )
    extension, extension_obs = spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=0.0)
    return stock, observed, observed_obs, extension, extension_obs


@pytest.fixture(scope="module")
def diagnostic_run(cuda0, fixture_data):
    return spike.fit_spike_realmlp(
        fixture_data, seed=SEED, adversary_lambda=LAMBDA, record_gradient_diagnostics=True
    )


# --- CPU-only contract, fixture and selector tests ---------------------------


def test_locked_v2_contract_matches_spike_constants() -> None:
    contract = spike.load_rq4b_spike_contract()
    assert contract["protocol_version"] == spike.RQ4B_PROTOCOL_VERSION
    assert contract["technical_spike"]["maximum_epochs_per_smoke_fit"] == 2


def test_contract_conflict_is_reported(tmp_path) -> None:
    text = spike.RQ4B_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "maximum_epochs_per_smoke_fit: 2", "maximum_epochs_per_smoke_fit: 3"
    )
    altered = tmp_path / "altered.yaml"
    altered.write_text(text, encoding="utf-8")
    with pytest.raises(spike.RQ4bContractError, match="maximum_epochs_per_smoke_fit"):
        spike.load_rq4b_spike_contract(altered)


def test_synthetic_fixture_matches_native_schema(fixture_data) -> None:
    info = spike.validate_fixture(fixture_data)
    assert info["columns"] == list(spike.TWO_CHANNEL_COLUMNS)
    assert set(info["group_code_counts"]) == {"-1", "0", "1", "2"}
    assert info["approved"] > 0 and info["denied"] > 0
    assert len(set(fixture_data.record_ids)) == len(fixture_data.X)
    assert spike.make_synthetic_fixture().record_ids.tolist() == fixture_data.record_ids.tolist()


def test_cuda_unavailable_is_rejected(monkeypatch, fixture_data) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(spike.RQ4bCudaUnavailableError):
        spike.require_cuda0()
    with pytest.raises(spike.RQ4bCudaUnavailableError):
        spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=LAMBDA)
    with pytest.raises(spike.RQ4bCudaUnavailableError):
        spike.fit_stock_realmlp(fixture_data, seed=SEED)


def test_non_cuda0_device_is_rejected(fixture_data) -> None:
    settings = spike.RQ4bSpikeSettings(
        adversary_lambda=LAMBDA,
        group_codes=fixture_data.group_codes,
        alignment_labels=fixture_data.y,
        adversary_seed=SEED,
    )
    params = {**spike.spike_estimator_parameters(SEED, 2), "device": "cpu"}
    estimator = spike.SpikeAdversarialRealMLPClassifier(**params).attach_rq4b(settings)
    with pytest.raises(spike.RQ4bCudaUnavailableError):
        estimator.fit(fixture_data.X, fixture_data.y)


def test_smoke_epoch_limit_is_enforced(fixture_data) -> None:
    with pytest.raises(ValueError):
        spike.RQ4bSpikeSettings(
            adversary_lambda=0.0,
            group_codes=fixture_data.group_codes,
            alignment_labels=fixture_data.y,
            adversary_seed=SEED,
            max_executed_epochs=3,
        )
    with pytest.raises(spike.RQ4bSpikeError):
        spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=0.0, n_epochs=3)
    with pytest.raises(spike.RQ4bSpikeError):
        spike.fit_stock_realmlp(fixture_data, seed=SEED, n_epochs=3)


def _curves(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.4, 0.6, size=(5, 256))


def test_epoch_selector_equal_fold_weighting_and_earliest_tie() -> None:
    curves = np.ones((5, 256))
    curves[0, 3] = 0.0  # mean 0.8 at epoch 4
    curves[1:, 5] = 0.76  # mean 0.808 at epoch 6; median would prefer epoch 6
    assert spike.select_epoch(curves)["selected_epoch"] == 4
    tie = np.ones((5, 256))
    tie[:, [9, 19, 200]] = 0.25
    result = spike.select_epoch(tie)
    assert result["selected_epoch"] == 10
    assert result["tied_epochs"] == [10, 20, 201]
    assert result["epoch_cap_hit"] is False
    assert result["finite_values"] == 1280


def test_epoch_selector_cap_hit_and_mapping_input() -> None:
    curves = np.tile(np.linspace(1.0, 0.5, 256), (5, 1))
    result = spike.select_epoch({fold: curves[fold - 1].tolist() for fold in range(1, 6)})
    assert result["selected_epoch"] == 256
    assert result["epoch_cap_hit"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c[:4],
        lambda c: c[:, :255],
        lambda c: np.where(np.eye(5, 256, dtype=bool), np.nan, c),
        lambda c: np.where(np.eye(5, 256, dtype=bool), np.inf, c),
    ],
    ids=["missing_fold", "missing_epoch", "nan", "inf"],
)
def test_epoch_selector_rejects_missing_or_nonfinite(mutate) -> None:
    with pytest.raises(spike.RQ4bEpochSelectionError):
        spike.select_epoch(mutate(_curves(np.random.default_rng(0))))


def test_epoch_selector_rejects_missing_mapping_entries() -> None:
    curves = {fold: [0.5] * 256 for fold in range(1, 6)}
    curves[3][7] = None
    with pytest.raises(spike.RQ4bEpochSelectionError):
        spike.select_epoch(curves)
    del curves[5]
    with pytest.raises(spike.RQ4bEpochSelectionError):
        spike.select_epoch(curves)


def test_stable_binary_ce_matches_probability_form() -> None:
    rng = np.random.default_rng(1)
    margin = rng.normal(scale=3.0, size=500)
    y = rng.integers(0, 2, size=500)
    p = 1.0 / (1.0 + np.exp(-margin))
    naive = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    assert spike.stable_binary_ce_from_margin(margin, y) == pytest.approx(naive, rel=1e-12)
    assert np.isfinite(spike.stable_binary_ce_from_margin(np.array([800.0, -800.0]), np.array([0, 1])))


def test_failure_receipt_is_immutable(tmp_path) -> None:
    path = spike.write_failure_receipt(tmp_path, gate="gX", category="implementation_defect", message="m")
    assert not os.stat(path).st_mode & stat.S_IWRITE
    with pytest.raises(FileExistsError):
        spike.write_failure_receipt(tmp_path, gate="gY", category="environment_failure", message="m")
    with pytest.raises(ValueError):
        spike.write_failure_receipt(tmp_path / "x", gate="g", category="other", message="m")


def test_attempt_root_must_be_new(tmp_path) -> None:
    attempt = spike.create_attempt_root("a1", root=tmp_path)
    assert attempt.is_dir()
    with pytest.raises(FileExistsError):
        spike.create_attempt_root("a1", root=tmp_path)


# --- GPU smoke tests (two epochs, synthetic) ---------------------------------


def test_score_tensor_signed_gradients_and_updates(diagnostic_run) -> None:
    model, observer = diagnostic_run
    log = model.rq4b_log_
    diag = log["gradient_diagnostics"]
    assert diag["logits_shape"] == [1, 256, 2]
    assert diag["margin_shape"] == [256]
    assert diag["adversary_input_shape"][1] == 2 and diag["adversary_output_shape"][1] == 3
    assert diag["signed_identity_holds"]
    assert diag["additive_variant_max_abs_diff"] > 0
    assert diag["adversary_logit1_grad_nonzero_g3_rows"] > 0
    assert diag["total_minus_pred_opposes_adv_g3_rows"] == diag["adversary_logit1_grad_nonzero_g3_rows"]
    assert diag["outside_rows"] > 0 and diag["roster_rows"] > 0
    assert diag["adversary_grad_outside_g3_max_abs"] == 0.0
    assert diag["adversary_margin_grad_outside_g3_max_abs"] == 0.0
    assert diag["adversary_grad_g3_nonzero_rows"] > 0
    assert diag["prediction_grad_outside_g3_nonzero_rows"] == diag["outside_rows"]
    assert diag["predictor_lr_at_diagnosed_batch"] > 0
    assert diag["predictor_unchanged_by_adversary_step"]
    assert diag["predictor_grads_empty_after_adversary_backward"]
    assert diag["adversary_max_change_by_adversary_step"] > 0
    assert diag["adversary_unchanged_by_predictor_step"]
    assert diag["adversary_grads_unchanged_by_predictor_backward"]
    assert diag["predictor_max_change_by_predictor_step"] > 0
    assert not diag["adversary_params_in_predictor_optimizer"]
    assert log["adversary_updates"] == len(observer.batch_fingerprints) == 8
    assert observer.completed_epochs == 2
    devices = log["devices"]
    assert devices["predictor_parameters"] == ["cuda:0"]
    assert devices["adversary_parameters"] == ["cuda:0"]
    assert devices["batch_x_cont"] == devices["batch__rq4b_row_idx"] == devices["group_metadata"] == "cuda:0"
    assert devices["train_loader"]["batch_size"] == 256 and devices["train_loader"]["drop_last"] is True
    assert model.backend_.alg_interface_.model.creator.config["predict_batch_size"] == 1024


def test_lambda_zero_equivalence_to_preserved_recipe(lambda_zero_runs, fixture_data) -> None:
    stock, observed, observed_obs, extension, extension_obs = lambda_zero_runs
    assert extension_obs.batch_fingerprints == observed_obs.batch_fingerprints
    assert stock.backend_parameters() == extension.backend_parameters()
    assert stock.rq4b_input_digest_ == extension.rq4b_input_digest_ == observed.rq4b_input_digest_
    assert extension.rq4b_log_["adversary_updates"] == 0
    assert extension.rq4b_log_["adversary_bypassed_lambda_zero"] == 8
    assert all(extension.rq4b_log_["adversary_rng_isolation"].values())
    rtol, atol = spike.EQUIVALENCE_RTOL, spike.EQUIVALENCE_ATOL
    params = spike.compare_tensors(spike.predictor_tensors(stock), spike.predictor_tensors(extension), rtol=rtol, atol=atol)
    probs = spike.compare_probabilities(
        stock.predict_proba(fixture_data.X), extension.predict_proba(fixture_data.X), rtol=rtol, atol=atol
    )
    assert params["within_tolerance"], params
    assert probs["within_tolerance"], probs


def test_batch_rows_align_with_record_ids(lambda_zero_runs, fixture_data) -> None:
    extension = lambda_zero_runs[3]
    batches = extension.rq4b_log_["batches"]
    assert [b["epoch"] for b in batches] == [1] * 4 + [2] * 4
    for start in (0, 4):
        rows = sum((b["rows"] for b in batches[start : start + 4]), [])
        assert len(rows) == len(set(rows)) == 1024  # drop_last omits 76 rows per epoch
        assert sum(b["roster_rows"] for b in batches[start : start + 4]) == int(
            (fixture_data.group_codes[rows] >= 0).sum()
        )


def test_alignment_corruption_is_rejected(cuda0, fixture_data) -> None:
    corrupted = np.roll(fixture_data.y, 1)
    with pytest.raises(spike.RQ4bAlignmentError):
        spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=LAMBDA, alignment_labels=corrupted)
    with pytest.raises(spike.RQ4bAlignmentError):
        spike.fit_spike_realmlp(
            fixture_data, seed=SEED, adversary_lambda=LAMBDA, group_codes=fixture_data.group_codes[:-1]
        )


@pytest.mark.parametrize("method", ["_prediction_loss", "_adversary_loss"])
def test_nonfinite_losses_fail_closed(monkeypatch, cuda0, fixture_data, method) -> None:
    original = getattr(spike.AdversarialTabNNModule, method)

    def poisoned(self, *args, **kwargs):
        return original(self, *args, **kwargs) * float("nan")

    monkeypatch.setattr(spike.AdversarialTabNNModule, method, poisoned)
    with pytest.raises(spike.RQ4bNonFiniteError):
        spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=LAMBDA)


# --- R1 regressions: nonfinite before backward/step --------------------------


def _inject_nonfinite_criterion(monkeypatch, *, only_lambda_zero: bool = False) -> dict:
    """Poison the predictor criterion right after compile and snapshot initial state."""
    seen: dict = {}
    original_compile = spike.AdversarialTabNNModule.compile_model

    def compile_and_poison(self, *args, **kwargs):
        original_compile(self, *args, **kwargs)
        if only_lambda_zero and self.rq4b_lambda != 0.0:
            return
        criterion = self.criterion
        self.criterion = lambda y_pred, y: criterion(y_pred, y) * float("nan")
        seen["module"] = self
        seen["predictor_initial"] = [p.detach().cpu().clone() for p in self.model.parameters()]
        seen["adversary_initial"] = [p.detach().cpu().clone() for p in self.rq4b_adversary.parameters()]

    monkeypatch.setattr(spike.AdversarialTabNNModule, "compile_model", compile_and_poison)
    return seen


@pytest.mark.parametrize("adversary_lambda", [0.0, LAMBDA], ids=["lambda_zero", "lambda_positive"])
def test_nonfinite_prediction_loss_rejected_before_backward_and_step(monkeypatch, cuda0, fixture_data, adversary_lambda) -> None:
    from pytabkit.models.optim.optimizers import OptimizerBase

    calls = {"predictor_step": 0, "backward": 0}
    original_step = OptimizerBase.step
    original_backward = spike.pl.LightningModule.manual_backward

    def counting_step(self, *args, **kwargs):
        calls["predictor_step"] += 1
        return original_step(self, *args, **kwargs)

    def counting_backward(self, *args, **kwargs):
        calls["backward"] += 1
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(OptimizerBase, "step", counting_step)
    monkeypatch.setattr(spike.pl.LightningModule, "manual_backward", counting_backward)
    seen = _inject_nonfinite_criterion(monkeypatch)
    with pytest.raises(spike.RQ4bNonFiniteError, match="backward and optimizer step refused|Prediction loss"):
        spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=adversary_lambda)
    module = seen["module"]
    assert calls == {"predictor_step": 0, "backward": 0}
    assert all(p.grad is None for p in module.model.parameters())
    for before, after in zip(seen["predictor_initial"], module.model.parameters()):
        assert torch.equal(before, after.detach().cpu())
    for before, after in zip(seen["adversary_initial"], module.rq4b_adversary.parameters()):
        assert torch.equal(before, after.detach().cpu())
    assert module._rq4b_adversary_opt.state_dict()["state"] == {}
    assert module.rq4b_log["batches"] == [] and module.rq4b_log["adversary_updates"] == 0


def test_finite_lambda_zero_path_still_uses_stock_step(monkeypatch, cuda0, fixture_data) -> None:
    calls = {"stock_step": 0}
    original = spike.TabNNModule.training_step

    def counting(self, *args, **kwargs):
        calls["stock_step"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(spike.TabNNModule, "training_step", counting)
    model, observer = spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=0.0)
    assert calls["stock_step"] == len(observer.batch_fingerprints) == 8
    assert model.rq4b_log_["adversary_bypassed_lambda_zero"] == 8


# --- R1 regressions: mandatory gates proceed only on PASS --------------------


def _forbid_fits(monkeypatch) -> list:
    calls: list = []
    for name in ("fit_spike_realmlp", "fit_stock_realmlp", "make_synthetic_fixture", "make_alignment_fixture"):
        monkeypatch.setattr(spike, name, lambda *a, _n=name, **k: calls.append(_n) or (_ for _ in ()).throw(AssertionError(_n)))
    return calls


def _attempt_in(monkeypatch, root):
    original = spike.create_attempt_root
    monkeypatch.setattr(spike, "create_attempt_root", lambda name=None: original(name, root=root))


def _receipt(attempt):
    import json

    path = attempt / "FAILURE_RECEIPT.json"
    assert path.exists()
    assert not os.stat(path).st_mode & stat.S_IWRITE
    return json.loads(path.read_text(encoding="utf-8"))


def test_missing_upstream_evidence_terminates_before_any_fit(monkeypatch, tmp_path) -> None:
    import json

    calls = _forbid_fits(monkeypatch)
    _attempt_in(monkeypatch, tmp_path)
    assert spike.main(["--attempt-name", "missing_upstream"]) == 1
    attempt = tmp_path / "missing_upstream"
    receipt = _receipt(attempt)
    assert receipt["gate"] == "g01_pin_pytabkit_source_revision"
    assert receipt["category"] == "environment_failure"
    assert "UNRESOLVED" in receipt["message"]
    g01 = json.loads((attempt / "g01_pin_pytabkit_source_revision.json").read_text(encoding="utf-8"))
    assert g01["status"] == "UNRESOLVED"
    assert calls == []
    assert not (attempt / "g02_cuda_preflight.json").exists()
    assert not (attempt / "fixture.json").exists()
    assert json.loads((attempt / "gates.json").read_text(encoding="utf-8"))["g01_pin_pytabkit_source_revision"]["status"] == "UNRESOLVED"


@pytest.mark.parametrize("status", ["FAIL", "NOT_RUN", "UNRESOLVED", "MAYBE"])
def test_non_pass_mandatory_status_terminates(monkeypatch, tmp_path, status) -> None:
    calls = _forbid_fits(monkeypatch)
    _attempt_in(monkeypatch, tmp_path)
    monkeypatch.setattr(spike, "pytabkit_installed_lineage", lambda upstream_dir=None: {"lineage_status": status, "stub": "test-only"})
    assert spike.main(["--attempt-name", f"status_{status}"]) == 1
    receipt = _receipt(tmp_path / f"status_{status}")
    assert receipt["gate"] == "g01_pin_pytabkit_source_revision"
    assert receipt["category"] == ("implementation_defect" if status == "MAYBE" else "environment_failure")
    assert calls == []


@pytest.mark.parametrize(
    "only_lambda_zero, expected_gate",
    [(False, "g03_training_step_and_score_tensor"), (True, "g04_lambda_zero_equivalence")],
    ids=["lambda_positive_gate", "lambda_zero_gate"],
)
def test_nonfinite_loss_writes_attempt_failure_receipt(monkeypatch, cuda0, tmp_path, only_lambda_zero, expected_gate) -> None:
    import json

    _attempt_in(monkeypatch, tmp_path)
    # Test-only lineage stub: the real lineage gate needs network evidence.
    monkeypatch.setattr(spike, "pytabkit_installed_lineage", lambda upstream_dir=None: {"lineage_status": "PASS", "stub": "test-only"})
    _inject_nonfinite_criterion(monkeypatch, only_lambda_zero=only_lambda_zero)
    name = f"nonfinite_{expected_gate}"
    assert spike.main(["--attempt-name", name]) == 1
    attempt = tmp_path / name
    receipt = _receipt(attempt)
    assert receipt["gate"] == expected_gate
    assert receipt["category"] == "implementation_defect"
    assert "RQ4bNonFiniteError" in receipt["message"]
    assert not (attempt / f"{expected_gate}.json").exists()
    gates = json.loads((attempt / "gates.json").read_text(encoding="utf-8"))
    assert gates[expected_gate]["status"] == "FAIL"


# --- R1 regressions: row -> feature -> group alignment -----------------------


def test_alignment_fixture_has_same_outcome_rows_with_distinct_identity_and_group() -> None:
    fixture = spike.make_alignment_fixture()
    for label in (0, 1):
        rows = fixture.y == label
        assert len(set(fixture.group_codes[rows].tolist())) == 4
        assert len(set(fixture.record_ids[rows].tolist())) == int(rows.sum())


def test_row_feature_group_correspondence_in_batches(cuda0) -> None:
    fixture = spike.make_alignment_fixture()
    model, observer = spike.fit_spike_realmlp(
        fixture, seed=SEED, adversary_lambda=LAMBDA,
        group_feature_column=spike.ALIGNMENT_GROUP_COLUMN, log_raw_batch_features=True,
    )
    log = model.rq4b_log_
    assert log["alignment"]["dataset_rows_verified"] == len(fixture.X)
    assert log["alignment"]["batches_feature_verified"] == log["alignment"]["batches_group_feature_verified"] == 8
    for batch in log["batches"]:
        raw = np.asarray(batch["raw_x_cont"], dtype=np.float32)
        rows = np.asarray(batch["rows"])
        ids = np.asarray([f"SYN-{int(v):06d}" for v in raw[:, spike.ALIGNMENT_ID_COLUMN]], dtype=object)
        assert (fixture.record_ids[rows] == ids).all()
        assert (raw[:, spike.ALIGNMENT_GROUP_COLUMN].astype(np.int64) == np.asarray(batch["groups"])).all()
        assert (fixture.group_codes[rows] == np.asarray(batch["groups"])).all()
        labels = fixture.y[rows]
        assert any(len(set(np.asarray(batch["groups"])[labels == label].tolist())) > 1 for label in (0, 1))


def test_label_preserving_corruptions_are_rejected(cuda0, fixture_data) -> None:
    fixture = spike.make_alignment_fixture()
    swapped, perm, info = spike.label_preserving_group_swap(fixture)
    assert np.array_equal(fixture.y[perm], fixture.y)  # a label-only check cannot see these
    assert info["groups_before"][0] != info["groups_before"][1]
    with pytest.raises(spike.RQ4bAlignmentError, match="group-encoding feature"):
        spike.fit_spike_realmlp(
            fixture, seed=SEED, adversary_lambda=LAMBDA, group_codes=swapped,
            group_feature_column=spike.ALIGNMENT_GROUP_COLUMN,
        )
    # Standard fixture: metadata table rows swapped within the same outcome.
    _, perm, _ = spike.label_preserving_group_swap(fixture_data)
    assert np.array_equal(fixture_data.y[perm], fixture_data.y)
    with pytest.raises(spike.RQ4bAlignmentError, match="external row-aligned features"):
        spike.fit_spike_realmlp(
            fixture_data, seed=SEED, adversary_lambda=LAMBDA,
            group_codes=fixture_data.group_codes[perm],
            alignment_features=spike.native_numeric_features(fixture_data.X)[perm],
        )


# --- R1: tolerance utilization reporting ------------------------------------


def test_tolerance_utilization_matches_allclose_boundary() -> None:
    reference = {"a": np.array([0.0, 1.0, -2.0], dtype=np.float32)}
    candidate = {"a": np.array([5e-7, 1.0 + 2e-5, -2.0], dtype=np.float32)}
    result = spike.compare_tensors(reference, candidate, rtol=1e-5, atol=1e-6)
    util = result["tolerance_utilization"]
    expected = spike.tolerance_utilization(reference["a"], candidate["a"], rtol=1e-5, atol=1e-6)
    assert util["max_utilization"] == pytest.approx(float(expected.max()))
    assert util["worst_index"] == [1]
    assert util["margin_to_tolerance"] == pytest.approx(1.0 - util["max_utilization"])
    assert result["within_tolerance"] == (util["elements_over_1_0"] == 0)
    probs = spike.compare_probabilities(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]), rtol=1e-5, atol=1e-6)
    assert probs["tolerance_utilization"]["max_utilization"] == 0.0


def test_empty_masks_skip_only_adversary_update(cuda0, fixture_data, lambda_zero_runs) -> None:
    codes = np.full(len(fixture_data.X), spike.OUTSIDE_ROSTER_CODE, dtype=np.int64)
    model, observer = spike.fit_spike_realmlp(fixture_data, seed=SEED, adversary_lambda=LAMBDA, group_codes=codes)
    log = model.rq4b_log_
    assert log["adversary_updates"] == 0
    assert log["adversary_skipped_empty_mask"] == len(observer.batch_fingerprints) == 8
    assert all(batch["adversary"] == "skipped_empty_mask" for batch in log["batches"])
    stock = lambda_zero_runs[0]
    # With no roster rows the objective is L_prediction only, so the predictor
    # update must reproduce the preserved recipe under the baseline policy.
    change = spike.compare_tensors(
        spike.predictor_tensors(stock),
        spike.predictor_tensors(model),
        rtol=spike.EQUIVALENCE_RTOL,
        atol=spike.EQUIVALENCE_ATOL,
    )
    assert change["within_tolerance"], change
    assert all(batch["total_loss"] == batch["prediction_loss"] for batch in log["batches"])


def test_reload_exact_and_failure_paths(lambda_zero_runs, diagnostic_run, fixture_data, tmp_path) -> None:
    model = diagnostic_run[0]
    evidence = spike.serialize_and_verify_reload(model, fixture_data.X, tmp_path / "model.joblib")
    assert evidence["probabilities_exactly_equal"]
    assert evidence["adversary_present_after_reload"]
    assert not evidence["training_metadata_pickled"]
    with pytest.raises(spike.RQ4bReloadError, match="overwrite"):
        spike.serialize_and_verify_reload(model, fixture_data.X, tmp_path / "model.joblib")

    def truncated_loader(path):
        data = path.read_bytes()
        broken = path.with_name("broken.joblib")
        broken.write_bytes(data[: len(data) // 2])
        import joblib

        return joblib.load(broken)

    with pytest.raises(spike.RQ4bReloadError, match="failed to reload"):
        spike.serialize_and_verify_reload(model, fixture_data.X, tmp_path / "m2.joblib", loader=truncated_loader)

    def perturbed_loader(path):
        import joblib

        reloaded = joblib.load(path)
        with torch.no_grad():
            next(reloaded.backend_.alg_interface_.model.model.parameters()).add_(0.5)
        return reloaded

    with pytest.raises(spike.RQ4bReloadError, match="changed eval-mode probabilities"):
        spike.serialize_and_verify_reload(model, fixture_data.X, tmp_path / "m3.joblib", loader=perturbed_loader)
