"""Production glue exercised with synthetic disk inputs and mocked engines only.

All generated files must live in the approved integration/N2 audit basetemp.
No production run, project data, fitted model, CUDA operation or real inference.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_b.mitigation import rq4b_evaluation as ev
from src.dataset_b.mitigation import rq4b_inputs as ri
from src.dataset_b.mitigation import rq4b_runner as rr
from src.dataset_b.mitigation import rq4b_training as rt


def forbidden(*args, **kwargs):
    raise AssertionError("Real execution is outside this synthetic integration task")


@pytest.fixture(autouse=True)
def guard_real_execution(monkeypatch):
    """Fail if a test accidentally reaches GPU/fit or a non-scratch CSV/NPZ."""
    monkeypatch.setattr(rt, "require_cuda0", forbidden)
    monkeypatch.setattr(rt, "_fit_engine", forbidden)
    monkeypatch.setattr(rt, "open_production_context", forbidden)
    csv_reader, npz_reader = pd.read_csv, np.load

    def checked_csv(path, *args, **kwargs):
        assert rr.SYNTHETIC_ROOT.resolve() in Path(path).resolve().parents
        return csv_reader(path, *args, **kwargs)

    def checked_npz(path, *args, **kwargs):
        assert rr.SYNTHETIC_ROOT.resolve() in Path(path).resolve().parents
        return npz_reader(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", checked_csv)
    monkeypatch.setattr(np, "load", checked_npz)


@pytest.fixture
def local(tmp_path):
    assert any(p.name.startswith(("CODEX_INTEGRATION_2_", "LOGGING_FIX_")) or p.name == "N2_real_probe_20261019_01" for p in tmp_path.parents)
    config = yaml.safe_load(ri.RQ4B_CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ri.PROJECT_ROOT / "config/dataset_b/rq3_protocol.yaml").read_text(encoding="utf-8"))
    contract = ri.validate_contract(config, protocol, project_root=tmp_path)
    contract = dataclasses.replace(contract, split_rows={"train": 4, "validation": 2, "test": 2})
    for relative in ("config/dataset_b", "artifacts/dataset_b", "data/dataset_b/split_v1", "data/raw"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ri.PROJECT_ROOT / ri.FEATURE_CONTRACT_PATH, contract.resources["feature_contract"])
    config_path = tmp_path / "config/dataset_b/rq4b_adversarial.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    actions = ["1", "3", "2", "3", "1", "3", "3", "1"]
    codes = ["1", "2", "3", "4", "6", "1", "2", "1"]
    ids = [f"dataset_b-{k:09d}" for k in range(1, 9)]
    race = ["White", "Black or African American", "Asian", "Joint", "Asian", "White", "Asian", "White"]
    raw = {"loan_type": "1", "loan_purpose": "1", "lien_status": "1", "reverse_mortgage": "2",
           "open-end_line_of_credit": "2", "business_or_commercial_purpose": "2", "income": "100",
           "debt_to_income_ratio": "<20%", "loan_amount": "200000", "loan_to_value_ratio": "80",
           "loan_term": "360", "occupancy_type": "1", "preapproval": "2", "interest_only_payment": "2", "balloon_payment": "2"}
    audit = {"lei": "L", "county_code": "1", **{f"applicant_race-{k}": "1" for k in range(1, 6)}, "applicant_sex": "1", "applicant_age": "35-44"}
    members = {"train": ids[:4], "validation": ids[4:6], "test": ids[6:]}
    manifest = {"dataset": "dataset_b", "split_unit": "record_id", "grouping_constraint": "none", "random_state": 42, "split_version": "split_v1",
                **{f"{name}_record_ids": values for name, values in members.items()}}
    contract.resources["split_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    for name, values in members.items():
        rows = []
        for record_id in values:
            k = ids.index(record_id)
            rows.append({"record_id": record_id, "action_taken": actions[k], "y_approved": int(actions[k] != "3"),
                         **raw, **audit, "applicant_sex": codes[k]})
        pd.DataFrame(rows).to_csv(contract.resources[f"split_{name}"], index=False)
    pd.DataFrame({"action_taken": actions, "derived_race": race, "lei": ["L"] * 8, "applicant_sex": codes}).to_csv(contract.resources["derived_race_source"], index=False)
    return SimpleNamespace(contract=contract, config=config, config_path=config_path, ids=ids, race=race,
                           backend=rr.ProductionBackend(contract, config_path=config_path), root=tmp_path)


@pytest.mark.parametrize("stage", ["cv_epoch_selection", "full_train_refit"])
def test_population_real_loaders_preserve_ids_labels_features_and_race(local, stage):
    pop = local.backend._population(ri.StageAccess(stage, local.contract))
    assert pop.record_ids == tuple(local.ids[:4])
    assert pop.y.tolist() == [1, 0, 1, 0]
    assert pop.derived_race.tolist() == local.race[:4]
    assert tuple(pop.X_raw) == ri.RAW_PREDICTOR_COLUMNS
    assert len(pop.X_raw) == 4 and not set(pop.X_raw) & ri.FORBIDDEN_PREDICTOR_COLUMNS


@pytest.mark.parametrize("field,value", [("action_taken", "3"), ("lei", "wrong"), ("applicant_sex", "6")])
def test_population_reconciliation_rejects_corrupt_raw_metadata(local, field, value):
    path = local.contract.resources["derived_race_source"]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ri.RQ4bInputError):
        local.backend._population(ri.StageAccess("cv_epoch_selection", local.contract))


def test_train_glue_never_opens_validation_or_test(local, monkeypatch):
    original = pd.read_csv
    opened = []

    def tracked(path, *args, **kwargs):
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracked)
    local.backend._population(ri.StageAccess("cv_epoch_selection", local.contract))
    assert set(opened) == {local.contract.resources["split_train"], local.contract.resources["derived_race_source"]}


def make_indices(local):
    ids = local.ids
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
    rng = np.random.default_rng(42)
    indices = np.array([np.r_[rng.choice(np.flatnonzero(y == 0), 4), rng.choice(np.flatnonzero(y == 1), 4)] for _ in range(1000)], dtype=np.int32)
    path = local.contract.resources["locked_test_bootstrap_indices"]
    np.savez_compressed(path, indices=indices)
    manifest = {"indices_sha256": rr.sha256_file(path), "test_record_id_sha256": rr.sha256_ids(ids), "test_rows": len(ids)}
    mpath = path.parent / "rq1_test_bootstrap_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return y, indices, path, mpath, manifest


def test_index_loader_reads_exact_synthetic_payload_without_regeneration(local, monkeypatch):
    y, expected, path, _, _ = make_indices(local)
    before = path.read_bytes()
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    actual = local.backend.load_test_indices(y, local.ids)
    np.testing.assert_array_equal(actual, expected)
    assert path.read_bytes() == before


@pytest.mark.parametrize("case", ["hash", "ids", "row_count", "missing_manifest", "missing_npz", "key", "extra_key", "dtype", "shape", "range", "strata"])
def test_index_loader_negative_cases_fail_closed_without_regeneration(local, monkeypatch, case):
    y, array, path, mpath, manifest = make_indices(local)
    if case == "hash":
        manifest["indices_sha256"] = "0" * 64
    elif case == "ids":
        manifest["test_record_id_sha256"] = rr.sha256_ids(local.ids[::-1])
    elif case == "row_count":
        manifest["test_rows"] = 9
    elif case == "missing_npz":
        path.unlink()  # only this test's synthetic fixture
    elif case not in ("missing_manifest",):
        if case == "dtype": array = array.astype(np.int64)
        if case == "shape": array = array[:999]
        if case == "range": array[0, 0] = len(y)
        if case == "strata": array[0, 0] = np.flatnonzero(y == 1)[0]
        payload = {"wrong" if case == "key" else "indices": array}
        if case == "extra_key": payload["other"] = np.array([1])
        np.savez_compressed(path, **payload)
        manifest["indices_sha256"] = rr.sha256_file(path)
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    if case == "missing_manifest": mpath.unlink()
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    with pytest.raises((rr.RQ4bRunnerError, ev.RQ4bEvaluationError, FileNotFoundError)):
        local.backend.load_test_indices(y, local.ids)


@pytest.mark.parametrize("missing", [None, *rr.REQUIRED_ACCEPTANCE, "cuda0", "indices"])
def test_backend_preflight_statuses_use_local_config_and_mocked_context(local, monkeypatch, missing):
    make_indices(local)
    if missing in rr.REQUIRED_ACCEPTANCE:
        del local.config["execution_governance"][missing]
        local.config_path.write_text(yaml.safe_dump(local.config), encoding="utf-8")
    if missing == "indices": local.contract.resources["locked_test_bootstrap_indices"].unlink()
    context = SimpleNamespace(prerequisites={"fixture_gate": "PASS"}, cuda={"device": "cpu" if missing == "cuda0" else "cuda:0"})
    monkeypatch.setattr(local.backend, "context", Mock(return_value=context))
    statuses = local.backend.preflight()  # no real context, GPU or run directory
    assert set(rr.REQUIRED_ACCEPTANCE) <= set(statuses)
    if missing is None: assert all(v == "PASS" for v in statuses.values())
    else: assert any(v != "PASS" for v in statuses.values())
    assert not local.contract.artifact_namespace.exists()


@pytest.mark.parametrize("value", [None, "FAIL", "pass", True, 1])
def test_real_preflight_status_collection_is_rejected_by_stage_guard(local, monkeypatch, value):
    make_indices(local)
    local.config["execution_governance"]["t5_runner_acceptance"]["result"] = value
    local.config_path.write_text(yaml.safe_dump(local.config), encoding="utf-8")
    monkeypatch.setattr(local.backend, "context", lambda: SimpleNamespace(prerequisites={"mock": "PASS"}, cuda={"device": "cuda:0"}))
    # Exercise the real status collector and stage guard via a synthetic adapter,
    # never construct a production-mode run or publish in its namespace.
    adapter = SimpleNamespace(kind="synthetic_preflight_adapter", preflight=local.backend.preflight)
    layout = rr.create_run(local.root / "runs", "rq4b3_adv_20261018_01", rr.MODE_SYNTHETIC, local.contract, config_path=local.config_path)
    runner = rr.RQ4bRunner(layout, adapter, local.contract)
    with pytest.raises(rr.RQ4bRunnerError, match="not PASS"):
        runner.preflight()
    assert not layout.stage_dir("preflight").exists()
    assert len(list(layout.failures.glob("preflight.*/FAILURE_RECEIPT.json"))) == 1
    assert not local.contract.artifact_namespace.exists()


def test_context_is_initialized_once_using_the_selected_config(local, monkeypatch):
    expected = object()
    opener = Mock(return_value=expected)
    monkeypatch.setattr(local.backend, "_training", lambda: SimpleNamespace(open_production_context=opener))
    assert local.backend.context() is expected and local.backend.context() is expected
    opener.assert_called_once_with(local.config_path)


def test_capability_ingests_explicit_real_evidence_without_fit(local, monkeypatch):
    from src.dataset_b.mitigation import rq4b_probe as rp
    monkeypatch.setattr(rr, "PROJECT_ROOT", local.root)
    gate = local.config["execution_governance"]["t3_capability_probe_acceptance"]["synthetic_gate"]
    gate["evidence_path"] = "evidence/synthetic_gate.json"
    local.config_path.write_text(yaml.safe_dump(local.config), encoding="utf-8")
    gate_path = local.root / gate["evidence_path"]
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text("synthetic accepted gate")
    path = local.root / "real_probe_evidence.json"
    evidence = {"status": "completed", "capability_status": "capability_inconclusive", "evidence": {"holdout": {"ce_gain_nats": -0.1}}}
    loader = Mock(return_value=evidence)
    monkeypatch.setattr(rp, "load_real_probe_evidence", loader)
    monkeypatch.setattr(rp, "fit_probe", forbidden)
    local.backend.real_probe_evidence = path
    result = local.backend.capability_evidence()
    loader.assert_called_once_with(path, local.contract)
    assert result["real_probe"] == evidence and result["synthetic_gate_status"] == "PASS"
    local.backend.real_probe_evidence = None
    assert local.backend.capability_evidence()["real_probe"]["status"] == "not_run_requires_separate_authorization"


def test_capability_invalid_explicit_evidence_propagates_failure(local, monkeypatch):
    from src.dataset_b.mitigation import rq4b_probe as rp
    monkeypatch.setattr(rp, "load_real_probe_evidence", Mock(side_effect=rp.RQ4bProbeError("invalid evidence")))
    local.backend.real_probe_evidence = local.root / "missing.json"
    with pytest.raises(rp.RQ4bProbeError, match="invalid evidence"):
        local.backend.capability_evidence()


def test_cli_forwards_real_evidence_only_to_capability(local, monkeypatch):
    monkeypatch.setattr(ri, "load_contract", lambda: local.contract)
    monkeypatch.setattr(rr, "open_run", Mock())
    backend, runner = Mock(), Mock()
    runner.capability.return_value = {"stage_key": "capability", "status": "PASS", "counts": {}}
    monkeypatch.setattr(rr, "ProductionBackend", backend)
    monkeypatch.setattr(rr, "RQ4bRunner", Mock(return_value=runner))
    path = local.root / "real_probe_evidence.json"
    assert rr.main(["--stage", "capability", "--run-id", "rq4b3_adv_20261019_01", "--confirm-production", "--real-probe-evidence", str(path)]) == 0
    backend.assert_called_once_with(local.contract, real_probe_evidence=path)
    runner.capability.assert_called_once_with()


@pytest.mark.parametrize("stage", [s for s in rr.STAGE_COMMANDS if s != "capability"])
def test_cli_rejects_evidence_on_other_stages_before_side_effects(stage, monkeypatch):
    monkeypatch.setattr(ri, "load_contract", forbidden)
    with pytest.raises(rr.RQ4bRunnerError, match="only valid for capability"):
        rr.main(["--stage", stage, "--run-id", "rq4b3_adv_20261019_01", "--confirm-production", "--real-probe-evidence", "missing.json"])
def test_capability_glue_uses_synthetic_evidence_and_does_not_fit_real_probe(local, monkeypatch):
    record = local.config["execution_governance"]["t3_capability_probe_acceptance"]
    # Keep this fixture below Windows path limits even with a long basetemp.
    record["synthetic_gate"]["evidence_path"] = "evidence/synthetic_gate.json"
    local.config_path.write_text(yaml.safe_dump(local.config), encoding="utf-8")
    evidence = local.root / record["synthetic_gate"]["evidence_path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"synthetic_fixture_only": true}', encoding="utf-8")
    monkeypatch.setattr(rr, "PROJECT_ROOT", local.root)
    result = local.backend.capability_evidence()
    assert result["synthetic_gate_status"] == "PASS"
    assert result["evidence_sha256"] == rr.sha256_file(evidence)
    assert result["real_probe"] == {"status": "not_run_requires_separate_authorization", "capability_status": "capability_inconclusive"}


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_predict_glue_with_real_synthetic_loaders_and_mocked_models(local, monkeypatch, partition):
    model_dir = local.root / "mock_models"
    preprocessor = Mock()
    preprocessor.transform.side_effect = lambda X: X.copy()
    estimator = Mock()
    estimator.predict_proba.side_effect = lambda X: np.tile([0.7, 0.3], (len(X), 1))
    loader = Mock(side_effect=lambda path: preprocessor if Path(path).name == "preprocessor.joblib" else estimator)
    monkeypatch.setattr(joblib, "load", loader)
    result = local.backend.predict(model_dir, partition)
    start = 4 if partition == "validation" else 6
    assert result.record_ids == tuple(local.ids[start:start + 2])
    assert result.y.tolist() == ([1, 0] if partition == "validation" else [0, 1])
    assert result.race.tolist() == local.race[start:start + 2]
    assert result.sex.tolist() == (["Applicant selected both male and female", "Male"] if partition == "validation" else ["Female", "Male"])
    assert result.p_approved.tolist() == [0.3, 0.3]
    assert [call.args[0] for call in loader.call_args_list] == [model_dir / "preprocessor.joblib", model_dir / "estimator.joblib"]
    assert tuple(preprocessor.transform.call_args.args[0]) == ri.RAW_PREDICTOR_COLUMNS
    assert estimator.predict_proba.call_count == 1


@pytest.mark.parametrize("probabilities", [np.array([[np.nan, np.nan], [0.5, 0.5]]), np.array([[0.0, 1.2], [0.5, 0.5]]), np.array([[0.5], [0.5]]), np.array([[0.5, 0.5]])])
def test_predict_invalid_outputs_are_not_silently_repaired(local, monkeypatch, probabilities):
    preprocessor, estimator = Mock(), Mock()
    preprocessor.transform.side_effect = lambda X: X
    estimator.predict_proba.return_value = probabilities
    monkeypatch.setattr(joblib, "load", lambda p: preprocessor if Path(p).name == "preprocessor.joblib" else estimator)
    with pytest.raises((ev.RQ4bEvaluationError, IndexError)):
        local.backend.predict(local.root / "mock_models", "validation")
    assert estimator.predict_proba.call_count == 1


def test_fit_unit_handshake_passes_selection_population_and_paths_without_fits(local, monkeypatch):
    population = local.backend._population(ri.StageAccess("cv_epoch_selection", local.contract))
    record = SimpleNamespace(curves=np.ones((5, 256)), adversary_lambda=0.1, model_seed=42, selection={"selected_epoch": 37},
                             fold_membership_sha256="a" * 64, fold_summaries=(), fold_outputs=({"fold_index": 1},),
                             evidence_kind="mock_not_observed")
    preprocessor = Mock()
    preprocessor.transform.side_effect = lambda X: X
    refit = SimpleNamespace(preprocessor=preprocessor, estimator=object(), history={"mock": True}, log={"mock": True}, epoch_cap_hit=False)
    cv, fit, reload = Mock(return_value=record), Mock(return_value=refit), Mock(return_value={"mock_reload": True})
    progress = Mock()
    fake = SimpleNamespace(DurableProgress=Mock(return_value=progress), run_cv_epoch_selection=cv,
                           run_full_train_refit=fit, save_and_verify_reload=reload)
    monkeypatch.setattr(local.backend, "_training", lambda: fake)
    context = object()
    monkeypatch.setattr(local.backend, "context", lambda: context)
    populations = Mock(return_value=population)
    monkeypatch.setattr(local.backend, "_population", populations)
    dump = Mock()
    monkeypatch.setattr(joblib, "dump", dump)
    out_dir = local.root / "mock_fit"
    result = local.backend.fit_unit(0.1, 42, out_dir)
    assert [c.args[0].stage for c in populations.call_args_list] == ["cv_epoch_selection", "full_train_refit"]
    assert cv.call_args.args == (context, populations.call_args_list[0].args[0], population)
    assert fit.call_args.args == (context, populations.call_args_list[1].args[0], population, record)
    assert cv.call_args.kwargs == fit.call_args.kwargs == {"adversary_lambda": 0.1, "model_seed": 42, "progress": progress}
    assert dump.call_args.args == (preprocessor, out_dir / "preprocessor.joblib")
    assert reload.call_args.args[0] is refit.estimator and reload.call_args.args[2] == out_dir / "estimator.joblib"
    assert reload.call_args.kwargs == {"contract": local.contract}
    assert result["cv_fold_fits"] == 5 and result["refit_fits"] == 1
    assert result["cv_fold_outputs"] == ({"fold_index": 1},)
    assert result["selection_record"]["evidence_kind"] == "mock_not_observed"
    progress.finalize.assert_called_once_with()


@pytest.mark.parametrize("stage", rr.STAGE_COMMANDS)
def test_cli_dispatches_exactly_one_mocked_stage_without_production_effects(local, monkeypatch, capsys, stage):
    monkeypatch.setattr(ri, "load_contract", lambda: local.contract)
    layout = SimpleNamespace(run_dir=local.root / "never_created")
    create, open_run, backend = Mock(return_value=layout), Mock(return_value=layout), Mock()
    monkeypatch.setattr(rr, "create_run", create)
    monkeypatch.setattr(rr, "open_run", open_run)
    monkeypatch.setattr(rr, "ProductionBackend", backend)
    runner = Mock()
    for name in ("preflight", "capability", "development", "development_validation", "validation_selection", "final_fit", "final_validation", "test_evaluation", "package_validation"):
        getattr(runner, name).return_value = {"stage_key": stage, "status": "PASS", "counts": {}}
    monkeypatch.setattr(rr, "RQ4bRunner", Mock(return_value=runner))
    args = ["--stage", stage, "--run-id", "rq4b3_adv_20261018_01", "--confirm-production"]
    unit = stage in ("development", "development_validation", "final_fit", "final_validation")
    if unit: args += ["--lambda", "0.1", "--seed", "42"]
    if stage in ("development", "final_fit"): args += ["--new-attempt-after-failure"]
    assert rr.main(args) == 0
    if stage == "create":
        assert create.call_count == 1 and open_run.call_count == 0 and not runner.method_calls
    else:
        assert create.call_count == 0 and open_run.call_count == 1
        assert len(runner.method_calls) == 1
        call = runner.method_calls[0]
        assert call[0] == ("validation_selection" if stage == "selection" else stage)
        if unit: assert call.args == (0.1, 42)
        if stage in ("development", "final_fit"): assert call.kwargs == {"allow_new_attempt_after_failure": True}
    assert not layout.run_dir.exists() and not local.contract.artifact_namespace.exists()


@pytest.mark.parametrize("args", [[], ["--stage", "all"], ["--stage", "preflight", "--run-id", "rq4b3_adv_20261018_01"]])
def test_invalid_cli_rejected_before_contract_or_stage_effects(monkeypatch, args):
    monkeypatch.setattr(ri, "load_contract", forbidden)
    with pytest.raises(SystemExit): rr.main(args)


@pytest.mark.parametrize("stage,extra", [("development", []), ("final_fit", ["--lambda", "0.1"]), ("preflight", ["--seed", "42"]), ("selection", ["--lambda", "0.1"])])
def test_cli_argument_mismatch_never_invokes_a_stage(local, monkeypatch, stage, extra):
    monkeypatch.setattr(ri, "load_contract", lambda: local.contract)
    monkeypatch.setattr(rr, "open_run", Mock())
    monkeypatch.setattr(rr, "ProductionBackend", Mock())
    runner = Mock()
    monkeypatch.setattr(rr, "RQ4bRunner", Mock(return_value=runner))
    with pytest.raises(rr.RQ4bRunnerError):
        rr.main(["--stage", stage, "--run-id", "rq4b3_adv_20261018_01", "--confirm-production", *extra])
    assert not runner.method_calls and not local.contract.artifact_namespace.exists()
