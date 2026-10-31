from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.dataset_b import ConfigResolutionError, config_path
from src.dataset_b.cli import build_parser, main
from src.dataset_b.core.config import validate_catalog
from src.dataset_b.core.navigation import (
    report_index,
    status_snapshot,
    validate_configuration,
    validate_frozen_artifacts,
    validate_notebooks,
)
from src.utils.path import PROJECT_ROOT


def test_public_package_import_and_catalog_resolution_are_cpu_safe() -> None:
    assert config_path("contracts.features") == (
        PROJECT_ROOT / "config" / "dataset_b" / "feature_set_v1.yaml"
    )
    assert config_path("models.realmlp").name == "realmlp_v1.yaml"
    assert config_path("research.rq4b").name == "rq4b_adversarial.yaml"


def test_model_and_classical_namespaces_are_cpu_safe_in_fresh_process() -> None:
    code = """
import sys
import src.dataset_b.models as models
assert 'torch' not in sys.modules
assert 'fairlearn' not in sys.modules
expected = {
    'PlainMLPClassifier', 'RealMLPClassifier', 'RealMLPConfigurationError',
    'load_realmlp_v1_config', 'TabMMiniConfigurationError', 'TabMMiniModel',
    'compute_pos_weight', 'expand_member_targets', 'load_tabmmini_v1_config',
    'mean_member_probabilities', 'member_wise_weighted_bce',
}
assert set(models.__all__) == expected
import src.dataset_b.models.classical as classical
assert 'torch' not in sys.modules
assert 'fairlearn' not in sys.modules
assert classical.make_preprocessing_pipeline.__module__ == 'src.dataset_b.models.classical.profiles'
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("role", ["features", "unknown.value", "models.unknown"])
def test_config_resolver_rejects_unknown_or_malformed_roles(role: str) -> None:
    with pytest.raises(ConfigResolutionError):
        config_path(role)


def test_catalog_validation_covers_each_unique_target() -> None:
    result = validate_catalog()

    assert result["status"] == "PASS"
    assert result["dataset"] == "dataset_b"
    assert result["target_count"] == 11
    assert len(result["targets"]) == len(set(result["targets"].values()))


def test_status_snapshot_is_explicitly_non_executing() -> None:
    result = status_snapshot()

    assert result["status"] == "PASS"
    assert result["active_dataset"] == "dataset_b"
    assert all(result["frozen_markers"].values())
    assert result["safety"] == {
        "training_allowed": False,
        "inference_allowed": False,
        "notebook_execution_allowed": False,
    }


def test_static_validators_cover_config_artifacts_and_notebooks() -> None:
    config_result = validate_configuration()
    artifact_result = validate_frozen_artifacts()
    notebook_result = validate_notebooks(scope="final")

    assert config_result["target_count"] == 11
    assert artifact_result["frozen_marker_count"] == 6
    assert artifact_result["checkpoints_loaded"] == 0
    assert artifact_result["inference_calls"] == 0
    assert notebook_result["notebook_count"] == 14
    assert notebook_result["saved_error_count"] == 0
    assert notebook_result["executed"] is False


def test_report_index_lists_current_final_notebooks() -> None:
    result = report_index()

    assert result["status"] == "PASS"
    assert result["notebook_count"] == 14
    assert all("notebooks/dataset_b/final/" in path for path in result["final_notebooks"])


@pytest.mark.parametrize(
    "argv",
    [
        ["status"],
        ["validate-config"],
        ["validate-artifacts", "--scope", "frozen"],
        ["validate-notebooks", "--scope", "final"],
        ["report-index"],
    ],
)
def test_public_cli_commands_return_json_without_experiment_execution(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] in {"PASS", "INCOMPLETE"}


@pytest.mark.parametrize("command", ["train", "evaluate", "predict", "serve", "inference"])
def test_public_cli_has_no_model_execution_commands(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_new_navigation_modules_have_no_model_backend_or_execution_calls() -> None:
    source_paths = [
        PROJECT_ROOT / "src" / "dataset_b" / "__init__.py",
        PROJECT_ROOT / "src" / "dataset_b" / "core" / "__init__.py",
        PROJECT_ROOT / "src" / "dataset_b" / "core" / "config.py",
        PROJECT_ROOT / "src" / "dataset_b" / "core" / "navigation.py",
        PROJECT_ROOT / "src" / "dataset_b" / "cli.py",
    ]
    forbidden_imports = {"torch", "fairlearn", "joblib", "pickle", "subprocess"}
    forbidden_calls = {"fit", "fit_transform", "predict", "predict_proba"}

    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert imported.isdisjoint(forbidden_imports), path
        assert calls.isdisjoint(forbidden_calls), path


def test_removed_empty_scaffolds_remain_absent() -> None:
    assert not (PROJECT_ROOT / "src" / "api").exists()
    assert not (PROJECT_ROOT / "src" / "pipeline").exists()
