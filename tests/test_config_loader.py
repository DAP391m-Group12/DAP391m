import os
import tempfile
from pathlib import Path
import pytest

from src.utils.config_loader import (
    Config,
    ConfigDict,
    get_project_root,
    load_config,
)
from src.utils.path import CONFIG_CATALOG_PATH, PROJECT_CONFIG_PATH


class TestConfig:
    def test_attribute_access(self):
        config = Config({"server": {"host": "localhost", "port": 8080}})
        assert config.server.host == "localhost"
        assert config.server.port == 8080

    def test_nested_assignment(self):
        config = Config({"server": {"host": "localhost"}})
        config.server.port = 9000
        assert config.server.port == 9000

        config.new_section = {"nested_key": 123}
        assert isinstance(config.new_section, Config)
        assert config.new_section.nested_key == 123

    def test_missing_attribute_raises_attribute_error(self):
        config = Config({"a": 1})
        with pytest.raises(AttributeError, match="Không tìm thấy cấu hình"):
            _ = config.missing_key

    def test_list_of_dicts_conversion(self):
        config = Config({"items": [{"name": "item1"}, {"name": "item2"}]})
        assert config.items[0].name == "item1"
        assert config.items[1].name == "item2"


class TestPathResolutionAndLoading:
    def test_project_root(self):
        root = get_project_root()
        assert root.is_dir()
        assert (root / "config").is_dir() or (root / "pyproject.toml").exists()

    def test_load_default_config(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.project.name == "dap391m_gr12"
        assert cfg.project.active_dataset == "dataset_b"
        assert cfg.logging.default_level == "INFO"

    def test_load_by_filename(self):
        cfg = load_config("project.yaml")
        assert cfg.source.raw_file == "data/raw/state_NY_2024.csv"

    def test_load_from_relative_path(self):
        cfg = load_config("config/project.yaml")
        assert cfg.configuration.catalog == "config/dataset_b/catalog.yaml"

    def test_load_nested_config_from_config_namespace(self):
        cfg = load_config("dataset_b/split_v1.yaml")
        assert cfg.dataset == "dataset_b"
        assert cfg.random_state == 42

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config_12345.yaml")

    def test_removed_legacy_config_paths_raise(self):
        for legacy_path in (
            "config.yaml",
            "features.yaml",
            "dataset_b/data_contract.yaml",
        ):
            with pytest.raises(FileNotFoundError):
                load_config(legacy_path)

    def test_configuration_entry_points_exist(self):
        assert PROJECT_CONFIG_PATH.is_file()
        assert CONFIG_CATALOG_PATH.is_file()

    def test_catalog_targets_exist_without_self_reference(self):
        catalog = load_config("dataset_b/catalog.yaml")
        assert catalog.dataset == "dataset_b"

        targets = []
        for group_name in ("contracts", "models", "research", "runtime"):
            targets.extend(catalog[group_name].values())

        assert len(targets) == len(set(targets))
        assert "config/dataset_b/catalog.yaml" not in targets
        for target in targets:
            assert (get_project_root() / target).is_file(), target


class TestEnvVarInterpolation:
    def test_env_var_default_casting(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(
                """
                test_int: ${MY_PORT:8080}
                test_bool_true: ${MY_BOOL:true}
                test_bool_false: ${MY_BOOL_F:FALSE}
                test_string: ${MY_HOST:localhost}
                test_embedded: http://${MY_HOST:localhost}:${MY_PORT:8080}/api
                """
            )
            temp_path = f.name

        try:
            cfg = load_config(temp_path)
            assert cfg.test_int == 8080
            assert cfg.test_bool_true is True
            assert cfg.test_bool_false is False
            assert cfg.test_string == "localhost"
            assert cfg.test_embedded == "http://localhost:8080/api"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_DB_PORT", "9999")
        monkeypatch.setenv("CUSTOM_DB_HOST", "db.production.internal")

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(
                """
                db:
                  host: ${CUSTOM_DB_HOST:localhost}
                  port: ${CUSTOM_DB_PORT:5432}
                """
            )
            temp_path = f.name

        try:
            cfg = load_config(temp_path)
            assert cfg.db.host == "db.production.internal"
            assert cfg.db.port == 9999
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

