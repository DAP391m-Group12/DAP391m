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
        assert cfg.lop.host == "localhost"
        assert cfg.lop.port == 5432

    def test_load_by_filename(self):
        cfg = load_config("config.yaml")
        assert cfg.lop.database == "mydb"

    def test_load_from_relative_path(self):
        cfg = load_config("config/config.yaml")
        assert cfg.lop.port == 5432

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config_12345.yaml")


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


class TestSaveModelConfig:
    def test_save_model_config_appends_history(self):
        from src.utils.config_loader import save_model_config
        import yaml
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name
            
        try:
            # Lần chạy 1
            save_model_config(
                "LogisticRegression",
                training_time=1.23,
                config_path=temp_path,
                C=1.5,
                max_iter=500,
            )
            # Lần chạy 2
            save_model_config(
                "RandomForest",
                training_time=5.67,
                config_path=temp_path,
                n_estimators=100,
            )
            
            with open(temp_path, "r", encoding="utf-8") as f:
                history = yaml.safe_load(f)
                
            assert isinstance(history, list)
            assert len(history) == 2
            
            assert history[0]["model_name"] == "LogisticRegression"
            assert history[0]["training_time"] == "1.23s"
            assert history[0]["hyperparameters"]["C"] == 1.5
            
            assert history[1]["model_name"] == "RandomForest"
            assert history[1]["training_time"] == "5.67s"
            assert history[1]["hyperparameters"]["n_estimators"] == 100
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
