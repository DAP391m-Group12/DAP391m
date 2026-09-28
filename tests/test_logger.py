from pathlib import Path

import src.utils.logger as logger_module
from src.utils.config_loader import Config
from src.utils.path import PROJECT_ROOT


def test_default_logger_uses_project_logging_config():
    assert Path(logger_module.logger.log_file) == PROJECT_ROOT / "logs" / "app_logs.jsonl"
    assert logger_module.logger.max_bytes == 5 * 1024 * 1024
    assert logger_module.logger.backup_count == 3
    assert logger_module.logger.mask_pii is True


def test_logger_settings_use_shared_config(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(
        logger_module,
        "load_config",
        lambda: Config(
            {
                "logging": {
                    "log_file": "logs/test.jsonl",
                    "default_level": "WARNING",
                    "max_bytes": 1024,
                    "backup_count": 1,
                    "mask_pii": False,
                }
            }
        ),
    )

    settings = logger_module._load_logger_settings()

    assert settings == {
        "log_file": str(PROJECT_ROOT / Path("logs/test.jsonl")),
        "min_level": "WARNING",
        "max_bytes": 1024,
        "backup_count": 1,
        "mask_pii": False,
    }


def test_log_level_environment_variable_overrides_shared_config(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setattr(
        logger_module,
        "load_config",
        lambda: Config({"logging": {"default_level": "INFO"}}),
    )

    assert logger_module._load_logger_settings()["min_level"] == "ERROR"
