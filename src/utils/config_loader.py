from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
import yaml

from src.utils.path import CONFIG_DIR, PROJECT_ROOT


class Config(dict):
    """Lớp cấu hình cho phép truy cập trực tiếp bằng dấu chấm (config.database.host)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for k, v in list(dict.items(self)):
            if isinstance(v, dict):
                dict.__setitem__(self, k, Config(v))
            elif isinstance(v, list):
                dict.__setitem__(self, k, [Config(i) if isinstance(i, dict) else i for i in v])

    def __getattribute__(self, name: str) -> Any:
        if name != "__dict__" and dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        return super().__getattribute__(name)

    def __getattr__(self, key: str) -> Any:
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            raise AttributeError(f"Không tìm thấy cấu hình: '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        dict.__setitem__(self, key, value)

    def __repr__(self) -> str:
        return f"Config({dict.__repr__(self)})"


# Alias để tương thích
ConfigDict = Config
get_project_root = lambda: PROJECT_ROOT


def _env_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    val = loader.construct_scalar(node)
    if not isinstance(val, str):
        return val

    pattern = re.compile(r"\${([A-Za-z0-9_]+)(?::([^}]*))?}")
    match = pattern.fullmatch(val)
    if match:
        var, default = match.groups()
        res = os.getenv(var, default if default is not None else "")
        if res.lower() == "true":
            return True
        if res.lower() == "false":
            return False
        if res.isdigit():
            return int(res)
        return res

    return pattern.sub(lambda m: os.getenv(m.group(1), m.group(2) if m.group(2) is not None else ""), val)


class _YamlLoader(yaml.SafeLoader):
    pass


_YamlLoader.add_implicit_resolver("!env", re.compile(r".*?\${[A-Za-z0-9_]+(?::[^}]*)?}.*", re.DOTALL), None)
_YamlLoader.add_constructor("!env", _env_constructor)


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Đọc file YAML và trả về object Config truy cập dạng thuộc tính (dấu chấm)."""
    path = Path(config_path)

    candidates = [
        path,
        CONFIG_DIR / path.name,
        CONFIG_DIR / f"{path.name}.yaml",
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                return Config(yaml.load(f, Loader=_YamlLoader) or {})

    raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

def save_model_config(
    model_name: str,
    training_time: float | str | None = None,
    config_path: str | Path = "model_config.yaml",
    **hyperparameters: Any,
) -> Path:
    """
    Lưu cấu hình mô hình (tên model, thời gian, siêu tham số) vào lịch sử file YAML.
    Nếu file đã tồn tại, hàm sẽ tự động thêm (append) lượt chạy mới vào danh sách.
    
    Ví dụ:
        save_model_config("LogisticRegression", training_time=1.25, C=1.0, max_iter=200)
    """
    from datetime import datetime
    
    now = datetime.now()
    run_entry: dict[str, Any] = {
        "run_id": f"{model_name}_{now.strftime('%Y%m%d_%H%M%S')}",
        "model_name": model_name,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if training_time is not None:
        run_entry["training_time"] = f"{training_time:.2f}s" if isinstance(training_time, (int, float)) else str(training_time)
        
    run_entry["hyperparameters"] = hyperparameters
    
    def _to_dict(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(v) for v in obj]
        return obj

    path = Path(config_path)
    if not path.is_absolute() and len(path.parts) == 1:
        path = CONFIG_DIR / path
    elif not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    # Đọc lịch sử cũ nếu file đã tồn tại
    history: list[Any] = []
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, list):
                    history = content
                elif isinstance(content, dict):
                    history = [content]
        except Exception:
            history = []

    history.append(_to_dict(run_entry))

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(history, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return path