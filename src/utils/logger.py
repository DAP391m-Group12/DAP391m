import json
import os
import sys
import datetime
import platform
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.path import PROJECT_ROOT

# Hỗ trợ khóa file cấp hệ điều hành (Process-safe)
if os.name == 'nt': # Windows
    import msvcrt
else: # Linux/Mac
    try:
        import fcntl
    except ImportError:
        fcntl = None

class ProductionLogger:
    # Định nghĩa mức độ ưu tiên để lọc Log
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40
    }

    def __init__(self, 
                 log_file="app_logs.jsonl", 
                 min_level="INFO", 
                 max_bytes=5 * 1024 * 1024, # 5MB
                 backup_count=3,
                 mask_pii=True):
        """
        Khởi tạo Logger chuẩn Production.
        - min_level: Bỏ qua các log thấp hơn mức này (Giảm tải I/O).
        - max_bytes, backup_count: Tự động cắt file khi quá 5MB, giữ tối đa 3 file cũ.
        - mask_pii: Che giấu thông tin cá nhân (Email).
        """
        normalized_level = min_level.upper()
        if normalized_level not in self.LEVELS:
            raise ValueError(f"Unknown log level: {min_level}")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if backup_count < 0:
            raise ValueError("backup_count must not be negative")

        self.min_level = self.LEVELS[normalized_level]
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.mask_pii = mask_pii
        
        # Giải quyết [LOW] Đường dẫn tương đối: Tự động lấy đường dẫn tuyệt đối
        self.log_file = os.path.abspath(log_file)
        
        # Đảm bảo thư mục chứa file log tồn tại
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        self.lock_file = f"{self.log_file}.lock"
        
        self.thread_lock = threading.Lock()
        
        # Chỉ lấy thông tin môi trường 1 lần lúc khởi động
        self.machine_info = {
            "os": platform.system(),
            "computer_name": platform.node()
        }
        self.git_info = self._get_git_user_info()

    def _get_git_user_info(self):
        """Lấy thông tin Git và tự động che giấu (Masking) nếu cần"""
        git_info = {"name": "Unknown", "email": "Unknown"}
        try:
            name = subprocess.check_output(["git", "config", "user.name"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            email = subprocess.check_output(["git", "config", "user.email"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            
            git_info["name"] = name
            # Giải quyết [HIGH] Rò rỉ PII: Che giấu email (VD: nguyen***@gmail.com)
            if self.mask_pii and "@" in email:
                user, domain = email.split("@", 1)
                git_info["email"] = f"***@{domain}"
            else:
                git_info["email"] = email
        except Exception:
            git_info["name"] = "No Git"
            git_info["email"] = "No Git"
            
        return git_info

    def _get_caller_info(self):
        """
        Giải quyết [MEDIUM] Hiệu năng: Dùng sys._getframe thay vì inspect.stack().
        Nhanh hơn gấp ~20 lần, không làm chậm chương trình.
        """
        try:
            # 0: Hàm này, 1: Hàm _write_log, 2: info/error, 3: Nơi gọi thực tế
            frame = sys._getframe(3)
            filename = os.path.basename(frame.f_code.co_filename)
            return f"{filename}:{frame.f_lineno}"
        except (ValueError, AttributeError):
            return "unknown:0"

    def _safe_json_fallback(self, obj):
        """Xử lý khi object không thể chuyển sang JSON"""
        try:
            return str(obj)
        except Exception:
            return "<Unserializable>"

    def _rotate_logs_if_needed(self):
        """Giải quyết [MEDIUM] Phình to ổ cứng: Tự động đổi tên file khi quá dung lượng"""
        if not os.path.exists(self.log_file):
            return

        if os.path.getsize(self.log_file) >= self.max_bytes:
            if self.backup_count == 0:
                os.remove(self.log_file)
                return

            # Xóa file cũ nhất nếu đã vượt backup_count
            oldest_log = f"{self.log_file}.{self.backup_count}"
            if os.path.exists(oldest_log):
                os.remove(oldest_log)

            # Đẩy các file cũ lùi lại (vd: file.1 thành file.2)
            for i in range(self.backup_count - 1, 0, -1):
                src = f"{self.log_file}.{i}"
                dst = f"{self.log_file}.{i + 1}"
                if os.path.exists(src):
                    os.rename(src, dst)

            # File hiện tại thành file.1
            os.rename(self.log_file, f"{self.log_file}.1")

    @contextmanager
    def _process_lock(self):
        """Khóa một file riêng để bảo vệ rotation và ghi giữa các process."""
        lock_file = open(self.lock_file, "a+b")
        try:
            if os.name == "nt":
                lock_file.seek(0)
                if not os.path.getsize(self.lock_file):
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            elif fcntl:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def _write_log(self, level_name, action, result):
        """Hàm lõi được thiết kế để không bao giờ làm Crash ứng dụng"""
        # 1. Bỏ qua nếu cấu hình log nhỏ hơn mức yêu cầu (Ví dụ: Đang để INFO thì bỏ qua DEBUG)
        level_value = self.LEVELS.get(level_name, 0)
        if level_value < self.min_level:
            return

        # Đưa vào Try-Except tổng để bắt mọi lỗi phát sinh từ logger
        try:
            log_entry = {
                # Giải quyết [MEDIUM] Timezone: Dùng chuẩn quốc tế ISO 8601 có UTC (Z)
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "level": level_name,
                "file": self._get_caller_info(),
                "action": action,
                "result": result,
                "context": {
                    "machine": self.machine_info,
                    "git": self.git_info
                }
            }

            # Giải quyết [LOW] Crash do dữ liệu lạ: Dùng default=fallback
            json_string = json.dumps(log_entry, ensure_ascii=False, default=self._safe_json_fallback)

            with self.thread_lock: # Khóa luồng (Thread-safe)
                with self._process_lock():
                    self._rotate_logs_if_needed() # Cắt file nếu quá nặng

                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json_string + "\n")
                        f.flush()          # Ép ghi xuống ổ cứng
                    
        except Exception as e:
            # Tối kỵ: Logger không được làm chết app. Nếu lỗi, in ra console.
            sys.stderr.write(f"[LOGGER ERROR] Failed to write log: {str(e)}\n")

    # --- CÁC HÀM API GỌI LOG ---
    def debug(self, action, result=""):
        self._write_log("DEBUG", action, result)

    def info(self, action, result=""):
        self._write_log("INFO", action, result)

    def warning(self, action, result=""):
        self._write_log("WARNING", action, result)

    def error(self, action, result=""):
        self._write_log("ERROR", action, result)

_DEFAULT_LOGGING_SETTINGS = {
    "log_file": "app_logs.jsonl",
    "min_level": "INFO",
    "max_bytes": 5 * 1024 * 1024,
    "backup_count": 3,
    "mask_pii": True,
}


def _load_logger_settings():
    """Read shared logging defaults and let LOG_LEVEL override the level locally."""
    settings = _DEFAULT_LOGGING_SETTINGS.copy()
    try:
        logging_config = load_config().get("logging", {})
        if isinstance(logging_config, dict):
            field_map = {
                "log_file": "log_file",
                "default_level": "min_level",
                "max_bytes": "max_bytes",
                "backup_count": "backup_count",
                "mask_pii": "mask_pii",
            }
            for config_key, setting_key in field_map.items():
                if config_key in logging_config:
                    settings[setting_key] = logging_config[config_key]
    except Exception as error:
        sys.stderr.write(f"[LOGGER WARNING] Could not load logging config: {error}\n")

    level_override = os.environ.get("LOG_LEVEL")
    if level_override:
        settings["min_level"] = level_override

    log_path = Path(settings["log_file"])
    if not log_path.is_absolute():
        settings["log_file"] = str(PROJECT_ROOT / log_path)
    return settings


def _build_default_logger():
    """Keep an invalid optional config from preventing the application import."""
    try:
        return ProductionLogger(**_load_logger_settings())
    except Exception as error:
        sys.stderr.write(f"[LOGGER WARNING] Falling back to built-in defaults: {error}\n")
        return ProductionLogger(**_DEFAULT_LOGGING_SETTINGS)


logger = _build_default_logger()
