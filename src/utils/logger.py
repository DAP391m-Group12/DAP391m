"""Shared logging setup for the HMDA project."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.path import REPORTS_DIR


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def get_project_logger(name: str, log_filename: str | None = None) -> logging.Logger:
    """Return a consistently configured logger without duplicate handlers.

    With ``log_filename``, messages are written to the terminal and
    ``reports/logs/<log_filename>``. Repeated calls can add a requested file
    handler even when the logger was initially created for console output.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT)

    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_filename:
        log_dir = REPORTS_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (log_dir / log_filename).resolve()
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == log_path
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = RotatingFileHandler(
                log_path,
                encoding="utf-8",
                maxBytes=MAX_LOG_BYTES,
                backupCount=LOG_BACKUP_COUNT,
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
