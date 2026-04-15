"""Structured logging with Rich console output."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

_LOG_FORMAT  = "%(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ERR_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Single shared error-file handler, attached once to the root logger
_error_handler: logging.FileHandler | None = None


def _get_error_handler() -> logging.FileHandler | None:
    """Return (creating if needed) the singleton errors.txt handler."""
    global _error_handler
    if _error_handler is not None:
        return _error_handler
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        h = logging.FileHandler(log_dir / "errors.txt", encoding="utf-8")
        h.setLevel(logging.WARNING)          # WARNING + ERROR + CRITICAL
        h.setFormatter(logging.Formatter(_ERR_FORMAT))
        _error_handler = h
        return h
    except (OSError, PermissionError):
        return None


def setup_logger(
    name: str = "sniper",
    level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _HAS_RICH:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(rich_handler)
    else:
        if sys.platform == "win32":
            stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        else:
            stream = sys.stdout
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    # Full log — all levels
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        full_handler = logging.FileHandler(log_dir / "sniper.log", encoding="utf-8")
        full_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(full_handler)
    except (OSError, PermissionError):
        pass

    # Error log — WARNING and above only
    err_handler = _get_error_handler()
    if err_handler:
        logger.addHandler(err_handler)

    logger.propagate = False
    return logger


log = setup_logger()
