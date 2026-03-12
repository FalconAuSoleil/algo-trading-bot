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

_LOG_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


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
        # Force UTF-8 on Windows to avoid cp1252 UnicodeEncodeError
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

    # File handler with UTF-8 encoding
    try:
        log_dir = Path("data")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "sniper.log", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        pass

    logger.propagate = False
    return logger


log = setup_logger()
