"""Structured logging with Rich console output."""

from __future__ import annotations

import logging
import sys
from rich.logging import RichHandler
from rich.console import Console

console = Console()

_LOG_FORMAT = "%(message)s"


def setup_logger(
    name: str = "sniper",
    level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(rich_handler)

    file_handler = logging.FileHandler("data/sniper.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    )
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


log = setup_logger()
