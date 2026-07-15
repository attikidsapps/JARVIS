"""
jarvis/config/logging_config.py

Compatibility shim for logging initialisation.

The canonical logging implementation lives in ``utils/logger.py``,
which sets up the colourised console handler, the rotating file
handler, and the "jarvis" namespace logger with full idempotency
protection.

This file exists because ``config/`` is the natural first place
anyone looks for logging configuration. Rather than duplicating
logic here (which would produce double handlers, conflicting log
levels, and duplicate console output), this module simply maps
its own public API to the canonical implementation.

Rule: ALL logging logic lives in utils/logger.py.
      This file ONLY delegates — it contains zero handler setup,
      zero formatter definitions, and zero logger instantiation.

Usage (the only correct call site is main.py):
    from jarvis.config.logging_config import setup_logging
    setup_logging(log_level="DEBUG", log_dir="jarvis/logs")

Do NOT call both setup_logging() and configure_logging() in the
same process. Use one or the other; they resolve to the same
underlying call and the idempotency guard in configure_logging()
will silently no-op the second invocation anyway, but calling
both is a code smell that signals confusion about ownership.
"""

from __future__ import annotations

import logging
from pathlib import Path

from utils.logger import configure_logging

__all__ = ["setup_logging"]


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "jarvis/logs",
) -> None:
    """Configure JARVIS logging by delegating to utils/logger.py.

    This is a convenience wrapper that accepts the string-based
    level names used in settings.yaml and converts them to the
    integer constants expected by ``configure_logging()``.

    Args:
        log_level: String log level name applied to both the console
            and file handlers. One of: DEBUG | INFO | WARNING |
            ERROR | CRITICAL. Defaults to "INFO".
        log_dir: Directory path (relative to CWD or absolute) where
            ``jarvis.log`` will be written. Created automatically if
            it does not exist. Defaults to "jarvis/logs".

    Raises:
        ValueError: If ``log_level`` is not a recognised level name.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: {log_level!r}. "
            f"Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    configure_logging(
        log_dir=Path(log_dir),
        console_level=numeric_level,
        file_level=numeric_level,
        root_level=numeric_level,
    )