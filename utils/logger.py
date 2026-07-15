"""
jarvis/utils/logger.py

Centralized logging configuration for JARVIS.

Sets up a structured, multi-handler logging pipeline that writes to
both the console (colourised for readability during development) and
a rotating file sink (for persistent diagnostics and post-mortem
analysis). All JARVIS modules obtain their loggers via the standard
`logging.getLogger("jarvis.<module>")` pattern; this module ensures
that every one of those loggers inherits a consistent format,
rotation policy, and level without any module having to repeat
boilerplate handler setup.

Design goals:
    - Single call to `configure_logging()` at process startup (in
      main.py) is all that is required. Every subsequent
      `logging.getLogger(...)` call in any module automatically
      inherits the configuration.
    - Console output is human-friendly with ANSI colour coding per
      level, making tail-reading during development fast and clear.
    - File output is plain-text, UTF-8, with full timestamps and
      module names, suitable for grep/awk and log aggregators.
    - Log files rotate at 5 MB with up to 5 backups retained,
      keeping disk usage bounded even during long-running sessions.
    - The root "jarvis" logger is the single aggregation point;
      third-party library noise is suppressed to WARNING by default.

This module has zero external dependencies beyond the Python standard
library.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "configure_logging",
    "get_logger",
]

# ---------------------------------------------------------------------------
# ANSI colour codes for console handler
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"

_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG:    "\033[36m",   # Cyan
    logging.INFO:     "\033[32m",   # Green
    logging.WARNING:  "\033[33m",   # Yellow
    logging.ERROR:    "\033[31m",   # Red
    logging.CRITICAL: "\033[35m",   # Magenta
}

_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotate at 5 MB, keep 5 backups
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5

# Third-party loggers that are overly verbose at DEBUG/INFO
_NOISY_LIBRARIES = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "charset_normalizer",
    "PIL",
    "pydub",
    "speechbrain",
    "pyaudio",
)


class _ColouredFormatter(logging.Formatter):
    """Formatter that prepends ANSI colour codes to the level name on terminals.

    Falls back to plain text automatically when stdout is not a TTY
    (e.g. when output is piped or redirected), so log files and CI
    output are never polluted with escape sequences.
    """

    def __init__(self, fmt: str, datefmt: str, use_colour: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_colour:
            return super().format(record)

        colour = _LEVEL_COLOURS.get(record.levelno, _RESET)
        original_levelname = record.levelname
        record.levelname = f"{colour}{_BOLD}{record.levelname}{_RESET}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted


def configure_logging(
    log_dir: Optional[Path] = None,
    console_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    root_level: int = logging.DEBUG,
) -> None:
    """Configure the JARVIS logging pipeline.

    Must be called exactly once, early in main.py, before any module
    instantiates a logger. Subsequent calls are idempotent -- handlers
    are not duplicated if the root "jarvis" logger already has them.

    Args:
        log_dir: Directory where ``jarvis.log`` is written. Defaults to
            ``jarvis/logs/`` relative to the current working directory.
            The directory is created if it does not exist.
        console_level: Minimum level emitted to stdout. Set to
            ``logging.INFO`` in production to reduce terminal noise;
            keep ``DEBUG`` during development.
        file_level: Minimum level written to the rotating log file.
            Keeping this at DEBUG provides a complete audit trail for
            post-mortem analysis even when console is set to INFO.
        root_level: Level set on the "jarvis" root logger itself.
            Should be the minimum of console_level and file_level so
            that neither handler silently drops records the other would
            have captured.
    """
    jarvis_logger = logging.getLogger("jarvis")

    # Idempotency guard: if handlers are already attached, do nothing.
    if jarvis_logger.handlers:
        return

    jarvis_logger.setLevel(root_level)

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    use_colour = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    console_handler.setFormatter(
        _ColouredFormatter(
            fmt=_CONSOLE_FORMAT,
            datefmt=_DATE_FORMAT,
            use_colour=use_colour,
        )
    )
    jarvis_logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Rotating file handler
    # ------------------------------------------------------------------
    resolved_log_dir = log_dir or Path("jarvis/logs")
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolved_log_dir / "jarvis.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(fmt=_FILE_FORMAT, datefmt=_DATE_FORMAT)
    )
    jarvis_logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Suppress third-party noise
    # ------------------------------------------------------------------
    for lib in _NOISY_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)

    jarvis_logger.info(
        "Logging initialised — console: %s | file: %s | path: %s",
        logging.getLevelName(console_level),
        logging.getLevelName(file_level),
        log_file.resolve(),
    )


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'jarvis' namespace.

    Convenience wrapper so modules can call::

        from jarvis.utils.logger import get_logger
        logger = get_logger(__name__)

    rather than importing ``logging`` directly. The returned logger
    inherits all handlers and levels from the root "jarvis" logger
    configured by ``configure_logging()``.

    Args:
        name: Typically ``__name__`` of the calling module, e.g.
            ``"jarvis.core.conversation_manager"``.

    Returns:
        A ``logging.Logger`` instance namespaced under ``"jarvis"``.
    """
    # If the caller already passes a fully-qualified jarvis.* name,
    # use it as-is. Otherwise, nest it under the jarvis namespace so
    # it inherits root handlers.
    if name.startswith("jarvis"):
        return logging.getLogger(name)
    return logging.getLogger(f"jarvis.{name}")