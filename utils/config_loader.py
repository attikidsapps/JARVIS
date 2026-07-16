"""
jarvis/utils/config_loader.py

Typed, validated settings loader for JARVIS.

Loads ``config/settings.yaml`` once at startup, merges any
environment-variable overrides, and exposes the fully-resolved
configuration as a typed ``Settings`` object. Every subsystem receives
a typed sub-config (e.g. ``DatabaseSettings``) rather than raw dicts,
so attribute access is IDE-friendly and typos in key names fail loudly
at startup rather than silently at runtime.

Override mechanism:
    Environment variables (from the shell or .env, loaded by
    python-dotenv in main.py) can override any YAML key using:

        JARVIS__<SECTION>__<KEY>=value

    Examples:
        JARVIS__LLM__MODEL=llama3          overrides llm.model
        JARVIS__DATABASE__PATH=/tmp/j.db   overrides database.path

    Values are coerced to the declared Python type of the target field.
    Boolean coercion treats "true"/"1"/"yes" as True (case-insensitive).

NOTE: This file intentionally does NOT use `from __future__ import
annotations`. That import activates PEP 563 lazy annotation evaluation,
which converts all annotations to strings at runtime. _populate_dataclass
relies on runtime type inspection via typing.get_type_hints(), which
re-resolves those strings — but only if the necessary names are in scope
at resolution time. To avoid that complexity and keep the implementation
straightforward, lazy annotations are disabled here. There are no forward
references in this file that would require them.

Dependencies:
    pyyaml   -- parses the YAML config file
    python-dotenv -- (used in main.py, not here) loads .env before
                     this module is imported
"""

# NOTE: No `from __future__ import annotations` here — see module docstring.

import dataclasses
import logging
import os
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union, get_type_hints

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc

__all__ = [
    "ConfigurationError",
    "Settings",
    "LLMSettings",
    "PersonalitySettings",
    "WakeWordSettings",
    "SpeechRecognitionSettings",
    "TextToSpeechSettings",
    "ShortTermMemorySettings",
    "LongTermMemorySettings",
    "MemorySettings",
    "DatabaseSettings",
    "PermissionsSettings",
    "BrowserSettings",
    "FileManagerSettings",
    "MouseKeyboardSettings",
    "AutomationSettings",
    "LoggingSettings",
    "load_settings",
]

logger = logging.getLogger("jarvis.utils.config_loader")

_DEFAULT_CONFIG_PATH = Path("config/settings.yaml")
_ENV_PREFIX = "JARVIS"
_ENV_SEPARATOR = "__"

T = TypeVar("T")


class ConfigurationError(Exception):
    """Raised when the configuration is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Typed sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    model: str = "dolphin-llama3"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: float = 120.0


@dataclass
class PersonalitySettings:
    assistant_name: str = "JARVIS"
    verbosity: str = "concise"
    user_name: Optional[str] = None


@dataclass
class WakeWordSettings:
    keyword: str = "jarvis"
    sensitivity: float = 0.5
    device_index: Optional[int] = None


@dataclass
class SpeechRecognitionSettings:
    engine: str = "faster_whisper"
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    silence_timeout_seconds: float = 1.5
    max_record_seconds: int = 30
    sample_rate: int = 16000
    chunk_size: int = 1024


@dataclass
class TextToSpeechSettings:
    engine: str = "pyttsx3"
    rate: int = 185
    volume: float = 1.0
    voice_id: Optional[str] = None


@dataclass
class ShortTermMemorySettings:
    max_turns: int = 20
    max_chars: int = 12000


@dataclass
class LongTermMemorySettings:
    max_facts_in_prompt: int = 10
    relevance_threshold: float = 0.3


@dataclass
class MemorySettings:
    short_term: ShortTermMemorySettings = field(default_factory=ShortTermMemorySettings)
    long_term: LongTermMemorySettings = field(default_factory=LongTermMemorySettings)


@dataclass
class DatabaseSettings:
    path: str = "data/jarvis.db"
    timeout_seconds: int = 30
    wal_mode: bool = True


@dataclass
class PermissionsSettings:
    trust_window_seconds: int = 300
    confirmation_timeout_seconds: float = 60.0
    audit_log_path: str = "logs/permissions_audit.jsonl"


@dataclass
class BrowserSettings:
    executable: str = "chrome"
    headless: bool = False


@dataclass
class FileManagerSettings:
    forbidden_paths: list = field(default_factory=lambda: [
        "C:\\Windows",
        "C:\\Windows\\System32",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
    ])
    max_read_bytes: int = 10 * 1024 * 1024  # 10 MB


@dataclass
class MouseKeyboardSettings:
    type_interval_seconds: float = 0.03
    action_pause_seconds: float = 0.1


@dataclass
class AutomationSettings:
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    file_manager: FileManagerSettings = field(default_factory=FileManagerSettings)
    mouse_keyboard: MouseKeyboardSettings = field(default_factory=MouseKeyboardSettings)


@dataclass
class LoggingSettings:
    console_level: str = "DEBUG"
    file_level: str = "DEBUG"
    log_dir: str = "logs"


@dataclass
class Settings:
    """Root settings object. Passed by reference through the JARVIS process.

    Every subsystem that needs configuration accepts a ``Settings``
    instance in its constructor rather than reaching into global state
    or re-reading the file, keeping dependencies explicit and testable.
    """

    llm: LLMSettings = field(default_factory=LLMSettings)
    personality: PersonalitySettings = field(default_factory=PersonalitySettings)
    wake_word: WakeWordSettings = field(default_factory=WakeWordSettings)
    speech_recognition: SpeechRecognitionSettings = field(default_factory=SpeechRecognitionSettings)
    text_to_speech: TextToSpeechSettings = field(default_factory=TextToSpeechSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    permissions: PermissionsSettings = field(default_factory=PermissionsSettings)
    automation: AutomationSettings = field(default_factory=AutomationSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_optional(tp: Any) -> tuple[bool, Any]:
    """Return (is_optional, inner_type) for a type annotation.

    Handles Optional[X] = Union[X, None] and bare types equally.
    Works with real runtime type objects (not PEP-563 strings).
    """
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())

    # Union[X, None]  (which is what Optional[X] desugars to)
    if origin is Union and args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return True, non_none[0]
        # Union of multiple non-None types — rare in this codebase
        return True, non_none[0] if non_none else (True, Any)

    return False, tp


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a raw value (from YAML or env-var string) to ``target_type``.

    Handles bool, int, float, str, list, and Optional[X].
    Nested dataclasses are handled by _populate_dataclass — do not
    pass them here directly.

    Args:
        value: The raw value to coerce.
        target_type: A real runtime type object (not a string annotation).

    Returns:
        The coerced value.

    Raises:
        ConfigurationError: If coercion is not possible.
    """
    # Unwrap Optional[X] → X
    is_opt, inner = _unwrap_optional(target_type)
    if is_opt:
        if value is None:
            return None
        target_type = inner

    if value is None:
        return None

    origin = getattr(target_type, "__origin__", None)

    # list / list[str] / list[X]
    if origin is list:
        if isinstance(value, list):
            return value
        # env-var override: comma-separated string
        return [item.strip() for item in str(value).split(",")]

    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")

    if target_type is int:
        return int(value)

    if target_type is float:
        return float(value)

    if target_type is str:
        return str(value)

    # Fallback: return as-is and let callers fail loudly if wrong
    return value


def _populate_dataclass(cls: Type[T], data: dict[str, Any]) -> T:
    """Recursively populate a dataclass from a dict.

    Uses typing.get_type_hints() to resolve annotations to real type
    objects at runtime. This is safe because this module does NOT use
    `from __future__ import annotations`, so annotations are already
    real type objects — get_type_hints() is used as an extra safety
    layer that would also work if the import were re-added later.

    Fields absent from ``data`` retain their dataclass defaults.
    Fields whose declared type is itself a dataclass are recursively
    populated from the corresponding nested dict.

    Args:
        cls: The dataclass type to instantiate.
        data: A dict whose keys map to field names of ``cls``.

    Returns:
        A fully populated instance of ``cls``.

    Raises:
        ConfigurationError: If a field value cannot be coerced.
    """
    if not dataclasses.is_dataclass(cls):
        raise ConfigurationError(
            f"_populate_dataclass called on non-dataclass: {cls}"
        )

    # Resolve all annotations to real type objects.
    # include_extras=False strips Annotated[...] wrappers if present.
    try:
        hints = get_type_hints(cls)
    except Exception as exc:
        raise ConfigurationError(
            f"Could not resolve type hints for {cls.__name__}: {exc}"
        ) from exc

    kwargs: dict[str, Any] = {}

    for f in fields(cls):  # type: ignore[arg-type]
        raw_value = data.get(f.name)

        # Get the resolved runtime type for this field.
        field_type = hints.get(f.name, Any)

        # Unwrap Optional to get the inner type for the is_dataclass check.
        _, inner_type = _unwrap_optional(field_type)

        if raw_value is not None and dataclasses.is_dataclass(inner_type) and isinstance(raw_value, dict):
            # Recursively hydrate nested dataclasses.
            kwargs[f.name] = _populate_dataclass(inner_type, raw_value)  # type: ignore[arg-type]
        elif raw_value is not None:
            try:
                kwargs[f.name] = _coerce(raw_value, field_type)
            except (ValueError, TypeError) as exc:
                raise ConfigurationError(
                    f"Cannot coerce config value {raw_value!r} for field "
                    f"'{cls.__name__}.{f.name}' to type {field_type}: {exc}"
                ) from exc
        # else: field absent from YAML — use dataclass default

    return cls(**kwargs)  # type: ignore[return-value]


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Scan environment variables for JARVIS__<SECTION>__<KEY> overrides.

    Supports two-level (section.key) and three-level
    (section.subsection.key) nesting. Mutates and returns ``raw``.
    """
    prefix = f"{_ENV_PREFIX}{_ENV_SEPARATOR}"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        remainder = env_key[len(prefix):]
        parts = remainder.split(_ENV_SEPARATOR)

        if len(parts) == 2:
            section, key = parts[0].lower(), parts[1].lower()
            if section in raw and isinstance(raw[section], dict):
                raw[section][key] = env_value
                logger.debug(
                    "Config override from env: %s.%s = %r", section, key, env_value
                )
        elif len(parts) == 3:
            section, subsection, key = (
                parts[0].lower(), parts[1].lower(), parts[2].lower()
            )
            if (
                section in raw
                and isinstance(raw[section], dict)
                and subsection in raw[section]
                and isinstance(raw[section][subsection], dict)
            ):
                raw[section][subsection][key] = env_value
                logger.debug(
                    "Config override from env: %s.%s.%s = %r",
                    section, subsection, key, env_value,
                )
        else:
            logger.warning(
                "Ignoring unrecognised JARVIS env override: %s", env_key
            )

    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load, validate, and return the JARVIS settings.

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``config/settings.yaml`` relative to CWD.

    Returns:
        A fully populated and validated ``Settings`` instance.

    Raises:
        ConfigurationError: If the file cannot be read, the YAML is
            malformed, or a field fails type coercion.
    """
    resolved_path = config_path or _DEFAULT_CONFIG_PATH

    if not resolved_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {resolved_path.resolve()}\n"
            "Ensure you are running JARVIS from the project root directory "
            "and that config/settings.yaml exists."
        )

    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse configuration file {resolved_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read configuration file {resolved_path}: {exc}"
        ) from exc

    raw = _apply_env_overrides(raw)

    try:
        settings = _populate_dataclass(Settings, raw)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Unexpected error while building Settings from config: {exc}"
        ) from exc

    _validate(settings)

    logger.info(
        "Configuration loaded from %s — model: %s | verbosity: %s",
        resolved_path.resolve(),
        settings.llm.model,
        settings.personality.verbosity,
    )
    return settings


def _validate(settings: Settings) -> None:
    """Run semantic validation rules that go beyond type coercion.

    Raises:
        ConfigurationError: On any invalid combination or out-of-range value.
    """
    valid_verbosity = {"concise", "standard", "detailed"}
    if settings.personality.verbosity not in valid_verbosity:
        raise ConfigurationError(
            f"personality.verbosity must be one of {valid_verbosity}, "
            f"got '{settings.personality.verbosity}'"
        )

    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    for attr, label in (
        (settings.logging.console_level, "logging.console_level"),
        (settings.logging.file_level, "logging.file_level"),
    ):
        if attr.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"{label} must be one of {valid_log_levels}, got '{attr}'"
            )

    if not 0.0 <= settings.wake_word.sensitivity <= 1.0:
        raise ConfigurationError(
            f"wake_word.sensitivity must be between 0.0 and 1.0, "
            f"got {settings.wake_word.sensitivity}"
        )

    if not 0.0 <= settings.llm.temperature <= 1.0:
        raise ConfigurationError(
            f"llm.temperature must be between 0.0 and 1.0, "
            f"got {settings.llm.temperature}"
        )

    if not 0.0 <= settings.text_to_speech.volume <= 1.0:
        raise ConfigurationError(
            f"text_to_speech.volume must be between 0.0 and 1.0, "
            f"got {settings.text_to_speech.volume}"
        )

    if settings.memory.short_term.max_turns < 1:
        raise ConfigurationError(
            "memory.short_term.max_turns must be at least 1"
        )

    if settings.llm.max_tokens is not None and settings.llm.max_tokens < 1:
        raise ConfigurationError(
            "llm.max_tokens must be null or a positive integer"
        )