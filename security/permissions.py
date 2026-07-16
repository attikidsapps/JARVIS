"""
jarvis/security/permissions.py

Centralized permission and authorization gate for JARVIS.

This module sits between Tool Selection and Tool Execution in the
JARVIS pipeline. No automation module (file_manager, browser_control,
app_control, mouse_keyboard, etc.) is permitted to execute a
destructive or sensitive action without first passing through
PermissionManager.authorize().

Design goals:
    - Fail closed: unknown or unclassified actions default to DENY
      until explicitly classified.
    - Every authorization decision (granted, denied, or timed out)
      is written to a persistent audit log.
    - Confirmation prompts are injected via a callback so this module
      stays decoupled from the voice/UI layer (no direct dependency
      on audio or Tkinter/PyQt).
    - Session-scoped trust windows reduce prompt fatigue for repeated
      low-risk actions without weakening protection on high-risk ones.

This module has zero external dependencies beyond the Python standard
library, so it carries no licensing or hosting cost.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

__all__ = [
    "RiskLevel",
    "PermissionDecision",
    "PermissionDeniedError",
    "PermissionRequest",
    "PermissionManager",
]

logger = logging.getLogger("jarvis.security.permissions")


class RiskLevel(Enum):
    """Risk classification for an automation action.

    Ordering matters: members are declared in ascending order of risk
    so that comparisons (e.g. `risk >= RiskLevel.HIGH`) behave
    intuitively when compared via `.value`.
    """

    SAFE = 0        # Read-only or fully reversible (e.g. list files, take screenshot)
    LOW = 1         # Reversible but modifies state (e.g. open an application)
    MEDIUM = 2      # Modifies user data but recoverable (e.g. create/rename a file)
    HIGH = 3        # Destructive or hard to reverse (e.g. delete file, close app with unsaved work)
    CRITICAL = 4    # Irreversible, external-facing, or financial (e.g. send email, purchase, enter password, system settings change)


class PermissionDecision(Enum):
    """Outcome of an authorization request."""

    GRANTED = auto()
    DENIED = auto()
    TIMED_OUT = auto()


class PermissionDeniedError(Exception):
    """Raised when an action is not authorized for execution.

    Callers in automation modules should catch this exception
    specifically -- it is a control-flow signal, not a bug indicator.
    """

    def __init__(self, action: str, reason: str) -> None:
        self.action = action
        self.reason = reason
        super().__init__(f"Action '{action}' denied: {reason}")


@dataclass
class PermissionRequest:
    """Represents a single request to perform a sensitive action.

    Attributes:
        action: Machine-readable action identifier, e.g. "file.delete".
        description: Human-readable description shown in confirmation
            prompts and audit logs, e.g. "Delete C:\\Users\\me\\report.docx".
        risk_level: The classified risk tier for this action.
        metadata: Arbitrary contextual data (file paths, recipients,
            amounts, etc.) preserved for audit purposes. Must be
            JSON-serializable.
        requested_at: UTC-aware timestamp of when the request was created.
        request_id: Unique identifier for correlating logs.
    """

    action: str
    description: str
    risk_level: RiskLevel
    metadata: dict = field(default_factory=dict)
    # FIX [WARNING 1]: use timezone-aware datetime to match the rest of the
    # codebase. datetime.utcnow() produces a naive datetime; appending "Z"
    # to its isoformat() is technically incorrect ISO-8601.
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class PermissionManager:
    """Authorization gate for all JARVIS automation actions.

    Usage:
        pm = PermissionManager(confirmation_callback=my_ui_confirm_fn)
        pm.register_action("file.delete", RiskLevel.HIGH)
        pm.register_action("file.list", RiskLevel.SAFE)

        try:
            pm.authorize(
                action="file.delete",
                description=f"Delete {path}",
                metadata={"path": str(path)},
            )
        except PermissionDeniedError as exc:
            logger.warning("Blocked: %s", exc)
            return

        # proceed with the actual deletion

    The `confirmation_callback` must be a callable accepting a
    PermissionRequest and returning a bool (True = user approved).
    If no callback is supplied, JARVIS defaults to auto-denying
    every action at MEDIUM risk or above, which effectively disables
    automation for anything but read-only operations -- this is the
    intentional fail-closed default for headless/unattended contexts.
    """

    #: Only RiskLevel.SAFE (value=0) is below this threshold. SAFE actions
    #: are auto-approved without any confirmation prompt or trust-window check.
    #: LOW and above always go through the confirmation path.
    ALWAYS_ALLOW_BELOW: RiskLevel = RiskLevel.LOW

    #: Actions at or above this risk level ALWAYS require fresh
    #: confirmation, regardless of any active trust window.
    ALWAYS_CONFIRM_AT_OR_ABOVE: RiskLevel = RiskLevel.CRITICAL

    def __init__(
        self,
        confirmation_callback: Optional[Callable[[PermissionRequest], bool]] = None,
        audit_log_path: Optional[Path] = None,
        trust_window_seconds: int = 300,
        confirmation_timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the permission manager.

        Args:
            confirmation_callback: Function invoked to ask the user
                to approve a MEDIUM/HIGH risk action. Must return
                bool. Called synchronously on the calling thread.

                IMPORTANT — timeout behaviour: PermissionManager
                measures how long the callback takes after it returns.
                If the elapsed time exceeds confirmation_timeout_seconds,
                the decision is recorded as TIMED_OUT. This check is
                post-hoc; it does not interrupt a blocking callback.
                The callback itself is responsible for enforcing any
                real-time timeout (e.g. via a threading.Timer that
                closes a dialog). If the callback raises, the request
                is treated as DENIED.

            audit_log_path: Path to a JSONL file where every decision
                is appended. Defaults to jarvis/logs/permissions_audit.jsonl
                relative to the current working directory.
            trust_window_seconds: How long a granted MEDIUM-risk
                action's approval is remembered for that exact action
                signature before requiring re-confirmation. Set to 0
                to disable trust windows entirely.
            confirmation_timeout_seconds: Advisory timeout (seconds).
                See confirmation_callback note above.
        """
        self._registry: dict[str, RiskLevel] = {}
        self._confirmation_callback = confirmation_callback
        self._trust_window_seconds = trust_window_seconds
        self._confirmation_timeout_seconds = confirmation_timeout_seconds
        self._trusted_until: dict[str, float] = {}
        self._lock = Lock()

        self._audit_log_path = audit_log_path or Path("jarvis/logs/permissions_audit.jsonl")
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._register_default_actions()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_default_actions(self) -> None:
        """Seed the registry with the baseline JARVIS action set."""
        defaults: dict[str, RiskLevel] = {
            # Application control
            "app.launch": RiskLevel.LOW,
            "app.close": RiskLevel.MEDIUM,
            "app.list_running": RiskLevel.SAFE,

            # File system
            "file.read": RiskLevel.SAFE,
            "file.list": RiskLevel.SAFE,
            "file.search": RiskLevel.SAFE,
            "file.create": RiskLevel.MEDIUM,
            "file.edit": RiskLevel.MEDIUM,
            "file.rename": RiskLevel.MEDIUM,
            "file.delete": RiskLevel.HIGH,
            "directory.create": RiskLevel.MEDIUM,
            "directory.delete": RiskLevel.HIGH,
            "directory.list": RiskLevel.SAFE,

            # Input control
            "mouse.move": RiskLevel.SAFE,
            "mouse.click": RiskLevel.LOW,
            "mouse.double_click": RiskLevel.LOW,
            "mouse.right_click": RiskLevel.LOW,
            "keyboard.type": RiskLevel.MEDIUM,
            "keyboard.hotkey": RiskLevel.MEDIUM,

            # Browser
            "browser.open": RiskLevel.LOW,
            "browser.search": RiskLevel.SAFE,
            "browser.navigate": RiskLevel.LOW,
            "browser.form_submit": RiskLevel.HIGH,

            # Communication and sensitive operations
            "email.send": RiskLevel.CRITICAL,
            "email.read": RiskLevel.MEDIUM,
            "credentials.enter_password": RiskLevel.CRITICAL,
            "purchase.execute": RiskLevel.CRITICAL,
            "system.settings_change": RiskLevel.CRITICAL,
            "system.shutdown_or_restart": RiskLevel.CRITICAL,
        }
        self._registry.update(defaults)

    def register_action(self, action: str, risk_level: RiskLevel) -> None:
        """Register or override the risk classification for an action."""
        with self._lock:
            previous = self._registry.get(action)
            self._registry[action] = risk_level
        if previous is not None and previous != risk_level:
            logger.info(
                "Action '%s' risk level changed: %s -> %s",
                action, previous.name, risk_level.name,
            )

    def get_risk_level(self, action: str) -> RiskLevel:
        """Return the risk level for an action.

        Unregistered actions default to CRITICAL (fail closed).
        """
        return self._registry.get(action, RiskLevel.CRITICAL)

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def authorize(
        self,
        action: str,
        description: str,
        metadata: Optional[dict] = None,
    ) -> PermissionRequest:
        """Authorize an action, raising PermissionDeniedError if refused.

        Args:
            action: Registered action identifier, e.g. "file.delete".
            description: Human-readable summary for confirmation
                prompts and audit logs.
            metadata: Optional contextual data (paths, recipients,
                amounts). Must be JSON-serializable for audit logging;
                non-serializable values are coerced to str().

        Returns:
            The PermissionRequest object representing the approved
            request, for callers that want the request_id for
            correlated logging.

        Raises:
            PermissionDeniedError: If the action is denied, times out
                waiting on user confirmation, or no confirmation
                mechanism is available for an action that requires one.
        """
        risk_level = self.get_risk_level(action)
        request = PermissionRequest(
            action=action,
            description=description,
            risk_level=risk_level,
            metadata=self._sanitize_metadata(metadata or {}),
        )

        # FIX [WARNING 2]: Only RiskLevel.SAFE (value=0) passes this check.
        # LOW actions (value=1) are NOT auto-approved; they go through
        # confirmation. ALWAYS_ALLOW_BELOW = LOW means "below LOW", i.e. SAFE only.
        if risk_level.value < self.ALWAYS_ALLOW_BELOW.value:
            self._record_decision(request, PermissionDecision.GRANTED, "SAFE action — below confirmation threshold")
            return request

        # Check trust window for repeatable, non-critical actions.
        if risk_level.value < self.ALWAYS_CONFIRM_AT_OR_ABOVE.value:
            if self._is_trusted(action, request.metadata):
                self._record_decision(request, PermissionDecision.GRANTED, "within trust window")
                return request

        decision, reason = self._request_confirmation(request)

        if decision == PermissionDecision.GRANTED:
            self._record_decision(request, decision, reason)
            if risk_level.value < self.ALWAYS_CONFIRM_AT_OR_ABOVE.value:
                self._extend_trust(action, request.metadata)
            return request

        self._record_decision(request, decision, reason)
        raise PermissionDeniedError(action=action, reason=reason)

    def _request_confirmation(self, request: PermissionRequest) -> tuple[PermissionDecision, str]:
        """Invoke the confirmation callback and interpret its result.

        Note on timeout: The elapsed-time check below is post-hoc. It
        records TIMED_OUT only after the callback has already returned.
        The callback itself must implement any real-time interrupt
        (e.g. a dialog that auto-closes after N seconds). A callback
        that blocks for longer than confirmation_timeout_seconds will
        still receive a TIMED_OUT decision, but the user interaction
        has already concluded inside the callback.
        """
        if self._confirmation_callback is None:
            return (
                PermissionDecision.DENIED,
                "no confirmation mechanism configured; failing closed",
            )

        start = time.monotonic()
        try:
            approved = self._confirmation_callback(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Confirmation callback raised an exception for action '%s'", request.action)
            return (PermissionDecision.DENIED, f"confirmation callback error: {exc}")

        elapsed = time.monotonic() - start
        if elapsed > self._confirmation_timeout_seconds:
            logger.warning(
                "Confirmation for action '%s' took %.2fs, exceeding advisory timeout of %.2fs",
                request.action, elapsed, self._confirmation_timeout_seconds,
            )
            return (PermissionDecision.TIMED_OUT, "confirmation exceeded timeout")

        if approved is True:
            return (PermissionDecision.GRANTED, "user confirmed")
        return (PermissionDecision.DENIED, "user declined")

    # ------------------------------------------------------------------
    # Trust window management
    # ------------------------------------------------------------------

    def _trust_key(self, action: str, metadata: dict) -> str:
        """Build a stable key identifying a specific action+target pair."""
        try:
            metadata_repr = json.dumps(metadata, sort_keys=True, default=str)
        except (TypeError, ValueError):
            metadata_repr = str(metadata)
        return f"{action}::{metadata_repr}"

    def _is_trusted(self, action: str, metadata: dict) -> bool:
        if self._trust_window_seconds <= 0:
            return False
        key = self._trust_key(action, metadata)
        with self._lock:
            expiry = self._trusted_until.get(key)
        return expiry is not None and time.monotonic() < expiry

    def _extend_trust(self, action: str, metadata: dict) -> None:
        if self._trust_window_seconds <= 0:
            return
        key = self._trust_key(action, metadata)
        with self._lock:
            self._trusted_until[key] = time.monotonic() + self._trust_window_seconds

    def clear_trust(self) -> None:
        """Clear all active trust windows.

        Should be called at the start of each new conversation session
        so trust never silently carries over across unrelated sessions.
        """
        with self._lock:
            self._trusted_until.clear()

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Ensure metadata is JSON-serializable and strips obvious secrets."""
        redacted_keys = {"password", "token", "secret", "api_key", "credential"}
        sanitized: dict = {}
        for key, value in metadata.items():
            if key.lower() in redacted_keys:
                sanitized[key] = "***REDACTED***"
                continue
            try:
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                sanitized[key] = str(value)
        return sanitized

    def _record_decision(
        self,
        request: PermissionRequest,
        decision: PermissionDecision,
        reason: str,
    ) -> None:
        """Append a single decision record to the audit log and app log."""
        record = {
            "request_id": request.request_id,
            # requested_at is now timezone-aware; isoformat() produces
            # a correct +00:00 suffix. We normalise to the 'Z' form for
            # consistency with the rest of the JARVIS timestamp convention.
            "timestamp": request.requested_at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            "action": request.action,
            "description": request.description,
            "risk_level": request.risk_level.name,
            "decision": decision.name,
            "reason": reason,
            "metadata": request.metadata,
        }

        log_level = logging.INFO if decision == PermissionDecision.GRANTED else logging.WARNING
        logger.log(
            log_level,
            "Permission %s for action '%s' (risk=%s): %s",
            decision.name, request.action, request.risk_level.name, reason,
        )

        try:
            with self._lock:
                with self._audit_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("Failed to write permission audit record to %s", self._audit_log_path)

    # ------------------------------------------------------------------
    # Introspection / maintenance helpers
    # ------------------------------------------------------------------

    def recent_denials(self, limit: int = 20) -> list[dict]:
        """Return the most recent denied/timed-out entries from the audit log."""
        if not self._audit_log_path.exists():
            return []

        denials: list[dict] = []
        with self._audit_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("decision") in (PermissionDecision.DENIED.name, PermissionDecision.TIMED_OUT.name):
                    denials.append(record)

        return denials[-limit:]