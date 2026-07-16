"""
jarvis/security/__init__.py

Public API surface for the JARVIS security package.

Re-exports all symbols needed by automation modules, the communication
layer, and the orchestrator so that callers use stable import paths.

Correct usage anywhere in JARVIS:
    from security import PermissionManager, PermissionDeniedError
    from security import RiskLevel, PermissionDecision, PermissionRequest
    from security import InputValidator, ValidationError
"""

from security.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionManager,
    PermissionRequest,
    RiskLevel,
)
from security.validators import InputValidator, ValidationError

__all__ = [
    # permissions.py
    "PermissionManager",
    "PermissionDeniedError",
    "PermissionRequest",
    "PermissionDecision",
    "RiskLevel",
    # validators.py
    "InputValidator",
    "ValidationError",
]