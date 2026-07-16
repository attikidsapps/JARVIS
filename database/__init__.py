"""
jarvis/database/__init__.py

Public API surface for the JARVIS database package.

Re-exports the symbols needed by all JARVIS subsystems that interact
with persistent storage, so callers use stable import paths regardless
of internal module reorganisation.

Correct usage anywhere in JARVIS:
    from database import DatabaseManager, DatabaseError
    from database.models import LongTermFact, ConversationSession
"""

from database.db_manager import DatabaseError, DatabaseManager

__all__ = [
    "DatabaseManager",
    "DatabaseError",
]