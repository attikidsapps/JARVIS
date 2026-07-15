"""
jarvis/database/models.py

SQLAlchemy ORM model definitions for all JARVIS persistent data.

This module is the single source of truth for the database schema.
Every table that JARVIS writes to or reads from is declared here as a
mapped dataclass. No raw SQL strings live anywhere else in the project;
all schema changes are made here and propagated via Alembic migrations.

Tables defined:
    ConversationSession  -- one row per JARVIS session (wake → sleep cycle)
    ConversationTurn     -- one row per user/assistant message exchange
    LongTermFact         -- user preferences and remembered facts
    ScheduledReminder    -- APScheduler-persisted reminders and tasks
    PermissionAuditLog   -- mirrors the JSONL audit trail for queryability

Design goals:
    - Mapped dataclasses (SQLAlchemy 2.x style) give full IDE type
      support without a separate schema declaration file.
    - All primary keys are UUIDs rather than auto-increment integers,
      enabling eventual multi-device sync without key collisions.
    - Timestamps are stored as UTC ISO-8601 strings for portability
      across SQLite (which has no native datetime type), avoiding the
      timezone ambiguity that plagues naive datetime columns.
    - Soft-delete pattern on LongTermFact: facts are never hard-deleted
      so that audit trails and memory decay analysis remain possible.
    - No foreign key constraints are enforced at the SQLite level
      (SQLite requires PRAGMA foreign_keys=ON per connection), but
      the relationships are declared for SQLAlchemy's ORM join helpers.

Dependencies:
    sqlalchemy >= 2.0  (in requirements.txt as SQLAlchemy==2.0.41)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "ConversationSession",
    "ConversationTurn",
    "LongTermFact",
    "ScheduledReminder",
    "PermissionAuditLog",
]


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with 'Z' suffix.

    Used as the ``default`` factory for all timestamp columns. Storing
    timestamps as strings avoids SQLite's notoriously inconsistent
    handling of Python datetime objects across driver versions.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 as a lowercase hex string without dashes.

    Compact form (32 hex chars) is used instead of the standard
    8-4-4-4-12 form to keep primary key columns narrow while
    remaining globally unique.
    """
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all JARVIS ORM models.

    All mapped classes must inherit from this base so that
    ``Base.metadata.create_all(engine)`` in db_manager.py creates
    every table in a single call.
    """


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------

class ConversationSession(Base):
    """Represents one continuous JARVIS interaction session.

    A session begins when the wake word is detected and ends when JARVIS
    enters sleep mode (either by timeout or explicit command). Short-term
    memory is scoped to a session; long-term facts may be promoted from
    session turns at session close.

    Columns:
        id:           UUID primary key (hex, 32 chars).
        started_at:   ISO-8601 UTC timestamp of session start.
        ended_at:     ISO-8601 UTC timestamp of session end; null if active.
        turn_count:   Running count of turns in this session (denormalised
                      for fast session summary queries).
        summary:      Optional LLM-generated one-paragraph session summary,
                      populated at session close for long-term memory seeding.
    """

    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=_new_uuid
    )
    started_at: Mapped[str] = mapped_column(
        String(30), nullable=False, default=_utc_now_iso
    )
    ended_at: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship: one session has many turns (for ORM join queries)
    turns: Mapped[list[ConversationTurn]] = relationship(
        "ConversationTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.sequence_number",
    )

    def __repr__(self) -> str:
        return (
            f"ConversationSession(id={self.id!r}, "
            f"started_at={self.started_at!r}, "
            f"turns={self.turn_count})"
        )


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------

class ConversationTurn(Base):
    """A single user-utterance / assistant-response exchange.

    Each row captures both sides of one conversational turn so that the
    full dialogue history can be reconstructed for context injection,
    session summarisation, and debugging.

    Columns:
        id:              UUID primary key.
        session_id:      FK reference to ConversationSession.id.
        sequence_number: 1-based ordinal within the session; used for
                         ordering and for capping short-term context.
        user_input:      Transcribed user speech (post-STT text).
        assistant_reply: JARVIS response text (pre-TTS text).
        tool_calls_json: JSON array of tool calls made during this turn,
                         stored as a string for portability. Null if no
                         tools were invoked.
        created_at:      ISO-8601 UTC timestamp of turn creation.
        input_tokens:    Approximate token count of the user input fed
                         to the LLM, for context-window budget tracking.
        output_tokens:   Approximate token count of the assistant reply.
        latency_ms:      Wall-clock time from user input to first TTS
                         byte, in milliseconds. Used for performance
                         monitoring.
    """

    __tablename__ = "conversation_turns"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=_new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_reply: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String(30), nullable=False, default=_utc_now_iso
    )
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationship: many turns belong to one session
    session: Mapped[ConversationSession] = relationship(
        "ConversationSession", back_populates="turns"
    )

    __table_args__ = (
        # Enforce (session_id, sequence_number) uniqueness so that sequence
        # gaps are detectable and duplicate inserts are rejected at the DB level.
        UniqueConstraint("session_id", "sequence_number", name="uq_turn_session_seq"),
        # Index for retrieving the last N turns in a session cheaply.
        Index("ix_turn_session_seq", "session_id", "sequence_number"),
    )

    def __repr__(self) -> str:
        return (
            f"ConversationTurn(id={self.id!r}, "
            f"session_id={self.session_id!r}, "
            f"seq={self.sequence_number})"
        )


# ---------------------------------------------------------------------------
# LongTermFact
# ---------------------------------------------------------------------------

class LongTermFact(Base):
    """A remembered fact about the user or their environment.

    Long-term facts are promoted from conversation turns by the memory
    layer (memory/long_term.py) and injected into the system prompt
    by personality.py to ground the LLM's responses.

    Facts use a soft-delete pattern (``is_active`` flag) so that
    revoked or superseded beliefs can be traced without losing history.
    The ``relevance_score`` is updated by the retrieval logic each time
    the fact is retrieved; frequently-used facts naturally rise in score,
    enabling rudimentary memory decay without a separate scheduler.

    Columns:
        id:               UUID primary key.
        content:          The fact text, e.g. "User prefers dark mode."
        category:         Broad classification: "preference", "identity",
                          "habit", "environment", "instruction", "other".
        source_session_id:Session that produced this fact; null if seeded
                          manually.
        source_turn_id:   Turn that produced this fact; null if seeded.
        relevance_score:  Float 0.0–1.0; updated by retrieval layer.
        created_at:       ISO-8601 UTC timestamp.
        last_accessed_at: ISO-8601 UTC timestamp of most recent retrieval;
                          used for decay calculations.
        access_count:     Number of times this fact has been retrieved.
        is_active:        False = soft-deleted / superseded.
    """

    __tablename__ = "long_term_facts"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=_new_uuid
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(
        String(64), nullable=False, default="other"
    )
    source_session_id: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    source_turn_id: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    relevance_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5
    )
    created_at: Mapped[str] = mapped_column(
        String(30), nullable=False, default=_utc_now_iso
    )
    last_accessed_at: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True
    )
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        # Most common query: active facts ordered by relevance score.
        Index("ix_fact_active_relevance", "is_active", "relevance_score"),
        # Category filter for structured memory retrieval.
        Index("ix_fact_category", "category"),
    )

    def __repr__(self) -> str:
        preview = self.content[:40] + "…" if len(self.content) > 40 else self.content
        return (
            f"LongTermFact(id={self.id!r}, "
            f"category={self.category!r}, "
            f"active={self.is_active}, "
            f"content={preview!r})"
        )


# ---------------------------------------------------------------------------
# ScheduledReminder
# ---------------------------------------------------------------------------

class ScheduledReminder(Base):
    """A user-requested reminder or deferred task.

    Reminders are created by the reminder tool (tools/reminder_tool.py)
    and fired by core/scheduler.py via APScheduler. This table acts as
    the persistent job store so that reminders survive JARVIS restarts.

    Columns:
        id:            UUID primary key; also used as the APScheduler job ID
                       so that the ORM record and the scheduler job stay
                       in 1:1 correspondence.
        label:         Human-readable description, e.g. "Call dentist".
        trigger_at:    ISO-8601 UTC timestamp when the reminder fires.
        repeat_rule:   Optional iCal RRULE string for recurring reminders,
                       e.g. "FREQ=DAILY;COUNT=5". Null for one-shot.
        tts_message:   The text JARVIS speaks when the reminder fires.
        created_at:    ISO-8601 UTC timestamp.
        fired_at:      ISO-8601 UTC timestamp of most recent firing; null
                       if not yet fired.
        fire_count:    Number of times this reminder has fired.
        is_active:     False = cancelled or expired.
        source_session_id: Session that created this reminder.
    """

    __tablename__ = "scheduled_reminders"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=_new_uuid
    )
    label: Mapped[str] = mapped_column(String(512), nullable=False)
    trigger_at: Mapped[str] = mapped_column(String(30), nullable=False)
    repeat_rule: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    tts_message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(
        String(30), nullable=False, default=_utc_now_iso
    )
    fired_at: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    fire_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    source_session_id: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )

    __table_args__ = (
        # Scheduler polls for active reminders due before a given time.
        Index("ix_reminder_active_trigger", "is_active", "trigger_at"),
    )

    def __repr__(self) -> str:
        return (
            f"ScheduledReminder(id={self.id!r}, "
            f"label={self.label!r}, "
            f"trigger_at={self.trigger_at!r}, "
            f"active={self.is_active})"
        )


# ---------------------------------------------------------------------------
# PermissionAuditLog
# ---------------------------------------------------------------------------

class PermissionAuditLog(Base):
    """Queryable mirror of the permissions_audit.jsonl audit trail.

    security/permissions.py writes every authorization decision to a
    JSONL file for immediate durability. This table is populated in
    parallel so that diagnostic queries ("show recent denials for
    file.delete") can be answered with SQL rather than grep.

    The JSONL file remains the authoritative audit record; this table
    is a read-optimised projection of it. If the two ever diverge (e.g.
    the DB is reset), the JSONL file can be replayed to rebuild this table.

    Columns:
        id:          UUID primary key (from PermissionRequest.request_id).
        timestamp:   ISO-8601 UTC timestamp from the audit record.
        action:      Machine-readable action identifier, e.g. "file.delete".
        description: Human-readable description from the request.
        risk_level:  RiskLevel name, e.g. "HIGH".
        decision:    PermissionDecision name: "GRANTED", "DENIED", or
                     "TIMED_OUT".
        reason:      Human-readable reason string from the decision.
        metadata_json: The sanitised metadata dict, stored as a JSON string.
    """

    __tablename__ = "permission_audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[str] = mapped_column(String(30), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Most common diagnostic query: recent denials by action.
        Index("ix_audit_decision_action", "decision", "action"),
        # Time-ordered log browsing.
        Index("ix_audit_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"PermissionAuditLog(id={self.id!r}, "
            f"action={self.action!r}, "
            f"decision={self.decision!r})"
        )