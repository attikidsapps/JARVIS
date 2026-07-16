"""
jarvis/database/db_manager.py

SQLAlchemy engine, session factory, and full database lifecycle
management for JARVIS.

This module is the single point of contact between all JARVIS subsystems
and the SQLite database. No other module creates an engine, opens a
connection, or manages a session directly — they all call
``DatabaseManager.get_session()`` and work inside that context manager.

Responsibilities:
    - Create and configure the SQLAlchemy engine with SQLite-specific
      pragmas (WAL mode, foreign keys, busy timeout).
    - Run ``Base.metadata.create_all()`` on first startup to initialise
      the schema without requiring a separate migration step for a
      fresh install.
    - Provide a thread-safe, context-managed session factory so callers
      never have to manage commit / rollback / close themselves.
    - Expose focused repository methods for each model so the rest of
      the codebase works with typed Python objects, not raw SQL.
    - Mirror every PermissionAuditLog entry written by
      security/permissions.py into the ``permission_audit_log`` table
      so diagnostic queries can be answered in SQL.

Design decisions:
    - SQLite is the target database. It requires no server process, has
      zero configuration overhead, and is fully sufficient for a
      single-user desktop assistant. The WAL journal mode enables
      concurrent reads without blocking writes, which matters for the
      background scheduler and foreground conversation threads.
    - Sessions are scoped to individual operations (Unit of Work pattern)
      rather than to the application lifetime. This avoids long-lived
      transactions holding write locks and makes error recovery trivial —
      a failed session is simply discarded and a new one opened.
    - All public write methods commit and close the session themselves.
      Callers that need to batch multiple writes should use
      ``get_session()`` directly and manage the transaction boundary
      explicitly.

Dependencies:
    sqlalchemy == 2.0.41  (requirements.txt)
    database/models.py    (Base, all ORM classes)
    utils/config_loader.py (DatabaseSettings)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from database.models import (
    Base,
    ConversationSession,
    ConversationTurn,
    LongTermFact,
    PermissionAuditLog,
    ScheduledReminder,
)
from utils.config_loader import DatabaseSettings

__all__ = [
    "DatabaseError",
    "DatabaseManager",
]

logger = logging.getLogger("jarvis.database.db_manager")


class DatabaseError(Exception):
    """Raised when a database operation fails in a way the caller must handle.

    SQLAlchemy exceptions that indicate programming errors (wrong column
    name, missing table, etc.) are allowed to propagate as-is so they
    surface loudly during development. DatabaseError is reserved for
    operational failures the caller might want to recover from (e.g.
    a write that fails due to a locked database during shutdown).
    """


class DatabaseManager:
    """Manages the SQLite database engine, schema, and session lifecycle.

    Usage:
        settings = load_settings()
        db = DatabaseManager(settings.database)
        db.initialise()

        # Short-lived session (preferred for isolated operations):
        with db.get_session() as session:
            facts = session.query(LongTermFact).filter_by(is_active=True).all()

        # Convenience repository methods:
        session_id = db.create_conversation_session()
        db.append_conversation_turn(
            session_id=session_id,
            sequence_number=1,
            user_input="What time is it?",
            assistant_reply="It is 3:42 PM.",
        )
        db.close()

    Thread safety:
        The SQLAlchemy engine and sessionmaker are both thread-safe.
        Individual Session objects are NOT thread-safe and must not be
        shared across threads. ``get_session()`` always creates a new
        Session per call, so concurrent callers each get their own.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        """Initialise the database manager.

        Does NOT open a connection or create the schema. Call
        ``initialise()`` explicitly after construction so that startup
        errors surface at a predictable point in main.py rather than
        inside a constructor.

        Args:
            settings: DatabaseSettings from config_loader.py. Provides
                the file path, timeout, and WAL mode flag.
        """
        self._settings = settings
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None
        self._init_lock = Lock()
        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create the engine, apply SQLite pragmas, and create all tables.

        Idempotent — safe to call more than once; subsequent calls are
        no-ops. Thread-safe via an internal lock.

        Raises:
            DatabaseError: If the database file cannot be created or
                the schema initialisation fails.
        """
        with self._init_lock:
            if self._initialised:
                return

            db_path = Path(self._settings.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            url = f"sqlite:///{db_path.resolve()}"
            logger.info("Initialising database: %s", url)

            try:
                self._engine = create_engine(
                    url,
                    connect_args={
                        "timeout": self._settings.timeout_seconds,
                        "check_same_thread": False,
                    },
                    # Echo SQL to debug log only — never to console in production.
                    echo=False,
                )

                # Register SQLite-specific PRAGMA configuration.
                # The ``connect`` event fires once per new connection,
                # so pragmas are applied regardless of connection pooling.
                event.listen(
                    self._engine, "connect", self._apply_sqlite_pragmas
                )

                self._session_factory = sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False,
                )

                # Create all tables defined in models.py.
                # checkfirst=True means existing tables are left untouched.
                Base.metadata.create_all(self._engine, checkfirst=True)

                self._initialised = True
                logger.info(
                    "Database initialised — path=%s, wal=%s, timeout=%ds",
                    db_path.resolve(),
                    self._settings.wal_mode,
                    self._settings.timeout_seconds,
                )

            except SQLAlchemyError as exc:
                raise DatabaseError(
                    f"Failed to initialise database at '{db_path}': {exc}"
                ) from exc

    def close(self) -> None:
        """Dispose of the connection pool and release all resources.

        Should be called during JARVIS shutdown. After close(), the
        DatabaseManager instance must not be used again.
        """
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialised = False
            logger.info("Database connection pool disposed")

    def _apply_sqlite_pragmas(self, connection: object, _: object) -> None:
        """Apply SQLite PRAGMA settings immediately after a connection opens.

        Args:
            connection: The raw DBAPI connection (sqlite3.Connection).
            _:          The connection record (unused).
        """
        # Import here to use the raw sqlite3 cursor, not SQLAlchemy text().
        cursor = connection.cursor()  # type: ignore[union-attr]
        try:
            if self._settings.wal_mode:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            # busy_timeout prevents "database is locked" errors under
            # concurrent access by waiting up to N ms before raising.
            cursor.execute(
                f"PRAGMA busy_timeout={self._settings.timeout_seconds * 1000}"
            )
            # synchronous=NORMAL is safe with WAL mode and much faster
            # than the default FULL setting.
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Session context manager
    # ------------------------------------------------------------------

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional database session as a context manager.

        Commits automatically on clean exit. Rolls back and re-raises
        on any exception. Always closes the session on exit regardless
        of outcome.

        Usage:
            with db.get_session() as session:
                session.add(some_model_instance)
                # commit happens automatically here

        Raises:
            DatabaseError: If the manager has not been initialised.
            SQLAlchemyError: Propagated from SQLAlchemy on query errors.
        """
        if not self._initialised or self._session_factory is None:
            raise DatabaseError(
                "DatabaseManager has not been initialised. "
                "Call initialise() before get_session()."
            )

        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Conversation session repository
    # ------------------------------------------------------------------

    def create_conversation_session(self) -> str:
        """Insert a new ConversationSession row and return its ID.

        Returns:
            The UUID hex string of the newly created session.

        Raises:
            DatabaseError: On database write failure.
        """
        try:
            with self.get_session() as session:
                record = ConversationSession()
                session.add(record)
                session.flush()  # Populate defaults before commit
                session_id = record.id
            logger.debug("Created ConversationSession id=%s", session_id)
            return session_id
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to create ConversationSession: {exc}") from exc

    def close_conversation_session(
        self,
        session_id: str,
        summary: Optional[str] = None,
    ) -> None:
        """Mark a ConversationSession as ended.

        Args:
            session_id: The session UUID returned by
                ``create_conversation_session()``.
            summary:    Optional LLM-generated session summary text.

        Raises:
            DatabaseError: If the session is not found or write fails.
        """
        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        try:
            with self.get_session() as session:
                record = session.get(ConversationSession, session_id)
                if record is None:
                    raise DatabaseError(
                        f"ConversationSession not found: {session_id!r}"
                    )
                record.ended_at = ended_at
                if summary is not None:
                    record.summary = summary
            logger.debug("Closed ConversationSession id=%s", session_id)
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to close ConversationSession {session_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Conversation turn repository
    # ------------------------------------------------------------------

    def append_conversation_turn(
        self,
        session_id: str,
        sequence_number: int,
        user_input: str,
        assistant_reply: str,
        tool_calls: Optional[list[str]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> str:
        """Insert a ConversationTurn and increment the parent session's counter.

        Args:
            session_id:      Parent ConversationSession UUID.
            sequence_number: 1-based ordinal within the session.
            user_input:      Transcribed user utterance.
            assistant_reply: JARVIS text response.
            tool_calls:      List of tool identifiers invoked this turn.
            input_tokens:    Approximate input token count.
            output_tokens:   Approximate output token count.
            latency_ms:      End-to-end latency in milliseconds.

        Returns:
            The UUID hex string of the newly inserted turn.

        Raises:
            DatabaseError: On database write failure.
        """
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        try:
            with self.get_session() as session:
                turn = ConversationTurn(
                    session_id=session_id,
                    sequence_number=sequence_number,
                    user_input=user_input,
                    assistant_reply=assistant_reply,
                    tool_calls_json=tool_calls_json,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                )
                session.add(turn)

                # Increment denormalised turn counter on the parent session.
                parent = session.get(ConversationSession, session_id)
                if parent is not None:
                    parent.turn_count += 1

                session.flush()
                turn_id = turn.id

            logger.debug(
                "Appended ConversationTurn session=%s seq=%d id=%s",
                session_id, sequence_number, turn_id,
            )
            return turn_id
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to append ConversationTurn for session {session_id!r}: {exc}"
            ) from exc

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ConversationTurn]:
        """Return the most recent turns for a session, oldest first.

        Args:
            session_id: The ConversationSession UUID.
            limit:      Maximum number of turns to return.

        Returns:
            List of ConversationTurn ORM objects, ordered ascending by
            sequence_number. Empty list if session has no turns.
        """
        try:
            with self.get_session() as session:
                turns = (
                    session.query(ConversationTurn)
                    .filter(ConversationTurn.session_id == session_id)
                    .order_by(ConversationTurn.sequence_number.desc())
                    .limit(limit)
                    .all()
                )
                # Reverse so oldest is first (natural conversation order).
                turns.reverse()
                return turns
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to retrieve turns for session {session_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Long-term fact repository
    # ------------------------------------------------------------------

    def save_fact(
        self,
        content: str,
        category: str = "other",
        source_session_id: Optional[str] = None,
        source_turn_id: Optional[str] = None,
        relevance_score: float = 0.5,
    ) -> str:
        """Insert a new LongTermFact and return its ID.

        Args:
            content:           The fact text.
            category:          One of: preference | identity | habit |
                               environment | instruction | other.
            source_session_id: Session that produced this fact.
            source_turn_id:    Turn that produced this fact.
            relevance_score:   Initial relevance score (0.0–1.0).

        Returns:
            The UUID hex string of the newly saved fact.

        Raises:
            DatabaseError: On database write failure.
        """
        try:
            with self.get_session() as session:
                fact = LongTermFact(
                    content=content,
                    category=category,
                    source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                    relevance_score=relevance_score,
                )
                session.add(fact)
                session.flush()
                fact_id = fact.id

            logger.debug("Saved LongTermFact id=%s category=%s", fact_id, category)
            return fact_id
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to save LongTermFact: {exc}") from exc

    def get_active_facts(
        self,
        limit: int = 10,
        category: Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> list[LongTermFact]:
        """Return active facts ordered by relevance score descending.

        Args:
            limit:         Maximum number of facts to return.
            category:      Optional category filter.
            min_relevance: Only return facts at or above this score.

        Returns:
            List of LongTermFact ORM objects.
        """
        try:
            with self.get_session() as session:
                query = (
                    session.query(LongTermFact)
                    .filter(
                        LongTermFact.is_active.is_(True),
                        LongTermFact.relevance_score >= min_relevance,
                    )
                )
                if category is not None:
                    query = query.filter(LongTermFact.category == category)

                return (
                    query.order_by(LongTermFact.relevance_score.desc())
                    .limit(limit)
                    .all()
                )
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to retrieve LongTermFacts: {exc}") from exc

    def update_fact_access(self, fact_id: str, new_relevance: float) -> None:
        """Record a fact retrieval: update last_accessed_at, access_count, and score.

        Called by memory/long_term.py each time a fact is surfaced into
        the system prompt so that frequently-used facts organically rise
        in relevance score.

        Args:
            fact_id:       The LongTermFact UUID.
            new_relevance: The updated relevance score to store.

        Raises:
            DatabaseError: If the fact is not found or write fails.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        try:
            with self.get_session() as session:
                fact = session.get(LongTermFact, fact_id)
                if fact is None:
                    raise DatabaseError(f"LongTermFact not found: {fact_id!r}")
                fact.last_accessed_at = now
                fact.access_count += 1
                fact.relevance_score = max(0.0, min(1.0, new_relevance))
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to update LongTermFact {fact_id!r}: {exc}"
            ) from exc

    def deactivate_fact(self, fact_id: str) -> None:
        """Soft-delete a fact by setting is_active=False.

        Args:
            fact_id: The LongTermFact UUID to deactivate.

        Raises:
            DatabaseError: If the fact is not found or write fails.
        """
        try:
            with self.get_session() as session:
                fact = session.get(LongTermFact, fact_id)
                if fact is None:
                    raise DatabaseError(f"LongTermFact not found: {fact_id!r}")
                fact.is_active = False
            logger.debug("Deactivated LongTermFact id=%s", fact_id)
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to deactivate LongTermFact {fact_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Scheduled reminder repository
    # ------------------------------------------------------------------

    def save_reminder(
        self,
        label: str,
        trigger_at: str,
        tts_message: str,
        repeat_rule: Optional[str] = None,
        source_session_id: Optional[str] = None,
    ) -> str:
        """Insert a new ScheduledReminder and return its ID.

        The returned ID is also used as the APScheduler job ID so the
        ORM record and the scheduler job remain in 1:1 correspondence.

        Args:
            label:             Human-readable reminder description.
            trigger_at:        ISO-8601 UTC timestamp when it fires.
            tts_message:       Text JARVIS speaks when the reminder fires.
            repeat_rule:       Optional iCal RRULE string for recurrence.
            source_session_id: Session that created this reminder.

        Returns:
            The UUID hex string of the newly created reminder.

        Raises:
            DatabaseError: On database write failure.
        """
        try:
            with self.get_session() as session:
                reminder = ScheduledReminder(
                    label=label,
                    trigger_at=trigger_at,
                    tts_message=tts_message,
                    repeat_rule=repeat_rule,
                    source_session_id=source_session_id,
                )
                session.add(reminder)
                session.flush()
                reminder_id = reminder.id

            logger.debug(
                "Saved ScheduledReminder id=%s trigger_at=%s",
                reminder_id, trigger_at,
            )
            return reminder_id
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to save ScheduledReminder: {exc}") from exc

    def get_active_reminders(self) -> list[ScheduledReminder]:
        """Return all active (not yet cancelled) reminders.

        Returns:
            List of ScheduledReminder ORM objects ordered by trigger_at
            ascending (soonest first).
        """
        try:
            with self.get_session() as session:
                return (
                    session.query(ScheduledReminder)
                    .filter(ScheduledReminder.is_active.is_(True))
                    .order_by(ScheduledReminder.trigger_at.asc())
                    .all()
                )
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to retrieve active reminders: {exc}") from exc

    def record_reminder_fired(self, reminder_id: str) -> None:
        """Update fired_at and fire_count when a reminder executes.

        Args:
            reminder_id: The ScheduledReminder UUID.

        Raises:
            DatabaseError: If the reminder is not found or write fails.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        try:
            with self.get_session() as session:
                reminder = session.get(ScheduledReminder, reminder_id)
                if reminder is None:
                    raise DatabaseError(
                        f"ScheduledReminder not found: {reminder_id!r}"
                    )
                reminder.fired_at = now
                reminder.fire_count += 1
                # Deactivate non-repeating reminders after first fire.
                if reminder.repeat_rule is None:
                    reminder.is_active = False
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to record reminder fired {reminder_id!r}: {exc}"
            ) from exc

    def cancel_reminder(self, reminder_id: str) -> None:
        """Cancel a reminder by setting is_active=False.

        Args:
            reminder_id: The ScheduledReminder UUID.

        Raises:
            DatabaseError: If the reminder is not found or write fails.
        """
        try:
            with self.get_session() as session:
                reminder = session.get(ScheduledReminder, reminder_id)
                if reminder is None:
                    raise DatabaseError(
                        f"ScheduledReminder not found: {reminder_id!r}"
                    )
                reminder.is_active = False
            logger.debug("Cancelled ScheduledReminder id=%s", reminder_id)
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to cancel reminder {reminder_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Permission audit log repository
    # ------------------------------------------------------------------

    def write_audit_record(
        self,
        request_id: str,
        timestamp: str,
        action: str,
        description: str,
        risk_level: str,
        decision: str,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Mirror a permission audit record into the database.

        Called by security/permissions.py alongside its JSONL write so
        that audit records are queryable via SQL. Failures here are
        logged as warnings but do NOT raise — the JSONL file is the
        authoritative audit trail and a DB write failure must never
        block a permission decision.

        Args:
            request_id:  UUID from PermissionRequest.request_id.
            timestamp:   ISO-8601 UTC timestamp string.
            action:      Action identifier, e.g. "file.delete".
            description: Human-readable action description.
            risk_level:  RiskLevel name, e.g. "HIGH".
            decision:    PermissionDecision name: GRANTED/DENIED/TIMED_OUT.
            reason:      Human-readable decision reason.
            metadata:    Sanitised metadata dict (already redacted by
                         PermissionManager._sanitize_metadata).
        """
        metadata_json = json.dumps(metadata) if metadata else None
        try:
            with self.get_session() as session:
                record = PermissionAuditLog(
                    id=request_id,
                    timestamp=timestamp,
                    action=action,
                    description=description,
                    risk_level=risk_level,
                    decision=decision,
                    reason=reason,
                    metadata_json=metadata_json,
                )
                session.add(record)
        except SQLAlchemyError as exc:
            # Warning only — never raise from here. See docstring above.
            logger.warning(
                "Failed to mirror audit record %s to database: %s",
                request_id, exc,
            )

    def get_recent_audit_denials(
        self,
        limit: int = 20,
        action_filter: Optional[str] = None,
    ) -> list[PermissionAuditLog]:
        """Return the most recent denied/timed-out audit records.

        Mirrors the ``recent_denials()`` method of PermissionManager
        but operates on the queryable database table rather than the
        JSONL file.

        Args:
            limit:         Maximum number of records to return.
            action_filter: Optional action identifier to filter by,
                           e.g. "file.delete".

        Returns:
            List of PermissionAuditLog ORM objects, most recent first.
        """
        try:
            with self.get_session() as session:
                query = session.query(PermissionAuditLog).filter(
                    PermissionAuditLog.decision.in_(["DENIED", "TIMED_OUT"])
                )
                if action_filter:
                    query = query.filter(
                        PermissionAuditLog.action == action_filter
                    )
                return (
                    query.order_by(PermissionAuditLog.timestamp.desc())
                    .limit(limit)
                    .all()
                )
        except SQLAlchemyError as exc:
            raise DatabaseError(
                f"Failed to retrieve audit denials: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verify the database connection is alive with a lightweight query.

        Returns:
            True if the database responds correctly, False otherwise.
        """
        if not self._initialised or self._engine is None:
            return False
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            logger.error("Database health check failed: %s", exc)
            return False