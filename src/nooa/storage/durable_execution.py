# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable operation coordination for resumable agent workflows.

This module deliberately separates durable operation state from the event
stream.  Events are an audit/context projection; an operation record is the
coordination primitive used to decide whether a side effect may run again.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ExecutionStatus(StrEnum):
    """Durable states exposed by the operation ledger."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class EffectClass(StrEnum):
    """Effect guarantees required to interpret a retry safely."""

    PURE = "pure"
    IDEMPOTENT = "idempotent"
    TRANSACTIONAL = "transactional"
    COMPENSATABLE = "compensatable"
    IRREVERSIBLE = "irreversible"


@dataclass(frozen=True, slots=True)
class ExecutionKey:
    """Stable identity of one logical operation."""

    workflow_id: str
    operation_id: str


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Immutable view of a row in the operation ledger."""

    key: ExecutionKey
    request_hash: str
    status: ExecutionStatus
    effect_class: EffectClass
    attempt: int
    lease_generation: int
    lease_expires_at: float | None
    result: Any
    error: dict[str, str] | None
    updated_at: float


@dataclass(frozen=True, slots=True)
class ExecutionClaim:
    """The result of claiming an operation.

    A claim with ``owner_token`` is executable by the caller.  A claim with a
    missing token is a terminal replay and must not execute the effect again.
    """

    record: ExecutionRecord
    owner_token: str | None

    @property
    def executable(self) -> bool:
        return self.owner_token is not None


@dataclass(frozen=True, slots=True)
class ExecutionTransition:
    """Append-only audit record for one operation state transition."""

    sequence: int
    key: ExecutionKey
    from_status: ExecutionStatus | None
    to_status: ExecutionStatus
    attempt: int
    lease_generation: int
    occurred_at: float
    detail: str


class ExecutionConflictError(RuntimeError):
    """Raised when an operation ID is reused with different input."""


class ExecutionAlreadyRunningError(RuntimeError):
    """Raised when a live lease belongs to another worker."""


class LeaseLostError(RuntimeError):
    """Raised when a stale worker tries to commit an operation."""


class CorruptExecutionStateError(RuntimeError):
    """Raised when durable operation state cannot be safely decoded."""


class ExecutionReconciliationError(RuntimeError):
    """Raised when an unknown operation cannot be explicitly reconciled."""


def request_hash(request: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible request."""

    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class ExecutionStore(Protocol):
    """Backend-neutral durable operation contract."""

    def get(self, key: ExecutionKey) -> ExecutionRecord | None:
        """Read the current durable state for ``key``."""
        ...

    def history(self, key: ExecutionKey) -> list[ExecutionTransition]:
        """Read the append-only transition history for ``key``."""
        ...

    def claim(
        self,
        key: ExecutionKey,
        request: Any,
        *,
        effect_class: EffectClass = EffectClass.IDEMPOTENT,
        lease_seconds: float = 30.0,
    ) -> ExecutionClaim:
        """Acquire an execution lease or return a terminal replay."""
        ...

    def renew(self, claim: ExecutionClaim, *, lease_seconds: float = 30.0) -> ExecutionClaim:
        """Renew an owned execution lease."""
        ...

    def complete_success(self, claim: ExecutionClaim, result: Any) -> ExecutionRecord:
        """Commit a successful result."""
        ...

    def complete_failure(self, claim: ExecutionClaim, error: Exception) -> ExecutionRecord:
        """Commit a known failure."""
        ...

    def complete_unknown(self, claim: ExecutionClaim, reason: str) -> ExecutionRecord:
        """Commit an ambiguous outcome."""
        ...

    def reconcile_success(self, key: ExecutionKey, result: Any) -> ExecutionRecord:
        """Explicitly resolve an UNKNOWN operation as successful."""
        ...

    def reconcile_failure(self, key: ExecutionKey, error: Exception) -> ExecutionRecord:
        """Explicitly resolve an UNKNOWN operation as failed."""
        ...


DURABLE_EXECUTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS durable_operations (
    workflow_id       TEXT NOT NULL,
    operation_id      TEXT NOT NULL,
    request_hash      TEXT NOT NULL,
    status            TEXT NOT NULL,
    effect_class      TEXT NOT NULL,
    attempt           INTEGER NOT NULL,
    lease_generation  INTEGER NOT NULL,
    lease_token       TEXT,
    lease_expires_at  REAL,
    result            TEXT,
    error             TEXT,
    updated_at        REAL NOT NULL,
    PRIMARY KEY (workflow_id, operation_id)
);
CREATE INDEX IF NOT EXISTS idx_durable_operations_status
    ON durable_operations(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS durable_operation_transitions (
    sequence          INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id       TEXT NOT NULL,
    operation_id      TEXT NOT NULL,
    from_status       TEXT,
    to_status         TEXT NOT NULL,
    attempt           INTEGER NOT NULL,
    lease_generation  INTEGER NOT NULL,
    occurred_at       REAL NOT NULL,
    detail            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_durable_operation_transitions_key
    ON durable_operation_transitions(workflow_id, operation_id, sequence);
"""


def initialize_execution_schema(connection: sqlite3.Connection) -> None:
    """Create the additive durable-operation tables for a SQLite database."""

    for statement in DURABLE_EXECUTION_SCHEMA.split(";"):
        statement = statement.strip()
        if statement:
            connection.execute(statement)


class SQLiteExecutionStore:
    """SQLite-backed operation ledger with leases and fencing tokens.

    The store provides at-most-one *live owner* for a logical operation.  It
    cannot make an external side effect atomic with SQLite; callers must use
    an idempotency key at the effect boundary or treat an interrupted call as
    ``UNKNOWN``.  This is the deliberate distributed-systems contract.
    """

    _SCHEMA = DURABLE_EXECUTION_SCHEMA

    _RETRYABLE_AFTER_LEASE = frozenset({EffectClass.PURE, EffectClass.IDEMPOTENT})

    def __init__(
        self,
        database: str | Path | sqlite3.Connection,
        *,
        clock: Callable[[], float] | None = None,
        lock: threading.RLock | None = None,
    ) -> None:
        self._lock = lock or threading.RLock()
        self._clock = clock
        if isinstance(database, (str, Path)):
            self._conn = sqlite3.connect(
                str(database),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._owns_connection = True
            self._conn.execute("PRAGMA busy_timeout = 30000")
        else:
            self._conn = database
            self._owns_connection = False
        initialize_execution_schema(self._conn)

    def close(self) -> None:
        """Close the connection when this store created it."""

        if self._owns_connection:
            self._conn.close()

    def get(self, key: ExecutionKey) -> ExecutionRecord | None:
        """Read the current durable state for ``key``."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                (key.workflow_id, key.operation_id),
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def history(self, key: ExecutionKey) -> list[ExecutionTransition]:
        """Return the append-only transition history for ``key``."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT sequence, workflow_id, operation_id, from_status,
                          to_status, attempt, lease_generation, occurred_at, detail
                   FROM durable_operation_transitions
                   WHERE workflow_id = ? AND operation_id = ?
                   ORDER BY sequence""",
                (key.workflow_id, key.operation_id),
            ).fetchall()
        return [self._transition_from_row(row) for row in rows]

    def claim(
        self,
        key: ExecutionKey,
        request: Any,
        *,
        effect_class: EffectClass = EffectClass.IDEMPOTENT,
        lease_seconds: float = 30.0,
    ) -> ExecutionClaim:
        """Claim a logical operation or return its terminal replay.

        A live lease raises ``ExecutionAlreadyRunningError``.  An expired
        lease is taken over with a higher fencing generation.  Terminal rows
        are returned without an owner token, so callers cannot accidentally
        execute a completed or uncertain effect twice.
        """

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        hashed_request = request_hash(request)
        token = secrets.token_urlsafe(24)

        with self._transaction():
            now = self._now()
            row = self._conn.execute(
                "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                (key.workflow_id, key.operation_id),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """INSERT INTO durable_operations
                    (workflow_id, operation_id, request_hash, status, effect_class,
                     attempt, lease_generation, lease_token, lease_expires_at,
                     result, error, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, NULL, NULL, ?)""",
                    (
                        key.workflow_id,
                        key.operation_id,
                        hashed_request,
                        ExecutionStatus.RUNNING,
                        effect_class,
                        token,
                        now + lease_seconds,
                        now,
                    ),
                )
                self._append_transition(
                    key,
                    from_status=None,
                    to_status=ExecutionStatus.RUNNING,
                    attempt=1,
                    lease_generation=1,
                    occurred_at=now,
                    detail="claimed",
                )
                return ExecutionClaim(
                    self._record_from_row(
                        self._conn.execute(
                            "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                            (key.workflow_id, key.operation_id),
                        ).fetchone()
                    ),
                    token,
                )

            record = self._record_from_row(row)
            if record.request_hash != hashed_request:
                raise ExecutionConflictError(
                    f"operation {key.workflow_id!r}/{key.operation_id!r} already has a different request"
                )
            if record.effect_class is not effect_class:
                raise ExecutionConflictError(
                    f"operation {key.workflow_id!r}/{key.operation_id!r} already has a different effect class"
                )
            if record.status is not ExecutionStatus.RUNNING:
                return ExecutionClaim(record, None)
            if record.lease_expires_at is not None and record.lease_expires_at > now:
                raise ExecutionAlreadyRunningError(
                    f"operation {key.workflow_id!r}/{key.operation_id!r} is leased until "
                    f"{record.lease_expires_at:.6f}"
                )

            if record.effect_class not in self._RETRYABLE_AFTER_LEASE:
                reason = (
                    "lease expired after a potentially applied side effect; "
                    "manual reconciliation is required"
                )
                self._conn.execute(
                    """UPDATE durable_operations
                    SET status = ?, lease_token = NULL, lease_expires_at = NULL,
                        error = ?, updated_at = ?
                    WHERE workflow_id = ? AND operation_id = ?
                      AND status = ? AND lease_generation = ?""",
                    (
                        ExecutionStatus.UNKNOWN,
                        json.dumps({"type": "UnknownOutcome", "message": reason}),
                        now,
                        key.workflow_id,
                        key.operation_id,
                        ExecutionStatus.RUNNING,
                        record.lease_generation,
                    ),
                )
                self._append_transition(
                    key,
                    from_status=ExecutionStatus.RUNNING,
                    to_status=ExecutionStatus.UNKNOWN,
                    attempt=record.attempt,
                    lease_generation=record.lease_generation,
                    occurred_at=now,
                    detail="unsafe lease expiry",
                )
                unknown_row = self._conn.execute(
                    "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                    (key.workflow_id, key.operation_id),
                ).fetchone()
                return ExecutionClaim(self._record_from_row(unknown_row), None)

            next_generation = record.lease_generation + 1
            self._conn.execute(
                """UPDATE durable_operations
                SET attempt = ?, lease_generation = ?, lease_token = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE workflow_id = ? AND operation_id = ?
                  AND status = ? AND lease_generation = ?""",
                (
                    record.attempt + 1,
                    next_generation,
                    token,
                    now + lease_seconds,
                    now,
                    key.workflow_id,
                    key.operation_id,
                    ExecutionStatus.RUNNING,
                    record.lease_generation,
                ),
            )
            self._append_transition(
                key,
                from_status=ExecutionStatus.RUNNING,
                to_status=ExecutionStatus.RUNNING,
                attempt=record.attempt + 1,
                lease_generation=next_generation,
                occurred_at=now,
                detail="expired lease reclaimed",
            )
            return ExecutionClaim(
                self._record_from_row(
                    self._conn.execute(
                        "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                        (key.workflow_id, key.operation_id),
                    ).fetchone()
                ),
                token,
            )

    def renew(self, claim: ExecutionClaim, *, lease_seconds: float = 30.0) -> ExecutionClaim:
        """Extend a live lease and return an updated claim.

        Renewal is fenced by both the opaque token and monotonic generation.
        An already-expired lease cannot be revived by its former owner.
        """

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if claim.owner_token is None:
            raise LeaseLostError("terminal claims cannot be renewed")
        with self._transaction():
            now = self._now()
            expires_at = now + lease_seconds
            cursor = self._conn.execute(
                """UPDATE durable_operations
                SET lease_expires_at = ?, updated_at = ?
                WHERE workflow_id = ? AND operation_id = ?
                  AND status = ? AND lease_token = ?
                  AND lease_generation = ? AND lease_expires_at > ?""",
                (
                    expires_at,
                    now,
                    claim.record.key.workflow_id,
                    claim.record.key.operation_id,
                    ExecutionStatus.RUNNING,
                    claim.owner_token,
                    claim.record.lease_generation,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise LeaseLostError(
                    f"lease lost for {claim.record.key.workflow_id!r}/{claim.record.key.operation_id!r}"
                )
            row = self._conn.execute(
                "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                (claim.record.key.workflow_id, claim.record.key.operation_id),
            ).fetchone()
        return ExecutionClaim(self._record_from_row(row), claim.owner_token)

    def complete_success(self, claim: ExecutionClaim, result: Any) -> ExecutionRecord:
        """Commit a result only if ``claim`` still owns the lease."""

        return self._complete(claim, ExecutionStatus.SUCCEEDED, result=result, error=None)

    def complete_failure(self, claim: ExecutionClaim, error: Exception) -> ExecutionRecord:
        """Commit a known execution failure only if the lease is current."""

        return self._complete(
            claim,
            ExecutionStatus.FAILED,
            result=None,
            error={"type": type(error).__name__, "message": str(error)},
        )

    def complete_unknown(self, claim: ExecutionClaim, reason: str) -> ExecutionRecord:
        """Record an ambiguous external outcome without retrying it."""

        return self._complete(
            claim,
            ExecutionStatus.UNKNOWN,
            result=None,
            error={"type": "UnknownOutcome", "message": reason},
        )

    def reconcile_success(self, key: ExecutionKey, result: Any) -> ExecutionRecord:
        """Resolve an UNKNOWN operation after an external status check."""

        return self._reconcile(key, ExecutionStatus.SUCCEEDED, result=result, error=None)

    def reconcile_failure(self, key: ExecutionKey, error: Exception) -> ExecutionRecord:
        """Resolve an UNKNOWN operation after proving the effect did not occur."""

        return self._reconcile(
            key,
            ExecutionStatus.FAILED,
            result=None,
            error={"type": type(error).__name__, "message": str(error)},
        )

    def _reconcile(
        self,
        key: ExecutionKey,
        status: ExecutionStatus,
        *,
        result: Any,
        error: dict[str, str] | None,
    ) -> ExecutionRecord:
        encoded_result = json.dumps(result, ensure_ascii=False) if result is not None else None
        encoded_error = json.dumps(error, ensure_ascii=False) if error is not None else None
        with self._transaction():
            now = self._now()
            row = self._conn.execute(
                "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                (key.workflow_id, key.operation_id),
            ).fetchone()
            if row is None:
                raise ExecutionReconciliationError(f"operation {key!r} does not exist")
            record = self._record_from_row(row)
            if record.status is not ExecutionStatus.UNKNOWN:
                raise ExecutionReconciliationError(
                    f"operation {key!r} is {record.status.value}, not unknown"
                )
            self._conn.execute(
                """UPDATE durable_operations
                   SET status = ?, result = ?, error = ?, updated_at = ?
                   WHERE workflow_id = ? AND operation_id = ? AND status = ?""",
                (
                    status,
                    encoded_result,
                    encoded_error,
                    now,
                    key.workflow_id,
                    key.operation_id,
                    ExecutionStatus.UNKNOWN,
                ),
            )
            self._append_transition(
                key,
                from_status=ExecutionStatus.UNKNOWN,
                to_status=status,
                attempt=record.attempt,
                lease_generation=record.lease_generation,
                occurred_at=now,
                detail="reconciled success"
                if status is ExecutionStatus.SUCCEEDED
                else "reconciled failure",
            )
            resolved = self._conn.execute(
                "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                (key.workflow_id, key.operation_id),
            ).fetchone()
        return self._record_from_row(resolved)

    def _complete(
        self,
        claim: ExecutionClaim,
        status: ExecutionStatus,
        *,
        result: Any,
        error: dict[str, str] | None,
    ) -> ExecutionRecord:
        if claim.owner_token is None:
            raise LeaseLostError("terminal claims cannot commit")
        encoded_result = json.dumps(result, ensure_ascii=False) if result is not None else None
        encoded_error = json.dumps(error, ensure_ascii=False) if error is not None else None
        with self._transaction():
            now = self._now()
            cursor = self._conn.execute(
                """UPDATE durable_operations
                SET status = ?, lease_token = NULL, lease_expires_at = NULL,
                    result = ?, error = ?, updated_at = ?
                WHERE workflow_id = ? AND operation_id = ?
                  AND status = ? AND lease_token = ?
                  AND lease_generation = ? AND lease_expires_at > ?""",
                (
                    status,
                    encoded_result,
                    encoded_error,
                    now,
                    claim.record.key.workflow_id,
                    claim.record.key.operation_id,
                    ExecutionStatus.RUNNING,
                    claim.owner_token,
                    claim.record.lease_generation,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise LeaseLostError(
                    f"lease lost for {claim.record.key.workflow_id!r}/{claim.record.key.operation_id!r}"
                )
            self._append_transition(
                claim.record.key,
                from_status=ExecutionStatus.RUNNING,
                to_status=status,
                attempt=claim.record.attempt,
                lease_generation=claim.record.lease_generation,
                occurred_at=now,
                detail={
                    ExecutionStatus.SUCCEEDED: "completed",
                    ExecutionStatus.FAILED: "failed",
                    ExecutionStatus.UNKNOWN: "outcome marked unknown",
                }[status],
            )
            row = self._conn.execute(
                "SELECT * FROM durable_operations WHERE workflow_id = ? AND operation_id = ?",
                (claim.record.key.workflow_id, claim.record.key.operation_id),
            ).fetchone()
        return self._record_from_row(row)

    def _append_transition(
        self,
        key: ExecutionKey,
        *,
        from_status: ExecutionStatus | None,
        to_status: ExecutionStatus,
        attempt: int,
        lease_generation: int,
        occurred_at: float,
        detail: str,
    ) -> None:
        self._conn.execute(
            """INSERT INTO durable_operation_transitions
            (workflow_id, operation_id, from_status, to_status, attempt,
             lease_generation, occurred_at, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key.workflow_id,
                key.operation_id,
                from_status,
                to_status,
                attempt,
                lease_generation,
                occurred_at,
                detail,
            ),
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Run one write transaction without leaking the shared lock.

        The lock is released even if SQLite rejects ``BEGIN IMMEDIATE``. If a
        commit fails, a best-effort rollback leaves the connection reusable.
        """

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def _now(self) -> float:
        """Return test-injected time or SQLite's connection-local UTC time."""

        if self._clock is not None:
            return self._clock()
        row = self._conn.execute("SELECT (julianday('now') - 2440587.5) * 86400.0").fetchone()
        if row is None:
            raise RuntimeError("SQLite did not return an authoritative timestamp")
        return float(row[0])

    @staticmethod
    def _record_from_row(row: sqlite3.Row | tuple[Any, ...]) -> ExecutionRecord:
        try:
            (
                workflow_id,
                operation_id,
                hashed_request,
                status,
                effect_class,
                attempt,
                lease_generation,
                _lease_token,
                lease_expires_at,
                encoded_result,
                encoded_error,
                updated_at,
            ) = row
            return ExecutionRecord(
                key=ExecutionKey(workflow_id, operation_id),
                request_hash=hashed_request,
                status=ExecutionStatus(status),
                effect_class=EffectClass(effect_class),
                attempt=attempt,
                lease_generation=lease_generation,
                lease_expires_at=lease_expires_at,
                result=json.loads(encoded_result) if encoded_result is not None else None,
                error=json.loads(encoded_error) if encoded_error is not None else None,
                updated_at=updated_at,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CorruptExecutionStateError("durable operation row is unreadable") from exc

    @staticmethod
    def _transition_from_row(row: sqlite3.Row | tuple[Any, ...]) -> ExecutionTransition:
        try:
            (
                sequence,
                workflow_id,
                operation_id,
                from_status,
                to_status,
                attempt,
                lease_generation,
                occurred_at,
                detail,
            ) = row
            return ExecutionTransition(
                sequence=sequence,
                key=ExecutionKey(workflow_id, operation_id),
                from_status=ExecutionStatus(from_status) if from_status is not None else None,
                to_status=ExecutionStatus(to_status),
                attempt=attempt,
                lease_generation=lease_generation,
                occurred_at=occurred_at,
                detail=detail,
            )
        except (TypeError, ValueError) as exc:
            raise CorruptExecutionStateError("durable transition row is unreadable") from exc
