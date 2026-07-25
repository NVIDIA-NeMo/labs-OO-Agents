# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OTLP trace storage and retrieval backed by SQLite.

Receives OTLP JSON ExportTraceServiceRequest payloads, stores individual
spans and session metadata in an SQLite database, and serves them back
for both the trace viewer and evaluation viewer APIs.

Schema design:
  - ``session_id``, ``experiment``, ``eval_passed`` are dedicated columns
    (the minimum required for the viewer UI to function).
  - All other ``eval.*`` attributes are stored in a single ``eval_metadata``
    JSON blob column, making the schema flexible for arbitrary experiment
    metadata without schema migrations.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_PATH = Path(
    os.environ.get("NOOA_TRACE_DB")
    or os.environ.get("NEMO_OO_TRACE_DB")
    or Path.cwd() / "traces.db"
)

_db: sqlite3.Connection | None = None  # used only by init_db() for schema setup

# Thread-local storage for per-thread SQLite connections.
#
# All route handlers (read path) and the write executor use thread-local
# connections so that:
#  - Route handlers never share a connection with each other or with the
#    writer — no concurrent-access corruption.
#  - Route handlers run as sync `def` functions in anyio's thread pool
#    (FastAPI auto-threads sync handlers), keeping the event loop free
#    for the POST /v1/traces ingest path.
#  - WAL mode allows many concurrent readers + one writer on separate
#    connections without blocking.
#
# Two TLS namespaces separate high-throughput ingest writes from everything else:
#  _read_tls  — route handler threads (anyio thread pool): reads + low-frequency
#               annotation writes (button clicks).  WAL serialises these fine.
#  _write_tls — single-writer executor thread: high-throughput span ingest only.
#               Batched commits amortise WAL sync across many payloads.
_read_tls = threading.local()
_write_tls = threading.local()


def _get_db() -> sqlite3.Connection:
    """Return (or lazily open) a per-thread connection for route handler threads.

    Used for reads and low-frequency annotation writes (create/update/delete).
    Safe to call from any thread — each thread gets its own sqlite3.Connection
    so there is no shared mutable state between concurrent callers.
    """
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    if not hasattr(_read_tls, "conn"):
        _read_tls.conn = sqlite3.connect(str(DB_PATH), timeout=30)
        _read_tls.conn.execute("PRAGMA journal_mode=WAL")
        _read_tls.conn.execute("PRAGMA synchronous=NORMAL")
        _read_tls.conn.execute("PRAGMA busy_timeout=30000")
        _read_tls.conn.row_factory = sqlite3.Row
    return _read_tls.conn


def _get_write_db() -> sqlite3.Connection:
    """Return (or lazily open) the single-writer thread's connection.

    Called only from the single-writer executor thread, never from the
    event loop or route-handler threads.

    ``busy_timeout=30000`` lets SQLite itself wait up to 30s for a
    writer lock before raising ``database is locked`` — covers the
    common case where another reader (or a long OTLP batch commit on
    the same file) is mid-fsync. Without it, lock contention surfaces
    as 500s on the client in under a second.
    """
    if not hasattr(_write_tls, "conn"):
        _write_tls.conn = sqlite3.connect(str(DB_PATH), timeout=30)
        _write_tls.conn.execute("PRAGMA journal_mode=WAL")
        _write_tls.conn.execute("PRAGMA synchronous=NORMAL")
        _write_tls.conn.execute("PRAGMA busy_timeout=30000")
        _write_tls.conn.row_factory = sqlite3.Row
    return _write_tls.conn


def _has_column(db: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether *table* has a column named *column*."""
    rows = db.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _migrate_v1_to_v2(db: sqlite3.Connection) -> None:
    """Migrate from the old per-field eval columns to the single JSON blob.

    Old schema had: eval_model, eval_test_id, eval_test_name, eval_tier,
    eval_score, eval_error, eval_input_tokens, eval_output_tokens,
    eval_agent_class, eval_method, eval_variant, eval_run_id,
    eval_display_name, eval_scores, eval_input, eval_output, eval_expected,
    eval_trace_file, eval_duration_seconds  (all separate columns)

    New schema has: eval_passed (INTEGER), eval_metadata (TEXT / JSON blob).
    """
    log.info("Migrating sessions table from v1 (per-field) to v2 (JSON blob) …")

    old_eval_cols = [
        "eval_model",
        "eval_test_id",
        "eval_test_name",
        "eval_tier",
        "eval_score",
        "eval_error",
        "eval_input_tokens",
        "eval_output_tokens",
        "eval_agent_class",
        "eval_method",
        "eval_variant",
        "eval_run_id",
        "eval_display_name",
        "eval_scores",
        "eval_input",
        "eval_output",
        "eval_expected",
        "eval_trace_file",
        "eval_duration_seconds",
    ]
    json_parse_cols = {"eval_scores", "eval_input", "eval_output", "eval_expected"}

    rows = db.execute("SELECT * FROM sessions").fetchall()

    db.execute("""
        CREATE TABLE sessions_v2 (
            session_id TEXT PRIMARY KEY,
            experiment TEXT NOT NULL,
            span_count INTEGER DEFAULT 0,
            total_size  INTEGER DEFAULT 0,
            modified REAL DEFAULT 0,
            resource_attrs TEXT,
            eval_passed INTEGER,
            eval_metadata TEXT
        )
    """)

    for row in rows:
        meta: dict[str, Any] = {}
        for col in old_eval_cols:
            try:
                val = row[col]
            except (IndexError, KeyError):
                continue
            if val is None:
                continue
            key = col[len("eval_") :]
            if col in json_parse_cols and isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            meta[key] = val

        eval_passed = None
        try:
            raw = row["eval_passed"]
            if raw is not None:
                eval_passed = int(raw)
        except (IndexError, KeyError):
            pass

        eval_metadata = json.dumps(meta, separators=(",", ":")) if meta else None

        db.execute(
            """INSERT INTO sessions_v2
               (session_id, experiment, span_count, modified, resource_attrs,
                eval_passed, eval_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                row["session_id"],
                row["experiment"],
                row["span_count"],
                row["modified"],
                row["resource_attrs"],
                eval_passed,
                eval_metadata,
            ),
        )

    db.execute("DROP TABLE sessions")
    db.execute("ALTER TABLE sessions_v2 RENAME TO sessions")
    db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_experiment ON sessions(experiment)")
    db.commit()
    log.info("Migration complete – %d sessions migrated.", len(rows))


class DatabaseBusyAtStartup(RuntimeError):
    """Raised when another process is holding a lock on the DB at startup."""


def _check_writable(path: Path) -> None:
    """Fail fast if another process is already locking the DB file.

    Opens a dedicated connection, runs ``BEGIN IMMEDIATE`` (acquires a
    reserved lock — compatible with concurrent WAL readers, but blocked
    by any other reserved/exclusive writer) and rolls back immediately.
    If that raises, another process is holding a writer lock and
    subsequent journal/ingest writes will stall every client. Much
    better to surface this at startup with a concrete fix than to let
    every POST time out later.
    """
    import sqlite3 as _sqlite

    probe_timeout_s = 3
    try:
        probe = _sqlite.connect(str(path), timeout=probe_timeout_s)
    except _sqlite.OperationalError as e:
        raise DatabaseBusyAtStartup(f"cannot open {path}: {e}") from e
    try:
        probe.execute(f"PRAGMA busy_timeout={probe_timeout_s * 1000}")
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
    except _sqlite.OperationalError as e:
        raise DatabaseBusyAtStartup(
            f"another process is holding a writer lock on {path}: {e}\n"
            f"Diagnose with:\n"
            f"  lsof {path}\n"
            f"  pgrep -af nooa.viewer\n"
            f"Then kill any stale viewer, or pick a different NOOA_TRACE_DB."
        ) from e
    finally:
        probe.close()


def init_db() -> int:
    """Create tables and return the number of existing sessions.

    Raises ``DatabaseBusyAtStartup`` if another process is holding a
    writer lock on the DB — catching the usual "stale second viewer"
    footgun at process start instead of as runtime 500s/503s later.
    """
    global _db
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    _check_writable(DB_PATH)

    # Reset any existing thread-local read connection so _get_db() opens a
    # fresh one pointing at the new DB_PATH (important in tests where each
    # test uses a different tmp database file).
    if hasattr(_read_tls, "conn"):
        try:
            _read_tls.conn.close()
        except Exception:
            log.debug(
                "Failed to close thread-local read connection during init_db()", exc_info=True
            )
        del _read_tls.conn

    if hasattr(_write_tls, "conn"):
        try:
            _write_tls.conn.close()
        except Exception:
            log.debug(
                "Failed to close thread-local write connection during init_db()", exc_info=True
            )
        del _write_tls.conn

    _db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("PRAGMA synchronous=NORMAL")
    _db.row_factory = sqlite3.Row

    _db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            experiment TEXT NOT NULL,
            span_count INTEGER DEFAULT 0,
            total_size  INTEGER DEFAULT 0,
            modified REAL DEFAULT 0,
            resource_attrs TEXT,
            eval_passed INTEGER,
            eval_metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            trace_id TEXT,
            span_id TEXT,
            parent_span_id TEXT,
            name TEXT,
            kind INTEGER,
            start_time_ns INTEGER,
            end_time_ns INTEGER,
            status_code INTEGER,
            status_message TEXT,
            attributes TEXT,
            resource TEXT,
            events TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
        CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(session_id, parent_span_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_experiment ON sessions(experiment);

        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            span_id TEXT,
            target TEXT,
            name TEXT NOT NULL,
            score REAL,
            label TEXT,
            comment TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            author_id TEXT,
            source TEXT NOT NULL DEFAULT 'human',
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_annotations_session ON annotations(session_id);
        CREATE INDEX IF NOT EXISTS idx_annotations_span ON annotations(session_id, span_id);

        CREATE TABLE IF NOT EXISTS llm_calls (
            call_id          TEXT PRIMARY KEY,
            session_id       TEXT NOT NULL,
            span_id          TEXT,
            model            TEXT,
            ts_start         REAL,
            ts_end           REAL,
            input_skeleton   TEXT NOT NULL,
            output_messages  TEXT NOT NULL,
            tokens           TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id);

        -- Content-addressed message blocks. The journal wire protocol POSTs
        -- block contents separately from call records; the call's
        -- input_skeleton/output_messages reference block hashes, and we
        -- resolve them from this table when the trace is read back out.
        CREATE TABLE IF NOT EXISTS msg_blocks (
            session_id  TEXT NOT NULL,
            hash        TEXT NOT NULL,
            content     TEXT NOT NULL,
            PRIMARY KEY (session_id, hash)
        );
    """)
    _db.commit()

    if _has_column(_db, "sessions", "eval_model"):
        _migrate_v1_to_v2(_db)

    # Journal v3 renamed input_hashes/output_hashes → input_skeleton/output_messages
    # and dropped the msg_content table. Older DBs still have the v2 shape, and
    # CREATE TABLE IF NOT EXISTS above won't replace them — so INSERTs into
    # input_skeleton raise OperationalError. Journal data is regenerable trace
    # display state; drop-recreate when the new column is missing.
    if not _has_column(_db, "llm_calls", "input_skeleton"):
        log.info("Migrating llm_calls to v3 schema (skeleton + blocks)")
        _db.execute("DROP TABLE IF EXISTS llm_calls")
        _db.execute("DROP TABLE IF EXISTS msg_content")
        _db.execute("""
            CREATE TABLE llm_calls (
                call_id          TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                span_id          TEXT,
                model            TEXT,
                ts_start         REAL,
                ts_end           REAL,
                input_skeleton   TEXT NOT NULL,
                output_messages  TEXT NOT NULL,
                tokens           TEXT
            )
        """)
        _db.execute("CREATE INDEX idx_llm_calls_session ON llm_calls(session_id)")

    # Add span_id to llm_calls if missing (added in journal v2)
    if not _has_column(_db, "llm_calls", "span_id"):
        _db.execute("ALTER TABLE llm_calls ADD COLUMN span_id TEXT")
    # Always ensure the span index exists (safe to re-create)
    _db.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_span ON llm_calls(span_id)")

    # FTS5 index for fast text search across span content
    _db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(
            session_id UNINDEXED,
            span_id UNINDEXED,
            name,
            content
        )
    """)

    # Add total_size to sessions if missing — cached sum of span payload sizes
    # so /api/traces doesn't need a full-table scan on spans.
    # Existing sessions will show size=0 until new spans are ingested.
    if not _has_column(_db, "sessions", "total_size"):
        _db.execute("ALTER TABLE sessions ADD COLUMN total_size INTEGER DEFAULT 0")
    _db.commit()

    row = _db.execute("SELECT COUNT(*) FROM sessions").fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# OTLP parsing helpers
# ---------------------------------------------------------------------------


def otlp_attrs_to_dict(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OTLP attribute array [{key, value}] to a flat dict."""
    result: dict[str, Any] = {}
    for attr in attrs:
        key = attr.get("key", "")
        value_obj = attr.get("value", {})
        if "stringValue" in value_obj:
            result[key] = value_obj["stringValue"]
        elif "intValue" in value_obj:
            result[key] = int(value_obj["intValue"])
        elif "doubleValue" in value_obj:
            result[key] = float(value_obj["doubleValue"])
        elif "boolValue" in value_obj:
            result[key] = value_obj["boolValue"]
        elif "arrayValue" in value_obj:
            result[key] = [_extract_any_value(v) for v in value_obj["arrayValue"].get("values", [])]
        elif "kvlistValue" in value_obj:
            result[key] = {
                kv["key"]: _extract_any_value(kv.get("value", {}))
                for kv in value_obj["kvlistValue"].get("values", [])
            }
        elif "bytesValue" in value_obj:
            result[key] = value_obj["bytesValue"]
    return result


def _extract_any_value(value_obj: dict[str, Any]) -> Any:
    """Extract a scalar from an OTLP AnyValue."""
    if "stringValue" in value_obj:
        return value_obj["stringValue"]
    if "intValue" in value_obj:
        return int(value_obj["intValue"])
    if "doubleValue" in value_obj:
        return float(value_obj["doubleValue"])
    if "boolValue" in value_obj:
        return value_obj["boolValue"]
    if "arrayValue" in value_obj:
        return [_extract_any_value(v) for v in value_obj["arrayValue"].get("values", [])]
    if "kvlistValue" in value_obj:
        return {
            kv["key"]: _extract_any_value(kv.get("value", {}))
            for kv in value_obj["kvlistValue"].get("values", [])
        }
    return None


def _extract_session_and_experiment(body: dict) -> tuple[str, str]:
    """Extract session.id and experiment from an OTLP payload.

    Checks resource attributes first (the correct OTLP location when the
    exporter's thread has the right context).  Falls back to individual span
    attributes (where SessionSpanProcessor stamps ``session.id`` in the event
    loop thread) for the case where BatchSpanProcessor exports from a background
    thread whose ContextVar context predates ``set_session()`` being called.
    """
    experiment = "default"

    # Primary: resource attributes — also collect experiment for the fallback path
    for rs in body.get("resourceSpans", []):
        resource = rs.get("resource", {})
        attrs = otlp_attrs_to_dict(resource.get("attributes", []))
        exp = attrs.get("experiment")
        if exp:
            experiment = exp
        session_id = attrs.get("session.id", "")
        if session_id:
            return session_id, experiment

    # Fallback: span attributes (SessionSpanProcessor always stamps session.id there)
    for rs in body.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                span_attrs = otlp_attrs_to_dict(span.get("attributes", []))
                session_id = span_attrs.get("session.id", "")
                if session_id:
                    return session_id, experiment

    return "", experiment


def _extract_eval_fields(body: dict) -> tuple[int | None, dict[str, Any]]:
    """Extract eval.* fields from resource and root span attributes.

    Returns (eval_passed, metadata_dict) where eval_passed is the dedicated
    column value and metadata_dict contains everything else as a flat dict
    with the ``eval.`` prefix stripped (e.g. ``eval.model`` -> ``model``).
    """
    eval_passed: int | None = None
    meta: dict[str, Any] = {}

    for rs in body.get("resourceSpans", []):
        resource = rs.get("resource", {})
        res_attrs = otlp_attrs_to_dict(resource.get("attributes", []))

        for key in list(res_attrs):
            if key.startswith("eval."):
                short = key[len("eval.") :]
                if short == "passed":
                    eval_passed = int(bool(res_attrs[key]))
                else:
                    meta[short] = res_attrs[key]

        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                if not span.get("parentSpanId"):
                    span_attrs = otlp_attrs_to_dict(span.get("attributes", []))
                    for key in list(span_attrs):
                        if key.startswith("eval."):
                            short = key[len("eval.") :]
                            val = span_attrs[key]
                            if short == "passed":
                                eval_passed = int(bool(val))
                                continue
                            if isinstance(val, str) and short not in ("trace_file", "error"):
                                try:
                                    val = json.loads(val)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            meta[short] = val

    return eval_passed, meta


def _extract_resource_json(body: dict) -> str | None:
    """Extract the first resource object as a JSON string."""
    for rs in body.get("resourceSpans", []):
        resource = rs.get("resource", {})
        if resource:
            return json.dumps(resource, separators=(",", ":"))
    return None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _ingest_one(body: dict, db: sqlite3.Connection) -> dict[str, Any]:
    """Write one OTLP payload to *db* without committing.

    The caller is responsible for calling db.commit() when the transaction
    should be flushed.  Keeping commit outside the hot path lets
    ingest_batch_write() amortise the WAL sync across many payloads.
    """
    import uuid as _uuid

    session_id, experiment = _extract_session_and_experiment(body)
    if not session_id:
        session_id = f"unknown_{_uuid.uuid4().hex[:8]}"

    eval_passed, eval_meta = _extract_eval_fields(body)
    now = time.time()

    span_rows: list[tuple] = []
    resource_json: str | None = None

    for rs in body.get("resourceSpans", []):
        resource = rs.get("resource", {})
        res_json = json.dumps(resource, separators=(",", ":"))
        if resource_json is None:
            resource_json = res_json

        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                span_rows.append(
                    (
                        session_id,
                        span.get("traceId"),
                        span.get("spanId"),
                        span.get("parentSpanId"),
                        span.get("name"),
                        span.get("kind"),
                        int(span.get("startTimeUnixNano", "0")),
                        int(span.get("endTimeUnixNano", span.get("startTimeUnixNano", "0"))),
                        span.get("status", {}).get("code"),
                        span.get("status", {}).get("message"),
                        json.dumps(span.get("attributes", []), separators=(",", ":")),
                        res_json,
                        json.dumps(span.get("events", []), separators=(",", ":")),
                    )
                )

    span_count = len(span_rows)
    # Compute total payload size for the spans in this batch.
    # Each span_row tuple has: ..., attributes(10), resource(11), events(12)
    batch_size = sum(len(r[10]) + len(r[11]) + len(r[12]) for r in span_rows)

    existing = db.execute(
        """SELECT span_count, total_size, experiment, resource_attrs, eval_metadata
           FROM sessions WHERE session_id = ?""",
        (session_id,),
    ).fetchone()

    if existing:
        new_count = existing["span_count"] + span_count
        new_size = (existing["total_size"] or 0) + batch_size
        merged_meta = {}
        if existing["eval_metadata"]:
            try:
                merged_meta = json.loads(existing["eval_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        merged_meta.update(eval_meta)

        updates: dict[str, Any] = {
            "span_count": new_count,
            "total_size": new_size,
            "modified": now,
        }

        # Post-processing imports may send a small eval span for a session that
        # was streamed live earlier with only the default experiment.  Treat a
        # non-default experiment/resource payload as enrichment for the existing
        # session so it appears under the Evaluations tab with its full trace.
        if experiment and experiment != "default" and experiment != existing["experiment"]:
            updates["experiment"] = experiment

        if resource_json:
            try:
                new_resource_attrs = otlp_attrs_to_dict(
                    json.loads(resource_json).get("attributes", [])
                )
            except (json.JSONDecodeError, TypeError):
                new_resource_attrs = {}
            if new_resource_attrs:
                existing_resource_attrs = {}
                if existing["resource_attrs"]:
                    try:
                        existing_resource_attrs = json.loads(existing["resource_attrs"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                existing_resource_attrs.update(new_resource_attrs)
                updates["resource_attrs"] = json.dumps(
                    existing_resource_attrs, separators=(",", ":")
                )

        if eval_passed is not None:
            updates["eval_passed"] = eval_passed
        if merged_meta:
            updates["eval_metadata"] = json.dumps(merged_meta, separators=(",", ":"))

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        db.execute(
            f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
            (*updates.values(), session_id),
        )

        harbor_trial_name = merged_meta.get("harbor_trial_name")
        if (
            harbor_trial_name
            and isinstance(harbor_trial_name, str)
            and harbor_trial_name != session_id
            and experiment
            and experiment != "default"
        ):
            stale_stub = db.execute(
                """
                SELECT span_count FROM sessions
                WHERE session_id = ? AND experiment = ? AND eval_metadata IS NOT NULL
                """,
                (harbor_trial_name, experiment),
            ).fetchone()
            if stale_stub and (stale_stub["span_count"] or 0) <= 2:
                db.execute("DELETE FROM spans WHERE session_id = ?", (harbor_trial_name,))
                db.execute("DELETE FROM sessions WHERE session_id = ?", (harbor_trial_name,))
                db.execute("DELETE FROM spans_fts WHERE session_id = ?", (harbor_trial_name,))
    else:
        resource_attrs = None
        if resource_json:
            resource_attrs = json.dumps(
                otlp_attrs_to_dict(json.loads(resource_json).get("attributes", [])),
                separators=(",", ":"),
            )

        eval_metadata_json = json.dumps(eval_meta, separators=(",", ":")) if eval_meta else None

        db.execute(
            """INSERT INTO sessions
               (session_id, experiment, span_count, total_size, modified,
                resource_attrs, eval_passed, eval_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                experiment,
                span_count,
                batch_size,
                now,
                resource_attrs,
                eval_passed,
                eval_metadata_json,
            ),
        )

    if span_rows:
        db.executemany(
            """INSERT INTO spans
               (session_id, trace_id, span_id, parent_span_id, name, kind,
                start_time_ns, end_time_ns, status_code, status_message,
                attributes, resource, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            span_rows,
        )

        # Populate FTS index with searchable content from each span
        fts_rows = []
        for row in span_rows:
            # row: (session_id, trace_id, span_id, parent_span_id, name, kind,
            #        start_time_ns, end_time_ns, status_code, status_message,
            #        attributes, resource, events)
            span_name = row[4] or ""
            attrs_json = row[10]
            # Extract string values from attributes for searchable content
            content_parts = []
            try:
                attrs = json.loads(attrs_json) if attrs_json else []
                for attr in attrs:
                    val = attr.get("value", {})
                    sv = val.get("stringValue")
                    if sv:
                        content_parts.append(sv)
            except (json.JSONDecodeError, TypeError):
                content_parts.append(attrs_json or "")
            fts_rows.append((row[0], row[2], span_name, " ".join(content_parts)))

        try:
            db.executemany(
                "INSERT INTO spans_fts (session_id, span_id, name, content) VALUES (?, ?, ?, ?)",
                fts_rows,
            )
        except Exception:
            log.debug(
                "FTS insert failed for %d spans (search falls back to LIKE)",
                len(fts_rows),
                exc_info=True,
            )

    return {"session_id": session_id, "experiment": experiment, "span_count": span_count}


def ingest(body: dict) -> dict[str, Any]:
    """Ingest an OTLP ExportTraceServiceRequest using the read connection.

    For use from the event-loop thread (e.g. tests, direct calls).
    Use ingest_batch_write() from the writer executor thread instead.
    """
    db = _get_db()
    result = _ingest_one(body, db)
    db.commit()
    return result


def _log_oversized_payload(body: dict, raw_bytes: int) -> None:
    """Log a breakdown of an oversized OTLP payload to help find truncation bugs."""
    session_id, experiment = _extract_session_and_experiment(body)
    span_sizes: list[tuple[str, int]] = []
    for rs in body.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                name = span.get("name", "?")
                attrs = span.get("attributes", [])
                attrs_size = sum(len(json.dumps(a, separators=(",", ":"))) for a in attrs)
                span_sizes.append((name, attrs_size))
    span_sizes.sort(key=lambda x: -x[1])
    top = span_sizes[:5]
    top_str = "; ".join(f"{name}={sz / 1024:.0f}KB" for name, sz in top)
    log.warning(
        "[OVERSIZED] %.1f MB  session=%s  experiment=%s  spans=%d  top_attrs=[%s]",
        raw_bytes / (1024 * 1024),
        session_id,
        experiment,
        len(span_sizes),
        top_str,
    )


def ingest_batch_write_bytes(payloads: list[bytes]) -> list[dict[str, Any]]:
    """Ingest a batch of raw OTLP JSON payloads (bytes) in a single transaction.

    Parses each payload in the write thread — the HTTP handler is pure I/O
    and never touches json.loads(), keeping the event loop free regardless
    of payload size.
    """
    import time as _time

    t0 = _time.monotonic()
    db = _get_write_db()
    results: list[dict[str, Any]] = []
    total_bytes = sum(len(r) for r in payloads)
    for raw in payloads:
        try:
            body = json.loads(raw)
            # Log oversized payloads with the biggest span names/attributes.
            # We deliberately do NOT dump the raw payload to disk: it contains
            # full trace contents and previously landed in a fixed, world-readable
            # /tmp path. _log_oversized_payload() provides the diagnostics needed.
            if len(raw) > 1 * 1024 * 1024:  # 1 MB
                _log_oversized_payload(body, len(raw))
            results.append(_ingest_one(body, db))
        except Exception:
            log.exception("[ingest_batch_write_bytes] Failed to process one payload, skipping")
    db.commit()
    elapsed_ms = (_time.monotonic() - t0) * 1000
    if elapsed_ms > 500 or total_bytes > 50 * 1024 * 1024:
        sessions = {r.get("session_id", "?") for r in results}
        log.warning(
            "[ingest_batch] %.0fms  payloads=%d  %.1f MB  sessions=%s",
            elapsed_ms,
            len(payloads),
            total_bytes / (1024 * 1024),
            ", ".join(sorted(sessions)),
        )
    return results


def ingest_batch_write(bodies: list[dict]) -> list[dict[str, Any]]:
    """Ingest a batch of OTLP payloads in a single SQLite transaction.

    Call this from the single-writer executor thread.  Batching amortises
    WAL sync (db.commit()) across all payloads in the batch — the dominant
    cost under parallel eval loads — giving roughly N x throughput improvement
    for an N-item batch compared to N separate ingest() calls.

    The thread-local write connection is separate from _db (the event-loop
    read connection), preventing concurrent-access corruption.

    Per-payload exceptions are caught and logged so one malformed payload
    cannot prevent the rest of the batch from being committed.
    """
    db = _get_write_db()
    results: list[dict[str, Any]] = []
    for body in bodies:
        try:
            results.append(_ingest_one(body, db))
        except Exception:
            log.exception("[ingest_batch_write] Failed to process one payload, skipping")
    db.commit()
    return results


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _parse_eval_metadata(raw: str | None) -> dict[str, Any]:
    """Parse the eval_metadata JSON blob, returning an empty dict on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _row_to_session_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sessions table row to the dict format expected by callers."""
    meta = _parse_eval_metadata(row["eval_metadata"])

    # Extract batch_id from resource_attrs JSON if present
    resource_attrs = {}
    try:
        raw_attrs = row["resource_attrs"]
        if raw_attrs:
            resource_attrs = json.loads(raw_attrs)
    except (json.JSONDecodeError, TypeError):
        pass

    d: dict[str, Any] = {
        "id": row["session_id"],
        "name": row["session_id"],
        "experiment": row["experiment"],
        "modified": str(row["modified"]),
        "span_count": row["span_count"],
        "batch_id": resource_attrs.get("batch_id"),
    }

    if meta or row["eval_passed"] is not None:
        d["eval"] = dict(meta)
        d["eval"]["passed"] = bool(row["eval_passed"]) if row["eval_passed"] is not None else None

    return d


def list_sessions(
    experiment: str | None = None,
    eval_only: bool = False,
    batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return session metadata, optionally filtered by experiment, eval-only, or batch_id."""
    db = _get_db()
    clauses: list[str] = []
    params: list[Any] = []

    if experiment:
        clauses.append("experiment = ?")
        params.append(experiment)
    if eval_only:
        clauses.append("eval_metadata IS NOT NULL")
    if batch_id:
        clauses.append("json_extract(resource_attrs, '$.batch_id') = ?")
        params.append(batch_id)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = db.execute(f"SELECT * FROM sessions{where}", params).fetchall()
    return [_row_to_session_dict(r) for r in rows]


def find_eval_session_for_task(
    task_name: str,
    *,
    model: str | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    experiment: str = "default",
    slop_seconds: float = 900.0,
) -> dict[str, Any] | None:
    """Find the live OTLP session for a Harbor trial.

    Live-streamed Harbor traces historically used timestamp session IDs and
    the default experiment, while Harbor result metadata uses trial names.
    Match by task identifier in span content, model metadata, and trial time
    window so eval-only imports enrich the full trace instead of creating a
    tiny metadata-only stub session.
    """
    if not task_name:
        return None

    db = _get_db()
    params: list[Any] = [experiment, f"%{task_name}%", f"%{task_name}%"]
    model_clause = ""
    if model:
        model_like = f"%{model}%"
        model_clause = """
          AND (
            s.resource_attrs LIKE ?
            OR s.eval_metadata LIKE ?
          )
        """
        params.extend([model_like, model_like])

    rows = db.execute(
        f"""
        SELECT
            s.session_id,
            s.experiment,
            s.span_count,
            MIN(sp.start_time_ns) / 1000000000.0 AS first_s,
            MAX(sp.end_time_ns) / 1000000000.0 AS last_s
        FROM sessions s
        JOIN spans sp ON sp.session_id = s.session_id
        WHERE s.experiment = ?
          AND s.span_count > 2
          AND EXISTS (
            SELECT 1 FROM spans hit
            WHERE hit.session_id = s.session_id
              AND (hit.attributes LIKE ? OR hit.events LIKE ?)
          )
          {model_clause}
        GROUP BY s.session_id
        """,
        params,
    ).fetchall()

    best = None
    best_score = None
    for row in rows:
        first_s = row["first_s"]
        last_s = row["last_s"]
        if first_s is None or last_s is None:
            continue
        if started_at is not None and last_s < started_at - slop_seconds:
            continue
        if finished_at is not None and first_s > finished_at + slop_seconds:
            continue
        if started_at is not None:
            score = abs(first_s - started_at)
        elif finished_at is not None:
            score = abs(last_s - finished_at)
        else:
            score = -float(row["span_count"] or 0)
        if best_score is None or score < best_score:
            best = row
            best_score = score

    if best is None:
        return None
    return {
        "session_id": best["session_id"],
        "experiment": best["experiment"],
        "span_count": best["span_count"],
        "first_s": best["first_s"],
        "last_s": best["last_s"],
    }


def get_session_durations_ms(session_ids: list[str]) -> dict[str, float | None]:
    """Return ``{session_id: duration_ms}`` (latest span end − earliest span start).

    ``None`` for sessions with no spans, so callers can render "—" rather than
    a misleading 0. Chunked under SQLite's SQLITE_MAX_VARIABLE_NUMBER (999).
    """
    if not session_ids:
        return {}
    db = _get_db()
    unique_ids = list(dict.fromkeys(session_ids))
    out: dict[str, float | None] = {}
    for i in range(0, len(unique_ids), 500):
        chunk = unique_ids[i : i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = db.execute(
            f"SELECT session_id, MIN(start_time_ns) AS t0, MAX(end_time_ns) AS t1 "
            f"FROM spans WHERE session_id IN ({placeholders}) GROUP BY session_id",
            chunk,
        ).fetchall()
        for row in rows:
            t0, t1 = row["t0"], row["t1"]
            out[row["session_id"]] = (
                (t1 - t0) / 1_000_000 if t0 is not None and t1 is not None and t1 >= t0 else None
            )
    return out


def list_experiments() -> list[str]:
    """Return list of known experiment names."""
    db = _get_db()
    rows = db.execute("SELECT DISTINCT experiment FROM sessions ORDER BY experiment").fetchall()
    return [r[0] for r in rows]


def get_experiment_summary(experiment: str) -> dict[str, Any]:
    """Compute aggregated eval metrics for an experiment."""
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM sessions WHERE experiment = ? AND eval_metadata IS NOT NULL",
        (experiment,),
    ).fetchall()

    if not rows:
        return {"experiment": experiment, "total": 0}

    total = len(rows)
    passed = sum(1 for r in rows if r["eval_passed"])

    scores: list[float] = []
    by_model: dict[str, dict[str, Any]] = {}
    by_tier: dict[str, dict[str, Any]] = {}
    by_test_name: dict[str, dict[str, Any]] = {}
    total_input = 0
    total_output = 0

    for r in rows:
        meta = _parse_eval_metadata(r["eval_metadata"])
        score = meta.get("score")
        if score is not None:
            try:
                score = float(score)
                scores.append(score)
            except (ValueError, TypeError):
                score = None

        for label, bucket in (
            (meta.get("model"), by_model),
            (meta.get("tier"), by_tier),
            (meta.get("test_name"), by_test_name),
        ):
            if label is None:
                continue
            if label not in bucket:
                bucket[label] = {"total": 0, "passed": 0, "score_sum": 0.0}
            bucket[label]["total"] += 1
            if r["eval_passed"]:
                bucket[label]["passed"] += 1
            if score is not None:
                bucket[label]["score_sum"] += score

        total_input += meta.get("input_tokens") or 0
        total_output += meta.get("output_tokens") or 0

    total_score = sum(scores)
    scored = len(scores)

    return {
        "experiment": experiment,
        "total": total,
        "passed": passed,
        "success_rate": passed / total if total else 0.0,
        "avg_score": total_score / scored if scored else 0.0,
        "by_model": by_model,
        "by_tier": by_tier,
        "by_test_name": by_test_name,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "modified": str(max(r["modified"] for r in rows)),
    }


def _resolve_message(msg: dict[str, Any], blocks: dict[str, str]) -> dict[str, Any]:
    """Resolve a journal-skeleton message back to its v2-shape, in-line content.

    Inverse of :func:`nooa.tracing._litellm_journal._skeleton_dict_message`
    (and the same for ``_journal_builder.build_journal_payload``):

    * ``parts: [{block_hash}]`` / ``[{text}]`` -> ``content`` (string, parts joined).
    * ``tool_calls[i].function.arguments_hash`` -> ``arguments``.
    * ``image_hashes`` -> ``images`` (placeholder URLs in resolution order).

    Block hashes that don't resolve (``hash`` not in *blocks*) are kept as a
    ``<missing block: {hash}>`` placeholder so the gap is visible in the
    download rather than silently empty.

    Messages that already have ``content`` (no ``parts``) pass through
    unchanged -- the v3 protocol can carry both shapes simultaneously.
    """
    out = dict(msg)

    parts = out.pop("parts", None)
    if parts is not None and "content" not in out:
        text_pieces: list[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if "block_hash" in p:
                h = p["block_hash"]
                resolved = blocks.get(h)
                if resolved is None:
                    text_pieces.append(f"<missing block: {h}>")
                else:
                    text_pieces.append(resolved)
            elif "text" in p:
                text_pieces.append(str(p["text"]))
        out["content"] = "".join(text_pieces) if text_pieces else None

    tcs = out.get("tool_calls")
    if tcs:
        new_tcs: list[dict[str, Any]] = []
        for tc in tcs:
            new_tc = dict(tc)
            fn = dict(new_tc.get("function") or {})
            if "arguments" not in fn and "arguments_hash" in fn:
                ah = fn.pop("arguments_hash")
                resolved = blocks.get(ah)
                fn["arguments"] = resolved if resolved is not None else f"<missing block: {ah}>"
            new_tc["function"] = fn
            new_tcs.append(new_tc)
        out["tool_calls"] = new_tcs

    img_hashes = out.pop("image_hashes", None)
    if img_hashes is not None and "images" not in out:
        out["images"] = [_resolve_image(blocks.get(h, f"<missing block: {h}>")) for h in img_hashes]

    return out


def _resolve_image(s: str) -> Any:
    """Reverse the canonical-JSON encoding ``_encode_image`` does for
    non-string images so the round-tripped ``images`` list matches what
    the runtime originally rendered.

    ``_encode_image`` is "if str: pass-through, else canonical JSON".
    Only dict/list inputs ever produce JSON output, so the inverse
    accepts only ``dict`` / ``list`` results.  Anything else (decode
    failure, or a string input that happens to be valid JSON for a
    scalar like ``"null"`` / ``"42"`` / ``"true"``) is returned
    verbatim, matching what the runtime saw.
    """
    if not isinstance(s, str):
        return s
    try:
        decoded = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s
    if isinstance(decoded, (dict, list)):
        return decoded
    # JSON scalar (str / int / float / bool / None) decoded from a
    # string -- ``_encode_image`` of a string never produces this, so
    # the source must be a plain-text image whose contents happen to
    # parse as a JSON scalar.  Keep the original string.
    return s


def _flatten_msg_to_attrs(msg: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Convert a message dict to OTLP-format flat attribute dicts.

    Inverse of the OpenInference ``_extract_messages`` logic::

        {"role": "user", "content": "hi"}
        → [{"key": "…role", "value": {"stringValue": "user"}},
           {"key": "…content", "value": {"stringValue": "hi"}}]

    ``content: None`` is omitted (not emitted as ``"None"``).
    """

    def _str(key: str, val: str) -> dict[str, Any]:
        return {"key": key, "value": {"stringValue": val}}

    attrs: list[dict[str, Any]] = [_str(f"{prefix}.role", msg.get("role", ""))]

    content = msg.get("content")
    if content is not None:
        attrs.append(_str(f"{prefix}.content", str(content)))

    tool_call_id = msg.get("tool_call_id")
    if tool_call_id:
        attrs.append(_str(f"{prefix}.tool_call_id", tool_call_id))

    for j, tc in enumerate(msg.get("tool_calls") or []):
        tcp = f"{prefix}.tool_calls.{j}.tool_call"
        attrs.append(_str(f"{tcp}.id", tc.get("id", "")))
        fn = tc.get("function") or {}
        attrs.append(_str(f"{tcp}.function.name", fn.get("name", "")))
        attrs.append(_str(f"{tcp}.function.arguments", fn.get("arguments", "")))

    return attrs


def _augment_span_attrs(
    attrs: list[dict[str, Any]],
    call: dict[str, Any],
    blocks: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Augment LLM message attrs with journal-reconstructed content.

    Each input/output message is in v3 wire shape (``parts:
    [{block_hash}]`` / ``arguments_hash``); *blocks* is the
    ``{hash: content}`` map used to resolve those references back to
    in-line content before flattening to OTLP attributes.

    "Soft" augmentation: only the indices the journal call actually
    supplies are replaced.  If the call record is missing some
    messages (hash miss, partial delivery, downgraded skeleton from
    the no-sideband fallback path) the original ``llm.input_messages.N.*``
    attrs at indices the journal didn't cover are preserved -- a
    truncated journal must not erase content the span already had.
    """
    blocks = blocks or {}
    call_id = call.get("call_id", "<unknown>")

    def _augmented(prefix: str, raw_messages: list) -> tuple[list[dict], set[int]]:
        out: list[dict] = []
        covered: set[int] = set()
        idx = 0
        for raw_i, msg in enumerate(raw_messages):
            if not msg:
                log.warning(
                    "_augment_span_attrs: message at %s index %d missing for call_id=%r",
                    prefix,
                    raw_i,
                    call_id,
                )
                continue
            resolved = _resolve_message(msg, blocks)
            out.extend(_flatten_msg_to_attrs(resolved, f"{prefix}.{idx}.message"))
            covered.add(idx)
            idx += 1
        return out, covered

    in_attrs, in_covered = _augmented("llm.input_messages", list(call.get("input_skeleton") or []))
    out_attrs, out_covered = _augmented(
        "llm.output_messages", list(call.get("output_messages") or [])
    )

    def _index_of(key: str, prefix: str) -> int | None:
        # ``llm.input_messages.7.message.foo`` -> 7
        rest = key[len(prefix) + 1 :]  # drop "{prefix}."
        head, _, _ = rest.partition(".")
        try:
            return int(head)
        except ValueError:
            return None

    kept: list[dict[str, Any]] = []
    for a in attrs:
        k = a["key"]
        if k.startswith("llm.input_messages."):
            i = _index_of(k, "llm.input_messages")
            if i is not None and i in in_covered:
                continue  # journal supplies this index; drop the stored attr
        elif k.startswith("llm.output_messages."):
            i = _index_of(k, "llm.output_messages")
            if i is not None and i in out_covered:
                continue
        kept.append(a)

    return kept + in_attrs + out_attrs


def reconstruct_full_spans(session_id: str) -> list[dict[str, Any]]:
    """Return OTLP spans for *session_id* with journal-reconstructed messages.

    The single read-side helper that resolves the journal sideband
    (``llm_calls`` + ``msg_blocks``) back into ``llm.input_messages.*`` /
    ``llm.output_messages.*`` attributes on each LLM span, so the on-disk
    storage shape (stripped) is invisible to consumers.

    Used by both ``get_session_spans`` (UI render path) and
    ``export_session_otlp`` (download path) so they always agree on what
    a session "is" -- the on-the-wire compression of the journal protocol
    is a strictly internal concern.
    """
    db = _get_db()
    exists = db.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not exists:
        raise FileNotFoundError(f"Session not found: {session_id}")

    rows = db.execute(
        "SELECT * FROM spans WHERE session_id = ? ORDER BY start_time_ns",
        (session_id,),
    ).fetchall()

    spans: list[dict[str, Any]] = []
    for r in rows:
        span = _row_to_span(r)
        span["_resource"] = json.loads(r["resource"]) if r["resource"] else {}
        spans.append(span)

    journal_by_span = _get_journal_calls_by_span(session_id)
    if not journal_by_span:
        return spans

    blocks = get_session_blocks(session_id)
    for span in spans:
        span_id = span.get("spanId")
        if not span_id or span_id not in journal_by_span:
            continue
        attrs = span.get("attributes", [])
        is_llm = any(
            a["key"] == "openinference.span.kind" and a.get("value", {}).get("stringValue") == "LLM"
            for a in attrs
        )
        if is_llm:
            span["attributes"] = _augment_span_attrs(attrs, journal_by_span[span_id], blocks)

    return spans


def session_exists(session_id: str) -> bool:
    """Return True if the session is present in the sessions table."""
    db = _get_db()
    return (
        db.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        is not None
    )


def _get_journal_calls_by_span(session_id: str) -> dict[str, dict[str, Any]]:
    """Return {span_id: call_record} for all journal calls in a session that
    have a span_id.  call_record includes reconstructed input/output messages.
    """
    calls = get_session_calls(session_id)
    return {c["span_id"]: c for c in calls if c.get("span_id")}


def _row_to_span(r: sqlite3.Row) -> dict[str, Any]:
    """Convert a spans table row to an OTLP-format span dict."""
    span: dict[str, Any] = {
        "traceId": r["trace_id"],
        "spanId": r["span_id"],
        "name": r["name"],
        "kind": r["kind"],
        "startTimeUnixNano": str(r["start_time_ns"]),
        "endTimeUnixNano": str(r["end_time_ns"]),
        "attributes": json.loads(r["attributes"]) if r["attributes"] else [],
        "status": {},
    }
    if r["parent_span_id"]:
        span["parentSpanId"] = r["parent_span_id"]
    if r["status_code"] is not None:
        span["status"]["code"] = r["status_code"]
    if r["status_message"]:
        span["status"]["message"] = r["status_message"]
    if r["events"]:
        events = json.loads(r["events"])
        if events:
            span["events"] = events
    return span


def get_session_spans(session_id: str, augment: bool = True) -> list[dict[str, Any]]:
    """Return all spans for a session as OTLP-format dicts.

    With ``augment=True`` (default), LLM spans receive
    journal-reconstructed ``llm.input_messages.*`` /
    ``llm.output_messages.*`` attributes via :func:`reconstruct_full_spans`.
    Pass ``augment=False`` to see the raw stored shape (debugging only;
    consumers should never rely on the stripped form leaking through).
    """
    if not augment:
        # Fast path: no augmentation, return raw stored spans.
        db = _get_db()
        exists = db.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not exists:
            raise FileNotFoundError(f"Session not found: {session_id}")
        rows = db.execute(
            "SELECT * FROM spans WHERE session_id = ? ORDER BY start_time_ns",
            (session_id,),
        ).fetchall()
        spans: list[dict[str, Any]] = []
        for r in rows:
            span = _row_to_span(r)
            span["_resource"] = json.loads(r["resource"]) if r["resource"] else {}
            spans.append(span)
        return spans

    return reconstruct_full_spans(session_id)


def export_session_otlp(session_id: str) -> list[dict[str, Any]]:
    """Return OTLP ExportTraceServiceRequest bodies for a session.

    Reconstructs the ``resourceSpans`` wrapper that ``/v1/traces`` expects,
    grouping spans by their stored resource JSON so the result round-trips
    cleanly through ``import-traces``.

    Spans are augmented with journal-reconstructed messages via
    :func:`reconstruct_full_spans`, so the download is byte-equivalent
    (modulo timestamps / batching) to a JSONL file written by
    ``exporters.jsonl`` for the same run.  Journaling stays an internal
    wire optimization -- consumers see complete OTLP either way.
    """
    spans = reconstruct_full_spans(session_id)

    # Group spans by their original resource JSON (carried on each span as
    # ``_resource`` from reconstruct_full_spans / _row_to_span lookups).
    groups: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        resource = span.pop("_resource", {}) if "_resource" in span else {}
        res_key = json.dumps(resource, separators=(",", ":"), sort_keys=True)
        groups.setdefault(res_key, []).append(span)

    bodies: list[dict[str, Any]] = []
    for res_json, spans_list in groups.items():
        resource = json.loads(res_json)
        bodies.append(
            {
                "resourceSpans": [
                    {
                        "resource": resource,
                        "scopeSpans": [{"spans": spans_list}],
                    }
                ]
            }
        )

    return bodies


def get_session_eval_detail(session_id: str) -> dict[str, Any]:
    """Return the full eval metadata for a session (input, output, expected, scores, etc.)."""
    db = _get_db()
    row = db.execute(
        "SELECT eval_metadata FROM sessions WHERE session_id = ? AND eval_metadata IS NOT NULL",
        (session_id,),
    ).fetchone()

    if row is None:
        return {}

    return _parse_eval_metadata(row["eval_metadata"])


def get_session_resource(session_id: str) -> dict[str, Any]:
    """Get the resource attributes for a session."""
    db = _get_db()
    row = db.execute(
        "SELECT resource_attrs FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if row is None:
        return {}

    if row["resource_attrs"]:
        try:
            return json.loads(row["resource_attrs"])
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def get_stats() -> dict[str, Any]:
    """Return lightweight stats about the store."""
    db = _get_db()
    sessions = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    spans = db.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    experiments = db.execute("SELECT COUNT(DISTINCT experiment) FROM sessions").fetchone()[0]
    return {"sessions": sessions, "spans": spans, "experiments": experiments}


def get_session_sizes() -> dict[str, int]:
    """Return approximate stored size in bytes per session from the cached total_size column."""
    db = _get_db()
    rows = db.execute("SELECT session_id, total_size FROM sessions").fetchall()
    return {r["session_id"]: int(r["total_size"] or 0) for r in rows}


def delete_session(session_id: str) -> bool:
    """Delete a single session and all associated data (spans, spans_fts, llm_calls, msg_blocks, annotations). Returns True if deleted."""
    db = _get_db()
    exists = db.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not exists:
        return False
    db.execute("DELETE FROM annotations WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM spans WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM spans_fts WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM llm_calls WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM msg_blocks WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    db.commit()
    return True


def delete_all_sessions() -> dict[str, int]:
    """Delete all sessions and associated data (spans, spans_fts, llm_calls, msg_blocks, annotations). Returns counts before deletion."""
    db = _get_db()
    stats = get_stats()
    db.execute("DELETE FROM annotations")
    db.execute("DELETE FROM spans")
    db.execute("DELETE FROM spans_fts")
    db.execute("DELETE FROM llm_calls")
    db.execute("DELETE FROM msg_blocks")
    db.execute("DELETE FROM sessions")
    db.commit()
    return stats


def delete_sessions_by_batch(batch_id: str) -> int:
    """Delete all sessions whose resource_attrs contain the given batch_id. Returns count deleted."""
    db = _get_db()
    rows = db.execute(
        "SELECT session_id FROM sessions WHERE json_extract(resource_attrs, '$.batch_id') = ?",
        (batch_id,),
    ).fetchall()
    if not rows:
        return 0
    session_ids = [r["session_id"] for r in rows]
    placeholders = ",".join("?" * len(session_ids))
    db.execute(f"DELETE FROM annotations WHERE session_id IN ({placeholders})", session_ids)
    db.execute(f"DELETE FROM spans WHERE session_id IN ({placeholders})", session_ids)
    db.execute(f"DELETE FROM spans_fts WHERE session_id IN ({placeholders})", session_ids)
    db.execute(f"DELETE FROM llm_calls WHERE session_id IN ({placeholders})", session_ids)
    db.execute(f"DELETE FROM msg_blocks WHERE session_id IN ({placeholders})", session_ids)
    db.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", session_ids)
    db.commit()
    return len(session_ids)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

_DEFAULT_TAGS = [
    "correct",
    "incorrect",
    "hallucination",
    "hypothesis:prompt-unclear",
    "hypothesis:model-limitation",
]


def _row_to_annotation(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an annotations table row to a dict with parsed JSON fields."""
    d = dict(row)
    if d.get("tags"):
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    else:
        d["tags"] = []
    if d.get("metadata"):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
    return d


def list_annotations(session_id: str, span_id: str | None = None) -> list[dict[str, Any]]:
    """Return annotations for a session, optionally filtered by span_id."""
    db = _get_db()
    if span_id:
        rows = db.execute(
            "SELECT * FROM annotations WHERE session_id = ? AND span_id = ?",
            (session_id, span_id),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM annotations WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    return [_row_to_annotation(r) for r in rows]


def create_annotation(data: dict[str, Any]) -> dict[str, Any]:
    """Insert a new annotation and return it.

    Uses _get_db() (thread-local read/write connection) rather than routing
    through the single-writer executor.  Annotations are low-frequency human
    operations (button clicks), so WAL-mode's built-in writer serialisation
    is sufficient — no risk of the contention that justifies the batched
    executor path for high-throughput span ingest.
    """
    import uuid
    from datetime import UTC, datetime

    db = _get_db()

    ann_id = data.get("id") or str(uuid.uuid4())
    created_at = data.get("created_at") or datetime.now(UTC).isoformat()
    tags = json.dumps(data.get("tags", [])) if data.get("tags") else None
    metadata = json.dumps(data["metadata"]) if data.get("metadata") else None

    db.execute(
        """INSERT INTO annotations
           (id, session_id, span_id, target, name, score, label, comment,
            tags, created_at, author_id, source, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ann_id,
            data["session_id"],
            data.get("span_id"),
            data.get("target"),
            data["name"],
            data.get("score"),
            data.get("label"),
            data.get("comment"),
            tags,
            created_at,
            data.get("author_id"),
            data.get("source", "human"),
            metadata,
        ),
    )
    db.commit()

    row = db.execute("SELECT * FROM annotations WHERE id = ?", (ann_id,)).fetchone()
    return _row_to_annotation(row)


def update_annotation(annotation_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Update an annotation. Raises ValueError if not found."""
    db = _get_db()

    row = db.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
    if not row:
        raise ValueError(f"Annotation not found: {annotation_id}")

    allowed = {"score", "label", "comment", "tags", "target", "author_id", "metadata"}
    sets: list[str] = []
    vals: list[Any] = []
    for k, v in updates.items():
        if k not in allowed:
            continue
        if k == "tags" and v is not None:
            v = json.dumps(v)
        elif k == "metadata" and v is not None:
            v = json.dumps(v)
        sets.append(f"{k} = ?")
        vals.append(v)

    if not sets:
        return _row_to_annotation(row)

    vals.append(annotation_id)
    db.execute(f"UPDATE annotations SET {', '.join(sets)} WHERE id = ?", vals)
    db.commit()

    updated = db.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
    return _row_to_annotation(updated)


def delete_annotation(annotation_id: str) -> None:
    """Delete an annotation. Raises ValueError if not found."""
    db = _get_db()
    row = db.execute("SELECT 1 FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
    if not row:
        raise ValueError(f"Annotation not found: {annotation_id}")
    db.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    db.commit()


def list_tags() -> list[dict[str, Any]]:
    """Return all tags with usage counts, including defaults."""
    db = _get_db()
    rows = db.execute("SELECT tags FROM annotations WHERE tags IS NOT NULL").fetchall()

    counts: dict[str, int] = {}
    for tag in _DEFAULT_TAGS:
        counts[tag] = 0

    for r in rows:
        try:
            tag_list = json.loads(r["tags"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(tag_list, list):
            for tag in tag_list:
                if isinstance(tag, str) and tag:
                    counts[tag] = counts.get(tag, 0) + 1

    result = [{"tag": tag, "count": count} for tag, count in counts.items()]
    result.sort(key=lambda x: (-x["count"], x["tag"]))
    return result


# ---------------------------------------------------------------------------
# Message journal
# ---------------------------------------------------------------------------


def ingest_journal_messages(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Accept a batch of content-addressed messages (v2 compat).

    Journal v3 stores messages inline in ``llm_calls.input_skeleton`` /
    ``output_messages``, so the separate ``msg_content`` table is gone.
    This endpoint is kept for backward-compatible clients that still POST
    messages; it simply acknowledges them without persisting.

    Returns ``{"stored": 0}`` unconditionally.
    """
    return {"stored": 0}


def ingest_journal_blocks(session_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist a batch of content-addressed blocks for *session_id*.

    Each item is ``{"hash": "<sha256:...>", "content": "<utf-8 string>"}``.
    Idempotent on ``(session_id, hash)`` — repeated POSTs of the same
    block are a no-op.  Returns ``{"stored": N}`` where ``N`` counts items
    successfully written (existing rows count toward N).

    Called from the single-writer executor thread via main.py.
    """
    if not items:
        return {"stored": 0}
    db = _get_write_db()
    rows = []
    for item in items:
        h = item.get("hash")
        c = item.get("content")
        if not h or c is None:
            continue
        rows.append((session_id, h, c))
    if not rows:
        return {"stored": 0}
    db.executemany(
        "INSERT OR IGNORE INTO msg_blocks (session_id, hash, content) VALUES (?, ?, ?)",
        rows,
    )
    db.commit()
    return {"stored": len(rows)}


def get_session_blocks(session_id: str) -> dict[str, str]:
    """Return ``{hash: content}`` for every block stored under *session_id*.

    Returns ``{}`` when the table is missing (older DBs that predate the
    msg_blocks schema, or in-memory test DBs that didn't run init_db).
    """
    db = _get_db()
    try:
        rows = db.execute(
            "SELECT hash, content FROM msg_blocks WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return {}
        raise
    return {r["hash"]: r["content"] for r in rows}


def ingest_journal_call(call: dict[str, Any]) -> dict[str, Any]:
    """Upsert a single LLM call record.

    Expected keys: ``call_id``, ``session_id``, ``model``, ``ts_start``,
    ``ts_end``, ``input_skeleton`` (list), ``output_messages`` (list),
    ``tokens`` (dict or None), ``span_id`` (str or None).

    Called from the single-writer executor thread via main.py.
    """
    required = ["call_id", "session_id"]
    missing = [k for k in required if k not in call]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    db = _get_write_db()
    try:
        input_skeleton_json = json.dumps(call.get("input_skeleton", []), separators=(",", ":"))
        output_messages_json = json.dumps(call.get("output_messages", []), separators=(",", ":"))
        tokens_json = (
            json.dumps(call["tokens"], separators=(",", ":")) if call.get("tokens") else None
        )
    except (TypeError, ValueError) as e:
        log.warning(
            "ingest_journal_call: JSON serialization failed for call_id=%s: %s",
            call.get("call_id"),
            e,
        )
        raise ValueError(f"Invalid call data: {e}") from e
    import time as _time

    t0 = _time.monotonic()
    db.execute(
        """
        INSERT OR REPLACE INTO llm_calls
            (call_id, session_id, span_id, model, ts_start, ts_end,
             input_skeleton, output_messages, tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call["call_id"],
            call["session_id"],
            call.get("span_id"),
            call.get("model", ""),
            call.get("ts_start"),
            call.get("ts_end"),
            input_skeleton_json,
            output_messages_json,
            tokens_json,
        ),
    )
    db.commit()
    elapsed_ms = (_time.monotonic() - t0) * 1000
    if elapsed_ms > 200:
        log.warning(
            "[journal_call] sqlite commit took %.0fms  call_id=%s",
            elapsed_ms,
            call.get("call_id", "?"),
        )
    return {"ok": True}


def get_session_calls(session_id: str) -> list[dict[str, Any]]:
    """Return all LLM call records for a session, ordered by start time.

    The returned ``input_skeleton`` / ``output_messages`` keys match
    both the column names and the ``/v1/journal/calls`` POST body --
    no rename across the storage / read boundary.  Both values are v3
    skeletons; block bodies live in ``msg_blocks`` and aren't inlined
    here.  ``_augment_span_attrs`` resolves the hashes via
    ``get_session_blocks`` when stamping them onto LLM spans as
    ``llm.input_messages.*`` / ``llm.output_messages.*``.  The
    column-name → OTLP-attribute-name change happens *only* at that
    flattening boundary, not here.
    """
    db = _get_db()
    call_rows = db.execute(
        "SELECT * FROM llm_calls WHERE session_id = ? ORDER BY ts_start, rowid",
        (session_id,),
    ).fetchall()

    if not call_rows:
        return []

    result = []
    for row in call_rows:
        rec: dict[str, Any] = {
            "call_id": row["call_id"],
            "session_id": row["session_id"],
            "model": row["model"],
            "ts_start": row["ts_start"],
            "ts_end": row["ts_end"],
            "input_skeleton": json.loads(row["input_skeleton"]),
            "output_messages": json.loads(row["output_messages"]),
            "tokens": json.loads(row["tokens"]) if row["tokens"] else None,
        }
        if row["span_id"]:
            rec["span_id"] = row["span_id"]
        result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Incremental queries for trace explorer thin-client
# ---------------------------------------------------------------------------


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    """Return lightweight session summary from DB without loading all spans.

    Returns a dict with session_id, span_count, duration_ms, has_errors,
    or None if the session doesn't exist.
    """
    db = _get_db()
    row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        return None

    # Get duration from min/max span timestamps
    time_row = db.execute(
        "SELECT MIN(start_time_ns) as t_start, MAX(end_time_ns) as t_end, COUNT(*) as cnt "
        "FROM spans WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    t_start = time_row["t_start"] or 0
    t_end = time_row["t_end"] or 0
    duration_ms = (t_end - t_start) / 1_000_000 if t_end > t_start else 0.0

    # Check for errors
    error_count = db.execute(
        "SELECT COUNT(*) as cnt FROM spans WHERE session_id = ? AND status_code = 2",
        (session_id,),
    ).fetchone()["cnt"]

    return {
        "session_id": session_id,
        "span_count": time_row["cnt"],
        "duration_ms": duration_ms,
        "has_errors": error_count > 0,
        "error_count": error_count,
        "experiment": row["experiment"],
    }


def get_agent_spans(session_id: str) -> list[dict[str, Any]]:
    """Return only AGENT-kind spans for a session.

    Filters by the openinference.span.kind attribute being AGENT
    at the SQL level using JSON extraction, avoiding full span loading.
    """
    db = _get_db()
    # Check session exists
    exists = db.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not exists:
        return []

    # Filter by attributes containing AGENT kind
    # SQLite JSON: attributes is stored as a JSON array of {key, value} objects
    rows = db.execute(
        "SELECT * FROM spans WHERE session_id = ? "
        "AND attributes LIKE ? "
        "AND attributes LIKE ? "
        "ORDER BY start_time_ns",
        (session_id, "%openinference.span.kind%", "%AGENT%"),
    ).fetchall()

    spans = []
    for r in rows:
        span = _row_to_span(r)
        # Verify it's actually an AGENT span (the LIKE is a coarse filter)
        attrs = json.loads(r["attributes"]) if r["attributes"] else []
        is_agent = any(
            a.get("key") == "openinference.span.kind"
            and a.get("value", {}).get("stringValue") == "AGENT"
            for a in attrs
        )
        if is_agent:
            spans.append(span)
    return spans


def get_error_spans(session_id: str) -> list[dict[str, Any]]:
    """Return spans with error status (status_code=2) for a session.

    Uses SQL filtering on the status_code column - no need to load all spans.
    """
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM spans WHERE session_id = ? AND status_code = 2 ORDER BY start_time_ns",
        (session_id,),
    ).fetchall()

    return [_row_to_span(r) for r in rows]


def get_descendant_spans(session_id: str, root_span_id: str) -> list[dict[str, Any]]:
    """Return a span and all its descendants within a session.

    Walks the parent_span_id tree starting from root_span_id,
    loading only the relevant subtree rather than all session spans.
    Returns OTLP-format span dicts (via _row_to_span).

    Returns empty list if root_span_id is not found in the session.
    """
    db = _get_db()

    # Verify the root span exists in this session
    root_row = db.execute(
        "SELECT * FROM spans WHERE session_id = ? AND span_id = ?",
        (session_id, root_span_id),
    ).fetchone()
    if not root_row:
        return []

    # BFS to collect all descendant span_ids
    collected_rows = [root_row]
    queue = [root_span_id]

    while queue:
        parent_id = queue.pop(0)
        child_rows = db.execute(
            "SELECT * FROM spans WHERE session_id = ? AND parent_span_id = ?",
            (session_id, parent_id),
        ).fetchall()
        for row in child_rows:
            collected_rows.append(row)
            queue.append(row["span_id"])

    return [_row_to_span(r) for r in collected_rows]


def search_spans_fts(session_id: str, query: str, limit: int = 100) -> list[dict[str, Any]]:
    """Search span content using FTS5 full-text search.

    Much faster than LIKE queries on large traces. Falls back gracefully
    if the FTS table doesn't exist or the query is invalid.

    Args:
        session_id: Session to search within.
        query: Search query (FTS5 syntax: supports AND, OR, NOT, phrases).
        limit: Maximum results to return.

    Returns:
        List of matching spans with span_id, name, and a snippet of matching content.
    """
    db = _get_db()
    try:
        # FTS5 MATCH query with snippet extraction
        rows = db.execute(
            "SELECT span_id, name, snippet(spans_fts, 3, '>>>', '<<<', '...', 32) as snippet "
            "FROM spans_fts WHERE session_id = ? AND spans_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (session_id, query, limit),
        ).fetchall()
        return [{"span_id": r["span_id"], "name": r["name"], "snippet": r["snippet"]} for r in rows]
    except Exception:
        # FTS table might not exist or query syntax invalid — return empty
        return []


def backfill_fts(session_id: str | None = None) -> int:
    """Backfill the FTS index for existing spans.

    Populates spans_fts from all spans in the given session (or all sessions
    if session_id is None). Idempotent — clears existing FTS entries for the
    session before re-inserting.

    Args:
        session_id: If provided, backfill only this session. Otherwise backfill all.

    Returns:
        Number of spans indexed.
    """
    db = _get_db()

    # Clear existing FTS entries for the scope
    if session_id:
        db.execute("DELETE FROM spans_fts WHERE session_id = ?", (session_id,))
    else:
        db.execute("DELETE FROM spans_fts")

    # Process in batches to avoid OOM on large databases
    batch_size = 5000
    total = 0

    if session_id:
        cursor = db.execute(
            "SELECT session_id, span_id, name, attributes FROM spans WHERE session_id = ?",
            (session_id,),
        )
    else:
        cursor = db.execute("SELECT session_id, span_id, name, attributes FROM spans")

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        fts_rows = []
        for r in rows:
            attrs_json = r["attributes"]
            content_parts = []
            try:
                attrs = json.loads(attrs_json) if attrs_json else []
                for attr in attrs:
                    val = attr.get("value", {})
                    sv = val.get("stringValue")
                    if sv:
                        content_parts.append(sv)
            except (json.JSONDecodeError, TypeError):
                if attrs_json:
                    content_parts.append(attrs_json)
            fts_rows.append(
                (r["session_id"], r["span_id"], r["name"] or "", " ".join(content_parts))
            )

        db.executemany(
            "INSERT INTO spans_fts (session_id, span_id, name, content) VALUES (?, ?, ?, ?)",
            fts_rows,
        )
        total += len(fts_rows)

    db.commit()
    return total
