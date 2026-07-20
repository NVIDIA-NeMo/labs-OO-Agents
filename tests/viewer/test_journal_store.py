# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the content-addressed message journal in otlp_store.

Covers:
- _flatten_msg_to_attrs: plain messages, null content, tool_calls, tool role
- _augment_span_attrs: replaces input/output message attrs, preserves others
- ingest_journal_call: stores span_id, round-trip via get_session_calls
- get_session_spans(augment=True): augments LLM spans, skips non-LLM spans
- Migration: span_id column added to existing llm_calls table
- 100-message roundtrip integration test
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_attr(key: str, val: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": val}}


def _make_db() -> sqlite3.Connection:
    """Create an in-memory DB with the full journal schema."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE llm_calls (
            call_id       TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            span_id       TEXT,
            model         TEXT,
            ts_start      REAL,
            ts_end        REAL,
            input_skeleton   TEXT NOT NULL,
            output_messages  TEXT NOT NULL,
            tokens        TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id);
        CREATE INDEX IF NOT EXISTS idx_llm_calls_span ON llm_calls(span_id);

        -- v3: content-addressed message blocks resolved on read.
        CREATE TABLE msg_blocks (
            session_id  TEXT NOT NULL,
            hash        TEXT NOT NULL,
            content     TEXT NOT NULL,
            PRIMARY KEY (session_id, hash)
        );

        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            experiment TEXT NOT NULL,
            span_count INTEGER DEFAULT 0,
            modified REAL DEFAULT 0,
            resource_attrs TEXT,
            eval_passed INTEGER,
            eval_metadata TEXT
        );

        CREATE TABLE spans (
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
            events TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
    """)
    db.commit()
    return db


@pytest.fixture
def store(monkeypatch):
    """Patch otlp_store._db with an in-memory DB, return the module.

    Also patches the thread-local _read_tls and _write_tls so that
    _get_db() and _get_write_db() return the same in-memory connection
    instead of opening file-based connections from DB_PATH.
    """
    import nooa.viewer.otlp_store as store_mod

    db = _make_db()
    monkeypatch.setattr(store_mod, "_db", db)
    # _get_db() and _get_write_db() check thread-local .conn attrs
    store_mod._read_tls.conn = db
    store_mod._write_tls.conn = db
    yield store_mod
    # Clean up thread-local state
    if hasattr(store_mod._read_tls, "conn"):
        del store_mod._read_tls.conn
    if hasattr(store_mod._write_tls, "conn"):
        del store_mod._write_tls.conn


# ---------------------------------------------------------------------------
# _flatten_msg_to_attrs
# ---------------------------------------------------------------------------


class TestFlattenMsgToAttrs:
    def test_plain_user_message(self, store):
        msg = {"role": "user", "content": "hello"}
        attrs = store._flatten_msg_to_attrs(msg, "llm.input_messages.0.message")
        keys = [a["key"] for a in attrs]
        assert "llm.input_messages.0.message.role" in keys
        assert "llm.input_messages.0.message.content" in keys
        role_attr = next(a for a in attrs if a["key"].endswith(".role"))
        assert role_attr["value"]["stringValue"] == "user"

    def test_null_content_omitted(self, store):
        msg = {"role": "assistant", "content": None}
        attrs = store._flatten_msg_to_attrs(msg, "llm.input_messages.0.message")
        keys = [a["key"] for a in attrs]
        assert not any("content" in k for k in keys)

    def test_tool_role_with_tool_call_id(self, store):
        msg = {"role": "tool", "tool_call_id": "call_abc", "content": "result"}
        attrs = store._flatten_msg_to_attrs(msg, "llm.input_messages.2.message")
        keys = [a["key"] for a in attrs]
        assert "llm.input_messages.2.message.tool_call_id" in keys
        assert "llm.input_messages.2.message.content" in keys

    def test_tool_calls_expanded(self, store):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "search", "arguments": '{"q":"hi"}'}},
                {"id": "call_2", "function": {"name": "fetch", "arguments": "{}"}},
            ],
        }
        attrs = store._flatten_msg_to_attrs(msg, "llm.input_messages.1.message")
        keys = [a["key"] for a in attrs]
        assert "llm.input_messages.1.message.tool_calls.0.tool_call.id" in keys
        assert "llm.input_messages.1.message.tool_calls.0.tool_call.function.name" in keys
        assert "llm.input_messages.1.message.tool_calls.1.tool_call.id" in keys

    def test_system_message(self, store):
        msg = {"role": "system", "content": "You are helpful."}
        attrs = store._flatten_msg_to_attrs(msg, "llm.input_messages.0.message")
        content_attr = next(a for a in attrs if "content" in a["key"])
        assert content_attr["value"]["stringValue"] == "You are helpful."

    def test_no_tool_calls_key_omits_tool_calls(self, store):
        msg = {"role": "assistant", "content": "hi"}
        attrs = store._flatten_msg_to_attrs(msg, "llm.output_messages.0.message")
        assert not any("tool_calls" in a["key"] for a in attrs)


# ---------------------------------------------------------------------------
# _augment_span_attrs
# ---------------------------------------------------------------------------


class TestAugmentSpanAttrs:
    def _make_call(self, input_msgs, output_msgs):
        return {"input_skeleton": input_msgs, "output_messages": output_msgs}

    def test_replaces_input_and_output_message_attrs(self, store):
        existing = [
            _str_attr("llm.input_messages.0.message.role", "user"),
            _str_attr("llm.input_messages.0.message.content", "old truncated"),
            _str_attr("openinference.span.kind", "LLM"),
        ]
        call = self._make_call(
            [{"role": "user", "content": "full content"}],
            [{"role": "assistant", "content": "reply"}],
        )
        result = store._augment_span_attrs(existing, call)
        keys = [a["key"] for a in result]
        # Old attr preserved (non-message)
        assert "openinference.span.kind" in keys
        # New input content
        content_attrs = [a for a in result if "input_messages.0.message.content" in a["key"]]
        assert content_attrs[0]["value"]["stringValue"] == "full content"
        # Output messages added
        assert any("output_messages.0.message.role" in k for k in keys)

    def test_preserves_non_message_attrs(self, store):
        existing = [
            _str_attr("openinference.span.kind", "LLM"),
            _str_attr("llm.model_name", "gpt-4o"),
            _str_attr("llm.input_messages.0.message.role", "user"),
        ]
        call = self._make_call([{"role": "user", "content": "hi"}], [])
        result = store._augment_span_attrs(existing, call)
        keys = [a["key"] for a in result]
        assert "openinference.span.kind" in keys
        assert "llm.model_name" in keys

    def test_handles_empty_messages(self, store):
        existing = [_str_attr("openinference.span.kind", "LLM")]
        call = self._make_call([], [])
        result = store._augment_span_attrs(existing, call)
        assert result == existing

    def test_null_msg_entry_skipped_dense_reindex(self, store):
        """None entries (hash miss) are skipped; remaining messages are densely re-indexed."""
        existing = [_str_attr("openinference.span.kind", "LLM")]
        call = self._make_call([None, {"role": "user", "content": "hi"}], [])
        result = store._augment_span_attrs(existing, call)
        keys = [a["key"] for a in result]
        # None at position 0 is skipped; the user message is re-indexed to 0
        assert "llm.input_messages.0.message.role" in keys
        # There is no index 1 (only one non-None message)
        assert not any("input_messages.1" in k for k in keys)


# ---------------------------------------------------------------------------
# ingest_journal_messages
# ---------------------------------------------------------------------------


class TestIngestJournalMessages:
    """v3: ingest_journal_messages is a backward-compat no-op (messages are inline in llm_calls)."""

    def test_stores_messages(self, store):
        items = [
            {"h": "sha256:aaa", "msg": {"role": "user", "content": "hi"}},
            {"h": "sha256:bbb", "msg": {"role": "assistant", "content": "hello"}},
        ]
        result = store.ingest_journal_messages(items)
        assert result == {"stored": 0}

    def test_empty_list_returns_stored_zero(self, store):
        """Empty batch returns {stored: 0} without touching the DB."""
        result = store.ingest_journal_messages([])
        assert result == {"stored": 0}

    def test_dedup_insert_or_ignore(self, store):
        items = [{"h": "sha256:dup", "msg": {"role": "user", "content": "x"}}]
        result1 = store.ingest_journal_messages(items)
        result2 = store.ingest_journal_messages(items)
        assert result1 == {"stored": 0}
        assert result2 == {"stored": 0}

    def test_message_json_round_trips(self, store):
        """v3: messages round-trip via input_skeleton in llm_calls, not msg_content."""
        msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}
        store.ingest_journal_call(
            {
                "call_id": "call_rt",
                "session_id": "sess_rt",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [msg],
                "output_messages": [],
            }
        )
        calls = store.get_session_calls("sess_rt")
        assert calls[0]["input_skeleton"][0] == msg


# ---------------------------------------------------------------------------
# ingest_journal_call / get_session_calls
# ---------------------------------------------------------------------------


class TestIngestJournalCall:
    def test_stores_and_retrieves_call(self, store):
        store.ingest_journal_call(
            {
                "call_id": "cid1",
                "session_id": "sess1",
                "model": "gpt-4o",
                "ts_start": 1000.0,
                "ts_end": 1001.0,
                "input_skeleton": [{"role": "user", "content": "hi"}],
                "output_messages": [{"role": "assistant", "content": "hello"}],
                "tokens": {"prompt": 10, "completion": 5},
                "span_id": "abc123",
            }
        )
        calls = store.get_session_calls("sess1")
        assert len(calls) == 1
        assert calls[0]["call_id"] == "cid1"
        assert calls[0]["span_id"] == "abc123"
        assert calls[0]["input_skeleton"] == [{"role": "user", "content": "hi"}]
        assert calls[0]["output_messages"] == [{"role": "assistant", "content": "hello"}]
        assert calls[0]["tokens"] == {"prompt": 10, "completion": 5}

    def test_span_id_is_none_when_not_provided(self, store):
        store.ingest_journal_call(
            {
                "call_id": "cid2",
                "session_id": "sess2",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [],
                "output_messages": [],
            }
        )
        calls = store.get_session_calls("sess2")
        assert len(calls) == 1
        assert "span_id" not in calls[0]

    def test_same_call_id_upserts(self, store):
        """ingest_journal_call uses INSERT OR REPLACE — second call overwrites first."""
        store.ingest_journal_call(
            {
                "call_id": "cid_dup",
                "session_id": "sess_dup",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [],
                "output_messages": [],
                "span_id": "span_first",
            }
        )
        store.ingest_journal_call(
            {
                "call_id": "cid_dup",
                "session_id": "sess_dup",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [],
                "output_messages": [],
                "span_id": "span_second",
            }
        )
        calls = store.get_session_calls("sess_dup")
        assert len(calls) == 1
        assert calls[0]["span_id"] == "span_second"

    def test_empty_session_returns_empty(self, store):
        calls = store.get_session_calls("nonexistent")
        assert calls == []

    def test_inline_messages_round_trip(self, store):
        """v3: messages stored inline in input_skeleton are returned as-is."""
        msg = {"role": "user", "content": "hello"}
        store.ingest_journal_call(
            {
                "call_id": "cid3",
                "session_id": "sess3",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [msg],
                "output_messages": [],
            }
        )
        calls = store.get_session_calls("sess3")
        assert calls[0]["input_skeleton"] == [msg]

    def test_multiple_calls_ordered_by_ts_start(self, store):
        for i, ts in [(2, 200.0), (1, 100.0), (3, 300.0)]:
            store.ingest_journal_call(
                {
                    "call_id": f"cid{i}",
                    "session_id": "sess_order",
                    "model": "gpt-4o",
                    "ts_start": ts,
                    "ts_end": ts + 1,
                    "input_skeleton": [],
                    "output_messages": [],
                }
            )
        calls = store.get_session_calls("sess_order")
        assert [c["call_id"] for c in calls] == ["cid1", "cid2", "cid3"]

    def test_compound_system_message_roundtrip(self, store):
        """v3: compound system messages are stored inline (no block assembly)."""
        sys_msg = {
            "role": "system",
            "content": "<notes>\nFirst note.\n</notes>\n\n<status>\nIdle.\n</status>",
        }
        user_msg = {"role": "user", "content": "hello"}
        store.ingest_journal_call(
            {
                "call_id": "cid_blocks",
                "session_id": "sess_blocks",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [sys_msg, user_msg],
                "output_messages": [],
            }
        )
        calls = store.get_session_calls("sess_blocks")
        assert len(calls) == 1
        assert calls[0]["input_skeleton"][0] == sys_msg
        assert calls[0]["input_skeleton"][1] == user_msg


# ---------------------------------------------------------------------------
# _get_journal_calls_by_span
# ---------------------------------------------------------------------------


class TestGetJournalCallsBySpan:
    def test_returns_dict_keyed_by_span_id(self, store):
        store.ingest_journal_call(
            {
                "call_id": "c1",
                "session_id": "sess_span",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [{"role": "user", "content": "hi"}],
                "output_messages": [],
                "span_id": "sp_abc",
            }
        )
        result = store._get_journal_calls_by_span("sess_span")
        assert "sp_abc" in result
        assert result["sp_abc"]["call_id"] == "c1"

    def test_calls_without_span_id_excluded(self, store):
        store.ingest_journal_call(
            {
                "call_id": "c_nospam",
                "session_id": "sess_nospam",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [],
                "output_messages": [],
            }
        )
        result = store._get_journal_calls_by_span("sess_nospam")
        assert result == {}

    def test_empty_session_returns_empty_dict(self, store):
        result = store._get_journal_calls_by_span("no_such_session")
        assert result == {}


# ---------------------------------------------------------------------------
# get_session_spans — augmentation
# ---------------------------------------------------------------------------


def _seed_session(store, session_id: str = "sess_aug") -> None:
    db = store._get_db()
    db.execute(
        "INSERT INTO sessions (session_id, experiment, span_count, modified) VALUES (?, 'default', 1, 0)",
        (session_id,),
    )
    db.commit()


def _make_llm_span_attrs(messages: list[dict], *, truncated: bool = False) -> list[dict]:
    """Build OTLP attribute list for an LLM span (only first N messages if truncated)."""
    attrs = [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}]
    msgs = messages[:3] if truncated else messages
    for i, msg in enumerate(msgs):
        prefix = f"llm.input_messages.{i}.message"
        attrs.append({"key": f"{prefix}.role", "value": {"stringValue": msg["role"]}})
        if msg.get("content"):
            attrs.append({"key": f"{prefix}.content", "value": {"stringValue": msg["content"]}})
    return attrs


class TestGetSessionSpansAugment:
    def test_non_llm_span_untouched(self, store):
        _seed_session(store)
        db = store._get_db()
        attrs = [
            {"key": "openinference.span.kind", "value": {"stringValue": "CHAIN"}},
            {"key": "some.attr", "value": {"stringValue": "val"}},
        ]
        db.execute(
            """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
               attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("sess_aug", "span_chain", "chain_op", 1, 0, 1, json.dumps(attrs), "{}", "[]"),
        )
        db.commit()
        spans = store.get_session_spans("sess_aug", augment=True)
        span = next(s for s in spans if s["spanId"] == "span_chain")
        # attrs unchanged
        assert span["attributes"] == attrs

    def test_llm_span_without_journal_call_untouched(self, store):
        _seed_session(store)
        db = store._get_db()
        attrs = _make_llm_span_attrs([{"role": "user", "content": "hi"}])
        db.execute(
            """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
               attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("sess_aug", "span_llm_no_journal", "llm_call", 1, 0, 1, json.dumps(attrs), "{}", "[]"),
        )
        db.commit()
        spans = store.get_session_spans("sess_aug", augment=True)
        span = next(s for s in spans if s["spanId"] == "span_llm_no_journal")
        assert span["attributes"] == attrs

    def test_llm_span_augmented_with_journal_messages(self, store):
        _seed_session(store)
        store.ingest_journal_call(
            {
                "call_id": "cid_aug",
                "session_id": "sess_aug",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [{"role": "user", "content": "what is 2+2?"}],
                "output_messages": [{"role": "assistant", "content": "4"}],
                "span_id": "span_aug",
            }
        )

        # Seed span with truncated attrs (missing output_messages)
        db = store._get_db()
        attrs = [
            {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
            {"key": "llm.input_messages.0.message.role", "value": {"stringValue": "user"}},
        ]
        db.execute(
            """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
               attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("sess_aug", "span_aug", "llm_call", 1, 0, 1, json.dumps(attrs), "{}", "[]"),
        )
        db.commit()

        spans = store.get_session_spans("sess_aug", augment=True)
        span = next(s for s in spans if s["spanId"] == "span_aug")
        keys = [a["key"] for a in span["attributes"]]

        # Output messages now present (from journal)
        assert any("output_messages.0.message.role" in k for k in keys)
        # Content reconstructed
        content_attr = next(
            a for a in span["attributes"] if a["key"] == "llm.input_messages.0.message.content"
        )
        assert content_attr["value"]["stringValue"] == "what is 2+2?"
        # span.kind preserved
        assert "openinference.span.kind" in keys

    def test_llm_span_augmented_with_compound_system_message(self, store):
        """v3: system message stored inline is augmented into span attrs."""
        session_id = "sess_compound_aug"
        span_id = "span_compound_aug"
        _seed_session(store, session_id)

        sys_content = "<persona>\nYou are helpful.\n</persona>\n\n<notes>\nSome notes.\n</notes>"
        sys_msg = {"role": "system", "content": sys_content}
        user_msg = {"role": "user", "content": "hello"}

        store.ingest_journal_call(
            {
                "call_id": "cid_compound_aug",
                "session_id": session_id,
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [sys_msg, user_msg],
                "output_messages": [],
                "span_id": span_id,
            }
        )

        # Span has no useful message attrs (simulating truncation / placeholder)
        db = store._get_db()
        attrs = [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}]
        db.execute(
            """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
               attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, span_id, "llm_call", 1, 0, 1, json.dumps(attrs), "{}", "[]"),
        )
        db.commit()

        spans = store.get_session_spans(session_id, augment=True)
        span = next(s for s in spans if s["spanId"] == span_id)
        keys = [a["key"] for a in span["attributes"]]

        # System message content present
        sys_content_attr = next(
            (a for a in span["attributes"] if a["key"] == "llm.input_messages.0.message.content"),
            None,
        )
        assert sys_content_attr is not None, "System message content missing after augmentation"
        assert "<persona>" in sys_content_attr["value"]["stringValue"]
        assert "<notes>" in sys_content_attr["value"]["stringValue"]

        # User message present at index 1
        assert "llm.input_messages.1.message.role" in keys

    def test_mixed_session_only_matching_span_augmented(self, store):
        """In a session with two LLM spans, only the one with a journal call is augmented."""
        _seed_session(store, "sess_mixed")

        store.ingest_journal_call(
            {
                "call_id": "cid_mix",
                "session_id": "sess_mixed",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [{"role": "user", "content": "full content"}],
                "output_messages": [{"role": "assistant", "content": "full reply"}],
                "span_id": "span_with_journal",
            }
        )

        db = store._get_db()
        # Span with matching journal call — has truncated attrs
        attrs_with = [
            {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
            {"key": "llm.input_messages.0.message.role", "value": {"stringValue": "user"}},
            {"key": "llm.input_messages.0.message.content", "value": {"stringValue": "TRUNCATED"}},
        ]
        # Span without journal call
        attrs_without = [
            {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
            {"key": "llm.input_messages.0.message.role", "value": {"stringValue": "user"}},
            {"key": "llm.input_messages.0.message.content", "value": {"stringValue": "unchanged"}},
        ]
        for sid, attrs in [
            ("span_with_journal", attrs_with),
            ("span_without_journal", attrs_without),
        ]:
            db.execute(
                """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
                   attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("sess_mixed", sid, "llm_call", 1, 0, 1, json.dumps(attrs), "{}", "[]"),
            )
        db.commit()

        spans = store.get_session_spans("sess_mixed", augment=True)
        span_with = next(s for s in spans if s["spanId"] == "span_with_journal")
        span_without = next(s for s in spans if s["spanId"] == "span_without_journal")

        # Augmented span has full content from journal
        content_with = next(
            a for a in span_with["attributes"] if "content" in a["key"] and "input" in a["key"]
        )
        assert content_with["value"]["stringValue"] == "full content"

        # Non-augmented span retains original truncated content
        content_without = next(
            a for a in span_without["attributes"] if "content" in a["key"] and "input" in a["key"]
        )
        assert content_without["value"]["stringValue"] == "unchanged"

        # Augmented span also has output messages from journal
        assert any("output_messages" in a["key"] for a in span_with["attributes"])
        # Non-augmented span has no output messages
        assert not any("output_messages" in a["key"] for a in span_without["attributes"])

    def test_augment_false_returns_raw(self, store):
        _seed_session(store, "sess_raw")
        store.ingest_journal_call(
            {
                "call_id": "cid_raw",
                "session_id": "sess_raw",
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [{"role": "user", "content": "hi"}],
                "output_messages": [],
                "span_id": "span_raw",
            }
        )
        db = store._get_db()
        attrs = [
            {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
            {"key": "llm.input_messages.0.message.role", "value": {"stringValue": "user"}},
            {"key": "llm.input_messages.0.message.content", "value": {"stringValue": "TRUNCATED"}},
        ]
        db.execute(
            """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
               attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("sess_raw", "span_raw", "llm_call", 1, 0, 1, json.dumps(attrs), "{}", "[]"),
        )
        db.commit()

        spans = store.get_session_spans("sess_raw", augment=False)
        span = next(s for s in spans if s["spanId"] == "span_raw")
        content_attr = next(a for a in span["attributes"] if "content" in a["key"])
        assert content_attr["value"]["stringValue"] == "TRUNCATED"


# ---------------------------------------------------------------------------
# Migration: span_id column on existing llm_calls
# ---------------------------------------------------------------------------


class TestMigrationSpanId:
    def test_init_db_adds_span_id_if_missing(self, tmp_path, monkeypatch):
        import nooa.viewer.otlp_store as store_mod

        db_path = tmp_path / "old.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db_path)
        monkeypatch.setattr(store_mod, "_db", None)

        # Create an old-style DB without span_id column
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE llm_calls (
                call_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                model TEXT,
                ts_start REAL,
                ts_end REAL,
                input_hashes TEXT NOT NULL,
                output_hashes TEXT NOT NULL,
                tokens TEXT
            )
        """)
        old_db.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, experiment TEXT NOT NULL, span_count INTEGER DEFAULT 0, modified REAL DEFAULT 0, resource_attrs TEXT, eval_passed INTEGER, eval_metadata TEXT)"
        )
        old_db.execute(
            "CREATE TABLE spans (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, trace_id TEXT, span_id TEXT, parent_span_id TEXT, name TEXT, kind INTEGER, start_time_ns INTEGER, end_time_ns INTEGER, status_code INTEGER, status_message TEXT, attributes TEXT, resource TEXT, events TEXT)"
        )
        old_db.execute("CREATE TABLE msg_content (hash TEXT PRIMARY KEY, msg TEXT NOT NULL)")
        old_db.execute(
            "CREATE TABLE annotations (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, span_id TEXT, target TEXT, name TEXT NOT NULL, score REAL, label TEXT, comment TEXT, tags TEXT, created_at TEXT NOT NULL, author_id TEXT, source TEXT NOT NULL DEFAULT 'human', metadata TEXT)"
        )
        old_db.commit()
        old_db.close()

        store_mod.init_db()

        db = store_mod._get_db()
        assert store_mod._has_column(db, "llm_calls", "span_id")

        # Cleanup
        monkeypatch.setattr(store_mod, "_db", None)


# ---------------------------------------------------------------------------
# 100-message roundtrip integration test
# ---------------------------------------------------------------------------


class TestHundredMessageRoundtrip:
    def test_100_message_roundtrip(self, store):
        """Seed 100 messages inline; span has only first 10; augment returns all 100."""
        session_id = "sess_100"
        span_id = "span_100"

        _seed_session(store, session_id)

        # Build 100 unique messages
        all_msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg_{i:03d}"}
            for i in range(100)
        ]

        # Seed call record with all 100 messages inline (v3)
        store.ingest_journal_call(
            {
                "call_id": "call_100",
                "session_id": session_id,
                "model": "gpt-4o",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": all_msgs,
                "output_messages": [],
                "span_id": span_id,
            }
        )

        # Seed span with only first 10 messages (simulating OTel truncation)
        truncated_attrs = [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}]
        for i, msg in enumerate(all_msgs[:10]):
            p = f"llm.input_messages.{i}.message"
            truncated_attrs.append({"key": f"{p}.role", "value": {"stringValue": msg["role"]}})
            truncated_attrs.append(
                {"key": f"{p}.content", "value": {"stringValue": msg["content"]}}
            )

        db = store._get_db()
        db.execute(
            """INSERT INTO spans (session_id, span_id, name, kind, start_time_ns, end_time_ns,
               attributes, resource, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, span_id, "llm_call", 1, 0, 1, json.dumps(truncated_attrs), "{}", "[]"),
        )
        db.commit()

        spans = store.get_session_spans(session_id, augment=True)
        span = next(s for s in spans if s["spanId"] == span_id)

        # Count reconstructed input message roles
        input_role_attrs = [
            a
            for a in span["attributes"]
            if "llm.input_messages." in a["key"] and a["key"].endswith(".role")
        ]
        assert len(input_role_attrs) == 100, (
            f"Expected 100 input message role attrs, got {len(input_role_attrs)}"
        )

        # Verify a message from the middle (would have been truncated)
        content_50 = next(
            (a for a in span["attributes"] if a["key"] == "llm.input_messages.50.message.content"),
            None,
        )
        assert content_50 is not None, "Message 50 missing after augmentation"
        assert content_50["value"]["stringValue"] == "msg_050"
