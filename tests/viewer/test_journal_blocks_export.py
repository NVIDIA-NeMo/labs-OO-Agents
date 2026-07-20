# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T5 (viewer): the receiver round-trips blocks + calls into a complete OTLP export.

The journal wire protocol (MR !128) sends spans with their
``llm.input_messages.*`` / ``llm.output_messages.*`` attributes stripped, plus
two sideband POSTs:

    POST /v1/journal/blocks   [{"hash": "<sha>", "content": "<utf-8 string>"}, ...]
    POST /v1/journal/calls    {"call_id", "session_id", "span_id",
                                "input_skeleton": [{"role", "parts":[{"block_hash"}]}],
                                "output_messages": [...]}

The receiver must:
  1.  Persist each block by ``(session_id, hash)``, idempotent.
  2.  Persist the call record.
  3.  When the same session is exported via ``GET /api/trace/export``,
      resolve each ``block_hash`` reference back to its content and stamp
      the resulting messages onto the corresponding LLM span as
      ``llm.input_messages.N.message.*`` attributes.

That's the contract that makes "downloaded jsonl is complete" hold even
when traces went over the journal wire.

These tests fail today because:
  * ``/v1/journal/blocks`` doesn't exist (405) -> blocks never stored.
  * ``export_session_otlp`` doesn't augment from journal calls.
  * Block-hash references in ``input_skeleton`` are never resolved into
    actual ``content`` on the way out.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient

from nooa.viewer import main as main_module
from nooa.viewer import otlp_store
from nooa.viewer.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with a fresh DB (mirrors ``tests/viewer/test_main.py::client``)."""
    monkeypatch.setattr(otlp_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(otlp_store, "_db", None)
    monkeypatch.setattr(
        main_module,
        "_write_executor",
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-writer-test"),
    )
    if hasattr(otlp_store._read_tls, "conn"):
        del otlp_store._read_tls.conn
    if hasattr(otlp_store._write_tls, "conn"):
        del otlp_store._write_tls.conn
    with TestClient(app) as c:
        db = otlp_store._db
        otlp_store._read_tls.conn = db
        otlp_store._write_tls.conn = db
        yield c
    if hasattr(otlp_store._read_tls, "conn"):
        del otlp_store._read_tls.conn
    if hasattr(otlp_store._write_tls, "conn"):
        del otlp_store._write_tls.conn


def _hash(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _seed_session(db: sqlite3.Connection, session_id: str) -> None:
    db.execute(
        "INSERT INTO sessions (session_id, experiment, span_count, modified) "
        "VALUES (?, 'default', 0, 0)",
        (session_id,),
    )
    db.commit()


def _seed_stripped_llm_span(db: sqlite3.Connection, *, session_id: str, span_id: str) -> None:
    """Inject the wire-shape stripped span directly into ``spans``.

    Bypasses the ``/v1/traces`` ingest queue (which TestClient + the worker
    task can deadlock on) so the test stays focused on the journal layer.
    """
    attrs = [
        {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
        {"key": "session.id", "value": {"stringValue": session_id}},
    ]
    db.execute(
        "INSERT INTO spans (session_id, trace_id, span_id, parent_span_id, "
        "name, kind, start_time_ns, end_time_ns, status_code, "
        "status_message, attributes, resource, events) "
        "VALUES (?, ?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL)",
        (
            session_id,
            "0" * 32,
            span_id,
            "acompletion",
            3,
            1,
            2,
            json.dumps(attrs),
            json.dumps({}),
        ),
    )
    db.commit()


def _attrs_dict(span: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in span.get("attributes", []):
        v = a.get("value", {})
        out[a["key"]] = v.get("stringValue", json.dumps(v))
    return out


# ---------------------------------------------------------------------------
# T5a — /v1/journal/blocks endpoint exists and persists per-session
# ---------------------------------------------------------------------------


class TestBlocksEndpoint:
    def test_post_blocks_returns_200(self, client):
        """The block-sideband POST must be accepted (currently 405)."""
        r = client.post(
            "/v1/journal/blocks",
            json=[
                {"hash": _hash("hello"), "content": "hello"},
                {"hash": _hash("world"), "content": "world"},
            ],
            headers={"X-Session-Id": "sess-blocks-1"},
        )
        assert r.status_code == 200, (
            f"POST /v1/journal/blocks should accept the wire payload but got "
            f"{r.status_code}: {r.text!r}.  This route does not exist on the "
            f"viewer today; the journal exporter posts to it and gets 405, "
            f"silently dropping all block content."
        )

    def test_repeated_post_is_idempotent(self, client):
        """Idempotency on ``(session_id, hash)`` so retries don't duplicate."""
        payload = [{"hash": _hash("dup"), "content": "dup"}]
        headers = {"X-Session-Id": "sess-dup"}
        r1 = client.post("/v1/journal/blocks", json=payload, headers=headers)
        r2 = client.post("/v1/journal/blocks", json=payload, headers=headers)
        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text


# ---------------------------------------------------------------------------
# T5b — viewer download reconstructs full messages from journal sideband
# ---------------------------------------------------------------------------


class TestExportReconstructsMessages:
    def test_export_resolves_block_hashes_into_message_content(self, client):
        """After blocks + a call referencing them land, ``GET /api/trace/export``
        returns OTLP whose LLM span carries the original message text as
        ``llm.input_messages.*`` / ``llm.output_messages.*`` -- block hashes
        resolved to content."""
        session_id = "sess-roundtrip"
        span_id = "1234567890abcdef"

        input_text = "T5_INPUT what's the weather"
        output_text = "T5_OUTPUT sunny"
        h_in = _hash(input_text)
        h_out = _hash(output_text)

        db = otlp_store._get_db()
        _seed_session(db, session_id)
        _seed_stripped_llm_span(db, session_id=session_id, span_id=span_id)

        # 1. Blocks land via the wire endpoint.
        rb = client.post(
            "/v1/journal/blocks",
            json=[
                {"hash": h_in, "content": input_text},
                {"hash": h_out, "content": output_text},
            ],
            headers={"X-Session-Id": session_id},
        )
        assert rb.status_code == 200, rb.text

        # 2. Call references those blocks.
        rc = client.post(
            "/v1/journal/calls",
            json={
                "call_id": "c-1",
                "session_id": session_id,
                "span_id": span_id,
                "model": "gpt-3.5-turbo",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "input_skeleton": [
                    {"role": "user", "parts": [{"block_hash": h_in, "key": "u"}]},
                ],
                "output_messages": [
                    {"role": "assistant", "parts": [{"block_hash": h_out, "key": "o"}]},
                ],
            },
        )
        assert rc.status_code == 200, rc.text

        # 3. Export.
        re = client.get(f"/api/trace/export?session_id={session_id}")
        assert re.status_code == 200, re.text
        bodies = [json.loads(line) for line in re.text.splitlines() if line.strip()]
        spans = [
            sp
            for body in bodies
            for rs in body.get("resourceSpans", [])
            for ss in rs.get("scopeSpans", [])
            for sp in ss.get("spans", [])
        ]
        target = next((s for s in spans if s.get("spanId") == span_id), None)
        assert target is not None, (
            f"export missing span {span_id}: returned span ids {[s.get('spanId') for s in spans]}"
        )
        attrs = _attrs_dict(target)

        in_role_key = "llm.input_messages.0.message.role"
        in_content_key = "llm.input_messages.0.message.content"
        out_role_key = "llm.output_messages.0.message.role"
        out_content_key = "llm.output_messages.0.message.content"

        assert attrs.get(in_role_key) == "user", (
            f"export missing/incorrect input role.  attrs={attrs!r}"
        )
        assert attrs.get(in_content_key) == input_text, (
            f"export did not resolve input block hash to content.\n"
            f"  expected: {input_text!r}\n"
            f"  got:      {attrs.get(in_content_key)!r}\n"
            f"  full attrs: {attrs!r}\n"
            f"This is the missing /v1/journal/blocks storage + the missing "
            f"augmentation in export_session_otlp."
        )
        assert attrs.get(out_role_key) == "assistant"
        assert attrs.get(out_content_key) == output_text, (
            f"export did not resolve output block hash to content. "
            f"got: {attrs.get(out_content_key)!r}"
        )
