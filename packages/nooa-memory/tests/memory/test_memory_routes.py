# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP-layer tests for the Memory tab routes (/api/memory/*).

Exercises the read-only endpoints against a real MemoryStore built in
tmp_path. The TestClient is used without entering its context manager, so the
app lifespan (otlp_store init) never runs — memory routes don't need it.
"""

from __future__ import annotations

import time

import pytest
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.schema import (
    AccessRecord,
    EdgeType,
    Memory,
    MemoryRef,
    MemoryType,
)
from nooa_memory.store import MemoryStore
from starlette.testclient import TestClient

from nooa.viewer import memory_routes
from nooa.viewer.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _serve_from_tmp(tmp_path, monkeypatch):
    """Run each test with cwd=tmp_path — ``?db=`` only accepts paths under cwd."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def memory_db(tmp_path):
    """A real memory sqlite with 4 memories, accesses, an edge and maintenance log."""
    path = tmp_path / "memory.sqlite"
    store = MemoryStore(path)
    embedder = HashingEmbedder(dim=256)
    now = time.time()

    info = Memory(
        id="m-info",
        type=MemoryType.INFO,
        title="Deploy procedure",
        content="Deploy with make ship after the tests pass.",
        owner="alice",
        importance=8.0,
        tags=["deploy"],
    )
    todo = Memory(
        id="m-todo",
        type=MemoryType.TODO,
        content="Fix the flaky retrieval benchmark test.",
        owner="bob",
        status="open",
    )
    episode = Memory(
        id="m-episode",
        type=MemoryType.EPISODE,
        content="Refactored the retrieval pipeline; the benchmark improved by 12%.",
        owner="alice",
        references=[MemoryRef(kind="file", key="docs/spec.md", preview="Spec v1 snapshot")],
    )
    shared = Memory(id="m-shared", content="Prefer uv for package management.", owner="")
    for m in (info, todo, episode, shared):
        store.add(m, embedding=embedder.embed(m.embedding_text()))

    # Structured accesses on the referenced memory: one recall (with trace/session
    # refs + scoring), then a spontaneous injection with no deliberate use after.
    m = store.get("m-episode")
    m.log_access(
        AccessRecord(
            ts=now,
            channel="recalled",
            reader_owner="bob",
            session_ref="sess-1",
            trace_ref="abc123def456",
            query="retrieval benchmark",
            score=0.91,
            rank=0,
            components={"rel": 0.9, "rec": 0.5, "imp": 0.4, "spread": 0.0},
        )
    )
    m.log_access(AccessRecord(ts=now + 1, channel="injected", reader_owner="bob"))
    store.save(m)

    store.add_edge("m-info", "m-episode", EdgeType.RELATED, 0.8)
    store.archive("m-shared")
    store.log_maintenance("reflection", {"merged": 1, "pruned": 0})
    store.log_maintenance(
        "reflect",
        {
            "trigger": "idle",
            "interrupted": True,
            "stopped_in": "form_edges",
            "duration_ms": 12.5,
            "merged": 2,
            "pruned": 0,
        },
    )
    store.close()
    memory_routes.close_stores()  # never reuse a cached handle across tests
    return path


# ---------------------------------------------------------------------------
# /dbs — discovery
# ---------------------------------------------------------------------------


def test_dbs_discovery(client, tmp_path, monkeypatch):
    for rel in (".nooa/memory/project.sqlite", ".nooa/sessions/abc-memory.db"):
        MemoryStore(tmp_path / rel).close()  # MemoryStore mkdir-s parents itself
    # A non-memory session db must NOT be discovered.
    (tmp_path / ".nooa/sessions/abc.db").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    resp = client.get("/api/memory/dbs")
    assert resp.status_code == 200
    dbs = resp.json()
    paths = [d["path"] for d in dbs]
    assert str(tmp_path / ".nooa/memory/project.sqlite") in paths
    assert str(tmp_path / ".nooa/sessions/abc-memory.db") in paths
    assert str(tmp_path / ".nooa/sessions/abc.db") not in paths
    assert all(set(d) == {"path", "size_bytes", "mtime"} for d in dbs)
    assert all(d["size_bytes"] > 0 for d in dbs)


# ---------------------------------------------------------------------------
# /records — listing, filters, pagination
# ---------------------------------------------------------------------------


def test_records_shape(client, memory_db):
    resp = client.get("/api/memory/records", params={"db": str(memory_db)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3  # archived m-shared excluded by default
    assert body["page"] == 1 and body["has_more"] is False
    row = next(r for r in body["records"] if r["id"] == "m-info")
    assert set(row) == {
        "id",
        "type",
        "status",
        "owner",
        "importance",
        "importance_label",
        "title",
        "preview",
        "tags",
        "created_at",
        "last_accessed_at",
        "archived",
        "fetches",
        "edge_count",
    }
    assert row["type"] == "info"
    assert row["title"] == "Deploy procedure"
    assert row["importance_label"] == "HIGH"
    assert row["tags"] == ["deploy"]
    assert row["edge_count"] == 1  # m-info -> m-episode
    episode = next(r for r in body["records"] if r["id"] == "m-episode")
    assert episode["fetches"] == 2  # 1 recalled + 1 injected


def test_records_filters(client, memory_db):
    db = str(memory_db)

    resp = client.get("/api/memory/records", params={"db": db, "include_archived": "true"})
    assert resp.json()["total"] == 4

    resp = client.get("/api/memory/records", params={"db": db, "type": "todo"})
    assert [r["id"] for r in resp.json()["records"]] == ["m-todo"]

    resp = client.get("/api/memory/records", params={"db": db, "status": "open"})
    assert [r["id"] for r in resp.json()["records"]] == ["m-todo"]

    # owner filter matches that owner's rows + unowned shared rows
    resp = client.get("/api/memory/records", params={"db": db, "owner": "alice"})
    assert {r["id"] for r in resp.json()["records"]} == {"m-info", "m-episode"}

    # keyword search hydrates in rank order
    resp = client.get("/api/memory/records", params={"db": db, "q": "retrieval benchmark"})
    ids = {r["id"] for r in resp.json()["records"]}
    assert "m-episode" in ids and "m-todo" in ids and "m-info" not in ids

    resp = client.get("/api/memory/records", params={"db": db, "limit": 2, "page": 1})
    body = resp.json()
    assert len(body["records"]) == 2 and body["has_more"] is True


# ---------------------------------------------------------------------------
# /record — detail
# ---------------------------------------------------------------------------


def test_record_detail(client, memory_db):
    resp = client.get("/api/memory/record", params={"db": str(memory_db), "id": "m-episode"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"].startswith("Refactored the retrieval pipeline")
    assert body["importance_label"] and body["salience_label"] and body["confidence_label"]

    ref = body["references"][0]
    assert ref["kind"] == "file" and ref["key"] == "docs/spec.md"
    assert ref["preview"] == "Spec v1 snapshot"

    channels = [a["channel"] for a in body["access_log"]]
    assert channels == ["created", "recalled", "injected"]
    recalled = body["access_log"][1]
    assert recalled["trace_ref"] == "abc123def456"
    assert recalled["session_ref"] == "sess-1"
    assert recalled["query"] == "retrieval benchmark"
    assert recalled["score"] == 0.91 and recalled["rank"] == 0
    assert recalled["components"]["rel"] == 0.9

    usage = body["usage"]
    assert usage["fetches"] == 2
    assert usage["recalled"] == 1 and usage["injected"] == 1
    assert usage["injected_never_used"] is True  # nothing deliberate after the injection
    assert usage["last_channel"] == "injected"
    assert 0.0 < usage["retention"] <= 1.0
    assert usage["strength"] >= 1


def test_record_detail_edges(client, memory_db):
    resp = client.get("/api/memory/record", params={"db": str(memory_db), "id": "m-info"})
    edges = resp.json()["edges"]
    assert len(edges) == 1
    edge = edges[0]
    assert edge["target_id"] == "m-episode"
    assert edge["type"] == "related" and edge["weight"] == 0.8
    assert edge["target_type"] == "episode"
    assert edge["target_preview"].startswith("Refactored the retrieval pipeline")


def test_record_not_found(client, memory_db):
    resp = client.get("/api/memory/record", params={"db": str(memory_db), "id": "nope"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /stats — dashboard KPIs
# ---------------------------------------------------------------------------


def test_stats(client, memory_db):
    resp = client.get("/api/memory/stats", params={"db": str(memory_db)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["by_type"] == {"info": 1, "todo": 1, "episode": 1}
    assert body["by_owner"] == {"alice": 2, "bob": 1}
    assert body["with_references"] == 1
    assert body["todos_open"] == 1 and body["todos_closed"] == 0
    assert body["total_fetches"] == 2
    assert body["cross_owner_reads"] == 2  # bob read alice's m-episode twice
    maint = body["maintenance"]
    # newest first: the interrupted idle run rides through with its flags
    assert maint[0]["kind"] == "reflect"
    assert maint[0]["report"]["trigger"] == "idle"
    assert maint[0]["report"]["interrupted"] is True
    assert maint[0]["report"]["stopped_in"] == "form_edges"
    assert maint[0]["report"]["duration_ms"] == 12.5
    assert maint[1]["kind"] == "reflection"
    assert maint[1]["report"] == {"merged": 1, "pruned": 0}


# ---------------------------------------------------------------------------
# /explain — retrieval debugger
# ---------------------------------------------------------------------------


def test_explain(client, memory_db):
    resp = client.get(
        "/api/memory/explain",
        params={"db": str(memory_db), "q": "retrieval benchmark", "k": 5},
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert rows, "expected scored candidates"
    assert {
        "rank",
        "id",
        "score",
        "source",
        "cos",
        "rel",
        "rec",
        "imp",
        "spread",
        "type",
        "owner",
        "head",
    } <= set(rows[0])
    assert rows[0]["rank"] == 0
    assert "m-episode" in {r["id"] for r in rows}


def test_explain_dim_mismatch_is_422(client, memory_db):
    resp = client.get(
        "/api/memory/explain",
        params={"db": str(memory_db), "q": "retrieval", "dim": 64},
    )
    assert resp.status_code == 422
    assert "dim" in resp.json()["detail"]


def test_explain_owner_filter_and_wildcard_validation(client, memory_db):
    resp = client.get(
        "/api/memory/explain",
        params={"db": str(memory_db), "q": "retrieval benchmark", "k": 5, "owner": "alice"},
    )
    assert resp.status_code == 200
    assert {row["owner"] for row in resp.json()} <= {"alice", ""}

    for bad in ("ali%", "a_ice"):
        resp = client.get(
            "/api/memory/explain",
            params={"db": str(memory_db), "q": "retrieval", "owner": bad},
        )
        assert resp.status_code == 422, bad
        assert "owner may not contain" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# db param guardrails — no silent fallbacks
# ---------------------------------------------------------------------------


def test_missing_db_is_404(client, tmp_path):
    resp = client.get("/api/memory/records", params={"db": str(tmp_path / "nope.sqlite")})
    assert resp.status_code == 404


def test_bad_suffix_is_422(client, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    resp = client.get("/api/memory/records", params={"db": str(txt)})
    assert resp.status_code == 422


def test_non_sqlite_file_is_422(client, tmp_path):
    fake = tmp_path / "fake.sqlite"
    fake.write_text("this is not a sqlite database at all, padded to 16+ bytes")
    resp = client.get("/api/memory/records", params={"db": str(fake)})
    assert resp.status_code == 422


def test_db_outside_working_dir_is_403(client, tmp_path):
    outside = tmp_path.parent / "outside-working-dir.sqlite"
    MemoryStore(outside).close()
    try:
        resp = client.get("/api/memory/records", params={"db": str(outside)})
        assert resp.status_code == 403
        assert "working directory" in resp.json()["detail"]
    finally:
        outside.unlink()


def test_store_cache_is_bounded_lru(client, tmp_path):
    cap = memory_routes._MAX_STORES
    memory_routes.close_stores()
    for i in range(cap + 1):
        MemoryStore(tmp_path / f"m{i}.sqlite").close()
        # Relative paths resolve against the working directory.
        resp = client.get("/api/memory/stats", params={"db": f"m{i}.sqlite"})
        assert resp.status_code == 200

    assert len(memory_routes._stores) == cap
    assert str((tmp_path / "m0.sqlite").resolve()) not in memory_routes._stores  # LRU evicted
    memory_routes.close_stores()
    assert not memory_routes._stores


def test_owner_like_wildcards_are_422(client, memory_db):
    """% and _ would act as LIKE wildcards in the store's role-scope clause."""
    for bad in ("ali%", "a_ice", "%", "_"):
        resp = client.get("/api/memory/records", params={"db": str(memory_db), "owner": bad})
        assert resp.status_code == 422, bad
        assert "owner may not contain" in resp.json()["detail"]
    # Plain owners keep working.
    resp = client.get("/api/memory/records", params={"db": str(memory_db), "owner": "alice"})
    assert resp.status_code == 200
