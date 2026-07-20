# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the v2 store: owner/status columns, versioned migration,
cross-connection refresh, and owner-scoped read filters."""

import json
import sqlite3

import pytest
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.schema import Memory
from nooa_memory.store import SCHEMA_VERSION, MemorySchemaError, MemoryStore

# The v1 DDL verbatim (pre owner/status), to fabricate legacy files.
_V1_SCHEMA = """
CREATE TABLE memories (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 5.0,
    salience      REAL NOT NULL DEFAULT 0.0,
    strength      INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    archived      INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB,
    data          TEXT NOT NULL
);
CREATE INDEX idx_mem_type ON memories(type);
CREATE INDEX idx_mem_archived ON memories(archived);

CREATE TABLE memory_edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    type       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    PRIMARY KEY (src, dst, type)
);
"""


@pytest.fixture
def emb():
    return HashingEmbedder(dim=128)


def _make_v1_file(path) -> str:
    """Create a legacy (user_version 0) store with one v1-shaped row."""
    conn = sqlite3.connect(path)
    conn.executescript(_V1_SCHEMA)
    payload = {
        "id": "legacy1",
        "type": "info",
        "content": "written by v1 code",
        "importance": 5.0,
        "created_at": 1.0,
        "last_accessed_at": 1.0,
    }
    conn.execute(
        "INSERT INTO memories (id, type, content, importance, created_at,"
        " last_accessed, data) VALUES (?,?,?,?,?,?,?)",
        ("legacy1", "info", "written by v1 code", 5.0, 1.0, 1.0, json.dumps(payload)),
    )
    conn.commit()
    conn.close()
    return "legacy1"


def _add(store, emb, content, **kw):
    m = Memory(content=content, **kw)
    return store.add(m, emb.embed(m.embedding_text()))


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------
def test_v1_file_migrates_in_place(tmp_path, emb):
    path = tmp_path / "legacy.sqlite"
    mid = _make_v1_file(path)

    store = MemoryStore(path)
    version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION

    got = store.get(mid)
    assert got is not None
    assert got.owner == ""  # legacy rows are unowned/shared
    assert got.status is None
    # new writes with owner work on the migrated file
    m = _add(store, emb, "post-migration memory", owner="alice")
    assert store.get(m.id).owner == "alice"
    store.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "legacy.sqlite"
    _make_v1_file(path)
    MemoryStore(path).close()
    store = MemoryStore(path)  # second open: ALTERs must not re-run/fail
    assert store.count() == 1
    store.close()


def test_newer_file_raises(tmp_path):
    path = tmp_path / "future.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    with pytest.raises(MemorySchemaError, match="update nooa"):
        MemoryStore(path)


# ---------------------------------------------------------------------------
# owner column + filters
# ---------------------------------------------------------------------------
def test_owner_roundtrip(emb):
    store = MemoryStore(":memory:")
    m = _add(store, emb, "alice's fact", owner="alice")
    got = store.get(m.id)
    assert got.owner == "alice"
    got.owner = "alice"  # save() keeps the column in sync
    store.save(got)
    assert store.get(m.id).owner == "alice"
    store.close()


def test_owner_filter_matches_own_plus_unowned(emb):
    store = MemoryStore(":memory:")
    a = _add(store, emb, "alice memory", owner="alice")
    b = _add(store, emb, "bob memory", owner="bob")
    shared = _add(store, emb, "shared memory")  # owner=""

    assert store.count() == 3
    assert store.count(owner="alice") == 2
    ids = {m.id for m in store.all_memories(owner="alice")}
    assert ids == {a.id, shared.id}
    assert b.id not in ids

    found = store.keyword_search("memory", 10, owner="alice")
    assert a.id in found and shared.id in found and b.id not in found
    store.close()


def test_knn_owner_filter_oversamples_past_other_owners(emb):
    store = MemoryStore(":memory:")
    # 20 bob memories that dominate the top of the ranking...
    for i in range(20):
        _add(store, emb, "deploy production ship release", owner="bob", title=f"bob{i}")
    # ...and one alice memory slightly further from the query.
    alice = _add(store, emb, "deploy production ship release rollout notes", owner="alice")

    q = emb.embed("deploy production ship release")
    unfiltered = store.knn(q, 4)
    assert alice.id not in [i for i, _ in unfiltered]  # buried below the bobs

    filtered = store.knn(q, 1, owner="alice")
    assert [i for i, _ in filtered] == [alice.id]  # found via oversampling
    store.close()


def test_knn_owner_filter_exhausts_cleanly(emb):
    store = MemoryStore(":memory:")
    _add(store, emb, "only bob here", owner="bob")
    q = emb.embed("only bob here")
    assert store.knn(q, 5, owner="alice") == []
    store.close()


# ---------------------------------------------------------------------------
# cross-connection refresh
# ---------------------------------------------------------------------------
def test_refresh_sees_other_connection_writes(tmp_path, emb):
    path = tmp_path / "shared.sqlite"
    reader = MemoryStore(path)
    writer = MemoryStore(path)

    m = _add(writer, emb, "written by the other connection")
    # The reader's knn must pick up the new row without reopening.
    ranked = reader.knn(emb.embed("written by the other connection"), 1)
    assert ranked and ranked[0][0] == m.id

    reader.close()
    writer.close()


def test_refresh_noop_for_own_writes(emb):
    store = MemoryStore(":memory:")
    _add(store, emb, "self write")
    assert store.refresh_if_changed() is False  # own commits don't trigger reloads
    store.close()


# ---------------------------------------------------------------------------
# owner renaming (legacy-spelling heal)
# ---------------------------------------------------------------------------
def test_rename_owner_restamps_rows(emb):
    store = MemoryStore(":memory:")
    a = _add(store, emb, "legacy row one", owner="pkg.module:TUIAgent")
    b = _add(store, emb, "legacy row two", owner="pkg.module:TUIAgent")
    keep = _add(store, emb, "someone else's row", owner="bob")

    assert store.rename_owner("pkg.module:TUIAgent", "TUIAgent") == 2
    assert store.get(a.id).owner == "TUIAgent"
    assert store.get(b.id).owner == "TUIAgent"
    assert store.get(keep.id).owner == "bob"
    # idempotent: nothing left under the old spelling
    assert store.rename_owner("pkg.module:TUIAgent", "TUIAgent") == 0
    store.close()


def test_rename_owner_never_claims_unowned(emb):
    store = MemoryStore(":memory:")
    _add(store, emb, "shared row")  # owner=""
    assert store.rename_owner("", "TUIAgent") == 0
    assert store.all_memories()[0].owner == ""
    store.close()
