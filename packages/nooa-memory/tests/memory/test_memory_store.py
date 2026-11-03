# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SQLite-centric memory store + numpy vector index."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from nooa_memory.config import ForgetPolicy, ReflectionPolicy
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.reflection import ReflectionEngine
from nooa_memory.schema import EdgeType, Memory, MemoryType
from nooa_memory.store import MemoryStore, NumpyVectorIndex


@pytest.fixture
def emb():
    return HashingEmbedder(dim=128)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _add(store, emb, content, **kw):
    m = Memory(content=content, **kw)
    return store.add(m, emb.embed(m.embedding_text()))


def test_add_get_roundtrip(store, emb):
    m = _add(store, emb, "deploy uses make ship", type=MemoryType.SKILL, importance=7.0)
    got = store.get(m.id)
    assert got is not None
    assert got.content == "deploy uses make ship"
    assert got.type == MemoryType.SKILL
    assert got.importance == 7.0


def test_save_persists_mutation(store, emb):
    m = _add(store, emb, "fact")
    m.touch()
    m.importance = 9.0
    store.save(m)
    got = store.get(m.id)
    assert got.importance == 9.0
    assert got.access_count == 1


def test_edges_roundtrip(store, emb):
    a = _add(store, emb, "alpha")
    b = _add(store, emb, "beta")
    a.add_edge(b.id, EdgeType.CAUSES, 0.8)
    store.save(a)
    got = store.get(a.id)
    assert any(e.target_id == b.id and e.type == EdgeType.CAUSES for e in got.edges)
    assert any(e.target_id == b.id for e in store.neighbors(a.id))


def test_add_edge_method(store, emb):
    a = _add(store, emb, "a")
    b = _add(store, emb, "b")
    store.add_edge(a.id, b.id, EdgeType.RELATED, 0.5)
    assert store.neighbors(a.id)[0].target_id == b.id


def test_knn_returns_nearest_first(store, emb):
    _add(store, emb, "kubernetes pods crash loop backoff")
    target = _add(store, emb, "deploy ship release production rollout")
    _add(store, emb, "totally different banana mango fruit")
    q = emb.embed("how to deploy and ship a release to production")
    ranked = store.knn(q, 3)
    assert ranked[0][0] == target.id
    assert ranked[0][1] >= ranked[-1][1]


def test_keyword_search_finds_by_token(store, emb):
    m = _add(store, emb, "the rollback procedure uses undeploy")
    _add(store, emb, "unrelated content here")
    ids = store.keyword_search("rollback undeploy", 5)
    assert m.id in ids


def test_archive_excludes_from_index_and_listing(store, emb):
    m = _add(store, emb, "ephemeral note")
    assert store.count() == 1
    store.archive(m.id)
    assert store.count() == 1 - 1  # excluded from default count
    assert store.count(include_archived=True) == 1
    q = emb.embed("ephemeral note")
    assert m.id not in [i for i, _ in store.knn(q, 5)]
    assert store.get(m.id).archived is True  # still retrievable (tombstone)


def test_delete_removes_everything(store, emb):
    a = _add(store, emb, "a")
    b = _add(store, emb, "b")
    store.add_edge(a.id, b.id)
    store.delete(a.id)
    assert store.get(a.id) is None
    assert store.neighbors(a.id) == []


def test_get_embedding_roundtrip(store, emb):
    m = _add(store, emb, "vector me")
    v = store.get_embedding(m.id)
    assert v is not None
    assert np.allclose(v, emb.embed(m.embedding_text()), atol=1e-6)


def test_persistence_reopen(tmp_path, emb):
    path = tmp_path / "mem.sqlite"
    s1 = MemoryStore(path)
    m = _add(s1, emb, "persisted across sessions")
    s1.close()

    s2 = MemoryStore(path)
    assert s2.count() == 1
    got = s2.get(m.id)
    assert got is not None
    assert got.content == "persisted across sessions"
    # index rebuilt from disk -> knn works
    ranked = s2.knn(emb.embed("persisted across sessions"), 1)
    assert ranked and ranked[0][0] == m.id
    s2.close()


def test_numpy_index_add_remove():
    idx = NumpyVectorIndex()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([0.0, 1.0], dtype=np.float32))
    assert len(idx) == 2
    res = idx.query(np.array([1.0, 0.0], dtype=np.float32), 2)
    assert res[0][0] == "a"
    idx.remove("a")
    assert len(idx) == 1
    assert idx.query(np.array([1.0, 0.0], dtype=np.float32), 2)[0][0] == "b"


class _TrackedRLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()

    def __enter__(self):
        self._lock.acquire()
        self._local.depth = getattr(self._local, "depth", 0) + 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._local.depth -= 1
        self._lock.release()

    def owned(self) -> bool:
        return getattr(self._local, "depth", 0) > 0


class _GuardedConnection:
    def __init__(self, conn, lock: _TrackedRLock) -> None:
        self._conn = conn
        self._lock = lock

    def _assert_locked(self) -> None:
        assert self._lock.owned(), "MemoryStore touched sqlite connection without its lock"

    def execute(self, *args, **kwargs):
        self._assert_locked()
        return self._conn.execute(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        self._assert_locked()
        return self._conn.executescript(*args, **kwargs)

    def commit(self):
        self._assert_locked()
        return self._conn.commit()

    def close(self):
        self._assert_locked()
        return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _GuardedIndex:
    def __init__(self, index, lock: _TrackedRLock) -> None:
        self._index = index
        self._lock = lock

    def _assert_locked(self) -> None:
        assert self._lock.owned(), "MemoryStore touched vector index without its lock"

    def add(self, id, vector):
        self._assert_locked()
        return self._index.add(id, vector)

    def remove(self, id):
        self._assert_locked()
        return self._index.remove(id)

    def query(self, vector, k):
        self._assert_locked()
        return self._index.query(vector, k)

    def __len__(self):
        self._assert_locked()
        return len(self._index)


def test_store_serializes_connection_and_index_access(store, emb):
    lock = _TrackedRLock()
    store._lock = lock
    store._conn = _GuardedConnection(store._conn, lock)
    store._index = _GuardedIndex(store._index, lock)

    m = _add(store, emb, "alpha beta", type=MemoryType.INFO)
    got = store.get(m.id)
    assert got is not None
    got.add_edge("other", EdgeType.RELATED, 0.4)
    store.save(got)
    store.add_edge(got.id, "other", EdgeType.SUPPORTS)

    assert store.resolve_id(m.id[:8]) == m.id
    assert store.owner_of(m.id) == ""
    assert store.get_embedding(m.id) is not None
    assert store.neighbors(m.id)
    assert store.all_memories()
    assert store.count() == 1
    assert store.knn(emb.embed("alpha beta"), 1)
    assert store.keyword_search("alpha", 1) == [m.id]

    store.rename_owner("missing-owner", "new-owner")
    store.log_maintenance("test", {"ok": True})
    assert store.maintenance_history()[0]["report"] == {"ok": True}
    store.archive(m.id)
    store.delete(m.id)


def test_store_survives_foreground_and_reflection_thread_overlap(tmp_path, emb):
    path = tmp_path / "memory.sqlite"
    store = MemoryStore(path)
    try:
        for i in range(12):
            _add(store, emb, f"seed memory {i}", type=MemoryType.INFO)

        engine = ReflectionEngine(
            store,
            emb,
            ReflectionPolicy(merge_threshold=0.999, edge_threshold=0.999),
            ForgetPolicy(),
        )

        def reflect_worker() -> None:
            for _ in range(20):
                engine.consolidate()

        def foreground_worker(worker_id: int) -> None:
            for i in range(40):
                m = _add(store, emb, f"foreground {worker_id} memory {i}")
                store.count()
                store.keyword_search("foreground memory", 5)
                store.knn(emb.embed(f"foreground {worker_id}"), 5)
                got = store.get(m.id)
                if got is not None:
                    got.touch()
                    store.save(got)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(reflect_worker)]
            futures.extend(pool.submit(foreground_worker, n) for n in range(3))
            for fut in futures:
                fut.result(timeout=30)

        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            rows = conn.execute("SELECT data FROM memories").fetchall()
        assert rows
        for (payload,) in rows:
            Memory.model_validate(json.loads(payload))
    finally:
        store.close()
