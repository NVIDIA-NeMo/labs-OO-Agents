# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SQLite-centric memory store + numpy vector index."""

import numpy as np
import pytest
from nooa_memory.embeddings import HashingEmbedder
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
