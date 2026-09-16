# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SQLite-centric memory store + numpy vector index."""

import multiprocessing
import queue
import threading

import numpy as np
import pytest
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.schema import EdgeType, Memory, MemoryType
from nooa_memory.store import MemoryStore, NumpyVectorIndex


def _process_writer(path: str, start, result, worker: int, count: int) -> None:
    """Write distinct rows from a spawned process and report any exception."""
    store = None
    try:
        start.wait(timeout=10)
        store = MemoryStore(path)
        for i in range(count):
            store.add(Memory(id=f"worker-{worker}-{i}", content=f"memory {worker} {i}"))
        result.put(None)
    except BaseException as exc:  # pragma: no cover - only returned to parent
        result.put(repr(exc))
    finally:
        if store is not None:
            store.close()


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


def test_file_store_uses_durable_wal(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite")
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    store.close()


def test_malformed_json_payload_is_reconstructed_from_columns(tmp_path, caplog):
    path = tmp_path / "recovered.sqlite"
    store = MemoryStore(path)
    memory = Memory(
        type=MemoryType.EPISODE,
        content="survived recovery",
        importance=7.0,
        salience=0.8,
        owner="agent@test",
    )
    store.add(memory)
    store._conn.execute("UPDATE memories SET data = '' WHERE id = ?", (memory.id,))
    store._conn.commit()

    recovered = store.get(memory.id)

    assert recovered is not None
    assert recovered.id == memory.id
    assert recovered.type == MemoryType.EPISODE
    assert recovered.content == "survived recovery"
    assert recovered.importance == 7.0
    assert recovered.salience == 0.8
    assert recovered.owner == "agent@test"
    assert "has a malformed JSON payload; reconstructing it from columns" in caplog.text
    store.close()


def test_malformed_stored_embeddings_do_not_disable_memory(tmp_path, emb, caplog):
    path = tmp_path / "recovered.sqlite"
    store = MemoryStore(path, embedding_dim=128)
    valid = _add(store, emb, "valid vector")
    empty = _add(store, emb, "recovered empty vector")
    wrong_dim = _add(store, emb, "stale vector dimension")
    store._conn.execute("UPDATE memories SET embedding = X'' WHERE id = ?", (empty.id,))
    store._conn.execute(
        "UPDATE memories SET embedding = ? WHERE id = ?",
        (np.ones(64, dtype=np.float32).tobytes(), wrong_dim.id),
    )
    store._conn.commit()
    store.close()

    reopened = MemoryStore(path, embedding_dim=128)
    assert len(reopened._index) == 1
    assert reopened.get_embedding(empty.id) is None
    assert reopened.get_embedding(wrong_dim.id) is None
    assert reopened.knn(emb.embed("valid vector"), 1)[0][0] == valid.id
    assert "has an empty embedding; ignoring it" in caplog.text
    assert "has embedding dimension 64, expected 128; ignoring it" in caplog.text
    reopened.close()


def test_rejects_new_embeddings_with_invalid_dimensions(store):
    with pytest.raises(ValueError, match="must not be empty"):
        store.add(Memory(content="empty"), np.array([], dtype=np.float32))

    store.add(Memory(content="sets dimension"), np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="dimension 3 does not match store dimension 4"):
        store.add(Memory(content="wrong dimension"), np.ones(3, dtype=np.float32))


def test_same_store_operations_are_atomic_across_threads(store):
    entered = threading.Event()
    release = threading.Event()
    original_add = store._index.add

    def blocking_add(memory_id, vector):
        entered.set()
        assert release.wait(timeout=5)
        original_add(memory_id, vector)

    store._index.add = blocking_add
    vector = np.array([1.0, 0.0], dtype=np.float32)
    writer = threading.Thread(target=store.add, args=(Memory(content="threaded"), vector))
    writer.start()
    assert entered.wait(timeout=5)

    read_finished = threading.Event()
    reader = threading.Thread(target=lambda: (store.count(), read_finished.set()))
    reader.start()
    assert not read_finished.wait(timeout=0.1)

    release.set()
    writer.join(timeout=5)
    reader.join(timeout=5)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert read_finished.is_set()
    assert store.count() == len(store._index) == 1


def test_concurrent_process_writers_share_store_safely(tmp_path):
    path = tmp_path / "shared.sqlite"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    result = ctx.Queue()
    worker_count, rows_per_worker = 4, 25
    workers = [
        ctx.Process(target=_process_writer, args=(str(path), start, result, i, rows_per_worker))
        for i in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    start.set()
    errors = []
    for _ in workers:
        try:
            error = result.get(timeout=20)
        except queue.Empty:
            error = "writer timed out"
        if error is not None:
            errors.append(error)
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            errors.append("writer process did not exit")
        elif worker.exitcode != 0:
            errors.append(f"writer exited with {worker.exitcode}")

    assert errors == []
    store = MemoryStore(path)
    assert store.count() == worker_count * rows_per_worker
    assert store._conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    store.close()


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
