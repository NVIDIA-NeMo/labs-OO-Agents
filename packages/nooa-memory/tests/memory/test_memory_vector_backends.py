# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for swappable vector backends (numpy / sqlite-vec / chroma / qdrant).

The numpy backend always runs; the others ``importorskip`` their optional
dependency so the suite stays green whether or not they are installed.
"""

import os
import uuid

import numpy as np
import pytest
from nooa_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    NumpyVectorIndex,
    VectorConfig,
    make_vector_index,
)
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.schema import Memory, MemoryType
from nooa_memory.store import MemoryStore

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

DIM = 64
ALL_BACKENDS = [
    "numpy",
    "sqlite_vec",
    "chroma_embedded",
    "chroma_http",
    "qdrant_server",
]
# Run the qdrant_server cases against a server, e.g. QDRANT_URL=http://localhost:6333.
QDRANT_URL = os.environ.get("QDRANT_URL")
_qdrant_collections: list[str] = []


def _require(backend: str) -> None:
    if backend == "sqlite_vec":
        pytest.importorskip("sqlite_vec")
    if backend in ("chroma_embedded", "chroma_http"):
        pytest.importorskip("chromadb")
    if backend == "chroma_http":
        pytest.skip("chroma_http needs a running server")
    if backend == "qdrant_server":
        pytest.importorskip("qdrant_client")
        if not QDRANT_URL:
            pytest.skip("qdrant_server needs a running server (set QDRANT_URL)")


def _vector_config(backend: str) -> VectorConfig:
    _require(backend)
    if backend == "qdrant_server":
        name = f"nooa-test-{uuid.uuid4().hex[:8]}"
        _qdrant_collections.append(name)
        return VectorConfig(backend=backend, url=QDRANT_URL, collection=name)
    return VectorConfig(backend=backend)


@pytest.fixture(autouse=True, scope="module")
def _drop_qdrant_collections():
    yield
    if _qdrant_collections:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)
        for name in _qdrant_collections:
            client.delete_collection(name)
        client.close()


def _store(backend: str, path: str = ":memory:", vector: VectorConfig | None = None) -> MemoryStore:
    _require(backend)
    return MemoryStore(path, vector_config=vector or _vector_config(backend), embedding_dim=DIM)


@pytest.fixture
def emb():
    return HashingEmbedder(dim=DIM)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_factory_returns_working_index(backend):
    store = _store(backend)
    emb = HashingEmbedder(dim=DIM)
    m = Memory(content="deploy ship release", type=MemoryType.INFO)
    store.add(m, emb.embed(m.embedding_text()))
    ranked = store.knn(emb.embed("deploy ship"), 1)
    assert ranked and ranked[0][0] == m.id
    assert 0.0 < ranked[0][1] <= 1.0  # cosine score
    store.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_ranks_relevant_first(backend, emb):
    store = _store(backend)
    target = Memory(content="deploy ship release production rollout", type=MemoryType.INFO)
    others = [
        Memory(content="the cat sat on the mat", type=MemoryType.INFO),
        Memory(content="kubernetes pod crashloop backoff", type=MemoryType.INFO),
    ]
    for m in [target, *others]:
        store.add(m, emb.embed(m.embedding_text()))
    ranked = store.knn(emb.embed("how do I deploy and ship a release"), 3)
    assert ranked[0][0] == target.id
    store.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_archive_removes_from_index(backend, emb):
    store = _store(backend)
    m = Memory(content="ephemeral", type=MemoryType.INFO)
    store.add(m, emb.embed(m.embedding_text()))
    assert len(store._index) == 1
    store.archive(m.id)
    assert m.id not in [i for i, _ in store.knn(emb.embed("ephemeral"), 5)]
    store.close()


def test_all_available_backends_agree_on_ranking(emb):
    """The whole point of the protocol: identical data -> identical top-k order."""
    texts = [
        "deploy ship release production",
        "the cat sat on the mat",
        "kubernetes pod crashloop backoff",
        "rollback with make undeploy",
    ]
    query = "how to deploy and ship a release to production"

    rankings = {}
    for backend in ALL_BACKENDS:
        try:
            _require(backend)
        except pytest.skip.Exception:
            continue
        store = _store(backend)
        ids = []
        for t in texts:
            m = Memory(content=t, type=MemoryType.INFO)
            store.add(m, emb.embed(m.embedding_text()))
            ids.append(m.id)
        order = [mid for mid, _ in store.knn(emb.embed(query), len(texts))]
        # normalise to positions within this store's own ids
        rankings[backend] = [ids.index(mid) for mid in order]
        store.close()

    assert "numpy" in rankings
    baseline = rankings["numpy"]
    for backend, order in rankings.items():
        assert order == baseline, f"{backend} ranking {order} != numpy {baseline}"


def test_factory_unknown_backend_raises():
    # VectorConfig.backend is a Literal, so build a stand-in to reach the factory's guard.
    import types

    with pytest.raises(ValueError, match="Unknown vector backend"):
        make_vector_index(types.SimpleNamespace(backend="bogus"))  # type: ignore[arg-type]


def test_factory_numpy_is_default():
    assert isinstance(make_vector_index(VectorConfig()), NumpyVectorIndex)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_manager_wires_backend_end_to_end(backend):
    """Flipping MemoryConfig.vector.backend changes the store's index, no code change."""
    _require(backend)

    class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
        pass

    agent = MemAgent()
    mgr = MemoryManager.install(
        agent,
        config=MemoryConfig(enabled=True, path=":memory:", vector=_vector_config(backend)),
    )
    agent.remember("the deploy command is make ship", type="info")
    res = agent.recall("how do I deploy", k=1)
    assert res and "make ship" in res[0].content
    mgr.uninstall()


def test_numpy_dedup_on_write_still_works_per_backend():
    """sanity: dedup-on-write (which uses knn) holds for the numpy default."""
    store = _store("numpy")
    emb = HashingEmbedder(dim=DIM)
    a = Memory(content="identical fact", type=MemoryType.INFO)
    store.add(a, emb.embed(a.embedding_text()))
    hits = store.knn(emb.embed("identical fact"), 1)
    assert hits[0][1] > 0.99  # near-identical cosine -> dedup would trigger
    store.close()


def test_npindex_unit():
    idx = NumpyVectorIndex()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([0.0, 1.0], dtype=np.float32))
    assert idx.query(np.array([1.0, 0.0], dtype=np.float32), 1)[0][0] == "a"


def test_qdrant_persists_across_reopen(tmp_path, emb):
    path = str(tmp_path / "mem.db")
    vector = _vector_config("qdrant_server")
    store = _store("qdrant_server", path, vector)
    m = Memory(content="deploy ship release", type=MemoryType.INFO)
    store.add(m, emb.embed(m.embedding_text()))
    store.close()

    store = _store("qdrant_server", path, vector)
    assert store.knn(emb.embed("deploy ship"), 1)[0][0] == m.id
    store.close()


def test_qdrant_index_unit():
    vector = _vector_config("qdrant_server")
    from nooa_memory import QdrantVectorIndex

    idx = QdrantVectorIndex(vector, dim=2)
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([0.0, 1.0], dtype=np.float32))
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    assert len(idx) == 2
    assert [mid for mid, _ in idx.query(np.array([1.0, 0.0], dtype=np.float32), 2)] == ["a", "b"]
    idx.remove("a")
    assert len(idx) == 1
    idx.close()


def test_qdrant_rejects_collection_with_other_dim():
    vector = _vector_config("qdrant_server")
    _store("qdrant_server", vector=vector).close()
    with pytest.raises(ValueError, match="128-dim cosine"):
        MemoryStore(":memory:", vector_config=vector, embedding_dim=DIM * 2)
