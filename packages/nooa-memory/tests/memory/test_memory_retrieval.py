# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval: scoring, recency/importance, multi-hop spread."""

import pytest
from nooa_memory.config import RetrievalConfig
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.retrieval import RetrievalEngine, base_level_activation
from nooa_memory.schema import AccessRecord, EdgeType, Memory
from nooa_memory.store import MemoryStore


@pytest.fixture
def emb():
    return HashingEmbedder(dim=512)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _add(store, emb, content, **kw):
    m = Memory(content=content, **kw)
    return store.add(m, emb.embed(m.embedding_text()))


def test_recall_ranks_relevant_first(store, emb):
    target = _add(store, emb, "to deploy the service run make ship in CI")
    _add(store, emb, "the weather today is sunny and warm")
    _add(store, emb, "my favourite colour is blue")
    eng = RetrievalEngine(store, emb, RetrievalConfig())
    res = eng.recall("how do I deploy the service?", k=3)
    assert res[0].id == target.id


def test_importance_breaks_relevance_ties(store, emb):
    low = _add(store, emb, "identical content about shipping", importance=2.0)
    high = _add(store, emb, "identical content about shipping", importance=9.0)
    eng = RetrievalEngine(store, emb, RetrievalConfig())
    res = eng.recall("identical content about shipping", k=2)
    assert res[0].id == high.id
    assert {m.id for m in res} == {low.id, high.id}


def test_recall_touches_returned_memories(store, emb):
    m = _add(store, emb, "remember to touch me on recall")
    eng = RetrievalEngine(store, emb, RetrievalConfig())
    eng.recall("touch me", k=1, touch=True)
    assert store.get(m.id).access_count == 1


def test_recall_no_touch_leaves_access_count(store, emb):
    m = _add(store, emb, "do not touch me")
    eng = RetrievalEngine(store, emb, RetrievalConfig())
    eng.recall("do not touch", k=1, touch=False)
    assert store.get(m.id).access_count == 0


def test_recall_empty_store_returns_empty(store, emb):
    eng = RetrievalEngine(store, emb, RetrievalConfig())
    assert eng.recall("anything", k=5) == []


def test_base_level_activation_recent_beats_old():
    now = 1_000_000.0
    recent = base_level_activation([AccessRecord(ts=now - 10, channel="recalled")], now, 0.5)
    old = base_level_activation([AccessRecord(ts=now - 1_000_000, channel="recalled")], now, 0.5)
    assert recent > old


def test_spread_decays_per_hop(store, emb):
    a = _add(store, emb, "seed node alpha")
    b = _add(store, emb, "node beta")
    c = _add(store, emb, "node gamma")
    store.add_edge(a.id, b.id, EdgeType.CAUSES, 1.0)
    store.add_edge(b.id, c.id, EdgeType.RELATED, 1.0)
    eng = RetrievalEngine(store, emb, RetrievalConfig(per_hop_decay=0.6, per_hop_fanout=5))
    spread = eng._spread({a.id: 1.0}, hops=2)
    assert spread[b.id] > spread[c.id] > 0  # decays with distance
    # 1 hop: only direct neighbour reachable
    spread1 = eng._spread({a.id: 1.0}, hops=1)
    assert b.id in spread1 and c.id not in spread1


def test_spread_decay_is_one_factor_per_hop(store, emb):
    # The docstring and ``per_hop_decay``'s own comment both say the decay is
    # per hop, so at hop h the activation is delta**h along unit-weight causal
    # edges -- not a compounded power that dies under ``activation_floor``.
    a = _add(store, emb, "seed node alpha")
    b = _add(store, emb, "node beta")
    c = _add(store, emb, "node gamma")
    d = _add(store, emb, "node delta")
    for src, dst in ((a, b), (b, c), (c, d)):
        store.add_edge(src.id, dst.id, EdgeType.CAUSES, 1.0)
    eng = RetrievalEngine(
        store,
        emb,
        RetrievalConfig(per_hop_decay=0.6, per_hop_fanout=5, activation_floor=0.05),
    )
    spread = eng._spread({a.id: 1.0}, hops=3)
    assert spread[b.id] == pytest.approx(0.6)
    assert spread[c.id] == pytest.approx(0.6**2)
    assert spread[d.id] == pytest.approx(0.6**3)


def test_multi_hop_surfaces_linked_dissimilar_memory(store, emb):
    # A is relevant to the query; B is dissimilar but linked from A. Several
    # unlinked, equally-dissimilar distractors compete for the second slot.
    a = _add(store, emb, "deploy ship release production rollout pipeline")
    b = _add(store, emb, "xyzzy plugh frobnicate quux")
    store.add_edge(a.id, b.id, EdgeType.CAUSES, 1.0)
    for i in range(5):
        _add(store, emb, f"unrelated distractor number {i} wibble wobble")
    eng = RetrievalEngine(store, emb, RetrievalConfig(top_k=2))
    res = eng.recall("deploy ship release", k=2, hops=1)
    ids = {m.id for m in res}
    # A wins on relevance; B is pulled into the top-2 by associative spread
    # over the causal edge, beating the unlinked distractors.
    assert a.id in ids and b.id in ids
