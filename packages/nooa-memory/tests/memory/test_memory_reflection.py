# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reflection / offline consolidation."""

import time

import pytest
from nooa_memory.config import ForgetPolicy, ReflectionPolicy
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.reflection import ReflectionEngine
from nooa_memory.schema import EdgeType, Memory, MemoryType
from nooa_memory.store import MemoryStore

DAY = 24 * 3600.0


@pytest.fixture
def emb():
    return HashingEmbedder(dim=256)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _add(store, emb, content, **kw):
    m = Memory(content=content, **kw)
    return store.add(m, emb.embed(m.embedding_text()))


def _engine(store, emb, **reflect_kw):
    return ReflectionEngine(store, emb, ReflectionPolicy(**reflect_kw), ForgetPolicy())


def test_merge_folds_duplicates(store, emb):
    a = _add(store, emb, "deploy with make ship", type=MemoryType.INFO)
    b = _add(store, emb, "deploy with make ship", type=MemoryType.INFO)  # identical -> cos 1.0
    eng = _engine(store, emb, merge_threshold=0.95)
    report = eng.consolidate()
    assert report.merged >= 1
    # exactly one survives in the active set
    active = store.all_memories()
    assert len(active) == 1
    survivor = active[0]
    assert survivor.reinforcement_count >= 1
    # the merged-away one is archived with a provenance edge from the survivor
    assert store.get(b.id).archived or store.get(a.id).archived


def test_distinct_memories_are_not_merged(store, emb):
    _add(store, emb, "completely different topic alpha beta")
    _add(store, emb, "unrelated subject gamma delta epsilon")
    eng = _engine(store, emb, merge_threshold=0.95)
    report = eng.consolidate()
    assert report.merged == 0
    assert len(store.all_memories()) == 2


def test_edge_formation_links_related_memories(store, emb):
    # Share some tokens -> cosine in the [edge, merge) band.
    _add(store, emb, "deploy ship release alpha")
    _add(store, emb, "deploy ship rollback beta")
    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.2, max_edges_per_node=3)
    report = eng.consolidate()
    assert report.edges_added >= 1
    assert any(m.edges for m in store.all_memories())


def test_rescore_importance_clamped_and_salience_boosts(store, emb):
    m = _add(store, emb, "a salient lesson", importance=3.0, salience=1.0)
    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.999)
    eng.consolidate()
    out = store.get(m.id)
    assert 0.0 <= out.importance <= 10.0
    assert out.importance > 3.0  # salience pushed importance up


def test_reflection_prunes_decayed_memory(store, emb):
    now = time.time()
    keep = _add(store, emb, "fresh relevant note", importance=5.0)
    _add(
        store,
        emb,
        "ancient trivia nobody needs",
        importance=1.0,
        created_at=now - 200 * DAY,
        last_accessed_at=now - 200 * DAY,
    )
    eng = ReflectionEngine(
        store, emb, ReflectionPolicy(merge_threshold=0.999, edge_threshold=0.999), ForgetPolicy()
    )
    report = eng.consolidate()
    assert report.pruned >= 1
    assert keep.id in {m.id for m in store.all_memories()}


def test_abstraction_with_reasoner_creates_skill_linked_to_episodes(store, emb):
    e1 = _add(store, emb, "episode: tried X, it worked", type=MemoryType.EPISODE)
    e2 = _add(store, emb, "episode: tried X again, worked", type=MemoryType.EPISODE)

    def reasoner(episodes):
        assert len(episodes) >= 2
        return [Memory(content="skill: X reliably works", type=MemoryType.SKILL)]

    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.999)
    report = eng.consolidate(reasoner=reasoner)
    assert report.created == 1
    skills = [m for m in store.all_memories() if m.type == MemoryType.SKILL]
    assert len(skills) == 1
    targets = {e.target_id for e in skills[0].edges if e.type == EdgeType.DERIVED_FROM}
    assert {e1.id, e2.id} <= targets


def test_reconsolidate_archives_stale_and_keeps_current(store, emb):
    older = _add(store, emb, "my 5k personal best is 27:00", type=MemoryType.INFO)
    _add(store, emb, "my 5k personal best is now 25:50", type=MemoryType.INFO)

    def reconciler(cluster):
        # cluster is oldest -> newest; supersede everything but the latest
        outdated = [c.id for c in cluster[:-1]]
        keep = Memory(
            content="current 5k personal best: 25:50", type=MemoryType.INFO, importance=8.0
        )
        return keep, outdated

    eng = ReflectionEngine(
        store,
        emb,
        ReflectionPolicy(merge_threshold=0.999, edge_threshold=0.999, recon_threshold=0.3),
        ForgetPolicy(),
    )
    report = eng.consolidate(reconciler=reconciler)
    assert report.reconciled >= 1 and report.superseded >= 1
    active = [m.content for m in store.all_memories()]
    assert any("25:50" in c for c in active)  # current value retained
    assert not any("27:00" in c for c in active)  # stale value archived
    assert store.get(older.id).archived is True


def test_reconciler_noop_when_nothing_archived(store, emb):
    _add(store, emb, "fact one alpha", type=MemoryType.INFO)
    _add(store, emb, "fact two beta", type=MemoryType.INFO)
    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.999)
    report = eng.consolidate(reconciler=lambda cluster: (None, []))
    assert report.reconciled == 0 and report.superseded == 0
    assert len(store.all_memories()) == 2


def test_consolidate_empty_store(store, emb):
    eng = _engine(store, emb)
    report = eng.consolidate()
    assert report.merged == 0 and report.pruned == 0
