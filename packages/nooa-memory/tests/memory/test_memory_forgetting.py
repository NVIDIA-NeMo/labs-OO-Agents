# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for forgetting: online decay + offline pruning."""

import time

import pytest
from nooa_memory.config import ForgetPolicy
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.forgetting import ForgettingEngine, retention
from nooa_memory.schema import Memory, MemoryType
from nooa_memory.store import MemoryStore

HOUR = 3600.0
DAY = 24 * HOUR


@pytest.fixture
def emb():
    return HashingEmbedder(dim=64)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_retention_decays_with_time():
    now = time.time()
    cfg = ForgetPolicy()
    fresh = Memory(content="x", last_accessed_at=now - HOUR)
    stale = Memory(content="x", last_accessed_at=now - 100 * DAY)
    assert retention(fresh, now, cfg) > retention(stale, now, cfg)


def test_strength_slows_decay():
    now = time.time()
    cfg = ForgetPolicy()
    weak = Memory(content="x", last_accessed_at=now - 30 * DAY, strength=1)
    strong = Memory(content="x", last_accessed_at=now - 30 * DAY, strength=50)
    assert retention(strong, now, cfg) > retention(weak, now, cfg)


def test_should_prune_old_low_value_memory(store, emb):
    now = time.time()
    cfg = ForgetPolicy()
    eng = ForgettingEngine(store, cfg)
    old = Memory(
        content="trivial old note",
        type=MemoryType.INFO,
        importance=2.0,
        created_at=now - 100 * DAY,
        last_accessed_at=now - 100 * DAY,
    )
    assert eng.should_prune(old, now) is True


def test_protected_type_is_never_pruned(store, emb):
    now = time.time()
    eng = ForgettingEngine(store, ForgetPolicy(protected_types=("skill",)))
    old_skill = Memory(
        content="how to deploy",
        type=MemoryType.SKILL,
        importance=2.0,
        created_at=now - 100 * DAY,
        last_accessed_at=now - 100 * DAY,
    )
    assert eng.should_prune(old_skill, now) is False


def test_high_importance_is_never_pruned(store, emb):
    now = time.time()
    eng = ForgettingEngine(store, ForgetPolicy())
    old_important = Memory(
        content="critical fact",
        type=MemoryType.INFO,
        importance=9.0,
        created_at=now - 100 * DAY,
        last_accessed_at=now - 100 * DAY,
    )
    assert eng.should_prune(old_important, now) is False


def test_young_memory_is_not_pruned(store, emb):
    now = time.time()
    eng = ForgettingEngine(store, ForgetPolicy(prune_min_age_hours=24.0))
    young = Memory(
        content="brand new", importance=1.0, created_at=now - HOUR, last_accessed_at=now - HOUR
    )
    assert eng.should_prune(young, now) is False


def test_prune_archives_by_default(store, emb):
    now = time.time()
    m = Memory(
        content="forget me",
        type=MemoryType.INFO,
        importance=1.0,
        created_at=now - 100 * DAY,
        last_accessed_at=now - 100 * DAY,
    )
    store.add(m, emb.embed(m.embedding_text()))
    eng = ForgettingEngine(store, ForgetPolicy(archive_vs_delete="archive"))
    pruned = eng.prune(now=now)
    assert m.id in pruned
    assert store.count() == 0  # excluded from active set
    assert store.get(m.id).archived is True  # tombstoned, recoverable


def test_prune_deletes_when_configured(store, emb):
    now = time.time()
    m = Memory(
        content="delete me",
        importance=1.0,
        created_at=now - 100 * DAY,
        last_accessed_at=now - 100 * DAY,
    )
    store.add(m, emb.embed(m.embedding_text()))
    eng = ForgettingEngine(store, ForgetPolicy(archive_vs_delete="delete"))
    eng.prune(now=now)
    assert store.get(m.id) is None


def test_prune_disabled_is_noop(store, emb):
    now = time.time()
    m = Memory(
        content="keep me",
        importance=1.0,
        created_at=now - 100 * DAY,
        last_accessed_at=now - 100 * DAY,
    )
    store.add(m, emb.embed(m.embedding_text()))
    eng = ForgettingEngine(store, ForgetPolicy(enabled=False))
    assert eng.prune(now=now) == []
    assert store.count() == 1
