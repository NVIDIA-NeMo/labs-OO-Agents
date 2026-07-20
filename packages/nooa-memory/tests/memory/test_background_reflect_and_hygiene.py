# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical memory fixes: background manual reflect, loud consolidation, compact reprs.

Defects (20260716 ARC fleet):
- ``MemoryManager.reflect()`` (manual trigger) runs its consolidation LLM calls
  synchronously on the caller's critical path even when
  ``ReflectionPolicy.background=True`` — 600-1,500s/game of blocking latency.
- A reasoner failure (e.g. empty LLM reply -> ``_extract_json`` ValueError) is
  swallowed by ``except Exception: return 0`` with no log line: on game cd82,
  17/17 consolidation calls failed silently and memory maintenance was dead all run.
- ``Memory.__repr__`` dumps the full record (~22KB): one print in a cell blew
  prompts to 180k tokens.
"""

from __future__ import annotations

import logging
import time

import pytest
from nooa_memory.config import ForgetPolicy, MemoryConfig, ReflectionPolicy
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.manager import MemoryManager
from nooa_memory.reflection import ReflectionEngine
from nooa_memory.schema import Memory, MemoryType
from nooa_memory.store import MemoryStore

from nooa.agent import Agent
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def emb():
    return HashingEmbedder(dim=256)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


class _HostAgent(Agent, llm=FakeLLMClient()):
    pass


def _manager(reasoner=None, *, background: bool) -> MemoryManager:
    config = MemoryConfig(
        enabled=True,
        path=":memory:",
        reflection=ReflectionPolicy(trigger="manual", background=background),
    )
    return MemoryManager(
        _HostAgent(),
        config,
        embedder=HashingEmbedder(dim=256),
        reasoner=reasoner,
    )


# --------------------------------------------------------------------------- #
# manual reflect() must honor background=True
# --------------------------------------------------------------------------- #

_SLOW_S = 0.6


def _slow_reasoner(episodes):
    time.sleep(_SLOW_S)
    return []


def test_manual_reflect_with_background_true_returns_immediately():
    mgr = _manager(_slow_reasoner, background=True)
    mgr.remember("tried the blue switch, door opened", type=MemoryType.EPISODE)

    t0 = time.monotonic()
    mgr.reflect()
    elapsed = time.monotonic() - t0
    assert elapsed < _SLOW_S / 2, (
        f"reflect() blocked for {elapsed:.2f}s — consolidation ran on the "
        "caller's critical path despite reflection.background=True"
    )

    # ... and the consolidation still actually happens (off the critical path).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if mgr.stats.reflections >= 1:
            break
        time.sleep(0.05)
    assert mgr.stats.reflections >= 1, "background consolidation never completed"


def test_manual_reflect_with_background_false_stays_synchronous():
    mgr = _manager(_slow_reasoner, background=False)
    mgr.remember("tried the red switch, nothing happened", type=MemoryType.EPISODE)

    t0 = time.monotonic()
    report = mgr.reflect()
    elapsed = time.monotonic() - t0
    assert elapsed >= _SLOW_S * 0.8  # deterministic inline behavior preserved
    assert mgr.stats.reflections == 1
    assert report is not None


# --------------------------------------------------------------------------- #
# consolidation failures must be loud (and retried once)
# --------------------------------------------------------------------------- #


def _add_episode(store, emb):
    m = Memory(content="explored level three, found the key", type=MemoryType.EPISODE)
    store.add(m, emb.embed(m.embedding_text()))
    return m


def test_reasoner_failure_is_logged_and_retried(store, emb, caplog):
    _add_episode(store, emb)
    calls: list[int] = []

    def flaky_reasoner(episodes):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("no JSON object in LLM reply: ''")
        return [Memory(content="insight: keys open doors", type=MemoryType.REFLECTION)]

    engine = ReflectionEngine(store, emb, ReflectionPolicy(), ForgetPolicy())
    with caplog.at_level(logging.WARNING):
        report = engine.consolidate(reasoner=flaky_reasoner)

    assert len(calls) == 2, "a failed reasoner call must be retried once"
    assert report.created == 1, "the retry's result must be committed"
    assert any(
        "reasoner" in rec.message.lower() or "consolidat" in rec.message.lower()
        for rec in caplog.records
    ), "a reasoner failure must produce a WARNING (cd82: 17/17 silent failures)"


def test_reasoner_failing_twice_is_loud_but_safe(store, emb, caplog):
    _add_episode(store, emb)
    calls: list[int] = []

    def dead_reasoner(episodes):
        calls.append(1)
        raise ValueError("no JSON object in LLM reply: ''")

    engine = ReflectionEngine(store, emb, ReflectionPolicy(), ForgetPolicy())
    with caplog.at_level(logging.WARNING):
        report = engine.consolidate(reasoner=dead_reasoner)

    assert len(calls) == 2  # one retry, then give up
    assert report.created == 0
    assert any(
        "reasoner" in r.message.lower() or "consolidat" in r.message.lower() for r in caplog.records
    )


def test_legitimately_empty_reasoner_result_stays_silent(store, emb, caplog):
    """Zero abstractions is a valid outcome — no warning, no retry."""
    _add_episode(store, emb)
    calls: list[int] = []

    def empty_reasoner(episodes):
        calls.append(1)
        return []

    engine = ReflectionEngine(store, emb, ReflectionPolicy(), ForgetPolicy())
    with caplog.at_level(logging.WARNING):
        engine.consolidate(reasoner=empty_reasoner)

    assert len(calls) == 1
    assert not [r for r in caplog.records if "reasoner" in r.message.lower()]


# --------------------------------------------------------------------------- #
# Memory repr must be compact
# --------------------------------------------------------------------------- #


def test_memory_repr_is_compact():
    m = Memory(content="x" * 5000, type=MemoryType.INFO)
    text = repr(m)
    assert len(text) < 500, (
        f"repr(Memory) is {len(text)} chars — full-record reprs blew ARC prompts "
        "to 180k tokens when printed in a cell"
    )
    assert m.id[:8] in text  # identity survives
    assert "info" in text.lower()  # type survives
    assert "xxx" in text  # a content head survives


def test_memory_str_is_compact_too():
    m = Memory(content="y" * 5000, type=MemoryType.SKILL)
    assert len(str(m)) < 500
