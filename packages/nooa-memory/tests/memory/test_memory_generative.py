# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-backed reflection steps (memory/generative.py): the reconciler folds
paraphrase-duplicate clusters, the reasoner abstracts episodes into
reflections. The headline case is the real dogfooding scenario: three
paraphrase 'Reflection harness duplicate' memories (pairwise cosine ~0.7 —
below every deterministic threshold, above the recon-cluster bar) that only
an LLM step can consolidate."""

import json
import re

from nooa_memory import (
    EdgeType,
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    MemoryType,
    llm_reasoner,
    llm_reconciler,
)
from nooa_memory.config import ReflectionPolicy

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


class ScriptedLLM:
    """A stand-in LLM: sync .call() answering via a function of the prompt."""

    def __init__(self, answer):
        self.answer = answer  # Callable[[str], str]
        self.calls: list[str] = []

    def call(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)

        class _Resp:
            content = self.answer(prompt)

        return _Resp()


def _install(agent, *, reasoner=None, reconciler=None, **cfg_kw):
    cfg = MemoryConfig(enabled=True, path=":memory:", **cfg_kw)
    return MemoryManager.install(agent, config=cfg, reasoner=reasoner, reconciler=reconciler)


# The user's real dogfooding memories — content AND tags verbatim from the
# project store (tags are part of embedding_text(); they lift the pairwise
# hashing cosine to ~0.7, above the 0.6 recon-cluster bar — measured on the
# real rows).
HARNESS_TAGS = ["reflection_harness_test", "duplicate_cluster_A", "idle_reflection"]
HARNESS_DUPES = [
    (
        "Reflection harness duplicate A1",
        "Reflection harness test: duplicate cluster A says the system should "
        "consolidate overlapping memory records when idle reflection is "
        "triggered by the runtime, not manually by the agent.",
    ),
    (
        "Reflection harness duplicate A2",
        "Reflection harness test: duplicate cluster A records that idle/runtime "
        "reflection should merge or link redundant memories about the harness "
        "trigger without the agent calling reflect() directly.",
    ),
    (
        "Reflection harness duplicate A3",
        "Reflection harness test: redundant memory for cluster A; expected "
        "consolidation target is these similar records about testing the "
        "harness reflection trigger while avoiding intentional "
        "self.memory.reflect().",
    ),
]

CONSOLIDATED_TEXT = (
    "Reflection harness test (cluster A): idle/runtime-triggered reflection "
    "should consolidate these overlapping records without the agent calling "
    "self.memory.reflect() directly."
)


def _reconciler_script(prompt: str) -> str:
    """Behave like a competent model: supersede every id shown in the cluster."""
    ids = re.findall(r"id=([0-9a-f]{32})", prompt)
    return json.dumps({"redundant": True, "consolidated": CONSOLIDATED_TEXT, "supersede": ids})


# --------------------------------------------------------------------------
# the dogfooding regression: A1/A2/A3 get consolidated
# --------------------------------------------------------------------------
def test_harness_duplicates_are_consolidated_by_reconciler():
    agent = MemAgent()
    llm = ScriptedLLM(_reconciler_script)
    mgr = _install(agent, reconciler=llm_reconciler(lambda: llm))
    for title, content in HARNESS_DUPES:
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)
    assert mgr.store.count() == 3  # paraphrases: dedup-on-write did NOT fold them

    report = mgr.reflect_interruptible(lambda: False, trigger="idle")

    assert report.merged == 0  # cosine ~0.7 << 0.95: deterministic merge untouched
    assert report.reconciled == 1
    assert report.superseded == 3
    assert len(llm.calls) == 1  # one cluster, one LLM call

    active = mgr.store.all_memories()
    harness = [m for m in active if "Reflection harness" in m.content]
    assert len(harness) == 1  # ONE consolidated record survives
    assert harness[0].content == CONSOLIDATED_TEXT
    assert set(HARNESS_TAGS) <= set(harness[0].tags)  # cluster tags carried over
    # (importance is subsequently renormalized by the rescore step — not pinned)
    # provenance: REFINES edges back to every superseded record
    refines = {e.target_id for e in harness[0].edges if e.type is EdgeType.REFINES}
    archived = {m.id for m in mgr.store.all_memories(include_archived=True) if m.archived}
    assert refines == archived and len(archived) == 3
    # and the consolidated record is what recall now surfaces
    found = agent.recall("reflection harness duplicate cluster")
    assert any(m.content == CONSOLIDATED_TEXT for m in found)
    assert all("duplicate cluster A says" not in m.content for m in found)


def test_distinct_facts_survive_reconciliation():
    agent = MemAgent()
    llm = ScriptedLLM(lambda prompt: json.dumps({"redundant": False}))
    mgr = _install(agent, reconciler=llm_reconciler(lambda: llm))
    for title, content in HARNESS_DUPES:
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)

    report = mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert report.reconciled == 0 and report.superseded == 0
    assert mgr.store.count() == 3  # the model said "distinct" -> nothing archived


def test_malformed_llm_output_is_contained():
    agent = MemAgent()
    llm = ScriptedLLM(lambda prompt: "sorry, I can't produce JSON today")
    mgr = _install(agent, reconciler=llm_reconciler(lambda: llm))
    for title, content in HARNESS_DUPES:
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)

    report = mgr.reflect_interruptible(lambda: False, trigger="idle")  # no raise
    assert report.reconciled == 0
    assert mgr.store.count() == 3  # cluster skipped, store untouched


def test_hallucinated_ids_cannot_archive_unrelated_memories():
    agent = MemAgent()
    llm = ScriptedLLM(
        lambda prompt: json.dumps(
            {"redundant": True, "consolidated": None, "supersede": ["f" * 32]}
        )
    )
    mgr = _install(agent, reconciler=llm_reconciler(lambda: llm))
    for title, content in HARNESS_DUPES:
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)
    victim = agent.remember("an unrelated fact about deployments", type="info")

    mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert mgr.store.get(victim).archived is False  # ids validated against the cluster


def test_cluster_cap_bounds_llm_calls():
    agent = MemAgent()
    llm = ScriptedLLM(lambda prompt: json.dumps({"redundant": False}))
    mgr = _install(
        agent,
        reconciler=llm_reconciler(lambda: llm),
        reflection=ReflectionPolicy(max_clusters_per_reflection=1),
    )
    for title, content in HARNESS_DUPES:  # one cluster's worth
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)
    agent.remember("apples pears and fruit baskets at the market", type="info")
    agent.remember("apples pears and fruit stalls at the market fair", type="info")

    mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert len(llm.calls) <= 1  # the cap held, leftovers wait for the next window


# --------------------------------------------------------------------------
# the reasoner step: episodes -> reflection memories
# --------------------------------------------------------------------------
def _reasoner_script(prompt: str) -> str:
    return json.dumps(
        {
            "insights": [
                {
                    "title": "Deploy retries",
                    "content": "Deploys that fail on the first attempt usually "
                    "succeed after a registry-cache retry; retry before paging.",
                    "tags": ["deploy", "retry"],
                }
            ]
        }
    )


def test_reasoner_abstracts_episodes_into_reflections():
    agent = MemAgent()
    llm = ScriptedLLM(_reasoner_script)
    mgr = _install(agent, reasoner=llm_reasoner(lambda: llm))
    e1 = mgr.remember("Episode: deploy failed once, retry succeeded", type=MemoryType.EPISODE)
    e2 = mgr.remember(
        "Episode: deploy failed, cache purge then retry worked", type=MemoryType.EPISODE
    )

    report = mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert report.created == 1
    reflections = [m for m in mgr.store.all_memories() if m.type is MemoryType.REFLECTION]
    assert len(reflections) == 1
    insight = reflections[0]
    assert "retry before paging" in insight.content
    assert insight.title == "Deploy retries"
    assert insight.importance == 6.0
    derived = {e.target_id for e in insight.edges if e.type is EdgeType.DERIVED_FROM}
    assert derived == {e1, e2}  # provenance back to the episodes it saw


def test_reasoner_never_called_without_episodes():
    agent = MemAgent()
    llm = ScriptedLLM(_reasoner_script)
    mgr = _install(agent, reasoner=llm_reasoner(lambda: llm))
    agent.remember("just an info fact, no episodes here", type="info")

    report = mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert report.created == 0
    assert llm.calls == []  # starved reasoner: no LLM call at all


def test_lazy_model_binding_follows_the_getter():
    agent = MemAgent()
    first = ScriptedLLM(lambda prompt: json.dumps({"redundant": False}))
    second = ScriptedLLM(lambda prompt: json.dumps({"redundant": False}))
    current = {"llm": first}
    mgr = _install(agent, reconciler=llm_reconciler(lambda: current["llm"]))
    for title, content in HARNESS_DUPES:
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)

    mgr.reflect_interruptible(lambda: False, trigger="idle")
    current["llm"] = second  # the host switched models (/model)
    mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert len(first.calls) == 1 and len(second.calls) == 1


def test_stop_flag_prevents_reconciler_calls():
    agent = MemAgent()
    llm = ScriptedLLM(_reconciler_script)
    mgr = _install(agent, reconciler=llm_reconciler(lambda: llm))
    for title, content in HARNESS_DUPES:
        agent.remember(content, type="info", title=title, tags=HARNESS_TAGS)

    report = mgr.reflect_interruptible(lambda: True, trigger="idle")
    assert report.interrupted is True
    assert llm.calls == []  # stopping never starts an LLM call
    assert mgr.store.count() == 3


# --------------------------------------------------------------------------
# the episode writer (phase 1 of a generative idle reflection)
# --------------------------------------------------------------------------
def test_episode_writer_produces_episode_text():
    from nooa_memory.generative import llm_episode_writer

    llm = ScriptedLLM(
        lambda prompt: json.dumps(
            {"noteworthy": True, "episode": "Episode: shipped the owner-role change."}
        )
    )
    writer = llm_episode_writer(lambda: llm)
    out = writer("- [Task] work on the owner change\n- [Message] done, tests green")
    assert out == "Episode: shipped the owner-role change."
    assert "owner change" in llm.calls[0]  # the transcript reached the prompt


def test_episode_writer_not_noteworthy_returns_none():
    from nooa_memory.generative import llm_episode_writer

    llm = ScriptedLLM(lambda prompt: json.dumps({"noteworthy": False, "episode": None}))
    writer = llm_episode_writer(lambda: llm)
    assert writer("- [Task] hi") is None


def test_episode_writer_empty_transcript_skips_llm():
    from nooa_memory.generative import llm_episode_writer

    llm = ScriptedLLM(lambda prompt: "should never be called")
    writer = llm_episode_writer(lambda: llm)
    assert writer("") is None
    assert llm.calls == []


def test_render_recent_events_oldest_first():
    from nooa_memory.generative import render_recent_events

    agent = MemAgent()
    agent.event_manager.add(Task(prompt="first: investigate the bug"))
    agent.event_manager.add(Task(prompt="second: write the fix"))
    text = render_recent_events(agent)
    assert "investigate the bug" in text and "write the fix" in text
    assert text.index("investigate the bug") < text.index("write the fix")
