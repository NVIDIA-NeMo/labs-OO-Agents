# SPDX-FileCopyrightText: Copyright (c) 2026 Sampoorn Nagpal
# SPDX-License-Identifier: Apache-2.0
"""Injection accounting follows the budgeted context, using a real manager/store."""

import pytest
from nooa_memory import Memory, MemoryConfig, MemoryManager, MemoryRef, MemoryToolsMixin, MemoryType
from nooa_memory.config import ObservabilityConfig, SpontaneousConfig

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class BudgetAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


@pytest.fixture
def setup(monkeypatch):
    agent = BudgetAgent()
    manager = MemoryManager.install(
        agent,
        config=MemoryConfig(enabled=True, path=":memory:", spontaneous=SpontaneousConfig(top_k=2)),
    )
    first = Memory(id="first001", type=MemoryType.INFO, content="A" * 200)
    second = Memory(id="second02", type=MemoryType.INFO, content="SECOND_MARKER")
    for memory in (first, second):
        manager.store.add(memory)
    # Keep candidate order fixed; persistence, formatting, references, context
    # installation, access logging, and events all use the real implementation.
    monkeypatch.setattr(
        manager.retrieval,
        "recall",
        lambda *args, **kwargs: [manager.store.get(first.id), manager.store.get(second.id)],
    )
    yield agent, manager, first, second
    manager.uninstall()
    manager.store.close()


def _budget(manager, value, **extra):
    manager.config = manager.config.model_copy(
        update={
            "spontaneous": manager.config.spontaneous.model_copy(
                update={"context_char_budget": value, **extra}
            )
        }
    )


def _expected_text(manager, memories):
    return "\n".join(
        [
            "## Recalled memories (associative)",
            *[line for memory in memories for line in manager._memory_lines(memory)],
        ]
    )


def test_wholly_truncated_memory_is_not_logged(setup):
    _, manager, first, second = setup
    _budget(manager, 80)
    text = manager._format_recall(["deployment"])
    assert second.id not in text and second.content not in text
    assert manager._last_injected_ids == [first.id]
    assert manager.store.get(first.id).injected_count == 1
    assert manager.store.get(second.id).injected_count == 0


@pytest.mark.parametrize("cut", [0, 10, 35, 50, 51, 52, 80, 250, 270, 1000])
def test_partial_record_requires_complete_identity_marker(setup, cut):
    _, manager, first, second = setup
    full = _expected_text(manager, [first, second])
    prefix = full[:cut].rstrip() if len(full) > cut else full
    expected = [m.id for m in (first, second) if f"[info#{m.id}]" in prefix]
    _budget(manager, cut)
    text = manager._format_recall(["deployment"])
    assert text == (prefix + " …" if len(full) > cut else full)
    assert manager._last_injected_ids == expected
    for memory in (first, second):
        assert manager.store.get(memory.id).injected_count == int(memory.id in expected)


def test_all_fit_preserves_output_and_passive_actr_state(setup):
    _, manager, first, second = setup
    full = _expected_text(manager, [first, second])
    before = {m.id: (m.strength, m.access_count, m.last_accessed_at) for m in (first, second)}
    _budget(manager, len(full))
    assert manager._format_recall(["deployment"]) == full
    assert manager._last_injected_ids == [first.id, second.id]
    for memory in (first, second):
        saved = manager.store.get(memory.id)
        assert saved.injected_count == 1
        assert (saved.strength, saved.access_count, saved.last_accessed_at) == before[memory.id]


@pytest.mark.parametrize("cut", [40, 80, 1000])
def test_event_counts_and_ids_agree_with_retained_memories(setup, cut):
    agent, manager, first, second = setup
    _budget(manager, cut)
    events = []
    agent.event_manager.on("MemoryInjected", events.append)
    agent.event_manager.add(Task(prompt="deployment"))
    manager._on_before_turn(None)
    assert len(events) == 1
    event = events[0]
    text = agent.context_manager[manager.config.spontaneous.context_block_key]
    expected = [m.id for m in (first, second) if f"[info#{m.id}]" in text]
    assert event.memory_ids == expected
    assert event.n_memories == len(expected)
    assert event.chars == len(text)
    assert manager.stats.injected_chars == len(text)


def test_logging_disabled_still_reports_only_retained_ids(setup):
    _, manager, first, second = setup
    manager.config = manager.config.model_copy(
        update={"observability": ObservabilityConfig(log_injections=False)}
    )
    _budget(manager, 80)
    manager._format_recall(["deployment"])
    assert manager._last_injected_ids == [first.id]
    assert (
        manager.store.get(first.id).injected_count
        == manager.store.get(second.id).injected_count
        == 0
    )


def test_empty_query_clears_prior_injection_metadata(setup):
    _, manager, _, _ = setup
    manager._format_recall(["deployment"])
    assert manager._last_injected_ids
    assert manager._format_recall([]) == ""
    assert manager._last_injected_ids == []


def test_reference_truncation_does_not_log_later_record(setup):
    agent, manager, first, second = setup
    agent.vars = {"live": "reference " + "R" * 200}
    first.content = "short"
    first.references = [MemoryRef(kind="var", key="live")]
    manager.store.save(first)
    full = _expected_text(manager, [first, second])
    cut = full.index("reference") + len("reference")
    _budget(manager, cut)
    text = manager._format_recall(["deployment"])
    assert "reference" in text and second.id not in text
    assert manager._last_injected_ids == [first.id]
    assert manager.store.get(second.id).injected_count == 0


@pytest.mark.parametrize("cut", [80, 1000])
def test_always_visible_todos_use_same_accounting_rule(setup, monkeypatch, cut):
    _, manager, first, second = setup
    second.type = MemoryType.TODO
    second.status = "open"
    manager.store.save(second)
    monkeypatch.setattr(
        manager.retrieval, "recall", lambda *args, **kwargs: [manager.store.get(first.id)]
    )
    _budget(manager, cut, inject_open_todos="always")
    text = manager._format_recall(["deployment"])
    expected = [first.id] + ([second.id] if f"[todo:open#{second.id}]" in text else [])
    assert manager._last_injected_ids == expected
    assert manager.store.get(second.id).injected_count == int(second.id in expected)
