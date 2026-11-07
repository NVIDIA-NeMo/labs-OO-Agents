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


@pytest.mark.parametrize(
    ("cut", "n_kept"),
    # 52 is the header, its newline and the first marker; below that nothing fits.
    [(0, 0), (10, 0), (35, 0), (50, 0), (51, 0), (52, 1), (80, 1), (250, 2), (270, 2), (1000, 2)],
)
def test_record_is_kept_only_with_its_complete_identity_marker(setup, cut, n_kept):
    _, manager, first, second = setup
    expected = [m.id for m in (first, second)][:n_kept]
    _budget(manager, cut)
    text = manager._format_recall(["deployment"])
    assert len(text) <= cut
    assert text.count("[info#") == n_kept
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


@pytest.mark.parametrize("cut", [60, 80, 1000])
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


def test_budget_below_one_marker_injects_nothing(setup):
    agent, manager, first, second = setup
    key = manager.config.spontaneous.context_block_key
    agent.context_manager.set_dynamic(key, value="stale block")
    _budget(manager, 40)
    events = []
    agent.event_manager.on("MemoryInjected", events.append)
    agent.event_manager.add(Task(prompt="deployment"))
    manager._on_before_turn(None)
    assert key not in agent.context_manager
    assert events == []
    assert manager._last_injected_ids == []
    assert manager.store.get(first.id).injected_count == 0
    assert manager.store.get(second.id).injected_count == 0


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


def test_reference_that_does_not_fit_is_dropped_whole(setup):
    agent, manager, first, second = setup
    agent.vars = {"live": "reference " + "R" * 200}
    first.content = "short"
    first.references = [MemoryRef(kind="var", key="live")]
    manager.store.save(first)
    full = _expected_text(manager, [first, second])
    cut = full.index("reference") + len("reference")
    _budget(manager, cut)
    text = manager._format_recall(["deployment"])
    assert "reference" not in text
    assert f"[info#{first.id}] short" in text and second.content in text
    assert manager._last_injected_ids == [first.id, second.id]
    assert manager.store.get(second.id).injected_count == 1


def test_reference_that_fits_survives_packing(setup):
    agent, manager, first, second = setup
    agent.vars = {"live": "reference value"}
    first.references = [MemoryRef(kind="var", key="live")]
    manager.store.save(first)
    full = _expected_text(manager, [first, second])
    _budget(manager, len(full) - 1)  # over budget only because of the long head
    text = manager._format_recall(["deployment"])
    assert "reference value" in text and second.content in text
    assert manager._last_injected_ids == [first.id, second.id]


@pytest.mark.parametrize(
    ("cut", "keeps_first"),
    [(80, False), (1000, True)],  # at 80 the todo claims the room the record would need
)
def test_always_visible_todos_use_same_accounting_rule(setup, monkeypatch, cut, keeps_first):
    _, manager, first, second = setup
    second.type = MemoryType.TODO
    second.status = "open"
    manager.store.save(second)
    monkeypatch.setattr(
        manager.retrieval, "recall", lambda *args, **kwargs: [manager.store.get(first.id)]
    )
    _budget(manager, cut, inject_open_todos="always")
    text = manager._format_recall(["deployment"])
    assert f"[todo:open#{second.id}]" in text
    assert manager._last_injected_ids == ([first.id] if keeps_first else []) + [second.id]
    assert manager.store.get(first.id).injected_count == int(keeps_first)
    assert manager.store.get(second.id).injected_count == 1


def test_bracket_inside_memory_id_does_not_end_identity_marker(setup, monkeypatch):
    _, manager, _, _ = setup
    odd = Memory(id="ab]cdefgh", type=MemoryType.INFO, content="ODD_MARKER")
    manager.store.add(odd)
    monkeypatch.setattr(
        manager.retrieval, "recall", lambda *args, **kwargs: [manager.store.get(odd.id)]
    )
    header = "## Recalled memories (associative)\n"
    marker = "- [info#ab]cdefg]"
    # Cut right after the ']' that belongs to the id: the marker is still incomplete.
    _budget(manager, len(header) + marker.index("]") + 1)
    manager._format_recall(["deployment"])
    assert manager._last_injected_ids == []
    # Cut right after the real closing ']': the marker is complete.
    _budget(manager, len(header) + len(marker))
    manager._format_recall(["deployment"])
    assert manager._last_injected_ids == [odd.id]


def _recall_returns(monkeypatch, manager, memories):
    for memory in memories:
        manager.store.add(memory)
    monkeypatch.setattr(
        manager.retrieval,
        "recall",
        lambda *args, **kwargs: [manager.store.get(m.id) for m in memories],
    )


def _info(letter, length):
    return Memory(id=f"{letter}" * 8, type=MemoryType.INFO, content=letter.upper() * length)


@pytest.mark.parametrize("over", [1, 6, 100])
def test_block_just_over_budget_loses_only_the_overflow(setup, monkeypatch, over):
    _, manager, _, _ = setup
    memories = [_info(letter, 474) for letter in "abcd"]
    _recall_returns(monkeypatch, manager, memories)
    full = _expected_text(manager, memories)
    _budget(manager, len(full) - over, top_k=4)
    text = manager._format_recall(["deployment"])
    assert len(text) == len(full) - over
    assert manager._last_injected_ids == [m.id for m in memories]
    # The earlier ranks stay whole; the shortfall comes off the last head.
    for memory in memories[:3]:
        assert memory.content in text
    assert memories[3].content not in text
    assert text.endswith("…")


def test_regrowth_keeps_short_records_beside_a_long_one(setup, monkeypatch):
    _, manager, _, _ = setup
    long_one = _info("a", 4000)
    short = [_info(letter, 40) for letter in "bcde"]
    _recall_returns(monkeypatch, manager, [long_one, *short])
    _budget(manager, 2000, top_k=5)
    text = manager._format_recall(["deployment"])
    assert len(text) == 2000
    assert manager._last_injected_ids == [m.id for m in (long_one, *short)]
    for memory in short:
        assert memory.content in text


def test_regrowth_does_not_bring_back_a_dropped_reference(setup, monkeypatch):
    agent, manager, _, _ = setup
    agent.vars = {"live": "reference " + "R" * 150}
    first = _info("a", 300)
    first.references = [MemoryRef(kind="var", key="live")]
    second = _info("b", 13)
    _recall_returns(monkeypatch, manager, [first, second])
    full = _expected_text(manager, [first, second])
    # Room for the whole head and the second record, but not for the reference
    # beside even a capped head.
    head_only = [manager._head_line(first), manager._head_line(second)]
    _budget(manager, len("\n".join(["## Recalled memories (associative)", *head_only])))
    assert len(full) > manager.config.spontaneous.context_char_budget
    text = manager._format_recall(["deployment"])
    assert first.content in text and second.content in text
    assert "reference" not in text


def test_long_always_todo_regrows_without_starving_recalled_records(setup, monkeypatch):
    _, manager, _, _ = setup
    recalled = _info("a", 30)
    todo = Memory(id="t" * 8, type=MemoryType.TODO, status="open", content="T" * 3000)
    _recall_returns(monkeypatch, manager, [recalled])
    manager.store.add(todo)
    _budget(manager, 500, inject_open_todos="always")
    text = manager._format_recall(["deployment"])
    assert len(text) == 500
    assert recalled.content in text
    assert manager._last_injected_ids == [recalled.id, todo.id]


def test_regrowth_follows_the_always_section_first(setup, monkeypatch):
    _, manager, _, _ = setup
    recalled = _info("a", 400)
    todo = Memory(id="t" * 8, type=MemoryType.TODO, status="open", content="T" * 400)
    _recall_returns(monkeypatch, manager, [recalled])
    manager.store.add(todo)
    _budget(manager, 700, inject_open_todos="always")
    text = manager._format_recall(["deployment"])
    assert len(text) == 700
    assert todo.content in text and recalled.content not in text
    assert manager._last_injected_ids == [recalled.id, todo.id]


def test_packing_invariants_hold_across_budgets(setup, monkeypatch):
    agent, manager, _, _ = setup
    agent.vars = {"short": "S" * 20, "long": "L" * 190}
    lengths = {"a": 700, "b": 30, "c": 300, "d": 90}
    memories = [_info(letter, length) for letter, length in lengths.items()]
    memories[0].references = [MemoryRef(kind="var", key="short")]
    memories[2].references = [
        MemoryRef(kind="var", key="short"),
        MemoryRef(kind="var", key="long"),
    ]
    todo = Memory(id="t" * 8, type=MemoryType.TODO, status="open", content="T" * 250)
    _recall_returns(monkeypatch, manager, memories)
    manager.store.add(todo)
    records = [*memories, todo]
    for budget in range(0, 1700, 3):
        _budget(manager, budget, inject_open_todos="always", top_k=4)
        text = manager._format_recall(["deployment"])
        assert len(text) <= budget
        expected = [m.id for m in records if manager._identity_marker(m) in text]
        assert manager._last_injected_ids == expected
        assert text.count("[info#") + text.count("[todo:open#") == len(expected)
        # While any head is still cut, no budget is left unspent.
        heads = [line for line in text.split("\n") if line.startswith("- [")]
        if any(line.endswith("…") for line in heads):
            assert len(text) == budget


def _todo(letter, importance):
    return Memory(
        id=f"{letter}" * 8,
        type=MemoryType.TODO,
        status="open",
        content=f"todo {letter}",
        importance=importance,
    )


@pytest.mark.parametrize(("cut", "keeps_low"), [(60, False), (80, True)])
def test_recalled_todo_keeps_its_priority_over_a_less_important_todo(
    setup, monkeypatch, cut, keeps_low
):
    _, manager, _, _ = setup
    other = _info("a", 30)
    high, low = _todo("h", 9.0), _todo("l", 1.0)
    _recall_returns(monkeypatch, manager, [other, high])
    manager.store.add(low)
    _budget(manager, cut, inject_open_todos="always")
    text = manager._format_recall(["deployment"])
    assert len(text) <= cut
    assert manager._last_injected_ids == ([high.id, low.id] if keeps_low else [high.id])
    assert manager.store.get(high.id).injected_count == 1
    assert manager.store.get(low.id).injected_count == int(keeps_low)
    assert manager.store.get(other.id).injected_count == 0
    assert text.startswith("## Open todos")


def test_recalled_todo_stays_in_its_section_when_the_block_fits(setup, monkeypatch):
    _, manager, _, _ = setup
    high, low = _todo("h", 9.0), _todo("l", 1.0)
    _recall_returns(monkeypatch, manager, [high])
    manager.store.add(low)
    _budget(manager, 2000, inject_open_todos="always")
    text = manager._format_recall(["deployment"])
    assert text.index("## Recalled memories") < text.index(manager._identity_marker(high))
    assert text.index(manager._identity_marker(high)) < text.index("## Open todos")
    assert text.index("## Open todos") < text.index(manager._identity_marker(low))
    assert manager._last_injected_ids == [high.id, low.id]


def test_recalled_todo_outside_the_selected_todos_keeps_its_rank(setup, monkeypatch):
    _, manager, _, _ = setup
    high, low = _todo("h", 9.0), _todo("l", 1.0)
    _recall_returns(monkeypatch, manager, [low])
    manager.store.add(high)
    # open_todos_k=1 selects only the important todo, so the recalled one is an
    # ordinary associative hit and packs after the "always" section.
    _budget(manager, 72, inject_open_todos="always", open_todos_k=1)
    text = manager._format_recall(["deployment"])
    assert manager._last_injected_ids == [high.id]
    assert manager._identity_marker(low) not in text
    # One character over budget, the recalled todo stays in its own section.
    _budget(manager, 107, inject_open_todos="always", open_todos_k=1)
    text = manager._format_recall(["deployment"])
    assert manager._last_injected_ids == [low.id, high.id]
    assert text.index("## Recalled memories") < text.index(manager._identity_marker(low))
    assert text.index(manager._identity_marker(low)) < text.index("## Open todos")


def test_recalled_todo_is_never_rendered_twice_or_lost_to_a_lesser_todo(setup, monkeypatch):
    _, manager, _, _ = setup
    other = _info("a", 200)
    high, low = _todo("h", 9.0), _todo("l", 1.0)
    _recall_returns(monkeypatch, manager, [other, high])
    manager.store.add(low)
    for cut in range(0, 400):
        _budget(manager, cut, inject_open_todos="always")
        text = manager._format_recall(["deployment"])
        assert len(text) <= cut
        for memory in (other, high, low):
            assert text.count(manager._identity_marker(memory)) <= 1
        if text:
            assert high.id in manager._last_injected_ids
        visible = {m.id for m in (other, high, low) if manager._identity_marker(m) in text}
        assert set(manager._last_injected_ids) == visible
