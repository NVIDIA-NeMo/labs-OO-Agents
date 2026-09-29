# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventsApi — Skill wrapper for event queries."""

import pytest

from nooa import Agent
from nooa.context_blocks import ResultStatus
from nooa.events import PythonOutput, Task
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


class _TestAgent(Agent, llm=_LLM):
    pass


@pytest.fixture
def api_with_event():
    agent = _TestAgent()
    event = Task(prompt="test")
    agent.event_manager.add(event)
    api = agent.events
    tag = list(agent.event_manager.keys())[0]
    return api, tag


def test_query_works():
    agent = _TestAgent()
    assert isinstance(agent.events.query(), list)


def test_get_single_returns_event(api_with_event):
    api, tag = api_with_event
    assert api.get(tag) is not None


def test_get_list_returns_found(api_with_event):
    api, tag = api_with_event
    result = api.get([tag, "missing"])
    assert len(result) == 1


def test_get_list_empty_for_all_missing():
    agent = _TestAgent()
    assert agent.events.get(["nonexistent-1", "nonexistent-2"]) == []


def test_getitem_returns_event(api_with_event):
    api, tag = api_with_event
    assert api[tag] is not None


def test_getitem_list_returns_events(api_with_event):
    api, tag = api_with_event
    result = api[[tag]]
    assert len(result) == 1


def test_getitem_raises_for_missing():
    agent = _TestAgent()
    with pytest.raises(KeyError):
        _ = agent.events["nonexistent"]


def test_getitem_list_raises_for_missing():
    agent = _TestAgent()
    with pytest.raises(KeyError):
        _ = agent.events[["nonexistent-1", "nonexistent-2"]]


def test_contains(api_with_event):
    api, tag = api_with_event
    assert tag in api
    assert "missing" not in api


def test_repr(api_with_event):
    api, _ = api_with_event
    assert "EventsApi" in repr(api)


def test_collapse_returns_summary_tag():
    """collapse() delegates to EventManager and returns the summary tag."""
    agent = _TestAgent()
    em = agent.event_manager
    for _ in range(5):
        em.add(Task(prompt="filler"))
    tags = list(em.keys())
    first, last = tags[0], tags[-1]
    summary_tag = agent.events.collapse(first, last, summary_text="collapsed")
    assert ".." in summary_tag
    assert agent.events.get(summary_tag) is not None
    # Individual collapsed tags remain accessible
    assert agent.events.get(first) is not None
    assert agent.events.get(last) is not None


@pytest.mark.parametrize(
    "start,end,previous",
    [
        (1, 3, None),
        ("1", 3, None),
        (1, "3", None),
        ("1", "3", None),
        ("1..2", 3, ("1", "2")),
        (1, "2..3", ("2", "3")),
        (1, 3, ("1", "2")),
    ],
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_collapse_accepts_existing_integer_tags(start, end, previous, backend, tmp_path, capsys):
    from nooa.storage.sqlite import SQLiteStorageManager

    agent = _TestAgent()
    storage = SQLiteStorageManager(tmp_path / "events.db") if backend == "sqlite" else None
    if storage is not None:
        agent.event_manager.set_backend(storage.event_backend)
    try:
        tags = [agent.event_manager.add(Task(prompt=f"original {i}")) for i in range(3)]
        assert tags == ["1", "2", "3"]
        if previous:
            agent.events.collapse(*previous, summary_text="earlier summary")
        children = agent.events.keys()
        capsys.readouterr()

        result = agent.events.collapse(start, end, summary_text="combined summary")

        assert result == "1..3"
        assert agent.events.keys() == [result]
        assert agent.events[result].children_tags == children
        assert agent.events[result].summary_text == "combined summary"
        assert all(agent.events.get(tag) is not None for tag in tags)
        assert capsys.readouterr().out == ""
    finally:
        if storage is not None:
            storage.close()


@pytest.mark.parametrize("bad", [0, 999, True, False, 1.0, None])
@pytest.mark.parametrize("side", ["start", "end"])
def test_collapse_invalid_numeric_boundary_does_not_change_history(bad, side, capsys):
    agent = _TestAgent()
    tags = [agent.event_manager.add(Task(prompt="original")) for _ in range(3)]
    before = agent.events.keys()
    start, end = (bad, tags[-1]) if side == "start" else (tags[0], bad)
    error = ValueError if type(bad) is int else TypeError
    with pytest.raises(error, match="tag"):
        agent.events.collapse(start, end, summary_text="must not be written")
    assert agent.events.keys() == before
    assert all(agent.events[tag].prompt == "original" for tag in tags)
    assert "Please use strings" not in capsys.readouterr().out


@pytest.mark.parametrize("nested", [False, True])
def test_summary_documents_working_archive_recovery(nested):
    """Recovery instructions come from the archive, not generated summary text."""
    agent = _TestAgent()
    originals = [Task(prompt="needle: exact decision"), Task(prompt="other detail")]
    tags = [agent.event_manager.add(event) for event in originals]
    summary_tag = agent.events.collapse(*tags, summary_text="short recap")
    if nested:
        last = agent.event_manager.add(Task(prompt="later detail"))
        summary_tag = agent.events.collapse(summary_tag, last, summary_text="second recap")
    summary = agent.events[summary_tag]
    search = 'self.events.query(query="keyword", limit=10)'
    read = f'self.events["{tags[0]}"]'
    expand = f'self.events[self.events["{summary_tag}"].children_tags]'
    for expression in (search, read, expand):
        assert expression in summary.doc
    assert "archived" in summary.doc
    assert "nested" in summary.doc
    # Execute the documented API operations: collapse must not hide source data.
    assert agent.events.query(query="needle", limit=10) == [originals[0]]
    assert agent.events[tags[0]] is originals[0]
    children = agent.events[summary.children_tags]
    if nested:
        children = agent.events[children[0].children_tags]
    assert children == originals


@pytest.mark.parametrize("formatter_name", ["MarkdownBlockFormatter", "XMLBlockFormatter"])
def test_summary_recovery_instructions_reach_model_context(formatter_name):
    from nooa.context_blocks import formatter

    agent = _TestAgent()
    tag = agent.event_manager.add(Task(prompt="original"))
    summary = agent.events[agent.events.collapse(tag, tag, summary_text="recap")]
    rendered = getattr(formatter, formatter_name)().format_event(summary)
    assert "self.events.query" in rendered
    assert "limit=10" in rendered
    assert "children_tags" in rendered


def test_keys_reflects_active_tags():
    """keys() exposes the active tag list from the manager."""
    agent = _TestAgent()
    em = agent.event_manager
    assert agent.events.keys() == list(em.keys())
    em.add(Task(prompt="one"))
    em.add(Task(prompt="two"))
    keys = agent.events.keys()
    assert len(keys) == 2
    assert keys == list(em.keys())
    # After collapse, range tag replaces individual tags in keys()
    first, last = keys[0], keys[-1]
    agent.events.collapse(first, last)
    post_keys = agent.events.keys()
    assert len(post_keys) == 1
    assert ".." in post_keys[0]


def test_collapse_invalid_range_raises():
    """collapse() with invalid range propagates ValueError."""
    agent = _TestAgent()
    with pytest.raises((ValueError, KeyError)):
        agent.events.collapse("999", "1000")


def test_query_filters_execution_status_and_limit():
    agent = _TestAgent()
    for count, status in enumerate(
        (ResultStatus.ERROR, ResultStatus.COMPLETE, ResultStatus.ERROR), start=1
    ):
        agent.event_manager.add(
            PythonOutput(
                tool_call_id=str(count),
                execution_count=count,
                execution_status=status,
                error="failed" if status is ResultStatus.ERROR else "",
            )
        )

    failures = agent.events.query(type="PythonOutput", execution_status="error", limit=1)

    assert len(failures) == 1
    assert failures[0].execution_count == 3
