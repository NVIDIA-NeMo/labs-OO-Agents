# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventsApi — Skill wrapper for event queries."""

import pytest

from nooa import Agent
from nooa.events import Task
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
