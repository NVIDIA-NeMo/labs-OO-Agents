# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nooa.events import Message, Summary
from nooa.runtime.event_manager import EventManager


def test_summary_provenance_survives_nested_collapse():
    manager = EventManager()
    tags = [manager.add(Message(content="fact", call_id=call_id)) for call_id in ("B", "A", "C")]
    first = manager.collapse(tags[0], tags[1], "AB")
    assert manager[first].source_call_ids == ["A", "B"]
    final = manager.collapse(first, tags[2], "ABC")
    summary = manager[final]
    assert summary.source_call_ids == ["A", "B", "C"]
    assert "A" in summary.related_call_ids
    assert "D" not in summary.related_call_ids
    assert Summary.model_validate_json(summary.model_dump_json()).source_call_ids == ["A", "B", "C"]


def test_query_origin_and_relevance():
    from nooa import Agent
    from nooa.unifiedllm import FakeLLMClient

    agent = Agent(llm=FakeLLMClient())
    manager = agent.event_manager
    a = manager.add(Message(content="A", call_id="A"))
    b = manager.add(Message(content="B", call_id="B"))
    summary = manager.collapse(a, b, "AB")
    assert [e.tag for e in agent.events.query(call_id="A")] == [a]
    assert {e.tag for e in agent.events.query(related_call_ids={"A"})} == {a, summary}
    assert {e.tag for e in agent.events.query(related_call_ids={"A", "B"})} == {a, b, summary}
    assert agent.events.query(related_call_ids=set()) == []
    assert agent.events.query(call_id="A", related_call_ids={"B"}) == []
    assert len(agent.events.query(related_call_ids=None)) == 3
    with pytest.raises(TypeError):
        agent.events.query(related_call_ids="A")


def test_missing_origin_and_required_summary_provenance():
    manager = EventManager()
    a = manager.add(Message(content="A", call_id="A"))
    b = manager.add(Message(content="unattributed"))
    old = manager.collapse(a, b, "old")
    c = manager.add(Message(content="C", call_id="C"))
    new = manager.collapse(old, c, "new")
    assert manager[new].source_call_ids == ["A", "C"]
    assert manager[a].related_call_ids == frozenset({"A"})
    assert manager[b].related_call_ids == frozenset()
    assert manager[new].call_id is None
    with pytest.raises(ValueError):
        Summary(summary_tag="1..2", replaced_range=(1, 2))


def test_lifecycle_origin_is_consistent():
    from nooa.events import AfterAgentCall, BeforeAgentCall

    for cls in (BeforeAgentCall, AfterAgentCall):
        event = cls(method_name="run", call_id="A", is_top_level=True)
        assert event.call_id == "A"
        assert event.related_call_ids == frozenset({"A"})
        assert event.model_dump()["call_id"] == "A"
        assert "call_id" not in event.metadata
        assert cls.model_validate_json(event.model_dump_json()).call_id == "A"


def test_call_id_is_a_typed_field_not_metadata():
    event = Message(content="fact", call_id="A", metadata={"source": "test"})
    assert event.model_dump()["call_id"] == "A"
    assert event.metadata == {"source": "test"}
    assert "call_id" in Message.model_json_schema()["properties"]
    assert Message.model_validate_json(event.model_dump_json()).call_id == "A"
    assert Message(content="fact", metadata={"call_id": "ignored"}).call_id is None
    with pytest.raises(ValueError):
        Message(content="fact", call_id=123)
