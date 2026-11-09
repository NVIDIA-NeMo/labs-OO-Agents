# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""replayable() repairs a selection for sending, never stored state; rendering only checks."""

import pytest

from nooa import Agent, collect_context_items, is_model_visible, replayable
from nooa.context_blocks import OpenAIProviderFormatter, XMLBlockFormatter, render_context
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.context_blocks.exceptions import UnsupportedContextLayout
from nooa.default_context_view import ManagedContextView, visible_events
from nooa.llm_types import LLMResponse, ToolCall
from nooa.strategies.current_call import CurrentCall


def render(items):
    return render_context(
        items, block_formatter=XMLBlockFormatter(), provider_formatter=OpenAIProviderFormatter()
    ).messages


def execution(call_id, *, response=None, result=None):
    return ToolCallEvent(
        tool_call_id=call_id,
        name="execute_python",
        arguments={"code": "work()"},
        llm_response_id=response.id if response else None,
        result=result,
    )


def response(*ids):
    return LLMResponse(
        tool_calls=[
            ToolCall(id=call_id, name="execute_python", arguments='{"code":"work()"}')
            for call_id in ids
        ],
        finish_reason="tool_calls",
    )


async def test_default_view_omits_incomplete_turn_and_keeps_task():
    from nooa import Context

    class NoOwnBlocksView(ManagedContextView):
        async def view_blocks(self, owner, call):
            return ()

    agent = Agent(llm=object())
    agent.context["task"] = Context("continue", prefix=False)
    event = response("missing")
    agent.event_manager.add(event)
    original = event.model_dump_json()
    try:
        call = CurrentCall(id="next", method_name="run", decorator="plan", agent=agent)
        items = await collect_context_items(NoOwnBlocksView(), agent, call)
        result = render_context(
            items, block_formatter=XMLBlockFormatter(), provider_formatter=OpenAIProviderFormatter()
        ).output
        # Nothing replayable precedes the task, so there is no prefix to mark.
        assert len(result) == 1
        assert "continue" in result[0]["content"]
        assert event.model_dump_json() == original
    finally:
        await agent.aclose()


async def test_default_view_recovers_standalone_missing_result_without_mutation():
    agent = Agent(llm=object())
    event = execution("interrupted")
    agent.event_manager.add(event)
    original = event.model_dump_json()
    call = CurrentCall(id="next", method_name="run", decorator="plan", agent=agent)
    items = await collect_context_items(ManagedContextView(), agent, call)
    messages = render(items)
    tools = [message for message in messages if message.tool_call_id]
    assert len(tools) == 1
    assert "execution outcome is unknown" in tools[0].content
    assert event.model_dump_json() == original
    # Rendering rejects history that did not go through replayable().
    with pytest.raises(UnsupportedContextLayout, match="no result"):
        render([event])
    await agent.aclose()


@pytest.mark.parametrize("missing", ["result", "event", "archived_event"])
def test_canonical_batch_recovers_missing_result_atomically(missing):
    agent = Agent(llm=object())
    canonical = response("a", "b")
    first = execution("a", response=canonical, result=ToolResult(tool_call_id="a", content="done"))
    second = execution("b", response=canonical)
    agent.event_manager.add(canonical)
    agent.event_manager.add(first)
    if missing != "event":
        tag = agent.event_manager.add(second)
        if missing == "archived_event":
            agent.event_manager.collapse(tag, tag, summary_text="Archived execution")
    before = [event.model_dump_json() for event in agent.event_manager.values()]
    items = replayable(visible_events(agent))
    assert items == replayable(visible_events(agent)), "Projection must be deterministic"
    messages = render(items)
    if missing == "result":
        assert [message.role.value for message in messages] == ["assistant", "tool", "tool"]
        assert [call.id for call in messages[0].tool_calls] == ["a", "b"]
        assert messages[0].replay_message is canonical
        assert messages[1].content == "done"
        assert "execution outcome is unknown" in messages[2].content
    elif missing == "event":
        assert messages == [], "Absent sidecars must not be reconstructed"
    else:
        assert len(messages) == 1 and "Archived execution" in messages[0].content
        assert not messages[0].tool_calls and not messages[0].tool_call_id
    assert [event.model_dump_json() for event in agent.event_manager.values()] == before
    assert second.result is None


@pytest.mark.parametrize("status", [ResultStatus.COMPLETE, ResultStatus.RUNNING])
def test_existing_results_and_inline_return_are_not_changed(status):
    agent = Agent(llm=object())
    canonical = response("a")
    completed = execution(
        "a",
        response=canonical,
        result=ToolResult(tool_call_id="a", content="original feedback", result_status=status),
    )
    inline = ToolCallEvent(
        tool_call_id="inline",
        name="return_result",
        arguments={"result": 42},
        result=ToolResult(tool_call_id="inline", content="accepted"),
        metadata={"synthetic_type": "codeact_inline_return"},
    )
    for event in (canonical, completed, inline):
        agent.event_manager.add(event)
    projected = replayable(visible_events(agent))
    assert all(a is b for a, b in zip(projected, (canonical, completed, inline), strict=True))
    messages = render(projected)
    assert len(messages) == 2
    assert [call.name for call in messages[0].tool_calls] == ["execute_python"]
    assert messages[1].content == "original feedback"


def test_orphan_sidecar_does_not_invent_canonical_turn():
    agent = Agent(llm=object())
    agent.event_manager.add(execution("orphan", response=response("orphan")))
    assert render(replayable(visible_events(agent))) == []


def test_duplicate_canonical_ids_remain_unreplayable():
    agent = Agent(llm=object())
    agent.event_manager.add(response("duplicate", "duplicate"))
    assert render(replayable(visible_events(agent))) == []


@pytest.mark.parametrize("gap", ["incomplete_batch", "orphan_result"])
def test_rendering_rejects_gaps_instead_of_dropping_them(gap):
    canonical = response("a", "b")
    done = ToolResult(tool_call_id="a", content="done")
    events = (
        [canonical, execution("a", response=canonical, result=done)]
        if gap == "incomplete_batch"
        else [execution("a", response=canonical, result=done)]
    )
    with pytest.raises(UnsupportedContextLayout, match="replayable"):
        render(events)
    assert render(replayable(events)) == []


def test_filtered_selection_is_repaired_after_filtering_without_changing_history():
    """The current-call recipe: the filter separates a result from its turn."""
    agent = Agent(llm=object())
    earlier = response("a")
    agent.event_manager.add(earlier)
    result = execution("a", response=earlier, result=ToolResult(tool_call_id="a", content="x"))
    pending = execution("pending")
    agent.event_manager.add(result)
    agent.event_manager.add(pending)
    before = list(agent.event_manager.values())
    snapshot = [event.model_dump_json() for event in before]

    current = {result.id, pending.id}  # stands in for related_call_ids filtering
    selection = [e for e in agent.event_manager.values() if e.id in current and is_model_visible(e)]
    with pytest.raises(UnsupportedContextLayout):
        render(selection)
    messages = render(replayable(selection))

    assert [m.tool_call_id for m in messages if m.tool_call_id] == ["pending"]
    assert "execution outcome is unknown" in messages[-1].content
    assert list(agent.event_manager.values()) == before
    assert [event.model_dump_json() for event in before] == snapshot
    assert pending.result is None
