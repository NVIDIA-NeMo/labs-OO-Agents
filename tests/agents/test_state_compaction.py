# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored-state summaries are independent of prompt projection."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nooa import Agent, Block
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.events import Message
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage, ToolCall


def setup(preserve=1):
    agent = Agent(llm=FakeLLMClient())
    for i in range(4):
        agent.event_manager.add(Message(content=f"STORED FACT {i}"))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=preserve)
    )
    ctx = LLMCallContext(
        agent=agent,
        runtime=agent.runtime,
        client=agent.llm,
        messages=[{"role": "user", "content": "PARENT PROJECTION ONLY"}],
        params={"tools": [object()], "prompt_cache_key": "parent-shard"},
    )
    return agent, summarizer, ctx


async def dispatch(ctx, tokens=1000):
    ctx.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=tokens))
    return ctx


async def trigger(agent, ctx):
    return await agent.event_manager.run_middleware("llm_call", ctx, dispatch)


async def test_source_is_independent_of_view_and_frozen_before_dispatch():
    agent, summarizer, ctx = setup()
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def summarize(text, target):
        seen.append(text)
        entered.set()
        await release.wait()
        return "summary"

    summarizer.summarize = AsyncMock(side_effect=summarize)

    async def parent(request):
        agent.event_manager.add(Message(content="NEW WORK"))
        return await dispatch(request)

    try:
        await agent.event_manager.run_middleware("llm_call", ctx, parent)
        await asyncio.wait_for(entered.wait(), 2)
        assert all(f"STORED FACT {i}" in seen[0] for i in range(3))
        assert "STORED FACT 3" not in seen[0]
        assert "PARENT PROJECTION ONLY" not in seen[0] and "NEW WORK" not in seen[0]
        assert agent.event_manager.keys() == ["1", "2", "3", "4", "5"]
        release.set()
        await summarizer._pending_task
        assert agent.event_manager.keys() == ["1", "2", "3", "4", "5"]
        summarizer._apply_pending_summary()
        assert agent.event_manager.keys() == ["1..3", "4", "5"]
        assert agent.events["1"].content == "STORED FACT 0"
    finally:
        release.set()
        await agent.aclose()


@pytest.mark.parametrize("change", ["edit", "collapse", "during_dispatch"])
async def test_stale_source_never_replaces_history(change):
    agent, summarizer, ctx = setup()
    summarizer.summarize = AsyncMock(return_value="summary")

    async def parent(request):
        if change == "during_dispatch":
            agent.event_manager["1"].content = "changed"
        return await dispatch(request)

    await agent.event_manager.run_middleware("llm_call", ctx, parent)
    await summarizer._pending_task
    if change == "edit":
        agent.event_manager["1"].content = "changed"
    elif change == "collapse":
        agent.event_manager.collapse("2", "3", "other")
    before = agent.event_manager.keys()
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == before
    await agent.aclose()


async def test_mutated_output_hidden_by_bounded_serializer_invalidates_summary():
    from nooa.context_blocks.events import ResultStatus
    from nooa.events import PythonOutput

    class Payload:
        values: list[int]

        def __init__(self, values):
            self.values = values

    agent, summarizer, ctx = setup(preserve=0)
    payload = Payload(list(range(200)))
    output = PythonOutput(
        tool_call_id="python",
        execution_status=ResultStatus.COMPLETE,
        execution_count=1,
        value=payload,
    )
    agent.event_manager.clear()
    agent.event_manager.add(output)
    summarizer.summarize = AsyncMock(return_value="summary")
    await trigger(agent, ctx)
    await summarizer._pending_task
    serialized = output.model_dump_json()
    payload.values[100] = 987654
    assert output.model_dump_json() == serialized
    summarizer._apply_pending_summary()
    assert agent.event_manager.values() == [output]
    await agent.aclose()


@pytest.mark.parametrize("kind", ["task", "output"])
async def test_image_bearing_events_remain_active(kind):
    from nooa.context_blocks.events import ResultStatus
    from nooa.events import PythonOutput, Task

    agent, summarizer, ctx = setup(preserve=0)
    images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    event = (
        Task(prompt="image", images=images)
        if kind == "task"
        else PythonOutput(
            tool_call_id="python",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            images=images,
        )
    )
    agent.event_manager.clear()
    agent.event_manager.add(event)
    agent.event_manager.add(Message(content="independent text"))
    source, text = summarizer._snapshot_source(ctx.client)
    assert [tag for tag, _, _ in source] == ["2"]
    assert "independent text" in text
    summarizer.summarize = AsyncMock(return_value="summary")
    await trigger(agent, ctx)
    await summarizer._pending_task
    summarizer._apply_pending_summary()
    assert agent.event_manager["1"] is event
    await agent.aclose()


async def test_close_during_parent_dispatch_cannot_schedule_background_work():
    agent, summarizer, ctx = setup()
    summarizer.summarize = AsyncMock(return_value="summary")

    async def parent(request):
        await agent.aclose()
        return await dispatch(request)

    await agent.event_manager.run_middleware("llm_call", ctx, parent)
    assert summarizer._pending_task is None
    summarizer.summarize.assert_not_awaited()


async def test_summary_overflow_cannot_retry_without_its_source_task(caplog):
    agent, summarizer, ctx = setup()
    ctx.client.acall = AsyncMock(
        side_effect=[
            RuntimeError(
                "This model's maximum context length is 10000 tokens. "
                "However, your request has 8000 input tokens."
            ),
            LLMResponse(content='{"value":"ungrounded summary"}'),
        ]
    )
    # Force the runtime's generic recovery to archive the summary's Task.
    summarizer.runtime._archive_on_context_error = lambda *args, **kwargs: (
        summarizer.event_manager.clear()
    )
    await trigger(agent, ctx)
    await summarizer._pending_task
    summarizer._apply_pending_summary()
    assert ctx.client.acall.await_count == 1
    assert agent.event_manager.keys() == ["1", "2", "3", "4"]
    assert summarizer._pending_summary is None
    assert "Summary source task is no longer active" in caplog.text
    await agent.aclose()


def test_bounded_prefix_never_omits_a_collapsed_event():
    agent, summarizer, ctx = setup(preserve=0)
    first = agent.event_manager["1"]
    text = summarizer._event_markdown("1", first)
    # summary input budget is 70% of the effective client window.
    cost = ctx.client.count_tokens(repr(text)) + ctx.client.count_tokens("\n\n")
    ctx.client._context_window = int(cost / 0.7) + 2
    source, rendered = summarizer._snapshot_source(ctx.client)
    assert [tag for tag, _, _ in source] == ["1"]
    assert "STORED FACT 0" in rendered and "STORED FACT 1" not in rendered
    ctx.client._context_window = 1
    assert summarizer._snapshot_source(ctx.client) == ((), "")
    summarizer._uninstall()


def add_batch(agent, *, complete=True):
    response = LLMResponse(
        tool_calls=[
            ToolCall(id="a", name="execute_python", arguments="{}"),
            ToolCall(id="b", name="execute_python", arguments="{}"),
        ],
        finish_reason="tool_calls",
    )
    agent.event_manager.add(response)
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="a",
            llm_response_id=response.id,
            name="execute_python",
            arguments={},
            result=ToolResult(tool_call_id="a", content="a") if complete else None,
        )
    )
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="b",
            llm_response_id=response.id,
            name="execute_python",
            arguments={},
            result=ToolResult(tool_call_id="b", content="b") if complete else None,
        )
    )
    return response


@pytest.mark.parametrize("cut", ["recent", "budget", "unfinished", "nested", "running"])
def test_compaction_keeps_whole_tool_batches(cut):
    agent, summarizer, ctx = setup(preserve=0)
    add_batch(agent, complete=cut not in {"unfinished", "nested"})
    if cut == "recent":
        summarizer.config = summarizer.config.model_copy(update={"preserve_recent": 1})
    elif cut == "budget":
        parts = [summarizer._event_markdown(tag, e) for tag, e in agent.event_manager.items()[:5]]
        cost = sum(
            ctx.client.count_tokens(repr(part)) + ctx.client.count_tokens("\n\n") for part in parts
        )
        ctx.client._context_window = int(cost / 0.7) + 2
    elif cut == "nested":
        agent.event_manager.add(Message(content="child task", metadata={"call_id": "child"}))
    elif cut == "running":
        from nooa.context_blocks.events import ResultStatus

        agent.event_manager["6"].result.result_status = ResultStatus.RUNNING
    source, _ = summarizer._snapshot_source(ctx.client)
    assert [tag for tag, _, _ in source] == ["1", "2", "3", "4"]
    summarizer._uninstall()


@pytest.mark.parametrize("bad", ["", "   ", None, RuntimeError("failed")])
async def test_bad_summary_preserves_history(bad):
    agent, summarizer, ctx = setup()
    summarizer.summarize = AsyncMock(
        side_effect=bad if isinstance(bad, Exception) else None,
        return_value=bad if not isinstance(bad, Exception) else None,
    )
    await trigger(agent, ctx)
    await summarizer._pending_task
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1", "2", "3", "4"]
    await agent.aclose()


def test_overlapping_batches_recheck_the_earlier_group_after_shrinking():
    first = LLMResponse(
        tool_calls=[ToolCall(id="a", name="execute_python", arguments="{}")],
        finish_reason="tool_calls",
    )
    second = LLMResponse(
        tool_calls=[ToolCall(id="b", name="execute_python", arguments="{}")],
        finish_reason="tool_calls",
    )
    results = [
        ToolCallEvent(
            name="execute_python",
            arguments={},
            tool_call_id=call_id,
            llm_response_id=response.id,
            result=ToolResult(tool_call_id=call_id, content="done"),
        )
        for call_id, response in [("a", first), ("b", second)]
    ]
    events = [first, second, *results]
    assert TokenBudgetSummarizer._history_units(events) == [(0, 4, True)]


async def test_hidden_records_are_neither_summarized_nor_archived():
    from nooa.events import DebugTrace

    agent = Agent(llm=FakeLLMClient())
    agent.event_manager.add(DebugTrace(content="PRIVATE DIAGNOSTICS"))
    agent.event_manager.add(Message(content="visible one"))
    agent.event_manager.add(Message(content="visible two"))
    agent.event_manager.add(DebugTrace(content="PRIVATE BARRIER"))
    agent.event_manager.add(Message(content="recent"))
    summarizer = TokenBudgetSummarizer.install(
        agent,
        config=TokenBudgetConfig(
            max_tokens=1,
            preserve_recent=1,
        ),
    )
    summarizer.summarize = AsyncMock(return_value="summary")
    ctx = LLMCallContext(client=agent.llm, runtime=agent.runtime, messages=[])
    await trigger(agent, ctx)
    await summarizer._pending_task
    text = summarizer.summarize.call_args.args[0]
    assert "visible one" in text and "visible two" in text
    assert "PRIVATE" not in text
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1", "2..3", "4", "5"]
    await agent.aclose()


def test_existing_summary_before_hidden_barrier_does_not_starve_later_history():
    from nooa.events import DebugTrace

    agent, summarizer, ctx = setup(preserve=0)
    agent.event_manager.collapse("1", "4", "prior summary")
    agent.event_manager.add(DebugTrace(content="private"))
    agent.event_manager.add(Message(content="new fact one"))
    agent.event_manager.add(Message(content="new fact two"))
    source, text = summarizer._snapshot_source(ctx.client)
    assert [tag for tag, _, _ in source] == ["6", "7"]
    assert "new fact one" in text and "prior summary" not in text
    summarizer._uninstall()


def test_summary_text_uses_public_content_not_persistence_payloads():
    from nooa.unifiedllm import AssistantReasoning, AssistantText

    agent, summarizer, ctx = setup(preserve=0)
    agent.event_manager.clear()
    response = LLMResponse(
        parts=(
            AssistantReasoning(
                text="public reasoning", native={"encrypted_content": "OPAQUE STATE"}
            ),
            AssistantText(text="public answer"),
        ),
        metadata={"private": "PRIVATE METADATA"},
    )
    agent.event_manager.add(response)
    source, text = summarizer._snapshot_source(ctx.client)
    assert "OPAQUE STATE" in source[0][2] and "PRIVATE METADATA" in source[0][2]
    assert "public answer" in text and "public reasoning" in text
    assert "OPAQUE STATE" not in text and "PRIVATE METADATA" not in text
    summarizer._uninstall()


def test_hidden_record_inside_tool_batch_retains_the_whole_group():
    from nooa.events import DebugTrace

    agent, summarizer, ctx = setup(preserve=0)
    agent.event_manager.clear()
    response = LLMResponse(
        tool_calls=[ToolCall(id="a", name="execute_python", arguments="{}")],
        finish_reason="tool_calls",
    )
    agent.event_manager.add(response)
    agent.event_manager.add(DebugTrace(content="private"))
    agent.event_manager.add(
        ToolCallEvent(
            name="execute_python",
            arguments={},
            tool_call_id="a",
            llm_response_id=response.id,
            result=ToolResult(tool_call_id="a", content="done"),
        )
    )
    agent.event_manager.add(Message(content="later public record"))
    source, text = summarizer._snapshot_source(ctx.client)
    assert [tag for tag, _, _ in source] == ["4"]
    assert "later public record" in text and "private" not in text
    summarizer._uninstall()


@pytest.mark.parametrize("finish", ["length", "error"])
async def test_truncated_valid_json_summary_preserves_history(finish):
    agent, summarizer, ctx = setup()
    agent.llm.acall = AsyncMock(
        return_value=LLMResponse(
            content='{"value":"apparently valid summary"}',
            finish_reason=finish,
        )
    )
    await trigger(agent, ctx)
    await summarizer._pending_task
    assert summarizer._pending_summary is None
    assert agent.llm.acall.await_count == 1
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1", "2", "3", "4"]
    await agent.aclose()


async def test_dedicated_request_uses_effective_client_and_not_parent_tools():
    agent, summarizer, ctx = setup()
    effective = FakeLLMClient(scripted_responses=[LLMResponse(content='{"value":"summary"}')])
    ctx = ctx.model_copy(update={"client": effective})
    await trigger(agent, ctx)
    await summarizer._pending_task
    assert summarizer._pending_summary == "summary"
    assert agent.llm.call_count == 0
    call = effective.calls[0]
    text = str(call.messages)
    assert "STORED FACT 0" in text and "STORED FACT 2" in text
    assert "PARENT PROJECTION ONLY" not in text and "STORED FACT 3" not in text
    assert not call.tools
    assert call.output_model is not None
    assert call.kwargs.get("prompt_cache_key") != "parent-shard"
    await agent.aclose()


async def test_custom_view_cannot_hide_compactor_source():
    from nooa import strategy
    from nooa.strategies import PredictStrategy

    class View:
        async def assemble(self, owner, call):
            yield Block(key="projection", content="PARENT PROJECTION ONLY")

    client = FakeLLMClient(
        scripted_responses=[
            LLMResponse(content='{"value":"parent"}', usage=LLMUsage(input_tokens=1000)),
            LLMResponse(content='{"value":"summary"}'),
        ]
    )

    class Parent(Agent, llm=client, context_view=View()):
        @strategy(PredictStrategy())
        async def run(self) -> str: ...

    agent = Parent()
    for i in range(3):
        agent.event_manager.add(Message(content=f"STORED FACT {i}"))
    summarizer = TokenBudgetSummarizer.install(
        agent,
        config=TokenBudgetConfig(
            max_tokens=100,
            preserve_recent=1,
        ),
    )
    assert await agent.run() == "parent"
    await summarizer._pending_task
    parent, summary = client.calls
    assert "STORED FACT" not in str(parent.messages)
    assert "STORED FACT 0" in str(summary.messages)
    assert "PARENT PROJECTION ONLY" not in str(summary.messages)
    await agent.aclose()


@pytest.mark.parametrize("outcomes", [["", ""], ["", "summary", ""]])
async def test_consecutive_failures_disable_compaction_and_success_resets(outcomes):
    agent, summarizer, ctx = setup()
    summarizer.summarize = AsyncMock(side_effect=outcomes)
    for _ in outcomes:
        agent.event_manager.add(Message(content="new work"))
        await trigger(agent, ctx)
        await summarizer._pending_task
        summarizer._apply_pending_summary()
    assert (summarizer._unsub_llm is None) is (outcomes == ["", ""])
    await agent.aclose()


@pytest.mark.parametrize(
    "tokens,preserve,expected",
    [
        (1000, 1, True),
        (100, 1, False),
        (0, 1, False),
        (1000, 4, False),
        (1000, 0, True),
    ],
)
async def test_trigger_uses_actual_usage(tokens, preserve, expected):
    agent, summarizer, ctx = setup(preserve)
    summarizer.summarize = AsyncMock(return_value="summary")

    async def parent(request):
        return await dispatch(request, tokens)

    await agent.event_manager.run_middleware("llm_call", ctx, parent)
    assert (summarizer._pending_task is not None) is expected
    if expected:
        await summarizer._pending_task
    await agent.aclose()


async def test_aclose_cancels_and_awaits_background_work():
    agent, summarizer, ctx = setup()
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def wait(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    summarizer.summarize = AsyncMock(side_effect=wait)
    await trigger(agent, ctx)
    await asyncio.wait_for(entered.wait(), 2)
    await agent.aclose()
    assert cancelled.is_set()
    assert summarizer._pending_task is None
    assert not agent.event_manager._middleware["llm_call"]


async def test_snapshot_failure_leaves_parent_and_history_intact():
    agent, summarizer, ctx = setup()
    summarizer._snapshot_source = lambda _: (_ for _ in ()).throw(ValueError("bad source"))
    result = await trigger(agent, ctx)
    assert result.response.content == "parent"
    assert summarizer._pending_task is None
    assert agent.event_manager.keys() == ["1", "2", "3", "4"]
    await agent.aclose()


async def test_close_callbacks_are_awaited_once_in_reverse_order(caplog):
    agent = Agent(llm=FakeLLMClient())
    calls = []

    async def first():
        await asyncio.sleep(0)
        calls.append("first")

    async def broken():
        calls.append("broken")
        raise ValueError("cleanup failed")

    removed = AsyncMock()
    agent.event_manager.on_close(first)
    unsubscribe = agent.event_manager.on_close(removed)
    agent.event_manager.on_close(broken)
    unsubscribe()
    await agent.aclose()
    await agent.aclose()
    assert calls == ["broken", "first"]
    removed.assert_not_awaited()
    assert "cleanup failed" in caplog.text
