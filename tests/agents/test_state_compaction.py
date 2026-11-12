# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored-state summaries are independent of prompt projection."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from nooa import Agent, Block, strategy
from nooa.agents import TokenBudgetSummarizer
from nooa.agents.summarization import summary_fork_active
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.context_blocks import ScopedContext
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.default_context_view import PredictView
from nooa.events import Message
from nooa.runtime.middleware import LLMCallContext
from nooa.strategies import PredictStrategy
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


def carrying_source(agent, ctx, reply="summary"):
    """A parent request that contains every stored event, so the summary can reuse it."""
    return ctx.model_copy(
        update={
            "event_ids": frozenset(event.id for event in agent.event_manager.values()),
            "params": {},
            "client": FakeLLMClient([LLMResponse(content=reply)]),
        }
    )


async def dispatch(ctx, tokens=1000):
    ctx.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=tokens))
    return ctx


async def trigger(agent, ctx):
    return await agent.event_manager.run_middleware("llm_call", ctx, dispatch)


async def test_under_budget_does_not_prepare_summary_source(monkeypatch):
    agent, summarizer, ctx = setup()

    def unexpected(*args, **kwargs):
        pytest.fail("Summary preparation ran below the token threshold")

    monkeypatch.setattr(summarizer, "_select_source", unexpected)
    monkeypatch.setattr(summarizer, "_render_range_to_markdown", unexpected)
    monkeypatch.setattr(Message, "model_dump_json", unexpected)
    monkeypatch.setattr(ctx.client, "count_tokens", unexpected)
    result = await agent.event_manager.run_middleware(
        "llm_call", ctx, lambda request: dispatch(request, tokens=1)
    )
    assert result.response.content == "parent"
    assert summarizer._pending_task is None
    await agent.aclose()


async def test_cached_summary_does_not_prepare_standalone_prompt(monkeypatch):
    agent, summarizer, ctx = setup()
    ctx = carrying_source(agent, ctx)

    def unexpected(*args, **kwargs):
        pytest.fail("Standalone prompt preparation ran for a cached summary")

    monkeypatch.setattr(summarizer, "_render_range_to_markdown", unexpected)
    monkeypatch.setattr(ctx.client, "count_tokens", unexpected)
    await trigger(agent, ctx)
    assert summarizer._pending_summary == "summary"
    await agent.aclose()


class HidesMessages(PredictView):
    """Leaves stored messages out of the request."""

    async def assemble(self, owner, call):
        async for item in super().assemble(owner, call):
            if not isinstance(item, Message):
                yield item


class AddsBlock(PredictView):
    """A customized default view that still carries the full history."""

    async def assemble(self, owner, call):
        async for item in super().assemble(owner, call):
            yield item
        yield Block(key="extra", content="EXTRA BLOCK")


@pytest.mark.parametrize(
    ("view", "reuses"),
    [(PredictView, True), (AddsBlock, True), (HidesMessages, False)],
    ids=["default", "subclass", "hides-source"],
)
async def test_real_parent_reuses_request_only_when_it_carries_the_source(view, reuses):
    class RecordingClient(FakeLLMClient):
        def __init__(self):
            super().__init__(
                [
                    LLMResponse(content='{"value":"parent"}', usage=LLMUsage(input_tokens=1000)),
                    LLMResponse(content='{"value":"summary"}'),
                ]
            )
            self.requests = []

        async def acall(self, messages, **kwargs):
            self.requests.append(json.dumps(messages, default=lambda value: value.model_dump()))
            return await super().acall(messages, **kwargs)

    class Parent(Agent):
        @strategy(PredictStrategy(), context={"parent_only": "PARENT_DECORATOR_MARKER"})
        async def run(self) -> str:
            """Return parent."""
            ...

    client = RecordingClient()
    agent = Parent(llm=client)
    agent.event_manager.add(Message(content="STORED_SOURCE_FACT"))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=1, preserve_recent=1)
    )
    try:
        with ScopedContext(context={"system_prompt": "PARENT_SCOPED_MARKER"}):
            assert await agent.run(context_view=view()) == "parent"
            await summarizer._pending_task
        assert summarizer._pending_summary == ('{"value":"summary"}' if reuses else "summary")
        assert len(client.requests) == 2
        parent, summary = client.requests
        for marker in ("PARENT_DECORATOR_MARKER", "PARENT_SCOPED_MARKER"):
            assert marker in parent
            assert (marker in summary) is reuses
        if reuses:
            assert json.loads(summary)[:-1] == json.loads(parent)
        assert "STORED_SOURCE_FACT" in summary
    finally:
        await agent.aclose()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_summary_activity_is_task_local_and_resets(outcome):
    agent, summarizer, ctx = setup()
    entered, release = asyncio.Event(), asyncio.Event()
    after = []

    async def summarize(text, target):
        assert summary_fork_active()
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("summary transport failed")
        return "summary"

    async def run():
        try:
            await summarizer._run_state_summary("facts", ctx.client)
        finally:
            after.append(summary_fork_active())

    summarizer.summarize = AsyncMock(side_effect=summarize)
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not summary_fork_active(), "Concurrent parent work is not summarization"
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await task
        assert after == [False]
    finally:
        release.set()
        await agent.aclose()


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
        parent_call = asyncio.create_task(
            agent.event_manager.run_middleware("llm_call", ctx, parent)
        )
        await asyncio.wait_for(entered.wait(), 2)
        assert not parent_call.done()
        assert all(f"STORED FACT {i}" in seen[0] for i in range(3))
        assert "STORED FACT 3" not in seen[0]
        assert "PARENT PROJECTION ONLY" not in seen[0] and "NEW WORK" not in seen[0]
        assert agent.event_manager.keys() == ["1", "2", "3", "4", "5"]
        release.set()
        await parent_call
        assert agent.event_manager.keys() == ["1", "2", "3", "4", "5"]
        summarizer._apply_pending_summary()
        assert agent.event_manager.keys() == ["1..3", "4", "5"]
        assert agent.events["1"].content == "STORED FACT 0"
    finally:
        release.set()
        await agent.aclose()


async def test_collapsed_source_never_replaces_history():
    agent, summarizer, ctx = setup()
    summarizer.summarize = AsyncMock(return_value="summary")
    await trigger(agent, ctx)
    await summarizer._pending_task
    agent.event_manager.collapse("2", "3", "other")
    before = agent.event_manager.keys()
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == before
    await agent.aclose()


@pytest.mark.parametrize("reuse_prefix", [False, True])
@pytest.mark.parametrize(
    "make_value",
    [lambda: {"a": 1}.keys(), lambda: (i for i in range(3)), object],
    ids=["not-deep-copyable", "generator", "address-repr"],
)
async def test_python_values_never_block_or_invalidate_compaction(reuse_prefix, make_value):
    from nooa.context_blocks.events import ResultStatus
    from nooa.events import PythonOutput

    agent, summarizer, ctx = setup(preserve=0)
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="python",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            value=make_value(),
        )
    )
    if reuse_prefix:
        ctx = carrying_source(agent, ctx)
    summarizer.summarize = AsyncMock(return_value="summary")
    await trigger(agent, ctx)
    await summarizer._pending_task
    assert summarizer._pending_summary == "summary"
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1..5"]
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


async def test_interleaved_hidden_records_collapse_into_one_summary():
    from nooa.events import DebugTrace

    agent = Agent(llm=FakeLLMClient())
    for turn in range(4):
        agent.event_manager.add(DebugTrace(content=f"session record {turn}"))
        agent.event_manager.add(Message(content=f"turn {turn}"))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=1, preserve_recent=1)
    )
    summarizer.summarize = AsyncMock(return_value="summary")
    await trigger(agent, LLMCallContext(client=agent.llm, runtime=agent.runtime, messages=[]))
    await summarizer._pending_task
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1..7", "8"]
    await agent.aclose()


async def test_adjacent_summaries_merge():
    agent, summarizer, ctx = setup(preserve=1)
    agent.event_manager.collapse("1", "2", "first summary")
    agent.event_manager.collapse("3", "3", "second summary")
    summarizer.summarize = AsyncMock(return_value="merged")
    await trigger(agent, ctx)
    await summarizer._pending_task
    text = summarizer.summarize.call_args.args[0]
    assert "first summary" in text and "second summary" in text
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1..3", "4"]
    await agent.aclose()


async def test_hidden_records_are_archived_with_their_range_but_not_summarized():
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
    assert agent.event_manager.keys() == ["1..4", "5"]
    await agent.aclose()


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

    class Parent(Agent, llm=client):
        @strategy(PredictStrategy(), context_view=View())
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
    parent_call = asyncio.create_task(trigger(agent, ctx))
    await asyncio.wait_for(entered.wait(), 2)
    await agent.aclose()
    with pytest.raises(asyncio.CancelledError):
        await parent_call
    assert cancelled.is_set()
    assert summarizer._pending_task is None
    assert not agent.event_manager._middleware["llm_call"]


async def test_snapshot_failure_leaves_parent_and_history_intact():
    agent, summarizer, ctx = setup()
    summarizer._render_range_to_markdown = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("bad source")
    )
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
