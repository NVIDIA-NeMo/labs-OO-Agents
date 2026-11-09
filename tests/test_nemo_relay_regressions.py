# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Relay regressions for upgrade review findings; no model calls."""
# ruff: noqa: E402

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Any

import pytest

relay = pytest.importorskip("nemo_relay")

from nooa import Agent
from nooa.events import ExecutionResult
from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse
from nooa.nemo_relay_middleware import (
    nemo_relay_agent_call_middleware,
    nemo_relay_llm_middleware,
    nemo_relay_scope,
    nemo_relay_tool_middleware,
)
from nooa.runtime.middleware import AgentCallContext, ExecutePythonContext, LLMCallContext
from nooa.unifiedllm import FakeLLMClient


class ReviewAgent(Agent, llm=FakeLLMClient()):
    pass


@dataclass
class CollectionResult:
    value: Any


@pytest.fixture(autouse=True)
async def flush_events_before_loop_closes():
    yield
    await relay.subscribers.flush_async()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [{1, 2}, frozenset({1, 2}), range(3), deque([1, 2])])
async def test_dataclass_collection_does_not_fail_after_execution(value):
    result = ExecutionResult(returned_value=CollectionResult(value))
    events = []
    relay.subscribers.register("review-collections", events.append)
    try:
        async with nemo_relay_scope(ReviewAgent(), "collections"):

            async def execute(ctx):
                ctx.result = result
                return ctx

            ctx = await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
            assert ctx.result is result
        await relay.subscribers.flush_async()
        data = next(
            e.data
            for e in events
            if getattr(e, "category", None) == "tool" and e.scope_category == "end"
        )
        assert data["schema"] == "nemo.relay.tool_result@1"
        assert "1" in data["content"] and "2" in data["content"]
    finally:
        relay.subscribers.deregister("review-collections")


@pytest.mark.asyncio
@pytest.mark.parametrize("body_error", [False, True])
async def test_scope_cleanup_does_not_replace_method_outcome(monkeypatch, body_error):
    original = relay.scope.pop

    def bad_pop(handle, **kwargs):
        original(handle, **kwargs)
        raise RuntimeError("cleanup failure")

    async with nemo_relay_scope(ReviewAgent(), "cleanup"):
        monkeypatch.setattr(relay.scope, "pop", bad_pop)

        async def body(ctx):
            if body_error:
                raise ValueError("original error")
            ctx.result = 42
            return ctx

        try:
            ctx = AgentCallContext(method_name="body")
            if body_error:
                with pytest.raises(ValueError, match="original error"):
                    await nemo_relay_agent_call_middleware(ctx, body)
            else:
                assert (await nemo_relay_agent_call_middleware(ctx, body)).result == 42
        finally:
            monkeypatch.setattr(relay.scope, "pop", original)


@pytest.mark.asyncio
async def test_legacy_sanitizers_redact_before_projection():
    events = []
    relay.subscribers.register("review-redaction", events.append)

    def sanitize_llm(payload, context):
        message = payload.get("message", {})
        for key in ("content", "reasoning_content"):
            if key in message:
                message[key] = message[key].replace("SECRET", "REDACTED")
        return payload

    def sanitize_tool(name, value):
        return value.replace("SECRET", "REDACTED") if isinstance(value, str) else value

    relay.guardrails.register_llm_sanitize_response("review-legacy", 1, sanitize_llm)
    relay.guardrails.register_tool_sanitize_response("review-legacy", 1, sanitize_tool)
    try:
        async with nemo_relay_scope(ReviewAgent(), "redaction"):

            async def model(ctx):
                ctx.response = LLMResponse(
                    parts=(
                        AssistantText(text="SECRET answer"),
                        AssistantReasoning(text="SECRET rationale"),
                    )
                )
                return ctx

            async def execute(ctx):
                ctx.result = ExecutionResult(returned_value="SECRET tool")
                return ctx

            await nemo_relay_llm_middleware(LLMCallContext(messages=[]), model)
            await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        await relay.subscribers.flush_async()
        outputs = [e.data for e in events if getattr(e, "scope_category", None) == "end"]
        assert "SECRET" not in json.dumps(outputs)
        llm = next(
            e.data
            for e in events
            if getattr(e, "category", None) == "llm" and e.scope_category == "end"
        )
        assert llm["assistant_message"]["content"] == "REDACTED answer"
        assert llm["reasoning"] == "REDACTED rationale"
        assert "reasoning_content" not in llm["assistant_message"]
    finally:
        relay.guardrails.deregister_llm_sanitize_response("review-legacy")
        relay.guardrails.deregister_tool_sanitize_response("review-legacy")
        relay.subscribers.deregister("review-redaction")


@pytest.mark.asyncio
async def test_legacy_event_sanitizer_redacts_before_projection():
    events = []
    relay.subscribers.register("review-event-redaction", events.append)
    try:
        async with nemo_relay_scope(ReviewAgent(), "event-redaction") as root:

            def sanitize(event, fields):
                data = fields["data"]
                if event.category == "llm" and isinstance(data, dict):
                    data["message"]["content"] = data["message"]["content"].replace(
                        "SECRET", "REDACTED"
                    )
                elif event.category == "tool" and isinstance(data, str):
                    data = data.replace("SECRET", "REDACTED")
                return {**fields, "data": data, "metadata": None}

            relay.scope_local.register_scope_sanitize_end(root, "legacy-event", 1, sanitize)

            async def model(ctx):
                ctx.response = LLMResponse(content="SECRET answer")
                return ctx

            async def execute(ctx):
                ctx.result = ExecutionResult(returned_value="SECRET tool")
                return ctx

            await nemo_relay_llm_middleware(LLMCallContext(messages=[]), model)
            await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        await relay.subscribers.flush_async()
        outputs = [e.data for e in events if getattr(e, "scope_category", None) == "end"]
        assert "SECRET" not in json.dumps(outputs)
        assert "REDACTED answer" in json.dumps(outputs)
        assert "REDACTED tool" in json.dumps(outputs)
        assert any(isinstance(value, dict) and "assistant_message" in value for value in outputs)
        assert any(
            isinstance(value, dict) and value.get("schema") == "nemo.relay.tool_result@1"
            for value in outputs
        )
        assert all(
            e.metadata is None
            for e in events
            if getattr(e, "category", None) in {"llm", "tool"} and e.scope_category == "end"
        )
    finally:
        relay.subscribers.deregister("review-event-redaction")


@pytest.mark.asyncio
async def test_nested_method_in_managed_callback_keeps_scope_local_policy():
    seen = []
    async with nemo_relay_scope(ReviewAgent(), "nested-policy") as root:
        relay.scope_local.register_tool_conditional_execution(
            root, "review-local", 1, lambda name, args: seen.append(name)
        )

        async def inner(ctx):
            await relay.tools.execute("inner", {}, lambda _: relay.ToolExecutionResult("ok"))
            return ctx

        async def execute(ctx):
            await nemo_relay_agent_call_middleware(AgentCallContext(method_name="inner"), inner)
            ctx.result = ExecutionResult(returned_value="ok")
            return ctx

        await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
    assert seen == ["execute_python", "inner"]


@pytest.mark.asyncio
async def test_concurrent_roots_do_not_share_mutable_stack():
    with relay.use_scope_stack(relay.create_scope_stack()):
        with relay.scope.scope("ambient", relay.ScopeType.Agent) as ambient:
            entered = asyncio.Event()
            release = asyncio.Event()

            async def first():
                async with nemo_relay_scope(ReviewAgent(), "first"):
                    entered.set()
                    await release.wait()

            async def second():
                await entered.wait()
                async with nemo_relay_scope(ReviewAgent(), "second"):
                    release.set()
                    await asyncio.sleep(0.01)

            await asyncio.gather(first(), second())
            assert relay.scope.get_handle().uuid == ambient.uuid


@pytest.mark.asyncio
async def test_method_exception_text_is_not_copied_to_scope_metadata():
    events = []
    relay.subscribers.register("review-errors", events.append)
    try:
        async with nemo_relay_scope(ReviewAgent(), "exceptions"):

            async def body(ctx):
                raise ValueError("SECRET_EXCEPTION")

            with pytest.raises(ValueError, match="SECRET_EXCEPTION"):
                await nemo_relay_agent_call_middleware(AgentCallContext(method_name="body"), body)
        await relay.subscribers.flush_async()
        assert "SECRET_EXCEPTION" not in json.dumps([e.metadata for e in events])
    finally:
        relay.subscribers.deregister("review-errors")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "x" * 20000, object()])
async def test_tool_display_is_bounded_and_does_not_print_pickle(value):
    events = []
    relay.subscribers.register("review-display", events.append)
    try:
        async with nemo_relay_scope(ReviewAgent(), "display"):

            async def execute(ctx):
                ctx.result = ExecutionResult(returned_value=value)
                return ctx

            await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        await relay.subscribers.flush_async()
        data = next(
            e.data
            for e in events
            if getattr(e, "category", None) == "tool" and e.scope_category == "end"
        )
        if value is None:
            assert data is None
        else:
            assert len(data["content"]) < 4200
            if isinstance(value, str):
                assert data["content"].endswith("… [truncated]")
                assert data["result"] == value
            else:
                assert data["content"] == "[opaque Python value]"
    finally:
        relay.subscribers.deregister("review-display")


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_response_sanitizer_omission_is_not_repopulated(raises):
    events = []
    relay.subscribers.register("review-omission", events.append)

    def omit(payload, context):
        if raises:
            raise ValueError("sanitizer failed")
        return None

    relay.guardrails.register_llm_sanitize_response("review-omit", 1, omit)
    try:
        async with nemo_relay_scope(ReviewAgent(), "omission"):
            response = LLMResponse(content="SECRET")

            async def model(ctx):
                ctx.response = response
                return ctx

            assert (
                await nemo_relay_llm_middleware(LLMCallContext(messages=[]), model)
            ).response is response
        await relay.subscribers.flush_async()
        assert all(
            e.data is None
            for e in events
            if getattr(e, "category", None) == "llm" and e.scope_category == "end"
        )
    finally:
        relay.guardrails.deregister_llm_sanitize_response("review-omit")
        relay.subscribers.deregister("review-omission")


@pytest.mark.asyncio
async def test_empty_tool_id_is_absent_in_native_execution_context():
    ids = []
    async with nemo_relay_scope(ReviewAgent(), "empty-id") as root:

        async def intercept(context, next_call):
            ids.append(context.tool_call_id)
            result = await next_call(context.args)
            return relay.ToolExecutionInterceptOutcome(result.result)

        relay.scope_local.register_tool_execution(root, "review-id", 1, intercept)

        async def execute(ctx):
            ctx.result = ExecutionResult(returned_value=42)
            return ctx

        await nemo_relay_tool_middleware(
            ExecutePythonContext(code="pass", params={"tool_call_id": ""}), execute
        )
    assert ids == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "llm"])
async def test_native_short_circuit_is_not_reported_as_guardrail_rejection(kind):
    async with nemo_relay_scope(ReviewAgent(), "short-circuit") as root:

        async def execute(ctx):
            pytest.fail("A replacement intercept must skip the NOOA callback")

        if kind == "tool":
            relay.scope_local.register_tool_execution(
                root,
                "review-cache",
                1,
                lambda context, next_call: relay.ToolExecutionInterceptOutcome("cached"),
            )
            call = nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        else:
            relay.scope_local.register_llm_execution(
                root,
                "review-cache",
                1,
                lambda name, request, context, next_call: {"message": {"content": "cached"}},
            )
            call = nemo_relay_llm_middleware(LLMCallContext(messages=[]), execute)
        with pytest.raises(RuntimeError, match="short-circuited") as error:
            await call
        assert "guardrail" not in str(error.value).lower()


@pytest.mark.asyncio
async def test_overlapping_scope_installations_do_not_duplicate_tool_calls():
    agent = ReviewAgent()
    events = []
    relay.subscribers.register("review-installation", events.append)
    try:
        async with nemo_relay_scope(agent, "outer"):
            async with nemo_relay_scope(agent, "inner"):

                async def execute(ctx):
                    ctx.result = ExecutionResult(returned_value=42)
                    return ctx

                await agent.event_manager.run_middleware(
                    "execute_python", ExecutePythonContext(code="pass"), execute
                )
        await relay.subscribers.flush_async()
        calls = [
            e
            for e in events
            if getattr(e, "category", None) == "tool" and e.scope_category == "start"
        ]
        assert len(calls) == 1
    finally:
        relay.subscribers.deregister("review-installation")
