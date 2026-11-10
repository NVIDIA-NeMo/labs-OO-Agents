# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Relay regressions for upgrade review findings; no model calls."""
# ruff: noqa: E402

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
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
@pytest.mark.parametrize(
    "value",
    [
        {1, 2},
        frozenset({1, 2}),
        range(3),
        deque([1, 2]),
        datetime(2026, 1, 2),
        2**64,
        -(2**63) - 1,
        "bad\ud800",
    ],
)
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
        assert data is not None
        json.dumps(data).encode("utf-8")
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
async def test_legacy_sanitizers_keep_payload_contract():
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
        assert llm["message"]["content"] == "REDACTED answer"
        assert llm["message"]["reasoning_content"] == "REDACTED rationale"
    finally:
        relay.guardrails.deregister_llm_sanitize_response("review-legacy")
        relay.guardrails.deregister_tool_sanitize_response("review-legacy")
        relay.subscribers.deregister("review-redaction")


@pytest.mark.asyncio
async def test_max_priority_legacy_event_sanitizer_keeps_payload_contract():
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

            relay.scope_local.register_scope_sanitize_end(
                root, "zzzz-late-legacy-event", 2**31 - 1, sanitize
            )

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
        assert any(isinstance(value, dict) and "message" in value for value in outputs)
        assert "REDACTED tool" in outputs
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
                async with nemo_relay_scope(ReviewAgent(), "first", isolate=True):
                    entered.set()
                    await release.wait()

            async def second():
                await entered.wait()
                async with nemo_relay_scope(ReviewAgent(), "second", isolate=True):
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
        from nooa.nemo_relay_middleware import _tool_display

        if value is None:
            assert data is None
        else:
            display = _tool_display(data)
            assert len(display) < 4200
            if isinstance(value, str):
                assert display.endswith("… [truncated]")
                assert data == value
            else:
                assert display == "[opaque Python value]"
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


@pytest.mark.asyncio
async def test_enclosing_local_policy_and_subscriber_survive_default_root():
    seen = []
    with relay.scope.scope("policy-owner", relay.ScopeType.Agent) as parent:
        relay.scope_local.register_subscriber(parent, "policy-owner-events", seen.append)
        relay.scope_local.register_tool_conditional_execution(
            parent, "policy-owner-deny", 1, lambda name, args: "denied by enclosing policy"
        )
        ran = False

        async def execute(ctx):
            nonlocal ran
            ran = True
            return ctx

        async with nemo_relay_scope(ReviewAgent(), "default-child"):
            with pytest.raises(Exception, match="denied"):
                await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        await relay.subscribers.flush_async()
        assert not ran
        assert any(event.name == "default-child" for event in seen)


@pytest.mark.asyncio
async def test_child_task_cannot_silently_discard_local_policy():
    ran = False
    async with nemo_relay_scope(ReviewAgent(), "safe-default"):

        async def body(ctx):
            nonlocal ran
            ran = True
            return ctx

        with pytest.raises(RuntimeError, match="explicit.*isolate=True"):
            await asyncio.create_task(
                nemo_relay_agent_call_middleware(AgentCallContext(method_name="child"), body)
            )
        assert not ran


@pytest.mark.asyncio
async def test_independent_roots_cannot_share_ambient_stack_concurrently():
    relay.get_scope_stack()  # Both tasks inherit this same mutable ambient stack.
    entered, release = asyncio.Event(), asyncio.Event()

    async def first():
        async with nemo_relay_scope(ReviewAgent(), "first-default"):
            entered.set()
            await release.wait()

    child = asyncio.create_task(first())
    try:
        await entered.wait()
        with pytest.raises(RuntimeError, match="Concurrent Relay roots"):
            async with nemo_relay_scope(ReviewAgent(), "second-default"):
                pytest.fail("unsafe root entered")
    finally:
        release.set()
        await child


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["llm", "tool"])
async def test_callback_exception_preserves_original_without_exporting_text(kind):
    events = []
    relay.subscribers.register("review-callback-error", events.append)
    original = ValueError("PRIVATE_CALLBACK_ERROR")
    try:
        async with nemo_relay_scope(ReviewAgent(), "callback-error"):

            async def fail(ctx):
                raise original

            with pytest.raises(ValueError) as caught:
                if kind == "llm":
                    await nemo_relay_llm_middleware(LLMCallContext(messages=[]), fail)
                else:
                    await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), fail)
            assert caught.value is original
        await relay.subscribers.flush_async()
        assert "PRIVATE_CALLBACK_ERROR" not in json.dumps(
            [{"data": e.data, "metadata": e.metadata} for e in events]
        )
    finally:
        relay.subscribers.deregister("review-callback-error")


@pytest.mark.parametrize("installed", ["0.0.6", "0.10.0", "0.11.0", "0.12.0", "0.11.1rc1"])
def test_incompatible_relay_version_fails_before_install(monkeypatch, installed):
    import nooa.nemo_relay_middleware as middleware

    monkeypatch.setattr(middleware, "version", lambda name: installed)
    with pytest.raises(ImportError, match="requires nemo-relay"):
        middleware.install_nemo_relay(ReviewAgent().event_manager)


def test_unflushed_interpreter_exit_has_no_native_callback_crash():
    """Exercise actual process teardown, where an in-process test cannot detect SIGSEGV."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio
        from nooa import Agent
        from nooa.events import ExecutionResult
        from nooa.nemo_relay_middleware import nemo_relay_scope, nemo_relay_tool_middleware
        from nooa.runtime.middleware import ExecutePythonContext
        from nooa.unifiedllm import FakeLLMClient

        class TestAgent(Agent, llm=FakeLLMClient()):
            pass

        async def execute(ctx):
            ctx.result = ExecutionResult(returned_value={'value': 42})
            return ctx

        async def main():
            async with nemo_relay_scope(TestAgent(), 'unflushed-exit'):
                for _ in range(5):
                    await nemo_relay_tool_middleware(ExecutePythonContext(code='pass'), execute)
            # Deliberately leave pending telemetry to test native interpreter teardown.

        asyncio.run(main())
        """
    )
    for _ in range(10):
        child = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        assert child.returncode == 0, child.stderr
        assert "Fatal Python error" not in child.stderr
