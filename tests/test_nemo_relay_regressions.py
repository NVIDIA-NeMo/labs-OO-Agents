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
async def test_global_policy_covers_nested_methods_in_native_callback():
    seen = []
    relay.guardrails.register_tool_conditional_execution(
        "review-global", 1, lambda name, args: seen.append(name)
    )
    try:
        async with nemo_relay_scope(ReviewAgent(), "nested-policy"):

            async def inner(ctx):
                await relay.tools.execute("inner", {}, lambda _: relay.ToolExecutionResult("ok"))
                return ctx

            async def execute(ctx):
                await nemo_relay_agent_call_middleware(AgentCallContext(method_name="inner"), inner)
                ctx.result = ExecutionResult(returned_value="ok")
                return ctx

            await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        assert seen == ["execute_python", "inner"]
    finally:
        relay.guardrails.deregister_tool_conditional_execution("review-global")


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
        from nooa._relay_atif import _tool_display

        if value is None:
            assert data is None
        else:
            display = _tool_display(data)
            assert len(display) < 4200
            if isinstance(value, str):
                assert "omitted" in display
                assert data == value
            else:
                assert "opaque" in display
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
async def test_global_guardrail_blocks_parallel_agent_calls():
    ran = []
    relay.guardrails.register_tool_conditional_execution(
        "review-parallel-deny", 1, lambda name, args: "denied by shared policy"
    )
    try:
        async with nemo_relay_scope(ReviewAgent(), "parallel-policy"):

            async def execute(ctx):
                ran.append(True)
                return ctx

            async def body(ctx):
                await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
                return ctx

            results = await asyncio.gather(
                *(
                    nemo_relay_agent_call_middleware(AgentCallContext(method_name=name), body)
                    for name in ("a", "b")
                ),
                return_exceptions=True,
            )
        assert all(isinstance(result, Exception) and "denied" in str(result) for result in results)
        assert not ran
    finally:
        relay.guardrails.deregister_tool_conditional_execution("review-parallel-deny")


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


@pytest.mark.parametrize(
    "installed", ["0.0.6", "0.10.0", "0.11.0", "0.11.1", "0.12.0", "0.11.2rc1"]
)
def test_incompatible_relay_version_fails_before_install(monkeypatch, installed):
    import nooa.nemo_relay_middleware as middleware

    middleware._check_relay_version.cache_clear()
    monkeypatch.setattr(middleware, "version", lambda name: installed)
    try:
        with pytest.raises(ImportError, match="requires nemo-relay"):
            middleware.install_nemo_relay(ReviewAgent().event_manager)
    finally:
        middleware._check_relay_version.cache_clear()


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


@pytest.mark.parametrize("fail_at", [2, 3])
def test_installation_rolls_back_if_registration_fails(monkeypatch, fail_at):
    from nooa.nemo_relay_middleware import install_nemo_relay
    from nooa.runtime.event_manager import EventManager

    manager = EventManager()
    original = manager.intercept
    calls = 0

    def intercept(kind, fn):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("registration failed")
        return original(kind, fn)

    monkeypatch.setattr(manager, "intercept", intercept)
    with pytest.raises(RuntimeError, match="registration failed"):
        install_nemo_relay(manager)
    assert not hasattr(manager, "_nooa_relay_installation")
    assert all(not entries for entries in manager._middleware.values())
    monkeypatch.setattr(manager, "intercept", original)
    uninstall = install_nemo_relay(manager)
    assert sum(map(len, manager._middleware.values())) == 3
    uninstall()
    assert all(not entries for entries in manager._middleware.values())


def test_supported_relay_version_is_checked_once(monkeypatch):
    import nooa.nemo_relay_middleware as middleware

    calls = []
    middleware._check_relay_version.cache_clear()
    monkeypatch.setattr(middleware, "version", lambda name: calls.append(name) or "0.11.2")
    try:
        for _ in range(10):
            middleware._require_relay()
        assert calls == ["nemo-relay"]
    finally:
        middleware._check_relay_version.cache_clear()


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_hung_nested_callback_does_not_hold_run_open(monkeypatch):
    import nooa._relay_callback as callback_bridge

    monkeypatch.setattr(callback_bridge, "_CANCELLATION_GRACE_SECONDS", 0.02)
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    relay.subscribers.register("review-hung-nested", events.append)

    async def inner(ctx):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            finished.set()

    async def execute(ctx):
        return await nemo_relay_agent_call_middleware(AgentCallContext(method_name="hung"), inner)

    async def run():
        async with nemo_relay_scope(ReviewAgent(), "bounded-run"):
            await nemo_relay_agent_call_middleware(
                AgentCallContext(method_name="outer"),
                lambda ctx: nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute),
            )

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel("caller timeout")
        done, _ = await asyncio.wait({task}, timeout=0.5)
        assert task in done
        with pytest.raises(asyncio.CancelledError, match="caller timeout"):
            await task
        await relay.subscribers.flush_async()
        ended = {event.name for event in events if event.scope_category == "end"}
        assert {"bounded-run", "NoneType.outer"} <= ended
        # A later run has a clean stack even while the abandoned child unwinds.
        async with nemo_relay_scope(ReviewAgent(), "next-run") as handle:
            assert relay.scope.get_handle().uuid == handle.uuid
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await relay.subscribers.flush_async()
        relay.subscribers.deregister("review-hung-nested")


@pytest.mark.asyncio
async def test_tool_error_preview_does_not_call_user_string_methods():
    class HostileError(Exception):
        def __str__(self):
            raise AssertionError("error stringification must not run")

        def __repr__(self):
            raise AssertionError("error repr must not run")

    error = HostileError("safe detail")
    result = ExecutionResult(error=error)

    async def execute(ctx):
        ctx.result = result
        return ctx

    async with nemo_relay_scope(ReviewAgent(), "hostile-error"):
        context = await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        assert context.result is result
        assert context.result.error is error


@pytest.mark.asyncio
async def test_large_llm_preview_preserves_usage_and_provider_call_ids():
    from nooa.llm_types import LLMUsage, ToolCall

    events = []
    relay.subscribers.register("review-large-model-response", events.append)
    response = LLMResponse(
        content="x" * 1_000_000,
        tool_calls=[
            ToolCall(id="provider-large", name="execute_python", arguments='{"code":"42"}')
        ],
        usage=LLMUsage(input_tokens=7, output_tokens=5),
    )

    async def model(ctx):
        ctx.response = response
        return ctx

    try:
        async with nemo_relay_scope(ReviewAgent(), "large-model-response"):
            ctx = await nemo_relay_llm_middleware(LLMCallContext(messages=[]), model)
            assert ctx.response is response
        await relay.subscribers.flush_async()
        data = next(e.data for e in events if e.category == "llm" and e.scope_category == "end")
        assert data["usage"]["prompt_tokens"] == 7
        assert data["message"]["tool_calls"][0]["id"] == "provider-large"
        assert len(data["message"]["content"]) < 50000
        assert data["nooa.output.preview.incomplete"] is True
    finally:
        relay.subscribers.deregister("review-large-model-response")


@pytest.mark.asyncio
async def test_tool_snapshot_metadata_handles_invalid_unicode_keys():
    # The long value makes snapshot metadata contain a path through this key.
    result = ExecutionResult(returned_value={"key\ud800": "x" * 100000})

    async def execute(ctx):
        ctx.result = result
        return ctx

    async with nemo_relay_scope(ReviewAgent(), "unicode-preview-metadata"):
        ctx = await nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute)
        assert ctx.result is result
        assert ctx.result.returned_value["key\ud800"] == "x" * 100000
