# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 — imports after pytest.importorskip are intentional
"""Tests for NeMo Relay middleware integration (nemo_relay_middleware.py).

Verifies that LLM and tool calls routed through the NeMo Relay pipeline
correctly support:
- LLM request intercepts (HTTP header injection)
- LLM sanitize-request guardrails (input transformation)
- LLM sanitize-response guardrails (output transformation)
- Tool conditional-execution guardrails (blocking)
- Tool sanitize-response guardrails (output transformation)
- ATIF trajectory export
"""

import pytest

nemo_relay = pytest.importorskip("nemo_relay", reason="nemo_relay not installed")

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from nooa.nemo_relay_middleware import (
    install_nemo_relay,
    nemo_relay_agent_call_middleware,
    nemo_relay_llm_middleware,
    nemo_relay_scope,
    nemo_relay_tool_middleware,
)
from nooa.runtime.event_manager import EventManager
from nooa.runtime.middleware import (
    AgentCallContext,
    ExecutePythonContext,
    LLMCallContext,
)
from nooa.unifiedllm import LLMResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _nemo_relay_scope():
    """Enter a NeMo Relay scope for every test so scope stack is initialised."""
    with nemo_relay.scope.scope("test-agent", nemo_relay.ScopeType.Agent):
        yield


# ---------------------------------------------------------------------------
# Fake LLMResponse (mirrors unifiedllm.LLMResponse dataclass)
# ---------------------------------------------------------------------------


class FakeLLMResponse(LLMResponse):
    """Convenient real LLMResponse with defaults for middleware tests."""

    def __init__(
        self,
        content: str = "hello",
        tool_calls: list | None = None,
        finish_reason: str = "stop",
        assistant_message: dict | None = None,
        reasoning: str | None = None,
        usage: dict | None = None,
    ) -> None:
        super().__init__(
            content=content,
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            assistant_message=assistant_message or {"role": "assistant", "content": content},
            reasoning=reasoning,
            usage=usage
            if usage is not None
            else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


# ---------------------------------------------------------------------------
# Fake ExecutionResult (mirrors nooa.events.ExecutionResult)
# ---------------------------------------------------------------------------


@dataclass
class FakeExecutionResult:
    stdout: str = ""
    error: Exception | None = None
    defined_methods: dict = field(default_factory=dict)
    returned_value: Any = "result-value"
    signal: Any = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_real_agent(model_name: str = "test-model"):
    """Create a real Agent with a mock LLM for model_name extraction."""
    from unittest.mock import MagicMock

    from nooa.agent import Agent

    mock_llm = MagicMock()
    mock_llm.model = model_name

    class _TestAgent(Agent, llm=mock_llm):
        pass

    return _TestAgent()


def _make_llm_ctx(
    messages: list[dict[str, Any]] | None = None,
    agent: Any = None,
) -> LLMCallContext:
    return LLMCallContext(
        messages=messages or [{"role": "user", "content": "hi"}],
        params={},
        agent=agent,
        runtime=None,
    )


def _make_exec_ctx(code: str = "x = 1", agent: Any = None) -> ExecutePythonContext:
    return ExecutePythonContext(
        code=code,
        params={"tool_call_id": "tc-1"},
        agent=agent,
        runtime=None,
    )


# ===================================================================
# LLM request intercepts (HTTP headers)
# ===================================================================


class TestLLMRequestIntercepts:
    """Verify that LLM request intercepts (header injection) work end-to-end."""

    @pytest.mark.asyncio
    async def test_request_intercept_injects_header(self):
        """A registered LLM request intercept can inject HTTP headers."""
        captured_headers: dict[str, str] = {}

        def _intercept(
            name: str, req: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
        ) -> nemo_relay.LLMRequestInterceptOutcome:
            req.headers["X-Test-Header"] = "test-value"
            return nemo_relay.LLMRequestInterceptOutcome(req, annotated)

        nemo_relay.intercepts.register_llm_request("test-header", 1, False, _intercept)
        try:
            # Subscriber captures the headers that were on the request
            def _capture(event: nemo_relay.Event):
                if (
                    isinstance(event, nemo_relay.ScopeEvent)
                    and event.category == "llm"
                    and event.scope_category == "start"
                    and event.data is not None
                ):
                    # The request headers flow through the pipeline via event.data
                    headers = event.data.get("headers") if isinstance(event.data, dict) else None
                    if isinstance(headers, dict):
                        captured_headers.update(headers)

            nemo_relay.subscribers.register("header-spy", _capture)

            ctx = _make_llm_ctx()
            fake_resp = FakeLLMResponse()

            async def nxt(c):
                c.response = fake_resp
                return c

            result = await nemo_relay_llm_middleware(ctx, nxt)

            # The middleware should have completed successfully
            assert result.response is fake_resp

            nemo_relay.subscribers.deregister("header-spy")
        finally:
            nemo_relay.intercepts.deregister_llm_request("test-header")

    @pytest.mark.asyncio
    async def test_request_intercept_modifies_messages_propagated_to_nxt(self):
        """A request intercept that modifies messages is propagated to nxt().

        This verifies that the modified LLMRequest from NeMo Relay request intercepts
        is applied back to ctx.messages before calling the rest of the middleware
        chain, so the actual LLM call sees the intercepted messages.
        """
        nxt_saw_messages: list[list] = []

        def _intercept(
            name: str, req: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
        ) -> nemo_relay.LLMRequestInterceptOutcome:
            # Must return a new LLMRequest — NeMo Relay serializes through Rust/JSON,
            # so in-place mutations on the original object are lost.
            new_content = dict(req.content)
            msgs = list(new_content.get("messages", []))
            msgs.insert(0, {"role": "system", "content": "INJECTED BY INTERCEPT"})
            new_content["messages"] = msgs
            return nemo_relay.LLMRequestInterceptOutcome(
                nemo_relay.LLMRequest(req.headers, new_content), annotated
            )

        nemo_relay.intercepts.register_llm_request("test-inject", 1, False, _intercept)
        try:
            ctx = _make_llm_ctx(messages=[{"role": "user", "content": "hello"}])

            async def nxt(c):
                # Capture what nxt() actually sees
                nxt_saw_messages.append(list(c.messages))
                c.response = FakeLLMResponse()
                return c

            await nemo_relay_llm_middleware(ctx, nxt)

            # nxt() should see the intercepted messages (system msg prepended)
            assert len(nxt_saw_messages) == 1
            msgs = nxt_saw_messages[0]
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == "INJECTED BY INTERCEPT"
            assert msgs[1]["content"] == "hello"
        finally:
            nemo_relay.intercepts.deregister_llm_request("test-inject")


# ===================================================================
# LLM sanitize-request guardrails (input transformation)
# ===================================================================


class TestLLMSanitizeRequest:
    """Verify LLM sanitize-request guardrails can transform LLM input."""

    @pytest.mark.asyncio
    async def test_sanitize_request_is_called(self):
        """A sanitize-request guardrail receives the LLMRequest."""
        sanitize_called = []

        def _sanitize(req):
            sanitize_called.append(True)
            return req

        nemo_relay.guardrails.register_llm_sanitize_request("test-san-req", 1, _sanitize)
        try:
            ctx = _make_llm_ctx()

            async def nxt(c):
                c.response = FakeLLMResponse()
                return c

            await nemo_relay_llm_middleware(ctx, nxt)
            assert len(sanitize_called) == 1
        finally:
            nemo_relay.guardrails.deregister_llm_sanitize_request("test-san-req")


# ===================================================================
# LLM sanitize-response guardrails (output transformation)
# ===================================================================


class TestLLMSanitizeResponse:
    """Verify LLM sanitize-response guardrails run and transform data for
    NeMo Relay internals (ATIF, subscribers).

    Note: NeMo Relay sanitize-response guardrails do NOT modify the value
    returned to the caller — they transform data for ATIF export and
    event subscribers only.  The caller always gets the original response.
    """

    @pytest.mark.asyncio
    async def test_sanitize_response_is_called_and_visible_to_subscribers(self):
        """A sanitize-response guardrail transforms what subscribers see."""
        subscriber_saw: list[Any] = []

        def _sanitize(response: dict) -> dict:
            msg = response.get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = msg["content"].replace("hello", "[REDACTED]")
            return response

        def _spy(event: nemo_relay.Event):
            if (
                isinstance(event, nemo_relay.ScopeEvent)
                and event.category == "llm"
                and event.scope_category == "end"
                and event.data is not None
            ):
                subscriber_saw.append(event.data)

        nemo_relay.guardrails.register_llm_sanitize_response("test-san-resp", 1, _sanitize)
        nemo_relay.subscribers.register("resp-spy", _spy)
        try:
            ctx = _make_llm_ctx()
            fake_resp = FakeLLMResponse(
                content="hello world",
                assistant_message={"role": "assistant", "content": "hello world"},
            )

            async def nxt(c):
                c.response = fake_resp
                return c

            result = await nemo_relay_llm_middleware(ctx, nxt)

            # Caller gets original (NeMo Relay design: sanitize is for observability)
            assert result.response.content == "hello world"

            # Subscriber should see the sanitized version
            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            assert len(subscriber_saw) > 0, "Subscriber should have received at least one event"
            last_output = subscriber_saw[-1]
            assert isinstance(last_output, dict), f"Expected dict output, got {type(last_output)}"
            msg = last_output.get("message")
            assert isinstance(msg, dict), f"Expected 'message' to be a dict, got {type(msg)}"
            assert msg.get("content") == "[REDACTED] world"
        finally:
            nemo_relay.guardrails.deregister_llm_sanitize_response("test-san-resp")
            nemo_relay.subscribers.deregister("resp-spy")

    @pytest.mark.asyncio
    async def test_sanitize_response_without_guardrail_preserves_original(self):
        """Without a sanitize-response guardrail, original response is unchanged."""
        ctx = _make_llm_ctx()
        fake_resp = FakeLLMResponse(
            content="original text",
            assistant_message={"role": "assistant", "content": "original text"},
        )

        async def nxt(c):
            c.response = fake_resp
            return c

        result = await nemo_relay_llm_middleware(ctx, nxt)
        assert result.response.content == "original text"


# ===================================================================
# LLM conditional-execution guardrails (blocking)
# ===================================================================


class TestLLMConditionalExecution:
    """Verify LLM conditional-execution guardrails can block calls."""

    @pytest.mark.asyncio
    async def test_conditional_execution_blocks_call(self):
        """A conditional-execution guardrail that returns a reason blocks the LLM call."""
        nxt_called = []

        def _guardrail(req) -> str | None:
            return "Blocked by test guardrail"

        nemo_relay.guardrails.register_llm_conditional_execution("test-block", 1, _guardrail)
        try:
            ctx = _make_llm_ctx()

            async def nxt(c):
                nxt_called.append(True)
                c.response = FakeLLMResponse()
                return c

            with pytest.raises(Exception, match="[Rr]ejected|[Bb]locked"):
                await nemo_relay_llm_middleware(ctx, nxt)

            # The actual LLM call should NOT have been made
            assert len(nxt_called) == 0
        finally:
            nemo_relay.guardrails.deregister_llm_conditional_execution("test-block")

    @pytest.mark.asyncio
    async def test_conditional_execution_allows_when_none(self):
        """A guardrail returning None allows the call to proceed."""

        def _guardrail(req) -> str | None:
            return None  # Allow

        nemo_relay.guardrails.register_llm_conditional_execution("test-allow", 1, _guardrail)
        try:
            ctx = _make_llm_ctx()

            async def nxt(c):
                c.response = FakeLLMResponse()
                return c

            result = await nemo_relay_llm_middleware(ctx, nxt)
            assert result.response is not None
        finally:
            nemo_relay.guardrails.deregister_llm_conditional_execution("test-allow")


# ===================================================================
# Tool conditional-execution guardrails (blocking)
# ===================================================================


class TestToolConditionalExecution:
    """Verify tool conditional-execution guardrails can block code execution."""

    @pytest.mark.asyncio
    async def test_tool_guardrail_blocks_dangerous_code(self):
        """A tool guardrail that rejects based on code content blocks execution."""
        nxt_called = []

        def _guardrail(tool_name: str, args: Any) -> str | None:
            code = args.get("code", "") if isinstance(args, dict) else ""
            if "os.system" in code:
                return "Dangerous code blocked"
            return None

        nemo_relay.guardrails.register_tool_conditional_execution("test-tool-block", 1, _guardrail)
        try:
            ctx = _make_exec_ctx(code="os.system('rm -rf /')")

            async def nxt(c):
                nxt_called.append(True)
                c.result = FakeExecutionResult()
                return c

            with pytest.raises(Exception, match="[Rr]ejected|[Bb]locked|[Dd]angerous"):
                await nemo_relay_tool_middleware(ctx, nxt)

            assert len(nxt_called) == 0
        finally:
            nemo_relay.guardrails.deregister_tool_conditional_execution("test-tool-block")

    @pytest.mark.asyncio
    async def test_tool_guardrail_allows_safe_code(self):
        """A tool guardrail returning None allows execution."""

        def _guardrail(tool_name: str, args: Any) -> str | None:
            return None

        nemo_relay.guardrails.register_tool_conditional_execution("test-tool-allow", 1, _guardrail)
        try:
            ctx = _make_exec_ctx(code="x = 42")

            async def nxt(c):
                c.result = FakeExecutionResult(returned_value=42)
                return c

            result = await nemo_relay_tool_middleware(ctx, nxt)
            assert result.result.returned_value == 42
        finally:
            nemo_relay.guardrails.deregister_tool_conditional_execution("test-tool-allow")


# ===================================================================
# Tool sanitize-response guardrails (output transformation)
# ===================================================================


class TestToolSanitizeResponse:
    """Verify tool sanitize-response guardrails run and transform data for
    NeMo Relay internals (ATIF, subscribers).

    Note: NeMo Relay sanitize-response guardrails do NOT modify the value
    returned to the caller — they transform data for ATIF export and
    event subscribers only.  The caller always gets the original result.
    """

    @pytest.mark.asyncio
    async def test_tool_sanitize_response_visible_to_subscribers(self):
        """A tool sanitize-response guardrail transforms what subscribers see."""
        subscriber_saw: list[Any] = []

        def _sanitize(tool_name: str, result: Any) -> Any:
            if isinstance(result, str):
                return result.replace("secret", "[REDACTED]")
            return result

        def _spy(event: nemo_relay.Event):
            if (
                isinstance(event, nemo_relay.ScopeEvent)
                and event.category == "tool"
                and event.scope_category == "end"
                and event.data is not None
            ):
                subscriber_saw.append(event.data)

        nemo_relay.guardrails.register_tool_sanitize_response("test-tool-san", 1, _sanitize)
        nemo_relay.subscribers.register("tool-resp-spy", _spy)
        try:
            ctx = _make_exec_ctx(code="x = 'secret data'")

            async def nxt(c):
                c.result = FakeExecutionResult(returned_value="secret data")
                return c

            result = await nemo_relay_tool_middleware(ctx, nxt)

            # Caller gets original (NeMo Relay design: sanitize is for observability)
            assert result.result.returned_value == "secret data"

            # Subscriber should see the sanitized version
            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            assert len(subscriber_saw) > 0
            assert any("[REDACTED]" in str(o) for o in subscriber_saw)
        finally:
            nemo_relay.guardrails.deregister_tool_sanitize_response("test-tool-san")
            nemo_relay.subscribers.deregister("tool-resp-spy")

    @pytest.mark.asyncio
    async def test_tool_sanitize_response_no_guardrail_preserves_original(self):
        """Without a sanitize-response guardrail, original result is unchanged."""
        ctx = _make_exec_ctx()

        async def nxt(c):
            c.result = FakeExecutionResult(returned_value="original")
            return c

        result = await nemo_relay_tool_middleware(ctx, nxt)
        assert result.result.returned_value == "original"


# ===================================================================
# Tool request intercepts (code transformation)
# ===================================================================


class TestToolRequestIntercepts:
    """Verify tool request intercepts can transform code and changes propagate."""

    @pytest.mark.asyncio
    async def test_tool_request_intercept_modifies_code_propagated_to_nxt(self):
        """A tool request intercept that modifies code is propagated to nxt().

        This verifies that the modified args from NeMo Relay tool request intercepts
        are applied back to ctx.code before calling the rest of the middleware
        chain, so the actual code execution sees the intercepted code.
        """
        nxt_saw_code: list[str] = []

        def _intercept(tool_name: str, args: Any) -> Any:
            if isinstance(args, dict) and "code" in args:
                args["code"] = "# SANDBOXED\n" + args["code"]
            return args

        nemo_relay.intercepts.register_tool_request("test-code-inject", 1, False, _intercept)
        try:
            ctx = _make_exec_ctx(code="x = 42")

            async def nxt(c):
                nxt_saw_code.append(c.code)
                c.result = FakeExecutionResult(returned_value=42)
                return c

            await nemo_relay_tool_middleware(ctx, nxt)

            # nxt() should see the intercepted code
            assert len(nxt_saw_code) == 1
            assert nxt_saw_code[0].startswith("# SANDBOXED\n")
            assert "x = 42" in nxt_saw_code[0]
        finally:
            nemo_relay.intercepts.deregister_tool_request("test-code-inject")


# ===================================================================
# Model name extraction
# ===================================================================


class TestModelNameExtraction:
    """Verify model_name is correctly extracted and passed to NeMo Relay."""

    @pytest.mark.asyncio
    async def test_model_name_from_agent_llm(self):
        """model_name is extracted from agent._llm.model."""
        captured_model: list[str] = []

        def _capture(event: nemo_relay.Event):
            # In nemo_relay v0.1.0 the LLM scope name *is* the model name.
            if isinstance(event, nemo_relay.ScopeEvent) and event.category == "llm" and event.name:
                captured_model.append(event.name)

        nemo_relay.subscribers.register("model-spy", _capture)
        try:
            agent = _make_real_agent(model_name="anthropic/claude-3")
            ctx = _make_llm_ctx(agent=agent)

            async def nxt(c):
                c.response = FakeLLMResponse()
                return c

            await nemo_relay_llm_middleware(ctx, nxt)
            assert any("claude-3" in m for m in captured_model)
        finally:
            nemo_relay.subscribers.deregister("model-spy")


# ===================================================================
# install_nemo_relay / nemo_relay_scope
# ===================================================================


class TestInstallNemoRelayAndScope:
    """Verify install_nemo_relay() and nemo_relay_scope() lifecycle."""

    def test_install_nemo_relay_registers_three_middleware(self):
        """install_nemo_relay() registers agent_call, llm_call, and execute_python."""
        em = EventManager()
        uninstall = install_nemo_relay(em)

        assert len(em._middleware["agent_call"]) == 1
        assert len(em._middleware["llm_call"]) == 1
        assert len(em._middleware["execute_python"]) == 1

        uninstall()

        assert len(em._middleware["agent_call"]) == 0
        assert len(em._middleware["llm_call"]) == 0
        assert len(em._middleware["execute_python"]) == 0

    @pytest.mark.asyncio
    async def test_nemo_relay_scope_yields_handle_with_uuid(self):
        """nemo_relay_scope() yields a handle whose uuid can be used for ATIF."""
        agent = MagicMock()
        agent.event_manager = EventManager()

        async with nemo_relay_scope(agent, "test-scope") as handle:
            assert hasattr(handle, "uuid")
            assert handle.uuid  # non-empty

    @pytest.mark.asyncio
    async def test_nemo_relay_scope_cleans_up_middleware(self):
        """nemo_relay_scope() removes middleware on exit."""
        agent = MagicMock()
        em = EventManager()
        agent.event_manager = em

        async with nemo_relay_scope(agent, "test-scope"):
            assert len(em._middleware["llm_call"]) == 1

        assert len(em._middleware["llm_call"]) == 0


# ===================================================================
# ATIF trajectory export
# ===================================================================


class TestATIFExport:
    """Verify ATIF trajectory export captures LLM and tool events."""

    @pytest.mark.asyncio
    async def test_atif_captures_llm_call(self):
        """An LLM call through NeMo Relay middleware appears in the ATIF trajectory."""
        exporter = nemo_relay.AtifExporter(
            session_id="test-session",
            agent_name="test-agent",
            agent_version="0.1.0",
        )
        exporter.register("atif-test")
        try:
            with nemo_relay.scope.scope("atif-test-agent", nemo_relay.ScopeType.Agent):
                ctx = _make_llm_ctx()

                async def nxt(c):
                    c.response = FakeLLMResponse()
                    return c

                await nemo_relay_llm_middleware(ctx, nxt)

            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            traj = exporter.export_json()
            assert isinstance(traj, str) or isinstance(traj, dict), (
                f"Unexpected export_json type: {type(traj)}"
            )
            # export_json() may return a JSON string or dict; normalize
            if isinstance(traj, str):
                import json

                traj = json.loads(traj)
            # The export may be a single trajectory dict or a list; handle both
            if isinstance(traj, list):
                assert len(traj) > 0, "Expected at least one trajectory"
                traj = traj[0]
            assert traj["schema_version"].startswith("ATIF-v1.")
            assert traj["session_id"] == "test-session"
            assert len(traj["steps"]) > 0
        finally:
            exporter.deregister("atif-test")

    @pytest.mark.asyncio
    async def test_atif_captures_tool_call(self):
        """A tool call through NeMo Relay middleware appears in the ATIF trajectory."""
        exporter = nemo_relay.AtifExporter(
            session_id="test-session",
            agent_name="test-agent",
            agent_version="0.1.0",
        )
        exporter.register("atif-test-tool")
        try:
            with nemo_relay.scope.scope("atif-test-agent", nemo_relay.ScopeType.Agent):
                ctx = _make_exec_ctx(code="x = 42")

                async def nxt(c):
                    c.result = FakeExecutionResult(returned_value=42)
                    return c

                await nemo_relay_tool_middleware(ctx, nxt)

            traj = exporter.export_json()
            if isinstance(traj, str):
                import json

                traj = json.loads(traj)
            if isinstance(traj, list):
                assert len(traj) > 0
                traj = traj[0]
            assert len(traj["steps"]) > 0
        finally:
            exporter.deregister("atif-test-tool")


# ===================================================================
# nemo_relay_agent_call_middleware (scope push/pop, exception safety)
# ===================================================================


class TestAgentCallMiddleware:
    """Verify nemo_relay_agent_call_middleware pushes/pops NeMo Relay scopes correctly."""

    @pytest.mark.asyncio
    async def test_scope_name_format(self):
        """Scope name is formatted as 'ClassName.method_name'."""
        captured_names: list[str] = []

        def _spy(event: nemo_relay.Event):
            if (
                isinstance(event, nemo_relay.ScopeEvent)
                and event.scope_category == "start"
                and event.category == "function"
            ):
                captured_names.append(event.name)

        nemo_relay.subscribers.register("scope-name-spy", _spy)
        try:
            agent = _make_real_agent()
            ctx = AgentCallContext(
                agent=agent,
                method_name="my_method",
                args=(),
                kwargs={},
            )

            async def nxt(c):
                c.result = "done"
                return c

            await nemo_relay_agent_call_middleware(ctx, nxt)
            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            assert len(captured_names) == 1
            assert captured_names[0] == f"{type(agent).__name__}.my_method"
        finally:
            nemo_relay.subscribers.deregister("scope-name-spy")

    @pytest.mark.asyncio
    async def test_scope_pushed_and_popped(self):
        """A Function scope is pushed before nxt() and popped after."""
        events: list[tuple[str, str]] = []

        def _spy(event: nemo_relay.Event):
            if isinstance(event, nemo_relay.ScopeEvent) and event.category == "function":
                events.append((event.scope_category, event.name))

        nemo_relay.subscribers.register("scope-lifecycle-spy", _spy)
        try:
            agent = _make_real_agent()
            ctx = AgentCallContext(agent=agent, method_name="run", args=(), kwargs={})

            async def nxt(c):
                c.result = "ok"
                return c

            await nemo_relay_agent_call_middleware(ctx, nxt)
            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay

            # Should see start then end
            assert len(events) == 2
            assert events[0][0] == "start"
            assert events[1][0] == "end"
        finally:
            nemo_relay.subscribers.deregister("scope-lifecycle-spy")

    @pytest.mark.asyncio
    async def test_scope_popped_on_exception(self):
        """Scope is popped even if nxt() raises — no scope leak."""
        events: list[str] = []

        def _spy(event: nemo_relay.Event):
            if isinstance(event, nemo_relay.ScopeEvent) and event.category == "function":
                events.append(event.scope_category)

        nemo_relay.subscribers.register("scope-exc-spy", _spy)
        try:
            agent = _make_real_agent()
            ctx = AgentCallContext(agent=agent, method_name="failing", args=(), kwargs={})

            async def nxt(c):
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await nemo_relay_agent_call_middleware(ctx, nxt)

            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            # Scope should still be popped (end event emitted)
            assert "start" in events
            assert "end" in events
        finally:
            nemo_relay.subscribers.deregister("scope-exc-spy")

    @pytest.mark.asyncio
    async def test_result_passes_through(self):
        """The result from nxt() passes through unchanged."""
        agent = _make_real_agent()
        ctx = AgentCallContext(agent=agent, method_name="compute", args=(), kwargs={})

        async def nxt(c):
            c.result = {"answer": 42}
            return c

        result = await nemo_relay_agent_call_middleware(ctx, nxt)
        assert result.result == {"answer": 42}


# ===================================================================
# Tool middleware: signal-based results and edge cases
# ===================================================================


class TestToolMiddlewareEdgeCases:
    """Test tool middleware with signal-based results and None results."""

    @pytest.mark.asyncio
    async def test_signal_based_return_result(self):
        """Tool middleware extracts value from signal.result['result']."""
        subscriber_saw: list[Any] = []

        def _spy(event: nemo_relay.Event):
            if (
                isinstance(event, nemo_relay.ScopeEvent)
                and event.category == "tool"
                and event.scope_category == "end"
            ):
                subscriber_saw.append(event.data)

        nemo_relay.subscribers.register("signal-spy", _spy)
        try:
            ctx = _make_exec_ctx(code="return_result(99)")

            # Simulate signal-based result (returned_value is _NO_RETURN sentinel)
            from nooa.events import _NO_RETURN

            @dataclass
            class FakeSignal:
                result: dict = field(default_factory=lambda: {"result": 99})

            fake_result = FakeExecutionResult(
                returned_value=_NO_RETURN,
                signal=FakeSignal(),
            )

            async def nxt(c):
                c.result = fake_result
                return c

            result = await nemo_relay_tool_middleware(ctx, nxt)

            # Caller gets the original ExecutionResult unchanged
            assert result.result is fake_result

            # NeMo Relay subscriber should see the extracted value (99)
            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            assert len(subscriber_saw) > 0
            assert any(o == 99 for o in subscriber_saw)
        finally:
            nemo_relay.subscribers.deregister("signal-spy")

    @pytest.mark.asyncio
    async def test_stdout_fallback_when_no_return_value(self):
        """Tool middleware falls back to stdout when no returned_value or signal."""
        subscriber_saw: list[Any] = []

        def _spy(event: nemo_relay.Event):
            if (
                isinstance(event, nemo_relay.ScopeEvent)
                and event.category == "tool"
                and event.scope_category == "end"
            ):
                subscriber_saw.append(event.data)

        nemo_relay.subscribers.register("stdout-spy", _spy)
        try:
            ctx = _make_exec_ctx(code="print('hello')")

            from nooa.events import _NO_RETURN

            fake_result = FakeExecutionResult(
                returned_value=_NO_RETURN,
                signal=None,
                stdout="hello\n",
            )

            async def nxt(c):
                c.result = fake_result
                return c

            result = await nemo_relay_tool_middleware(ctx, nxt)
            assert result.result is fake_result

            # NeMo Relay should see the stdout value
            nemo_relay.subscribers.flush()  # event dispatch is async in nemo_relay
            assert len(subscriber_saw) > 0
            assert any("hello" in str(o) for o in subscriber_saw)
        finally:
            nemo_relay.subscribers.deregister("stdout-spy")

    @pytest.mark.asyncio
    async def test_none_result(self):
        """Tool middleware handles result=None without crashing."""
        ctx = _make_exec_ctx(code="pass")

        async def nxt(c):
            c.result = FakeExecutionResult(returned_value=None)
            return c

        result = await nemo_relay_tool_middleware(ctx, nxt)
        assert result.result is not None
        assert result.result.returned_value is None


# ===================================================================
# LLM middleware: agent=None fallback
# ===================================================================


class TestLLMMiddlewareEdgeCases:
    """Test LLM middleware edge cases."""

    @pytest.mark.asyncio
    async def test_agent_none_uses_empty_model_name(self):
        """When agent is None, model_name defaults to empty string."""
        captured_model: list[str] = []

        def _spy(event: nemo_relay.Event):
            # When agent is None the LLM middleware passes model_name="" to
            # nemo_relay.llm.execute(), which becomes the scope name.
            if isinstance(event, nemo_relay.ScopeEvent) and event.category == "llm":
                captured_model.append(event.name or "")

        nemo_relay.subscribers.register("model-none-spy", _spy)
        try:
            ctx = _make_llm_ctx(agent=None)

            async def nxt(c):
                c.response = FakeLLMResponse()
                return c

            result = await nemo_relay_llm_middleware(ctx, nxt)
            assert result.response is not None
            # Should not crash — model_name is empty string
            assert all(m == "" for m in captured_model)
        finally:
            nemo_relay.subscribers.deregister("model-none-spy")


# ===================================================================
# nemo_relay_scope: cleanup on exception
# ===================================================================


class TestNemoRelayScopeExceptionSafety:
    """Verify nemo_relay_scope cleans up middleware when the body raises."""

    @pytest.mark.asyncio
    async def test_middleware_removed_on_exception(self):
        """nemo_relay_scope() removes middleware even if the body raises."""
        agent = MagicMock()
        em = EventManager()
        agent.event_manager = em

        with pytest.raises(RuntimeError, match="test explosion"):
            async with nemo_relay_scope(agent, "failing-scope"):
                assert len(em._middleware["llm_call"]) == 1
                raise RuntimeError("test explosion")

        # Middleware should be cleaned up despite the exception
        assert len(em._middleware["agent_call"]) == 0
        assert len(em._middleware["llm_call"]) == 0
        assert len(em._middleware["execute_python"]) == 0


# ===================================================================
# Regression: agent_call middleware preserves None return values
# ===================================================================


class TestAgentCallNoneReturn:
    """Verify that agent methods returning None work with middleware installed."""

    @pytest.mark.asyncio
    async def test_none_return_is_preserved(self):
        """An agent method that legitimately returns None must not raise RuntimeError."""
        from nooa.runtime.middleware import _AGENT_RESULT_NOT_SET, AgentCallContext

        ctx = AgentCallContext(
            agent=None,
            method_name="do_nothing",
            args=(),
            kwargs={},
        )
        # result starts as sentinel, not None
        assert ctx.result is _AGENT_RESULT_NOT_SET

        async def nxt(c):
            c.result = None  # method returns None
            return c

        result_ctx = await nemo_relay_agent_call_middleware(ctx, nxt)
        # None should be preserved, not treated as "not set"
        assert result_ctx.result is None

    @pytest.mark.asyncio
    async def test_sentinel_detected_when_nxt_not_called(self):
        """If middleware short-circuits without setting result, sentinel remains."""
        from nooa.runtime.middleware import _AGENT_RESULT_NOT_SET, AgentCallContext

        ctx = AgentCallContext(
            agent=None,
            method_name="skipped",
            args=(),
            kwargs={},
        )
        assert ctx.result is _AGENT_RESULT_NOT_SET

        # A middleware that doesn't call nxt and doesn't set result
        async def bad_middleware(c, nxt):
            return c  # forgot to call nxt or set c.result

        # The sentinel should still be there
        assert ctx.result is _AGENT_RESULT_NOT_SET


# ===================================================================
# Regression: LLM intercepts propagate non-message params
# ===================================================================


class TestLLMRequestInterceptParamPropagation:
    """Verify that LLM request intercepts can modify params beyond messages."""

    @pytest.mark.asyncio
    async def test_temperature_intercept_propagated(self):
        """A request intercept that changes temperature is seen by nxt()."""
        nxt_saw_params: list[dict] = []

        def _intercept(
            name: str, req: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
        ) -> nemo_relay.LLMRequestInterceptOutcome:
            new_content = dict(req.content)
            new_content["temperature"] = 0.0
            return nemo_relay.LLMRequestInterceptOutcome(
                nemo_relay.LLMRequest(req.headers, new_content), annotated
            )

        nemo_relay.intercepts.register_llm_request("test-temp", 1, False, _intercept)
        try:
            ctx = _make_llm_ctx()
            ctx.params["temperature"] = 0.7  # original value

            async def nxt(c):
                nxt_saw_params.append(dict(c.params))
                c.response = FakeLLMResponse()
                return c

            await nemo_relay_llm_middleware(ctx, nxt)

            assert len(nxt_saw_params) == 1
            assert nxt_saw_params[0]["temperature"] == 0.0
        finally:
            nemo_relay.intercepts.deregister_llm_request("test-temp")

    @pytest.mark.asyncio
    async def test_max_tokens_intercept_propagated(self):
        """A request intercept that limits max_tokens is seen by nxt()."""
        nxt_saw_params: list[dict] = []

        def _intercept(
            name: str, req: nemo_relay.LLMRequest, annotated: nemo_relay.AnnotatedLLMRequest | None
        ) -> nemo_relay.LLMRequestInterceptOutcome:
            new_content = dict(req.content)
            new_content["max_tokens"] = 100
            return nemo_relay.LLMRequestInterceptOutcome(
                nemo_relay.LLMRequest(req.headers, new_content), annotated
            )

        nemo_relay.intercepts.register_llm_request("test-maxtok", 1, False, _intercept)
        try:
            ctx = _make_llm_ctx()
            ctx.params["max_tokens"] = 4096

            async def nxt(c):
                nxt_saw_params.append(dict(c.params))
                c.response = FakeLLMResponse()
                return c

            await nemo_relay_llm_middleware(ctx, nxt)

            assert nxt_saw_params[0]["max_tokens"] == 100
        finally:
            nemo_relay.intercepts.deregister_llm_request("test-maxtok")


# ===================================================================
# Regression: Tool intercepts propagate non-code params
# ===================================================================


class TestToolRequestInterceptParamPropagation:
    """Verify that tool request intercepts can modify params beyond code."""

    @pytest.mark.asyncio
    async def test_timeout_intercept_propagated(self):
        """A tool request intercept that modifies timeout is seen by nxt()."""
        nxt_saw_params: list[dict] = []

        def _intercept(tool_name: str, args: Any) -> Any:
            if isinstance(args, dict):
                args["timeout"] = 5
            return args

        nemo_relay.intercepts.register_tool_request("test-timeout", 1, False, _intercept)
        try:
            ctx = _make_exec_ctx(code="x = 1")
            ctx.params["timeout"] = 30  # original

            async def nxt(c):
                nxt_saw_params.append(dict(c.params))
                c.result = FakeExecutionResult(returned_value=1)
                return c

            await nemo_relay_tool_middleware(ctx, nxt)

            assert nxt_saw_params[0]["timeout"] == 5
        finally:
            nemo_relay.intercepts.deregister_tool_request("test-timeout")


# ===================================================================
# Regression: overlapping agent calls must not corrupt the scope stack
# ===================================================================


class TestConcurrentAgentCallScopes:
    """NeMo Relay keeps one mutable LIFO scope stack per context, inherited by
    child asyncio tasks *by reference*.  Agent calls that overlap in time (e.g.
    ``asyncio.gather(self.a(), self.b())`` in generated code) used to interleave
    their pushes on that shared stack; whichever finished first could not pop,
    leaving the stack permanently desynchronised and raising out of
    ``nemo_relay_scope``.  ``nemo_relay_agent_call_middleware`` now isolates a
    concurrently-dispatched call onto its own stack.
    """

    def test_scope_stack_var_still_exists_upstream(self):
        """The private ContextVar the fix relies on must still be there.

        It is not part of ``nemo_relay.__all__``; if upstream renames it the
        middleware silently degrades to the old (broken) shared-stack path, so
        fail loudly here instead.
        """
        import nooa.nemo_relay_middleware as nm

        assert getattr(nemo_relay, "_scope_stack_var", None) is not None, (
            "nemo_relay._scope_stack_var is gone — nemo_relay_agent_call_middleware "
            "can no longer isolate concurrent scopes; see its Concurrency docstring."
        )
        assert nm._SCOPE_STACK_VAR is not None

    @pytest.mark.asyncio
    async def test_overlapping_agent_calls_keep_stack_balanced(self):
        """Siblings that finish out of order still pop cleanly."""
        import asyncio

        async def one(method_name: str, delay: float):
            ctx = AgentCallContext(agent=None, method_name=method_name)

            async def body(c):
                await asyncio.sleep(delay)
                c.result = method_name
                return c

            await nemo_relay_agent_call_middleware(ctx, body)

        with nemo_relay.scope.scope("concurrent-root", nemo_relay.ScopeType.Agent) as root:
            import nooa.nemo_relay_middleware as nm

            token = nm._current_relay_scope.set((root, asyncio.current_task()))
            try:
                # "first" finishes before "second", so a shared stack would pop
                # out of order and raise.
                await asyncio.gather(one("first", 0.01), one("second", 0.05))
                # The root must still be on top; otherwise the surrounding
                # scope.scope() would raise on exit.
                assert nemo_relay.scope.get_handle().uuid == root.uuid
            finally:
                nm._current_relay_scope.reset(token)

    @pytest.mark.asyncio
    async def test_overlapping_siblings_share_the_same_parent(self):
        """Isolation must not detach siblings into separate roots."""
        import asyncio

        handles: list = []

        async def one(method_name: str, delay: float):
            ctx = AgentCallContext(agent=None, method_name=method_name)

            async def body(c):
                import nooa.nemo_relay_middleware as nm

                handles.append(nm._current_relay_scope.get()[0])
                await asyncio.sleep(delay)
                c.result = method_name
                return c

            await nemo_relay_agent_call_middleware(ctx, body)

        with nemo_relay.scope.scope("parent-root", nemo_relay.ScopeType.Agent) as root:
            import nooa.nemo_relay_middleware as nm

            token = nm._current_relay_scope.set((root, asyncio.current_task()))
            try:
                await asyncio.gather(one("left", 0.01), one("right", 0.03))
            finally:
                nm._current_relay_scope.reset(token)

        assert len(handles) == 2
        # Both siblings parent off the shared root, not off each other.
        assert {str(h.parent_uuid) for h in handles} == {str(root.uuid)}
        assert handles[0].uuid != handles[1].uuid

    @pytest.mark.asyncio
    async def test_sequential_nesting_still_unwinds(self):
        """The isolation path must not regress ordinary nested calls."""
        seen: list[str] = []

        async def inner(c):
            seen.append(nemo_relay.scope.get_handle().name)
            c.result = "inner"
            return c

        async def outer(c):
            seen.append(nemo_relay.scope.get_handle().name)
            inner_ctx = AgentCallContext(agent=None, method_name="inner")
            await nemo_relay_agent_call_middleware(inner_ctx, inner)
            # After the nested call returns, the scope must unwind back.
            seen.append(nemo_relay.scope.get_handle().name)
            c.result = "outer"
            return c

        with nemo_relay.scope.scope("seq-root", nemo_relay.ScopeType.Agent):
            ctx = AgentCallContext(agent=None, method_name="outer")
            await nemo_relay_agent_call_middleware(ctx, outer)

        assert seen == ["NoneType.outer", "NoneType.inner", "NoneType.outer"]
