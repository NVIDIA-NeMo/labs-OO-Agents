# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for L4 context eviction.

Inline tests use FakeLLMClient (no API calls). Tests marked @pytest.mark.integration
require a real LLM and should run in nightly CI only.
"""

import pytest

from nooa import Agent
from nooa.config.truncation_config import TruncationConfig
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.events import PythonOutput
from nooa.runtime.actor import _current_llm_var
from nooa.unifiedllm import FakeLLMClient

# ── Helpers ──────────────────────────────────────────────────────────────


class _FakeLLM(FakeLLMClient):
    """FakeLLM with a settable context_window for tests."""

    _cw = 4096

    @property
    def context_window(self):  # type: ignore[override]
        return self._cw

    def count_tokens(self, text: str) -> int:
        import any_llm

        return any_llm.token_counter(model="anthropic/claude-3-5-sonnet-20240620", text=text)


def _mk_agent(context_window: int = 4096, max_context_tokens: int | None = None) -> Agent:
    """Create a test agent with configurable window and context budget."""

    class _LLM(_FakeLLM):
        _cw = context_window

    llm = _LLM()
    llm.model = "anthropic/claude-3-5-sonnet-20240620"  # type: ignore[attr-defined]

    if max_context_tokens is not None:
        tc = TruncationConfig(max_context_tokens=max_context_tokens)

        class A(Agent, llm=llm, truncation=tc):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
    else:

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
    return agent


def _count_words(s: str) -> int:
    """Simple word-count token approximation for tests."""
    return len(s.split())


# ── Context Block EVICTION ───────────────────────────────────────────────


# ── pformat(unquote_strings=True) ───────────────────────────────────────


class TestUnquoteStrings:
    """Context blocks render strings verbatim (no quotes) via unquote_strings."""

    def test_short_string_no_quotes(self):
        from nooa.agentdoc import pformat

        assert pformat("Hello world", unquote_strings=True) == "Hello world"

    def test_long_string_gets_truncation_marker(self):
        from nooa.agentdoc import pformat

        result = pformat("a" * 1000, unquote_strings=True, max_string=100)
        assert result.startswith("str(len=1000")
        assert "[:50]=" in result

    def test_non_string_unaffected(self):
        from nooa.agentdoc import pformat

        assert pformat([1, 2, 3], unquote_strings=True) == pformat([1, 2, 3])

    def test_multiline_string_verbatim(self):
        from nooa.agentdoc import pformat

        text = "line1\nline2\nline3"
        result = pformat(text, unquote_strings=True)
        assert result == text

    def test_triple_quote_string_preserved(self):
        """String consisting of triple quotes (''') is not deleted by unquote logic."""
        from nooa.agentdoc import pformat

        # Edge case: input is literally 3 single-quote chars
        result = pformat("'''", unquote_strings=True)
        assert result == "'''", f"Expected '''' but got {repr(result)}"

    def test_string_with_embedded_triple_quotes(self):
        """String containing triple quotes inside is preserved."""
        from nooa.agentdoc import pformat

        text = "before''' after"
        result = pformat(text, unquote_strings=True)
        assert result == text


# ── Post-render event collapse ───────────────────────────────────────────


class TestPostRenderEventCollapse:
    """When rendered payload exceeds context_window, events are collapsed."""

    @pytest.mark.asyncio
    async def test_overflow_passes_all_messages_through(self):
        """Without proactive clamping, all events pass through _build_messages.

        The API will reject if context is too large, and the recovery path
        in generate() handles archival.
        """
        agent = _mk_agent(context_window=4096)

        # Fill with many events (will exceed 4096 token window)
        for i in range(20):
            tc_id = f"tc_{i}"
            agent.event_manager.add(
                ToolCallEvent(
                    tool_call_id=tc_id,
                    name="execute_python",
                    arguments={"code": "x " * 100},
                    result=ToolResult(
                        tool_call_id=tc_id,
                        content="done",
                        result_status=ResultStatus.COMPLETE,
                    ),
                )
            )
            agent.event_manager.add(
                PythonOutput(
                    tool_call_id=tc_id,
                    execution_count=i,
                    stdout="y " * 100,
                    stderr="",
                    execution_status=ResultStatus.COMPLETE,
                )
            )

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method,
                call_args=(agent, "hi"),
                call_kwargs={},
                max_output_tokens=2048,
            )
        finally:
            _current_llm_var.reset(token)

        assert messages is not None
        # No clamping — all events should be in the message list
        active = len(list(agent.event_manager.keys()))
        assert active == 40, f"All 40 events should still be active, got {active}"


# ── Default budget split ─────────────────────────────────────────────────


# ── Nightly integration tests (real LLM) ────────────────────────────────
