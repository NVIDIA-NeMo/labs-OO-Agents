# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for token calibration from API response usage stats.

After each LLM call, the runtime extracts response.usage.prompt_tokens and
uses that provider-reported actual for the headline context stat. Local estimates
remain fallback diagnostics and recovery inputs, not the primary token signal.
"""

import pytest

from nooa import Agent
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.events import Message
from nooa.unifiedllm import FakeLLMClient, LLMResponse


class _CalibratingFakeLLM(FakeLLMClient):
    """FakeLLM that reports usage.prompt_tokens in the response."""

    _cw = 200_000

    @property
    def context_window(self):
        return self._cw

    def count_tokens(self, text: str) -> int:
        import litellm

        return litellm.token_counter(model="anthropic/claude-3-5-sonnet-20240620", text=text)


def _mk_calibrating_llm(context_window: int = 200_000):
    """Create a FakeLLM with a configurable context window."""

    class _LLM(_CalibratingFakeLLM):
        _cw = context_window

    llm = _LLM()
    llm.model = "anthropic/claude-3-5-sonnet-20240620"
    return llm


class TestTokenCalibration:
    """Token calibration populates calibrated context stats without a per-actor cap ratio."""

    @pytest.mark.asyncio
    async def test_context_stats_populated_after_llm_call(self):
        """After a successful LLM call, _last_context_stats carries the
        provider-reported prompt tokens (the single source of truth) and the
        structural event count from the render pass."""
        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="ok",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": "ok"},
                    usage={"prompt_tokens": 500, "completion_tokens": 7},
                )
            ]
        )

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        # Add some events so there's actual content to measure
        for i in range(10):
            tc_id = f"call_{i}"
            agent.event_manager.add(
                ToolCallEvent(
                    tool_call_id=tc_id,
                    name="execute_python",
                    arguments={"code": f"x = {i}"},
                    result=ToolResult(
                        tool_call_id=tc_id,
                        content=f"done_{i}",
                        result_status=ResultStatus.COMPLETE,
                    ),
                )
            )

        runtime = agent.runtime
        assert not hasattr(runtime, "_token_calibration_ratio")

        await agent.respond("hello")

        # Populated from provider usage, not a local estimate.
        assert runtime._last_context_stats is not None
        assert runtime._last_context_stats.total_tokens == 500
        assert runtime._last_context_stats.events_count >= 10

    @pytest.mark.asyncio
    async def test_headline_is_raw_provider_total_no_ratio(self):
        """The headline token count is the provider's prompt_tokens verbatim —
        no per-actor calibration ratio or 70%-style scaling is applied to it."""
        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="ok",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": "ok"},
                    usage={"prompt_tokens": 150_000, "completion_tokens": 9},
                )
            ]
        )

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        runtime = agent.runtime
        assert not hasattr(runtime, "_token_calibration_ratio")

        agent.event_manager.add(Message(content="some event"))

        await agent.respond("test calibration")

        stats = runtime._last_context_stats
        assert stats is not None
        # Raw provider value, unscaled.
        assert stats.total_tokens == 150_000
        assert runtime._last_prompt_tokens_actual == 150_000
        assert not hasattr(runtime, "_token_calibration_ratio")

    @pytest.mark.asyncio
    async def test_provider_response_recalibrates_tokens_per_char(self):
        """A provider response updates _tokens_per_char to exactly
        prompt_tokens / total_chars, replacing the cold-start default."""
        import pytest as _pytest

        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="ok",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": "ok"},
                    usage={"prompt_tokens": 5_000, "completion_tokens": 4},
                )
            ]
        )

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="x" * 400))
        runtime = agent.runtime

        # Before any response: the cold-start default.
        from nooa.runtime.actor import _DEFAULT_TOKENS_PER_CHAR

        assert runtime._tokens_per_char == _DEFAULT_TOKENS_PER_CHAR

        await agent.respond("hi")

        stats = runtime._last_context_stats
        total_chars = stats.context_blocks_chars + stats.events_chars
        assert total_chars > 0
        # Ratio is exactly prompt_tokens / total_chars from this response...
        assert runtime._tokens_per_char == _pytest.approx(5_000 / total_chars)
        # ...and it actually moved off the default (small prompt → large ratio).
        assert runtime._tokens_per_char != _DEFAULT_TOKENS_PER_CHAR


class TestTokenCalibrationEdgeCases:
    """Edge cases for the calibration logic."""

    def test_ratio_not_set_when_usage_missing(self):
        """If response has no usage, ratio stays None."""
        llm = _mk_calibrating_llm()

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        runtime = agent.runtime
        assert not hasattr(runtime, "_token_calibration_ratio")

    def test_ratio_not_set_when_stats_missing(self):
        """If _last_context_stats is None (no prior render), ratio stays None."""
        llm = _mk_calibrating_llm()

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        runtime = agent.runtime
        assert runtime._last_context_stats is None
        assert not hasattr(runtime, "_token_calibration_ratio")


class TestActualTokenStats:
    """Provider usage is the authoritative headline token count."""

    @pytest.mark.asyncio
    async def test_response_usage_overwrites_context_stats_total_tokens(self):
        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="ok",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": "ok"},
                    usage={"prompt_tokens": 12_345, "completion_tokens": 7},
                )
            ]
        )

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="small event"))

        result = await agent.respond("hi")

        assert result == "ok"
        stats = agent.runtime._last_context_stats
        assert stats is not None
        assert stats.total_tokens == 12_345
        assert agent.runtime._last_prompt_tokens_actual == 12_345

    @pytest.mark.asyncio
    async def test_missing_usage_leaves_total_tokens_none(self):
        """No provider usage → no token count at all. There is no local
        estimate to fall back to: total_tokens stays None, but structural
        fields from the render pass are still populated."""
        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="ok",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": "ok"},
                    usage=None,
                )
            ]
        )

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="small event"))

        result = await agent.respond("hi")

        assert result == "ok"
        stats = agent.runtime._last_context_stats
        assert stats is not None
        assert stats.total_tokens is None
        assert stats.context_blocks_tokens is None
        # Structural facts are still recorded.
        assert stats.events_count >= 1
        assert agent.runtime._last_prompt_tokens_actual is None
