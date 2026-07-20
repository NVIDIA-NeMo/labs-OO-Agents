# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that empty response with finish_reason='length' raises immediately with an actionable message."""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TEST_LLM = FakeLLMClient()


def _resp(
    content: str, tool_calls: list | None = None, finish_reason: str | None = None
) -> LLMResponse:
    """Create a test LLM response."""
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


class TestMaxTokensExhaustedError:
    """Tests for the max_tokens exhaustion error in CodeAct strategy."""

    @pytest.mark.asyncio
    async def test_finish_reason_length_raises_immediately(self):
        """When model hits max_tokens (finish_reason='length') with no output,
        GenerationError is raised immediately (no retry) with an actionable message."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3, max_iterations=10)))
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", finish_reason="length"),  # Reasoning ate all tokens
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()

        # Verify a DebugTrace was emitted (not an Error event in LLM context)
        all_events = agent_instance.event_manager.values()
        debug_events = [e for e in all_events if e.event_type == "DebugTrace"]
        error_events = [e for e in all_events if e.event_type == "Error"]
        assert any("finish_reason='length'" in e.content for e in debug_events), (
            f"Expected DebugTrace with finish_reason, got: {[e.content for e in debug_events]}"
        )
        # No Error event should be injected into LLM context for this case
        max_tokens_errors = [e for e in error_events if "max_tokens" in e.content]
        assert len(max_tokens_errors) == 0, (
            f"max_tokens error should not be an Error event (LLM-visible), got: {max_tokens_errors}"
        )

    @pytest.mark.asyncio
    async def test_empty_response_without_length_retries_normally(self):
        """When empty response has finish_reason != 'length', normal retry logic applies."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(
                    config=CodeActConfig(
                        max_retries=2,
                        text_only_stop_behavior="synthetic_comment",
                    )
                )
            )
            async def my_task(self) -> str:
                """A task."""
                ...

        def _ret(val, cid="c2"):
            return ToolCall(id=cid, name="return_result", arguments=json.dumps({"result": val}))

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", finish_reason="stop"),  # Empty but not length
                _resp("", tool_calls=[_ret("hello")], finish_reason="tool_calls"),  # Recovery
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.my_task()
        assert result == "hello"
