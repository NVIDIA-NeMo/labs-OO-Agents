# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test CodeActStrategy calling PredictStrategy on the same agent.

Regression test for two properties of scoped block inheritance:
1. Strategy-specific blocks (e.g., CodeAct's execution_context) do NOT leak
   to nested calls that use a different strategy.
2. User-supplied blocks (e.g., from @strategy(blocks={...})) DO pass through
   to nested calls.

The original bug: CodeAct adds an "execution_context" block with
    expr="strategy.execution_context(runtime)"
When a nested call on the same agent uses a different strategy (e.g., Predict),
the scoped block persisted but `strategy` now points to PredictStrategy,
which doesn't have `execution_context()`, causing AttributeError.
"""

import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks import ScopedContext
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# --- Helpers ---


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    """Create a ToolCall for execute_python with the given code."""
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _return_result(result: str, call_id: str = "call_return") -> ToolCall:
    """Create a ToolCall for return_result with the given result."""
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


def _resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    """Create an LLMResponse with the given content and tool calls."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


class RecordingFakeLLM(FakeLLMClient):
    """FakeLLMClient that records ALL calls (not just the last one)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.all_calls: list[dict[str, Any]] = []

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.all_calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "output_model": output_model,
            }
        )
        return await super().acall(messages, tools, output_model, **kwargs)


# Unique marker that user-supplied blocks inject into the system prompt.
USER_BLOCK_MARKER = "CUSTOM_USER_BLOCK_MARKER_e7b3f9a2"


def _get_system_prompt(call: dict[str, Any]) -> str:
    """Extract the system prompt (first system message) from a recorded call."""
    for msg in call["messages"]:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Anthropic-style: list of text blocks
                return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content
    return ""


# --- Tests ---


@pytest.mark.asyncio
async def test_strategy_blocks_dont_leak_and_user_blocks_pass_through():
    """CodeAct method calling PredictStrategy method on the same agent:

    1. Strategy blocks (execution_context) must NOT leak to the nested call.
    2. User-supplied blocks (via @strategy(blocks=...)) MUST pass through.
    """
    responses = [
        # 1. CodeAct turn 1: execute_python that calls self.classify_sentiment()
        _resp(
            tool_calls=[
                _tool_call(
                    'result = await self.classify_sentiment("I love this!")',
                    call_id="call_exec_1",
                )
            ]
        ),
        # 2. PredictStrategy: classify_sentiment returns positive
        _resp('{"value": "positive"}'),
        # 3. CodeAct turn 2: return_result with the answer
        _resp(
            tool_calls=[
                _return_result("Sentiment: positive", call_id="call_ret"),
            ]
        ),
        # Extra responses for any template strategy calls (strategy_instructions, etc.)
        *[_resp("filler") for _ in range(20)],
    ]

    llm = RecordingFakeLLM(scripted_responses=responses)

    class AnalysisAgent(Agent, llm=llm):
        @strategy(PredictStrategy())
        async def classify_sentiment(self, text: str) -> str:
            """Classify as positive, negative, or neutral."""
            ...

        @strategy(
            CodeActStrategy(config=CodeActConfig(max_iterations=5, max_retries=1)),
            context=ScopedContext(
                context={
                    "user_hint": USER_BLOCK_MARKER,
                }
            ),
        )
        async def analyze(self, text: str) -> str:
            """Classify the text and return a summary."""
            ...

    agent = AnalysisAgent()
    result = await agent.analyze("I love this!")

    # --- Basic success assertion ---
    assert result is not None, "analyze() should return a result"
    assert "positive" in str(result).lower(), f"Expected 'positive' in result, got: {result}"

    # --- Find the PredictStrategy call ---
    # PredictStrategy call has output_model set (it's the one generating structured data).
    # CodeAct calls have tools set (execute_python, return_result).
    predict_strategy_calls = [c for c in llm.all_calls if c.get("output_model") is not None]
    assert len(predict_strategy_calls) >= 1, (
        f"Expected at least 1 PredictStrategy call, got {len(predict_strategy_calls)}. "
        f"Total calls: {len(llm.all_calls)}"
    )

    predict_prompt = _get_system_prompt(predict_strategy_calls[0])

    # --- Assertion 1: User-supplied blocks PASS THROUGH ---
    assert USER_BLOCK_MARKER in predict_prompt, (
        f"User-supplied block content '{USER_BLOCK_MARKER}' should be present in "
        f"PredictStrategy system prompt, but was not found.\n"
        f"System prompt preview: {predict_prompt[:500]}..."
    )

    # --- Assertion 2: Strategy blocks do NOT leak ---
    # CodeAct's execution_context produces content containing these markers.
    # If the block leaked, these would appear in the PredictStrategy prompt.
    assert "Execution Context" not in predict_prompt, (
        "CodeAct's 'execution_context' block content leaked to PredictStrategy. "
        "Strategy-specific blocks should be filtered out for nested calls with "
        "different strategies."
    )
    assert "Available in execute_python" not in predict_prompt, (
        "CodeAct's 'execution_context' block content leaked to PredictStrategy. "
        "Found 'Available in execute_python' which is CodeAct-specific."
    )
