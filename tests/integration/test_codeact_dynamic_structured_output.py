# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test CodeActStrategy dynamically defining @strategy(PredictStrategy()) functions.

Reproduces the scenario where the LLM generates code inside execute_python() that:
1. Defines a standalone async function (no `self`) with @strategy(PredictStrategy())
   and an ellipsis body.
2. Calls that function in parallel via `asyncio.gather` to process batch inputs.
3. Returns the aggregated results via inline return_result().

Helpers are not attached to the agent — calls go through bare names, not
`self.helper`. This exercises the full pipeline: HelperFunctionManager
pre-compilation (for `_generated_source` / decorator detection) →
standalone wrapper → PredictStrategy nested generation → fan-out.
"""

import json
from typing import Literal

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TEST_LLM = FakeLLMClient()


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


# Standalone async function decorated with @strategy(PredictStrategy()),
# called via asyncio.gather for parallel fan-out.
CLASSIFY_BATCH_CODE = '''\
@strategy(PredictStrategy())
async def classify_one(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the sentiment of this single text."""
    ...

results = await asyncio.gather(*(classify_one(t) for t in texts))
return_result(results)
'''


class SentimentAgent(Agent, llm=_TEST_LLM):
    """Test agent for batch sentiment classification."""

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def classify(self, texts: list[str]) -> list[Literal["positive", "negative", "neutral"]]:
        """Classify the sentiment of multiple texts."""
        ...


@pytest.mark.asyncio
async def test_dynamic_structured_output_in_codeact_loop():
    """LLM defines @strategy(PredictStrategy()) method and calls it in a loop.

    This is the core batch-processing pattern: CodeAct generates a sub-method
    that delegates per-item classification to PredictStrategy, then
    loops over the batch and aggregates results.

    """
    texts = ["Great product!", "Terrible service.", "It's okay."]

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # 1. CodeAct: execute_python with classify_one definition + loop
            _resp(tool_calls=[_tool_call(CLASSIFY_BATCH_CODE)]),
            # 2-4. Predict calls for each text (value-wrapped basic type)
            _resp('{"value": "positive"}'),
            _resp('{"value": "negative"}'),
            _resp('{"value": "neutral"}'),
        ]
    )

    agent = SentimentAgent(llm=fake_llm)
    result = await agent.classify(texts)

    assert result == ["positive", "negative", "neutral"]
    assert fake_llm.call_count == 4  # 1 CodeAct + 3 Predict


@pytest.mark.asyncio
async def test_dynamic_structured_output_single_item():
    """Same pattern but with a single item — minimal case."""
    texts = ["Amazing!"]

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_tool_call(CLASSIFY_BATCH_CODE)]),
            _resp('{"value": "positive"}'),
        ]
    )

    agent = SentimentAgent(llm=fake_llm)
    result = await agent.classify(texts)

    assert result == ["positive"]
    assert fake_llm.call_count == 2


@pytest.mark.asyncio
async def test_dynamic_helper_is_not_bound_to_instance():
    """Helpers defined in CodeAct cells are not attached to the agent.

    The standalone wrapper makes `classify_one` callable as a bare name in the
    REPL scope. It must not appear on the agent instance.
    """
    texts = ["Test"]

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_tool_call(CLASSIFY_BATCH_CODE)]),
            _resp('{"value": "neutral"}'),
        ]
    )

    agent = SentimentAgent(llm=fake_llm)
    await agent.classify(texts)

    assert not hasattr(agent, "classify_one"), "classify_one must NOT be on the agent instance"
    assert not hasattr(SentimentAgent, "classify_one"), (
        "classify_one must NOT be on the agent class either"
    )


# Code that includes `from typing import Literal` — models do this by habit
CLASSIFY_BATCH_CODE_WITH_IMPORT = '''\
from typing import Literal

@strategy(PredictStrategy())
async def classify_one(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the sentiment of this single text."""
    ...

results = await asyncio.gather(*(classify_one(t) for t in texts))
return_result(results)
'''


@pytest.mark.asyncio
async def test_from_typing_import_is_allowed():
    """LLM code that starts with 'from typing import Literal' should not fail.

    Models frequently prepend typing imports by habit. Since typing is a pure
    metadata module, it should be allowed.
    """
    texts = ["Nice!"]

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_tool_call(CLASSIFY_BATCH_CODE_WITH_IMPORT)]),
            _resp('{"value": "positive"}'),
        ]
    )

    agent = SentimentAgent(llm=fake_llm)
    result = await agent.classify(texts)

    assert result == ["positive"]


# Code with bogus imports that the LLM might generate by habit
CLASSIFY_BATCH_CODE_WITH_BOGUS_IMPORTS = '''\
from typing import Literal
from strategy import PredictStrategy, strategy

@strategy(PredictStrategy())
async def classify_one(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the sentiment of this single text."""
    ...

results = await asyncio.gather(*(classify_one(t) for t in texts))
return_result(results)
'''


@pytest.mark.asyncio
async def test_redundant_imports_stripped_silently():
    """Imports of names already in scope (e.g. 'from strategy import ...') are stripped.

    LLMs write these by habit even though strategy, PredictStrategy,
    Literal etc. are all pre-loaded.  The framework should silently strip them
    rather than raising a validation error.
    """
    texts = ["Wonderful!"]

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_tool_call(CLASSIFY_BATCH_CODE_WITH_BOGUS_IMPORTS)]),
            _resp('{"value": "positive"}'),
        ]
    )

    agent = SentimentAgent(llm=fake_llm)
    result = await agent.classify(texts)

    assert result == ["positive"]
