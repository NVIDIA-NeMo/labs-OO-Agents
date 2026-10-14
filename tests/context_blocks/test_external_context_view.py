# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end seam test for a context system outside ``nooa``."""

import pytest

import nooa.agent as agent_module
import nooa.runtime.method_wrapper as method_wrapper
from nooa import Agent, strategy
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse
from tests.context_blocks.fixtures.external_context_view import ExternalContextView


class PoisonContextManager:
    """Fail if runtime code touches the legacy manager behind a custom view."""

    def __getattr__(self, name):
        raise AssertionError(f"custom context view unexpectedly accessed {name}")


def _predict_response(value: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=f'{{"value": "{value}"}}',
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": value},
        usage={"prompt_tokens": 5, "completion_tokens": 1},
    )


@pytest.mark.asyncio
async def test_external_view_completes_predict_and_codeact_calls(monkeypatch):
    async def no_flush():
        pass

    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", True)
    monkeypatch.setattr(method_wrapper, "_flush_litellm_journal", no_flush)
    predict_llm = FakeLLMClient(scripted_responses=[_predict_response("predicted")])

    class PredictAgent(Agent, llm=predict_llm, context_view=ExternalContextView()):
        @strategy(PredictStrategy())
        async def run(self) -> str: ...

    predict_agent = PredictAgent()
    predict_agent.context_manager = PoisonContextManager()
    assert await predict_agent.run() == "predicted"
    assert "external context for run on fake-model" in str(predict_llm.last_messages)

    code_llm = FakeLLMClient.with_tool_call("return_result", {"result": "executed"})

    class CodeAgent(Agent, llm=code_llm, context_view=ExternalContextView()):
        async def run(self) -> str: ...

    code_agent = CodeAgent()
    code_agent.context_manager = PoisonContextManager()
    assert await code_agent.run() == "executed"
    assert "external context for run on fake-model" in str(code_llm.last_messages)
