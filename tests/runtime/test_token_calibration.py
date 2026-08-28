# SPDX-License-Identifier: Apache-2.0
"""Provider usage is passive telemetry and never drives runtime control flow."""

import pytest

from nooa import Agent
from nooa.events import LLMComplete, Message
from nooa.unifiedllm import FakeLLMClient, LLMResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_tokens": 12345, "completion_tokens": 7}, (12345, 7)),
        (None, (0, 0)),
    ],
)
async def test_provider_usage_is_passive_telemetry(usage, expected):
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                content="ok",
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": "ok"},
                usage=usage,
            )
        ]
    )

    class A(Agent, llm=llm):
        async def respond(self, prompt: str) -> str:
            """Respond."""
            ...

    agent = A()
    agent.event_manager.add(Message(content="small event"))
    seen = []
    agent.event_manager.on("LLMComplete", lambda event: seen.append(event))
    assert await agent.respond("hi") == "ok"
    complete = next(event for event in seen if isinstance(event, LLMComplete))
    assert (complete.prompt_tokens, complete.completion_tokens) == expected
    assert not hasattr(agent.runtime, "_last_prompt_tokens_actual")
    assert not hasattr(agent.runtime, "_tokens_per_char")
    assert not hasattr(agent.runtime._last_context_stats, "prompt_tokens")
