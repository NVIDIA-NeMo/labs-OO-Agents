# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral replay of persisted plain reasoning."""

from unittest.mock import patch

from litellm.types.utils import Choices, Message, ModelResponse

from nooa.context_blocks.formatter import (
    OpenAIProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import ResolvedBlock, Role
from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient


def _render(response: LLMResponse, *, responses: bool = False) -> list[dict]:
    neutral = XMLBlockFormatter().format(
        [ResolvedBlock(key="turn", content=response.content, role=Role.ASSISTANT, event=response)]
    )
    formatter = ResponsesProviderFormatter() if responses else OpenAIProviderFormatter()
    return formatter.format(neutral)


def _chat_response() -> ModelResponse:
    return ModelResponse(
        model="test-model",
        choices=[Choices(message=Message(role="assistant", content="done"), finish_reason="stop")],
    )


def test_reasoning_backed_structured_output_replays_after_persistence() -> None:
    response = LLMResponse(
        content="",
        parsed={"value": "positive"},
        reasoning='{"value":"positive"}',
    )
    restored = LLMResponse.model_validate_json(response.model_dump_json())
    rendered = _render(restored)

    # The renderer exposes public JSON plus identity, never native state.
    assert rendered[-1] == {**restored.public_message(), "nooa_turn": restored.id}

    client = CompletionClient(model="openai/gpt-4o")
    try:
        with patch("litellm.completion", return_value=_chat_response()) as completion:
            client.call(rendered, turns={restored.id: restored})

        assert restored.parsed is None
        assert restored.content == ""
        assert restored.reasoning == '{"value":"positive"}'
        assistant = next(
            message
            for message in completion.call_args.kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert assistant == {"role": "assistant", "content": '{"value":"positive"}'}
    finally:
        client.close()


def test_reasoning_only_response_demotes_for_responses_api() -> None:
    response = LLMResponse(content="", reasoning="portable thought")
    client = ResponsesClient(model="openai/gpt-5")
    try:
        transformed, instructions = client._transform_messages(_render(response, responses=True))
    finally:
        client.close()

    assert instructions is None
    assert transformed == [{"role": "assistant", "content": "portable thought"}]


def test_opaque_only_response_is_withheld_without_a_provider_gate() -> None:
    response = LLMResponse.model_validate(
        {"content": "", "llm_state": {"opaque": "provider state"}}, context={"archive": True}
    )
    client = CompletionClient(model="openai/gpt-4o")
    try:
        with patch("litellm.completion", return_value=_chat_response()) as completion:
            client.call(_render(response))

        assert all(
            message.get("role") != "assistant"
            for message in completion.call_args.kwargs["messages"]
        )
        assert "provider state" not in repr(completion.call_args.kwargs["messages"])
    finally:
        client.close()
