# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider's explicit rejection of one attachment becomes a note, not a failed turn."""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.media_inputs import attachment_kind, explicitly_rejected

# Rejection bodies captured from HCB I01.corrupt_image (2026-10-29), as the
# client receives them through the LiteLLM proxy.
GPT = (
    'OpenAIException - {\n  "error": {\n    "message": "The image data you provided does not '
    'represent a valid image. Please check your input and try again.",\n    "type": '
    '"invalid_request_error",\n    "param": "input",\n    "code": "invalid_value"\n  }\n}\n\n'
    "LiteLLM: model group 'openai/openai/gpt-6.1-sol' failed with the error above."
)
BEDROCK = (
    'BedrockException - {"message":"The model returned the following errors: Could not process '
    "image\"}\n\nLiteLLM: model group 'aws/anthropic/bedrock-claude-opus-5-5' failed with the "
    "error above. No fallback was attempted."
)
GLM = (
    'OpenAIException - {\n\t"error": {\n\t\t"code": "400",\n\t\t"message": "Media could not be '
    "processed (400): xInvalid media for url: \\u003curl\\u003e Failed to load image: image file "
    'is truncated\\" }",\n\t\t"type": "Bad Request"\n\t}\n}'
)
GENERIC = "OpenAIException - Invalid value for 'temperature': expected a number."
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
OTHER = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgp="}}
NOTE = "[An image was removed here because the model provider could not read it.]"


def bad_request(text):
    return litellm.BadRequestError(message=text, model="test", llm_provider="openai")


def ok(responses):
    if not responses:
        return litellm.ModelResponse(choices=[{"message": {"content": "UNREADABLE"}}])
    from types import SimpleNamespace

    return SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "UNREADABLE"}]}],
        output_text="UNREADABLE",
        status="completed",
        usage=None,
    )


def sent_blocks(call, responses):
    messages = call.kwargs["input" if responses else "messages"]
    return [b for m in messages if isinstance(m.get("content"), list) for b in m["content"]]


def images(blocks):
    return [b for b in blocks if attachment_kind(b) == "image"]


def texts(blocks):
    return " ".join(b.get("text", "") for b in blocks)


@pytest.mark.parametrize("text", [GPT, BEDROCK, GLM])
def test_real_provider_phrasings_are_explicit_image_rejections(text):
    assert explicitly_rejected(bad_request(text), "image")


@pytest.mark.parametrize(
    "error",
    [
        bad_request(GENERIC),
        bad_request("Invalid model: vision-model does not exist"),
        litellm.ServiceUnavailableError(
            message="The model is temporarily unavailable.", model="test", llm_provider="openai"
        ),
    ],
)
def test_generic_errors_are_not_media_rejections(error):
    assert not explicitly_rejected(error, "image")


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_rejected_image_is_retried_once_as_a_note(client_type, asynchronous):
    responses = client_type is ResponsesClient
    method = ("a" if asynchronous else "") + ("responses" if responses else "completion")
    messages = [{"role": "user", "content": [{"type": "text", "text": "Read this:"}, IMAGE]}]

    async def send(client):
        return await client.acall(messages) if asynchronous else client.call(messages)

    with client_type("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire:
            wire.side_effect = [bad_request(GPT), ok(responses)]
            assert (await send(client)).content == "UNREADABLE"
            first, second = (sent_blocks(call, responses) for call in wire.call_args_list)
            assert len(images(first)) == 1 and NOTE not in texts(first)
            assert images(second) == [] and NOTE in texts(second)
        with patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire:
            # The same attachment does not fail again on this connection.
            wire.return_value = ok(responses)
            await send(client)
            assert wire.call_count == 1
            assert images(sent_blocks(wire.call_args, responses)) == []


async def test_generic_400_still_fails_without_a_retry():
    messages = [{"role": "user", "content": [IMAGE]}]
    with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
            wire.side_effect = bad_request(GENERIC)
            with pytest.raises(litellm.BadRequestError):
                await client.acall(messages)
            assert wire.call_count == 1


async def test_newest_attachment_goes_first_then_history_media():
    history = {"role": "user", "content": [OTHER]}
    fresh = {"role": "user", "content": [IMAGE]}
    messages = [history, {"role": "assistant", "content": "seen"}, fresh]
    with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
            wire.side_effect = [bad_request(BEDROCK), bad_request(BEDROCK), ok(False)]
            await client.acall(messages)
            sent = [call.kwargs["messages"] for call in wire.call_args_list]
            assert [len(images(m[0]["content"] + m[2]["content"])) for m in sent] == [2, 1, 0]
            assert images(sent[1][0]["content"]) == [OTHER]  # the newest went first


async def test_exclusion_is_cleared_when_the_connection_changes():
    messages = [{"role": "user", "content": [IMAGE]}]
    with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
            wire.side_effect = [bad_request(GLM), ok(False), ok(False), ok(False)]
            await client.acall(messages)
            await client.acall(messages, base_url="https://replacement.test/v1")
            # Back on the original connection: the image is tried again.
            await client.acall(messages)
            sent = [sent_blocks(call, False) for call in wire.call_args_list]
            assert [len(images(blocks)) for blocks in sent] == [1, 0, 0, 1]
            assert NOTE in texts(sent[1])
            assert NOTE not in texts(sent[2])  # a replacement route has its own permissions
