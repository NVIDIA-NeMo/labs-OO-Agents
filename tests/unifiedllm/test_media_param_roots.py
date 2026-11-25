# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 re-review: a structured error ``param`` outside the conversation is never media.

The provider is a fake (``litellm.completion``/``acompletion`` patched); no
request leaves the process.
"""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.media_inputs import attachment_kind

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1nMDAwMA=="}}
WITH_IMAGE = [{"role": "user", "content": [{"type": "text", "text": "Look:"}, IMAGE]}]


@pytest.fixture(autouse=True)
def no_sleep():
    with (
        patch("nooa.unifiedllm.retry.asyncio.sleep", new_callable=AsyncMock),
        patch("nooa.unifiedllm.retry.time.sleep"),
    ):
        yield


def client():
    return CompletionClient(
        "openai/test",
        media_inputs=["image"],
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=3),
    )


def images_sent(kwargs):
    return sum(
        1
        for m in kwargs["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if attachment_kind(b) == "image"
    )


def fails_first(error, times=1):
    calls = []

    def wire(**kwargs):
        calls.append(images_sent(kwargs))
        if len(calls) <= times:
            raise error
        return litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])

    return wire, calls


def _run(c, messages, asynchronous):
    if asynchronous:
        return c.acall(messages)

    async def done():
        return c.call(messages)

    return done()


def _patch_wire(asynchronous):
    if asynchronous:
        return patch("litellm.acompletion", new_callable=AsyncMock)
    return patch("litellm.completion")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_tool_schema_param_named_image_url_is_not_a_media_rejection(asynchronous):
    # #597 re-review P2: the param root (tools) wins over a media-named descendant.
    from nooa.unifiedllm.media_inputs import rejected_media_kinds
    from nooa.unifiedllm.unifiedllm import _media_fallback_eligible

    error = litellm.BadRequestError(
        message="Invalid schema: expected an object", model="test", llm_provider="openai"
    )
    error.param = "tools[0].function.parameters.properties.image_url"
    body = {"error": {"message": "Invalid schema: expected an object", "param": error.param}}
    error.body = body
    assert rejected_media_kinds(error) == frozenset()
    assert not _media_fallback_eligible(error)
    wire_reply, calls = fails_first(error, times=100)
    with client() as c, _patch_wire(asynchronous) as wire:
        wire.side_effect = wire_reply
        with pytest.raises(litellm.BadRequestError):
            await _run(c, WITH_IMAGE, asynchronous)
    assert calls == [1]  # no recovery request at all


@pytest.mark.parametrize(
    "param", ["messages[0].content[1].image_url", "image_url", "input[0].content[1].image_url"]
)
def test_a_content_path_or_bare_media_param_still_counts(param):
    from nooa.unifiedllm.media_inputs import rejected_media_kinds

    error = litellm.BadRequestError(message="Invalid value", model="test", llm_provider="openai")
    error.param = param
    error.body = {"error": {"message": "Invalid value", "param": param}}
    assert rejected_media_kinds(error) == frozenset({"image"})


@pytest.mark.parametrize(
    "param",
    ["tool_choice.image_url", "response_format.json_schema.image_url", "functions[0].image_url"],
)
def test_other_param_roots_never_count(param):
    from nooa.unifiedllm.media_inputs import rejected_media_kinds

    error = litellm.BadRequestError(
        message="Unsupported image format: svg+xml", model="test", llm_provider="openai"
    )
    error.param = param
    error.body = {"error": {"message": "Unsupported image format: svg+xml", "param": param}}
    assert rejected_media_kinds(error) == frozenset()
