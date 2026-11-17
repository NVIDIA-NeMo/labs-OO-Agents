# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 reviews: a transient failure never drops a good attachment for good.

The provider is a fake (``litellm.acompletion`` patched); no request leaves
the process.
"""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.media_inputs import attachment_kind

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1nMDAwMA=="}}


@pytest.fixture(autouse=True)
def no_sleep():
    with (
        patch("nooa.unifiedllm.retry.asyncio.sleep", new_callable=AsyncMock),
        patch("nooa.unifiedllm.retry.time.sleep"),
    ):
        yield


def client(max_retries=3):
    return CompletionClient(
        "openai/test",
        media_inputs=["image", "pdf"],
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=max_retries),
    )


def server_error(text):
    return litellm.InternalServerError(message=text, model="test", llm_provider="openai")


def reply(text):
    return litellm.ModelResponse(choices=[{"message": {"content": text}}])


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
        return reply(f"saw {calls[-1]} images")

    return wire, calls


TEXT_ONLY = [{"role": "user", "content": "hello"}]
WITH_IMAGE = [{"role": "user", "content": [{"type": "text", "text": "Look:"}, IMAGE]}]


@pytest.mark.parametrize(
    "text", ["Failed to load the file", "Could not process image", "The PDF is corrupted"]
)
async def test_a_media_worded_500_on_a_text_only_request_is_retried(text):
    wire_reply, calls = fails_first(server_error(text))
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        assert (await c.acall(TEXT_ONLY)).content == "saw 0 images"
    assert len(calls) == 2  # the 500, then the endpoint retry


async def test_an_image_500_with_a_timeout_is_retried_and_the_image_kept():
    error = server_error("Could not process image: upstream request timed out")
    wire_reply, calls = fails_first(error)
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        assert (await c.acall(WITH_IMAGE)).content == "saw 1 images"
        assert (await c.acall(WITH_IMAGE)).content == "saw 1 images"
    assert calls == [1, 1, 1]  # retried with the image; the next call keeps it


async def test_an_image_500_on_a_request_with_an_image_goes_to_media_recovery():
    def wire_reply(**kwargs):
        if images_sent(kwargs):
            raise server_error("Could not process image")
        return reply("ok")

    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        assert (await c.acall(WITH_IMAGE)).content == "ok"
        # No endpoint retry of the refused image: the request, then the note.
        assert [images_sent(call.kwargs) for call in wire.call_args_list] == [1, 0]
