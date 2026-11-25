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
        # No endpoint retry of the refused image: the request, the note, then
        # one confirmation (a 500 may be a one-off) that fails again.
        assert [images_sent(call.kwargs) for call in wire.call_args_list] == [1, 0, 1]
        assert (await c.acall(WITH_IMAGE)).content == "ok"
        # Kept: the next call sends the note at once.
        assert [images_sent(call.kwargs) for call in wire.call_args_list] == [1, 0, 1, 0]


UNNAMED = "Provided request is malformed."


async def test_a_one_off_unnamed_400_drops_no_image():
    images = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,aW1n{i:04d}"}}
        for i in range(4)
    ]
    messages = [{"role": "user", "content": images}]
    error = litellm.BadRequestError(message=UNNAMED, model="test", llm_provider="openai")
    wire_reply, calls = fails_first(error)
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        assert (await c.acall(messages)).content == "saw 4 images"
        assert (await c.acall(messages)).content == "saw 4 images"
    # Failed, the probe without media, the newest image excluded, then the
    # confirmation (the original unchanged) succeeds: its reply, nothing
    # excluded, and the next call sends all four.
    assert calls == [4, 0, 3, 4, 4]


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
async def test_a_one_off_media_worded_500_keeps_the_image(asynchronous):
    # #597 re-review P1: "Could not process image" once, then fine.
    wire_reply, calls = fails_first(server_error("Could not process image"))
    with client() as c, _patch_wire(asynchronous) as wire:
        wire.side_effect = wire_reply
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "saw 1 images"
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "saw 1 images"
    # The 500, the note (succeeds), the confirmation with the image (succeeds:
    # a one-off, its reply is returned, nothing kept), the next call keeps it.
    assert calls == [1, 0, 1, 1]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_persistent_media_worded_500_drops_the_image(asynchronous):
    def wire_reply(**kwargs):
        if images_sent(kwargs):
            raise server_error("Could not process image")
        return reply("ok")

    with client() as c, _patch_wire(asynchronous) as wire:
        wire.side_effect = wire_reply
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "ok"
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "ok"
        assert [images_sent(call.kwargs) for call in wire.call_args_list] == [1, 0, 1, 0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("text", [UNNAMED, "Request contains an invalid argument."])
async def test_a_persistent_unnamed_400_costs_exactly_one_extra_request(asynchronous, text):
    error = litellm.BadRequestError(message=text, model="test", llm_provider="openai")
    wire_reply, calls = fails_first(error, times=100)
    with client() as c, _patch_wire(asynchronous) as wire:
        wire.side_effect = wire_reply
        for _ in range(2):
            with pytest.raises(litellm.BadRequestError):
                await _run(c, WITH_IMAGE, asynchronous)
    # Each call: the request, then the probe without media; never a re-send.
    assert calls == [1, 0, 1, 0]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_one_off_unnamed_400_with_one_image_commits_nothing(asynchronous):
    error = litellm.BadRequestError(message=UNNAMED, model="test", llm_provider="openai")
    wire_reply, calls = fails_first(error)
    with client() as c, _patch_wire(asynchronous) as wire:
        wire.side_effect = wire_reply
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "saw 1 images"
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "saw 1 images"
    assert calls == [1, 0, 1, 1]  # the probe, the confirmation wins, the image stays


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_named_4xx_is_definitive_and_not_confirmed(asynchronous):
    error = litellm.BadRequestError(
        message="Unsupported image format: svg+xml", model="test", llm_provider="openai"
    )

    def wire_reply(**kwargs):
        if images_sent(kwargs):
            raise error
        return reply("ok")

    with client() as c, _patch_wire(asynchronous) as wire:
        wire.side_effect = wire_reply
        assert (await _run(c, WITH_IMAGE, asynchronous)).content == "ok"
        assert [images_sent(call.kwargs) for call in wire.call_args_list] == [1, 0]
