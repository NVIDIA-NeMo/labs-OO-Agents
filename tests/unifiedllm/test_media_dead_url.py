# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 review missed case 1: a dead remote image URL must not fail every later request."""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.media_inputs import attachment_kind, rejected_media_kinds, sent_media_kinds
from nooa.unifiedllm.retry import _is_retryable_error


def retryable(exc, sent=frozenset({"image"})):
    """``_is_retryable_error`` for a request that sent media of the ``sent`` kinds."""
    token = sent_media_kinds.set(frozenset(sent))
    try:
        return _is_retryable_error(exc, RetryConfig())
    finally:
        sent_media_kinds.reset(token)


DEAD = {"type": "image_url", "image_url": {"url": "https://example.invalid/gone.png"}}
NOTE = "[An image was removed here because the model provider could not read it.]"


def fetch_error(
    text="Error: Unable to fetch image from URL. Status code: 404, url=https://x/a.png",
):
    # Raised by LiteLLM's image_handling.convert_url_to_base64 (installed source).
    return litellm.ImageFetchError(text)


def test_image_fetch_error_is_an_image_rejection():
    exc = fetch_error()
    assert exc.status_code == 400
    assert rejected_media_kinds(exc) == {"image"}
    assert rejected_media_kinds(fetch_error("Error: Image size (60.00MB) exceeds maximum")) == {
        "image"
    }


@pytest.mark.parametrize(
    "exc",
    [
        # Review wording for OpenAI's own server-side fetch (not captured live).
        litellm.BadRequestError(
            message="Error while downloading https://example.invalid/gone.png.",
            model="m",
            llm_provider="openai",
        ),
        # The same LiteLLM error passed through the Hub proxy as a 500.
        litellm.InternalServerError(
            message="litellm.ImageFetchError: Error: Unable to fetch image from URL after 3 "
            "attempts. url=https://example.invalid/gone.png",
            model="m",
            llm_provider="openai",
        ),
    ],
)
def test_download_failures_are_image_rejections_and_not_retried(exc):
    assert rejected_media_kinds(exc) == {"image"}
    assert retryable(exc) == (False, False)


@pytest.mark.parametrize(
    "text",
    [
        "Model weights are downloading; temporarily unavailable",
        # Not about an image or a URL: the server's own download.
        "Error while downloading model weights",
        # A timeout may pass: retried, the image is kept.
        "Timeout while downloading https://example.invalid/slow.png.",
    ],
)
def test_unrelated_or_transient_downloads_are_not_rejections(text):
    exc = litellm.InternalServerError(message=text, model="m", llm_provider="openai")
    assert not rejected_media_kinds(exc)
    assert _is_retryable_error(exc, RetryConfig())[0]
    assert not rejected_media_kinds(
        litellm.BadRequestError(message=text, model="m", llm_provider="openai")
    )


async def test_dead_url_is_replaced_by_a_note_and_stays_replaced():
    messages = [{"role": "user", "content": [{"type": "text", "text": "Look:"}, DEAD]}]

    def wire(**kwargs):
        blocks = [b for m in kwargs["messages"] for b in m["content"]]
        if any(attachment_kind(b) == "image" for b in blocks):
            raise fetch_error()
        return litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])

    with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = wire
            assert (await client.acall(messages)).content == "ok"
            assert (await client.acall(messages)).content == "ok"
        sent = [
            b for call in mock.call_args_list for m in call.kwargs["messages"] for b in m["content"]
        ]
        assert mock.call_count == 3  # one refused request, then the note every time
        assert sum(1 for b in sent if b.get("text") == NOTE) == 2
