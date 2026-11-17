# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 item 1: more provider wordings count as media rejections, narrowly and by status.

The fixtures follow the wording in the provider documentation and the #597
issue/reviews. Where a wording was not captured from a live provider the
test name or comment says so ("docs" / "review"); none of these strings was
checked against a live endpoint.
"""

import litellm
import pytest

from nooa.unifiedllm import RetryConfig
from nooa.unifiedllm.media_inputs import (
    explicitly_rejected,
    is_media_rejection,
    rejected_media_kinds,
)
from nooa.unifiedllm.retry import _is_retryable_error


def bad_request(text, body=None):
    exc = litellm.BadRequestError(message=text, model="test", llm_provider="anthropic")
    if body is not None:
        exc.body = body
    return exc


def server_error(text):
    return litellm.InternalServerError(message=text, model="test", llm_provider="openai")


class StatusError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message}}


# Anthropic docs/issue wording: per-image size and dimension limits.
ANTHROPIC_SIZE = (
    'AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":'
    '"messages.0.content.1.image.source.base64: image exceeds 5 MB maximum: 7340032 bytes > '
    '5242880 bytes"}}'
)
ANTHROPIC_DIMENSIONS = (
    'AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":'
    '"messages.0.content.1.image.source.base64.data: At least one of the image dimensions '
    'exceed max allowed size: 8000 pixels"}}'
)
# Issue wording: Anthropic's schema error for a media_type it does not take.
ANTHROPIC_MEDIA_TYPE = (
    "AnthropicException - messages.0.content.1.image.source.base64.media_type: Input should be "
    "'image/jpeg', 'image/png', 'image/gif' or 'image/webp'"
)
# Review wording (not captured live): OpenAI's audio format error and its param.
OPENAI_AUDIO_FORMAT = "Invalid value: 'mpeg'. Supported values are: 'wav' and 'mp3'."
OPENAI_AUDIO_PARAM = "messages.[1].content.[1].input_audio.format"


@pytest.mark.parametrize(
    "text,kind",
    [
        (ANTHROPIC_SIZE, "image"),
        (ANTHROPIC_DIMENSIONS, "image"),
        (ANTHROPIC_MEDIA_TYPE, "image"),
        ("Could not process image", "image"),
        ("Failed to decode image: invalid PNG header", "image"),
        # Review wording (Gemini-style, unverified).
        ("Unsupported MIME type: image/gif", "image"),
        ("audio format mpeg is not supported", "audio"),
        # Anthropic docs wording for PDF limits and bad PDFs (unverified live).
        ("A maximum of 100 PDF pages may be provided.", "pdf"),
        ("The PDF specified was not valid.", "pdf"),
        ("PDF is password protected", "pdf"),
        ("The document is encrypted and cannot be read", "pdf"),
        ("Invalid PDF: corrupt cross-reference table", "pdf"),
        ("corrupt pdf", "pdf"),
    ],
)
def test_docs_and_review_wordings_are_rejections_at_400(text, kind):
    assert explicitly_rejected(bad_request(text), kind)


def test_openai_audio_format_error_is_an_audio_rejection_by_its_param():
    body = {"error": {"message": OPENAI_AUDIO_FORMAT, "param": OPENAI_AUDIO_PARAM}}
    exc = bad_request(OPENAI_AUDIO_FORMAT, body)
    assert rejected_media_kinds(exc) == {"audio"}
    # The same field path in the message alone also counts.
    assert explicitly_rejected(
        bad_request(f"{OPENAI_AUDIO_FORMAT} ({OPENAI_AUDIO_PARAM})"), "audio"
    )


def test_413_counts_only_with_media_wording():
    assert explicitly_rejected(StatusError(413, "image too large"), "image")
    assert not is_media_rejection(StatusError(413, "Request Entity Too Large"))
    assert not is_media_rejection(StatusError(413, "request_too_large: the request is too big"))
    # A 413 never counts for the strict "invalid image" wording, only for size.
    assert not is_media_rejection(StatusError(413, "invalid image"))


@pytest.mark.parametrize(
    "text",
    [
        "BedrockException - Could not process image",  # Bedrock/Hub wording (issue)
        "Unable to decode image data",
        ANTHROPIC_SIZE,
    ],
)
def test_media_worded_500_is_a_rejection_and_not_retried(text):
    exc = server_error(text)
    assert explicitly_rejected(exc, "image")
    assert _is_retryable_error(exc, RetryConfig()) == (False, False)


@pytest.mark.parametrize(
    "exc",
    [
        server_error("Internal server error"),
        server_error("Image service temporarily unavailable, try again later"),
        StatusError(502, "Bad gateway while processing image"),
        StatusError(503, "Could not process image: backend overloaded"),
        StatusError(504, "Gateway timeout"),
        server_error("Router misconfiguration: unsupported image format conversion plugin"),
    ],
)
def test_generic_5xx_stays_retryable_and_not_a_rejection(exc):
    assert not is_media_rejection(exc)
    assert _is_retryable_error(exc, RetryConfig())[0]


def test_429_timeouts_and_network_errors_are_unchanged():
    rate = litellm.RateLimitError(message="rate limit", model="m", llm_provider="openai")
    assert not is_media_rejection(rate)
    assert _is_retryable_error(rate, RetryConfig()) == (True, True)
    timeout = litellm.Timeout(message="Request timed out", model="m", llm_provider="openai")
    assert not is_media_rejection(timeout)
    assert _is_retryable_error(timeout, RetryConfig())[0]
    conn = litellm.APIConnectionError(message="Connection reset", model="m", llm_provider="openai")
    assert not is_media_rejection(conn)
    assert _is_retryable_error(conn, RetryConfig())[0]


@pytest.mark.parametrize(
    "text",
    [
        "This model's maximum context length is 8192 tokens; your images resulted in 9000 tokens",
        "prompt is too long: 210000 tokens > 200000 maximum (images exceed the budget)",
        "Too many images: 7 (max 5)",
        "Invalid value for 'temperature': expected a number.",
        "max_tokens: 100000 > 64000, which is the maximum allowed",
    ],
)
def test_request_wide_and_unrelated_400s_are_not_media_rejections(text):
    assert not is_media_rejection(bad_request(text))


def test_local_bedrock_value_error_is_a_format_rejection():
    # LiteLLM's Bedrock adapter (factory.py BedrockImageProcessor._validate_format)
    # raises this before any request when it is not wrapped in a status error.
    exc = ValueError("Unsupported image format: svg+xml. Supported formats: ['png', 'jpeg']")
    assert rejected_media_kinds(exc) == {"image"}
    assert not is_media_rejection(ValueError("Could not process image"))
    assert not is_media_rejection(RuntimeError("Unsupported image format: svg+xml."))
    # Wrapped by LiteLLM's generic exception mapping it arrives as a 500 APIConnectionError.
    wrapped = litellm.APIConnectionError(
        message="litellm.APIConnectionError: Unsupported image format: svg+xml. Supported "
        "formats: ['png', 'jpeg']",
        model="m",
        llm_provider="bedrock",
    )
    assert explicitly_rejected(wrapped, "image")
    assert _is_retryable_error(wrapped, RetryConfig()) == (False, False)


@pytest.mark.parametrize(
    "text",
    [
        "Could not process image: upstream request timed out",
        "Could not process image: connection reset by peer",
        "Failed to decode image: upstream unavailable",
        "image processing backend overloaded",
        "Unsupported image format: png. The service is temporarily degraded",
        "Could not process image, please try again later",
    ],
)
@pytest.mark.parametrize("status", [400, 500])
def test_transient_wording_is_never_a_media_rejection(text, status):
    exc = bad_request(text) if status == 400 else server_error(text)
    assert not is_media_rejection(exc)
    if status == 500:
        assert _is_retryable_error(exc, RetryConfig())[0]


@pytest.mark.parametrize(
    "text",
    [
        "Failed to load the file",
        "could not read file /models/config.json",
        "Failed to parse the document",
        "corrupted file in cache",
        "The file is too large to cache",
        "Error while downloading model weights",
    ],
)
def test_generic_file_wording_in_a_500_is_not_a_media_rejection(text):
    assert not is_media_rejection(server_error(text))
    assert _is_retryable_error(server_error(text), RetryConfig())[0]


def test_an_explicit_pdf_in_a_500_still_counts():
    assert rejected_media_kinds(server_error("Failed to load the PDF")) == {"pdf"}
    assert rejected_media_kinds(server_error("The PDF is corrupted")) == {"pdf"}


def test_a_bracketed_non_media_param_wins_over_media_words():
    text = "Invalid schema for function image_url: expected an object"
    body = {"error": {"message": text, "param": "tools[0].function.parameters"}}
    assert not is_media_rejection(bad_request(text, body))
    # Without the param, a bare field name (a function name) is no content path.
    assert not is_media_rejection(bad_request(text))
    assert not is_media_rejection(bad_request("Function 'input_audio' is not defined"))
    # A media field inside a content path still counts.
    path = "messages[0].content[1].image_url.url"
    assert rejected_media_kinds(bad_request(f"Invalid value at {path}")) == {"image"}
    body = {"error": {"message": "Invalid value", "param": path}}
    assert rejected_media_kinds(bad_request("Invalid value", body)) == {"image"}
