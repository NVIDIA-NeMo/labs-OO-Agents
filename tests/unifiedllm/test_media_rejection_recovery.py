# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider's explicit rejection of one attachment becomes a note, not a failed turn."""

import json
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


async def test_rejected_image_retry_is_a_harness_metric():
    from nooa.runtime.actor import _make_llm_metrics_bridge
    from nooa.runtime.harness_metrics import HarnessMetrics
    from nooa.unifiedllm.unifiedllm import _llm_metrics_callback

    metrics = HarnessMetrics()
    token = _llm_metrics_callback.set(_make_llm_metrics_bridge(metrics))
    try:
        messages = [{"role": "user", "content": [IMAGE]}]
        with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as c:
            with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
                wire.side_effect = [bad_request(GPT), ok(False)]
                await c.acall(messages)
    finally:
        _llm_metrics_callback.reset(token)
    attrs = metrics.to_span_attributes()
    assert attrs["harness.media.rejected_retry.count"] == 1
    assert attrs["harness.media.rejected_retry.kinds"] == ["image"]


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


def numbered_images(count):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,aW1n{i:04d}"}}
        for i in range(count)
    ]


@pytest.mark.parametrize("max_retries", [1, 3])
async def test_an_unnamed_bad_older_image_never_costs_good_images_on_failure(max_retries):
    """Bedrock names no image. The bad one is the oldest of six."""
    from nooa.unifiedllm import RetryConfig

    bad, *good = numbered_images(6)
    messages = [{"role": "user", "content": [block]} for block in (bad, *good)]

    def wire_reply(*args, **kwargs):
        sent = [b for m in kwargs["messages"] for b in m["content"]]
        if bad in sent:
            raise bad_request(BEDROCK)
        return ok(False)

    with CompletionClient(
        "openai/test",
        media_inputs=["image"],
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=max_retries),
    ) as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
            wire.side_effect = wire_reply
            if max_retries == 1:
                # One retry (the newest image) is not enough: the request fails
                # and no good image stays replaced for later requests.
                with pytest.raises(litellm.BadRequestError):
                    await client.acall(messages)
                assert wire.call_count == 2
                view = client._prepare_media_messages(messages, client._prepare_call_config({}))
                assert NOTE not in texts([b for m in view for b in m["content"]])
            else:
                # Newest first, then every remaining image at once: it succeeds.
                assert (await client.acall(messages)).content == "UNREADABLE"
                sent = [sent_blocks(call, False) for call in wire.call_args_list]
                assert [len(images(blocks)) for blocks in sent] == [6, 5, 0]
                assert texts(sent[-1]).count(NOTE) == 6


def test_a_still_failing_sync_request_undoes_its_exclusions():
    from nooa.unifiedllm import RetryConfig

    messages = [{"role": "user", "content": [IMAGE]}]
    with CompletionClient(
        "openai/test",
        media_inputs=["image"],
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=1),
    ) as client:
        with patch("litellm.completion") as wire:
            wire.side_effect = [bad_request(BEDROCK), bad_request(GENERIC)]
            with pytest.raises(litellm.BadRequestError):
                client.call(messages)
        view = client._prepare_media_messages(messages, client._prepare_call_config({}))
        assert images(view[0]["content"]) == [IMAGE]


# Hub (LiteLLM proxy to Bedrock Claude), 2026-11-16: an SVG from show() came back
# as HTTP 500, so the existing 400-only recovery never ran and every later turn
# failed on the same image.
HUB_SVG = (
    "OpenAIException - litellm.APIConnectionError: Unsupported image format: svg+xml. "
    "Supported formats: ['mp4', 'mov', 'mkv', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', '3gp', "
    "'png', 'jpeg', 'gif', 'webp']"
)


def server_error(text):
    return litellm.InternalServerError(message=text, model="test", llm_provider="openai")


def test_unsupported_format_500_is_an_explicit_rejection():
    assert explicitly_rejected(server_error(HUB_SVG), "image")
    assert not explicitly_rejected(server_error(HUB_SVG), "pdf")


def test_other_500s_are_not_media_rejections():
    assert not explicitly_rejected(server_error("Could not process image"), "image")
    assert not explicitly_rejected(server_error("Internal server error"), "image")


def server_error_with_body(message):
    exc = server_error("Internal Server Error")
    exc.body = {"error": {"message": message}}
    return exc


def test_the_500_gate_reads_the_structured_error_message():
    assert explicitly_rejected(server_error_with_body(HUB_SVG), "image")
    pdf = "Unsupported document format: docx. Supported formats: ['pdf']"
    assert explicitly_rejected(server_error(pdf), "pdf")
    assert explicitly_rejected(server_error("unsupported pdf type 'application/x-pdf'"), "pdf")


def test_a_500_that_only_mentions_a_format_is_not_a_rejection():
    text = "Router misconfiguration: unsupported image format conversion plugin is unavailable"
    assert not explicitly_rejected(server_error(text), "image")


def test_an_unsupported_format_500_is_not_retried_by_the_endpoint_retry_layer():
    from nooa.unifiedllm import RetryConfig
    from nooa.unifiedllm.retry import _is_retryable_error

    assert _is_retryable_error(server_error(HUB_SVG), RetryConfig()) == (False, False)
    assert _is_retryable_error(server_error("Internal server error"), RetryConfig()) == (
        True,
        False,
    )


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unsupported_format_500_is_sent_twice_then_a_note(client_type, asynchronous):
    responses = client_type is ResponsesClient
    method = ("a" if asynchronous else "") + ("responses" if responses else "completion")
    messages = [{"role": "user", "content": [{"type": "text", "text": "Read this:"}, IMAGE]}]

    def hub(**kwargs):
        # Like the Hub: every attempt that still carries the image fails the same way.
        sent = kwargs["input" if responses else "messages"]
        if any(images(m["content"]) for m in sent if isinstance(m.get("content"), list)):
            raise server_error(HUB_SVG)
        return ok(responses)

    with client_type("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with (
            patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire,
            patch("nooa.unifiedllm.retry.asyncio.sleep", new_callable=AsyncMock),
            patch("nooa.unifiedllm.retry.time.sleep"),
        ):
            wire.side_effect = hub
            if asynchronous:
                assert (await client.acall(messages)).content == "UNREADABLE"
            else:
                assert client.call(messages).content == "UNREADABLE"
            assert wire.call_count == 2  # no unchanged resend of the refused image
            last = sent_blocks(wire.call_args_list[-1], responses)
            assert images(last) == [] and NOTE in texts(last)


SVG = {"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,PHN2Zy8+"}}


async def test_an_image_already_replaced_by_the_filter_is_never_excluded():
    # Bedrock names no image; the corrupt PNG is the newest one actually sent.
    good, other, corrupt = numbered_images(3)
    messages = [{"role": "user", "content": [block]} for block in (good, other, corrupt, SVG)]

    def wire_reply(*args, **kwargs):
        sent = [b for m in kwargs["messages"] for b in m["content"]]
        if corrupt in sent:
            raise bad_request(BEDROCK)
        return ok(False)

    with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
            wire.side_effect = wire_reply
            assert (await client.acall(messages)).content == "UNREADABLE"
            sent = [call.kwargs["messages"] for call in wire.call_args_list]
            assert len(sent) == 2
            last = [m["content"][0] for m in sent[-1]]
            assert last[:2] == [good, other]  # the good PNGs are kept
            assert last[2]["text"] == NOTE  # only the corrupt one is replaced
            assert "image/svg+xml" in last[3]["text"]


def top_level(item_type="input_image", url="data:image/png;base64,iVBORw0KGgo="):
    return {"type": item_type, "image_url": url}


def test_a_top_level_attachment_item_in_a_refused_format_becomes_a_note():
    from nooa.unifiedllm.media_inputs import filter_media_messages, media_occurrences

    svg = top_level(url="data:image/svg+xml;base64,PHN2Zy8+")
    png = top_level()
    filtered = filter_media_messages([svg, png], ("image",))
    assert filtered[0]["role"] == "user" and "image/svg+xml" in filtered[0]["content"]
    assert filtered[1] == png
    assert [kind for kind, _ in media_occurrences([svg, png], ("image",))] == ["image"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_rejected_top_level_image_item_is_retried_as_a_note(asynchronous):
    method = "aresponses" if asynchronous else "responses"
    messages = [{"role": "user", "content": "Read this:"}, top_level()]

    async def send(client):
        return await client.acall(messages) if asynchronous else client.call(messages)

    def items(call):
        return call.kwargs["input"]

    with ResponsesClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire:
            wire.side_effect = [bad_request(GPT), ok(True)]
            assert (await send(client)).content == "UNREADABLE"
            first, second = (items(call) for call in wire.call_args_list)
            assert top_level() in first
            assert top_level() not in second and NOTE in json.dumps(second)
        with patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire:
            wire.return_value = ok(True)
            await send(client)
            await send(client)
            assert wire.call_count == 2
            assert all(top_level() not in items(call) for call in wire.call_args_list)


@pytest.mark.parametrize(
    "block",
    [
        # A format hint that names a refused type, beside a PNG data URL.
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBO", "format": "image/svg+xml"},
        },
        # Native Anthropic image block.
        {"type": "image", "source": {"type": "base64", "media_type": "image/BMP", "data": "Qk0="}},
        # Upper-case scheme.
        {"type": "image_url", "image_url": {"url": "DATA:image/svg+xml;base64,PHN2Zy8+"}},
        {"type": "input_image", "image_url": "data:image/tiff;base64,SUkq"},
    ],
)
def test_every_mime_hint_of_an_image_is_checked(block):
    from nooa.unifiedllm.media_inputs import filter_media_messages, unsendable_image_type

    assert unsendable_image_type(block) is not None
    filtered = filter_media_messages([{"role": "user", "content": [block]}], ("image",))
    assert filtered[0]["content"][0]["type"] == "text"


def test_accepted_aliases_are_normalised_on_the_wire_only():
    from nooa.unifiedllm.media_inputs import filter_media_messages, unsendable_image_type

    blocks = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/PNG;base64,iVBO", "format": "image/JPG; q=1"},
        },
        {"type": "image", "source": {"type": "base64", "media_type": "Image/Jpg", "data": "/9j/"}},
        {"type": "image_url", "image_url": {"url": "DATA:image/jpg;base64,/9j/"}},
    ]
    original = json.loads(json.dumps(blocks))
    assert all(unsendable_image_type(block) is None for block in blocks)
    sent = filter_media_messages([{"role": "user", "content": blocks}], ("image",))[0]["content"]
    assert sent[0]["image_url"] == {"url": "data:image/png;base64,iVBO", "format": "image/jpeg"}
    assert sent[1]["source"]["media_type"] == "image/jpeg"
    assert sent[2]["image_url"]["url"] == "data:image/jpeg;base64,/9j/"
    assert blocks == original  # the caller's history is unchanged


def test_completion_client_sends_the_normalised_block():
    block = {
        "type": "image_url",
        "image_url": {"url": "data:image/JPG;base64,/9j/", "format": "IMAGE/JPG"},
    }
    messages = [{"role": "user", "content": [block]}]
    with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as client:
        with patch("litellm.completion") as wire:
            wire.return_value = ok(False)
            client.call(messages)
            (sent,) = images(sent_blocks(wire.call_args, False))
    assert sent["image_url"] == {"url": "data:image/jpeg;base64,/9j/", "format": "image/jpeg"}
    assert block["image_url"]["format"] == "IMAGE/JPG"
