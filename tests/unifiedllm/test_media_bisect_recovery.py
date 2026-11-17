# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597: media recovery finds the bad attachment(s) without dropping good ones.

Items 6 (bisect instead of "all of that kind") and 7 (the general fallback:
one probe without media when no rejection rule matched). The provider is a
fake: ``litellm.acompletion``/``litellm.completion`` patched with a function
that fails while any "bad" attachment is still in the request.
"""

import math
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.media_inputs import attachment_kind

NOTE = "[An image was removed here because the model provider could not read it.]"
# Bedrock names no image (captured, see test_media_rejection_recovery.py).
BEDROCK = 'BedrockException - {"message":"The model returned the following errors: Could not process image"}'
# A 400 that names no media and no parameter: only the general fallback can help.
UNNAMED = (
    "OpenAIException - Error code: 400 - {'error': {'message': 'Provided request is malformed.', "
    "'type': 'invalid_request_error'}}"
)


def numbered_images(count):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,aW1n{i:04d}"}}
        for i in range(count)
    ]


def bad_request(text):
    return litellm.BadRequestError(message=text, model="test", llm_provider="openai")


def sent_images(kwargs):
    return [
        b
        for m in kwargs["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if attachment_kind(b) == "image"
    ]


def reply(text):
    return litellm.ModelResponse(choices=[{"message": {"content": text}}])


def fake_provider(bad, error=BEDROCK):
    def wire(**kwargs):
        images = sent_images(kwargs)
        if any(b in images for b in bad):
            raise bad_request(error)
        return reply(f"saw {len(images)} images")

    return wire


def client(max_retries=3):
    return CompletionClient(
        "openai/test",
        media_inputs=["image"],
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=max_retries),
    )


def messages_with(blocks):
    return [{"role": "user", "content": [block]} for block in blocks]


@pytest.fixture(autouse=True)
def no_sleep():
    with (
        patch("nooa.unifiedllm.retry.asyncio.sleep", new_callable=AsyncMock),
        patch("nooa.unifiedllm.retry.time.sleep"),
    ):
        yield


@pytest.mark.parametrize("error", [BEDROCK, UNNAMED])
@pytest.mark.parametrize("bad_index", [0, 5, 11, 15])
async def test_one_bad_image_among_sixteen_is_found_with_log2_requests(error, bad_index):
    images = numbered_images(16)
    bad = images[bad_index]
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = fake_provider([bad], error)
        response = await c.acall(messages_with(images))
        assert response.content == "saw 15 images"  # the final request's own reply
        requests = wire.call_count - 1  # after the original request
        # newest alone, (all or probe), bisect over 16, final: log2(16) + 3;
        # the fallback first re-sends the original once: one more
        assert requests <= math.ceil(math.log2(16)) + 3 + (error == UNNAMED)
        last = sent_images(wire.call_args.kwargs)
        assert bad not in last and len(last) == 15
    # The exclusion is kept for later requests on this connection.
    with patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = fake_provider([bad], error)
        await c.acall(messages_with(images))
        assert wire.call_count == 1


@pytest.mark.parametrize("error", [BEDROCK, UNNAMED])
async def test_two_bad_images_are_both_found_and_good_ones_kept(error):
    images = numbered_images(10)
    bad = [images[2], images[7]]
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = fake_provider(bad, error)
        assert (await c.acall(messages_with(images))).content == "saw 8 images"
        last = sent_images(wire.call_args.kwargs)
        assert all(b not in last for b in bad) and len(last) == 8


def test_sync_call_uses_the_same_search():
    images = numbered_images(8)
    with client() as c, patch("litellm.completion") as wire:
        wire.side_effect = fake_provider([images[1]])
        assert c.call(messages_with(images)).content == "saw 7 images"


async def test_a_failed_probe_raises_the_original_error_and_excludes_nothing():
    images = numbered_images(4)

    def wire_reply(**kwargs):
        raise bad_request(UNNAMED)  # fails with or without media: not the media

    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        with pytest.raises(litellm.BadRequestError, match="malformed"):
            await c.acall(messages_with(images))
        assert wire.call_count == 3  # the request, the same request again, one probe
        assert sent_images(wire.call_args.kwargs) == []  # the probe carried no media
        view = c._prepare_media_messages(messages_with(images), c._prepare_call_config({}))
        assert NOTE not in str(view)


async def test_the_probe_has_no_endpoint_retries_and_its_reply_is_discarded():
    images = numbered_images(2)
    calls = []

    def wire_reply(**kwargs):
        calls.append(len(sent_images(kwargs)))
        if len(calls) == 3:  # the probe: a retryable 500 must not be retried
            raise litellm.InternalServerError(message="boom", model="m", llm_provider="openai")
        raise bad_request(UNNAMED)

    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        with pytest.raises(litellm.BadRequestError):
            await c.acall(messages_with(images))
    assert calls == [2, 2, 0]  # the request, re-sent once, the probe

    # When the probe succeeds its reply ("PROBE") is never returned.
    def probe_ok(**kwargs):
        n = len(sent_images(kwargs))
        if images[0] in sent_images(kwargs):
            raise bad_request(UNNAMED)
        return reply("PROBE" if n == 0 else f"saw {n} images")

    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = probe_ok
        assert (await c.acall(messages_with(images))).content == "saw 1 images"


@pytest.mark.parametrize(
    "error",
    [
        litellm.InternalServerError(message="Internal server error", model="m", llm_provider="x"),
        litellm.RateLimitError(message="rate limit exceeded", model="m", llm_provider="x"),
        litellm.Timeout(message="Request timed out", model="m", llm_provider="x"),
        litellm.APIConnectionError(message="Connection reset by peer", model="m", llm_provider="x"),
        litellm.ContextWindowExceededError(
            message="This model's maximum context length is 8192 tokens",
            model="m",
            llm_provider="x",
        ),
        bad_request("prompt is too long: 210000 tokens > 200000 maximum"),
        litellm.AuthenticationError(message="Invalid API key", model="m", llm_provider="x"),
        litellm.NotFoundError(message="model not found", model="m", llm_provider="x"),
        litellm.ContentPolicyViolationError(
            message="Your request was flagged by our content policy", model="m", llm_provider="x"
        ),
        bad_request("Invalid schema for function 'run': 'type' is required (tools[0])"),
        bad_request("Invalid value for 'temperature': expected a number."),
    ],
)
async def test_errors_that_are_not_about_media_never_probe(error):
    images = numbered_images(3)
    with client(max_retries=1) as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = error
        with pytest.raises(type(error)):
            await c.acall(messages_with(images))
        # Only endpoint retries, never a request with media removed.
        assert all(len(sent_images(call.kwargs)) == 3 for call in wire.call_args_list)


async def test_max_retries_zero_disables_media_recovery():
    images = numbered_images(2)
    with client(max_retries=0) as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = fake_provider([images[0]])
        with pytest.raises(litellm.BadRequestError):
            await c.acall(messages_with(images))
        assert wire.call_count == 1


async def test_only_kept_exclusions_are_counted():
    from nooa.runtime.actor import _make_llm_metrics_bridge
    from nooa.runtime.harness_metrics import HarnessMetrics
    from nooa.unifiedllm.unifiedllm import _llm_metrics_callback

    images = numbered_images(6)
    metrics = HarnessMetrics()
    token = _llm_metrics_callback.set(_make_llm_metrics_bridge(metrics))
    try:
        # Budget 4 (max_retries=1) is too small to find the oldest of six: nothing kept.
        with client(max_retries=1) as c, patch("litellm.acompletion", new_callable=AsyncMock) as w:
            w.side_effect = fake_provider([images[0]])
            with pytest.raises(litellm.BadRequestError):
                await c.acall(messages_with(images))
        assert metrics.to_span_attributes().get("harness.media.rejected_retry.count", 0) == 0
        with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as w:
            w.side_effect = fake_provider([images[0]])
            await c.acall(messages_with(images))
    finally:
        _llm_metrics_callback.reset(token)
    attrs = metrics.to_span_attributes()
    assert attrs["harness.media.rejected_retry.count"] == 1
    assert attrs["harness.media.recovery_probes.count"] > 1


class StatusError(Exception):
    def __init__(self, status, message, param=None):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message, "param": param}}


@pytest.mark.parametrize(
    "text",
    [
        "messages.2: unexpected `tool_use_id` found in `tool_result` blocks: toolu_01",
        "Each `tool_result` block must have a corresponding `tool_use` block",
        "An assistant message with 'tool_calls' must be followed by tool messages",
        "Invalid tool call id: call_abc",
        "roles must alternate between user and assistant",
        "max_tokens: 64000 > 32000, which is the maximum allowed",
        "This model's maximum is 4096 completion tokens",
        "temperature and top_p cannot both be specified for this model",
        "Extra inputs are not permitted",
        "Invalid model ID was provided",
        "invalid model: gpt-9",
        "Function declaration is invalid: required property missing",
        "Invalid JSON: schema does not match",
    ],
)
def test_ordinary_400s_are_not_eligible_for_the_fallback(text):
    from nooa.unifiedllm.unifiedllm import _media_fallback_eligible

    assert not _media_fallback_eligible(bad_request(text))
    assert _media_fallback_eligible(bad_request(UNNAMED))


def test_413_and_bracketed_params_are_not_eligible_for_the_fallback():
    from nooa.unifiedllm.unifiedllm import _media_fallback_eligible

    assert not _media_fallback_eligible(StatusError(413, "Request Entity Too Large"))
    assert not _media_fallback_eligible(StatusError(413, "payload exceeds 20MB"))
    assert not _media_fallback_eligible(
        StatusError(400, "expected an object", "tools[0].function.parameters")
    )
    assert not _media_fallback_eligible(StatusError(400, "bad value", "max_tokens"))


async def test_a_tool_use_400_with_images_sends_no_probe():
    images = numbered_images(3)
    error = bad_request("unexpected `tool_use_id` found in `tool_result` blocks")
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = error
        with pytest.raises(litellm.BadRequestError):
            await c.acall(messages_with(images))
        assert wire.call_count == 1


async def test_a_search_does_not_remove_attachments_from_a_concurrent_call():
    import asyncio

    images = numbered_images(3)
    bad = images[0]
    a_messages = [{"role": "user", "content": [{"type": "text", "text": "A"}, *images]}]
    b_messages = [{"role": "user", "content": [{"type": "text", "text": "B"}, *images[1:]]}]
    probe_started, release = asyncio.Event(), asyncio.Event()

    async def wire_reply(**kwargs):
        sent = sent_images(kwargs)
        if kwargs["messages"][0]["content"][0]["text"] == "B":
            return reply(f"B saw {len(sent)} images")
        if not sent:  # A's probe without media: hold it while B runs
            probe_started.set()
            await release.wait()
        if bad in sent:
            raise bad_request(UNNAMED)
        return reply(f"A saw {len(sent)} images")

    with client() as c, patch("litellm.acompletion", new=wire_reply):
        task_a = asyncio.create_task(c.acall(a_messages))
        await asyncio.wait_for(probe_started.wait(), 5)
        assert (await c.acall(b_messages)).content == "B saw 2 images"
        release.set()
        assert (await task_a).content == "A saw 2 images"
        # Committed only now: B's images are good, so B still sends both.
        assert (await c.acall(b_messages)).content == "B saw 2 images"


@pytest.mark.parametrize("error", [BEDROCK, UNNAMED])
async def test_the_budget_counts_endpoint_requests_with_500s_mid_search(error):
    images = numbered_images(16)
    bad = images[5]
    seen = set()

    def wire_reply(**kwargs):
        sent = tuple(b["image_url"]["url"] for b in sent_images(kwargs))
        if wire.call_count > 1 and sent not in seen:
            seen.add(sent)  # each new search request fails once with a generic 500
            raise litellm.InternalServerError(message="boom", model="m", llm_provider="openai")
        if bad in sent_images(kwargs):
            raise bad_request(error)
        return reply(f"saw {len(sent)} images")

    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        budget = c._media_recovery_attempts()
        try:
            await c.acall(messages_with(images))
        except (litellm.BadRequestError, litellm.InternalServerError):
            pass  # the budget may run out before the search ends
        assert wire.call_count - 1 <= budget


@pytest.mark.parametrize("bad_ok", [True, False])
async def test_one_image_sends_no_identical_repeat_in_named_mode(bad_ok):
    images = numbered_images(1)

    def wire_reply(**kwargs):
        if sent_images(kwargs) or not bad_ok:
            raise bad_request(BEDROCK)
        return reply("ok")

    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = wire_reply
        if bad_ok:
            assert (await c.acall(messages_with(images))).content == "ok"
        else:
            with pytest.raises(litellm.BadRequestError):
                await c.acall(messages_with(images))
        # The request, then the image replaced by a note; never a third identical one.
        assert [len(sent_images(call.kwargs)) for call in wire.call_args_list] == [1, 0]


async def test_one_image_in_fallback_mode_reuses_the_probe_reply():
    images = numbered_images(1)
    with client() as c, patch("litellm.acompletion", new_callable=AsyncMock) as wire:
        wire.side_effect = fake_provider(images, UNNAMED)
        assert (await c.acall(messages_with(images))).content == "saw 0 images"
        # The request, re-sent once, the probe (its reply is the answer).
        assert [len(sent_images(call.kwargs)) for call in wire.call_args_list] == [1, 1, 0]


async def test_a_bad_image_and_a_bad_pdf_are_both_found():
    images = numbered_images(3)
    pdf = {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjQ="}}
    good_pdf = {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjU="}}
    blocks = [images[0], pdf, images[1], good_pdf, images[2]]

    def wire_reply(**kwargs):
        sent = [b for m in kwargs["messages"] for b in m["content"]]
        if images[1] in sent:
            raise bad_request(BEDROCK)
        if pdf in sent:
            raise bad_request("The PDF specified was not valid.")
        return reply(f"saw {len(sent_images(kwargs))} images, {good_pdf in sent}")

    with (
        CompletionClient(
            "openai/test",
            media_inputs=["image", "pdf"],
            cache_breakpoint=None,
            retry_config=RetryConfig(max_retries=3),
        ) as c,
        patch("litellm.acompletion", new_callable=AsyncMock) as wire,
    ):
        wire.side_effect = wire_reply
        assert (await c.acall(messages_with(blocks))).content == "saw 2 images, True"
        assert wire.call_count - 1 <= c._media_recovery_attempts()
        wire.reset_mock()
        assert (await c.acall(messages_with(blocks))).content == "saw 2 images, True"
        assert wire.call_count == 1  # both exclusions kept, no new search


@pytest.mark.parametrize("scenario", ["one_bad", "two_bad", "probe_fails", "one_off"])
async def test_sync_and_async_fallback_send_the_same_requests(scenario):
    images = numbered_images(6)

    def make_wire():
        calls = []

        def wire_reply(**kwargs):
            sent = sent_images(kwargs)
            calls.append(len(sent))
            if scenario == "probe_fails":
                raise bad_request(UNNAMED)
            if scenario == "one_off":
                if len(calls) == 1:
                    raise bad_request(UNNAMED)
                return reply(f"saw {len(sent)} images")
            bad = [images[2]] if scenario == "one_bad" else [images[1], images[4]]
            if any(b in sent for b in bad):
                raise bad_request(UNNAMED)
            return reply(f"saw {len(sent)} images")

        return wire_reply, calls

    outcomes = []
    for mode in ("sync", "async"):
        wire_reply, calls = make_wire()
        target = "litellm.completion" if mode == "sync" else "litellm.acompletion"
        mock = {} if mode == "sync" else {"new_callable": AsyncMock}
        with client() as c, patch(target, **mock) as wire:
            wire.side_effect = wire_reply
            try:
                if mode == "sync":
                    result = c.call(messages_with(images)).content
                else:
                    result = (await c.acall(messages_with(images))).content
            except litellm.BadRequestError:
                result = "failed"
            # The next request shows what the connection kept.
            later = c._prepare_media_messages(messages_with(images), c._prepare_call_config({}))
        outcomes.append((result, calls, str(later).count(NOTE)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] != "failed" or scenario == "probe_fails"
