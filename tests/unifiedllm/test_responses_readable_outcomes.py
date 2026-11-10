# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Successful answers remain usable when their full wire shape cannot be replayed."""

from types import SimpleNamespace

import pytest

from nooa.llm_types import LLMResponse, LLMUsage
from nooa.unifiedllm import ResponsesClient
from nooa.unifiedllm.response_parts import project_turn


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("shape", ["builtin", "refusal", "both"])
@pytest.mark.asyncio
async def test_builtin_output_and_refusal_keep_readable_outcome(
    monkeypatch, caplog, is_async, shape
):
    refusal = shape in {"refusal", "both"}
    text = "I cannot help with that." if refusal else "The search found an answer."
    block = (
        {"type": "refusal", "refusal": text} if refusal else {"type": "output_text", "text": text}
    )
    raw = SimpleNamespace(
        output=[
            {"type": "reasoning", "encrypted_content": "secret", "summary": []},
            *(
                [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]
                if shape != "refusal"
                else []
            ),
            {"type": "message", "role": "assistant", "content": [block]},
        ],
        model="gpt-5.6",
        status="completed",
    )
    monkeypatch.setattr("litellm.responses", lambda **_: raw)

    async def respond(**_):
        return raw

    monkeypatch.setattr("litellm.aresponses", respond)
    monkeypatch.setattr("nooa.unifiedllm.unifiedllm._extract_usage", lambda _: LLMUsage())
    async with ResponsesClient("openai/gpt-5.6", api_key="test") as client:
        result = await client.acall([]) if is_async else client.call([])
    assert result.content == text
    assert all(part.native is None for part in result.parts)
    assert result.replay_scope is None
    assert "Unsupported Responses output" in caplog.text
    assert "secret" not in caplog.text
    restored = LLMResponse.model_validate_json(result.model_dump_json())
    assert project_turn(restored, "responses:openai:test") == [
        {"role": "assistant", "content": text}
    ]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_responses_calibration_counts_instructions(monkeypatch, is_async):
    from nooa.unifiedllm import unifiedllm as implementation

    calibration = implementation.TokenCalibration()
    monkeypatch.setattr(implementation, "_token_calibration", calibration)
    raw = SimpleNamespace(output=[], model="gpt-5.6", status="completed")
    monkeypatch.setattr("litellm.responses", lambda **_: raw)

    async def respond(**_):
        return raw

    monkeypatch.setattr("litellm.aresponses", respond)
    monkeypatch.setattr(implementation, "_extract_usage", lambda _: LLMUsage(input_tokens=1001))
    estimates = []

    def count(**kwargs):
        estimates.append(kwargs["messages"])
        return sum(len(m["content"]) for m in kwargs["messages"])

    monkeypatch.setattr("litellm.token_counter", count)
    messages = [{"role": "system", "content": "x" * 1000}, {"role": "user", "content": "y"}]
    async with ResponsesClient("openai/gpt-5.6", api_key="test") as client:
        await client.acall(messages) if is_async else client.call(messages)
    assert estimates == [
        [
            messages[0],
            {"role": "user", "content": [{"type": "text", "text": "y"}]},
        ]
    ]
    assert calibration.ratio("openai/gpt-5.6") == 1.0


@pytest.mark.parametrize("bad", [None, 42, {}])
def test_malformed_refusal_still_raises(bad):
    from nooa.unifiedllm.errors import ReasoningReplayError
    from nooa.unifiedllm.response_parts import capture_parts

    with pytest.raises(ReasoningReplayError, match="text must be a string"):
        capture_parts(
            [{"type": "message", "content": [{"type": "refusal", "refusal": bad}]}],
            "responses:openai:test",
        )


def test_calibration_survives_input_text_wrapped_content(monkeypatch):
    """litellm.token_counter chokes on Responses input_text/output_text blocks,
    and the fallback only recognizes type=="text" -- so wrapped content used
    to silently undercount to near-zero, driving the shared per-model
    calibration ratio to absurd multiples (observed ~41x) from one call. The
    estimate must survive the real (unmocked) litellm.token_counter.
    """
    from nooa.unifiedllm import unifiedllm as implementation

    calibration = implementation.TokenCalibration()
    monkeypatch.setattr(implementation, "_token_calibration", calibration)
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. " * 20},
        {"role": "user", "content": [{"type": "input_text", "text": "Please help me. " * 50}]},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": "result data. " * 50}],
        },
    ]
    implementation._update_token_calibration(
        "gpt-5.6", messages, LLMUsage(input_tokens=1000), tools=None, instructions=None
    )
    ratio = calibration.ratio("gpt-5.6")
    assert 0.3 <= ratio <= 5.0, f"calibration ratio blew up to {ratio}; estimate likely collapsed"


def test_calibration_counts_wrapped_tool_output_text():
    """litellm.token_counter silently ignores function_call_output.output
    text entirely (confirmed against the real, unmocked counter: a
    2000-word tool result counts identically to a 2-word one), so large
    tool results were invisible to the estimate -- collapsing it and
    inflating the calibration ratio for any tool-heavy conversation.
    _token_counter_messages must give that text a countable representation.
    """
    import litellm

    from nooa.unifiedllm import unifiedllm as implementation

    short = [
        {"role": "user", "content": "hi"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": "ok"}],
        },
    ]
    long = [
        {"role": "user", "content": "hi"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": "result data. " * 500}],
        },
    ]
    short_count = litellm.token_counter(
        model="gpt-5.6", messages=implementation._token_counter_messages(short)
    )
    long_count = litellm.token_counter(
        model="gpt-5.6", messages=implementation._token_counter_messages(long)
    )
    assert long_count > short_count + 500, (
        f"tool-output text change ({short_count} -> {long_count}) barely moved the "
        "estimate; function_call_output.output text is still not being counted"
    )


def test_calibration_counts_wrapped_input_image_blocks():
    """litellm.token_counter raises on Responses' input_image block shape
    (confirmed against the real, unmocked counter), which _update_token_
    calibration's outer except then falls back to a per-message text sum
    that skips non-"text" blocks entirely -- an image billed several hundred
    tokens by the real API silently contributes 0 to the estimate.
    _token_counter_messages must give it a countable representation.
    """
    import litellm

    from nooa.unifiedllm import unifiedllm as implementation

    no_image = [{"role": "user", "content": [{"type": "input_text", "text": "describe this"}]}]
    with_image = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": "https://example.test/photo.png"},
            ],
        }
    ]
    # The real, unmocked counter raises on the raw input_image shape.
    with pytest.raises(Exception, match="input_image"):
        litellm.token_counter(
            model="gpt-5.6",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.test/x"}],
                }
            ],
        )

    bare_count = litellm.token_counter(
        model="gpt-5.6", messages=implementation._token_counter_messages(no_image)
    )
    image_count = litellm.token_counter(
        model="gpt-5.6", messages=implementation._token_counter_messages(with_image)
    )
    assert image_count > bare_count + 50, (
        f"image block ({bare_count} -> {image_count}) barely moved the estimate; "
        "input_image is still not being counted"
    )
