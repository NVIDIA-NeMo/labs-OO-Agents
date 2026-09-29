# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bounded reasoning probe with independently checkable final-answer evidence."""

import base64
import json
from itertools import permutations

import httpx
import pytest

from nooa.unifiedllm import AssistantReasoning, LLMResponse, LLMUsage, connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def test_scheduling_puzzle_has_one_solution():
    solutions = []
    for order in permutations("ABCDEFGH"):
        p = {job: order.index(job) for job in order}
        if (
            p["A"] == p["B"] + 3
            and p["E"] == p["C"] + 1
            and p["F"] == p["E"] + 1
            and p["D"] == p["G"] + 1
            and p["E"] > p["A"]
            and p["H"] == 7
        ):
            solutions.append("".join(order))
    assert solutions == ["BGDACEFH"]


def test_wrong_puzzle_answer_alone_is_not_a_check_failure():
    """The puzzle exists to elicit reasoning, not to prove the model can
    solve it. A wrong answer with reasoning genuinely observed must pass.
    """
    from nooa.unifiedllm.connect._records import check_status

    record = {"outcome": "accepted", "reasoning_observed": True, "answer_correct": False}
    assert check_status("level:high", record, missing_reasoning=False) == "passed"


def test_missing_reasoning_is_still_a_check_failure():
    """Unlike a wrong answer, a genuinely missing reasoning signal is exactly
    what this check exists to verify — that must still flag attention.
    """
    from nooa.unifiedllm.connect._records import check_status

    record = {"outcome": "accepted", "reasoning_observed": False, "answer_correct": True}
    assert check_status("level:high", record, missing_reasoning=True) == "attention"


@pytest.mark.asyncio
async def test_reasoning_observed_counts_an_empty_text_reasoning_part(monkeypatch):
    """Claude Sonnet 5/Opus 5 via Azure or Bedrock return a reasoning part with a
    signature but deliberately empty text. response.reasoning joins only
    non-empty parts, so a probe that checked that property alone (instead of
    the part's presence) reported no reasoning even though one was returned.
    """

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(AssistantReasoning(text=""),),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=20, output_tokens=2, total_tokens=22),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "claude-opus-5",
        "anthropic",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"thinking": {"type": "adaptive"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is False


@pytest.mark.asyncio
async def test_redacted_thinking_block_is_flagged_as_encrypted_not_missing(monkeypatch):
    """Anthropic's redacted_thinking is a distinct, detectable wire type (real
    reasoning occurred; the provider withholds the text) — not the same as a
    generic empty-text reasoning part from some other cause.
    """

    encoded_blob = base64.b64encode(b"x" * 100).decode()

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="",
                    native={"thinking_blocks": {"type": "redacted_thinking", "data": encoded_blob}},
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=147, output_tokens=687, total_tokens=834),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "claude-opus-5",
        "anthropic",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"thinking": {"type": "adaptive"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is True
    assert record["output_tokens"] == 687
    assert record["reasoning_encrypted_bytes"] == 100


@pytest.mark.asyncio
async def test_redacted_thinking_falls_back_to_char_count_for_non_base64_data(monkeypatch):
    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="",
                    # Not valid base64 (wrong padding length) — the opaque
                    # blob format isn't guaranteed, so this must degrade to a
                    # raw character count rather than raise.
                    native={"thinking_blocks": {"type": "redacted_thinking", "data": "opaque"}},
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=147, output_tokens=687, total_tokens=834),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "claude-opus-5",
        "anthropic",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"thinking": {"type": "adaptive"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_encrypted"] is True
    assert record["reasoning_encrypted_bytes"] == len("opaque")


@pytest.mark.asyncio
async def test_signed_thinking_block_with_empty_text_is_flagged_with_no_size(monkeypatch):
    """The dialect actually observed live for Claude Sonnet 5/Opus 5 via Azure
    or Bedrock: a normal *signed* "thinking" block (not redacted_thinking)
    whose visible text simply comes back empty. Detected the same way, but
    with no byte size — a signature's length doesn't scale with how much was
    thought, unlike a redacted_thinking block's opaque data blob.
    """

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="",
                    native={
                        "thinking_blocks": {
                            "type": "thinking",
                            "signature": "a-real-signature-value",
                        }
                    },
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=147, output_tokens=687, total_tokens=834),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "claude-opus-5",
        "anthropic",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"thinking": {"type": "adaptive"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is True
    assert record["reasoning_encrypted_bytes"] is None


@pytest.mark.asyncio
async def test_responses_encrypted_reasoning_item_is_flagged_as_encrypted(monkeypatch):
    """Responses-style routes (response_parts.py) store the raw output item on
    .native directly rather than wrapping it under "thinking_blocks" the way
    Chat-style routes do. Detection must recognize this shape too, or every
    Responses/OpenAI-style encrypted reasoning route (observed live for
    gpt-6-astra) always reads as "not encrypted".
    """

    encoded_blob = base64.b64encode(b"x" * 64).decode()

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="",
                    native={"type": "reasoning", "encrypted_content": encoded_blob},
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=101, output_tokens=259, total_tokens=360),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "gpt-6-astra",
        "responses",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"reasoning": {"effort": "medium"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is True
    assert record["reasoning_encrypted_bytes"] == 64


@pytest.mark.asyncio
async def test_chat_encrypted_reasoning_item_is_flagged_as_encrypted(monkeypatch):
    """openai/azure Chat Completions routes (chat_parts.py) store their
    reasoning item under a third, distinct shape: part.native["reasoning_items"]
    holding the raw item with encrypted_content — neither the Anthropic-style
    "thinking_blocks" wrapper nor the unwrapped Responses-style native.
    Missing this shape means every openai/azure Chat-API encrypted-reasoning
    route always reads as "not encrypted".
    """

    encoded_blob = base64.b64encode(b"x" * 48).decode()

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="",
                    native={
                        "reasoning_items": {"type": "reasoning", "encrypted_content": encoded_blob}
                    },
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=101, output_tokens=259, total_tokens=360),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "gpt-6-astra",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"reasoning_effort": "medium"}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is True
    assert record["reasoning_encrypted_bytes"] == 48


@pytest.mark.asyncio
async def test_responses_visible_summary_with_encrypted_content_is_not_flagged_as_withheld(
    monkeypatch,
):
    """An OpenAI-style Responses reasoning item can carry a visible summary
    (real reasoning text) alongside encrypted_content at the same time — the
    encrypted blob is opaque replay state, not proof the readable text was
    withheld. Flagging this as "withheld" hides the real char count the
    provider actually returned.
    """

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="Here is a short visible summary of my reasoning.",
                    native={"type": "reasoning", "encrypted_content": "abcd1234=="},
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=101, output_tokens=259, total_tokens=360),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "gpt-6-astra",
        "responses",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"reasoning": {"effort": "medium"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is False
    assert record["reasoning_text_chars"] == len("Here is a short visible summary of my reasoning.")


@pytest.mark.asyncio
async def test_chat_visible_summary_with_encrypted_content_is_not_flagged_as_withheld(monkeypatch):
    """Same caveat as the Responses-style case above, for the openai/azure
    Chat Completions "reasoning_items" shape.
    """

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="Here is a short visible summary of my reasoning.",
                    native={
                        "reasoning_items": {
                            "type": "reasoning",
                            "encrypted_content": "abcd1234==",
                        }
                    },
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=101, output_tokens=259, total_tokens=360),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "gpt-6-astra",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"reasoning_effort": "medium"}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is False
    assert record["reasoning_text_chars"] == len("Here is a short visible summary of my reasoning.")


@pytest.mark.asyncio
async def test_signed_thinking_block_with_real_text_is_not_flagged_as_withheld(monkeypatch):
    """A normal, fully visible signed thinking block must not be treated as
    withheld — only an empty one should be.
    """

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(
                AssistantReasoning(
                    text="Let me work through this step by step.",
                    native={
                        "thinking_blocks": {
                            "type": "thinking",
                            "signature": "a-real-signature-value",
                        }
                    },
                ),
            ),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=147, output_tokens=687, total_tokens=834),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "claude-opus-5",
        "anthropic",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"thinking": {"type": "adaptive"}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is False
    assert record["reasoning_text_chars"] == len("Let me work through this step by step.")


@pytest.mark.asyncio
async def test_visible_reasoning_text_with_no_litellm_estimate_reports_char_count(monkeypatch):
    """The dialect observed live for Qwen and DeepSeek routes on this
    gateway: real, non-empty reasoning text came back (no thinking_blocks
    structure at all, just a plain reasoning_content string), but litellm's
    text-length estimate -- which only exists for Anthropic/Bedrock -- never
    ran, leaving reasoning_tokens at 0 despite real reasoning having occurred.
    """
    reasoning_text = "First, note that H must be last. " * 20

    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(AssistantReasoning(text=reasoning_text),),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=115, output_tokens=2329, total_tokens=2444),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "qwen3.6-27b",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is True
    assert record["reasoning_encrypted"] is False
    assert record["reasoning_tokens"] == 0
    assert record["reasoning_text_chars"] == len(reasoning_text)


@pytest.mark.asyncio
async def test_no_visible_reasoning_text_reports_no_char_count(monkeypatch):
    async def fake_run_probe(alias, entry, probe, api_key):
        response = LLMResponse(
            parts=(),
            finish_reason="stop",
            usage=LLMUsage(input_tokens=115, output_tokens=20, total_tokens=135),
        )
        return response, True, "litellm"

    monkeypatch.setattr(connect, "_run_probe", fake_run_probe)
    proposal = connect.plan(
        "test",
        "some-model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"on": {"reasoning_effort": "high"}},
    )
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["reasoning_observed"] is False
    assert record["reasoning_text_chars"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize(
    "answer,correct", [("B G D A C E F H", True), ("ABCDEFGH", False), ("", False)]
)
async def test_puzzle_reaches_wire_and_scores_only_final_answer(
    monkeypatch, style, answer, correct
):
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        assert body.get("max_tokens", body.get("max_output_tokens")) == 32768
        messages = body.get("messages", body.get("input"))
        assert connect.REASONING_CHECK_PROMPT in json.dumps(messages, ensure_ascii=False).replace(
            "\\n", "\n"
        )
        return httpx.Response(200, json=response_body(style, answer))

    mock_http(monkeypatch, handle)
    settings = (
        {"thinking": {"type": "adaptive"}}
        if style == "anthropic"
        else {"reasoning": {"effort": "medium"}}
        if style == "responses"
        else {"reasoning_effort": "medium"}
    )
    proposal = connect.plan(
        "test",
        "claude-sonnet-4-6" if style == "anthropic" else "test-model",
        style,
        "https://api.test/v1",
        "",
        reasoning_levels={"on": settings},
    )
    assert "Compute 17 * 19" in str(proposal.probes[0].body)
    assert "Call probe_tool" in str(proposal.probes[1].body)
    result = await connect.check_stage(proposal, "reasoning", api_key="test-key")
    assert len(sent) == 1
    record = result.entry["provenance"]["probes"]["level:on"]
    assert record["settings_sent"] is True
    assert record["answer_correct"] is correct
    assert record["reasoning_observed"] is False  # Correct text is not reasoning evidence.
    assert record["input_tokens"] == 20
    assert record["output_tokens"] == 2
    assert "answer_correct" in connect.diagnostic_prompt(
        "reasoning", result.entry, {"level:on": record}
    )
    assert not {"content", "reasoning", "response"} & record.keys()
