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
                    native={
                        "thinking_blocks": {"type": "redacted_thinking", "data": encoded_blob}
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
