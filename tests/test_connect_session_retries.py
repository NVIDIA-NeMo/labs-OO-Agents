# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Length recovery retries identical input within consent, never partial history."""

import json
from copy import deepcopy

import httpx
import pytest

from nooa import connect
from nooa._connect_session import session_steps
from tests.connect_http import mock_http, response_body


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("truncated_turn", [0, 1, 2])
async def test_length_retry_preserves_input_and_replay_evidence(monkeypatch, style, truncated_turn):
    sent = []
    counts = [0, 0, 0]
    failed_pair = []

    def handle(request):
        body = json.loads(request.content)
        wire = json.dumps(body)
        turn = 2 if "Check 2:" in wire else 1 if "Check 1:" in wire else 0
        counts[turn] += 1
        sent.append(body)
        assert "PARTIAL-DO-NOT-REPLAY" not in wire
        if turn == truncated_turn:
            failed_pair.append(deepcopy(body))
        truncated = turn == truncated_turn and counts[turn] == 1
        data = response_body(style, "PARTIAL-DO-NOT-REPLAY" if truncated else "686")
        if style == "chat":
            data["choices"][0]["message"]["reasoning_content"] = "private-reasoning-state"
            data["choices"][0]["finish_reason"] = "length" if truncated else "stop"
            data["usage"]["prompt_tokens_details"] = {"cached_tokens": 12 if turn else 0}
        elif style == "responses":
            data["output"].insert(
                0,
                {
                    "type": "reasoning",
                    "id": "r1",
                    "summary": [],
                    "encrypted_content": "private-reasoning-state",
                },
            )
            data["usage"]["input_tokens_details"] = {"cached_tokens": 12 if turn else 0}
            if truncated:
                data.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        else:
            data["content"].insert(
                0,
                {
                    "type": "thinking",
                    "thinking": "private-reasoning-state",
                    "signature": "private-signature",
                },
            )
            data["stop_reason"] = "max_tokens" if truncated else "end_turn"
            data["usage"]["cache_read_input_tokens"] = 12 if turn else 0
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    entry = connect.plan(
        "test",
        "claude-sonnet-4-6" if style == "anthropic" else "gpt-5.1",
        style,
        "https://api.test/v1",
        "",
    ).entry
    original = deepcopy(entry)
    updates = [
        u async for u in session_steps("test", entry, api_key="test-key", budget_tokens=200000)
    ]
    records = {u.name: u.outcome for u in updates}
    assert len(sent) == 4
    assert entry == original
    assert records["session"]["outcome"] == "completed"
    assert records["cache"]["outcome"] == "confirmed"
    assert records["reasoning_retention"]["outcome"] == "confirmed"
    cap_key = next(
        k
        for k in ("max_tokens", "max_output_tokens", "max_completion_tokens")
        if k in failed_pair[0]
    )
    assert [b.pop(cap_key) for b in failed_pair] == [2048, 4096]
    assert failed_pair[0] == failed_pair[1]
    name = ("seed", "replay", "repeat")[truncated_turn]
    assert [a["finish_reason"] for a in records[f"session:{name}"]["attempts"]] == [
        "length",
        "stop",
    ]
    assert len([u for u in updates if u.outcome["outcome"] == "retrying"]) == 1
    assert "private-reasoning-state" not in repr(records)
    assert records["session"]["tokens_charged_to_budget"] <= 200000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected_calls",
    [
        ("length", 3),
        ("budget", 1),
        ("saved-cap", 1),
        ("http", 1),
        ("filter", 1),
        ("fixed-level", 1),
    ],
)
async def test_retry_stops_at_explicit_bounds_and_never_retries_errors(
    monkeypatch, mode, expected_calls
):
    caps = []

    def handle(request):
        body = json.loads(request.content)
        caps.append(body.get("max_tokens", body.get("max_completion_tokens")))
        if mode == "http":
            return httpx.Response(500, json={"error": {"message": "failed"}})
        data = response_body("chat", "partial")
        data["choices"][0]["finish_reason"] = "content_filter" if mode == "filter" else "length"
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    entry = connect.plan(
        "test",
        "gpt-5.1",
        "chat",
        "https://api.test/v1",
        "",
        reply_tokens=2048 if mode == "saved-cap" else 32768,
    ).entry
    if mode == "fixed-level":
        entry["reasoning_levels"] = {"fixed": {"max_tokens": 2048}}
        entry["reasoning_default"] = "fixed"
    budget = 43008 if mode == "budget" else 200000
    updates = [
        u async for u in session_steps("test", entry, api_key="test-key", budget_tokens=budget)
    ]
    assert len(caps) == expected_calls
    assert caps == [2048, 4096, 8192][:expected_calls]
    assert updates[-1].outcome["outcome"] == "not_confirmed"
    assert updates[-1].outcome["tokens_charged_to_budget"] <= budget
    assert not any(u.name in {"cache", "reasoning_retention"} for u in updates)
