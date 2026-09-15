# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interface detection uses the ordinary probes and one shared request budget."""

import json

import httpx
import pytest

from nooa import connect
from tests.connect_http import mock_http, mock_post, response_body

REPLIES = {style: response_body(style) for style in ("chat", "responses", "anthropic")}
PATHS = {"/v1/chat/completions": "chat", "/v1/responses": "responses", "/v1/messages": "anthropic"}


@pytest.fixture(autouse=True)
def sdk_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


async def check(**kwargs):
    events = [
        e
        async for e in connect.check_interfaces(
            "local", "model", "https://api.test/v1", "", **kwargs
        )
    ]
    assert isinstance(events[-1], connect.InterfaceResult)
    return events, events[-1]


@pytest.mark.asyncio
async def test_all_interfaces_use_correct_wire_shapes_and_one_call_each(monkeypatch):
    sent = []

    def handle(request):
        style = PATHS[request.url.path]
        body = json.loads(request.content)
        sent.append(body)
        assert body["model"] == "model"
        assert body["max_output_tokens" if style == "responses" else "max_tokens"] == 200
        assert ("input" if style == "responses" else "messages") in body
        assert "tools" not in body
        assert request.headers.get("x-api-key" if style == "anthropic" else "authorization") == (
            "temporary-secret" if style == "anthropic" else "Bearer temporary-secret"
        )
        return httpx.Response(200, json=REPLIES[style])

    mock_http(monkeypatch, handle)
    events, result = await check(api_key="temporary-secret")
    assert len(sent) == 3
    assert list(result.accepted) == ["chat", "responses", "anthropic"]
    assert [(e.name, e.outcome["outcome"]) for e in events[:-1]] == [
        (s, o) for s in REPLIES for o in ("running", "accepted")
    ]
    assert result.tokens_charged_to_budget == 3 * 712
    assert "temporary-secret" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [400, 401, 404, 429, 500, "timeout", "wrong-shape"])
async def test_failed_interface_is_not_offered_but_other_interfaces_are_checked(
    monkeypatch, failure
):
    calls = []

    async def post(self, url, **kwargs):
        calls.append(url)
        if url.endswith("chat/completions"):
            return httpx.Response(200, json=REPLIES["chat"])
        if failure == "timeout":
            raise httpx.ReadTimeout("secret must not escape")
        return (
            httpx.Response(200, json=REPLIES["chat"])
            if failure == "wrong-shape"
            else httpx.Response(failure, text="secret must not escape")
        )

    mock_post(monkeypatch, post)
    _, result = await check()
    assert result.accepted == ("chat",)
    assert len(calls) == 3
    assert "secret must not escape" not in repr(result)
    assert result.results["responses"].entry["provenance"]["probes"]["routing"]["outcome"] == (
        "rejected" if failure == 400 else "not_confirmed" if failure == "timeout" else "not_probed"
    )
    if failure == "timeout":
        record = result.results["responses"].entry["provenance"]["probes"]["routing"]
        # LiteLLM may replace the original httpx exception instead of chaining it.
        # Report the actual available class; do not invent connect/read attribution.
        assert record["timeout_kind"] in {"ReadTimeout", "Timeout", "APITimeoutError"}
        assert record["error_chain"]
        assert record["elapsed_seconds"] >= 0
        assert record["request_shape"] == {
            "api_style": "responses",
            "output_tokens": 200,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,usage,calls", [(1, 0, 0), (712, 0, 1), (1500, 1400, 1), (2136, 0, 3)]
)
async def test_interface_checks_share_budget_including_reported_usage(
    monkeypatch, budget, usage, calls
):
    sent = []

    async def post(self, url, **kwargs):
        sent.append(url)
        style = PATHS[httpx.URL(url).path]
        return httpx.Response(
            200,
            json={
                **REPLIES[style],
                "usage": {
                    "prompt_tokens" if style == "chat" else "input_tokens": usage,
                    "completion_tokens" if style == "chat" else "output_tokens": 0,
                    "total_tokens": usage,
                },
            },
        )

    mock_post(monkeypatch, post)
    _, result = await check(budget_tokens=budget)
    assert len(sent) == calls
    assert result.tokens_charged_to_budget == max(712, usage) * calls


@pytest.mark.asyncio
async def test_selected_routing_result_reused_even_when_levels_are_added(monkeypatch):
    sent = []

    async def post(self, url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(200, json=REPLIES[PATHS[httpx.URL(url).path]])

    mock_post(monkeypatch, post)
    _, result = await check()
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        existing_entry=result.results["chat"].entry,
        reasoning_levels={"high": {"reasoning_effort": "high"}},
    )
    await connect.run(proposal, approved="all")
    assert len(sent) == 5  # Three interfaces, tools, high; no second routing call.


@pytest.mark.asyncio
@pytest.mark.parametrize("style", list(REPLIES))
@pytest.mark.parametrize(
    "payload",
    [{}, {"error": {"message": "error"}}, {"choices": "bad", "output": "bad", "content": "bad"}],
)
async def test_success_status_without_expected_response_shape_is_inconclusive(
    monkeypatch, style, payload
):
    async def post(self, url, **kwargs):
        return httpx.Response(200, json=payload)

    mock_post(monkeypatch, post)
    proposal = connect.plan("local", "model", style, "https://api.test/v1", "")
    result = await connect.run(proposal, approved="minimal")
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "not_probed"


@pytest.mark.asyncio
async def test_responses_probe_recovers_from_explicit_include_rejection(monkeypatch):
    async def post(self, url, **kwargs):
        body = kwargs["json"]
        if url.endswith("/responses") and "include" not in body:
            assert body["store"] is False
            return httpx.Response(200, json=REPLIES["responses"])
        return httpx.Response(400, json={"error": {"message": "Unsupported field include"}})

    mock_post(monkeypatch, post)
    _, result = await check()
    assert result.accepted == ("responses",)
    entry = result.results["responses"].entry
    assert entry["include"] == []
    assert "encrypted_reasoning" not in entry


@pytest.mark.asyncio
async def test_cancelling_interface_checks_before_send_closes_client(monkeypatch):
    clients = []
    original = httpx.AsyncClient

    def create(**kwargs):
        client = original(**kwargs)
        clients.append(client)
        return client

    async def forbidden(*args, **kwargs):
        raise AssertionError("Cancellation must stop this request and later styles")

    monkeypatch.setattr(httpx, "AsyncClient", create)
    monkeypatch.setattr(original, "post", forbidden)
    steps = connect.check_interfaces("local", "model", "https://api.test/v1", "")
    event = await anext(steps)
    assert event.outcome["outcome"] == "running"
    await steps.aclose()
    assert len(clients) == 0  # Cancellation before dispatch constructs no runtime client.


def test_removed_level_does_not_leave_a_stale_accepted_probe():
    previous = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"high": {"reasoning_effort": "high"}},
    )
    previous.entry["provenance"]["probes"]["level:high"] = {"outcome": "accepted"}
    current = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={},
        existing_entry=previous.entry,
    )
    assert "level:high" not in current.entry["provenance"]["probes"]
