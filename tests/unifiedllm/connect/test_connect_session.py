# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn evidence is based on actual SDK requests and reported usage."""

import json

import httpx
import pytest

from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._session import TOKEN_RESERVATION, _wire_reasoning, session_steps
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize(
    "mode", ["success", "early-hit", "no-cache", "no-reasoning", "no-usage", "truncated"]
)
async def test_session_checks_observe_wire_and_usage(monkeypatch, style, mode):
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        assert (
            body.get("max_tokens", body.get("max_output_tokens", body.get("max_completion_tokens")))
            == 2048
        )
        data = response_body(style)
        if mode != "no-reasoning":
            if style == "chat":
                data["choices"][0]["message"]["reasoning_content"] = "Reasoning payload for replay"
            elif style == "responses":
                data["output"].insert(
                    0,
                    {
                        "type": "reasoning",
                        "id": "r1",
                        "summary": [],
                        "encrypted_content": "OPAQUE-secret",
                    },
                )
            else:
                data["content"].insert(
                    0,
                    {
                        "type": "thinking",
                        "thinking": "Reasoning payload for replay",
                        "signature": "OPAQUE-secret",
                    },
                )
        if mode == "truncated":
            if style == "chat":
                data["choices"][0]["finish_reason"] = "length"
            elif style == "responses":
                data["status"] = "incomplete"
                data["incomplete_details"] = {"reason": "max_output_tokens"}
            else:
                data["stop_reason"] = "max_tokens"
        cached = 0 if mode == "no-cache" or len(sent) < 3 else 12
        if mode == "early-hit":
            cached = 12 if len(sent) == 2 else 0
        if style == "anthropic":
            data["usage"]["cache_read_input_tokens"] = cached
        else:
            data["usage"][
                "input_tokens_details" if style == "responses" else "prompt_tokens_details"
            ] = {"cached_tokens": cached}
        if mode == "no-usage":
            data.pop("usage")
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "test",
        "claude-sonnet-4-6" if style == "anthropic" else "gpt-5.1",
        style,
        "https://api.test/v1",
        "",
        reply_tokens=2048,
    )
    updates = [
        u
        async for u in session_steps(
            "test", proposal.entry, api_key="temporary-secret", budget_tokens=TOKEN_RESERVATION
        )
    ]
    records = {u.name: u.outcome for u in updates if u.outcome["outcome"] != "running"}
    assert "temporary-secret" not in repr(records)
    assert "OPAQUE-secret" not in repr(records)
    assert "Reasoning payload for replay" not in repr(records)
    if mode == "truncated" or (mode == "no-usage" and style == "anthropic"):
        assert len(sent) == 1
        assert records["session"]["outcome"] == "not_confirmed"
        return
    assert len(sent) == 3
    assert records["session"]["outcome"] == "completed"
    assert records["cache"]["stable_prefix"]
    assert len(records["cache"]["readings"]) == 2
    if style == "responses":
        # #341 enables explicit caching from the formatter boundary by default.
        # A server miss is a warning, not evidence that the runtime lost it.
        assert records["cache"]["explicit_mode"] is True
        assert records["cache"]["marker_count"] == 1
        assert all(body["prompt_cache_options"] == {"mode": "explicit"} for body in sent)
    if mode == "early-hit":
        assert records["cache"]["cached_input_tokens"] == 12
        assert records["cache"]["readings"][1]["cached_input_tokens"] == 0
    assert records["cache"]["outcome"] == (
        "warning" if mode in {"no-cache", "no-usage"} else "confirmed"
    )
    assert records["reasoning_retention"]["outcome"] == (
        "not_confirmed" if mode == "no-reasoning" else "confirmed"
    )
    if mode != "no-reasoning":
        assert (
            "OPAQUE-secret" if style != "chat" else "Reasoning payload for replay"
        ) in json.dumps(sent[1])


@pytest.mark.asyncio
async def test_session_requires_budget_before_any_call(monkeypatch):
    def forbidden(request):
        raise AssertionError("No HTTP is approved")

    mock_http(monkeypatch, forbidden)
    proposal = connect.plan("test", "gpt-5.1", "chat", "https://api.test/v1", "", reply_tokens=2048)
    updates = [
        u
        async for u in session_steps(
            "test", proposal.entry, api_key="key", budget_tokens=TOKEN_RESERVATION - 1
        )
    ]
    assert len(updates) == 1
    assert updates[0].outcome == {"outcome": "not_probed", "reason": "budget exhausted"}


@pytest.mark.asyncio
async def test_plan_and_run_include_session_reservation(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    basic = connect.plan("test", "gpt-5.1", "chat", "https://api.test/v1", "", reply_tokens=2048)
    proposal = connect.plan(
        "test",
        "gpt-5.1",
        "chat",
        "https://api.test/v1",
        "",
        session_checks=True,
        budget_tokens=65536,
        reply_tokens=2048,
    )
    assert proposal.token_estimate == basic.token_estimate + TOKEN_RESERVATION
    result = await connect.run(proposal, approved="all", api_key="key")
    assert result.entry["provenance"]["session_checks"]["session"]["outcome"] == "completed"
    assert result.entry["provenance"]["tokens_charged_to_budget"] == proposal.token_estimate


def test_portable_text_does_not_count_as_reasoning_replay():
    assert (
        list(_wire_reasoning({"messages": [{"role": "assistant", "content": "reasoning"}]})) == []
    )
    assert list(_wire_reasoning({"reasoning_content": "reasoning"})) == ["reasoning"]
    assert list(_wire_reasoning({"tool_calls": [{"id": "call__thought__opaque"}]})) == ["opaque"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key_present", [True, False])
async def test_reused_probes_resolve_credentials_before_session(monkeypatch, key_present):
    from dataclasses import replace

    sent = []

    def handle(request):
        sent.append(request)
        assert request.headers["authorization"] == "Bearer session-test-key"
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    monkeypatch.setenv("CONNECT_SESSION_KEY", "session-test-key")
    proposal = connect.plan(
        "test", "gpt-5.1", "chat", "https://api.test/v1", "CONNECT_SESSION_KEY", reply_tokens=2048
    )
    checked = await connect.run(proposal, approved="all")
    assert len(sent) == 2
    sent.clear()
    proposal = replace(proposal, entry=checked.entry, session_checks=True)
    if not key_present:
        monkeypatch.delenv("CONNECT_SESSION_KEY")
        with pytest.raises(ValueError, match="Set CONNECT_SESSION_KEY"):
            await connect.run(proposal, approved="all")
        assert not sent
        return
    result = await connect.run(proposal, approved="all")
    assert len(sent) == 3  # Routing/tools reused; only the session sends requests.
    assert result.entry["provenance"]["session_checks"]["session"]["outcome"] == "completed"


@pytest.mark.asyncio
async def test_selected_reasoning_level_and_cache_defaults_reach_wire(monkeypatch):
    from copy import deepcopy

    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        data = response_body("chat")
        data["choices"][0]["message"]["reasoning_content"] = "calculation"
        data["usage"]["prompt_tokens_details"] = {"cached_tokens": 15}
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "test",
        "gpt-5.1",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"high": {"reasoning_effort": "high"}},
        reply_tokens=2048,
    )
    before = deepcopy(proposal.entry)
    updates = [
        u
        async for u in session_steps(
            "test", proposal.entry, api_key="key", budget_tokens=TOKEN_RESERVATION
        )
    ]
    records = {u.name: u.outcome for u in updates if u.outcome["outcome"] != "running"}
    assert records["session"]["outcome"] == "completed", records
    assert len(bodies) == 3
    assert all(body["reasoning_effort"] == "high" for body in bodies)
    assert records["reasoning_retention"]["settings_retained"] is True
    assert proposal.entry == before
    assert "cache_breakpoint" not in proposal.entry


@pytest.mark.asyncio
async def test_missing_settings_are_not_reported_as_missing_replay(monkeypatch):
    # Regression for legacy parameter filtering, which direct does not perform.
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "litellm")
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        data = response_body("chat")
        data["choices"][0]["message"]["reasoning_content"] = "test reasoning"
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "test",
        "vendor/unlisted-model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"max": {"reasoning_effort": "max"}},
        reply_tokens=2048,
    )
    proposal.entry.pop("allowed_openai_params", None)  # A pre-fix saved entry.
    updates = [
        u
        async for u in session_steps(
            "test", proposal.entry, api_key="key", budget_tokens=TOKEN_RESERVATION
        )
    ]
    retained = next(u.outcome for u in updates if u.name == "reasoning_retention")
    assert all("reasoning_effort" not in body for body in bodies)
    assert retained["state_retained"] is True
    assert retained["settings_retained"] is False
    assert "state was replayed" in retained["reason"]


@pytest.mark.asyncio
async def test_unlisted_model_declared_effort_reaches_level_and_session_requests(monkeypatch):
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        data = response_body("chat")
        data["choices"][0]["message"]["reasoning_content"] = "test reasoning"
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "test",
        "vendor/unlisted-model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"max": {"reasoning_effort": "max"}},
        session_checks=True,
        budget_tokens=65536,
        reply_tokens=2048,
    )
    result = await connect.run(proposal, approved="all", api_key="test-key")
    assert result.entry["provenance"]["probes"]["level:max"]["settings_sent"] is True
    assert all(body["reasoning_effort"] == "max" for body in sent[2:])
    retention = result.entry["provenance"]["session_checks"]["reasoning_retention"]
    assert retention["state_retained"] is True
    assert retention["settings_retained"] is True


@pytest.mark.asyncio
async def test_small_cache_hit_is_not_a_success(monkeypatch):
    def handle(request):
        data = response_body("chat")
        data["usage"]["prompt_tokens"] = 6000
        data["usage"]["prompt_tokens_details"] = {"cached_tokens": 100}
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan("test", "gpt-5.1", "chat", "https://api.test/v1", "", reply_tokens=2048)
    updates = [
        u
        async for u in session_steps(
            "test", proposal.entry, api_key="key", budget_tokens=TOKEN_RESERVATION
        )
    ]
    cache = next(u.outcome for u in updates if u.name == "cache")
    assert cache["outcome"] == "warning"
    assert cache["cached_input_tokens"] == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", ["none", "minimal"])
async def test_library_session_never_runs_without_full_approval(monkeypatch, approval):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "test",
        "gpt-5.1",
        "chat",
        "https://api.test/v1",
        "",
        session_checks=True,
        budget_tokens=65536,
        reply_tokens=2048,
    )
    result = await connect.run(proposal, approved=approval, api_key="key")
    assert len(bodies) == (1 if approval == "minimal" else 0)
    assert result.entry["provenance"]["session_checks"]["session"]["outcome"] == "not_probed"


@pytest.mark.asyncio
async def test_real_tool_reply_is_replayed_without_execution(monkeypatch):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        data = response_body("chat")
        choice = data["choices"][0]
        choice["message"]["reasoning_content"] = "calculation"
        if len(bodies) == 1:
            choice["finish_reason"] = "tool_calls"
            choice["message"]["tool_calls"] = [
                {
                    "type": "function",
                    "id": "tool-one",
                    "function": {"name": "probe_tool", "arguments": '{"value":"686"}'},
                }
            ]
        else:
            prior = next(m for m in body["messages"] if m.get("tool_calls"))
            assert prior["reasoning_content"] == "calculation"
            assert prior["tool_calls"][0]["id"] == "tool-one"
            assert any(
                m.get("role") == "tool" and m["tool_call_id"] == "tool-one"
                for m in body["messages"]
            )
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan("test", "gpt-5.1", "chat", "https://api.test/v1", "", reply_tokens=2048)
    updates = [
        u
        async for u in session_steps(
            "test", proposal.entry, api_key="key", budget_tokens=TOKEN_RESERVATION
        )
    ]
    # The tool callable raises if executed: completing proves it was read as data.
    assert updates[-1].outcome["outcome"] == "completed"
    assert len(bodies) == 3


@pytest.mark.asyncio
async def test_closing_progress_iterator_closes_owned_client(monkeypatch):
    import nooa.unifiedllm.registry as registry
    from nooa.unifiedllm.registry import client_from_config

    clients = []

    def track(*args, **kwargs):
        client = client_from_config(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(registry, "client_from_config", track)
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    proposal = connect.plan("test", "gpt-5.1", "chat", "https://api.test/v1", "", reply_tokens=2048)
    steps = session_steps("test", proposal.entry, api_key="key", budget_tokens=TOKEN_RESERVATION)
    assert (await anext(steps)).name == "session:seed"
    http = clients[0]._http.httpx_async
    await steps.aclose()
    assert http.is_closed
