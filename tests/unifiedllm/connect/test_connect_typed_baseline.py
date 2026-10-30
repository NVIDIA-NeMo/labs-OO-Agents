# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native configured baseline policy, real runtime/SDKs and offline HTTP only."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm import ContextWindowExceededError

from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._baseline import _TextBaseline, typed_baseline
from nooa.unifiedllm.connect._records import text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(style="chat", **kwargs):
    return connect.plan(
        "local", "model", style, "https://api.test/v1", "", reply_tokens=1000, **kwargs
    )


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize(
    "text,finish,passed",
    [
        ('{"value":323}', "stop", True),
        ('{"value":322}', "stop", False),
        ('{"value":323}', "length", False),
        ('{"value":323}', "content_filter", False),
        ('{"value":"private-wrong-answer"}', "stop", False),
        ("323", "stop", False),  # Native object-root scalar schema, no custom parser.
        ("", "stop", False),
    ],
)
async def test_native_baseline_scalar_completion_and_media_gate(
    monkeypatch, style, text, finish, passed
):
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        body = response_body(style, text if len(sent) == 1 else "RED")
        if len(sent) == 1:
            if style == "chat":
                body["choices"][0]["finish_reason"] = finish
            elif style == "responses" and finish != "stop":
                body["status"] = "incomplete" if finish == "length" else "failed"
                body["incomplete_details"] = {"reason": "max_output_tokens"}
            elif style == "anthropic":
                body["stop_reason"] = "end_turn" if finish == "stop" else "max_tokens"
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    proposal = plan(style, test_media_inputs=["image"])
    result = await connect.check_stage(proposal, "media", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert text_baseline_passed(record) is passed, record
    assert connect.check_status("routing", record) == ("passed" if passed else "attention")
    assert len(sent) == (2 if passed else 1)
    if not passed:
        skipped = result.entry["provenance"]["probes"]["media:image"]
        assert "media not tested" in skipped["reason"]
        assert "Text baseline not confirmed" not in skipped["reason"]
        if text == '{"value":322}' and finish == "stop":
            assert "request accepted but returned the wrong integer" in skipped["reason"]
    assert '"type": "integer"' in json.dumps(sent[0])
    assert "private-wrong-answer" not in repr(result)
    assert "parsed" not in record and "parts" not in record


async def test_reasoning_only_typed_result_is_not_a_final_answer(monkeypatch):
    sent = []

    def handle(request):
        sent.append(request)
        body = response_body("chat", "")
        body["choices"][0]["message"]["reasoning_content"] = '{"value":323}'
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=["image"]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["typed_result_validated"] and record["answer_correct"]
    assert not record["final_answer_observed"] and not text_baseline_passed(record)
    assert "reasoning alone" in record["reason"]
    assert len(sent) == 1


@pytest.mark.parametrize("observation", [False, None])
async def test_missing_wire_settings_rejects_baseline_and_media(monkeypatch, observation):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: sent.append(request) or httpx.Response(200, json=response_body("chat")),
    )
    original = connect._run_probe

    async def unobserved(*args):
        response, _, transport = await original(*args)
        return response, observation, transport

    monkeypatch.setattr(connect, "_run_probe", unobserved)
    result = await connect.check_stage(plan(test_media_inputs=["image"]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["answer_correct"] and not text_baseline_passed(record)
    assert "not observed" in record["reason"] and len(sent) == 1
    assert (
        "settings were not observed"
        in result.entry["provenance"]["probes"]["media:image"]["reason"]
    )


async def test_selected_configured_effort_and_temperature_must_reach_wire(monkeypatch):
    bodies = []
    mock_http(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("chat"))
        ),
    )
    proposal = plan(reasoning_levels={"high": {"reasoning_effort": "high"}})
    proposal.entry.update(reasoning_level="high", temperature=0.2)
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    assert text_baseline_passed(result.entry["provenance"]["probes"]["routing"])
    assert bodies[0]["reasoning_effort"] == "high" and bodies[0]["temperature"] == 0.2


async def test_discovery_acceptance_cannot_be_reused_as_typed_baseline(monkeypatch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(
                200, json=response_body("chat", "wrong" if len(sent) == 1 else '{"value":323}')
            )
        ),
    )
    events = [
        event
        async for event in connect.check_interfaces(
            "local",
            "model",
            "https://api.test/v1",
            "",
            styles=("chat",),
            api_key="fake",
            output_tokens=1000,
        )
    ]
    discovery = events[-1]
    assert discovery.accepted == ("chat",)  # Acceptance, not task correctness.
    legacy = discovery.results["chat"].entry
    assert not text_baseline_passed(legacy["provenance"]["probes"]["routing"])
    proposal = plan(existing_entry=legacy)
    result = await connect.run(proposal, approved="minimal", api_key="fake")
    assert len(sent) == 2
    assert text_baseline_passed(result.entry["provenance"]["probes"]["routing"])
    assert '"type": "integer"' not in json.dumps(sent[0])
    assert '"type": "integer"' in json.dumps(sent[1])
    # A complete native configured result is reusable only for identical settings.
    await connect.run(plan(existing_entry=result.entry), approved="minimal", api_key="fake")
    assert len(sent) == 2
    changed = plan(existing_entry=result.entry)
    changed.entry["temperature"] = 0.5
    await connect.run(changed, approved="minimal", api_key="fake")
    assert len(sent) == 3


async def test_legacy_answer_evidence_is_invalid_even_at_same_cap(monkeypatch):
    proposal = plan()
    proposal.entry["provenance"]["probes"]["routing"] = {
        "outcome": "accepted",
        "client": "unifiedllm",
        "settings_sent": True,
        "answer_correct": True,
        "finish_reason": "stop",
        "request": deepcopy(proposal.probes[0].body),
    }
    sent = []
    mock_http(
        monkeypatch,
        lambda request: sent.append(request) or httpx.Response(200, json=response_body("chat")),
    )
    result = await connect.run(proposal, approved="minimal", api_key="fake")
    assert len(sent) == 1 and text_baseline_passed(result.entry["provenance"]["probes"]["routing"])


async def test_tools_stage_remains_tools_only_no_typed_call(monkeypatch):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        body = response_body("chat", "")
        body["choices"][0].update(finish_reason="tool_calls")
        body["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "probe_tool", "arguments": '{"value":"ok"}'},
            }
        ]
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(), "tools", api_key="fake")
    assert list(result.entry["provenance"]["probes"]) == ["tools"]
    assert result.entry["provenance"]["probes"]["tools"]["tool_observed"]
    assert len(bodies) == 1 and "response_format" not in bodies[0]


async def test_runtime_cap_recovery_blocked_before_second_acall(monkeypatch):
    proposal = plan()
    proposal.entry.update(context_window=8000, max_tokens=5000)
    client = registry.client_from_config("local", proposal.entry, api_key="fake")
    client.acall = AsyncMock(
        side_effect=ContextWindowExceededError(
            message="maximum context length is 8000 tokens; requested 9000 tokens (4000 in messages, 5000 in completion)",
            model="model",
            llm_provider="openai",
        )
    )
    try:
        with pytest.raises(RuntimeError, match="only one outbound call"):
            await typed_baseline(client)
        assert client.acall.await_count == 1
    finally:
        await client.aclose()


async def test_sdk_retry_blocked_by_http_hook_even_if_retry_config_regresses(monkeypatch):
    sent = []
    real_factory = registry.client_from_config

    def factory(*args, **kwargs):
        kwargs["num_retries"] = 2
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(registry, "client_from_config", factory)
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or httpx.Response(500, json={"error": {"message": "private"}})
        ),
    )
    result = await connect.run(plan(), approved="minimal", api_key="fake")
    assert len(sent) == 1 and not text_baseline_passed(
        result.entry["provenance"]["probes"]["routing"]
    )


async def test_configured_include_rejection_never_sends_second_candidate(monkeypatch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request)
            or httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported include",
                        "param": "include",
                    }
                },
            )
        ),
    )
    result = await connect.run(plan("responses"), approved="all", api_key="fake")
    assert len(sent) == 1
    assert result.entry["include"] == ["reasoning.encrypted_content"]
    assert not text_baseline_passed(result.entry["provenance"]["probes"]["routing"])


async def test_transient_native_agent_never_records_raw_reply(monkeypatch):
    agents = []
    recorded = []
    original = _TextBaseline.__init__

    def init(agent, **kwargs):
        original(agent, **kwargs)
        agents.append(agent)
        # Observe history *during* generation, not just after cleanup.
        agent.event_manager.on("LLMResponse", lambda event: recorded.append(event))

    monkeypatch.setattr(_TextBaseline, "__init__", init)
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    client = registry.client_from_config("local", plan().entry, api_key="fake")
    try:
        response, scalar = await typed_baseline(client)
        assert scalar == 323 and response.content == '{"value":323}'
        assert agents[0].event_manager.keys() == []
        assert recorded
        assert all(
            not event.content and not event.reasoning and event.raw_response is None
            for event in recorded
        )
        assert all(not part.native for event in recorded for part in event.parts)
    finally:
        await client.aclose()


async def test_adapter_dropped_selected_effort_is_not_baseline_success(monkeypatch):
    from nooa.unifiedllm import CompletionClient

    original = CompletionClient._prepare_call_config

    def dropped(self, overrides):
        result = original(self, overrides)
        result.pop("reasoning_effort", None)
        return result

    monkeypatch.setattr(CompletionClient, "_prepare_call_config", dropped)
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    proposal = plan(reasoning_levels={"high": {"reasoning_effort": "high"}})
    proposal.entry["reasoning_level"] = "high"
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["answer_correct"] and record["settings_sent"] is False
    assert not text_baseline_passed(record)


async def test_no_approval_never_calls_native_baseline(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("unapproved inference")

    monkeypatch.setattr(registry, "client_from_config", forbidden)
    result = await connect.run(plan(test_media_inputs=["image"]), approved="none")
    assert all(
        record["outcome"] == "not_probed"
        for record in result.entry["provenance"]["probes"].values()
    )
