# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed media contract integration: real Predict, real SDK, offline HTTP only."""

import base64
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm import ContextWindowExceededError

from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._media import pdf_fixture, png_fixture, video_fixture, wav_fixture
from nooa.unifiedllm.connect._typed_media import typed_media
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(**kwargs):
    return connect.plan(
        "local", "model", "chat", "https://api.test/v1", "", reply_tokens=1000, **kwargs
    )


def attachments(body):
    return [
        block
        for message in body["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") in {"image_url", "input_audio", "video_url", "file"}
    ]


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
@pytest.mark.parametrize("value", [True, False])
async def test_actual_native_method_unwraps_typed_value_and_attaches_original_once(
    monkeypatch, kind, value
):
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_body("chat", json.dumps({"value": value})))

    mock_http(monkeypatch, handle)
    entry = plan().entry
    entry.update(
        media_inputs=[kind],
        temperature=0.2,
        top_p=0.9,
        reasoning_level="high",
        reasoning_levels={"high": {"reasoning_effort": "high"}},
        allowed_openai_params=["reasoning_effort"],
    )
    client = registry.client_from_config("local", entry, api_key="fake")
    try:
        response, result = await typed_media(client, kind)
    finally:
        await client.aclose()
    assert type(result) is bool
    assert result == value
    assert response.content == json.dumps({"value": value})
    assert len(sent) == 1
    wire = sent[0]
    schema = wire["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    value_schema = schema["properties"]["value"]
    assert value_schema["type"] == "boolean"
    assert wire["max_tokens"] == 1000
    assert wire["temperature"] == 0.2 and wire["top_p"] == 0.9
    assert wire["reasoning_effort"] == "high"
    blocks = attachments(wire)
    assert len(blocks) == 1
    block = blocks[0]
    if kind == "audio":
        assert block["input_audio"]["format"] == "wav"
        payload = block["input_audio"]["data"]
        expected = wav_fixture()
    else:
        slot, key, mime, fixture = {
            "image": ("image_url", "url", "image/png", png_fixture),
            "video": ("video_url", "url", "video/mp4", video_fixture),
            "pdf": ("file", "file_data", "application/pdf", pdf_fixture),
        }[kind]
        assert block[slot][key].startswith(f"data:{mime};base64,")
        payload = block[slot][key].split(",", 1)[1]
        expected = fixture()
    assert base64.b64decode(payload) == expected
    texts = json.dumps(wire["messages"])
    assert "Can you access and interpret" in texts and "Return false" in texts
    assert "Count the separate tones" not in texts and "dominant colors" not in texts


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
@pytest.mark.parametrize(
    "text,valid,access",
    [
        ('{"value":true}', True, True),
        ('{"value":false}', True, False),
        ('{"value":1}', False, None),
        ('{"value":0}', False, None),
        ('{"value":"true"}', False, None),
        ('{"value":"false"}', False, None),
        ('{"value":"private-invalid-answer"}', False, None),
        ('{"value":null}', False, None),
        ("true", False, None),
        ("I can access it", False, None),
        ('{"value":', False, None),
    ],
)
async def test_acceptance_is_separate_from_native_validation_and_access(
    monkeypatch, kind, text, valid, access
):
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json=response_body("chat", '{"value":323}' if len(sent) == 1 else text)
        )

    mock_http(monkeypatch, handle)
    proposal = plan(test_media_inputs=[kind])
    proposal.entry.update(
        temperature=0.2,
        reasoning_level="high",
        reasoning_levels={"high": {"reasoning_effort": "high"}},
        allowed_openai_params=["reasoning_effort"],
    )
    result = await connect.check_stage(proposal, "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(sent) == 2
    assert record["request_accepted"] is True
    assert record["attachment_observed"] is True
    assert record["output_settings_observed"] is True
    assert record["typed_result_validated"] is valid
    assert "answer_correct" not in record
    assert record.get("access_reported") is access
    assert ("access_reported" in record) is valid
    assert record["outcome"] == (
        "passed" if access is True else "unavailable" if valid else "inconclusive"
    )
    if not valid:
        assert record["status_code"] == 200
        assert record["reason"].startswith("Could not read the answer")
    elif access is False:
        assert record["reason"] == "Model cannot access this input"
        assert "error" not in record
    assert result.entry["media_inputs"] == []
    public = json.dumps(connect.public_record(record))
    assert "private-invalid-answer" not in public and "base64" not in public
    if text != "true":
        assert text not in public
    assert sent[-1]["temperature"] == 0.2 and sent[-1]["reasoning_effort"] == "high"


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
@pytest.mark.parametrize("mode", ["reasoning_only", "length", "error", "tool_calls", "empty"])
async def test_only_completed_public_final_answer_can_pass(monkeypatch, kind, mode):
    calls = []
    value = True

    def handle(request):
        calls.append(request)
        text = json.dumps({"value": 323 if len(calls) == 1 else value})
        body = response_body("chat", text)
        if len(calls) == 2:
            choice = body["choices"][0]
            if mode in {"reasoning_only", "empty"}:
                choice["message"]["content"] = ""
                if mode == "reasoning_only":
                    choice["message"]["reasoning_content"] = text
            else:
                choice["finish_reason"] = mode
                if mode == "tool_calls":
                    choice["message"]["tool_calls"] = [
                        {
                            "id": "pending",
                            "type": "function",
                            "function": {"name": "unfinished", "arguments": "{}"},
                        }
                    ]
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(calls) == 2 and record["request_accepted"]
    assert record["outcome"] == "inconclusive"
    if mode == "reasoning_only":
        assert record["typed_result_validated"] and record["access_reported"] is True
        assert record["final_answer_observed"] is False
    assert result.entry["media_inputs"] == []


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
@pytest.mark.parametrize("missing", ["attachment", "settings"])
async def test_typed_answer_never_replaces_wire_evidence(monkeypatch, kind, missing):
    from nooa.unifiedllm.unifiedllm import CompletionClient

    original = CompletionClient._prepare_media_messages

    def drop(self, messages, config):
        projected = original(self, messages, config)
        if missing == "attachment":
            return [
                {**m, "content": [b for b in m["content"] if b.get("type") == "text"]}
                if isinstance(m.get("content"), list)
                else m
                for m in projected
            ]
        return projected

    monkeypatch.setattr(CompletionClient, "_prepare_media_messages", drop)
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        value = 323 if len(sent) == 1 else (True)
        return httpx.Response(200, json=response_body("chat", json.dumps({"value": value})))

    mock_http(monkeypatch, handle)
    if missing == "settings":
        from nooa.unifiedllm.connect import _session

        original_settings = _session.settings_on_wire
        monkeypatch.setattr(
            _session,
            "settings_on_wire",
            lambda expected, body: (
                False if attachments(body) else original_settings(expected, body)
            ),
        )
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(sent) == 2 and record["request_accepted"] and record.get("access_reported")
    assert record["typed_result_validated"] and not record["settings_sent"]
    assert record["attachment_observed"] is (missing != "attachment")
    assert record["output_settings_observed"] is (missing != "settings")
    assert record["outcome"] == "inconclusive"


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
@pytest.mark.parametrize("failure", [400, 429, 503, "timeout", "connection"])
async def test_no_fake_acceptance_or_retry_on_provider_transport_errors(monkeypatch, kind, failure):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json=response_body("chat"))
        if failure in {"timeout", "connection"}:
            cls = httpx.ReadTimeout if failure == "timeout" else httpx.ConnectError
            raise cls("private reply sk-secret https://private.test", request=request)
        return httpx.Response(failure, json={"error": {"message": "private reply sk-secret"}})

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(calls) == 2
    assert record["request_accepted"] is False
    assert record["typed_result_validated"] is False
    assert record["outcome"] == "inconclusive"
    assert "sk-secret" not in json.dumps(connect.public_record(record))


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_context_recovery_blocked_before_second_runtime_call(monkeypatch, kind):
    entry = plan().entry
    entry.update(context_window=8000, max_tokens=5000, media_inputs=[kind])
    # Explicit credentials replace the alias's credential scope. This fixture
    # intentionally declares the same input cap for its synthetic replacement.
    client = registry.client_from_config(
        "local", entry, api_key="fake", context_window=entry["context_window"]
    )
    client.acall = AsyncMock(
        side_effect=ContextWindowExceededError(
            message="context too large", model="model", llm_provider="openai"
        )
    )
    try:
        with pytest.raises(RuntimeError, match="only one outbound call"):
            await typed_media(client, kind)
        assert client.acall.await_count == 1
    finally:
        await client.aclose()


@pytest.mark.parametrize("budget,count", [(6607, 1), (6608, 2)])
@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_exact_saved_cap_budget_boundary(monkeypatch, kind, budget, count):
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        value = 323 if len(sent) == 1 else (True)
        return httpx.Response(200, json=response_body("chat", json.dumps({"value": value})))

    mock_http(monkeypatch, handle)
    proposal = plan(test_media_inputs=[kind], budget_tokens=budget)
    result = await connect.check_stage(proposal, "media", api_key="fake")
    assert len(sent) == count
    assert all(body["max_tokens"] == 1000 for body in sent)
    assert result.entry["provenance"]["tokens_charged_to_budget"] == (6608 if count == 2 else 1512)
    assert proposal.budget_tokens == budget


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_native_parse_failure_keeps_usage_and_skips_next_sample(monkeypatch, kind):
    sent = []

    def handle(request):
        sent.append(request)
        body = response_body("chat", '{"value":323}' if len(sent) == 1 else "private-bad-schema")
        if len(sent) == 2:
            body["usage"] = {
                "prompt_tokens": 10000,
                "completion_tokens": 9000,
                "total_tokens": 19000,
            }
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(
        plan(test_media_inputs=[kind, "audio" if kind == "pdf" else "pdf"], budget_tokens=12000),
        "media",
        api_key="fake",
    )
    probes = result.entry["provenance"]["probes"]
    record = probes[f"media:{kind}"]
    assert len(sent) == 2 and record["request_accepted"]
    assert not record["typed_result_validated"]
    assert record["reported_tokens"] == 19000 and record["output_tokens"] == 9000
    assert probes["media:audio" if kind == "pdf" else "media:pdf"]["reason"] == "budget exhausted"
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 1512 + 19000
    assert "private-bad-schema" not in json.dumps(connect.public_record(record))


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_http_guard_blocks_sdk_retry_before_second_send(monkeypatch, kind):
    sent = []
    real_factory = registry.client_from_config

    def factory(*args, **kwargs):
        kwargs["num_retries"] = 2
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(registry, "client_from_config", factory)

    def handle(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(503, json={"error": {"message": "private-failure"}})

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    assert len(sent) == 2
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert record["request_accepted"] is False
    assert record["status_code"] == 503
    assert record["outcome"] == "inconclusive"


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_http_200_malformed_envelope_keeps_acceptance_not_capability(monkeypatch, kind):
    sent = []

    def handle(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(
            200,
            json={
                "private-failure": "malformed envelope",
                "usage": {"prompt_tokens": 111, "completion_tokens": 9, "total_tokens": 120},
            },
        )

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(sent) == 2
    assert record["request_accepted"] is True and record["status_code"] == 200
    assert record["reported_tokens"] == 120 and record["output_tokens"] == 9
    assert not record["typed_result_validated"] and not record.get("access_reported")
    assert record["outcome"] == "inconclusive"
    assert record["reason"].startswith("Could not read the answer")
    assert "private-failure" not in json.dumps(connect.public_record(record))


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_native_media_reply_never_enters_history_or_caller_hooks(monkeypatch, kind):
    from nooa.runtime.hooks import hooks_scope
    from nooa.unifiedllm.connect._typed_media import (
        _AudioAccess,
        _DocumentAccess,
        _ImageAccess,
        _VideoAccess,
    )

    cls = {
        "image": _ImageAccess,
        "audio": _AudioAccess,
        "video": _VideoAccess,
        "pdf": _DocumentAccess,
    }[kind]
    original = cls.__init__
    agents, recorded, hook_events = [], [], []

    def init(agent, **kwargs):
        original(agent, **kwargs)
        agents.append(agent)
        agent.event_manager.on("LLMResponse", lambda event: recorded.append(event))

    class Hooks:
        def on_generation_start(self, *args):
            hook_events.append(args)

        def on_generation_end(self, *args):
            hook_events.append(args)

        def on_llm_call(self, *args):
            hook_events.append(args)

    monkeypatch.setattr(cls, "__init__", init)
    value = True
    text = json.dumps({"value": value})

    def handle(request):
        body = response_body("chat", text)
        body["choices"][0]["message"]["reasoning_content"] = "private-reasoning"
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    entry = plan().entry
    entry["media_inputs"] = [kind]
    client = registry.client_from_config("local", entry, api_key="fake")
    try:
        with hooks_scope(Hooks(), compose=False):
            response, result = await typed_media(client, kind)
        assert result == value and response.content == text
        assert recorded and agents[0].event_manager.keys() == []
        assert not hook_events
        assert all(not e.content and not e.reasoning and e.raw_response is None for e in recorded)
        assert all(not part.native for e in recorded for part in e.parts)
    finally:
        await client.aclose()
