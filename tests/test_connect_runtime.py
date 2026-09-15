# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect must test the runtime client, not a second HTTP implementation."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest

from nooa import connect
from nooa.llm_types import LLMResponse, LLMUsage
from nooa.unifiedllm import registry
from tests.connect_http import mock_http, response_body


@pytest.mark.asyncio
async def test_reported_reply_ceiling_is_metadata_not_a_request_default(monkeypatch):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "local",
        "gpt-5.1",
        "chat",
        "https://api.test/v1",
        "",
        catalogue={
            "id": "example/model",
            "context_length": 262144,
            "top_provider": {"max_completion_tokens": 235929},
        },
    )
    assert proposal.entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == 235929
    assert proposal.entry["max_tokens"] == 32768
    client = registry.client_from_config("local", proposal.entry, api_key="test-key")
    try:
        await client.acall(messages=[{"role": "user", "content": "Hello"}])
    finally:
        await client.aclose()
    assert len(bodies) == 1
    assert all(
        bodies[0].get(key) != 235929
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_probes_use_registry_entry_and_unified_call(monkeypatch, style):
    proposal = connect.plan(
        "local",
        "openai/model",
        style,
        "https://api.test/v1",
        "",
        reasoning_levels={"low": {"reasoning_effort": "low"}},
    )
    before = deepcopy(proposal.entry)
    clients = []

    def create(name, config, **kwargs):
        assert name == "local"
        assert {k: v for k, v in config.items() if k != "provenance"} == {
            k: v for k, v in before.items() if k != "provenance"
        }
        assert kwargs["api_key"] == "transient-secret"
        assert kwargs["retry_config"].max_retries == 0
        client = AsyncMock()
        client.acall.return_value = LLMResponse(
            content="323",
            reasoning="Worked it out",
            usage=LLMUsage(input_tokens=10, output_tokens=2),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(registry, "client_from_config", create)
    result = await connect.run(proposal, approved="all", api_key="transient-secret")
    calls = [call for client in clients for call in client.acall.call_args_list]
    assert len(calls) == 3
    assert calls[0].kwargs["messages"] == proposal.probes[0].body.get(
        "messages", proposal.probes[0].body.get("input")
    )
    assert calls[1].kwargs["tools"][0].name == "probe_tool"
    assert calls[2].kwargs["reasoning_level"] == "low"
    assert all(client.aclose.await_count == 1 for client in clients)
    assert proposal.entry == before
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] == "accepted"
    assert record["reasoning_observed"] is True
    assert record["reported_tokens"] == 12
    assert "transient-secret" not in repr(result)


@pytest.mark.asyncio
async def test_runtime_failure_cannot_be_reported_as_success(monkeypatch):
    client = AsyncMock()
    client.acall.side_effect = ValueError("private server details and credentials")
    monkeypatch.setattr(registry, "client_from_config", lambda *a, **kw: client)
    proposal = connect.plan("local", "model", "responses", "https://api.test/v1", "")
    result = await connect.run(proposal, approved="all", api_key="key")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] != "accepted"
    assert record["error"] == "ValueError"
    assert "private" not in repr(result)
    assert client.acall.await_count == client.aclose.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_saved_entry_sends_the_same_request_as_connect(tmp_path, monkeypatch, style):
    sent = []

    def handle(request):
        sent.append((request.url.path, request.content))
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "local", "openai/model", style, "https://api.test/v1", "CONNECT_TEST_KEY"
    )
    monkeypatch.setenv("CONNECT_TEST_KEY", "temporary-key")
    result = await connect.run(proposal, approved="minimal")
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"
    path = tmp_path / "models.yaml"
    connect.write(result.entry, path, alias="local")
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    registry.reload_registry(path)
    client = registry.get_llm_client("local")
    try:
        await client.acall(
            messages=[{"role": "user", "content": "Compute 17 * 19. Reply with the number."}],
            **{"max_output_tokens" if style == "responses" else "max_tokens": 200},
        )
    finally:
        await client.aclose()
    assert len(sent) == 2
    assert sent[0] == sent[1]
    assert (
        sent[0][0]
        == "/v1/"
        + {"chat": "chat/completions", "responses": "responses", "anthropic": "messages"}[style]
    )
    assert json.loads(sent[0][1])["model"] == "openai/model"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("status", [429, 500])
async def test_runtime_and_sdk_do_not_retry_probe(monkeypatch, style, status):
    sent = []

    def handle(request):
        sent.append(request.url.path)
        return httpx.Response(status, json={"error": {"message": "private-secret"}})

    mock_http(monkeypatch, handle)
    proposal = connect.plan("local", "model", style, "https://api.test/v1", "")
    result = await connect.run(proposal, approved="all", api_key="key")
    assert len(sent) == 1
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] == "not_probed"
    assert record["status_code"] == status
    assert "private-secret" not in repr(result)


@pytest.mark.asyncio
async def test_old_http_only_success_is_not_reused(monkeypatch):
    proposal = connect.plan("local", "model", "chat", "https://api.test/v1", "")
    proposal.entry["provenance"]["probes"]["routing"] = {
        "outcome": "accepted",
        "request": deepcopy(proposal.probes[0].body),
        "fields_reached_wire": True,
    }
    client = AsyncMock()
    client.acall.side_effect = ValueError("Bad runtime configuration")
    monkeypatch.setattr(registry, "client_from_config", lambda *a, **kw: client)
    result = await connect.run(proposal, approved="minimal", api_key="key")
    assert client.acall.await_count == 1
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] != "accepted"


@pytest.mark.asyncio
async def test_level_changed_after_planning_cannot_exceed_approved_cap(monkeypatch):
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"high": {"max_tokens": 200}},
    )
    proposal.entry["reasoning_levels"]["high"]["max_tokens"] = 100000
    client = AsyncMock()
    client.acall.return_value = LLMResponse(content="323")
    monkeypatch.setattr(registry, "client_from_config", lambda *a, **kw: client)
    result = await connect.run(proposal, approved="all", api_key="key")
    assert result.entry["provenance"]["probes"]["level:high"]["outcome"] == "not_probed"
    assert client.acall.await_count == 2  # Routing and tools only.
