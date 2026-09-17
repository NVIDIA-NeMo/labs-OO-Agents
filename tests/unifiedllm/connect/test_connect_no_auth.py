# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A saved no-auth alias must not send ambient credentials to a custom endpoint."""

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect
from nooa.unifiedllm.registry import client_from_config
from tests.unifiedllm.connect.connect_http import response_body


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("no_auth", [False, True])
async def test_saved_auth_choice_survives_reload(
    style, asynchronous, no_auth, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-anthropic-token")
    sent = []

    def respond(request):
        sent.append(request)
        if no_auth:
            assert "authorization" not in request.headers
            assert "x-api-key" not in request.headers
        elif style == "anthropic":
            assert request.headers["x-api-key"] == "ambient-anthropic"
        else:
            assert request.headers["authorization"] == "Bearer ambient-openai"
        return httpx.Response(200, json=response_body(style))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: respond(req))

    async def handle_async(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    proposal = connect.plan(
        "local", "model", style, "https://custom.example/v1", "", reply_tokens=128
    )
    if not no_auth:
        del proposal.entry["api_key_env"]  # Absent preserves the ordinary SDK fallback.
    path = tmp_path / "llm_config.yaml"
    connect.write(proposal.entry, path, alias="local")
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    client = client_from_config("local", entry, api_key=None)
    try:
        messages = [{"role": "user", "content": "Hi"}]
        result = await client.acall(messages) if asynchronous else client.call(messages)
        assert result.content == "323"
        assert len(sent) == 1
    finally:
        await client.aclose()


async def test_explicit_credential_overrides_saved_no_auth(monkeypatch):
    from tests.unifiedllm.connect.connect_http import mock_http

    sent = []

    def respond(request):
        sent.append(request)
        assert request.headers["authorization"] == "Bearer supplied-key"
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, respond)
    plan = connect.plan("local", "model", "chat", "https://custom.example/v1", "", reply_tokens=128)
    result = await connect.run(plan, approved="minimal", api_key="supplied-key")
    assert "routing" in connect.verdict(result.entry).passed
    assert len(sent) == 1
