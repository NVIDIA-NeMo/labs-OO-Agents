# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route identity is derived once; request overrides still get validated."""

import json
from unittest.mock import patch

import httpx
import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient, connect
from nooa.unifiedllm.direct import DirectTransport


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_connect_saves_one_source_of_wire_style(style):
    entry = connect.plan("route", "custom-model", style, "https://api.example/v1", "").entry
    assert "api_style" not in entry
    assert (
        entry["model_name"]
        == ("anthropic/" if style == "anthropic" else "openai/") + "custom-model"
    )
    from nooa.unifiedllm.registry import client_from_config

    async with client_from_config("route", entry, api_key="test") as client:
        assert client._direct.api_style == style


@pytest.mark.parametrize("cls", [CompletionClient, ResponsesClient])
def test_constructor_rejects_removed_api_style(cls):
    with pytest.raises(TypeError, match="derived"):
        cls("openai/test", api_style="chat", api_key="test")


async def test_model_route_is_resolved_once_but_endpoint_guards_remain():
    with patch.object(
        DirectTransport, "_model_route", autospec=True, side_effect=DirectTransport._model_route
    ) as resolve:
        async with CompletionClient(
            "openai/test", transport="direct", api_key="test", api_base="https://example.test/v1"
        ) as client:
            route = client._direct
            params = {"api_base": "https://example.test/v1"}
            assert route.route(client.model, params) == ("test", "openai")
            assert route.route(client.model, params) == ("test", "openai")
            assert resolve.call_count == 1
            with pytest.raises(ValueError, match="Azure"):
                route.route(client.model, {"api_base": "https://resource.openai.azure.com"})
            with pytest.raises(ValueError, match="client/custom_llm_provider"):
                route.route(client.model, {**params, "client": object()})
            assert route.route("openai/other", params) == ("other", "openai")
            assert resolve.call_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_cached_route_keeps_call_specific_model_endpoint_and_key(monkeypatch, asynchronous):
    requests = []

    def respond(_, request):
        requests.append(
            (request.url.host, request.headers["authorization"], json.loads(request.content))
        )
        return httpx.Response(
            200,
            json={
                "id": "reply",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def arespond(transport, request):
        return respond(transport, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", arespond)
    async with CompletionClient(
        "openai/test",
        transport="direct",
        api_key="first-test-key",
        api_base="https://first.example/v1",
        max_tokens=32,
    ) as client:
        for overrides in (
            {},
            {
                "model": "openai/other",
                "api_base": "https://second.example/v1",
                "api_key": "second-test-key",
            },
            {},
        ):
            messages = [{"role": "user", "content": "test"}]
            response = (
                await client.acall(messages, **overrides)
                if asynchronous
                else client.call(messages, **overrides)
            )
            assert response.content == "ok"
    assert [host for host, _, _ in requests] == ["first.example", "second.example", "first.example"]
    assert [auth for _, auth, _ in requests] == [
        "Bearer first-test-key",
        "Bearer second-test-key",
        "Bearer first-test-key",
    ]
    assert [body["model"] for _, _, body in requests] == ["test", "other", "test"]
    assert all("api_style" not in body for _, _, body in requests)


async def test_cached_compatible_route_still_requires_endpoint():
    async with CompletionClient(
        "deepseek/test",
        transport="direct",
        api_key="test",
        api_base="https://example.test/v1",
    ) as client:
        with pytest.raises(ValueError, match="requires api_base"):
            client._direct.route(client.model, {})


async def test_gateway_anthropic_name_stays_chat():
    async with CompletionClient(
        "openai/anthropic/claude-test",
        transport="direct",
        api_key="test",
        api_base="https://example.test/v1",
    ) as client:
        assert client._direct.api_style == "chat"


@pytest.mark.parametrize("legacy_style", ["anthropic", "responses"])
def test_conflicting_legacy_registry_style_requires_migration(legacy_style):
    from nooa.unifiedllm.registry import client_from_config

    with pytest.raises(ValueError, match="no longer selects"):
        client_from_config(
            "old",
            {
                "model_name": "bare-model",
                "transport": "direct",
                "api_style": legacy_style,
                "api_key_env": "",
            },
            api_key="test",
        )
    with pytest.raises(ValueError, match="no longer selects"):
        connect.configure_entry({"model_name": "bare-model", "api_style": legacy_style})


async def test_connect_removes_consistent_legacy_style_without_changing_route():
    entry = connect.plan("old", "custom", "anthropic", "https://api.example/v1", "").entry
    entry["api_style"] = "anthropic"
    updated = connect.configure_entry(entry)
    assert "api_style" not in updated
    assert updated["model_name"] == "anthropic/custom"
