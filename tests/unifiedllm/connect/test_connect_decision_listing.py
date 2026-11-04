# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A shared catalogue is not the decision inference interface."""

from dataclasses import asdict

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

ENDPOINT = "https://hub.test/v1/decisions"
MODELS_URL = "https://hub.test/catalogue/models"
MODEL = {
    "id": "decision/model",
    "name": "Decision model",
    "client_type": "decision",
    "supported_interfaces": ["systemone", "openai"],
    "input_modalities": ["image", "text"],
    "output_modalities": ["decision"],
    "architecture": {
        "input_modalities": ["video"],
        "output_modalities": ["decision"],
        "private": "ignored",
    },
    "context_length": 8192,
    "max_input_tokens": 4096,
    "max_output_tokens": 1024,
    "api_key": "must-not-retain",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
async def test_explicit_decision_listing_preserves_hints_not_chat_settings(monkeypatch, style):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == MODELS_URL
        assert request.headers["authorization"] == "Bearer listing-key"
        return httpx.Response(200, json={"data": [MODEL]})

    mock_http(monkeypatch, handle)
    found = await connect.discover(
        ENDPOINT,
        api_style=style,
        models_url=MODELS_URL,
        listing_style="chat",
        api_key="listing-key",
    )
    assert found.api_base == ENDPOINT
    assert asdict(found)["models_url"] == MODELS_URL
    advertised = found.models[0]
    assert advertised["id"] == MODEL["id"]
    assert advertised["context_window"] == 8192
    assert advertised["max_input_tokens"] == 4096
    assert advertised["supported_interfaces"] == ["systemone", "openai"]
    assert advertised["input_modalities"] == ["image", "text"]
    assert advertised["architecture"] == {
        "input_modalities": ["video"],
        "output_modalities": ["decision"],
    }
    assert "must-not-retain" not in repr(found)
    proposal = connect.plan(
        "local", MODEL["id"], style, ENDPOINT, "INFERENCE_KEY", endpoint_model=advertised
    )
    entry = proposal.entry
    assert entry["endpoint"] == ENDPOINT
    assert entry["model_name"] == MODEL["id"]
    assert entry["provenance"]["advertised_model"] == advertised
    assert entry["provenance"]["probes"] == {}
    assert entry["provenance"]["requests_accepted"] == []
    assert entry["media_inputs"] == []
    assert not {
        "context_window",
        "max_tokens",
        "max_output_tokens",
        "reasoning_levels",
    }.intersection(entry)
    advertised["input_modalities"].append("audio")
    assert entry["provenance"]["advertised_model"]["input_modalities"] == ["image", "text"]
    assert len(requests) == 1  # Listing and planning never invoke inference.


@pytest.mark.asyncio
@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
async def test_decision_without_listing_url_never_attempts_http(monkeypatch, style):
    mock_http(monkeypatch, lambda request: pytest.fail("No guessed /models requests"))
    with pytest.raises(ValueError, match="models_url"):
        await connect.discover(ENDPOINT, api_style=style, listing_style="chat")


@pytest.mark.asyncio
async def test_listing_style_selects_anthropic_auth_and_pagination(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/catalogue/models"
        assert request.headers["x-api-key"] == "catalogue-key"
        assert "authorization" not in request.headers
        assert request.headers["anthropic-version"] == "2023-06-01"
        if len(requests) == 1:
            return httpx.Response(
                200, json={"data": [{"id": "a"}], "has_more": True, "last_id": "a"}
            )
        assert request.url.params["after_id"] == "a"
        return httpx.Response(200, json={"data": [{"id": "b"}]})

    mock_http(monkeypatch, handle)
    found = await connect.discover(
        ENDPOINT,
        api_style="openai",
        models_url=MODELS_URL,
        listing_style="anthropic",
        api_key="catalogue-key",
    )
    assert [m["id"] for m in found.models] == ["a", "b"]
    assert found.api_base == ENDPOINT


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 302, 401])
async def test_explicit_listing_never_falls_back_follows_redirects_or_retries(monkeypatch, status):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://other.test/models"},
            text="secret-provider-message",
        )

    mock_http(monkeypatch, handle)
    # Even an explicit root URL is used exactly, with no /models or /v1 suffix.
    with pytest.raises(connect.DiscoveryError) as exc:
        await connect.discover(ENDPOINT, api_style="systemone", models_url="https://hub.test/")
    assert exc.value.status_code == status
    assert "secret-provider-message" not in str(exc.value)
    assert [str(r.url) for r in requests] == ["https://hub.test/"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@hub.test/models",
        "http://remote.test/models",
        "https://hub.test/models?key=secret",
        "https://hub.test/#fragment",
    ],
)
async def test_invalid_listing_urls_fail_before_http(monkeypatch, url):
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP expected"))
    with pytest.raises(ValueError):
        await connect.discover(ENDPOINT, api_style="systemone", models_url=url)


@pytest.mark.asyncio
async def test_invalid_listing_style_fails_before_http(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP expected"))
    with pytest.raises(ValueError, match="listing_style"):
        await connect.discover(
            ENDPOINT, api_style="systemone", models_url=MODELS_URL, listing_style="openai"
        )  # Inference openai != listing style.


@pytest.mark.asyncio
async def test_invalid_advertised_hints_are_not_retained(monkeypatch):
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model",
                        "input_modalities": ["image", 2],
                        "supported_interfaces": "systemone",
                        "description": "unsafe\x01",
                        "context_window": True,
                        "max_input_tokens": -1,
                        "architecture": {"input_modalities": ["bad\x00"]},
                    }
                ]
            },
        ),
    )
    found = await connect.discover(ENDPOINT, api_style="systemone", models_url=MODELS_URL)
    assert found.models == ({"id": "model"},)
