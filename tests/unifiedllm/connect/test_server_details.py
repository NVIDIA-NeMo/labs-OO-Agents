# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Published metadata fixtures: GET-only, exact IDs, bounded and no admin calls."""

import copy

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

MODEL = "nvidia/alexwortega/openjev-4b"
BASE = "https://inference-api.nvidia.com/v1"
SCHEMA = {
    "openapi": "3.1.0",
    "info": {"title": "LiteLLM API", "version": "1.105.1"},
    "paths": {
        "/model_group/info": {
            "get": {
                "parameters": [{"name": "model_group", "in": "query", "schema": {"type": "string"}}]
            }
        }
    },
    # Untrusted followup locations must never be used.
    "servers": [{"url": "https://evil.test"}],
}
GROUP = {
    "model_group": MODEL,
    "mode": "evaluation",
    "max_input_tokens": None,
    "max_output_tokens": None,
    "supported_openai_params": [],
    "supported_reasoning_efforts": None,
    "supports_vision": False,
    "supports_function_calling": False,
    "supports_reasoning": False,
    "litellm_params": {"api_key": "private-config", "api_base": "https://evil.test"},
}


def setup(monkeypatch, *, group=None, schema=None, status=200, listing=None, base=BASE):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.url.host == httpx.URL(base).host
        prefix = httpx.URL(base.removesuffix("/v1")).path.rstrip("/")
        if request.url.path == prefix + "/v1/models":
            assert request.headers["authorization"] == "Bearer work-key"
            return httpx.Response(
                200,
                json={
                    "data": listing
                    or [{"id": MODEL, "max_input_tokens": 1048576, "max_output_tokens": 128000}]
                },
            )
        if request.url.path == prefix + "/openapi.json":
            assert "authorization" not in request.headers
            return httpx.Response(200, json=SCHEMA if schema is None else schema)
        assert request.url.path == prefix + "/model_group/info"
        assert request.url.params["model_group"] == MODEL
        assert request.headers["authorization"] == "Bearer work-key"
        return httpx.Response(
            status,
            json={"data": [GROUP if group is None else group]},
            headers={"location": "https://evil.test"},
        )

    mock_http(monkeypatch, handle)
    return calls


@pytest.mark.asyncio
async def test_litellm_actual_null_false_contract_and_listing_reuse(monkeypatch):
    calls = setup(monkeypatch)
    listing = await connect.discover(BASE, api_key="work-key")
    before = copy.deepcopy(listing)
    details = await connect.server_details(
        BASE, MODEL, listing=listing, api_key="work-key", litellm=True
    )
    assert listing == before
    assert details.requests == 2 and len(calls) == 3
    assert details.model["max_input_tokens"] == 1048576
    assert details.model["max_output_tokens"] == 128000
    assert details.model["model_type"] == "decision"
    group = details.records[-1]
    assert group["source"] == BASE.removesuffix("/v1") + "/model_group/info"
    assert group["product"] == {"title": "LiteLLM API", "version": "1.105.1"}
    assert group["facts"]["supports_vision"] is False
    assert group["facts"]["supported_openai_params"] == []
    assert "supported_reasoning_efforts" not in group["facts"]
    assert "max_input_tokens" not in group["facts"]
    assert "private-config" not in repr(details) and "evil.test" not in repr(details)
    proposal = connect.plan(
        "local",
        MODEL,
        "chat",
        BASE,
        "KEY",
        endpoint_model=details.model,
        endpoint_models_url=listing.models_url,
        media_inputs=["image"],
        reply_tokens=2000,
    )
    assert proposal.entry["api_style"] == "chat"
    assert proposal.entry["media_inputs"] == ["image"]
    assert proposal.entry.get("allowed_openai_params") is None
    assert not proposal.entry.get("reasoning_levels")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values,expected",
    [
        ({"max_input_tokens": 2000000, "max_output_tokens": 256000}, (1048576, 128000)),
        ({"max_input_tokens": 60000, "max_output_tokens": 8000}, (60000, 8000)),
        ({"max_input_tokens": 16385.0, "max_output_tokens": 4096.0}, (16385, 4096)),
        ({"max_input_tokens": True, "max_output_tokens": "123"}, (1048576, 128000)),
        ({"max_input_tokens": 1.5, "max_output_tokens": -1}, (1048576, 128000)),
        ({"max_input_tokens": 0, "max_output_tokens": float("inf")}, (1048576, 128000)),
    ],
)
async def test_group_limits_only_tighten_and_validate(monkeypatch, values, expected):
    calls = setup(monkeypatch, group={**GROUP, **values})
    details = await connect.server_details(BASE, MODEL, api_key="work-key", litellm=True)
    assert (details.model["max_input_tokens"], details.model["max_output_tokens"]) == expected
    assert len(calls) == details.requests == 3


@pytest.mark.asyncio
async def test_total_listing_ceiling_and_false_flags_do_not_replace_media(monkeypatch):
    calls = setup(
        monkeypatch,
        listing=[
            {
                "id": MODEL,
                "context_window": 64000,
                "max_input_tokens": 60000,
                "capabilities": {"image_input": {"supported": True}},
            }
        ],
        group={
            **GROUP,
            "max_input_tokens": 2000000,
            "supports_reasoning": "false",
            "supported_reasoning_efforts": ["low", 1],
        },
    )
    details = await connect.server_details(BASE, MODEL, api_key="work-key", litellm=True)
    assert details.model["max_input_tokens"] == 60000
    assert details.model["capabilities"]["image_input"]["supported"] is True
    assert "supports_reasoning" not in details.records[-1]["facts"]
    assert "supported_reasoning_efforts" not in details.records[-1]["facts"]
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 302])
async def test_group_denials_continue_no_redirect_or_retry(monkeypatch, status):
    calls = setup(monkeypatch, status=status)
    details = await connect.server_details(BASE, MODEL, api_key="work-key", litellm=True)
    assert len(calls) == 3 and details.requests == 3
    assert details.model["max_input_tokens"] == 1048576
    assert len(details.records) == 1
    assert f"HTTP {status}" in details.notices[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"info": {"title": "Not LiteLLM"}, "paths": SCHEMA["paths"]},
        {"info": SCHEMA["info"], "paths": {"/model_group/info": {"post": {}}}},
    ],
)
async def test_only_validated_product_get_contract_authorizes_group(monkeypatch, schema):
    calls = setup(monkeypatch, schema=schema)
    details = await connect.server_details(BASE, MODEL, api_key="work-key", litellm=True)
    assert len(calls) == 2 and details.requests == 2 and details.notices


@pytest.mark.asyncio
async def test_version_header_not_required_prefix_preserved_and_wrong_model_ignored(monkeypatch):
    schema = copy.deepcopy(SCHEMA)
    schema["info"].pop("version")
    base = "https://api.test/team/deployment/v1"
    calls = setup(
        monkeypatch,
        schema=schema,
        group={**GROUP, "model_group": "neighbour", "max_input_tokens": 1},
        base=base,
    )
    details = await connect.server_details(base, MODEL, api_key="work-key", litellm=True)
    assert details.model["max_input_tokens"] == 1048576
    assert len(calls) == 3 and details.notices
    assert all("/team/deployment/" in request.url.path for request in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "style,base,item",
    [
        (
            "responses",
            "https://api.openai.com/v1",
            {"id": "gpt-test", "object": "model", "owned_by": "openai", "created": 1},
        ),
        (
            "anthropic",
            "https://api.anthropic.com/v1",
            {
                "id": "claude-test",
                "display_name": "Claude Test",
                "max_input_tokens": 200000,
                "max_tokens": 64000,
                "capabilities": {
                    "image_input": {"supported": True},
                    "pdf_input": {"supported": False},
                },
            },
        ),
    ],
)
async def test_native_shapes_and_known_style_headers_no_invented_calls(
    monkeypatch, style, base, item
):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET" and request.url.path == "/v1/models"
        if style == "anthropic":
            assert request.headers["x-api-key"] == "native-key"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == "Bearer native-key"
            assert "x-api-key" not in request.headers
        return httpx.Response(200, json={"data": [item], "has_more": False})

    mock_http(monkeypatch, handle)
    details = await connect.server_details(base, item["id"], api_style=style, api_key="native-key")
    assert len(calls) == details.requests == 1
    assert not details.notices
    if style == "anthropic":
        assert details.model["max_output_tokens"] == 64000
        assert details.records[0]["fields"]["max_output_tokens"] == "data[].max_tokens"
        assert details.model["capabilities"]["pdf_input"]["supported"] is False
    else:
        assert details.model == {"id": "gpt-test"}


@pytest.mark.asyncio
async def test_openrouter_public_exact_metadata_no_key_and_no_route_config(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == connect.CATALOGUE_URL and request.method == "GET"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "openai/gpt-test",
                        "context_length": 200000,
                        "top_provider": {
                            "context_length": 128000,
                            "max_completion_tokens": 8000,
                            "api_key": "private",
                        },
                        "architecture": {"input_modalities": ["text", "image"]},
                        "supported_parameters": ["reasoning"],
                        "reasoning": {
                            "supported_efforts": ["high"],
                            "default_effort": "high",
                            "api_key": "private",
                        },
                        "api_key": "private",
                        "api_base": "https://evil.test",
                        "media_source": "forged",
                    }
                ]
            },
        )

    mock_http(monkeypatch, handle)
    details = await connect.server_details(
        "https://openrouter.ai/api/v1", "openai/gpt-test", api_key="inference-key"
    )
    assert len(calls) == details.requests == 1
    assert details.model["context_window"] == 128000 and details.model["max_output_tokens"] == 8000
    assert details.catalogue_model["id"] == "openai/gpt-test"
    assert details.records[0]["facts"]["context_length"] == 200000
    assert details.records[0]["facts"]["top_provider"]["context_length"] == 128000
    assert (
        "private" not in repr(details)
        and "evil.test" not in repr(details)
        and "forged" not in repr(details)
    )
    missing = await connect.server_details("https://openrouter.ai/api/v1", "gpt-test")
    assert missing.model == {"id": "gpt-test"} and missing.catalogue_model is None


@pytest.mark.asyncio
async def test_bypass_and_scope_isolation(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.host == "catalogue.test"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"id": MODEL, "max_input_tokens": 123}]})

    mock_http(monkeypatch, handle)
    ignored = connect.Discovery("https://evil.test/v1", ({"id": MODEL, "max_input_tokens": 1},))
    details = await connect.server_details(
        BASE, MODEL, api_key="work-key", listing=ignored, metadata=False, litellm=True
    )
    assert details.model == {"id": MODEL} and details.requests == 0 and not calls
    with pytest.raises(ValueError, match="different connection"):
        await connect.server_details(BASE, MODEL, listing=ignored)
    with pytest.raises(ValueError, match="explicit listing key"):
        await connect.server_details(
            BASE, MODEL, api_key="work-key", models_url="https://catalogue.test/models"
        )
    result = await connect.server_details(
        BASE,
        MODEL,
        api_key="work-key",
        listing_api_key=None,
        models_url="https://catalogue.test/models",
    )
    assert len(calls) == 1 and result.model["max_input_tokens"] == 123
    replay = connect.Discovery(
        BASE,
        (
            {
                "id": MODEL,
                "api_key": "private-config",
                "mode": "evaluation",
                "api_base": "https://evil.test",
            },
        ),
        BASE + "/models",
    )
    clean = await connect.server_details(BASE, MODEL, listing=replay)
    assert clean.model == {"id": MODEL} and clean.requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["bad-json", "oversize", "timeout", "too-many"])
async def test_metadata_bounds_and_errors_are_optional(monkeypatch, kind):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MODEL, "max_input_tokens": 123}]})
        if kind == "bad-json":
            return httpx.Response(200, text="not-json work-key")
        if kind == "oversize":
            return httpx.Response(200, content=b" " * (5 * 1024 * 1024 + 1))
        if kind == "timeout":
            raise httpx.ReadTimeout("work-key private query", request=request)
        if request.url.path.endswith("/openapi.json"):
            return httpx.Response(200, json=SCHEMA)
        return httpx.Response(200, json={"data": [GROUP] * 5001})

    mock_http(monkeypatch, handle)
    details = await connect.server_details(BASE, MODEL, api_key="work-key", litellm=True)
    assert details.model["max_input_tokens"] == 123 and details.notices
    assert "work-key" not in repr(details) and "private query" not in repr(details)
    assert details.requests == len(calls) == (3 if kind == "too-many" else 2)


@pytest.mark.asyncio
async def test_root_listing_fallback_reports_actual_get_count(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return (
            httpx.Response(404)
            if request.url.path == "/models"
            else httpx.Response(200, json={"data": [{"id": MODEL}]})
        )

    mock_http(monkeypatch, handle)
    details = await connect.server_details("https://api.test", MODEL)
    assert details.requests == len(calls) == 2


@pytest.mark.asyncio
async def test_reused_raw_context_length_clamps_group_ceiling(monkeypatch):
    calls = setup(monkeypatch, group={**GROUP, "max_input_tokens": 1000000.0})
    listing = connect.Discovery(BASE, ({"id": MODEL, "context_length": 100},), BASE + "/models")
    details = await connect.server_details(
        BASE, MODEL, listing=listing, api_key="work-key", litellm=True
    )
    assert details.model["max_input_tokens"] == 100
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_listing_url_binding_and_known_root_v1_reuse(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("Reused listing must not fetch"))
    listing = connect.Discovery(
        "https://api.test/v1", ({"id": MODEL},), "https://api.test/v1/models"
    )
    result = await connect.server_details("https://api.test", MODEL, listing=listing)
    assert result.requests == 0
    mismatch = connect.Discovery(BASE, ({"id": MODEL},), BASE + "/other-models")
    with pytest.raises(ValueError, match="listing URL"):
        await connect.server_details(BASE, MODEL, listing=mismatch)


@pytest.mark.parametrize(
    "style,base",
    [("anthropic", "https://api.anthropic.com/v1"), ("chat", "https://openrouter.ai/api/v1")],
)
def test_library_plan_never_imports_foreign_published_profiles(style, base):
    facts = {
        "id": "other/model",
        "context_length": 128000,
        "reasoning": {"supported_efforts": ["high"], "default_effort": "high"},
        "default_parameters": {"max_tokens": 6000},
        "pricing": {"prompt": "0.001", "completion": "0.002"},
    }
    original = copy.deepcopy(facts)
    proposal = connect.plan("local", "wire-model", style, base, "", catalogue=facts)
    assert proposal.entry["context_window"] == 128000
    assert not proposal.entry.get("reasoning_levels") and "reasoning_default" not in proposal.entry
    assert proposal.entry["max_tokens"] != 6000 and proposal.price_estimate is None
    assert facts == original
    explicit = connect.plan(
        "local",
        "wire-model",
        style,
        base,
        "",
        catalogue=facts,
        reasoning_levels={"chosen": {"temperature": 0.5}},
        reply_tokens=1234,
    )
    assert explicit.entry["reasoning_levels"] == {"chosen": {"temperature": 0.5}}
    assert explicit.entry["max_tokens"] == 1234


@pytest.mark.asyncio
async def test_legacy_listing_implied_source_not_relabelled_or_reused_elsewhere(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("Legacy reuse must not fetch"))
    listing = connect.Discovery("https://api.test/v1", ({"id": "model", "max_input_tokens": 123},))
    details = await connect.server_details("https://api.test", "model", listing=listing)
    assert details.records[0]["source"] == "https://api.test/v1/models"
    with pytest.raises(ValueError, match="listing URL"):
        await connect.server_details(
            "https://api.test/v1",
            "model",
            listing=listing,
            models_url="https://api.test/custom-models",
        )


@pytest.mark.asyncio
async def test_public_library_result_scrubs_both_active_credential_scopes(monkeypatch):
    schema = copy.deepcopy(SCHEMA)
    schema["info"]["version"] = "separate-listing-key"
    setup(
        monkeypatch,
        schema=schema,
        group={**GROUP, "supported_reasoning_efforts": ["separate-listing-key", "work-key"]},
    )
    listing = connect.Discovery(BASE, ({"id": MODEL},), BASE + "/models")
    details = await connect.server_details(
        BASE,
        MODEL,
        listing=listing,
        api_key="work-key",
        listing_api_key="separate-listing-key",
        litellm=True,
    )
    assert "work-key" not in repr(details) and "separate-listing-key" not in repr(details)
    assert details.model_id == MODEL and details.requests == 2
