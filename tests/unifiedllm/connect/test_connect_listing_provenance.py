# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Limits have exact-model, independently verified listing attribution."""

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

SOURCE = "https://catalogue.example/available"


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_live_limits_only_listing_uses_full_source(monkeypatch, style):
    sent = []

    def handle(request):
        sent.append(request)
        assert request.method == "GET"
        assert str(request.url).split("?")[0] == SOURCE
        return httpx.Response(200, json={"data": [{"id": "model", "max_input_tokens": 12345}]})

    mock_http(monkeypatch, handle)
    found = await connect.discover(
        "https://inference.example/v1/", api_style=style, models_url=SOURCE
    )
    assert "media_source" not in found.models[0]
    entry = connect.plan(
        "local",
        "model",
        style,
        found.api_base,
        "",
        endpoint_model=found.models[0],
        endpoint_models_url=found.models_url,
    ).entry
    assert entry["context_window"] == 12345
    assert entry["provenance"]["endpoint_limits"] == {"url": SOURCE, "max_input_tokens": 12345}
    assert len(sent) == 1


async def test_anthropic_normalized_inference_base_does_not_rewrite_listing(monkeypatch):
    mock_http(
        monkeypatch,
        lambda r: httpx.Response(200, json={"data": [{"id": "model", "context_window": 24000}]}),
    )
    found = await connect.discover("https://api.anthropic.com/v1/", api_style="anthropic")
    entry = connect.plan(
        "local",
        "model",
        "anthropic",
        found.api_base,
        "",
        endpoint_model=found.models[0],
        endpoint_models_url=found.models_url,
    ).entry
    assert entry["api_base"] == "https://api.anthropic.com"
    assert entry["provenance"]["endpoint_limits"]["url"] == "https://api.anthropic.com/v1/models"


@pytest.mark.parametrize("style", ["chat", "anthropic", "openai", "systemone"])
@pytest.mark.parametrize("catalogue", [None, {"id": "mapped/model", "context_length": 9876}])
def test_mismatched_model_cannot_contribute_limits_or_media(style, catalogue):
    entry = connect.plan(
        "local",
        "requested",
        style,
        "https://inference.example/v1",
        "",
        catalogue=catalogue,
        endpoint_model={
            "id": "neighbour",
            "context_window": 99999,
            "max_output_tokens": 99999,
            "input_modalities": ["image"],
        },
        endpoint_models_url=SOURCE,
    ).entry
    assert "endpoint_limits" not in entry["provenance"]
    assert entry.get("context_window") == (
        9876 if catalogue and style not in connect.DECISION_API_STYLES else None
    )
    assert "max_output_tokens" not in entry
    assert "image" not in entry.get("media_inputs", [])


def test_record_source_keys_do_not_override_independent_source():
    model = {
        "id": "model",
        "context_window": 12345,
        "media_source": "https://forged.example/models",
        "models_url": "https://forged.example/models",
        "source": "https://forged.example/models",
        "endpoint_limits": {"url": "https://forged.example/models"},
    }
    for source, expected in [(SOURCE, SOURCE), (None, "https://inference.example/v1/models")]:
        entry = connect.plan(
            "local",
            "model",
            "chat",
            "https://inference.example/v1",
            "",
            endpoint_model=model,
            endpoint_models_url=source,
        ).entry
        assert entry["provenance"]["endpoint_limits"] == {"url": expected, "context_window": 12345}


def test_unknown_limits_stay_unknown_even_with_verified_source():
    entry = connect.plan(
        "local",
        "model",
        "chat",
        "https://inference.example/v1",
        "",
        endpoint_model={"id": "model"},
        endpoint_models_url=SOURCE,
    ).entry
    assert "endpoint_limits" not in entry["provenance"]
    assert "context_window" not in entry
    assert "max_output_tokens" not in entry


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic", "openai", "systemone"])
@pytest.mark.parametrize("source", [SOURCE, None])
def test_independent_source_and_exact_id_override_record_media_attribution(style, source):
    entry = connect.plan(
        "local",
        "model",
        style,
        "https://inference.example/v1",
        "",
        endpoint_model={
            "id": "model",
            "input_modalities": ["image"],
            "media_source": "https://forged.example/models",
            "media_model_id": "neighbour",
        },
        endpoint_models_url=source,
    ).entry
    hints = entry["provenance"]["media_suggestions"]
    assert hints["source"] == (source or "endpoint listing")
    assert hints["model_id"] == "model"
    assert "image" not in entry.get("media_inputs", [])


async def test_live_catalogue_cannot_inject_internal_endpoint_or_media_attribution(monkeypatch):
    def handle(request):
        assert request.method == "GET"
        assert str(request.url) == connect.CATALOGUE_URL
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "vendor/model",
                        "context_length": 9876,
                        "input_modalities": ["image"],
                        "endpoint_limits": {
                            "url": "https://forged.example/models",
                            "context_window": 99999,
                        },
                        "limit_sources": {"context_length": "endpoint"},
                        "media_source": "https://forged.example/models",
                        "media_model_id": "neighbour",
                    }
                ]
            },
        )

    mock_http(monkeypatch, handle)
    record = (await connect.catalogue())[0]
    assert "endpoint_limits" not in record
    assert "limit_sources" not in record
    assert "media_model_id" not in record
    assert record["media_source"] == connect.CATALOGUE_URL
    for endpoint_model in [None, {"id": "model", "context_window": 12345}]:
        entry = connect.plan(
            "local",
            "model",
            "chat",
            "https://inference.example/v1",
            "",
            catalogue=record,
            endpoint_model=endpoint_model,
            endpoint_models_url=SOURCE,
        ).entry
        assert entry["context_window"] == (12345 if endpoint_model else 9876)
        if endpoint_model:
            assert entry["provenance"]["endpoint_limits"] == {
                "url": SOURCE,
                "context_window": 12345,
            }
        else:
            assert "endpoint_limits" not in entry["provenance"]
            assert entry["provenance"]["limit_sources"]["context_length"] == "catalogue"
        assert entry["provenance"]["media_suggestions"]["source"] == connect.CATALOGUE_URL
        assert entry["provenance"]["media_suggestions"]["model_id"] == "vendor/model"
