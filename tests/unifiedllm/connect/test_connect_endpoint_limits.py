# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint limits outrank third-party metadata without inventing total context."""

from copy import deepcopy

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

CATALOGUE = {
    "id": "vendor/model",
    "context_length": 200000,
    "top_provider": {"max_completion_tokens": 180000},
}


@pytest.mark.parametrize(
    "endpoint,window,ceiling,source",
    [
        (
            {"max_input_tokens": 100000, "max_output_tokens": 16000},
            100000,
            16000,
            "endpoint_input_limit",
        ),
        ({"context_window": 120000, "max_output_tokens": 24000}, 120000, 24000, "endpoint"),
        ({"max_input_tokens": 100000}, 100000, 180000, "endpoint_input_limit"),
        ({"max_input_tokens": True, "max_output_tokens": -1}, 200000, 180000, "catalogue"),
    ],
)
def test_plan_prefers_endpoint_limits(endpoint, window, ceiling, source):
    before = deepcopy(CATALOGUE)
    entry = connect.plan(
        "local",
        "vendor/model",
        "chat",
        "https://models.example/v1",
        "",
        catalogue=CATALOGUE,
        endpoint_model=endpoint,
    ).entry
    assert entry["context_window"] == window
    assert entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == ceiling
    assert entry["provenance"]["limit_sources"]["context_length"] == source
    assert entry["max_tokens"] <= min(window, ceiling)
    assert CATALOGUE == before


@pytest.mark.asyncio
async def test_discovery_keeps_input_and_context_distinct(monkeypatch):
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model",
                        "max_input_tokens": 100000,
                        "context_window": 120000,
                        "max_output_tokens": 16000,
                    }
                ]
            },
        ),
    )
    found = await connect.discover("https://models.example/v1")
    assert found.models[0]["max_input_tokens"] == 100000
    assert found.models[0]["context_window"] == 120000
    merged = connect.model_metadata("model", endpoint_model=found.models[0])
    assert merged["context_length"] == 100000
    assert merged["limit_sources"]["context_length"] == "endpoint_input_limit"
    assert merged["endpoint_limits"]["context_window"] == 120000


def test_unknown_context_is_visible_in_library_plan():
    entry = connect.plan("local", "model", "chat", "https://models.example/v1", "").entry
    assert "Context window unknown" in entry["provenance"]["warnings"][0]
