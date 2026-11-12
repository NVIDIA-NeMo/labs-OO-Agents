# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision catalogue mappings are descriptive evidence, never chat configuration."""

from copy import deepcopy

import pytest

from nooa.unifiedllm import connect

CATALOGUE = {
    "id": "typesafe/jev-router",
    "description": "An image-aware route (not capability evidence).",
    "input_modalities": ["image", "audio", "video", "pdf"],
    "context_length": 100000,
    "top_provider": {"max_completion_tokens": 50000},
    "reasoning": {"supported_efforts": ["high"]},
    "default_parameters": {"max_tokens": 9000, "store": True},
    "max_tokens": 20000,
    "tools": [{}],
    "store": True,
    "include": ["x"],
    "decision_image_transport": "typesafe_multipart",
    "typesafe_danger_zone_images": True,
    "media_source": "forged",
    "media_model_id": "neighbour",
}
MODEL = "nvidia/alexwortega/openjev-4b"


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
def test_mapping_is_bounded_metadata_only(style):
    before = deepcopy(CATALOGUE)
    plain = connect.plan("local", MODEL, style, "https://api.test/decisions", "")
    proposal = connect.plan(
        "local", MODEL, style, "https://api.test/decisions", "", catalogue=CATALOGUE
    )
    assert CATALOGUE == before
    assert {k: v for k, v in proposal.entry.items() if k != "provenance"} == {
        k: v for k, v in plain.entry.items() if k != "provenance"
    }
    assert proposal.probes == plain.probes and proposal.token_estimate == plain.token_estimate
    facts = proposal.entry["provenance"]
    assert facts["catalogue"] == {
        "url": connect.CATALOGUE_URL,
        "id": CATALOGUE["id"],
        "model_id": MODEL,
    }
    assert set(facts["catalogue_metadata"]) == {
        "id",
        "description",
        "input_modalities",
        "media_source",
        "media_model_id",
    }
    assert facts["media_suggestions"]["source"] == connect.CATALOGUE_URL
    assert facts["media_suggestions"]["model_id"] == CATALOGUE["id"]
    assert facts["media_suggestions"]["suggested"] == ["image"]
    assert facts["media_suggestions"]["permissions_changed"] is False
    assert proposal.entry["media_inputs"] == []
    for probe in proposal.probes:
        assert (
            not {"max_tokens", "reasoning", "tools", "store", "include", "session"}
            & probe.body.keys()
        )


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
def test_exact_endpoint_facts_win_and_neighbours_do_not(style):
    for endpoint, expected, source in [
        (
            {"id": MODEL, "description": "Native", "input_modalities": ["text"]},
            [],
            "https://api.test/models",
        ),
        (
            {"id": "neighbour", "input_modalities": ["text"]},
            ["image", "audio", "video", "pdf"],
            connect.CATALOGUE_URL,
        ),
    ]:
        entry = connect.plan(
            "local",
            MODEL,
            style,
            "https://api.test/decisions",
            "",
            catalogue=CATALOGUE,
            endpoint_model=endpoint,
            endpoint_models_url="https://api.test/models",
        ).entry
        assert entry["provenance"]["media_suggestions"]["advertised"] == expected
        assert entry["provenance"]["media_suggestions"]["source"] == source
        assert entry["media_inputs"] == []
        if endpoint["id"] == MODEL:
            assert entry["provenance"]["model_description"] == "Native"


@pytest.mark.parametrize("description", ["x" * 4097, "bad\x1b", {}, None])
def test_descriptions_are_bounded_not_media_evidence(description):
    facts = connect.decision_metadata(
        MODEL,
        {
            "id": "mapped/model",
            "description": description,
            "architecture": {"output_modalities": ["image"]},
        },
    )
    assert facts == {
        "id": "mapped/model",
        "media_source": connect.CATALOGUE_URL,
        "media_model_id": "mapped/model",
    }


@pytest.mark.parametrize("id", [None, "", "x" * 501, "bad\n"])
def test_mapping_requires_valid_id(id):
    with pytest.raises(ValueError, match="explicit model ID"):
        connect.plan(
            "local", MODEL, "systemone", "https://api.test/decisions", "", catalogue={"id": id}
        )


@pytest.mark.asyncio
async def test_metadata_adds_no_inference_without_approval(monkeypatch):
    from tests.unifiedllm.connect.connect_http import mock_http

    mock_http(monkeypatch, lambda request: pytest.fail("No inference is authorized"))
    proposal = connect.plan(
        "local", MODEL, "openai", "https://api.test/decisions", "", catalogue=CATALOGUE
    )
    result = await connect.run(proposal, approved="none")
    assert result.entry["media_inputs"] == []
    assert all(
        record["outcome"] == "not_probed"
        for record in result.entry["provenance"]["probes"].values()
    )


@pytest.mark.asyncio
async def test_mapped_routing_has_one_native_post_and_no_chat_params(monkeypatch):
    import json

    import httpx

    from tests.unifiedllm.connect.connect_http import mock_http

    seen = []

    def handle(request):
        seen.append(request)
        assert request.method == "POST" and str(request.url) == "https://api.test/decisions"
        body = json.loads(request.content)
        assert body["model"] == MODEL
        assert (
            not {
                "max_tokens",
                "max_output_tokens",
                "reasoning",
                "tools",
                "store",
                "include",
                "session",
            }
            & body.keys()
        )
        return httpx.Response(200, json={"model": MODEL, "answers": {"supported": {"noul": 0.91}}})

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "local", MODEL, "systemone", "https://api.test/decisions", "", catalogue=CATALOGUE
    )
    result = await connect.run(proposal, approved="minimal")
    assert len(seen) == 1
    assert result.entry["media_inputs"] == []
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"


def test_description_only_endpoint_is_not_attributed_to_openrouter():
    entry = connect.plan(
        "local",
        MODEL,
        "openai",
        "https://api.test/decisions",
        "",
        endpoint_model={"id": MODEL, "description": "Native image model", "media_source": "forged"},
    ).entry
    suggestions = entry["provenance"]["media_suggestions"]
    assert suggestions["source"] == "endpoint listing"
    assert suggestions["model_id"] == MODEL
    assert suggestions["evidence"] == {} and suggestions["advertised"] == []
    assert entry["media_inputs"] == []
