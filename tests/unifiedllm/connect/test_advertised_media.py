# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advertisements are bounded evidence, never runtime permissions or paid checks."""

from copy import deepcopy

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

SOURCE = "https://api.test/v1/models"


def metadata(modalities):
    return {"id": "model", "architecture": {"input_modalities": modalities}, "media_source": SOURCE}


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"capabilities": None},
        {"description": "image pdf"},
        {"supports_image": True},
        {"nested": {"capabilities": {"pdf_input": {"supported": True}}}},
    ],
)
def test_missing_and_unknown_are_manual(value):
    result = connect.advertised_media_suggestions(value)
    assert result["advertised"] == result["suggested"] == []
    assert result["notes"] == [
        "Media support isn't listed. You can test inputs or choose them yourself."
    ]
    assert result["permissions_changed"] is False


@pytest.mark.parametrize(
    "image,pdf,expected",
    [
        (True, False, ["image"]),
        (False, True, ["pdf"]),
        (False, False, []),
        (True, True, ["image", "pdf"]),
    ],
)
def test_exact_anthropic_sdk_flags(image, pdf, expected):
    value = {
        "id": "claude",
        "capabilities": {"image_input": {"supported": image}, "pdf_input": {"supported": pdf}},
    }
    before = deepcopy(value)
    result = connect.advertised_media_suggestions(value, source=SOURCE, api_style="anthropic")
    assert result["advertised"] == result["suggested"] == expected
    assert result["evidence"] == {
        "capabilities.image_input.supported": image,
        "capabilities.pdf_input.supported": pdf,
    }
    assert result["source"] == SOURCE and result["model_id"] == "claude"
    assert value == before


@pytest.mark.parametrize("bad", [1, 0, "true", "false", None, [], {"value": True}])
def test_flags_do_not_use_truthiness_or_recursion(bad):
    value = {
        "capabilities": {
            "image_input": {"supported": bad},
            "pdf_input": {"supported": bad},
            "audio_input": {"supported": True},
        }
    }
    assert connect.advertised_media_suggestions(value)["suggested"] == []


@pytest.mark.parametrize(
    "modalities,expected",
    [
        (["text", "image", "audio", "video", "file"], ["image", "audio", "video"]),
        (["file"], []),
        (["pdf", "file"], ["pdf"]),
        (["text", "unknown"], []),
    ],
)
def test_openrouter_categories_file_is_not_pdf(modalities, expected):
    result = connect.advertised_media_suggestions(metadata(modalities), source=SOURCE)
    assert result["advertised"] == result["suggested"] == expected
    if "file" in modalities:
        assert any(
            (
                "Files are listed separately from PDFs"
                if "pdf" in modalities
                else "PDF support isn't listed"
            )
            in note
            for note in result["notes"]
        )
    if "unknown" in modalities:
        assert any("Inputs unavailable in Connect" in note for note in result["notes"])


@pytest.mark.parametrize("bad", [["image"] * 101, ["image", 4], "image", ["image\n"], ["x" * 201]])
def test_bounds(bad):
    result = connect.advertised_media_suggestions(metadata(bad), source="bad\nsource")
    assert result["suggested"] == [] and result["source"] == "unknown"


@pytest.mark.parametrize("style", ["responses", "anthropic"])
def test_audio_video_advertised_but_checks_unavailable(style):
    result = connect.advertised_media_suggestions(
        metadata(["image", "audio", "video", "pdf"]), api_style=style
    )
    assert result["suggested"] == ["image", "audio", "video", "pdf"]
    assert result["checkable"] == ["image", "pdf"]
    assert set(result["unavailable_checks"]) == {"audio", "video"}


@pytest.mark.parametrize(
    "style,transport,optin,checkable",
    [
        ("openai", "json", False, ["image"]),
        ("systemone", "json", False, []),
        ("systemone", "typesafe_multipart", False, []),
        ("systemone", "typesafe_multipart", True, ["image"]),
    ],
)
def test_decision_transport_limits(style, transport, optin, checkable):
    result = connect.advertised_media_suggestions(
        metadata(["image", "audio", "video", "pdf"]),
        api_style=style,
        decision_image_transport=transport,
        typesafe_danger_zone_images=optin,
    )
    assert result["advertised"] == ["image", "audio", "video", "pdf"]
    assert result["suggested"] == ["image"]
    assert result["checkable"] == checkable
    if not checkable:
        assert (
            "Image testing isn't available for this connection"
            in result["unavailable_checks"]["image"]
        )
        assert (
            "OpenAI Decisions or direct TypeSafe multipart" in result["unavailable_checks"]["image"]
        )


@pytest.mark.parametrize("permissions", [None, [], ["pdf"], ["audio", "video"]])
def test_plan_does_not_adopt_suggestions_or_add_checks(permissions):
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        catalogue=metadata(["image", "audio", "video", "file"]),
        media_inputs=permissions,
    )
    assert proposal.entry["media_inputs"] == (permissions or [])
    assert proposal.entry["provenance"]["media_suggestions"]["suggested"] == [
        "image",
        "audio",
        "video",
    ]
    assert [probe.name for probe in proposal.probes] == ["routing", "tools"]


def test_same_route_edits_preserve_explicit_permissions_and_native_metadata_wins():
    existing = connect.plan(
        "local", "model", "anthropic", "https://api.test/v1", "", media_inputs=["pdf"]
    ).entry
    native = {
        "id": "model",
        "capabilities": {"image_input": {"supported": False}, "pdf_input": {"supported": True}},
        "media_source": SOURCE,
    }
    catalogue = metadata(["image", "audio"])
    result = connect.plan(
        "local",
        "model",
        "anthropic",
        "https://api.test/v1",
        "",
        existing_entry=existing,
        catalogue=catalogue,
        endpoint_model=native,
    )
    assert result.entry["media_inputs"] == ["pdf"]
    assert result.entry["provenance"]["media_suggestions"]["suggested"] == ["pdf"]
    wrong = connect.plan(
        "local",
        "model",
        "anthropic",
        "https://api.test/v1",
        "",
        catalogue=catalogue,
        endpoint_model={**native, "id": "neighbour"},
    )
    assert wrong.entry["provenance"]["media_suggestions"]["suggested"] == ["image", "audio"]


@pytest.mark.parametrize("style", ["chat", "anthropic"])
async def test_discovery_preserves_known_fields_and_limits(style, monkeypatch):
    item = {
        "id": "model",
        "max_input_tokens": 10000,
        "max_tokens": 1234,
        "architecture": {"input_modalities": ["image", "file"], "output_modalities": ["text"]},
        "capabilities": {
            "image_input": {"supported": False, "secret": "discard"},
            "pdf_input": {"supported": True},
            "unknown": {"supported": True},
        },
    }
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [item]}))
    discovery = await connect.discover("https://api.test/v1", api_style=style)
    model = discovery.models[0]
    assert model["architecture"]["output_modalities"] == ["text"]
    assert model["capabilities"] == {
        "image_input": {"supported": False},
        "pdf_input": {"supported": True},
    }
    assert model["media_source"] == SOURCE
    assert ("max_output_tokens" in model) == (style == "anthropic")
    if style == "anthropic":
        assert model["max_output_tokens"] == 1234
    assert connect.advertised_media_suggestions(model)["suggested"] == ["pdf"]


async def test_catalogue_keeps_architecture_without_generation(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET" and str(request.url) == connect.CATALOGUE_URL
        return httpx.Response(200, json={"data": [metadata(["image", "file"])]})

    mock_http(monkeypatch, handle)
    models = await connect.catalogue()
    assert models[0]["architecture"] == {"input_modalities": ["image", "file"]}
    assert models[0]["media_source"] == connect.CATALOGUE_URL
    assert len(calls) == 1


@pytest.mark.parametrize("supported", [True, False])
def test_file_note_respects_specific_pdf_flags(supported):
    result = connect.advertised_media_suggestions(
        {"input_modalities": ["file"], "capabilities": {"pdf_input": {"supported": supported}}}
    )
    assert result["suggested"] == (["pdf"] if supported else [])
    assert not any("PDF support isn't listed" in note for note in result["notes"])
    assert any("Files are listed separately from PDFs" in note for note in result["notes"])
