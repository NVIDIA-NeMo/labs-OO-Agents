# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision image permission/encoding is validated and route-local in YAML."""

import httpx
import pytest
from pydantic import ValidationError

from nooa.config.model_config import ModelConfig
from nooa.unifiedllm.registry import get_decision_model, reload_registry


@pytest.fixture
def decision_alias(tmp_path):
    path = tmp_path / "llm_config.yaml"
    path.write_text("""models:
  vision-decisions:
    client_type: decision
    model_name: jev-preview
    api_style: systemone
    endpoint: https://verified.example/system-one
    media_inputs: [image]
    decision_image_transport: typesafe_multipart
    typesafe_danger_zone_images: true
""")
    reload_registry(path)
    yield "vision-decisions"
    reload_registry()


@pytest.mark.asyncio
async def test_yaml_permissions_and_explicit_transport_reach_client(decision_alias):
    client = get_decision_model(decision_alias)
    assert client.media_inputs == ("image",)
    assert client.decision_image_transport == "typesafe_multipart"
    assert client.typesafe_danger_zone_images is True
    await client.aclose()


@pytest.mark.parametrize(
    "overrides",
    [
        {"endpoint": "https://replacement.example/decisions"},
        {"model": "replacement-model"},
        {"api_style": "openai"},
    ],
)
@pytest.mark.asyncio
async def test_replacement_route_clears_inherited_image_configuration(decision_alias, overrides):
    client = get_decision_model(decision_alias, **overrides)
    assert client.media_inputs == ()
    assert client.decision_image_transport == "json"
    assert client.typesafe_danger_zone_images is False
    await client.aclose()


@pytest.mark.asyncio
async def test_caller_client_clears_permission_unless_explicitly_redeclared(decision_alias):
    async with httpx.AsyncClient() as http:
        client = get_decision_model(decision_alias, client=http)
        assert client.media_inputs == ()
        assert client.decision_image_transport == "json"
        client = get_decision_model(
            decision_alias,
            client=http,
            media_inputs=["image"],
            decision_image_transport="typesafe_multipart",
            typesafe_danger_zone_images=True,
        )
        assert client.media_inputs == ("image",)
        assert client.decision_image_transport == "typesafe_multipart"
        assert client.typesafe_danger_zone_images is True


@pytest.mark.asyncio
async def test_explicit_permission_override_wins(decision_alias):
    client = get_decision_model(decision_alias, media_inputs=[])
    assert client.media_inputs == ()
    await client.aclose()


@pytest.mark.parametrize("permissions", [["audio"], ["image", "image"], "image"])
def test_registry_rejects_unsupported_or_malformed_permissions(decision_alias, permissions):
    with pytest.raises(ValueError):
        get_decision_model(decision_alias, media_inputs=permissions)


def test_typed_config_transport_fields_and_strict_opt_in():
    config = ModelConfig.from_registry(
        "native",
        {
            "client_type": "decision",
            "media_inputs": ["image"],
            "decision_image_transport": "typesafe_multipart",
            "typesafe_danger_zone_images": True,
        },
    )
    assert config.media_inputs == ("image",)
    assert config.decision_image_transport == "typesafe_multipart"
    assert config.typesafe_danger_zone_images is True
    for values in ({"decision_image_transport": "auto"}, {"typesafe_danger_zone_images": "true"}):
        with pytest.raises(ValidationError):
            ModelConfig(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides", [{"decision_image_transport": "json"}, {"typesafe_danger_zone_images": False}]
)
async def test_transport_override_resets_inherited_permissions(decision_alias, overrides):
    client = get_decision_model(decision_alias, **overrides)
    assert client.media_inputs == ()
    assert client.decision_image_transport == "json"
    assert client.typesafe_danger_zone_images is False
    await client.aclose()


@pytest.mark.asyncio
async def test_identical_explicit_transport_keeps_permissions(decision_alias):
    client = get_decision_model(
        decision_alias,
        decision_image_transport="typesafe_multipart",
        typesafe_danger_zone_images=True,
    )
    assert client.media_inputs == ("image",)
    await client.aclose()
