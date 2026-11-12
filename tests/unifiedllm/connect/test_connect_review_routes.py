# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permission inheritance is restricted to the same normalized route."""

from copy import deepcopy

import pytest

from nooa.unifiedllm import connect
from nooa.unifiedllm.registry import client_from_config

OLD = {
    "model_name": "openai/model",
    "api_base": "https://api.test/v1",
    "api_style": "chat",
    "client_type": "completion",
    "api_key_env": "",
    "media_inputs": ["image", "pdf"],
    "max_tokens": 1000,
}


@pytest.mark.parametrize(
    "model,style,endpoint,expected",
    [
        ("model", "chat", "https://api.test/v1/", ["image", "pdf"]),
        ("other", "chat", "https://api.test/v1", []),
        ("model", "responses", "https://api.test/v1", []),
        ("model", "anthropic", "https://api.test/v1", []),
        ("model", "chat", "https://other.test/v1", []),
    ],
)
async def test_route_local_plan_and_history(model, style, endpoint, expected):
    original = deepcopy(OLD)
    proposal = connect.plan("local", model, style, endpoint, "NEW_KEY", existing_entry=original)
    assert proposal.entry["media_inputs"] == expected
    client = client_from_config("local", proposal.entry, api_key="fake")
    history = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
            ],
        }
    ]
    before = deepcopy(history)
    try:
        projected = client._prepare_media_messages(history, dict(client.config))
        assert projected[0]["content"][0]["type"] == ("image_url" if expected else "text")
        assert history == before and original == OLD
    finally:
        await client.aclose()
    if style in {"anthropic", "responses"}:
        with pytest.raises(ValueError, match="Unsupported media input.*audio"):
            connect.plan(
                "local", model, style, endpoint, "", existing_entry=original, media_inputs=["audio"]
            )
    else:
        explicit = connect.plan(
            "local", model, style, endpoint, "", existing_entry=original, media_inputs=["audio"]
        )
        assert explicit.entry["media_inputs"] == ["audio"]
    cleared = connect.plan(
        "local", model, style, endpoint, "", existing_entry=original, media_inputs=[]
    )
    assert cleared.entry["media_inputs"] == []


def test_missing_identity_or_changed_client_type_cannot_inherit():
    for previous in ({"media_inputs": ["image"]}, {**OLD, "client_type": "custom"}):
        assert (
            connect.plan(
                "local", "model", "chat", "https://api.test/v1", "", existing_entry=previous
            ).entry["media_inputs"]
            == []
        )


def test_anthropic_normalized_endpoint_keeps_permission():
    old = {**OLD, "model_name": "anthropic/model", "api_style": "anthropic"}
    assert connect.plan(
        "local", "model", "anthropic", "https://api.test/v1/", "", existing_entry=old
    ).entry["media_inputs"] == ["image", "pdf"]
