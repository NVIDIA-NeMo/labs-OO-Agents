# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenRouter deployment limits must not become another provider's limits."""

from copy import deepcopy

import pytest

from nooa.unifiedllm import connect

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
PUBLIC = {
    "id": MODEL,
    "context_length": 262144,
    "top_provider": {"context_length": 202800, "max_completion_tokens": 182520},
}


@pytest.mark.parametrize(
    "base",
    [
        "https://integrate.api.nvidia.com/v1",
        "https://api.openai.com/v1",
        "https://inference-api.nvidia.com/v1",
    ],
)
def test_native_route_uses_general_model_context_not_openrouter_deployment(base):
    before = deepcopy(PUBLIC)
    entry = connect.plan("local", MODEL, "chat", base, "", catalogue=PUBLIC).entry
    assert entry["context_window"] == 262144
    assert "max_output_tokens" not in entry
    assert "max_completion_tokens" not in entry["provenance"].get("catalogue_limits", {})
    assert entry["max_tokens"] == 32768
    assert PUBLIC == before


def test_openrouter_route_still_uses_its_deployment_limits():
    entry = connect.plan(
        "local", MODEL, "chat", "https://openrouter.ai/api/v1", "", catalogue=PUBLIC
    ).entry
    assert entry["context_window"] == 202800
    assert entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == 182520


def test_native_reported_limits_win_and_explicit_reply_cap_is_kept():
    entry = connect.plan(
        "local",
        MODEL,
        "chat",
        "https://integrate.api.nvidia.com/v1",
        "",
        catalogue=PUBLIC,
        endpoint_model={"id": MODEL, "context_window": 131072, "max_output_tokens": 8192},
        reply_tokens=4096,
    ).entry
    assert entry["context_window"] == 131072
    assert entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == 8192
    assert entry["max_tokens"] == 4096


def test_native_wizard_does_not_apply_openrouter_deployment_limits(monkeypatch):
    from nooa_cli.commands._connect_wizard import WizardState, configure_checks, prepare_metadata

    async def server(*args, **kwargs):
        return connect.ModelDetails(
            MODEL,
            {"id": MODEL},
            records=(
                {
                    "source": "https://integrate.api.nvidia.com/v1/models",
                    "schema": "models",
                    "facts": {"id": MODEL},
                },
            ),
        )

    async def catalogue():
        return [deepcopy(PUBLIC)]

    monkeypatch.setattr(connect, "server_details", server)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    state = WizardState(
        model=MODEL,
        provider="nvidia",
        api_style="chat",
        endpoint="https://integrate.api.nvidia.com/v1",
        api_key_env="",
        data={},
        server_urls=[],
        endpoint_models=[],
        guided=True,
        budget_tokens=8192,
        output_tokens=2048,
        reasoning_output_tokens=4096,
        media_inputs=[],
        approval="none",
    )
    assert prepare_metadata(state)
    assert state.candidate["context_length"] == 262144
    assert not state.candidate.get("top_provider")
    assert configure_checks(state)
    assert state.proposal.entry["context_window"] == 262144
    assert "max_completion_tokens" not in state.proposal.entry["provenance"]["catalogue_limits"]
