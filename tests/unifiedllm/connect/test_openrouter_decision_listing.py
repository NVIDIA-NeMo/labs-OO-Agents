# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenRouter hides decision-output models unless the output filter includes them."""

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

JEV = {
    "id": "typesafe/jev-1.14",
    "name": "TypeSafe: Jev 1.14",
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["decisions"]},
}
ROUTER = {
    "id": "typesafe/jev-router",
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["catalogue", "discover", "details"])
async def test_openrouter_requests_text_and_decision_models(monkeypatch, operation):
    seen = []

    def handle(request):
        seen.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v1/models"
        data = (
            [ROUTER, JEV]
            if request.url.params.get("output_modalities") == "text,decisions"
            else [ROUTER]
        )
        return httpx.Response(200, json={"data": data})

    mock_http(monkeypatch, handle)
    if operation == "catalogue":
        models = await connect.catalogue()
    elif operation == "discover":
        models = (await connect.discover("https://openrouter.ai/api/v1")).models
    else:
        details = await connect.server_details("https://openrouter.ai/api/v1", JEV["id"])
        models = [details.model]
        assert details.requests == 1
    assert JEV["id"] in [model["id"] for model in models]
    selected = next(model for model in models if model["id"] == JEV["id"])
    assert selected["architecture"]["output_modalities"] == ["decisions"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_generic_model_servers_receive_no_openrouter_filter(monkeypatch):
    def handle(request):
        assert "output_modalities" not in request.url.params
        return httpx.Response(200, json={"data": [ROUTER]})

    mock_http(monkeypatch, handle)
    await connect.discover("https://gateway.test/v1")


@pytest.mark.asyncio
async def test_explicit_openrouter_shared_listing_includes_decisions(monkeypatch):
    def handle(request):
        assert request.url.params["output_modalities"] == "text,decisions"
        return httpx.Response(200, json={"data": [JEV]})

    mock_http(monkeypatch, handle)
    result = await connect.discover(
        "https://openrouter.ai/api/alpha/decisions",
        api_style="systemone",
        models_url=connect.CATALOGUE_URL,
    )
    assert result.models[0]["id"] == JEV["id"]
    assert result.api_base == "https://openrouter.ai/api/alpha/decisions"


@pytest.mark.parametrize(
    "outputs,expected",
    [(["decisions"], "systemone"), (["text"], "discover"), (["text", "decisions"], "discover")],
)
def test_interface_suggestion_uses_declared_output_not_name(monkeypatch, outputs, expected):
    from nooa_cli.commands import _connect_prompts as prompts
    from nooa_cli.commands._connect_wizard import WizardState, select_request_interface

    model = {**JEV, "architecture": {"output_modalities": outputs}}
    state = WizardState(
        model=JEV["id"],
        provider="openrouter",
        endpoint="https://openrouter.ai/api/v1",
        api_key_env="",
        selected_endpoint_model=model,
        interface_endpoints={"systemone": "https://openrouter.ai/api/alpha/decisions"},
        server_urls=[],
    )

    def choose(message, **options):
        assert message == "Request interface"
        assert options["default"] == expected
        return options["default"]

    monkeypatch.setattr(prompts, "prompt", choose)
    assert select_request_interface(state)
    if expected == "systemone":
        assert state.api_style == "systemone"
        assert state.endpoint == "https://openrouter.ai/api/alpha/decisions"
    else:
        assert state.api_style is None


def test_explicit_interface_is_preserved(monkeypatch):
    from nooa_cli.commands import _connect_prompts as prompts
    from nooa_cli.commands._connect_wizard import WizardState, select_request_interface

    monkeypatch.setattr(
        prompts, "prompt", lambda *args, **kwargs: pytest.fail("Explicit style must not prompt")
    )
    state = WizardState(model=JEV["id"], api_style="chat", selected_endpoint_model=JEV)
    assert select_request_interface(state)
    assert state.api_style == "chat"


@pytest.mark.parametrize("field", ["model_type", "client_type"])
@pytest.mark.parametrize("answer", ["systemone", "discover", "chat"])
def test_server_decision_type_suggests_systemone_but_keeps_user_choice(
    monkeypatch, capsys, field, answer
):
    from nooa_cli.commands import _connect_prompts as prompts
    from nooa_cli.commands._connect_wizard import WizardState, select_request_interface

    state = WizardState(
        model="nvidia/alexwortega/openjev-4b",
        provider="hub",
        endpoint="https://inference-api.nvidia.com/v1",
        api_key_env="",
        selected_endpoint_model={"id": "nvidia/alexwortega/openjev-4b", field: "decision"},
        interface_endpoints={"systemone": "https://inference-api.nvidia.com/v1/systemone"},
        server_urls=[],
    )

    def choose(message, **options):
        assert options["default"] == "systemone"
        return answer

    monkeypatch.setattr(prompts, "prompt", choose)
    assert select_request_interface(state)
    assert state.api_style == (None if answer == "discover" else answer)
    assert "Model type: Decision" in " ".join(capsys.readouterr().out.split())


def test_litellm_detection_and_decision_type_shown_before_interface(monkeypatch, capsys):
    from nooa_cli.commands import _connect_prompts as prompts
    from nooa_cli.commands._connect_wizard import (
        WizardState,
        prepare_metadata,
        select_request_interface,
    )

    details = connect.ModelDetails(
        "model",
        {"id": "model", "model_type": "decision"},
        records=(
            {
                "source": "https://inference-api.nvidia.com/model_group/info",
                "schema": "litellm.model_group.info",
                "product": {"title": "LiteLLM API", "version": "1.105.1"},
            },
        ),
    )

    async def server(*args, **options):
        assert options["litellm"] is True
        return details

    async def catalogue():
        return []

    monkeypatch.setattr(connect, "server_details", server)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    state = WizardState(
        model="model",
        provider="hub",
        endpoint="https://inference-api.nvidia.com/v1",
        api_key_env="",
        data={},
        server_urls=[],
        interface_endpoints={"systemone": "https://inference-api.nvidia.com/v1/systemone"},
        endpoint_models=[],
    )

    def choose(message, **options):
        assert "Server: LiteLLM 1.105.1" in " ".join(capsys.readouterr().out.split())
        assert options["default"] == "systemone"
        return "systemone"

    monkeypatch.setattr(prompts, "prompt", choose)
    assert prepare_metadata(state)
    assert select_request_interface(state)
    assert state.api_style == "systemone"
