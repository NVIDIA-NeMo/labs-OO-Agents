# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Effort declarations are data; consumers only select and inspect labels."""

import ast
import re
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest
import yaml
from pydantic import ValidationError

from nooa.config.model_config import ModelConfig
from nooa.unifiedllm import CompletionClient, FakeLLMClient, ResponsesClient, get_llm_client
from nooa.unifiedllm.reasoning import ReasoningConfig, apply_reasoning_level

LEVELS = {
    "low": {"reasoning": {"effort": "low", "context": "all_turns"}},
    "high": {"reasoning": {"effort": "high", "context": "all_turns"}},
}


def _response(responses):
    if not responses:
        return litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    return SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        output_text="ok",
        status="completed",
        usage=None,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
async def test_selected_level_reaches_both_dispatch_paths(client_type, asynchronous):
    responses = client_type is ResponsesClient
    method = ("a" if asynchronous else "") + ("responses" if responses else "completion")
    with client_type("openai/test", reasoning_levels=LEVELS, reasoning_default="low") as client:
        assert client.reasoning_levels == ("low", "high")
        assert client.reasoning_default == "low"
        with patch(
            f"litellm.{method}", new_callable=AsyncMock if asynchronous else None
        ) as transport:
            transport.return_value = _response(responses)
            messages = [{"role": "user", "content": "hello"}]
            if asynchronous:
                result = await client.acall(messages, reasoning_level="high")
            else:
                result = client.call(messages, reasoning_level="high")
            assert result.content == "ok"
            params = transport.call_args.kwargs
            assert params["reasoning"] == LEVELS["high"]["reasoning"]
            assert not {"reasoning_levels", "reasoning_default", "reasoning_level"} & params.keys()
            assert client.config == {}


@pytest.mark.parametrize(
    "levels,expected", [(None, "unknown"), ({}, "not supported"), (LEVELS, "allowed: low, high")]
)
def test_unknown_unsupported_and_invalid_are_distinct(levels, expected):
    with CompletionClient("openai/test", reasoning_levels=levels) as client:
        assert client.reasoning_levels == (None if levels is None else tuple(levels))
        with pytest.raises(ValueError, match=expected):
            client.call([], reasoning_level="max")


@pytest.mark.parametrize(
    "declaration",
    [
        {"levels": {"": {"reasoning_effort": "low"}}},
        {"levels": {"low": {}}},
        {"levels": {"low": "low"}},
        {"levels": {"low": {"reasoning_effort": "low"}}, "default": "high"},
        {"default": "high"},
    ],
)
def test_malformed_declarations_fail_early(declaration):
    with pytest.raises(ValidationError):
        ReasoningConfig(**declaration)


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "api_base",
        "base_url",
        "api_key",
        "custom_llm_provider",
        "messages",
        "input",
        "extra_body",
        "reasoning_levels",
        "reasoning_default",
        "reasoning_level",
        "client",
    ],
)
def test_level_settings_cannot_replace_framework_or_routing_fields(field):
    with pytest.raises(ValidationError, match="reserved.*" + field):
        CompletionClient("openai/test", reasoning_levels={"low": {field: "value"}})


def test_level_settings_allow_new_provider_fields_without_an_allowlist():
    config = ReasoningConfig(levels={"low": {"future_provider_control": {"budget": 12}}})
    assert config.settings("low") == {"future_provider_control": {"budget": 12}}


@pytest.mark.parametrize("field", ["model", "api_base", "extra_body", "reasoning_level"])
def test_mutating_a_declared_level_cannot_bypass_reserved_fields(field):
    config = ReasoningConfig(levels={"low": {"reasoning_effort": "low"}})
    config.levels["low"][field] = "injected"
    with pytest.raises(ValueError, match="reserved.*" + field):
        apply_reasoning_level(config, "openai/test", {}, {}, "low")


def test_no_selection_preserves_raw_controls_and_default_is_only_metadata():
    raw = {"reasoning": {"effort": "medium", "summary": "auto"}}
    with CompletionClient(
        "openai/test", reasoning_levels=LEVELS, reasoning_default="low", **raw
    ) as client:
        assert client._prepare_call_config({}) == raw
        assert client._prepare_call_config({"reasoning_effort": "future"}) == {
            **raw,
            "reasoning_effort": "future",
        }


def test_selection_replaces_whole_blocks_and_does_not_mutate_configuration():
    levels = {"low": {"reasoning": {"effort": "low", "context": "all_turns"}}}
    raw = {"reasoning": {"effort": "high", "summary": "auto"}}
    with ResponsesClient(
        "openai/test", reasoning_levels=levels, reasoning_level="low", **raw
    ) as client:
        levels["low"]["reasoning"]["effort"] = "changed by caller"
        first = client._prepare_call_config({})
        assert first == LEVELS["low"]  # No implicit merge of summary from raw defaults.
        first["reasoning"]["effort"] = "changed by SDK"
        assert client._prepare_call_config({}) == LEVELS["low"]
        assert client._prepare_call_config({"reasoning_level": None}) == raw
        assert client.config == raw


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning": {"effort": "high"}},
        {"extra_body": {"reasoning": {"effort": "high"}}},
        {"extra_body": MappingProxyType({"reasoning": {"effort": "high"}})},
    ],
)
def test_competing_request_settings_raise(overrides):
    with CompletionClient("openai/test", reasoning_levels=LEVELS) as client:
        with pytest.raises(ValueError, match="conflicts.*reasoning"):
            client.call([], reasoning_level="low", **overrides)


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
def test_selected_level_replaces_inherited_extra_body_without_mutating_it(client_type):
    inherited = {"reasoning": {"effort": "high"}, "other": {"enabled": True}}
    with client_type("openai/test", reasoning_levels=LEVELS, extra_body=inherited) as client:
        selected = client._prepare_call_config({"reasoning_level": "low"})
        assert selected["reasoning"] == LEVELS["low"]["reasoning"]
        assert selected["extra_body"] == {"other": {"enabled": True}}
        assert inherited["reasoning"] == {"effort": "high"}
        assert client._prepare_call_config({})["extra_body"] == inherited
        with pytest.raises(ValueError, match="conflicts.*reasoning"):
            client._prepare_call_config({"reasoning_level": "low", "extra_body": inherited})


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
def test_managed_level_rejects_sdk_client_override(client_type, monkeypatch):
    from openai import OpenAI

    responses = client_type is ResponsesClient
    monkeypatch.setattr(
        litellm, "responses" if responses else "completion", lambda **_: _response(responses)
    )
    with OpenAI(api_key="test", base_url="https://different-route.test/v1") as sdk:
        with client_type("openai/test", reasoning_levels=LEVELS) as client:
            with pytest.raises(ValueError, match="route-specific"):
                client.call([], reasoning_level="low", client=sdk)
            assert client._prepare_call_config({"client": sdk})["client"] is sdk


@pytest.mark.parametrize(
    "route",
    [
        {"model": "openai/other"},
        {"api_base": "https://other.test"},
        {"base_url": "https://other.test"},
        {"custom_llm_provider": "anthropic"},
    ],
)
def test_managed_level_cannot_follow_a_route_override(route):
    with CompletionClient("openai/test", reasoning_levels=LEVELS, reasoning_level="low") as client:
        with pytest.raises(ValueError, match="route-specific"):
            client.call([], **route)
        # An explicitly unmanaged call is still the existing raw-transport API.
        assert client._prepare_call_config({**route, "reasoning_level": None}) == route


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning_levels": LEVELS},
        {"reasoning_default": "low"},
        {"extra_body": {"reasoning_level": "low"}},
        {"extra_body": {"reasoning_levels": LEVELS}},
        {"extra_body": MappingProxyType({"reasoning_level": "low"})},
    ],
)
def test_framework_settings_cannot_leak_through_call_kwargs(overrides):
    with CompletionClient("openai/test") as client:
        with pytest.raises(ValueError, match="constructor|extra_body"):
            client.call([], **overrides)


def test_registry_declarations_reach_the_client(monkeypatch):
    from nooa.unifiedllm import registry

    config = {
        "model_name": "openai/test",
        "reasoning_levels": LEVELS,
        "reasoning_default": "low",
        "reasoning_level": "high",
    }
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", {"alias": config})
    assert ModelConfig.from_registry("alias", config).reasoning_default == "low"
    with get_llm_client("alias") as client:
        assert client.reasoning_levels == ("low", "high")
        assert client.reasoning_default == "low"
        assert client._prepare_call_config({})["reasoning"] == LEVELS["high"]["reasoning"]


@pytest.mark.parametrize(
    "route",
    [
        {"api_base": "https://different-route.test/v1"},
        {"base_url": "https://different-route.test/v1"},
        {"model": "openai/other"},
        {"custom_llm_provider": "openai"},
        {"client_type": "responses"},
        {"client": object()},
    ],
)
def test_registry_route_override_drops_inherited_reasoning(monkeypatch, route):
    from nooa.unifiedllm import registry

    config = {
        "model_name": "openai/test",
        "api_base": "https://original-route.test/v1",
        "reasoning_levels": LEVELS,
        "reasoning_default": "high",
        "reasoning_level": "high",
    }
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", {"alias": config})
    with get_llm_client("alias", api_key="test", **route) as client:
        assert client.reasoning_levels is None
        assert client.reasoning_default is None
        assert client.reasoning_level is None
        with pytest.raises(ValueError, match="unknown"):
            client._prepare_call_config({"reasoning_level": "low"})
    replacement = {"custom": {"reasoning_effort": "medium"}}
    with get_llm_client("alias", api_key="test", reasoning_levels=replacement, **route) as client:
        assert client.reasoning_levels == ("custom",)
        assert client.reasoning_default is None
        assert client.reasoning_level is None
        assert (
            client._prepare_call_config({"reasoning_level": "custom"})["reasoning_effort"]
            == "medium"
        )
    assert config["reasoning_levels"] is LEVELS


def test_registry_same_route_override_preserves_reasoning(monkeypatch):
    from nooa.unifiedllm import registry

    config = {
        "model_name": "openai/test",
        "api_base": "https://original-route.test/v1",
        "reasoning_levels": LEVELS,
        "reasoning_default": "low",
    }
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", {"alias": config})
    with get_llm_client(
        "alias",
        model=config["model_name"],
        api_base=config["api_base"],
        client_type="completion",
        api_key="replacement-key",
    ) as client:
        assert client.reasoning_levels == ("low", "high")
        assert client.reasoning_default == "low"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fake_does_not_report_success_for_unknown_reasoning(asynchronous):
    client = FakeLLMClient()
    with pytest.raises(ValueError, match="unknown"):
        if asynchronous:
            await client.acall([], reasoning_level="low")
        else:
            client.call([], reasoning_level="low")
    assert client.call_count == 0


async def test_agent_authoring_skill_reasoning_example(tmp_path, monkeypatch):
    """Execute the shipped skill's registry and selection example without inference."""
    from nooa.skill import _parse_skill_md
    from nooa.unifiedllm import registry

    def unexpected_request(*args, **kwargs):
        pytest.fail("The skill example must not make live HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", unexpected_request)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_request)
    name, _, skill = _parse_skill_md(
        Path(__file__).resolve().parents[2] / "skills/nooa-agent-authoring"
    )
    assert name == "nooa-agent-authoring"
    declarations = re.findall(r"```yaml\n(.*?)```", skill, re.DOTALL)
    assert len(declarations) == 1, "Provide one executable reasoning registry example"
    config = yaml.safe_load(declarations[0])["models"]["my-route"]
    (tmp_path / "llm_config.yaml").write_text(declarations[0])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    snippets = [
        block
        for block in re.findall(r"```python\n(.*?)```", skill, re.DOTALL)
        if 'reload_registry(Path("llm_config.yaml"))' in block
    ]
    assert len(snippets) == 1, "Load the example registry before selecting its alias"
    scope = {"messages": [{"role": "user", "content": "hello"}]}
    with patch("litellm.aresponses", new_callable=AsyncMock) as transport:
        transport.return_value = _response(True)
        try:
            await eval(
                compile(snippets[0], "SKILL.md", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), scope
            )
            client = scope["llm"]
            assert client.reasoning_levels == ("low", "medium", "high")
            assert client.reasoning_default == "medium"
            assert (
                client._prepare_call_config({})["reasoning"]
                == config["reasoning_levels"]["high"]["reasoning"]
            )
            assert (
                client._prepare_call_config({"reasoning_level": None})["reasoning"]
                == config["reasoning"]
            )
            assert (
                transport.call_args.kwargs["reasoning"]
                == config["reasoning_levels"]["low"]["reasoning"]
            )
            assert scope["reply"].content == "ok"
        finally:
            if "llm" in scope:
                await scope["llm"].aclose()
