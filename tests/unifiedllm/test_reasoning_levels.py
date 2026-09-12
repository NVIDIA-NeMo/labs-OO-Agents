# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Effort declarations are data; consumers only select and inspect labels."""

from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from pydantic import ValidationError

from nooa.config.model_config import ModelConfig
from nooa.unifiedllm import CompletionClient, FakeLLMClient, ResponsesClient, get_llm_client
from nooa.unifiedllm.reasoning import ReasoningConfig

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
    ],
)
def test_level_settings_cannot_replace_framework_or_routing_fields(field):
    with pytest.raises(ValidationError, match="reserved.*" + field):
        CompletionClient("openai/test", reasoning_levels={"low": {field: "value"}})


def test_level_settings_allow_new_provider_fields_without_an_allowlist():
    config = ReasoningConfig(levels={"low": {"future_provider_control": {"budget": 12}}})
    assert config.settings("low") == {"future_provider_control": {"budget": 12}}


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


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fake_does_not_report_success_for_unknown_reasoning(asynchronous):
    client = FakeLLMClient()
    with pytest.raises(ValueError, match="unknown"):
        if asynchronous:
            await client.acall([], reasoning_level="low")
        else:
            client.call([], reasoning_level="low")
    assert client.call_count == 0
