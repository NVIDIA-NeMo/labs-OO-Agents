# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocation-free native loader/bind settings; route facts are never guessed."""

import httpx
import pytest

from nooa.unifiedllm.adapters._binding import _UNSET, _thaw
from nooa.unifiedllm.adapters._native_settings import _derive_native, _resolve_native
from nooa.unifiedllm.errors import FailureKind, LLMFailure

FAMILIES = ["openai-chat", "openai-responses", "anthropic-messages"]


def block_value(family, value):
    return {
        "openai-chat": {"type": "content", "content": value},
        "openai-responses": {"effort": value},
        "anthropic-messages": {"type": "disabled"}
        if value == "low"
        else {"type": "adaptive", "display": "omitted" if value == "high" else "summarized"},
    }[family]


def resolve(family, config=None, **overrides):
    return _resolve_native(
        family,
        "alias",
        {
            "model_name": "literal/slash-model",
            "api_base": "https://fixture.invalid/v1",
            "api_key": "fixture-key",
            "max_tokens": 100,
            **(config or {}),
        },
        overrides,
    )[0]


@pytest.mark.parametrize("family", FAMILIES)
def test_snapshot_and_no_http(family, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("settings resolution allocated HTTP")

    monkeypatch.setattr(httpx.Client, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", forbidden)
    config = {
        "extra_body": {"host_knob": {"value": 1}},
        "reasoning_levels": {"low": {"temperature": 0.2}},
        "reasoning_level": "low",
        "media_inputs": ["image"],
        "context_window": 2000,
    }
    settings = resolve(family, config)
    assert settings.model == "literal/slash-model"
    assert settings.capabilities.context_window == 2000
    assert settings.capabilities.tools is None
    assert settings.selected["temperature"] == 0.2
    config["extra_body"]["host_knob"]["value"] = 9
    config["reasoning_levels"]["low"]["temperature"] = 1
    assert _thaw(settings.selected)["extra_body"] == {"host_knob": {"value": 1}}
    view = _derive_native(settings, {"temperature": _UNSET, "max_output_tokens": 50})
    assert view.route is settings.route and view.capabilities is settings.capabilities
    assert (
        next(
            view.selected[k]
            for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if k in view.selected
        )
        == 50
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "option",
    [
        "drop_params",
        "allowed_openai_params",
        "additional_drop_params",
        "azure_endpoint",
        "aws_access_key_id",
        "vertex_credentials",
        "unknown_setting",
    ],
)
def test_unsupported_load_controls_terminal(family, option):
    with pytest.raises(LLMFailure) as caught:
        resolve(family, **{option: False})
    assert caught.value.kind == FailureKind.CONFIG


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"max_retries": 9},
        {"num_retries": None},
        {"extra_body": {"model": "other"}},
        {"extra_headers": {"Authorization": "wrong"}},
        {"temperature": float("nan")},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_tokens": 2, "max_output_tokens": 3},
    ],
)
def test_invalid_settings_before_http(family, options):
    with pytest.raises(LLMFailure) as caught:
        resolve(family, **options)
    assert caught.value.kind == FailureKind.CONFIG


@pytest.mark.parametrize("family", FAMILIES)
def test_rebind_whole_blocks_and_none(family):
    native = {
        "openai-chat": "prediction",
        "openai-responses": "reasoning",
        "anthropic-messages": "thinking",
    }[family]
    settings = resolve(
        family,
        {
            native: block_value(family, "medium"),
            "reasoning_levels": {
                "low": {native: block_value(family, "low")},
                "high": {native: block_value(family, "high")},
            },
            "reasoning_level": "low",
        },
    )
    high = _derive_native(settings, {"reasoning_level": "high", "temperature": None})
    assert _thaw(high.selected)[native] == block_value(family, "high")
    assert "temperature" not in high.selected
    off = _derive_native(high, {"reasoning_level": None})
    assert _thaw(off.selected)[native] == block_value(family, "medium")
    with pytest.raises(LLMFailure):
        _derive_native(settings, {"reasoning_level": "high", native: {"explicit": 1}})
    for key in ("model", "api_base", "api_key", "extra_query", "headers"):
        with pytest.raises(LLMFailure):
            _derive_native(settings, {key: "wrong"})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "override", [{"model": "new"}, {"api_base": "https://other.invalid/v1"}, {"api_key": "new-key"}]
)
def test_route_change_clears_inherited_facts(family, override):
    settings = resolve(
        family,
        {
            "context_window": 2000,
            "media_inputs": ["image"],
            "reasoning_levels": {"low": {"temperature": 0.2}},
            "reasoning_level": "low",
        },
        **override,
    )
    assert settings.capabilities.context_window is None
    assert not settings.capabilities.media_kinds
    assert settings.declaration.levels is None
    assert settings.selection is None


@pytest.mark.parametrize("family", FAMILIES)
def test_endpoint_none_unset_aliases_and_explicit_fact(family):
    settings = resolve(
        family, {"context_window": 2000}, api_base=None, base_url=" ", temperature=_UNSET
    )
    assert settings.capabilities.context_window == 2000
    changed = resolve(family, {"context_window": 2000}, model="new", context_window=4000)
    assert changed.capabilities.context_window == 4000
    with pytest.raises(LLMFailure):
        resolve(family, api_base="https://one.invalid", base_url="https://two.invalid")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("explicit", [0.8, None])
def test_load_reasoning_conflicts_with_genuine_overrides(family, explicit):
    declaration = {"reasoning_levels": {"low": {"temperature": 0.2}}, "reasoning_level": "low"}
    with pytest.raises(LLMFailure) as caught:
        resolve(family, declaration, temperature=explicit)
    assert caught.value.kind == FailureKind.CONFIG
    assert resolve(family, {**declaration, "temperature": 0.8}).selected["temperature"] == 0.2
    assert (
        "temperature"
        not in resolve(family, declaration, reasoning_level=None, temperature=None).selected
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_load_reasoning_cap_alias_and_whole_block_conflicts(family):
    block = {
        "openai-chat": "prediction",
        "openai-responses": "reasoning",
        "anthropic-messages": "thinking",
    }[family]
    config = {
        block: block_value(family, "medium"),
        "reasoning_levels": {"low": {block: block_value(family, "low"), "max_tokens": 30}},
        "reasoning_level": "low",
    }
    for options in (
        {"max_output_tokens": 20},
        {"max_completion_tokens": None},
        {block: {"explicit": 1}},
        {"extra_body": {block: {"explicit": 1}}},
    ):
        with pytest.raises(LLMFailure):
            resolve(family, config, **options)
    selected = resolve(family, config).selected
    assert _thaw(selected)[block] == block_value(family, "low")
    settings = resolve(
        family,
        config,
        reasoning_level=None,
        **{block: block_value(family, "high"), "max_output_tokens": 20},
    )
    assert _thaw(settings.selected)[block] == block_value(family, "high")
    assert (
        len(settings.selected.keys() & {"max_tokens", "max_completion_tokens", "max_output_tokens"})
        == 1
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_route_alias_layers_and_auth_env(family, monkeypatch):
    settings = resolve(family, model_name="replacement")
    assert settings.model == "replacement"
    assert settings.capabilities.context_window is None
    settings = resolve(
        family,
        {"default_headers": {"x-host": "old"}, "default_query": {"host_version": "old"}},
        headers={"x-host": "new"},
        extra_query={"host_version": "new"},
    )
    assert _thaw(settings.route)["headers"] == {"x-host": "new"}
    assert _thaw(settings.route)["query"] == {"host_version": "new"}
    monkeypatch.setenv("NATIVE_FIXTURE_KEY", "replacement-key")
    assert resolve(family, api_key_env="NATIVE_FIXTURE_KEY").route["key"] == "replacement-key"
    for options in (
        {"adapter": "litellm"},
        {"model": "a", "model_name": "b"},
        {"headers": {"x-host": "a"}, "default_headers": {"x-host": "b"}},
        {"default_query": {"a": 1}, "extra_query": {"a": 2}},
        {"headers": {"Authorization": "secret"}},
        {"extra_query": {"api_key": "secret"}},
    ):
        with pytest.raises(LLMFailure):
            resolve(family, **options)
    with pytest.raises(LLMFailure):
        resolve(family, {"base_url": "https://one.invalid", "api_base": "https://two.invalid"})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "field",
    [
        "messages",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "response_format",
        "text",
        "output_config",
        "stream",
        "n",
    ],
)
def test_extra_body_cannot_override_request_authority(family, field):
    with pytest.raises(LLMFailure) as caught:
        resolve(family, extra_body={field: []})
    assert caught.value.kind == FailureKind.CONFIG


@pytest.mark.parametrize(
    "family,options",
    [
        ("openai-chat", {"n": True}),
        ("openai-chat", {"seed": True}),
        ("openai-chat", {"parallel_tool_calls": 1}),
        ("openai-responses", {"reasoning": "high"}),
        ("openai-responses", {"text": {"format": {}}}),
        ("anthropic-messages", {"thinking": True}),
        ("anthropic-messages", {"output_config": {"format": {}}}),
        ("anthropic-messages", {"stop_sequences": "stop"}),
        ("openai-chat", {"stream": 0}),
    ],
)
def test_known_control_types_fail_config(family, options):
    with pytest.raises(LLMFailure) as caught:
        resolve(family, **options)
    assert caught.value.kind == FailureKind.CONFIG


def test_anthropic_required_cap_and_unsupported_media():
    with pytest.raises(LLMFailure):
        resolve("anthropic-messages", max_tokens=None)
    assert "max_completion_tokens" not in resolve("openai-chat", max_tokens=None).selected
    with pytest.raises(LLMFailure):
        resolve("openai-chat", media_inputs=["pdf"])
    with pytest.raises(LLMFailure):
        resolve("anthropic-messages", media_inputs=["audio"])
