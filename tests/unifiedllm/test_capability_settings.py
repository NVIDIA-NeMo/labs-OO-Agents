# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The structured_output_format setting, detected developer role, and instruction_role.

Request shapes are checked on the prepared provider parameters (LiteLLM) and
on the native projection; no network.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from nooa.unifiedllm.adapters._native_projection import _project_request
from nooa.unifiedllm.adapters._native_settings import _resolve_native
from nooa.unifiedllm.capability_settings import (
    CapabilitySettingError,
    json_object_instruction,
    map_instruction_roles,
    resolve_instruction_role,
    resolve_structured_output_format,
)
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.registry import client_from_config, load_llm_from_config
from nooa.unifiedllm.request import LLMRequest


class Answer(BaseModel):
    value: int


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "developer", "content": "dev"},
]
NEUTRAL = {"structured_output_format": "json_schema", "instruction_role": "auto"}


def _client(style="completion", **entry):
    return client_from_config(
        "alias",
        {
            "model_name": "openai/model",
            "api_base": "https://fixture.invalid/v1",
            "api_key": "fixture-key",
            "client_type": style,
            **entry,
        },
    )


def _wire(client, output_type=Answer, messages=MESSAGES, **call):
    request = LLMRequest(messages=list(messages), output_type=output_type)
    config = client._prepare_call_config(call)
    if client._adapter.__class__.__name__ == "LiteLLMChatAdapter":
        params = client._adapter.prepare(request, config, native_output=False, is_async=False)
    else:
        params = client._adapter.prepare(request, config, is_async=False)
    params = dict(params.api_params)
    params.pop("client", None)
    return params


def _schema_text(content):
    return json.loads(content.split("\n", 1)[1])


# -- resolution ----------------------------------------------------------------


@pytest.mark.parametrize(
    "setting,expected",
    [(None, "json_schema"), ("json_schema", "json_schema"), ("json_object", "json_object")],
)
def test_structured_output_format_resolution(setting, expected):
    assert resolve_structured_output_format(setting) == expected


@pytest.mark.parametrize("setting", ["auto", "xml", ["json_schema"], ""])
def test_structured_output_format_rejects_other_values(setting):
    with pytest.raises(CapabilitySettingError, match="'json_schema' or 'json_object'") as info:
        resolve_structured_output_format(setting)
    assert info.value.kind is FailureKind.CONFIG


@pytest.mark.parametrize(
    "setting,detected,expected",
    [
        (None, None, "system"),
        ("auto", False, "system"),
        ("auto", True, "developer"),
        ("system", True, "system"),
        ("developer", None, "developer"),
        ("developer", True, "developer"),
    ],
)
def test_instruction_role_resolution(setting, detected, expected):
    assert resolve_instruction_role(setting, detected) == expected


def test_explicit_developer_on_route_without_support_is_config_error():
    with pytest.raises(CapabilitySettingError, match="developer_role: false"):
        resolve_instruction_role("developer", False)


@pytest.mark.parametrize(
    "key,value",
    [
        ("structured_output_format", "xml"),
        ("structured_output_format", "auto"),
        ("structured_output_format", ["json_schema"]),
        ("instruction_role", "assistant"),
        ("developer_role", "yes"),
    ],
)
def test_invalid_values_fail_at_construction(key, value):
    with pytest.raises(CapabilitySettingError):
        _client(**{key: value})


def test_map_instruction_roles_without_facts_returns_same_list():
    assert map_instruction_roles(MESSAGES, "system", None) is MESSAGES
    assert map_instruction_roles(MESSAGES, "system", True) is MESSAGES


# -- LiteLLM Chat Completions ----------------------------------------------------


def test_chat_default_and_neutral_settings_send_identical_requests():
    baseline = _wire(_client())
    assert baseline["messages"] == MESSAGES
    assert baseline["response_format"] is Answer
    assert _wire(_client(**NEUTRAL)) == baseline
    # Unknown developer role with the auto role keeps today's request too.
    assert _wire(_client(developer_role=None)) == baseline


def test_chat_json_object_sends_json_mode_and_schema_last():
    params = _wire(_client(structured_output_format="json_object"))
    assert params["response_format"] == {"type": "json_object"}
    assert params["messages"][:3] == MESSAGES  # cacheable prefix unchanged
    final = params["messages"][-1]
    assert final["role"] == "system" and "JSON" in final["content"]
    assert _schema_text(final["content"]) == Answer.model_json_schema()
    assert final == json_object_instruction(Answer)


def test_chat_json_object_only_for_typed_calls():
    params = _wire(_client(structured_output_format="json_object"), output_type=None)
    assert "response_format" not in params
    assert params["messages"] == MESSAGES


def test_chat_developer_role_maps_instructions():
    params = _wire(_client(developer_role=True, structured_output_format="json_object"))
    assert [m["role"] for m in params["messages"]] == [
        "developer",
        "user",
        "developer",
        "developer",
    ]


def test_chat_caller_developer_message_is_system_on_route_without_support():
    params = _wire(_client(developer_role=False))
    assert [m["role"] for m in params["messages"]] == ["system", "user", "system"]
    assert [m["content"] for m in params["messages"]] == ["sys", "hi", "dev"]


def test_chat_explicit_system_keeps_caller_developer_when_support_unknown():
    params = _wire(_client(instruction_role="system"))
    assert params["messages"] == MESSAGES


def test_route_override_does_not_inherit_capability_choices():
    client = _client(structured_output_format="json_object", developer_role=True)
    params = _wire(client, model="openai/other")
    assert params["response_format"] is Answer
    assert params["messages"] == MESSAGES


def test_registry_override_of_model_drops_entry_facts():
    client = client_from_config(
        "alias",
        {
            "model_name": "openai/model",
            "api_base": "https://fixture.invalid/v1",
            "api_key": "fixture-key",
            "structured_output_format": "json_object",
            "developer_role": False,
        },
        model="openai/other",
    )
    assert client._output_format == "json_schema" and client.developer_role is None
    assert "structured_output_format" not in client.config


def test_capability_keys_never_reach_provider_params():
    params = _wire(
        _client(
            developer_role=True,
            structured_output_format="json_schema",
            instruction_role="system",
        )
    )
    for key in (
        "developer_role",
        "structured_output_format",
        "instruction_role",
    ):
        assert key not in params


def test_json_object_conflicts_with_native_anthropic_mode():
    with pytest.raises(CapabilitySettingError, match="structured_output_mode"):
        client_from_config(
            "alias",
            {
                "model_name": "anthropic/model",
                "api_key": "fixture-key",
                "structured_output_mode": "anthropic",
                "structured_output_format": "json_object",
            },
        )


# -- LiteLLM Responses -----------------------------------------------------------


def test_responses_default_and_neutral_settings_send_identical_requests():
    baseline = _wire(_client("responses"))
    assert baseline["text_format"] is Answer
    assert _wire(_client("responses", **NEUTRAL)) == baseline
    assert _wire(_client("responses", developer_role=None)) == baseline


def test_responses_json_object_uses_text_format_and_final_instruction():
    params = _wire(_client("responses", structured_output_format="json_object"))
    assert params["text"] == {"format": {"type": "json_object"}}
    assert "text_format" not in params
    assert params["instructions"] == "sys"
    final = params["input"][-1]
    assert final["role"] == "system"
    assert _schema_text(final["content"][0]["text"]) == Answer.model_json_schema()


def test_responses_developer_role_maps_input_instructions():
    params = _wire(
        _client("responses", developer_role=True),
        messages=[{"role": "user", "content": "hi"}, {"role": "system", "content": "later"}],
    )
    assert [m["role"] for m in params["input"]] == ["user", "developer"]


def test_responses_caller_developer_is_system_without_support():
    params = _wire(_client("responses", developer_role=False))
    assert [m["role"] for m in params["input"]] == ["user", "system"]


# -- load-time facts and rejection -------------------------------------------------


def _entry(**extra):
    return {
        "model_name": "openai/model",
        "api_base": "https://fixture.invalid/v1",
        "api_key": "fixture-key",
        "media_inputs": [],
        **extra,
    }


def test_loaded_llm_exposes_detected_facts():
    llm = load_llm_from_config(
        "alias", _entry(structured_output_format="json_schema", developer_role=True)
    )
    try:
        assert llm.capabilities.developer_role is True
    finally:
        llm.close()


def test_loaded_llm_without_facts_reports_unknown():
    llm = load_llm_from_config("alias", _entry())
    try:
        assert llm.capabilities.developer_role is None
    finally:
        llm.close()


@pytest.mark.parametrize(
    "extra",
    [
        {"structured_output_format": "auto"},
        {"developer_role": False, "instruction_role": "developer"},
    ],
)
def test_load_rejects_invalid_or_contradicted_choice(extra):
    with pytest.raises(LLMFailure) as info:
        load_llm_from_config("alias", _entry(**extra))
    assert info.value.kind is FailureKind.CONFIG


# -- native projection -------------------------------------------------------------


def _native(family, **options):
    return _resolve_native(
        family,
        "alias",
        dict(
            model_name="model",
            api_key="fixture-key",
            api_base="https://fixture.invalid/v1",
            max_tokens=100,
            **options,
        ),
        {},
    )[0]


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
def test_native_neutral_settings_project_identical_requests(family):
    request = LLMRequest(messages=list(MESSAGES[:2]), output_type=Answer)
    baseline = _project_request(request, _native(family))
    assert _project_request(request, _native(family, **NEUTRAL)) == baseline
    assert _project_request(request, _native(family, developer_role=None)) == baseline


def test_native_chat_json_object_and_developer():
    settings = _native("openai-chat", structured_output_format="json_object", developer_role=True)
    assert settings.capabilities.developer_role is True
    params = _project_request(LLMRequest(messages=list(MESSAGES), output_type=Answer), settings)
    assert params["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in params["messages"]] == [
        "developer",
        "user",
        "developer",
        "developer",
    ]
    assert _schema_text(params["messages"][-1]["content"]) == Answer.model_json_schema()


def test_native_chat_caller_developer_is_system_without_support():
    params = _project_request(
        LLMRequest(messages=list(MESSAGES)), _native("openai-chat", developer_role=False)
    )
    assert [m["role"] for m in params["messages"]] == ["system", "user", "system"]


def test_native_responses_json_object():
    settings = _native("openai-responses", structured_output_format="json_object")
    params = _project_request(LLMRequest(messages=list(MESSAGES), output_type=Answer), settings)
    assert params["text"]["format"] == {"type": "json_object"}
    final = params["input"][-1]
    assert final["role"] == "system"
    assert final["content"][0]["type"] == "input_text"
    assert _schema_text(final["content"][0]["text"]) == Answer.model_json_schema()


@pytest.mark.parametrize(
    "options,match",
    [
        ({"structured_output_format": "json_object"}, "json_object"),
        ({"instruction_role": "developer"}, "developer"),
    ],
)
def test_native_anthropic_load_rejections(options, match):
    family = "anthropic-messages"
    with pytest.raises(LLMFailure, match=match) as info:
        _native(family, **options)
    assert info.value.kind is FailureKind.CONFIG


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
def test_native_load_rejects_auto_format(family):
    with pytest.raises(LLMFailure, match="not 'auto'") as info:
        _native(family, structured_output_format="auto")
    assert info.value.kind is FailureKind.CONFIG
