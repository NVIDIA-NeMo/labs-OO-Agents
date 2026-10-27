# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``structured_output: json_object`` for endpoints that reject ``json_schema``.

DeepSeek (chat and reasoner) answers a ``json_schema`` response_format with
"This response_format type is unavailable now" but accepts ``json_object``. In that
mode the schema has to travel in the prompt; these checks pin the request shape.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from nooa.unifiedllm.registry import client_from_config
from nooa.unifiedllm.unifiedllm import CompletionClient, ResponsesClient, _json_object_request


class _Verdict(BaseModel):
    label: str
    confidence: float


def test_json_object_request_sends_json_mode_and_schema_last():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    sent, response_format = _json_object_request(messages, _Verdict)

    assert response_format == {"type": "json_object"}
    assert sent[:2] == messages  # cacheable prefix untouched
    assert sent[-1]["role"] == "system"
    schema_text = sent[-1]["content"].split("\n", 1)[1]
    assert json.loads(schema_text) == _Verdict.model_json_schema()
    assert "JSON" in sent[-1]["content"]  # DeepSeek's JSON mode requires the word in the prompt


def test_json_object_request_does_not_mutate_caller_messages():
    messages = [{"role": "user", "content": "hi"}]
    _json_object_request(messages, _Verdict)
    assert messages == [{"role": "user", "content": "hi"}]


def test_client_consumes_structured_output_instead_of_forwarding_it():
    client = CompletionClient("openai/fast", api_base="http://x", structured_output="json_object")
    assert client.structured_output == "json_object"
    assert "structured_output" not in client.config


def test_client_defaults_to_json_schema():
    assert CompletionClient("openai/fast", api_base="http://x").structured_output == "json_schema"


def test_client_rejects_unknown_structured_output():
    with pytest.raises(ValueError, match="structured_output"):
        CompletionClient("openai/fast", api_base="http://x", structured_output="xml")


def test_registry_forwards_structured_output_to_completion_client():
    client = client_from_config(
        "ds", {"model_name": "deepseek/deepseek-chat", "structured_output": "json_object"}
    )
    assert isinstance(client, CompletionClient)
    assert client.structured_output == "json_object"
    assert "structured_output" not in client.config


def test_responses_client_never_forwards_structured_output():
    client = ResponsesClient("openai/gpt-5-mini", structured_output="json_schema")
    assert "structured_output" not in client.config
    with pytest.raises(ValueError, match="CompletionClient only"):
        ResponsesClient("openai/gpt-5-mini", structured_output="json_object")
