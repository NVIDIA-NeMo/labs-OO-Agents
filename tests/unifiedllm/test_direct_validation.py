# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bad public input is rejected before the provider SDK is called."""

import tomllib
from pathlib import Path

import pytest

from nooa.unifiedllm.direct import anthropic_request


@pytest.mark.parametrize(
    "package", ["opentelemetry-api", "opentelemetry-sdk", "openinference-semantic-conventions"]
)
def test_core_install_declares_tracing_dependencies(package):
    metadata = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    assert any(d.startswith(package) for d in metadata["project"]["dependencies"])


@pytest.mark.parametrize("limit", [None, False, 0, -1, "10"])
def test_anthropic_requires_positive_reply_limit(limit):
    with pytest.raises(ValueError, match="max_tokens"):
        anthropic_request({"messages": [], "max_tokens": limit})


@pytest.mark.parametrize(
    "fmt",
    [{"type": "text"}, {"type": "json_object"}, {"type": "json_schema", "json_schema": {}}, 1],
)
def test_response_format_has_actionable_error(fmt):
    with pytest.raises(ValueError, match="response_format.*json_schema"):
        anthropic_request({"messages": [], "max_tokens": 10, "response_format": fmt})


def test_null_response_format_is_absent():
    assert "output_config" not in anthropic_request(
        {"messages": [], "max_tokens": 10, "response_format": None}
    )


@pytest.mark.parametrize(
    "message,field",
    [
        ({"content": "x"}, "role"),
        ({"role": "tool", "content": "x"}, "tool_call_id"),
        ({"role": "assistant", "thinking_blocks": "oops"}, "thinking_blocks"),
        ({"role": "assistant", "tool_calls": [{}]}, "tool_calls"),
        (
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "not json"}}],
            },
            "arguments",
        ),
        *[
            ({"role": "user", "content": [block]}, "content")
            for block in [
                {"type": "video"},
                {"text": "x"},
                {"type": "input_image"},
                {"type": "image_url"},
                {"type": "image_url", "image_url": {}},
                {"type": "image_url", "image_url": {"url": None}},
                {"type": "image_url", "image_url": "data:image/png,abc"},
                {"type": "file"},
                {"type": "file", "file": {"file_data": "abc"}},
            ]
        ],
    ],
)
def test_invalid_message_names_index_and_field(message, field):
    with pytest.raises(ValueError, match=rf"message\[0\].*{field}"):
        anthropic_request({"messages": [message], "max_tokens": 10})


def test_tool_arguments_may_already_be_a_dictionary():
    result = anthropic_request(
        {
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": {"x": 1}}}],
                }
            ],
        }
    )
    assert result["messages"][0]["content"][0]["input"] == {"x": 1}
