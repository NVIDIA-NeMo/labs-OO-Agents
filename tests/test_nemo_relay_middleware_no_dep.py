# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NeMo Relay middleware when nemo_relay is NOT installed.

These tests verify that install_nemo_relay() and nemo_relay_scope() raise ImportError
with helpful install instructions when nemo_relay is not available. They run
regardless of whether nemo_relay is installed by monkeypatching the flag.
"""

from unittest.mock import MagicMock

import pytest

import nooa.nemo_relay_middleware as nm
from nooa.unifiedllm import AssistantReasoning, AssistantText, LLMResponse, LLMUsage, ToolCall


def test_canonical_response_projects_to_relay_shape_without_private_state():
    response = LLMResponse(
        raw_response=object(),
        parts=(
            AssistantText(text="hello"),
            AssistantReasoning(text="plain reasoning", native={"opaque": "provider-only"}),
            ToolCall(id="call-1", name="search", arguments='{"q":"x"}'),
        ),
        parsed={"value": 42},
        finish_reason="tool_calls",
        usage=LLMUsage(input_tokens=12, output_tokens=3, cached_input_tokens=8),
    )

    payload = nm._relay_response(response)

    assert payload["message"] == {
        "role": "assistant",
        "content": "hello",
        "reasoning_content": "plain reasoning",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            }
        ],
    }
    assert payload["usage"]["cached_tokens"] == 8
    assert "llm_state" not in payload
    assert "raw_response" not in payload
    assert "parsed" not in payload


def test_canonical_response_never_exposes_private_raw_response_to_relay():
    raw = MagicMock()
    raw.model_dump.return_value = {"encrypted_content": "provider-secret"}
    response = LLMResponse(
        raw_response=raw,
        parts=(
            AssistantText(text="public answer"),
            AssistantReasoning(text="", native={"encrypted_content": "provider-secret"}),
        ),
    )

    payload = nm._relay_response(response)

    assert payload["message"]["content"] == "public answer"
    assert "provider-secret" not in repr(payload)
    raw.model_dump.assert_not_called()


def test_state_only_response_does_not_create_a_relay_message():
    payload = nm._relay_response(
        LLMResponse(parts=(AssistantReasoning(text="", native={"opaque": "provider-only"}),))
    )

    assert "message" not in payload
    assert "provider-only" not in repr(payload)


@pytest.fixture()
def _no_nemo_relay(monkeypatch):
    """Simulate nemo_relay not being installed."""
    monkeypatch.setattr(nm, "_HAS_NEMO_RELAY", False)


class TestImportErrorWhenMissing:
    """Verify ImportError is raised when nemo_relay is not available."""

    @pytest.mark.usefixtures("_no_nemo_relay")
    def test_install_nemo_relay_raises_import_error(self):
        """install_nemo_relay() raises ImportError when nemo_relay is not installed."""
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        with pytest.raises(ImportError, match="nemo_relay is required"):
            nm.install_nemo_relay(em)

    @pytest.mark.usefixtures("_no_nemo_relay")
    @pytest.mark.asyncio
    async def test_nemo_relay_scope_raises_import_error(self):
        """nemo_relay_scope() raises ImportError when nemo_relay is not installed."""
        agent = MagicMock()
        agent.event_manager = MagicMock()
        with pytest.raises(ImportError, match="nemo_relay is required"):
            async with nm.nemo_relay_scope(agent, "test-scope"):
                pass
