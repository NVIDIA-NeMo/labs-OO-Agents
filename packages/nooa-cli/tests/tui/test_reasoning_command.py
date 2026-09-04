# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning slash-command behavior for Responses and completion clients."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nooa_cli.tui.commands import ReasoningCommand

from nooa.unifiedllm import CompletionClient, ResponsesClient


def _bare_client(client_type, config):
    client = object.__new__(client_type)
    client.config = config
    return client


def _command(llm):
    return ReasoningCommand(
        MagicMock(),
        SimpleNamespace(default_model="gpt-5.6-sol"),
        SimpleNamespace(llm=llm),
    )


def test_responses_effort_changes_preserve_reasoning_context() -> None:
    llm = _bare_client(
        ResponsesClient,
        {"reasoning": {"effort": "medium", "context": "all_turns"}},
    )
    command = _command(llm)

    command._set_reasoning("high")
    assert llm.config["reasoning"] == {"effort": "high", "context": "all_turns"}

    command._set_reasoning("off")
    assert llm.config["reasoning"] == {"effort": "none", "context": "all_turns"}
    assert command._get_reasoning_state() == ("off", "responses")


def test_completion_off_restores_provider_default() -> None:
    llm = _bare_client(
        CompletionClient,
        {
            "reasoning_effort": "high",
            "allowed_openai_params": ["reasoning_effort"],
        },
    )
    command = _command(llm)

    command._set_reasoning("off")

    assert "reasoning_effort" not in llm.config
    assert llm.config["allowed_openai_params"] == ["reasoning_effort"]


def test_reasoning_accepts_gpt_5_6_extra_effort_levels() -> None:
    command = _command(_bare_client(ResponsesClient, {}))

    assert command.validate_args(["xhigh"]) == (True, None)
    assert command.validate_args(["max"]) == (True, None)


def test_reasoning_rejects_responses_only_levels_for_completion() -> None:
    command = _command(_bare_client(CompletionClient, {}))

    assert command.validate_args(["xhigh"]) == (
        False,
        "xhigh and max currently require a Responses API model",
    )
    assert command.validate_args(["max"]) == (
        False,
        "xhigh and max currently require a Responses API model",
    )


@pytest.mark.asyncio
async def test_reasoning_status_reports_responses_context_policy() -> None:
    llm = _bare_client(
        ResponsesClient,
        {"reasoning": {"effort": "medium", "context": "all_turns"}},
    )

    result = await _command(llm).execute([])

    assert result.success is True
    assert "(responses)" in result.outputs[0].content
    assert "**medium**" in result.outputs[0].content
    assert "context: **all_turns**" in result.outputs[0].content
