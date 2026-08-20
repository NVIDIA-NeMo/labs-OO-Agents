# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for targeted slash-command help."""

from types import SimpleNamespace

import pytest
from nooa_cli.tui.commands import HelpCommand
from nooa_cli.tui.output import HelpOutput, TextOutput

_HELP = {
    "/help [COMMAND]": "Show all commands, or detailed help for one command",
    "/model [NAME]": "Switch and save a model (currently test-model)",
    "/exit": "Exit the TUI",
    "/quit": "Exit the TUI (alias for /exit)",
}


def _help_command() -> HelpCommand:
    registry = SimpleNamespace(get_builtin_help=lambda: _HELP)
    return HelpCommand(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), registry=registry)


@pytest.mark.asyncio
async def test_help_command_without_args_lists_all_commands():
    result = await _help_command().execute([])

    assert result.success is True
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert isinstance(output, HelpOutput)
    assert output.commands == _HELP


@pytest.mark.asyncio
async def test_help_command_filters_to_requested_builtin_command():
    result = await _help_command().execute(["model"])

    assert result.success is True
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert isinstance(output, HelpOutput)
    assert output.commands == {
        "/model [NAME]": "Switch and save a model (currently test-model)",
    }


@pytest.mark.asyncio
async def test_help_command_accepts_leading_slash_for_requested_command():
    result = await _help_command().execute(["/exit"])

    assert result.success is True
    output = result.outputs[0]
    assert isinstance(output, HelpOutput)
    assert output.commands == {
        "/exit": "Exit the TUI",
    }


@pytest.mark.asyncio
async def test_help_command_reports_unknown_command_without_full_listing():
    result = await _help_command().execute(["nonexistent"])

    assert result.success is False
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert isinstance(output, TextOutput)
    assert output.level == "error"
    assert output.content == "No help found for /nonexistent. Type /help for commands."


def test_help_command_rejects_too_many_args():
    is_valid, error = _help_command().validate_args(["model", "extra"])

    assert is_valid is False
    assert error == "Usage: /help [command]"
