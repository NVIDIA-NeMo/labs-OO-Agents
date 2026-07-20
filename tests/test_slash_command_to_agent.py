# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the user-only slash command flag (@slash_command(output_to_agent=...))."""

from nooa.skill import Skill, get_slash_commands, slash_command
from nooa.slash_dispatch import SlashCommandResult


def test_slash_result_defaults_output_to_agent():
    r = SlashCommandResult(command="x", args="", value="hi")
    assert r.output_to_agent is True


def test_slash_result_output_to_agent_false():
    r = SlashCommandResult(command="x", args="", value="hi", output_to_agent=False)
    assert r.output_to_agent is False
    assert str(r) == "hi"


def test_slash_command_meta_defaults_output_to_agent_true():
    class S(Skill):
        @slash_command("foo")
        async def foo(self, args: str) -> str:
            return args

    meta, _ = get_slash_commands(S())[0]
    assert meta.output_to_agent is True


def test_slash_command_output_to_agent_false_on_meta():
    class S(Skill):
        @slash_command("bar", output_to_agent=False)
        async def bar(self, args: str) -> str:
            return args

    meta, _ = get_slash_commands(S())[0]
    assert meta.output_to_agent is False
