# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-facing descriptions survive registry discovery and ACP advertisement."""

from nooa_atom.acp.server import _available_commands_update
from nooa_atom.agent import AtomAgent, SlashCommandRegistry
from nooa_atom.agent.slash_commands import SlashCommand
from nooa_atom.session.items import CommandInfo
from nooa_atom.workspace.controls import behavior_commands

from nooa.unifiedllm import FakeLLMClient


async def test_core_descriptions_in_registry_and_acp(tmp_path):
    expected = {
        "compact": (
            "Summarize older conversation history to free up context while preserving key details."
        ),
        "skills": (
            "Browse skills, add skill folders, and choose which skills the agent can use "
            "in this workspace."
        ),
        "mcp": "Review MCP server configurations, approve access, sign in, or revoke approval.",
        "mcp-add": "Ask the agent to configure a new MCP server from the details you provide.",
        "trace-url": "Show the link to this session in the trace viewer.",
        "usage": "Show token usage for this session, including subagents when available.",
        "recover": (
            "Continue a session marked in use in a new copy, leaving the original unchanged. "
            "Without arguments, list those sessions."
        ),
        "verbosity": "Set how much agent activity is shown: silent, concise, or verbose.",
    }
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    registry = SlashCommandRegistry(agent)
    try:
        registry.set_controls(
            [
                SlashCommand.for_control(control)
                for control in behavior_commands(
                    agent, None, workspace=tmp_path, command_registry=registry
                )
            ]
        )
        commands = registry.commands()
        registry_expected = {
            name: description
            for name, description in expected.items()
            if name not in {"recover", "verbosity"}
        }
        assert {command.name: command.description for command in commands} == registry_expected
        for command in commands:
            assert command.help_entry()[1] == registry_expected[command.name]

        update = _available_commands_update(
            [
                CommandInfo(
                    name=command.name,
                    description=command.description,
                    input_hint=command.argument_hint,
                )
                for command in commands
            ]
        )
        assert {
            command.name: command.description for command in update.available_commands
        } == expected
    finally:
        registry.close()
        await agent.aclose()
