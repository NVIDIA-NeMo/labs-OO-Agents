# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared coding-agent skill command discovery and dispatch."""

import asyncio
from typing import Literal

import pytest
from nooa_cli.coding import CodingAgent, CodingSlashCommandRegistry

from nooa.skill import Skill, slash_command
from nooa.slash_dispatch import CoercionError
from nooa.unifiedllm import FakeLLMClient


class _WorkflowSkill(Skill):
    @slash_command(
        "diagnose",
        argument_hint="<fast|deep>",
        completions=("fast", "deep"),
    )
    async def diagnose(self, mode: Literal["fast", "deep"]) -> str:
        """Diagnose the current failure."""
        return f"Diagnose in {mode} mode."


class _BackgroundWorkflowSkill(Skill):
    @slash_command("background")
    def background(self, args: str) -> str:
        channel = self._agent.queue_manager.queue("background-test", replace=True)

        async def produce() -> None:
            await asyncio.Event().wait()

        self._agent.queue_manager.spawn(produce(), channel=channel.name)
        return args


async def test_registry_discovers_metadata_and_dispatches_typed_arguments(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    agent.skills.register("test.workflow", _WorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        assert registry.commands()[0].name == "diagnose"
        assert registry.commands()[0].description == "Diagnose the current failure."
        assert registry.commands()[0].argument_hint == "<fast|deep>"
        assert registry.commands()[0].completions == ("fast", "deep")

        result = await registry.invoke("diagnose", "deep")

        assert result.command == "diagnose"
        assert result.text == "Diagnose in deep mode."
        assert result.output_to_agent is True
    finally:
        registry.close()
        await agent.close()


async def test_registry_reports_typed_argument_errors(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    agent.skills.register("test.workflow", _WorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        with pytest.raises(CoercionError, match="cannot convert"):
            await registry.invoke("diagnose", "invalid")
    finally:
        registry.close()
        await agent.close()


async def test_registry_refresh_callback_observes_new_skill_commands(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    registry = CodingSlashCommandRegistry(agent)
    updates: list[tuple[str, ...]] = []
    registry.set_on_change(
        lambda commands: updates.append(tuple(command.name for command in commands)),
        emit=True,
    )
    try:
        agent.skills.register("test.workflow", _WorkflowSkill())
        agent.skills.activate(["test.workflow"])

        assert updates == [(), ("diagnose",)]
    finally:
        registry.close()
        await agent.close()


async def test_sync_command_runs_on_agent_loop_and_can_spawn_background_job(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    agent.skills.register("test.background", _BackgroundWorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        result = await registry.invoke("background", "started")
        handle = agent.queue_manager.job("background-test")

        assert result.text == "started"
        assert handle is not None
        assert handle.state == "running"
    finally:
        registry.close()
        await agent.close()
    assert handle is not None
    assert handle.state == "cancelled"


async def test_async_command_is_cooperatively_cancellable_on_agent_loop(tmp_path):
    started = asyncio.Event()

    class AsyncWorkflowSkill(Skill):
        @slash_command("wait")
        async def wait(self) -> str:
            assert asyncio.get_running_loop() is host_loop
            started.set()
            await asyncio.Event().wait()
            return "never"

    host_loop = asyncio.get_running_loop()
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    agent.skills.register("test.async", AsyncWorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        invocation = asyncio.create_task(registry.invoke("wait", ""))
        # Bounded: a regression that stops the coroutine reaching the host loop
        # would otherwise wedge here forever, so CI stalls instead of going red.
        await asyncio.wait_for(started.wait(), timeout=30)
        invocation.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, timeout=30)
    finally:
        registry.close()
        await agent.close()


async def test_commands_are_sorted_deduplicated_and_case_insensitive(tmp_path):
    """Three registry behaviours that no test exercised.

    Every existing case registers one command and reads it back in canonical
    case, so unsorted output, a dropped duplicate guard, and lost case
    normalisation were all invisible.
    """

    class _Beta(Skill):
        @slash_command("beta")
        def beta(self, args: str) -> str:
            """Beta."""
            return "beta"

    class _Alpha(Skill):
        @slash_command("alpha")
        def alpha(self, args: str) -> str:
            """Alpha."""
            return "alpha"

    class _AlphaAgain(Skill):
        @slash_command("alpha")
        def alpha(self, args: str) -> str:
            """Duplicate."""
            return "duplicate"

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    # Registered out of order, and with a colliding name.
    # Registry names sort opposite to the command names they provide, so
    # _commands is built in [beta, alpha] order and sorting is observable.
    agent.skills.register("test.aaa", _Beta())
    agent.skills.register("test.zzz", _Alpha())
    agent.skills.register("test.zzzz", _AlphaAgain())
    registry = CodingSlashCommandRegistry(agent)
    try:
        names = [command.name for command in registry.commands()]
        assert names == sorted(names), names
        assert names.count("alpha") == 1, names
        assert registry.get("ALPHA") is not None, "lookup is not case-insensitive"
    finally:
        registry.close()
        await agent.close()
