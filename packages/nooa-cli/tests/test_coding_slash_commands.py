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
        await started.wait()
        invocation.cancel()

        with pytest.raises(asyncio.CancelledError):
            await invocation
    finally:
        registry.close()
        await agent.close()
