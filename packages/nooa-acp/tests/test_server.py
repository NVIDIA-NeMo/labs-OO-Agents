# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ACP server surface."""

import asyncio
import os
import sys
import threading
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from acp import PROTOCOL_VERSION, RequestError, resource_link_block, text_block
from acp.schema import (
    AgentMessageChunk,
    AvailableCommandsUpdate,
    EnvVariable,
    McpServerStdio,
    UserMessageChunk,
)
from click.testing import CliRunner
from nooa_acp.cli import command
from nooa_acp.server import CodingACPAdapter
from nooa_cli.commands import discover_commands

from nooa.errors import GenerationError
from nooa.interactive import RespondReason, RespondResult
from nooa.skill import Skill, slash_command
from nooa.slash_dispatch import SlashCommandResult
from nooa.unifiedllm import FakeLLMClient


def _completed_llm() -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                "self.message('ACP response')\n"
                "return_result(RespondReason.DONE, explanation='request complete')"
            )
        },
    )


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[object] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append(update)


class _RegistrationAwareClient:
    """Model clients that discard updates for sessions they do not know yet."""

    def __init__(self) -> None:
        self.registered: set[str] = set()
        self.accepted: list[tuple[str, object]] = []
        self.dropped: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        target = self.accepted if session_id in self.registered else self.dropped
        target.append((session_id, update))


class _RejectFirstCommandsClient(_RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.rejections = 0

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        if isinstance(update, AvailableCommandsUpdate) and self.rejections == 0:
            self.rejections += 1
            raise RuntimeError("session is not registered yet")
        await super().session_update(session_id, update, **kwargs)


async def _session(adapter: CodingACPAdapter, session_id: str):
    return (await adapter._sessions.get(session_id)).value


def test_acp_command_is_discovered_as_cli_plugin():
    assert dict(discover_commands())["acp"] is command


def test_acp_command_passes_the_nvidia_key_for_nvidia_models():
    runner = CliRunner()
    with (
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.get_llm_client") as get_llm_client,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(
            command,
            ["--model", "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"],
            env={"NVIDIA_API_KEY": "nvapi-test"},
        )

    assert result.exit_code == 0
    llm_factory = serve.call_args.args[0]
    llm_factory()
    get_llm_client.assert_called_once_with(
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        client_type=None,
        api_key="nvapi-test",
    )


def test_acp_command_leaves_the_key_alone_for_other_providers():
    runner = CliRunner()
    with (
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.get_llm_client") as get_llm_client,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(
            command,
            ["--model", "openai/gpt-4o-mini"],
            env={"NVIDIA_API_KEY": "nvapi-test"},
        )

    assert result.exit_code == 0
    serve.call_args.args[0]()
    get_llm_client.assert_called_once_with("openai/gpt-4o-mini", client_type=None)


class _MCPTools:
    async def lookup(self, query: str) -> str:
        """Look up a value in the test MCP server."""
        return query


def _write_external_workflow_skill(workspace, skills_root) -> None:
    package = skills_root / "workflow_skill"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "workflow-skill"\n\n'
        '[project.entry-points."nooa.skills"]\n'
        '"nvzurich.workflow" = "workflow_skill:WorkflowSkill"\n'
    )
    (package / "__init__.py").write_text(
        "from nooa.skill import Skill, slash_command\n\n"
        "class WorkflowSkill(Skill):\n"
        "    @slash_command('diagnose', argument_hint='<mode>')\n"
        "    def diagnose(self, args: str) -> str:\n"
        '        """Diagnose the workspace."""\n'
        "        return f'Diagnose using {args} mode.'\n\n"
        "    @slash_command('skill-status', output_to_agent=False)\n"
        "    def status(self, args: str) -> str:\n"
        '        """Show workflow status."""\n'
        "        return f'status:{args}'\n"
    )
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        f"tui:\n  additional_skills_dirs:\n    - {skills_root}\n"
    )


def _write_standalone_command_skill(workspace, command: str, value: str) -> None:
    skills_root = workspace / "external-skills"
    skills_root.mkdir()
    (skills_root / f"{command}.py").write_text(
        "from nooa.skill import Skill, slash_command\n\n"
        f"class {command.title()}Skill(Skill):\n"
        f"    @slash_command('{command}', output_to_agent=False)\n"
        "    def run(self, args: str) -> str:\n"
        f"        return '{value}:' + args\n"
    )
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {skills_root}\n"
    )


def _write_packaged_command_skill(workspace, value: str) -> None:
    skills_root = workspace / "external-skills"
    checkout = skills_root / "workflow-checkout"
    package = checkout / "src" / "shared_workflow"
    package.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "workflow-distribution"\n\n'
        '[project.entry-points."nooa.skills"]\n'
        '"test.workflow" = "shared_workflow:WorkflowSkill"\n'
    )
    (package / "helper.py").write_text(f"VALUE = {value!r}\n")
    (package / "__init__.py").write_text(
        "from nooa.skill import Skill, slash_command\n\n"
        "class WorkflowSkill(Skill):\n"
        "    @slash_command('workflow', output_to_agent=False)\n"
        "    def run(self, args: str) -> str:\n"
        "        from shared_workflow.helper import VALUE\n"
        "        return VALUE + ':' + args\n"
    )
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {skills_root}\n"
    )


async def test_adapter_completes_one_session_prompt(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    initialized = await adapter.initialize(PROTOCOL_VERSION)
    session = await adapter.new_session(str(tmp_path))
    response = await adapter.prompt(session.session_id, [text_block("do the work")])

    assert initialized.protocol_version == PROTOCOL_VERSION
    assert initialized.agent_info is not None
    assert initialized.agent_info.name == "nooa-acp"
    assert initialized.agent_capabilities.load_session is True
    capabilities = initialized.agent_capabilities.session_capabilities
    assert capabilities.list is not None
    assert capabilities.close is not None
    assert response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "ACP response"
        for update in client.updates
    )
    await adapter.close()


async def test_adapter_loads_workspace_skills_and_advertises_commands(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    skills_root = tmp_path / "nemo-oo-skills"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    _write_external_workflow_skill(workspace, skills_root)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    session = await adapter.new_session(str(workspace))
    runtime = await _session(adapter, session.session_id)
    await asyncio.sleep(0)
    await runtime.bridge.flush()

    assert "nvzurich.workflow" in runtime.agent.skills.loaded()
    advertised = [
        update for update in client.updates if isinstance(update, AvailableCommandsUpdate)
    ]
    assert len(advertised) == 1
    assert [command.name for command in advertised[0].available_commands] == [
        "diagnose",
        "skill-status",
    ]
    diagnose = advertised[0].available_commands[0]
    assert diagnose.description == "Diagnose the workspace."
    assert diagnose.input is not None
    assert diagnose.input.root.hint == "<mode>"
    await adapter.close()


async def test_rejected_bootstrap_commands_do_not_poison_first_prompt(tmp_path):
    client = _RejectFirstCommandsClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    created = await adapter.new_session(str(tmp_path))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    response = await adapter.prompt(created.session_id, [text_block("continue")])

    assert client.rejections == 1
    assert response.stop_reason == "end_turn"
    assert any(isinstance(update, AvailableCommandsUpdate) for update in client.updates)
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "ACP response"
        for update in client.updates
    )
    await adapter.close()


async def test_adapter_dispatches_agent_facing_skill_command(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    skills_root = tmp_path / "nemo-oo-skills"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    _write_external_workflow_skill(workspace, skills_root)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(workspace))
    runtime = await _session(adapter, created.session_id)
    notifications: list[dict[str, list[object]]] = []

    async def handle(notification):
        notifications.append(notification)
        return RespondResult(kind=RespondReason.DONE, explanation="done")

    with patch.object(runtime.agent, "handle", side_effect=handle):
        response = await adapter.prompt(
            created.session_id,
            [text_block("/diagnose deep")],
        )

    slash_result = notifications[0]["slash_commands"][0]
    assert isinstance(slash_result, SlashCommandResult)
    assert slash_result.command == "diagnose"
    assert slash_result.args == "deep"
    assert slash_result.text == "Diagnose using deep mode."
    assert response.stop_reason == "end_turn"
    assert runtime.handle.turns()[0].content == "/diagnose deep"
    await adapter.close()


async def test_adapter_renders_user_only_skill_command_without_llm_turn(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    skills_root = tmp_path / "nemo-oo-skills"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    _write_external_workflow_skill(workspace, skills_root)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(workspace))
    runtime = await _session(adapter, created.session_id)

    with patch.object(runtime.agent, "handle", AsyncMock()) as handle:
        response = await adapter.prompt(
            created.session_id,
            [text_block("/skill-status ready")],
        )

    handle.assert_not_awaited()
    assert response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "status:ready"
        for update in client.updates
    )
    await adapter.close()


async def test_adapter_renders_typed_command_error_without_llm_turn(tmp_path):
    class TypedSkill(Skill):
        @slash_command("typed", argument_hint="<fast|deep>")
        def typed(self, mode: Literal["fast", "deep"]) -> str:
            """Run a typed workflow."""
            return mode

    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, created.session_id)
    runtime.agent.skills.register("test.typed", TypedSkill())

    with patch.object(runtime.agent, "handle", AsyncMock()) as handle:
        response = await adapter.prompt(
            created.session_id,
            [text_block("/typed invalid")],
        )

    handle.assert_not_awaited()
    assert response.stop_reason == "end_turn"
    message = next(
        update.content.text
        for update in client.updates
        if isinstance(update, AgentMessageChunk) and update.content.text.startswith("/typed:")
    )
    assert "cannot convert 'invalid'" in message
    assert "Usage: `/typed <fast|deep>`" in message
    await adapter.close()


async def test_unknown_slash_command_is_forwarded_as_an_ordinary_prompt(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, created.session_id)
    result = RespondResult(kind=RespondReason.DONE, explanation="done")

    with patch.object(
        runtime.dispatcher,
        "submit",
        AsyncMock(return_value=result),
    ) as submit:
        response = await adapter.prompt(
            created.session_id,
            [text_block("/not-advertised keep this")],
        )

    submit.assert_awaited_once_with("/not-advertised keep this")
    assert response.stop_reason == "end_turn"
    await adapter.close()


async def test_initial_commands_are_deferred_until_new_session_returns(tmp_path):
    client = _RegistrationAwareClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    created = await adapter.new_session(str(tmp_path))
    client.registered.add(created.session_id)
    await asyncio.sleep(0)
    runtime = await _session(adapter, created.session_id)
    await runtime.bridge.flush()

    assert client.dropped == []
    assert len(client.accepted) == 1
    assert isinstance(client.accepted[0][1], AvailableCommandsUpdate)
    await adapter.close()


async def test_first_prompt_resends_commands_if_initial_update_was_dropped(tmp_path):
    client = _RegistrationAwareClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    created = await adapter.new_session(str(tmp_path))
    await asyncio.sleep(0)
    runtime = await _session(adapter, created.session_id)
    await runtime.bridge.flush()
    assert len(client.dropped) == 1

    client.registered.add(created.session_id)
    response = await adapter.prompt(created.session_id, [text_block("hello")])

    assert response.stop_reason == "end_turn"
    assert any(isinstance(update, AvailableCommandsUpdate) for _, update in client.accepted)
    await adapter.close()


async def test_load_session_advertises_commands_to_registered_session(tmp_path):
    create_client = _RecordingClient()
    create_adapter = CodingACPAdapter(_completed_llm)
    create_adapter.on_connect(create_client)  # type: ignore[arg-type]
    created = await create_adapter.new_session(str(tmp_path))
    await create_adapter.close()

    load_client = _RegistrationAwareClient()
    load_client.registered.add(created.session_id)
    load_adapter = CodingACPAdapter(_completed_llm)
    load_adapter.on_connect(load_client)  # type: ignore[arg-type]
    await load_adapter.load_session(str(tmp_path), created.session_id)
    await asyncio.sleep(0)
    runtime = await _session(load_adapter, created.session_id)
    await runtime.bridge.flush()

    assert load_client.dropped == []
    assert any(isinstance(update, AvailableCommandsUpdate) for _, update in load_client.accepted)
    await load_adapter.close()


async def test_cancel_interrupts_async_slash_command_and_session_remains_usable(tmp_path):
    class BlockingSkill(Skill):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.cancelled = threading.Event()

        @slash_command("block", output_to_agent=False)
        async def block(self, args: str) -> str:
            """Block until the command is cancelled."""
            del args
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            return "unreachable"

    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, created.session_id)
    skill = BlockingSkill()
    runtime.agent.skills.register("test.blocking", skill)

    prompt_task = asyncio.create_task(
        adapter.prompt(created.session_id, [text_block("/block now")])
    )
    assert await asyncio.to_thread(skill.started.wait, 1)
    await adapter.cancel(created.session_id)
    cancelled = await asyncio.wait_for(prompt_task, timeout=1)

    assert cancelled.stop_reason == "cancelled"
    assert await asyncio.to_thread(skill.cancelled.wait, 1)
    resumed = await asyncio.wait_for(
        adapter.prompt(created.session_id, [text_block("continue")]),
        timeout=2,
    )
    assert resumed.stop_reason == "end_turn"
    await adapter.close()


async def test_cancel_clears_agent_facing_slash_result_and_session_remains_usable(
    tmp_path,
):
    class AgentFacingSkill(Skill):
        @slash_command("agent-work")
        def run(self, args: str) -> str:
            """Send work to the agent."""
            return f"work:{args}"

    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, created.session_id)
    runtime.agent.skills.register("test.agent-facing", AgentFacingSkill())
    started = asyncio.Event()

    async def blocking_handle(notification):
        slash = notification["slash_commands"][0]
        assert isinstance(slash, SlashCommandResult)
        assert slash.text == "work:first"
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with patch.object(runtime.agent, "handle", side_effect=blocking_handle):
        prompt_task = asyncio.create_task(
            adapter.prompt(created.session_id, [text_block("/agent-work first")])
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        await adapter.cancel(created.session_id)
        cancelled = await asyncio.wait_for(prompt_task, timeout=1)

    assert cancelled.stop_reason == "cancelled"
    assert runtime.agent.queue_manager.get_channel("slash_commands").drain() == []

    observed: list[dict[str, list[object]]] = []

    async def resumed_handle(notification):
        observed.append(notification)
        return RespondResult(kind=RespondReason.DONE, explanation="done")

    with patch.object(runtime.agent, "handle", side_effect=resumed_handle):
        resumed = await adapter.prompt(created.session_id, [text_block("continue")])

    assert resumed.stop_reason == "end_turn"
    assert observed == [{"user_messages": ["continue"]}]
    await adapter.close()


async def test_adapter_republishes_commands_after_skill_activation(tmp_path):
    class LaterSkill(Skill):
        @slash_command("later")
        def later(self) -> str:
            """Run the later workflow."""
            return "later"

    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, created.session_id)
    await runtime.bridge.flush()

    runtime.agent.skills.register("test.later", LaterSkill())
    runtime.agent.skills.activate(["test.later"])
    await runtime.bridge.flush()

    advertised = [
        update for update in client.updates if isinstance(update, AvailableCommandsUpdate)
    ]
    assert len(advertised) == 2
    assert [command.name for command in advertised[-1].available_commands] == ["later"]
    await adapter.close()


async def test_adapter_replaces_advertised_commands_after_skill_reload(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    skills_root = tmp_path / "nemo-oo-skills"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    _write_external_workflow_skill(workspace, skills_root)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(workspace))
    runtime = await _session(adapter, created.session_id)
    await asyncio.sleep(0)
    await runtime.bridge.flush()
    client.updates.clear()

    module = skills_root / "workflow_skill" / "__init__.py"
    module.write_text(
        "from nooa.skill import Skill, slash_command\n\n"
        "class WorkflowSkill(Skill):\n"
        "    @slash_command('repair', argument_hint='<mode>')\n"
        "    def repair(self, args: str) -> str:\n"
        '        """Repair the workspace."""\n'
        "        return f'Repair using {args} mode (reloaded).'\n"
    )
    stat = module.stat()
    os.utime(module, (stat.st_atime + 2, stat.st_mtime + 2))

    result = await runtime.agent.skills.reload("nvzurich.workflow")
    await runtime.bridge.flush()

    assert result == "Reloaded nvzurich.workflow (self.workflow)"
    advertised = [
        update for update in client.updates if isinstance(update, AvailableCommandsUpdate)
    ]
    assert len(advertised) == 1
    assert [command.name for command in advertised[0].available_commands] == ["repair"]
    invoked = await runtime.commands.invoke("repair", "deep")
    assert invoked.text == "Repair using deep mode (reloaded)."
    await adapter.close()


async def test_failed_skill_reload_keeps_previous_command_and_advertisement(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    skills_root = tmp_path / "nemo-oo-skills"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    _write_external_workflow_skill(workspace, skills_root)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(workspace))
    runtime = await _session(adapter, created.session_id)
    await asyncio.sleep(0)
    await runtime.bridge.flush()
    client.updates.clear()

    module = skills_root / "workflow_skill" / "__init__.py"
    module.write_text("this is not valid Python !!!\n")
    stat = module.stat()
    os.utime(module, (stat.st_atime + 2, stat.st_mtime + 2))

    result = await runtime.agent.skills.reload("nvzurich.workflow")
    await runtime.bridge.flush()

    assert result.startswith("Reload failed for nvzurich.workflow:")
    assert not any(isinstance(update, AvailableCommandsUpdate) for update in client.updates)
    assert [command.name for command in runtime.commands.commands()] == [
        "diagnose",
        "skill-status",
    ]
    invoked = await runtime.commands.invoke("diagnose", "deep")
    assert invoked.text == "Diagnose using deep mode."
    await adapter.close()


async def test_adapter_includes_resource_links_in_prompt(tmp_path):
    llm = _completed_llm()
    client = _RecordingClient()
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    session = await adapter.new_session(str(tmp_path))
    await adapter.prompt(
        session.session_id,
        [resource_link_block("README", "file:///workspace/README.md")],
    )

    assert "Resource README: file:///workspace/README.md" in str(llm.last_messages)
    await adapter.close()


async def test_adapter_preserves_prompt_whitespace(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)
    result = RespondResult(kind=RespondReason.DONE, explanation="done")
    submit = AsyncMock(return_value=result)
    with patch.object(runtime.dispatcher, "submit", submit):
        await adapter.prompt(session.session_id, [text_block("  indented\n")])

    submit.assert_awaited_once_with("  indented\n")
    await adapter.close()


async def test_adapter_connects_baseline_stdio_mcp_servers(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(
        name="lookup",
        command="lookup-server",
        args=["--stdio"],
        env=[EnvVariable(name="TOKEN", value="test")],
    )

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ) as create:
        session = await adapter.new_session(str(tmp_path), mcp_servers=[server])

    create.assert_awaited_once_with(
        "lookup",
        command="lookup-server",
        args=["--stdio"],
        env={"TOKEN": "test"},
    )
    runtime = await _session(adapter, session.session_id)
    assert "mcp.lookup" in runtime.agent.skills.loaded()
    assert "mcp.lookup" in runtime.agent.skills.activated()
    await adapter.close()


async def test_adapter_allows_independent_session_creation(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(name="lookup", command="lookup-server", args=[], env=[])
    started = asyncio.Event()
    release = asyncio.Event()

    async def create_mcp(*args, **kwargs):
        started.set()
        await release.wait()
        return _MCPTools()

    with patch("nooa_acp.server.MCPManager.create_stdio_server", side_effect=create_mcp):
        first_session = asyncio.create_task(
            adapter.new_session(str(tmp_path), mcp_servers=[server])
        )
        await asyncio.wait_for(started.wait(), 1)
        second = await asyncio.wait_for(adapter.new_session(str(tmp_path)), 1)
        release.set()
        first = await first_session

    assert first.session_id != second.session_id
    await adapter.close()


async def test_adapter_routes_distinct_workspace_commands_to_their_sessions(tmp_path, monkeypatch):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    user_config = tmp_path / "user-config"
    alpha.mkdir()
    beta.mkdir()
    user_config.mkdir()
    _write_standalone_command_skill(alpha, "alpha", "from-alpha")
    _write_standalone_command_skill(beta, "beta", "from-beta")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RegistrationAwareClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    alpha_session = await adapter.new_session(str(alpha))
    client.registered.add(alpha_session.session_id)
    beta_session = await adapter.new_session(str(beta))
    client.registered.add(beta_session.session_id)
    await asyncio.sleep(0)
    for session_id in (alpha_session.session_id, beta_session.session_id):
        runtime = await _session(adapter, session_id)
        await runtime.bridge.flush()

    alpha_response = await adapter.prompt(
        alpha_session.session_id,
        [text_block("/alpha one")],
    )
    beta_response = await adapter.prompt(
        beta_session.session_id,
        [text_block("/beta two")],
    )

    assert alpha_response.stop_reason == "end_turn"
    assert beta_response.stop_reason == "end_turn"
    commands_by_session = {
        session_id: [command.name for command in update.available_commands]
        for session_id, update in client.accepted
        if isinstance(update, AvailableCommandsUpdate)
    }
    assert commands_by_session[alpha_session.session_id] == ["alpha"]
    assert commands_by_session[beta_session.session_id] == ["beta"]
    messages = {
        session_id: update.content.text
        for session_id, update in client.accepted
        if isinstance(update, AgentMessageChunk)
    }
    assert messages[alpha_session.session_id] == "from-alpha:one"
    assert messages[beta_session.session_id] == "from-beta:two"
    await adapter.close()


async def test_adapter_rejects_conflicting_packaged_skills_without_contamination(
    tmp_path, monkeypatch
):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    user_config = tmp_path / "user-config"
    alpha.mkdir()
    beta.mkdir()
    user_config.mkdir()
    _write_packaged_command_skill(alpha, "from-alpha")
    _write_packaged_command_skill(beta, "from-beta")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    alpha_session = await adapter.new_session(str(alpha))
    with pytest.raises(RequestError) as error:
        await adapter.new_session(str(beta))
    assert error.value.code == -32600
    assert "Launch a separate ACP server" in error.value.data["reason"]

    alpha_response = await adapter.prompt(
        alpha_session.session_id,
        [text_block("/workflow one")],
    )
    assert alpha_response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "from-alpha:one"
        for update in client.updates
    )

    await adapter.close_session(alpha_session.session_id)
    beta_session = await adapter.new_session(str(beta))
    beta_response = await adapter.prompt(
        beta_session.session_id,
        [text_block("/workflow two")],
    )
    assert beta_response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "from-beta:two"
        for update in client.updates
    )
    await adapter.close()


@pytest.mark.parametrize("first_to_close", [0, 1])
async def test_same_checkout_stays_live_until_last_acp_session_closes(
    tmp_path, monkeypatch, first_to_close
):
    workspace = tmp_path / "workspace"
    user_config = tmp_path / "user-config"
    workspace.mkdir()
    user_config.mkdir()
    _write_packaged_command_skill(workspace, "shared")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    sessions = [
        await adapter.new_session(str(workspace)),
        await adapter.new_session(str(workspace)),
    ]
    remaining = 1 - first_to_close
    await adapter.close_session(sessions[first_to_close].session_id)

    response = await adapter.prompt(
        sessions[remaining].session_id,
        [text_block("/workflow still-live")],
    )
    assert response.stop_reason == "end_turn"
    assert "shared_workflow" in sys.modules
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "shared:still-live"
        for update in client.updates
    )

    await adapter.close_session(sessions[remaining].session_id)
    assert "shared_workflow" not in sys.modules
    await adapter.close()


async def test_adapter_rejects_two_prompts_for_same_session(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def submit(_text: str):
        started.set()
        await release.wait()
        return RespondResult(kind=RespondReason.DONE, explanation="done")

    with patch.object(session.dispatcher, "submit", side_effect=submit):
        first = asyncio.create_task(adapter.prompt(created.session_id, [text_block("first")]))
        await asyncio.wait_for(started.wait(), 1)
        with pytest.raises(RequestError):
            await adapter.prompt(created.session_id, [text_block("second")])
        release.set()
        assert (await first).stop_reason == "end_turn"

    await adapter.close()


async def test_adapter_lists_closes_loads_and_replays_durable_session(tmp_path):
    first_client = _RecordingClient()
    first_adapter = CodingACPAdapter(_completed_llm)
    first_adapter.on_connect(first_client)  # type: ignore[arg-type]
    created = await first_adapter.new_session(str(tmp_path))
    await first_adapter.prompt(created.session_id, [text_block("remember this")])

    listed = await first_adapter.list_sessions(str(tmp_path))
    assert [session.session_id for session in listed.sessions] == [created.session_id]
    assert listed.sessions[0].cwd == str(tmp_path)
    await first_adapter.close_session(created.session_id)
    with pytest.raises(RequestError):
        await first_adapter.prompt(created.session_id, [text_block("closed")])
    await first_adapter.close()

    replay_client = _RecordingClient()
    replay_adapter = CodingACPAdapter(_completed_llm)
    replay_adapter.on_connect(replay_client)  # type: ignore[arg-type]
    await replay_adapter.load_session(str(tmp_path), created.session_id)

    replayed_text = [
        update.content.text
        for update in replay_client.updates
        if isinstance(update, AgentMessageChunk)
    ]
    assert replayed_text == ["ACP response"]
    replayed_user_text = [
        update.content.text
        for update in replay_client.updates
        if isinstance(update, UserMessageChunk)
    ]
    assert replayed_user_text == ["remember this"]
    await replay_adapter.close()


async def test_adapter_reports_unknown_session_on_load(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    with pytest.raises(RequestError):
        await adapter.load_session(str(tmp_path), "missing")

    await adapter.close()


async def test_adapter_rejects_duplicate_mcp_names_before_startup(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    servers = [
        McpServerStdio(name="lookup", command="first", args=[], env=[]),
        McpServerStdio(name="lookup", command="second", args=[], env=[]),
    ]

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ) as create:
        with pytest.raises(RequestError):
            await adapter.new_session(str(tmp_path), mcp_servers=servers)

    create.assert_not_awaited()
    await adapter.close()


@pytest.mark.parametrize(
    ("message", "stop_reason"),
    [
        (
            "Empty response: the model used all available output tokens on reasoning; "
            "increase `max_tokens`.",
            "max_tokens",
        ),
        (
            "Generation failed after 10 iterations (max_iterations=10).",
            "max_turn_requests",
        ),
    ],
)
async def test_adapter_maps_generation_limits_to_stop_reasons(
    tmp_path, message: str, stop_reason: str
):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)

    with patch.object(runtime.dispatcher, "submit", side_effect=GenerationError(message)):
        response = await adapter.prompt(session.session_id, [text_block("do the work")])

    assert response.stop_reason == stop_reason
    await adapter.close()


async def test_adapter_propagates_unrelated_generation_errors(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)

    with (
        patch.object(
            runtime.dispatcher,
            "submit",
            side_effect=GenerationError("LLM API error after retries"),
        ),
        pytest.raises(GenerationError, match="LLM API error"),
    ):
        await adapter.prompt(session.session_id, [text_block("do the work")])

    await adapter.close()


async def test_prompt_on_a_closing_session_is_a_clean_protocol_error(tmp_path):
    """A prompt racing a close must not surface as a raw internal error.

    turn() raises SessionRuntimeClosedError as well as SessionBusyError; only
    the latter was translated, so the client got -32603 with no actionable
    reason instead of a typed protocol error.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    await adapter.initialize(PROTOCOL_VERSION)
    session = await adapter.new_session(str(tmp_path))
    runtime = await adapter._sessions.get(session.session_id)
    await runtime.close()

    with pytest.raises(RequestError):
        await adapter.prompt(session.session_id, [text_block("do the work")])


async def test_sessions_on_different_workspaces_get_separate_library_dirs(tmp_path):
    """One ACP process serves many workspaces, so libs cannot be process-global.

    SkillWriting puts the directory on sys.path and activates local.*, so a
    shared one exposes code the agent wrote for one repo to a session working
    on another.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    first_root = tmp_path / "repo-a"
    second_root = tmp_path / "repo-b"
    first_root.mkdir()
    second_root.mkdir()

    first = await adapter.new_session(str(first_root))
    second = await adapter.new_session(str(second_root))

    first_agent = await _session(adapter, first.session_id)
    second_agent = await _session(adapter, second.session_id)
    assert first_agent.agent.libs._path == first_root / ".nooa" / "libs"
    assert second_agent.agent.libs._path == second_root / ".nooa" / "libs"
    await adapter.close()
