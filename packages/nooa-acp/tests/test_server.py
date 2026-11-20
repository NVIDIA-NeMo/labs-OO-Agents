# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ACP server surface."""

import asyncio
import os
import sys
import threading
from contextlib import suppress
from typing import Literal
from unittest.mock import AsyncMock, call, patch

import pytest
from acp import PROTOCOL_VERSION, RequestError, resource_link_block, text_block
from acp.schema import (
    AcpMcpServer,
    AgentMessageChunk,
    AvailableCommandsUpdate,
    EnvVariable,
    HttpHeader,
    HttpMcpServer,
    McpServerStdio,
    SseMcpServer,
    UserMessageChunk,
)
from click.testing import CliRunner
from nooa_acp.cli import command
from nooa_acp.server import CodingACPAdapter
from nooa_cli.commands import discover_commands

from nooa.context_blocks.events import ToolCallEvent
from nooa.errors import GenerationError
from nooa.interactive import Done
from nooa.skill import Skill, slash_command
from nooa.slash_dispatch import SlashCommandResult
from nooa.strategies.codeact import MAX_ITERATIONS_MESSAGE, OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path_factory, monkeypatch):
    """Keep every session out of the developer's real user configuration.

    CodingACPAdapter builds each session with load_coding_skills_dirs(root),
    which reads the user layer and conventional roots such as ~/.agents/skills.
    Without this, a developer who has any of those loads extra skills into these
    sessions, and assertions about advertised commands fail on a working machine
    while passing on a clean CI runner.
    """
    home = tmp_path_factory.mktemp("isolated-home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(home / "user-config"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    (home / "user-config").mkdir(parents=True, exist_ok=True)
    return home


# JSON-RPC code for resource_not_found; asserting it distinguishes a typed
# protocol error from the generic -32603 internal_error.
_RESOURCE_NOT_FOUND = -32002


@pytest.fixture(autouse=True)
async def close_every_adapter(monkeypatch):
    """Guarantee teardown for every adapter a test builds.

    Each test closes its adapter as the last statement, so a failing assertion
    leaks the pump task, bootstrap task, agent, MCP tools and SQLite handle onto
    the shared event loop for the rest of the module — which has been observed
    turning one failure into unrelated intermittent failures later. Closing is
    idempotent, so tests keep their explicit close and this only covers the
    paths that do not reach it.
    """
    built: list[CodingACPAdapter] = []
    original = CodingACPAdapter.__init__

    def tracking_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(CodingACPAdapter, "__init__", tracking_init)
    yield built

    for adapter in built:
        with suppress(Exception):
            await adapter.close()


def _completed_llm() -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                "self.message('ACP response')\nreturn_result(Done(explanation='request complete'))"
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
        patch("nooa.unifiedllm.load_llm") as load_llm,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(
            command,
            ["--model", "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"],
            env={"NVIDIA_API_KEY": "nvapi-test"},
        )

    assert result.exit_code == 0
    assert serve.call_args.kwargs["model_alias"] == ("nvidia_nim/nvidia/nemotron-3-super-120b-a12b")
    llm_factory = serve.call_args.args[0]
    llm_factory()
    load_llm.assert_called_once_with(
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        client_type=None,
        api_key="nvapi-test",
    )


def test_acp_command_leaves_the_key_alone_for_other_providers():
    runner = CliRunner()
    with (
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.load_llm") as load_llm,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(
            command,
            ["--model", "openai/gpt-4o-mini"],
            env={"NVIDIA_API_KEY": "nvapi-test"},
        )

    assert result.exit_code == 0
    assert serve.call_args.kwargs["model_alias"] == "openai/gpt-4o-mini"
    serve.call_args.args[0]()
    load_llm.assert_called_once_with("openai/gpt-4o-mini", client_type=None)


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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(workspace))
    runtime = await _session(adapter, created.session_id)
    notifications: list[dict[str, list[object]]] = []

    async def handle(notification):
        notifications.append(notification)
        return Done(explanation="done")

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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, created.session_id)
    result = Done(explanation="done")

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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    create_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    create_adapter.on_connect(create_client)  # type: ignore[arg-type]
    created = await create_adapter.new_session(str(tmp_path))
    await create_adapter.close()

    load_client = _RegistrationAwareClient()
    load_client.registered.add(created.session_id)
    load_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
        return Done(explanation="done")

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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(lambda: llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]

    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)
    result = Done(explanation="done")
    submit = AsyncMock(return_value=result)
    with patch.object(runtime.dispatcher, "submit", submit):
        await adapter.prompt(session.session_id, [text_block("  indented\n")])

    submit.assert_awaited_once_with("  indented\n")
    await adapter.close()


async def test_adapter_connects_baseline_stdio_mcp_servers(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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


async def test_adapter_connects_forwarded_http_and_sse_mcp_servers(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    servers = [
        HttpMcpServer(
            name="remote-http",
            url="https://mcp.example.test/mcp",
            headers=[HttpHeader(name="Authorization", value="Bearer test")],
            type="http",
        ),
        SseMcpServer(
            name="remote-sse",
            url="https://mcp.example.test/sse",
            headers=[],
            type="sse",
        ),
    ]

    with patch(
        "nooa_acp.server.MCPManager.create_url_server",
        new=AsyncMock(side_effect=[_MCPTools(), _MCPTools()]),
    ) as create:
        session = await adapter.new_session(str(tmp_path), mcp_servers=servers)

    assert create.await_args_list == [
        call(
            "remote-http",
            "https://mcp.example.test/mcp",
            headers={"Authorization": "Bearer test"},
            transport="streamable-http",
        ),
        call(
            "remote-sse",
            "https://mcp.example.test/sse",
            headers={},
            transport="sse",
        ),
    ]
    runtime = await _session(adapter, session.session_id)
    assert {"mcp.remote-http", "mcp.remote-sse"}.issubset(runtime.agent.skills.loaded())
    assert {"mcp.remote-http", "mcp.remote-sse"}.issubset(runtime.agent.skills.activated())
    await adapter.close()


async def test_adapter_skips_forwarded_acp_transport_mcp_server(tmp_path):
    client = _RegistrationAwareClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = AcpMcpServer(name="proxied", server_id="server-1", type="acp")

    with (
        patch(
            "nooa_acp.server.MCPManager.create_stdio_server",
            new=AsyncMock(),
        ) as create_stdio,
        patch(
            "nooa_acp.server.MCPManager.create_url_server",
            new=AsyncMock(),
        ) as create_url,
    ):
        created = await adapter.new_session(str(tmp_path), mcp_servers=[server])

    create_stdio.assert_not_awaited()
    create_url.assert_not_awaited()
    client.registered.add(created.session_id)
    await asyncio.sleep(0)
    runtime = await _session(adapter, created.session_id)
    await runtime.bridge.flush()
    warning = next(
        update.content.text
        for _, update in client.accepted
        if isinstance(update, AgentMessageChunk)
    )
    assert "session is still usable" in warning
    assert "unsupported ACP server type AcpMcpServer" in warning
    await adapter.close()


async def test_adapter_loads_healthy_mcp_when_another_server_fails(tmp_path):
    client = _RegistrationAwareClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    servers = [
        HttpMcpServer(
            name="offline",
            url="https://offline.example.test/mcp",
            headers=[],
            type="http",
        ),
        HttpMcpServer(
            name="healthy",
            url="https://healthy.example.test/mcp",
            headers=[],
            type="http",
        ),
    ]

    with patch(
        "nooa_acp.server.MCPManager.create_url_server",
        new=AsyncMock(
            side_effect=[
                RuntimeError("Could not connect: ConnectionRefusedError: refused"),
                _MCPTools(),
            ]
        ),
    ):
        created = await adapter.new_session(str(tmp_path), mcp_servers=servers)

    client.registered.add(created.session_id)
    await asyncio.sleep(0)
    runtime = await _session(adapter, created.session_id)
    await runtime.bridge.flush()
    assert "mcp.offline" not in runtime.agent.skills.loaded()
    assert "mcp.healthy" in runtime.agent.skills.loaded()
    warning = next(
        update.content.text
        for _, update in client.accepted
        if isinstance(update, AgentMessageChunk)
    )
    assert "MCP server 'offline' was not loaded" in warning
    assert "ConnectionRefusedError: refused" in warning
    await adapter.close()


async def test_adapter_allows_independent_session_creation(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Needs the external-package lifecycle work (SkillPackageConflictError, "
        "package release on last close) from dev/acp-on-foundation's bed5493. That "
        "commit rewrites skill_registry.py along a different line than the version "
        "merged in #127, so it is deferred to its own PR. Remove this marker when it "
        "lands — strict=True makes an unexpected pass fail here."
    ),
)
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Needs the external-package lifecycle work (SkillPackageConflictError, "
        "package release on last close) from dev/acp-on-foundation's bed5493. That "
        "commit rewrites skill_registry.py along a different line than the version "
        "merged in #127, so it is deferred to its own PR. Remove this marker when it "
        "lands — strict=True makes an unexpected pass fail here."
    ),
)
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def submit(_text: str):
        started.set()
        await release.wait()
        return Done(explanation="done")

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
    first_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
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
    replay_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    replay_adapter.on_connect(replay_client)  # type: ignore[arg-type]
    await replay_adapter.load_session(str(tmp_path), created.session_id)

    replayed_text = [
        update.content.text
        for update in replay_client.updates
        if isinstance(update, AgentMessageChunk)
    ]
    # Replay terminates each turn so consecutive turns from one speaker keep
    # their own boundary; see test_replay_separates_consecutive_turns_from_one_speaker.
    assert replayed_text == ["ACP response\n"]
    replayed_user_text = [
        update.content.text
        for update in replay_client.updates
        if isinstance(update, UserMessageChunk)
    ]
    assert replayed_user_text == ["remember this\n"]
    await replay_adapter.close()


async def test_loading_session_is_not_available_until_replay_finishes(tmp_path):
    create_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    create_adapter.on_connect(_RecordingClient())  # type: ignore[arg-type]
    created = await create_adapter.new_session(str(tmp_path))
    await create_adapter.close()

    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(_RecordingClient())  # type: ignore[arg-type]
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def blocked_replay(_handle):
        replay_started.set()
        await release_replay.wait()

    with patch.object(adapter, "_replay_session", side_effect=blocked_replay):
        loading = asyncio.create_task(adapter.load_session(str(tmp_path), created.session_id))
        await asyncio.wait_for(replay_started.wait(), timeout=1)
        with pytest.raises(RequestError) as exc_info:
            await adapter.prompt(created.session_id, [text_block("too soon")])
        assert exc_info.value.code == _RESOURCE_NOT_FOUND
        with pytest.raises(RequestError) as exc_info:
            await adapter.close_session(created.session_id)
        assert exc_info.value.code == _RESOURCE_NOT_FOUND
        release_replay.set()
        await loading

    response = await adapter.prompt(created.session_id, [text_block("now ready")])
    assert response.stop_reason == "end_turn"
    await adapter.close()


async def test_adapter_loads_durable_session_when_forwarded_mcp_is_unavailable(tmp_path):
    create_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    create_adapter.on_connect(_RecordingClient())  # type: ignore[arg-type]
    created = await create_adapter.new_session(str(tmp_path))
    await create_adapter.close()

    load_client = _RecordingClient()
    load_adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    load_adapter.on_connect(load_client)  # type: ignore[arg-type]
    server = HttpMcpServer(
        name="offline",
        url="https://offline.example.test/mcp",
        headers=[],
        type="http",
    )
    with patch(
        "nooa_acp.server.MCPManager.create_url_server",
        new=AsyncMock(
            side_effect=RuntimeError(
                "Could not connect to MCP server 'offline': ConnectionRefusedError: refused"
            )
        ),
    ):
        await load_adapter.load_session(str(tmp_path), created.session_id, mcp_servers=[server])

    await asyncio.sleep(0)
    runtime = await _session(load_adapter, created.session_id)
    await runtime.bridge.flush()
    assert "mcp.offline" not in runtime.agent.skills.loaded()
    warning = next(
        update.content.text
        for update in load_client.updates
        if isinstance(update, AgentMessageChunk)
    )
    assert "session is still usable" in warning
    assert "ConnectionRefusedError: refused" in warning
    await load_adapter.close()


async def test_adapter_reports_unknown_session_on_load(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]

    with pytest.raises(RequestError):
        await adapter.load_session(str(tmp_path), "missing")

    await adapter.close()


async def test_adapter_skips_duplicate_mcp_names_without_failing_startup(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    servers = [
        McpServerStdio(name="lookup", command="first", args=[], env=[]),
        HttpMcpServer(
            name="lookup",
            url="https://mcp.example.test/mcp",
            headers=[],
            type="http",
        ),
    ]

    with (
        patch(
            "nooa_acp.server.MCPManager.create_stdio_server",
            new=AsyncMock(return_value=_MCPTools()),
        ) as create_stdio,
        patch(
            "nooa_acp.server.MCPManager.create_url_server",
            new=AsyncMock(return_value=_MCPTools()),
        ) as create_url,
    ):
        created = await adapter.new_session(str(tmp_path), mcp_servers=servers)

    create_stdio.assert_awaited_once()
    create_url.assert_not_awaited()
    await asyncio.sleep(0)
    runtime = await _session(adapter, created.session_id)
    await runtime.bridge.flush()
    assert "mcp.lookup" in runtime.agent.skills.loaded()
    warning = next(
        update.content.text for update in client.updates if isinstance(update, AgentMessageChunk)
    )
    assert "another server has the same name" in warning
    await adapter.close()


@pytest.mark.parametrize(
    ("message", "stop_reason"),
    [
        (OUTPUT_TOKENS_EXHAUSTED_MESSAGE, "max_tokens"),
        (
            MAX_ITERATIONS_MESSAGE.format(iterations=10, max_iterations=10, method="work"),
            "max_turn_requests",
        ),
        (
            "Generation failed after 3 errors (max_retries=3). "
            "Unable to generate valid code for `work`.",
            "max_turn_requests",
        ),
    ],
)
async def test_adapter_maps_generation_limits_to_stop_reasons(
    tmp_path, message: str, stop_reason: str
):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)

    with patch.object(runtime.dispatcher, "submit", side_effect=GenerationError(message)):
        response = await adapter.prompt(session.session_id, [text_block("do the work")])

    assert response.stop_reason == stop_reason
    await adapter.close()


async def test_adapter_propagates_generation_error_message_to_acp_client(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)
    message = "LLM API error after retries: Missing credentials"

    with (
        patch.object(
            runtime.dispatcher,
            "submit",
            side_effect=GenerationError(message),
        ),
        pytest.raises(RequestError) as raised,
    ):
        await adapter.prompt(session.session_id, [text_block("do the work")])

    assert raised.value.code == -32603
    assert str(raised.value) == message
    assert raised.value.data == {"details": message}
    await adapter.close()


async def test_prompt_on_a_closing_session_is_a_clean_protocol_error(tmp_path):
    """A prompt racing a close must not surface as a raw internal error.

    turn() raises SessionRuntimeClosedError as well as SessionBusyError; only
    the latter was translated, so the client got -32603 with no actionable
    reason instead of a typed protocol error.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]

    await adapter.initialize(PROTOCOL_VERSION)
    session = await adapter.new_session(str(tmp_path))
    runtime = await adapter._sessions.get(session.session_id)
    await runtime.close()

    # The code, not merely "some RequestError": a bare raises() accepts the
    # -32603 internal_error this test exists to rule out.
    with pytest.raises(RequestError) as raised:
        await adapter.prompt(session.session_id, [text_block("do the work")])
    assert raised.value.code == _RESOURCE_NOT_FOUND, raised.value.code
    await adapter.close()


async def test_sessions_on_different_workspaces_get_separate_library_dirs(tmp_path):
    """One ACP process serves many workspaces, so agent-written skills cannot be process-global.

    SkillWriting puts the directory on sys.path and activates local.*, so a
    shared one exposes code the agent wrote for one repo to a session working
    on another.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    first_root = tmp_path / "repo-a"
    second_root = tmp_path / "repo-b"
    first_root.mkdir()
    second_root.mkdir()

    first = await adapter.new_session(str(first_root))
    second = await adapter.new_session(str(second_root))

    first_agent = await _session(adapter, first.session_id)
    second_agent = await _session(adapter, second.session_id)
    assert first_agent.agent.skill_writing._path == first_root / ".nooa" / "skills"
    assert second_agent.agent.skill_writing._path == second_root / ".nooa" / "skills"
    await adapter.close()


async def test_mcp_server_named_like_a_core_tool_cannot_replace_it(tmp_path):
    """A client-supplied server name must not remove the agent's own tools.

    `mcp.shell` resolved to the `shell` attribute and silently replaced
    ActivityShellTools, which also broke close() because the replacement has no
    close(). The session must survive with its real shell and a warning.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(name="shell", command="lookup-server", args=[], env=[])

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ):
        created = await adapter.new_session(str(tmp_path), mcp_servers=[server])

    session = await _session(adapter, created.session_id)
    assert type(session.agent.shell).__name__ == "ActivityShellTools"
    assert any("shell" in warning for warning in session.startup_warnings)
    await adapter.close()


async def test_mcp_server_with_a_reserved_name_does_not_kill_the_session(tmp_path):
    """A reserved attr name raised a bare ValueError out of session/new."""
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(name="runtime", command="lookup-server", args=[], env=[])

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ):
        created = await adapter.new_session(str(tmp_path), mcp_servers=[server])

    session = await _session(adapter, created.session_id)
    assert any("runtime" in warning for warning in session.startup_warnings)
    await adapter.close()


async def test_a_skipped_mcp_server_is_reported_to_the_client(tmp_path):
    """The startup warning must reach the user, not just session state.

    A server whose name collides is skipped so the session stays usable, but
    silently skipping it means the user configured a server and gets no tools
    and no explanation.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(name="shell", command="lookup-server", args=[], env=[])

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ):
        await adapter.new_session(str(tmp_path), mcp_servers=[server])

    await asyncio.sleep(0.1)  # let the deferred bootstrap publish
    rendered = "".join(str(update) for update in client.updates)
    assert "shell" in rendered, rendered[:400]
    await adapter.close()


async def test_replay_separates_consecutive_turns_from_one_speaker(tmp_path):
    """Consecutive same-role turns must not run together on reload.

    update_user_message emits a *chunk*, and ACP has no end-of-message marker:
    a boundary is implied by a different update type arriving. Cancelled turns
    recorded the prompt and produced nothing else, so several in a row replayed
    into one bubble — "run sleep(60)run sleep(60)One more time...".
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]

    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)
    session.handle.record_user_message("first")
    session.handle.record_user_message("second")
    await adapter.close_session(created.session_id)

    client.updates.clear()
    await adapter.load_session(str(tmp_path), created.session_id)

    replayed = [
        update.content.text
        for update in client.updates
        if type(update).__name__ == "UserMessageChunk"
    ]
    assert len(replayed) == 2, replayed
    assert "firstsecond" not in "".join(replayed)
    await adapter.close()


async def test_initialize_advertises_the_mcp_transports_it_supports():
    """Advertise HTTP and SSE, because the adapter connects them.

    McpCapabilities defaults to all-false. A client that honours the handshake
    filters its HTTP/SSE servers out of session/new, so the agent sees no MCP
    servers at all however they are configured — observed against Zed, which
    had authenticated servers and forwarded none.
    """
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    initialized = await adapter.initialize(PROTOCOL_VERSION)

    mcp = initialized.agent_capabilities.mcp_capabilities
    assert mcp is not None, "no MCP capabilities advertised at all"
    assert mcp.http is True
    assert mcp.sse is True


class _YieldingClient:
    """A client whose session_update yields, as a real transport does."""

    def __init__(self) -> None:
        self.updates: list[object] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        await asyncio.sleep(0)
        self.updates.append(update)


async def test_slash_command_still_reports_generation_limits(tmp_path):
    """GenerationError must keep its stop reason on the slash path.

    The catch-all that stops a raising command from killing the RPC also
    catches GenerationError, which subclasses Exception — so a command hitting
    the token ceiling was reported as an ordinary command failure with a normal
    end of turn, losing max_tokens.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)

    boom = GenerationError(OUTPUT_TOKENS_EXHAUSTED_MESSAGE)
    with (
        # The session has no workspace commands, so force the slash branch.
        patch.object(CodingACPAdapter, "_slash_invocation", return_value=("anything", "now")),
        patch.object(session.dispatcher, "invoke_slash", side_effect=boom),
    ):
        response = await adapter.prompt(created.session_id, [text_block("/anything now")])

    assert response.stop_reason == "max_tokens"
    await adapter.close()


async def test_a_turn_that_ends_early_closes_its_open_tool_cards(tmp_path):
    """An open card must not survive the turn that opened it.

    fail_open_tools was reachable only from cancel and session close, so a turn
    ending on a generation limit left its card in_progress for the session — and
    a later cancel then retitled that stale card "Cancelled".
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)

    async def open_a_card_then_fail(_text):
        session.agent.event_manager.add(
            ToolCallEvent(tool_call_id="stale", name="execute_python", arguments={"code": "x"})
        )
        raise GenerationError(OUTPUT_TOKENS_EXHAUSTED_MESSAGE)

    with patch.object(session.dispatcher, "submit", side_effect=open_a_card_then_fail):
        await adapter.prompt(created.session_id, [text_block("do the work")])

    assert session.bridge._open_tools == set(), session.bridge._open_tools
    await adapter.close()


async def test_bootstrap_updates_do_not_interleave_into_a_replay(tmp_path):
    """Restored history must arrive as one contiguous run.

    Replay writes straight to the client while the bridge pump drains bootstrap
    updates, so every await in the replay loop let a commands update or an MCP
    warning land in the middle of the restored conversation.
    """
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(_RecordingClient())  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)
    for index in range(6):
        session.handle.record_user_message(f"turn-{index}")
    await adapter.close_session(created.session_id)

    client = _YieldingClient()
    adapter.on_connect(client)  # type: ignore[arg-type]
    await adapter.load_session(str(tmp_path), created.session_id)
    await asyncio.sleep(0.1)

    kinds = [type(u).__name__ for u in client.updates]
    replayed = [i for i, k in enumerate(kinds) if k == "UserMessageChunk"]
    assert replayed, kinds
    # The replayed turns must be contiguous — nothing wedged between them.
    assert replayed == list(range(replayed[0], replayed[0] + len(replayed))), kinds
    await adapter.close()


def _remembering_llm() -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                "self.v.remembered = 'kept across a reload'\n"
                "self.message('noted')\n"
                "return_result(Done(explanation='noted'))"
            )
        },
    )


async def _session_that_remembered(tmp_path) -> tuple[CodingACPAdapter, str]:
    """A session whose one finished turn set ``self.v.remembered``; adapter still open."""
    adapter = CodingACPAdapter(_remembering_llm, model_alias="fake-model")
    adapter.on_connect(_RecordingClient())  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    await adapter.prompt(created.session_id, [text_block("remember this")])
    return adapter, created.session_id


async def _reload(tmp_path, session_id: str, client=None) -> tuple[CodingACPAdapter, object]:
    adapter = CodingACPAdapter(_completed_llm, model_alias="fake-model")
    adapter.on_connect(client or _RecordingClient())  # type: ignore[arg-type]
    await adapter.load_session(str(tmp_path), session_id)
    return adapter, await _session(adapter, session_id)


def _count_saves(session) -> list[str]:
    saved: list[str] = []
    session.handle.storage.save_snapshot_json = saved.append
    return saved


async def test_load_session_restores_the_agent_state_from_the_last_turn(tmp_path):
    first, session_id = await _session_that_remembered(tmp_path)
    await first.close()

    second, session = await _reload(tmp_path, session_id)
    assert session.agent.v.remembered == "kept across a reload"
    await second.close()


async def test_load_session_runs_after_restore_once(tmp_path, monkeypatch):
    """The restore itself runs after_restore() (AgentSnapshot.restore); the adapter adds no second call."""
    from nooa_acp.server import CodingAgent

    first, session_id = await _session_that_remembered(tmp_path)
    await first.close()

    calls: list[object] = []

    class CountingAgent(CodingAgent):
        def after_restore(self) -> None:
            calls.append(self)
            super().after_restore()

    monkeypatch.setattr("nooa_acp.server.CodingAgent", CountingAgent)
    second, session = await _reload(tmp_path, session_id)
    assert calls == [session.agent]
    await second.close()


async def test_an_unchanged_state_is_not_saved_again(tmp_path):
    adapter, session_id = await _session_that_remembered(tmp_path)
    session = await _session(adapter, session_id)
    saved = _count_saves(session)
    session.checkpoint()
    assert saved == []
    await adapter.close()


async def test_a_reloaded_session_does_not_save_its_unchanged_state_again(tmp_path):
    first, session_id = await _session_that_remembered(tmp_path)
    await first.close()

    second, session = await _reload(tmp_path, session_id)
    saved = _count_saves(session)
    session.checkpoint()
    assert saved == []
    await second.close()


async def test_a_cancelled_turn_does_not_save_its_state(tmp_path):
    adapter, session_id = await _session_that_remembered(tmp_path)
    session = await _session(adapter, session_id)
    saved = _count_saves(session)

    session.cancel_complete.clear()
    async with session.saving_state():
        session.agent.v.half_done = True
        session.cancel_complete.set()  # what cancel() does
    with pytest.raises(asyncio.CancelledError):
        async with session.saving_state():
            session.cancel_complete.clear()
            session.agent.v.half_done = 2
            raise asyncio.CancelledError
    assert saved == []

    session.cancel_complete.clear()
    async with session.saving_state():
        pass  # a turn that finishes saves what changed
    assert len(saved) == 1
    await adapter.close()


async def test_load_session_with_an_unreadable_snapshot_starts_a_fresh_agent(tmp_path, caplog):
    first, session_id = await _session_that_remembered(tmp_path)
    (await _session(first, session_id)).handle.storage.save_snapshot_json('{"version": 999}')
    await first.close()

    client = _RecordingClient()
    with caplog.at_level("WARNING", logger="nooa_acp.server"):
        second, session = await _reload(tmp_path, session_id, client)
    assert "could not restore its saved state" in caplog.text
    assert "remembered" not in session.agent.v
    replayed = [u.content.text for u in client.updates if isinstance(u, UserMessageChunk)]
    assert replayed == ["remember this\n"]
    await second.prompt(session_id, [text_block("still usable")])
    await second.close()


# D2: actual public-loader/core-SDK/CodingAgent integration, not fake lifecycle proof.
@pytest.fixture
async def d2_native_models(monkeypatch):
    from nooa.unifiedllm import registry
    from tests.runtime.public_llm_fixtures import NativeModels

    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})
    models = NativeModels()
    yield models
    await models.close()


def _d2_native_cell(family, code, call_id):
    import json

    from tests.unifiedllm.test_native_load_llm import paid_payload

    data = paid_payload(family, input_tokens=11, output_tokens=3)
    arguments = json.dumps({"code": code})
    if family == "openai-chat":
        data["choices"][0]["message"].update(
            content=None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": arguments},
                }
            ],
        )
        data["choices"][0]["finish_reason"] = "tool_calls"
    elif family == "openai-responses":
        data["output"] = [
            {
                "id": call_id,
                "type": "function_call",
                "call_id": call_id,
                "name": "execute_python",
                "arguments": arguments,
                "status": "completed",
            }
        ]
    else:
        data["content"] = [
            {
                "type": "tool_use",
                "id": call_id,
                "name": "execute_python",
                "input": {"code": code},
            }
        ]
        data["stop_reason"] = "tool_use"
    return data


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
async def test_d2_real_native_coding_new_prompt_save_close_load_prompt(
    family,
    d2_native_models,
    tmp_path,
):
    import json

    import httpx
    from nooa_cli.coding.agent import CodingAgent

    from nooa.unifiedllm.request import LLMRequest

    seen, made = [], []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["x-api-key" if family == "anthropic-messages" else "authorization"]
        assert body["model"] == "wire-model-not-the-alias"
        return httpx.Response(
            200,
            json=_d2_native_cell(
                family,
                "self.v.native_turns = self.v.get('native_turns', 0) + 1\n"
                "return_result(Done(message='native answer', explanation='native done'))",
                f"native_{len(seen)}",
            ),
        )

    # Required four fields are fixed on the public request, including output_type.
    assert set(LLMRequest.model_fields) == {"messages", "tools", "tool_choice", "output_type"}

    def factory():
        model = d2_native_models.make(family, handler=handler, model="wire-model-not-the-alias")
        made.append(model)
        return model

    alias = "configured-literal-alias"
    adapter = CodingACPAdapter(factory, model_alias=alias)
    adapter.on_connect(_RecordingClient())
    created = await adapter.new_session(str(tmp_path), [])
    first = await _session(adapter, created.session_id)
    assert isinstance(first.agent, CodingAgent)
    assert first.handle.info.model == alias
    assert (
        await adapter.prompt(created.session_id, [text_block("first")])
    ).stop_reason == "end_turn"
    assert first.agent.v.native_turns == 1
    assert first.handle.info.model == alias
    assert list(first.handle.storage.event_backend.all_events())
    await adapter.close_session(created.session_id)
    assert made[0]._closed and first.handle._closed
    assert all(not http.is_closed for http in d2_native_models.http)
    assert all(sdk.max_retries == 9 for sdk in d2_native_models.sdks)
    await adapter.load_session(str(tmp_path), created.session_id, [])
    second = await _session(adapter, created.session_id)
    assert isinstance(second.agent, CodingAgent) and second.agent is not first.agent
    assert second.agent.v.native_turns == 1
    assert (
        await adapter.prompt(created.session_id, [text_block("second")])
    ).stop_reason == "end_turn"
    assert second.agent.v.native_turns == 2
    assert second.handle.info.model == alias
    assert len(seen) == 2 and len(made) == 2
    await adapter.close()
    assert all(model._closed for model in made)
    assert all(not http.is_closed for http in d2_native_models.http)


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
async def test_d2_alias_rejection_precedes_native_factory_or_second_writer(
    family,
    d2_native_models,
    tmp_path,
):
    first = CodingACPAdapter(lambda: d2_native_models.make(family), model_alias="saved-alias")
    first.on_connect(_RecordingClient())
    created = await first.new_session(str(tmp_path), [])
    await first.close()
    calls = []

    def forbidden_factory():
        calls.append(True)
        return d2_native_models.make(family)

    other = CodingACPAdapter(forbidden_factory, model_alias="different-alias")
    other.on_connect(_RecordingClient())
    with pytest.raises(RequestError):
        await other.load_session(str(tmp_path), created.session_id, [])
    assert calls == [] and len(d2_native_models.views) == 1
    await other.close()


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize(
    "seam",
    [
        "binding",
        "partial-agent",
        "dispatcher",
        "bridge",
        "commands",
        "command-callback",
        "bootstrap-registration",
    ],
)
async def test_d2_native_partial_construction_closes_real_resources(
    family,
    seam,
    d2_native_models,
    monkeypatch,
    tmp_path,
):
    import nooa_acp.server as server
    from nooa_cli.coding.agent import CodingAgent

    from nooa.runtime.model_binding import ModelBinding

    sentinel = RuntimeError("native partial " + seam)
    made, handles, sessions = [], [], []
    store_create = server.SessionStore.create

    def create(store, *args, **kwargs):
        handle = store_create(store, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(server.SessionStore, "create", create)

    def factory():
        view = d2_native_models.make(family)
        made.append(view)
        return view

    adapter = CodingACPAdapter(factory, model_alias="native-partial-alias")
    adapter.on_connect(_RecordingClient())
    targets = {
        "binding": (ModelBinding, "__init__"),
        "partial-agent": (CodingAgent, "__init__"),
        "dispatcher": (server.InteractiveSessionDispatcher, "__init__"),
        "bridge": (server.ACPEventBridge, "__init__"),
        "commands": (server.CodingSlashCommandRegistry, "__init__"),
        "command-callback": (server.CodingSlashCommandRegistry, "set_on_change"),
    }
    if seam == "bootstrap-registration":
        original_defer = adapter._defer_bootstrap_updates

        class FailedRegistration(set):
            def add(self, task):
                raise sentinel

        def fail_registration(session):
            sessions.append(session)
            session.notification_tasks = FailedRegistration()
            original_defer(session)

        monkeypatch.setattr(adapter, "_defer_bootstrap_updates", fail_registration)
    else:
        cls, method = targets[seam]
        original = getattr(cls, method)

        def fail_after_actual(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise sentinel

        monkeypatch.setattr(cls, method, fail_after_actual)
    with pytest.raises(RuntimeError) as caught:
        await adapter.new_session(str(tmp_path), [])
    assert caught.value is sentinel
    assert len(made) == 1 and made[0]._closed
    assert all(handle._closed for handle in handles)
    assert await adapter._sessions.ids() == ()
    assert not adapter._construction
    assert all(not client.is_closed for client in d2_native_models.http)
    assert all(sdk.max_retries == 9 for sdk in d2_native_models.sdks)
    assert not d2_native_models.seen
    for session in sessions:
        assert session.notification_owner is None
        assert not session.notification_tasks
    await adapter.close()


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("seam", ["replay", "replacement-factory", "replacement-agent"])
async def test_d2_native_restore_failure_retains_and_closes_actual_owners(
    family,
    seam,
    d2_native_models,
    monkeypatch,
    tmp_path,
):
    import nooa_acp.server as server
    from nooa_cli.coding.agent import CodingAgent

    from nooa.storage.sqlite import SQLiteStorageManager

    views, handles, agents = [], [], []
    sentinel = RuntimeError("native restore " + seam)
    original_agent_init = CodingAgent.__init__
    original_store_open = server.SessionStore.open

    def factory():
        if seam == "replacement-factory" and len(views) == 2:
            raise sentinel
        view = d2_native_models.make(family)
        views.append(view)
        return view

    def agent_init(agent, *args, **kwargs):
        agents.append(agent)
        original_agent_init(agent, *args, **kwargs)
        if seam == "replacement-agent" and len(agents) == 3:
            raise sentinel

    def store_open(store, *args, **kwargs):
        handle = original_store_open(store, *args, **kwargs)
        handles.append(handle)
        return handle

    adapter = CodingACPAdapter(factory, model_alias="native-restore-alias")
    adapter.on_connect(_RecordingClient())
    monkeypatch.setattr(CodingAgent, "__init__", agent_init)
    created = await adapter.new_session(str(tmp_path), [])
    await adapter.close_session(created.session_id)
    monkeypatch.setattr(server.SessionStore, "open", store_open)

    def broken_restore(storage, agent):
        raise RuntimeError("actual snapshot restore boundary")

    monkeypatch.setattr(SQLiteStorageManager, "restore_latest_snapshot", broken_restore)
    if seam == "replay":

        async def broken_replay(session):
            raise sentinel

        monkeypatch.setattr(adapter, "_replay_session", broken_replay)
    with pytest.raises(RuntimeError) as caught:
        await adapter.load_session(str(tmp_path), created.session_id, [])
    assert caught.value is sentinel
    assert len(views) == (2 if seam == "replacement-factory" else 3)
    assert all(view._closed for view in views)
    assert all(handle._closed for handle in handles)
    assert await adapter._sessions.ids() == ()
    assert not adapter._construction
    assert all(not http.is_closed for http in d2_native_models.http)
    assert all(sdk.max_retries == 9 for sdk in d2_native_models.sdks)
    await adapter.close()


import os as _d2_control_os  # noqa: E402 (additive integration test section)


class _D2ACPControl(BaseException):
    pass


@pytest.mark.skipif(
    _d2_control_os.environ.get("NOOA_D2_ISOLATED_CONTROLS") != "1",
    reason="interpreter controls run only in the private subprocess driver",
)
@pytest.mark.parametrize("control_type", [SystemExit, KeyboardInterrupt, _D2ACPControl])
@pytest.mark.parametrize("stage", ["bridge", "commands", "agent", "model", "record", "bootstrap"])
async def test_d2_isolated_native_acp_complete_data_chain(
    control_type,
    stage,
    d2_native_models,
    monkeypatch,
    tmp_path,
    close_every_adapter,
):
    import asyncio
    import inspect

    from nooa_acp._runtime import _acp_capture, _acp_task, _ACPOutcome

    view = d2_native_models.make("openai-chat")
    adapter = CodingACPAdapter(lambda: view, model_alias="native-control-alias")
    adapter.on_connect(_RecordingClient())
    created = await adapter.new_session(str(tmp_path), [])
    runtime = await adapter._sessions.get(created.session_id)
    session = runtime.value
    control = control_type("actual cleanup control " + stage)
    calls = []
    if stage == "bootstrap":
        await asyncio.sleep(0)

        async def failed_notification():
            raise control

        worker = _acp_task(failed_notification(), name="d2-held-control-bootstrap")
        session.notification_owner = worker
        session.notification_tasks.add(worker)
        await asyncio.sleep(0)
    else:
        resource, method = {
            "bridge": (session.bridge, "close"),
            "commands": (session.commands, "close"),
            "agent": (session.agent, "_close_coding_resources"),
            "model": (view, "aclose"),
            "record": (session.handle, "close"),
        }[stage]
        # Inject at the real CodingAgent cleanup seam, not a new agent method.
        target = resource
        actual = getattr(target, method)

        def failed_after_actual(*args, **kwargs):
            monkeypatch.setattr(target, method, actual)
            calls.append(resource)
            result = actual(*args, **kwargs)
            if inspect.isawaitable(result):

                async def complete():
                    await result
                    raise control

                return complete()
            raise control

        monkeypatch.setattr(target, method, failed_after_actual)

    def contains(error):
        return error is control or (
            isinstance(error, BaseExceptionGroup)
            and any(contains(child) for child in error.exceptions)
        )

    outer = asyncio.create_task(_acp_capture(adapter.close()))
    outcome = await outer
    assert outer.exception() is None and isinstance(outcome, _ACPOutcome)
    assert contains(outcome.failure)
    assert await adapter._sessions.ids() == ()
    assert view._closed and session.handle._closed
    assert not d2_native_models.http[0].is_closed
    assert d2_native_models.sdks[0].max_retries == 9
    for task in (
        session.close_task,
        session.construction.close_task,
        runtime._close_task,
        adapter._sessions._close_task,
    ):
        assert task.done() and not task.cancelled() and task.exception() is None
        assert isinstance(task.result(), _ACPOutcome)
    coding_task = session.agent._coding_close_task
    assert coding_task.done() and coding_task.exception() is None
    if stage == "agent":
        assert coding_task.result() is control
    for _ in range(2):
        again = await _acp_capture(adapter.close())
        assert contains(again.failure)
    assert len(calls) == (0 if stage == "bootstrap" else 1)
    if stage == "bootstrap":
        assert worker.done() and worker.exception() is None
        assert worker.result().failure is control
    # All real resources have closed; avoid the legacy autouse fixture's
    # Exception-only catch re-raising a deliberately retained control.
    close_every_adapter.remove(adapter)


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("stage", ["retryable-cleanup", "terminal-delegated-release"])
async def test_d2_native_retirement_prunes_only_settled_obligations(
    family, stage, d2_native_models, monkeypatch, tmp_path, close_every_adapter
):
    import httpx
    from nooa_acp._runtime import _acp_capture

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=_d2_native_cell(
                family,
                "self.v.settled_native = True\n"
                "return_result(Done(message='paid native turn', explanation='settled'))",
                "native_retirement",
            ),
        )

    view = d2_native_models.make(family, handler=handler)
    adapter = CodingACPAdapter(lambda: view, model_alias="native-retirement-alias")
    adapter.on_connect(_RecordingClient())
    created = await adapter.new_session(str(tmp_path), [])
    runtime = await adapter._sessions.get(created.session_id)
    session = runtime.value
    assert (
        await adapter.prompt(created.session_id, [text_block("settle")])
    ).stop_reason == "end_turn"
    assert session.agent.v.settled_native and len(requests) == 1
    saved_events = list(session.handle.storage.event_backend.all_events())
    assert saved_events
    sentinel = RuntimeError("actual native retirement " + stage)
    calls = []
    if stage == "retryable-cleanup":
        actual = session.bridge.close

        async def close_bridge():
            calls.append("bridge")
            if len(calls) == 1:
                raise sentinel
            await actual()

        monkeypatch.setattr(session.bridge, "close", close_bridge)
    else:
        actual = session.handle.close

        def close_handle():
            calls.append("delegate")
            actual()
            raise sentinel

        monkeypatch.setattr(session.handle, "close", close_handle)
    close_every_adapter.remove(adapter)
    first = await _acp_capture(adapter.close_session(created.session_id))
    assert first.failure is sentinel
    assert await adapter._sessions.ids() == ()
    assert await adapter._sessions.owns(created.session_id)
    assert view._closed and session.handle._closed
    assert all(not client.is_closed for client in d2_native_models.http)
    assert all(sdk.max_retries == 9 for sdk in d2_native_models.sdks)
    second = await _acp_capture(adapter.close_session(created.session_id))
    assert second.failure is sentinel
    assert session.construction.failures[0].failure is sentinel
    if stage == "retryable-cleanup":
        assert calls == ["bridge", "bridge"]
        assert not session.construction.retryable
        assert session.construction.release_completed
        assert not await adapter._sessions.owns(created.session_id)
        assert created.session_id not in adapter._sessions._retired
        # A completed stage is not replayed merely because first-error DATA persists.
        await adapter.close()
        assert calls == ["bridge", "bridge"]
    else:
        assert calls == ["delegate"]
        assert session.construction.release_started
        assert not session.construction.release_completed
        assert session.construction.release_failure.failure is sentinel
        assert await adapter._sessions.owns(created.session_id)
        assert adapter._sessions._retired[created.session_id] is runtime
        # Real delegated close ran once, but its failure remains unresolved:
        # a repeat/no-op cannot establish settlement or remove the writer fence.
        third = await _acp_capture(adapter.close_session(created.session_id))
        assert third.failure is sentinel and calls == ["delegate"]
        with pytest.raises(RequestError):
            await adapter.load_session(str(tmp_path), created.session_id, [])


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("seam", ["store", "mcp", "skills", "publication"])
async def test_d2_native_partial_real_store_mcp_skills_publication(
    family, seam, d2_native_models, monkeypatch, tmp_path, close_every_adapter
):
    import sys
    from contextlib import asynccontextmanager
    from pathlib import Path

    import nooa_acp.server as server
    from nooa_cli.sessions.events import SessionStarted

    from nooa.mcp.client import MCPStdioClient
    from nooa.runtime.event_manager import EventManager
    from nooa.skill_registry import SkillRegistry
    from nooa.storage.sqlite import SQLiteStorageManager

    sentinel = RuntimeError("actual native host " + seam)
    views, handles, storage_closes, mcp_exits, tools = [], [], [], [], []
    actual_create = server.SessionStore.create
    actual_storage_close = SQLiteStorageManager.close
    actual_connect = MCPStdioClient.connect_to_server

    def factory():
        view = d2_native_models.make(family)
        views.append(view)
        return view

    def create(store, *args, **kwargs):
        handle = actual_create(store, *args, **kwargs)
        handles.append(handle)
        return handle

    def storage_close(storage):
        storage_closes.append(storage)
        return actual_storage_close(storage)

    @asynccontextmanager
    async def connect(client):
        try:
            async with actual_connect(client) as session:
                yield session
        finally:
            mcp_exits.append(client)

    monkeypatch.setattr(server.SessionStore, "create", create)
    monkeypatch.setattr(SQLiteStorageManager, "close", storage_close)
    monkeypatch.setattr(MCPStdioClient, "connect_to_server", connect)
    adapter = CodingACPAdapter(factory, model_alias="native-partial-host-alias")
    adapter.on_connect(_RecordingClient())
    servers = []
    if seam == "store":
        actual_add = EventManager.add

        def add(manager, event, *args, **kwargs):
            result = actual_add(manager, event, *args, **kwargs)
            if isinstance(event, SessionStarted):
                raise sentinel
            return result

        monkeypatch.setattr(EventManager, "add", add)
    elif seam == "mcp":
        actual_tools = adapter._create_mcp_tools

        async def create_tools(*args, **kwargs):
            result = await actual_tools(*args, **kwargs)
            tools.extend(result[0].values())
            assert len(tools) == 1
            raise sentinel

        monkeypatch.setattr(adapter, "_create_mcp_tools", create_tools)
        servers = [
            McpServerStdio(
                name="native-real-wiki",
                command=sys.executable,
                args=[
                    str(Path(__file__).resolve().parents[3] / "examples/assets/wiki_mcp_server.py")
                ],
                env=[],
            )
        ]
    elif seam == "skills":
        actual_register = SkillRegistry.register

        def register(registry, name, *args, **kwargs):
            result = actual_register(registry, name, *args, **kwargs)
            if name == "nemo.skill_writing":
                raise sentinel
            return result

        monkeypatch.setattr(SkillRegistry, "register", register)
    else:
        actual_add = adapter._sessions.add

        async def add_runtime(*args, **kwargs):
            await actual_add(*args, **kwargs)
            raise sentinel

        monkeypatch.setattr(adapter._sessions, "add", add_runtime)
    close_every_adapter.remove(adapter)
    with pytest.raises(RuntimeError) as caught:
        await adapter.new_session(str(tmp_path), mcp_servers=servers)
    assert caught.value is sentinel
    assert len(views) == 1 and views[0]._closed
    assert all(handle._closed for handle in handles)
    assert storage_closes and all(storage._closed for storage in storage_closes)
    assert await adapter._sessions.ids() == ()
    assert not adapter._sessions._runtimes and not adapter._sessions._retired
    assert not adapter._construction
    assert not d2_native_models.seen
    assert all(not client.is_closed for client in d2_native_models.http)
    assert all(sdk.max_retries == 9 for sdk in d2_native_models.sdks)
    if seam == "mcp":
        assert len(tools) == len(mcp_exits) == 1
        assert tools[0]._client is mcp_exits[0]
        # NOOA's actual MCP client scopes its subprocess to connect/discovery;
        # MCPTool itself has no persistent process or close/aclose protocol.
        assert not callable(getattr(tools[0], "close", None))
        assert not callable(getattr(tools[0], "aclose", None))
    await adapter.close()


@pytest.mark.skipif(
    _d2_control_os.environ.get("NOOA_D2_ISOLATED_CONTROLS") != "1",
    reason="native wrapper controls run only in the isolated offline subprocess",
)
@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize(
    "control_type", [RuntimeError, asyncio.CancelledError, _D2ACPControl, SystemExit]
)
async def test_d2_native_actual_coding_shell_owner_survives_skill_removal(
    family, control_type, d2_native_models, monkeypatch, tmp_path
):
    from nooa_cli.coding.agent import CodingAgent

    from nooa.interactive import SummarizationConfig
    from nooa.unifiedllm import registry
    from tests.runtime.test_public_llm_factory_calls import generate
    from tests.unifiedllm.test_native_load_llm import zero_delay

    root = d2_native_models.make(family)
    monkeypatch.setitem(
        registry.MODELS,
        "native-shell-owner-alias",
        {
            "adapter": family,
            "model": "fixture-model",
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid/v1",
            "client": d2_native_models.sdks[-1],
            "context_window": 12800,
            "max_tokens": 100,
            "retry_config": zero_delay(),
        },
    )
    agent = CodingAgent(
        llm="native-shell-owner-alias",
        cwd=tmp_path,
        libs_dir=tmp_path / "skills",
        summarization=SummarizationConfig(policy="none"),
    )
    view, binding = agent.llm, agent.llms["main"]
    wrapper = agent.shell
    assert agent._coding_shell_owner is wrapper
    assert view is not root and binding._owned[id(view)] is view
    await generate(agent)
    assert len(d2_native_models.seen) == 1
    control, calls = control_type("actual wrapper close control"), []
    actual = wrapper.close

    async def close_wrapper():
        calls.append(wrapper)
        assert getattr(agent, "shell", None) is None
        await actual()
        raise control

    monkeypatch.setattr(wrapper, "close", close_wrapper)
    from nooa_acp._runtime import _acp_capture

    for _ in range(3):
        outcome = await _acp_capture(agent.close())
        assert outcome.failure is control
        assert agent._coding_cleanup_failures[0] is control
        task = agent._coding_close_task
        assert task.done() and not task.cancelled() and task.exception() is None
        assert task.result() is control
    assert calls == [wrapper, wrapper, wrapper]
    assert agent._coding_cleanup_done == {"agent", "skills", "queues"}
    assert view._closed and binding._references == 0 and not binding._owned
    assert not root._closed and root._lease.leases == 1
    assert all(not http.is_closed for http in d2_native_models.http)
    assert all(sdk.max_retries == 9 for sdk in d2_native_models.sdks)
