# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ACP JSON-RPC subprocess test."""

import asyncio
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.connection import StreamDirection
from acp.schema import (
    AgentMessageChunk,
    AvailableCommandsUpdate,
    ContentToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

# Bounds a hang, not the expected duration. Spawning an interpreter and running
# a turn takes well under a second here, but a loaded CI runner is a different
# machine — and a flaky test gets deleted, which is worse than a slow one. A
# real deadlock still fails, just later.
_HANG_TIMEOUT = 30


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.tool_started = asyncio.Event()
        self.commands_updated = asyncio.Event()

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))
        if isinstance(update, ToolCallStart):
            self.tool_started.set()
        if isinstance(update, AvailableCommandsUpdate):
            self.commands_updated.set()


def _write_protocol_skill(workspace: Path) -> None:
    skills_root = workspace / "external-skills"
    package = skills_root / "protocol_skill"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "protocol-skill"\n\n'
        '[project.entry-points."nooa.skills"]\n'
        '"test.protocol" = "protocol_skill:ProtocolSkill"\n'
    )
    (package / "__init__.py").write_text(
        "from nooa.skill import Skill, slash_command\n\n"
        "class ProtocolSkill(Skill):\n"
        "    @slash_command(\n"
        "        'protocol-check', argument_hint='<value>', output_to_agent=False\n"
        "    )\n"
        "    def check(self, args: str) -> str:\n"
        '        """Check ACP command dispatch."""\n'
        "        return f'Check {args}.'\n"
    )
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {skills_root}\n"
    )


async def test_acp_subprocess_transcript(tmp_path, monkeypatch):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    _write_protocol_skill(tmp_path)
    incoming: list[dict[str, object]] = []

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        connection._conn.add_observer(  # type: ignore[attr-defined]
            lambda event: (
                incoming.append(event.message)
                if event.direction is StreamDirection.INCOMING
                else None
            )
        )
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(client.commands_updated.wait(), timeout=5)
        response = await connection.prompt(session.session_id, [text_block("run smoke test")])

    assert initialized.agent_info is not None
    assert initialized.agent_info.name == "nooa-acp"
    assert response.stop_reason == "end_turn"
    assert {update_session for update_session, _ in client.updates} == {session.session_id}
    new_session_response = next(
        index
        for index, message in enumerate(incoming)
        if message.get("id") == 1 and "result" in message
    )
    commands_notification = next(
        index
        for index, message in enumerate(incoming)
        if message.get("method") == "session/update"
        and message.get("params", {}).get("update", {}).get("sessionUpdate")
        == "available_commands_update"
    )
    assert new_session_response < commands_notification
    commands = next(
        update for _, update in client.updates if isinstance(update, AvailableCommandsUpdate)
    )
    assert [command.name for command in commands.available_commands] == ["protocol-check"]
    assert commands.available_commands[0].input is not None
    assert commands.available_commands[0].input.root.hint == "<value>"
    started = next(update for _, update in client.updates if isinstance(update, ToolCallStart))
    assert started.kind == "other"
    assert started.status == "in_progress"
    assert started.raw_input is None
    assert started.content is not None
    source_content = started.content[0]
    assert isinstance(source_content, ContentToolCallContent)
    assert isinstance(source_content.content, TextContentBlock)
    assert source_content.content.text.startswith("```python\n")
    assert "return_result" in source_content.content.text
    assert (
        started.model_dump(mode="json", by_alias=True, exclude_none=True)["content"][0]["content"][
            "type"
        ]
        == "text"
    )

    completed = next(update for _, update in client.updates if isinstance(update, ToolCallProgress))
    assert completed.content is not None
    assert len(completed.content) == 2
    assert completed.title == "Ran Python"
    assert any(
        isinstance(update, AgentMessageChunk)
        and update.content.text == "NOOA ACP smoke test passed."
        for _, update in client.updates
    )


async def test_acp_subprocess_dispatches_advertised_slash_command(tmp_path, monkeypatch):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    _write_protocol_skill(tmp_path)

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        response = await connection.prompt(
            session.session_id,
            [text_block("/protocol-check ready")],
        )

    assert response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "Check ready."
        for _, update in client.updates
    )
    assert not any(isinstance(update, ToolCallStart) for _, update in client.updates)


async def test_acp_subprocess_cancellation_finishes_open_tools(tmp_path):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        "--blocking",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt_task = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("wait forever")])
        )
        await asyncio.wait_for(client.tool_started.wait(), timeout=_HANG_TIMEOUT)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=_HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    started = next(update for _, update in client.updates if isinstance(update, ToolCallStart))
    failed = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress) and update.status == "failed"
    )
    assert failed.tool_call_id == started.tool_call_id


async def test_acp_subprocess_closes_a_session_over_the_wire(tmp_path):
    """session/close must work through the router, not just on the adapter.

    initialize advertises the close capability, and the library registers that
    method as unstable — so an adapter-level test passes while a real client
    gets "method not found" and can never release a session.
    """
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await connection.close_session(session.session_id)

    capabilities = initialized.agent_capabilities.session_capabilities
    assert capabilities is not None and capabilities.close is not None
