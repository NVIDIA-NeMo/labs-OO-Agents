# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ACP JSON-RPC subprocess test."""

import asyncio
import signal
import sys
from contextlib import suppress
from pathlib import Path

import pytest
from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process, text_block
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
        self.message_updated = asyncio.Event()

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))
        if isinstance(update, ToolCallStart):
            self.tool_started.set()
        if isinstance(update, AvailableCommandsUpdate):
            self.commands_updated.set()
        if isinstance(update, AgentMessageChunk):
            self.message_updated.set()


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
        response = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("run smoke test")]),
            timeout=_HANG_TIMEOUT,
        )

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
    assert [command.name for command in commands.available_commands] == [
        "mcp-add",
        "memory",
        "protocol-check",
        "reflection",
        "skills",
    ]
    protocol_command = next(
        command for command in commands.available_commands if command.name == "protocol-check"
    )
    assert protocol_command.input is not None
    assert protocol_command.input.root.hint == "<value>"
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
        response = await asyncio.wait_for(
            connection.prompt(
                session.session_id,
                [text_block("/protocol-check ready")],
            ),
            timeout=_HANG_TIMEOUT,
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

        # Routable is only half of it: a no-op handler leaks the runtime for the
        # process lifetime. Prompting a closed session must now be rejected.
        with pytest.raises(Exception, match="(?i)not found|no such|unknown"):
            await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block("still there?")]),
                timeout=_HANG_TIMEOUT,
            )

    capabilities = initialized.agent_capabilities.session_capabilities
    assert capabilities is not None and capabilities.close is not None


async def test_acp_lists_native_sessions_with_absolute_workspaces(tmp_path, monkeypatch):
    """One legacy relative cwd must not invalidate a client's whole resume list."""
    from nooa_cli.tui import session_manager

    from nooa.sessions import SessionStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    store = SessionStore(workspace / ".nooa" / "sessions")
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)
    monkeypatch.setattr(session_manager, "SESSIONS_DIR", store.root)
    monkeypatch.chdir(workspace)
    native = session_manager.SessionManager.create(working_dir=".")
    native_id = native.session_id
    assert native.working_dir == str(workspace)
    native.record_user("A native conversation")
    native.close()
    expected = {native_id: str(workspace)}
    # These are persisted legacy values, deliberately bypassing normalization
    # at native creation. Do not rewrite existing user databases to repair them.
    for index, cwd in enumerate((".", "../workspace", "", str(server_cwd))):
        with store.create(session_id=f"old-{index}", host="tui", working_directory=cwd) as old:
            old.record_user_message("A legacy native conversation")
            expected[old.id] = str(server_cwd) if cwd == str(server_cwd) else str(workspace)

    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(client, sys.executable, str(fixture), cwd=server_cwd) as (
        connection,
        _process,
    ):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        capabilities = initialized.agent_capabilities.session_capabilities
        assert capabilities is not None and capabilities.list is not None
        listed = await connection.list_sessions(cwd=str(workspace))
        assert {session.session_id: session.cwd for session in listed.sessions} == expected
        assert all(Path(session.cwd).is_absolute() for session in listed.sessions)


async def test_resume_hides_open_sessions_and_explains_a_stale_selection(tmp_path):
    """The picker and error message must work across the actual process boundary."""
    from nooa.sessions import SessionStore

    store = SessionStore(tmp_path / ".nooa" / "sessions")
    with store.create(working_directory=str(tmp_path), host="tui") as native:
        session_id = native.id
        native.record_user_message("Resume this conversation")

    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(client, sys.executable, str(fixture), cwd=tmp_path) as (
        connection,
        _process,
    ):
        await connection.initialize(PROTOCOL_VERSION)
        empty = await connection.new_session(str(tmp_path))
        await connection.close_session(empty.session_id)
        listed = await connection.list_sessions(cwd=str(tmp_path))
        assert [session.session_id for session in listed.sessions] == [session_id]
        assert store.path_for(empty.session_id).exists()

        # Another client opens it after the picker was populated.
        with store.open(session_id):
            assert (await connection.list_sessions(cwd=str(tmp_path))).sessions == []
            with pytest.raises(RequestError, match="already open") as caught:
                await connection.load_session(session_id=session_id, cwd=str(tmp_path))
            # Poolside renders error.message, not the diagnostic error.data.
            assert "Close it in the other client or tab" in str(caught.value)
            assert caught.value.data["sessionId"] == session_id

        # Releasing the owning client makes the existing session resumable.
        listed = await connection.list_sessions(cwd=str(tmp_path))
        assert [session.session_id for session in listed.sessions] == [session_id]
        await connection.load_session(session_id=session_id, cwd=str(tmp_path))
        await connection.close_session(session_id)


@pytest.mark.parametrize("shutdown", ["eof", "sigterm"])
@pytest.mark.parametrize("during_turn", [False, True])
async def test_client_shutdown_releases_sessions_for_resume(tmp_path, shutdown, during_turn):
    """A client need not call session/close before exiting its agent server."""
    from nooa.sessions import SessionStore
    from nooa.storage.sqlite import is_sqlite_database_active

    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    client = _RecordingClient()
    async with spawn_agent_process(
        client,
        sys.executable,
        str(fixture),
        "--blocking" if during_turn else "--idle",
        cwd=tmp_path,
    ) as (
        connection,
        process,
    ):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("Remember this conversation")])
        )
        if during_turn:
            await asyncio.wait_for(client.tool_started.wait(), _HANG_TIMEOUT)
        else:
            await prompt
        if shutdown == "sigterm":
            process.send_signal(signal.SIGTERM)
        else:
            process.stdin.close()
        await asyncio.wait_for(process.wait(), _HANG_TIMEOUT)
        assert process.returncode == 0
        with suppress(Exception):
            await prompt

    store = SessionStore(tmp_path / ".nooa" / "sessions")
    assert not is_sqlite_database_active(store.path_for(session.session_id))
    async with spawn_agent_process(
        _RecordingClient(), sys.executable, str(fixture), cwd=tmp_path
    ) as (
        connection,
        _process,
    ):
        await connection.initialize(PROTOCOL_VERSION)
        listed = await connection.list_sessions(cwd=str(tmp_path))
        assert [item.session_id for item in listed.sessions] == [session.session_id]
        assert listed.sessions[0].title == f"Untitled session [{session.session_id[:8]}]"
        assert store.list()[0].title is None  # A display fallback, not a persisted rename.
        await connection.load_session(cwd=str(tmp_path), session_id=session.session_id)
        await connection.close_session(session.session_id)


async def test_cancelling_a_turn_says_so_in_the_conversation(tmp_path):
    """A cancelled turn must leave a visible trace, not just stop.

    stop_reason=cancelled and the tool card carry the outcome, but a collapsed
    card shows the user nothing at all — the turn simply goes quiet.
    """
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
    messages = [
        update.content.text for _, update in client.updates if isinstance(update, AgentMessageChunk)
    ]
    assert any("stopped" in text.lower() for text in messages), messages


async def test_cancelling_a_shell_command_reports_it_as_cancellation(tmp_path):
    """Cancellation during a real shell command, which --blocking never reaches."""
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        "--shell",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt_task = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("run it")])
        )
        await asyncio.wait_for(client.tool_started.wait(), timeout=_HANG_TIMEOUT)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=_HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    rendered = "".join(str(update) for _, update in client.updates)
    assert "Cancelled by user." in rendered
    assert "CancelledError" not in rendered


async def test_acp_subprocess_advertises_and_invokes_markdown_skill(tmp_path, monkeypatch):
    from nooa.sessions import SessionStore

    _write_protocol_skill(tmp_path)
    skill = tmp_path / "external-skills" / "review" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "---\nname: protocol-review\ndescription: Review through ACP\n"
        "argument-hint: [target]\n---\nReview $ARGUMENTS"
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(
        client,
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(client.commands_updated.wait(), timeout=5)
        commands = next(u for _, u in client.updates if isinstance(u, AvailableCommandsUpdate))
        command = next(c for c in commands.available_commands if c.name == "protocol-review")
        assert command.input.root.hint == "[target]"
        response = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block('/protocol-review "two words"')]),
            timeout=_HANG_TIMEOUT,
        )
        assert response.stop_reason == "end_turn"
    turns = SessionStore(tmp_path / ".nooa" / "sessions").load_turns(session.session_id)
    assert [t.content for t in turns if t.role == "user"] == ["Review two words"]


async def test_acp_subprocess_behavior_controls_do_not_call_the_llm(tmp_path, monkeypatch):
    from nooa.sessions import SessionStore

    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"
    async with spawn_agent_process(client, sys.executable, str(fixture), cwd=tmp_path) as (
        connection,
        _,
    ):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        for prompt, expected in [
            ("/memory local", "Memory local (this session only) enabled"),
            ("/memory", "Memory: local (this session only)"),
            ("/reflection on", "Idle reflection enabled"),
            ("/reflection off", "Idle reflection disabled"),
            ("/memory off", "Memory disabled"),
            ("/memory invalid", "Usage: /memory"),
            ("/reflection on", "Memory is not attached"),
            ("/skills list", "Skills"),
            ("/compact", "NOOA /compact is not available through ACP yet"),
        ]:
            client.updates.clear()
            client.message_updated.clear()
            response = await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block(prompt)]), timeout=_HANG_TIMEOUT
            )
            assert response.stop_reason == "end_turn"
            await asyncio.wait_for(client.message_updated.wait(), timeout=5)
            text = "".join(
                u.content.text for _, u in client.updates if isinstance(u, AgentMessageChunk)
            )
            assert expected in text
            assert not any(isinstance(u, ToolCallStart) for _, u in client.updates)
    assert SessionStore(tmp_path / ".nooa" / "sessions").load_turns(session.session_id) == []
