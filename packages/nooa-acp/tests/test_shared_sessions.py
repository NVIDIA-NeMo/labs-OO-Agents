# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct shared-API hosts and ACP preserve the same durable agent behavior."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from acp import text_block
from nooa_acp.server import CodingACPAdapter
from nooa_cli.coding.factory import create_session_agent, load_agent_class
from nooa_cli.coding.identity import CODING_AGENT, EXPERIMENTAL_CODING_AGENT
from nooa_cli.coding.slash_commands import CodingSlashCommandRegistry
from nooa_cli.interactive.controls import behavior_commands
from nooa_cli.interactive.dispatcher import InteractiveSessionDispatcher
from nooa_cli.interactive.memory import configure_session_memory
from nooa_cli.interactive.options import (
    SessionOptions,
    configure_session_skills,
    connect_session_mcp,
)
from nooa_cli.interactive.session_paths import session_directory

from nooa.sessions import SessionResumed, SessionStore
from nooa.unifiedllm import AssistantReasoning, CacheBoundary, FakeLLMClient, LLMResponse, ToolCall


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


class RecordingLLM(FakeLLMClient):
    async def acall(self, messages, **kwargs):
        self.request_messages = list(messages)
        return await super().acall(messages, **kwargs)


def response(code, call_id="call_1", *, native=False):
    return LLMResponse(
        parts=(
            AssistantReasoning(
                text="portable reasoning", native={"signature": "fixture-state"} if native else None
            ),
            ToolCall(id=call_id, name="python_cell", arguments=json.dumps({"code": code})),
        ),
        replay_scope="fixture-issuer" if native else None,
        finish_reason="tool_calls",
    )


def parity_llm():
    return FakeLLMClient(
        scripted_responses=[
            response(
                "self.v.counter = self.vars.get('counter', 0) + 1\n"
                "self.rename_session('Shared session test')\n"
                "self.message('counter=' + str(self.v.counter))\n"
                "return_result(RespondReason.DONE, explanation='done')"
            )
        ]
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / ".nooa").mkdir(parents=True)
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(root / ".nooa"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: user)
    (root / "AGENTS.md").write_text("Shared workspace instruction.\n")
    (root / ".nooa/settings.yaml").write_text(
        "coding:\n  active_skills: [nemo.methodwriting]\n  inactive_skills: [nemo.libwriting]\n"
    )
    return root


@asynccontextmanager
async def open_host(workspace, host, llm, session_id=None):
    if host == "acp":
        adapter = CodingACPAdapter(lambda: llm)
        adapter.on_connect(RecordingClient())
        try:
            if session_id is None:
                session_id = (await adapter.new_session(str(workspace))).session_id
            else:
                await adapter.load_session(str(workspace), session_id)
            session = (await adapter._sessions.get(session_id)).value
            await asyncio.gather(*tuple(session.notification_tasks))

            async def submit(text):
                return await adapter.prompt(session_id, [text_block(text)])

            yield SimpleNamespace(
                agent=session.agent, handle=session.handle, commands=session.commands, submit=submit
            )
        finally:
            await adapter.close()
        return

    options = SessionOptions.load(workspace)
    store = SessionStore(session_directory(workspace))
    handle = (
        store.open(session_id)
        if session_id
        else store.create(host="direct", working_directory=str(workspace))
    )
    agent = create_session_agent(llm=llm, storage=handle.storage, options=options)
    dispatcher = None
    commands = None
    try:
        restored = handle.storage.restore_latest_snapshot(agent) if session_id else False
        agent._session_manager = handle
        configure_session_skills(agent, options)

        def memory():
            configure_session_memory(agent, options, agent_db=handle.path, session_id=handle.id)

        memory()
        assert not await connect_session_mcp(agent, options)
        dispatcher = InteractiveSessionDispatcher(agent)
        dispatcher.runtime.set_user_message_accepted_callback(handle.record_user_message)
        commands = CodingSlashCommandRegistry(agent, skills_dirs=options.skills_dirs)
        commands.set_controls(
            behavior_commands(
                agent,
                options,
                configure_memory=memory,
                workspace=workspace,
                command_registry=commands,
            )
        )
        agent.event_manager.add(SessionResumed(session_id=handle.id, restored=restored))
        yield SimpleNamespace(
            agent=agent, handle=handle, commands=commands, submit=dispatcher.submit
        )
    finally:
        try:
            if dispatcher is not None:
                await dispatcher.runtime.cancel_work()
                handle.storage.save_snapshot(agent)
            if commands is not None:
                commands.close()
            if dispatcher is not None:
                await dispatcher.close()
            else:
                await agent.close()
        finally:
            handle.close()


@pytest.mark.parametrize("source,target", [("direct", "acp"), ("acp", "direct")])
async def test_handoff_preserves_agent_state_skills_titles_and_provider_history(
    workspace, source, target
):
    original = response(
        "self.v.counter = 41\nself.rename_session('Shared session test')\n"
        "self.message('seeded')\nreturn_result(RespondReason.DONE, explanation='done')",
        native=True,
    )
    first_llm = RecordingLLM(scripted_responses=[original])
    async with open_host(workspace, source, first_llm) as first:
        session_id = first.handle.id
        await first.submit("seed")
        assert "[session-title]" in json.dumps(first_llm.last_messages)
        assert first.handle.info.title == "Shared session test"
        first.handle.set_title("User chosen title", user_set=True)
        initial_skills = first.agent.skills.activated()
        todo = first.agent.todo.add("Shared task", checkpoint="seed")
        todo_id = todo.id

    llm = RecordingLLM(
        scripted_responses=[
            response(
                "self.v.counter += 1\nself.rename_session('Automatic overwrite')\n"
                "self.message('resumed')\nreturn_result(RespondReason.DONE, explanation='resumed')",
                "next",
            )
        ]
    )
    async with open_host(workspace, target, llm, session_id) as resumed:
        assert type(resumed.agent).__name__ == "ExperimentalCodingAgent"
        assert resumed.agent.vars["counter"] == 41
        assert resumed.agent.todo.get(todo_id).v.checkpoint == "seed"
        assert resumed.agent.skills.activated() == initial_skills
        await resumed.submit("continue")
        assert resumed.agent.vars["counter"] == 42
        assert resumed.handle.info.title == "User chosen title"
        messages = llm.request_messages
        index = next(
            i for i, m in enumerate(messages) if isinstance(m, LLMResponse) and m.id == original.id
        )
        assert messages[index].parts == original.parts
        assert messages[index].replay_scope == original.replay_scope
        result = next(i for i, m in enumerate(messages) if m.get("tool_call_id") == "call_1")
        boundary = next(i for i, m in enumerate(messages) if isinstance(m, CacheBoundary))
        state = [
            i for i, m in enumerate(messages) if "## Python cell state" in str(m.get("content", ""))
        ]
        assert index < result < boundary
        assert state and all(i > boundary for i in state)
        assert "Shared workspace instruction." in json.dumps(llm.last_messages)
    assert [
        t.content for t in SessionStore(session_directory(workspace)).load_turns(session_id)
    ] == ["seed", "seeded", "continue", "resumed"]


@pytest.mark.parametrize("host", ["direct", "acp"])
async def test_agent_preferences_are_workspace_sticky_without_changing_other_live_agents(
    workspace, host
):
    other_host = "direct" if host == "acp" else "acp"
    async with open_host(workspace, host, parity_llm()) as source:
        async with open_host(workspace, other_host, parity_llm()) as other:
            settings = source.agent.workspace_settings
            await settings.remember_skill("nemo.libwriting")
            assert "nemo.libwriting" in source.agent.skills.activated()
            assert "nemo.libwriting" not in other.agent.skills.activated()
            await settings.configure_memory("session")
            await settings.configure_reflection(True)
            assert settings.status()["current"]["reflection_enabled"]
            assert settings.status()["agent_key"] == CODING_AGENT
            assert not hasattr(other.agent, "memory")
            assert source.agent.llm.call_count == 0
            source.agent.mcp.register("saved", command="fixture-command", args=["--fixture"])
            settings.remember_mcp("saved", auto_connect=False)
            assert not source.agent.mcp._is_approved("saved")
            saved = SessionOptions.load(workspace)
            assert saved.memory_agents[CODING_AGENT] == "session"
            assert saved.reflection_agents[CODING_AGENT] is True
            assert "saved" in saved.mcp_servers
        async with open_host(workspace, other_host, parity_llm()) as fresh:
            assert "nemo.libwriting" in fresh.agent.skills.activated()
            assert fresh.agent.workspace_settings.status()["current"]["reflection_enabled"]
            assert fresh.agent.memory._mgr.store.path.endswith(f"{fresh.handle.id}-memory.db")
            await fresh.agent.workspace_settings.forget_skill("nemo.libwriting")
            fresh.agent.workspace_settings.forget_mcp("saved")
        assert "nemo.libwriting" in source.agent.skills.activated()
        assert "saved" not in SessionOptions.load(workspace).mcp_servers
        assert "nemo.libwriting" in SessionOptions.load(workspace).inactive_skills


def test_legacy_settings_are_normalized_at_the_boundary(workspace):
    (workspace / ".nooa/settings.yaml").write_text("""
tui:
  memory_agents:
    'nooa_cli.tui.agent:TUIAgent': session
  reflection_agents:
    'nooa_cli.tui.agent:TUIAgent': true
coding:
  memory_agents:
    'nooa_cli.tui.agent:TUIAgent': session
    'nooa_cli.coding.agent:CodingAgent': project
""")
    options = SessionOptions.load(workspace)
    assert options.memory_agents == {CODING_AGENT: "project"}
    assert options.reflection_agents == {CODING_AGENT: True}
    assert not hasattr(options, "policy_config")
    assert load_agent_class("nooa_cli.tui.agent:TUIAgent") is load_agent_class(CODING_AGENT)
    assert load_agent_class(
        "nooa_cli.tui.experimental_agent:ExperimentalTUIAgent"
    ) is load_agent_class(EXPERIMENTAL_CODING_AGENT)


async def test_client_mcp_probe_is_callable_on_new_and_loaded_acp_sessions(workspace, tmp_path):
    import sys

    from acp.schema import McpServerStdio

    journal = tmp_path / "mcp-journal.jsonl"
    script = Path(__file__).parent / "fixtures" / "mcp_probe.py"
    server = McpServerStdio(
        name="client_probe",
        command=sys.executable,
        args=[str(script), "--journal", str(journal)],
        env=[],
    )
    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    try:
        created = await adapter.new_session(str(workspace), mcp_servers=[server])
        for nonce in ("first", "resumed"):
            if nonce == "resumed":
                await adapter.close_session(created.session_id)
                await adapter.load_session(str(workspace), created.session_id, mcp_servers=[server])
            agent = (await adapter._sessions.get(created.session_id)).value.agent
            assert "mcp.client_probe" in agent.skills.activated()
            response = await agent.client_probe.probe(nonce=nonce)
            payload = json.loads(response)
            assert payload["nonce"] == nonce
            entries = [json.loads(line) for line in journal.read_text().splitlines()]
            assert payload in entries
            assert payload["server_token"]
            # Client-supplied configuration has not become a NOOA default.
            assert "client_probe" not in agent.workspace_settings.status()["saved"]["mcp_servers"]
    finally:
        await adapter.close()


@pytest.mark.parametrize("source_host", ["direct", "acp"])
async def test_remembered_mcp_reconnects_across_hosts_and_forget_stops_startup(
    workspace, tmp_path, source_host
):
    import sys

    script = Path(__file__).parent / "fixtures" / "mcp_probe.py"
    journal = tmp_path / "remembered-mcp.jsonl"
    async with open_host(workspace, source_host, parity_llm()) as source:
        source.agent.mcp.register(
            "saved_probe", command=sys.executable, args=[str(script), "--journal", str(journal)]
        )
        source.agent.workspace_settings.remember_mcp("saved_probe")
        assert not source.agent.mcp._is_approved("saved_probe")
        # Use the actual shared host control, never hand-edit the approval store.
        request = source.agent.mcp._approval_request("saved_probe")
        control = source.commands.get("mcp")
        assert control is not None and control.is_control
        review = await control._method("approve saved_probe")
        assert review.success and request.confirmation in str(review)
        assert not source.agent.mcp._is_approved("saved_probe")
        for wrong in ("wrong", "é"):
            rejected = await control._method(f"approve saved_probe {wrong}")
            assert not rejected.success and "does not match" in str(rejected)
            assert not source.agent.mcp._is_approved("saved_probe")
            assert not journal.exists()
        approved = await control._method(f"approve saved_probe {request.confirmation}")
        assert approved.success, str(approved)
        assert source.agent.mcp._is_approved("saved_probe")
        assert "saved_probe" in source.agent.mcp.connected()
        assert "approved" in str(await control._method("status"))
        for host in ("direct", "acp"):
            async with open_host(workspace, host, parity_llm()) as fresh:
                assert "saved_probe" in fresh.agent.mcp.connected()
                payload = json.loads(await fresh.agent.saved_probe.probe(nonce=host))
                assert payload["nonce"] == host
        revoked = await control._method("revoke saved_probe")
        assert revoked.success, str(revoked)
        assert not source.agent.mcp._is_approved("saved_probe")
        assert "saved_probe" not in source.agent.mcp.connected()
        source.agent.workspace_settings.forget_mcp("saved_probe")
        for host in ("direct", "acp"):
            async with open_host(workspace, host, parity_llm()) as fresh:
                assert "saved_probe" not in fresh.agent.mcp.connected()
                assert "saved_probe" not in fresh.agent.mcp.discovered()


@pytest.mark.parametrize("host", ["direct", "acp"])
async def test_project_settings_cannot_execute_an_agent_module(workspace, host, caplog):
    sentinel = workspace / "unexpected-import"
    (workspace / "injected.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
        "from nooa_cli.coding.experimental_agent import ExperimentalCodingAgent as Cls\n"
    )
    (workspace / ".nooa/settings.yaml").write_text("coding:\n  agent_spec: ./injected.py:Cls\n")
    async with open_host(workspace, host, parity_llm()) as session:
        assert type(session.agent).__name__ == "ExperimentalCodingAgent"
        assert not sentinel.exists()
        assert "Ignoring coding.agent_spec" in caplog.text
    assert SessionOptions.load(workspace, agent_spec="./injected.py:Cls").agent_spec == (
        "./injected.py:Cls"
    )


async def test_legacy_memory_owners_preserve_session_identity(workspace):
    from nooa_memory.schema import Memory, MemoryType
    from nooa_memory.store import MemoryStore

    path = workspace / ".nooa/memory/memory.sqlite"
    path.parent.mkdir(parents=True)
    store = MemoryStore(str(path))
    owners = [
        "TUIAgent",
        "TUIAgent@12345678",
        "TUIAgent@archived",
        "AnotherAgent@12345678",
        "",
        "nooa_cli.tui.agent:TUIAgent",
        "nooa_cli.tui.agent:TUIAgent@archived",
    ]
    records = []
    try:
        for owner in owners:
            records.append(
                store.add(
                    Memory(
                        type=MemoryType.INFO,
                        content=owner or "shared",
                        owner=owner,
                        archived=owner.endswith("archived"),
                    )
                )
            )
    finally:
        store.close()
    (workspace / ".nooa/settings.yaml").write_text("coding:\n  memory: project\n")
    for host in ("direct", "acp"):
        async with open_host(workspace, host, parity_llm()) as session:
            store = session.agent.memory._mgr.store
            assert [store.owner_of(record.id) for record in records] == [
                "CodingAgent",
                "CodingAgent@12345678",
                "CodingAgent@archived",
                "AnotherAgent@12345678",
                "",
                "CodingAgent",
                "CodingAgent@archived",
            ]
            assert len(store.all_memories(include_archived=True)) == len(records)


async def test_resume_warns_when_host_selects_a_different_agent(workspace):
    store = SessionStore(session_directory(workspace))
    with store.create(agent="CodingAgent", working_directory=str(workspace)) as handle:
        session_id = handle.id
        handle.record_user_message("saved conversation")
    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    try:
        await adapter.load_session(str(workspace), session_id)
        session = (await adapter._sessions.get(session_id)).value
        assert any(
            "created with agent 'CodingAgent'" in warning and "ExperimentalCodingAgent" in warning
            for warning in session.startup_warnings
        )
    finally:
        await adapter.close()
