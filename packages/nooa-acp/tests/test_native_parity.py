# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native bootstrap and ACP must restore the same agent, skills and state."""

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from acp import RequestError, text_block
from acp.schema import SessionInfoUpdate
from nooa_acp.server import CodingACPAdapter
from nooa_cli.interactive.local_agent import LocalAgentRunner
from nooa_cli.tui import bootstrap as native
from nooa_cli.tui import config as native_config
from nooa_cli.tui import session_manager

from nooa.sessions import SessionStore
from nooa.unifiedllm import FakeLLMClient


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


def parity_llm():
    return FakeLLMClient.with_tool_call(
        "python_cell",
        {
            "code": (
                "self.v.counter = self.vars.get('counter', 0) + 1\n"
                "self.message('counter=' + str(self.v.counter))\n"
                "return_result(RespondReason.DONE, explanation='parity check complete')"
            )
        },
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(root / ".nooa"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: user)
    monkeypatch.setattr(session_manager, "SESSIONS_DIR", root / ".nooa" / "sessions")
    monkeypatch.setattr(native, "_scaffold_settings", lambda *_: None)
    monkeypatch.setattr(native, "_load_llm_registry", lambda *_: None)
    monkeypatch.setattr(native, "_enable_tracing", lambda *_: (False, None))
    monkeypatch.setattr(native_config, "get_llm", lambda *_: parity_llm())
    package = root / "skills" / "fixture" / "src" / "parity_fixture"
    package.mkdir(parents=True)
    (package.parents[1] / "pyproject.toml").write_text(
        '[project]\nname="parity-fixture"\n[project.entry-points."nooa.skills"]\n'
        '"parity.fixture"="parity_fixture:ParitySkill"\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        """from nooa.skill import Skill, slash_command
class ParitySkill(Skill):
    def attach(self, agent):
        super().attach(agent)
        self.unsubscribe = agent.event_manager.on("SessionResumed", lambda event: agent.vars.update(resumed_with_skill=True))
    def detach(self):
        self.unsubscribe()
        super().detach()
    @slash_command("parity-probe", output_to_agent=False)
    def probe(self, args: str) -> str:
        return "fixture:" + args
""",
        encoding="utf-8",
    )
    (root / ".nooa").mkdir()
    (root / ".nooa" / "settings.yaml").write_text(
        "coding:\n  additional_skills_dirs: [skills]\n  active_skills: [parity.fixture]\n",
        encoding="utf-8",
    )
    (root / "AGENTS.md").write_text("Parity workspace instruction.\n", encoding="utf-8")
    return root


async def open_native(root, session_id=None):
    result = await native.bootstrap(
        native_config.Config.load(model="parity-fixture", working_dir=str(root)),
        resume_session_id=session_id,
    )
    native.build_registry(result, MagicMock())
    return result


async def close_native(result):
    try:
        result.session_manager._storage.save_snapshot(result.agent)
        await result.agent.close()
    finally:
        result.session_manager.close()


async def test_both_hosts_request_and_persist_automatic_titles(workspace, monkeypatch):
    def title_llm():
        return FakeLLMClient.with_tool_call(
            "python_cell",
            {
                "code": (
                    "self.rename_session('Shared session housekeeping')\n"
                    "self.message('Title test complete.')\n"
                    "return_result(RespondReason.DONE, explanation='complete')"
                )
            },
        )

    monkeypatch.setattr(native_config, "get_llm", lambda *_: title_llm())
    result = await open_native(workspace)
    runner = LocalAgentRunner(result.agent, emit_text=lambda _: None, agent_id=result.session_id)
    runner.set_user_message_accepted_callback(result.session_manager.record_user)
    adapter = CodingACPAdapter(title_llm)
    client = RecordingClient()
    adapter.on_connect(client)
    try:
        prompt = "Make native and ACP session behavior agree"
        await runner.submit_and_wait(prompt)
        created = await adapter.new_session(str(workspace))
        await adapter.prompt(created.session_id, [text_block(prompt)])
        session = (await adapter._sessions.get(created.session_id)).value
        # Inspect the actual LLM inputs, so a forced rename alone cannot hide
        # a missing housekeeping instruction in either host's first turn.
        housekeeping = []
        for agent in (result.agent, session.agent):
            inputs = json.dumps(agent.llm.last_messages)
            assert "[session-title]" in inputs
            assert "self.rename_session" in inputs
            assert prompt in inputs
            assert "Parity workspace instruction." in inputs
            housekeeping.append(
                re.search(r"\[session-title\].*?</opening_user_message>", inputs).group()
            )
        assert housekeeping[0] == housekeeping[1]
        assert (
            result.session_manager.name
            == session.handle.info.title
            == "Shared session housekeeping"
        )
        assert any(
            isinstance(u, SessionInfoUpdate) and u.title == session.handle.info.title
            for u in client.updates
        )
        await adapter.close_session(created.session_id)
        listed = await adapter.list_sessions(str(workspace))
        assert [(s.session_id, s.title) for s in listed.sessions] == [
            (created.session_id, "Shared session housekeeping")
        ]
    finally:
        await adapter.close()
        await runner.shutdown()
        await close_native(result)


async def test_native_to_acp_to_native_preserves_state_and_skills(workspace):
    first = await open_native(workspace)
    session_id = first.session_id
    first.agent.v.counter = 41
    todo = first.agent.todo.add("Verify shared sessions", checkpoint="seed")
    todo_id = todo.id
    first.session_manager.rename("User chosen title", user_named=True)
    assert first.agent.vars["resumed_with_skill"] is True
    initial_skills = first.agent.skills.activated()
    initial_type = type(first.agent)
    await close_native(first)

    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    try:
        await adapter.load_session(str(workspace), session_id)
        session = (await adapter._sessions.get(session_id)).value
        await asyncio.gather(*tuple(session.notification_tasks))
        assert session.restored is True
        assert type(session.agent) is initial_type
        assert session.agent.skills.activated() == initial_skills
        assert session.agent.vars["resumed_with_skill"] is True
        assert session.agent.v.counter == 41
        assert session.agent.todo.get(todo_id).v.checkpoint == "seed"
        assert session.agent.cwd == workspace
        assert session.agent.shell.session is session.agent.repo.session
        assert session.agent.rename_session("automatic title") == "User chosen title"
        assert (await session.commands.invoke("parity-probe", "loaded")).text == "fixture:loaded"
        response = await adapter.prompt(session_id, [text_block("continue")])
        assert response.stop_reason == "end_turn"
        assert session.agent.v.counter == 42
        session.agent.todo.get(todo_id).v.checkpoint = "acp"
    finally:
        await adapter.close()

    resumed = await open_native(workspace, session_id)
    try:
        assert resumed.restored is True
        assert resumed.agent.v.counter == 42
        assert resumed.agent.todo.get(todo_id).v.checkpoint == "acp"
        assert resumed.agent.skills.activated() == initial_skills
        assert resumed.session_manager.name == "User chosen title"
        turns = SessionStore(workspace / ".nooa" / "sessions").load_turns(session_id)
        assert [turn.content for turn in turns] == ["continue", "counter=42"]
    finally:
        await close_native(resumed)


async def test_acp_to_native_handoff_and_active_writer_exclusion(workspace):
    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    created = await adapter.new_session(str(workspace))
    session_id = created.session_id
    await adapter.prompt(session_id, [text_block("seed")])
    await adapter.close()
    result = await open_native(workspace, session_id)
    contender = CodingACPAdapter(parity_llm)
    contender.on_connect(RecordingClient())
    try:
        assert result.agent.v.counter == 1
        assert "parity.fixture" in result.agent.skills.activated()
        with pytest.raises(RequestError):
            await contender.load_session(str(workspace), session_id)
    finally:
        await contender.close()
        await close_native(result)


async def test_native_admission_and_acp_produce_same_durable_turn(workspace):
    result = await open_native(workspace)
    runtime = LocalAgentRunner(result.agent, emit_text=lambda _: None, agent_id=result.session_id)
    runtime.set_user_message_accepted_callback(result.session_manager.record_user)
    done = asyncio.Event()
    runtime.set_dispatch_hooks(on_after_handle=lambda *_: done.set())
    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    try:
        assert runtime.submit("same prompt")
        await asyncio.wait_for(done.wait(), 5)
        created = await adapter.new_session(str(workspace))
        await adapter.prompt(created.session_id, [text_block("same prompt")])
        store = SessionStore(workspace / ".nooa" / "sessions")

        def project(session_id):
            return [(t.role, t.content) for t in store.load_turns(session_id)]

        assert (
            project(result.session_id)
            == project(created.session_id)
            == [("user", "same prompt"), ("agent", "counter=1")]
        )
    finally:
        await adapter.close()
        await runtime.shutdown()
        await close_native(result)


async def test_markdown_command_prepares_same_llm_input_and_durable_turn(workspace):
    from unittest.mock import AsyncMock

    from acp.schema import AvailableCommandsUpdate
    from nooa_cli.tui.commands import CommandHandler

    skill = workspace / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "---\nname: parity-review\ndescription: Review a file\nargument-hint: [target]\n"
        "---\nReview $ARGUMENTS"
    )
    target = workspace / "notes.md"
    target.write_text("A file both agents can read.")
    expected = f"Review [notes.md](<{target}>)"
    result = await open_native(workspace)
    runtime = LocalAgentRunner(result.agent, emit_text=lambda _: None, agent_id=result.session_id)
    runtime.set_user_message_accepted_callback(result.session_manager.record_user)
    adapter = CodingACPAdapter(parity_llm)
    client = RecordingClient()
    adapter.on_connect(client)
    try:
        handler = CommandHandler(result.agent._command_registry, AsyncMock())
        prepared = await handler.handle('/parity-review "@notes.md"')
        assert prepared.agent_message == expected
        await runtime.submit_and_wait(prepared.agent_message)
        created = await adapter.new_session(str(workspace))
        response = await adapter.prompt(
            created.session_id, [text_block('/parity-review "@notes.md"')]
        )
        assert response.stop_reason == "end_turn"
        session = (await adapter._sessions.get(created.session_id)).value
        assert any(
            isinstance(update, AvailableCommandsUpdate)
            and any(command.name == "parity-review" for command in update.available_commands)
            for update in client.updates
        )
        for agent in (result.agent, session.agent):
            inputs = json.dumps(agent.llm.last_messages)
            assert expected in inputs
            assert "[session-title]" in inputs
        store = SessionStore(workspace / ".nooa" / "sessions")
        for session_id in (result.session_id, created.session_id):
            assert [(t.role, t.content) for t in store.load_turns(session_id)] == [
                ("user", expected),
                ("agent", "counter=1"),
            ]
    finally:
        await adapter.close()
        await runtime.shutdown()
        await close_native(result)


async def test_behavior_controls_match_native_without_generating_turns(workspace):
    from acp.schema import AgentMessageChunk
    from nooa_cli.interactive.options import SessionOptions

    result = await open_native(workspace)
    adapter = CodingACPAdapter(parity_llm)
    client = RecordingClient()
    adapter.on_connect(client)
    created = await adapter.new_session(str(workspace))
    session = (await adapter._sessions.get(created.session_id)).value
    try:
        for name, args in [
            ("memory", ["local"]),
            ("reflection", ["on"]),
            ("reflection", ["off"]),
        ]:
            native_result = await result.agent._command_registry.get_command(name).execute(args)
            assert native_result.success
            client.updates.clear()
            reply = await adapter.prompt(
                created.session_id, [text_block("/" + name + " " + " ".join(args))]
            )
            assert reply.stop_reason == "end_turn"
            text = "".join(
                u.content.text for u in client.updates if isinstance(u, AgentMessageChunk)
            )
            assert text == "\n".join(o.content for o in native_result.outputs)

        for agent, sid in ((result.agent, result.session_id), (session.agent, created.session_id)):
            assert agent.memory._mgr.store.path.endswith(f"{sid}-memory.db")
            assert "counter" not in agent.vars
            assert SessionStore(workspace / ".nooa" / "sessions").load_turns(sid) == []
        saved = SessionOptions.load(workspace)
        assert saved.memory_agents["nooa_cli.tui.agent:TUIAgent"] == "session"

        # Controls remain in the catalog after memory replaces its skill instance.
        await adapter.prompt(created.session_id, [text_block("/memory off")])
        assert not hasattr(session.agent, "memory")
        assert {"skills", "memory", "reflection"} <= {c.name for c in session.commands.commands()}
    finally:
        await adapter.close()
        await close_native(result)


async def test_acp_skill_selection_persists_after_agent_loaded_it(workspace, external):
    from nooa_cli.interactive.options import SessionOptions

    adapter = CodingACPAdapter(parity_llm)
    client = RecordingClient()
    adapter.on_connect(client)
    native_result = None
    try:
        first = await adapter.new_session(str(workspace))
        session = (await adapter._sessions.get(first.session_id)).value
        # The user's initial path: the model loads/activates a package locally.
        session.agent.skills.discover_skills_dirs([external])
        session.agent.skills.activate(["parity.extra"])
        assert "parity.extra" in session.agent.skills.activated()
        assert "parity.extra" not in SessionOptions.load(workspace).active_skills

        await adapter.prompt(first.session_id, [text_block(f'/skills add "{external}"')])
        await adapter.prompt(first.session_id, [text_block("/skills activate parity.extra")])
        saved = SessionOptions.load(workspace)
        assert external in saved.skills_dirs, [getattr(u, "content", None) for u in client.updates]
        assert "parity.extra" in saved.active_skills
        await adapter.close_session(first.session_id)

        fresh = await adapter.new_session(str(workspace))
        fresh_session = (await adapter._sessions.get(fresh.session_id)).value
        native_result = await open_native(workspace)
        for agent in (fresh_session.agent, native_result.agent):
            assert "parity.extra" in agent.skills.activated()
        assert (
            await fresh_session.commands.invoke("extra-probe", '"two words"')
        ).text == 'extra:"two words"'
        assert native_result.agent._command_registry.get_user_skill("extra-probe") is not None
    finally:
        await adapter.close()
        if native_result is not None:
            await close_native(native_result)


@pytest.mark.parametrize("legacy", [False, True])
async def test_both_coding_agents_omit_web_publisher(workspace, monkeypatch, legacy):
    from nooa_cli.coding.factory import create_session_agent
    from nooa_cli.interactive.options import SessionOptions

    monkeypatch.setenv("NEMO_OO_RICH_URL", "http://localhost:9999")
    agent = create_session_agent(
        llm=parity_llm(),
        storage=None,
        options=SessionOptions.load(workspace, legacy_agent=legacy),
    )
    try:
        assert not hasattr(agent, "web")
        assert "web" not in agent.context
    finally:
        await agent.close()


@pytest.fixture
def external(tmp_path):
    external = tmp_path / "extra skills"
    package = external / "extra" / "src" / "parity_extra"
    package.mkdir(parents=True)
    (package.parents[1] / "pyproject.toml").write_text(
        '[project]\nname="parity-extra"\n[project.entry-points."nooa.skills"]\n'
        '"parity.extra"="parity_extra:ExtraSkill"\n'
    )
    (package / "__init__.py").write_text(
        "from nooa.skill import Skill, slash_command\n"
        "class ExtraSkill(Skill):\n"
        '    @slash_command("extra-probe", output_to_agent=False)\n'
        "    def probe(self, args: str):\n"
        '        return "extra:" + args\n'
    )
    return external


@pytest.mark.parametrize("host", ["native", "acp"])
async def test_agent_remembers_skills_for_both_clients(workspace, external, host, monkeypatch):
    from nooa_cli.interactive.options import SessionOptions

    def remembering_llm():
        return FakeLLMClient.with_tool_call(
            "python_cell",
            {
                "code": (
                    "self.message(await self.workspace_settings.remember_skill("
                    f"'parity.extra', directory={str(external)!r}))\n"
                    "return_result(RespondReason.DONE, explanation='Remembered skill')"
                )
            },
        )

    source_adapter = CodingACPAdapter(remembering_llm)
    source_adapter.on_connect(RecordingClient())
    reader = CodingACPAdapter(parity_llm)
    reader.on_connect(RecordingClient())
    source_native = fresh_native = runner = None
    try:
        if host == "acp":
            created = await source_adapter.new_session(str(workspace))
            source = (await source_adapter._sessions.get(created.session_id)).value.agent
            await source_adapter.prompt(
                created.session_id, [text_block("Remember the extra skill")]
            )
        else:
            monkeypatch.setattr(native_config, "get_llm", lambda *_: remembering_llm())
            source_native = await open_native(workspace)
            source = source_native.agent
            runner = LocalAgentRunner(
                source, emit_text=lambda _: None, agent_id=source_native.session_id
            )
            done = asyncio.Event()
            runner.set_dispatch_hooks(on_after_handle=lambda *_: done.set())
            assert runner.submit("Remember the extra skill")
            await asyncio.wait_for(done.wait(), 5)
            monkeypatch.setattr(native_config, "get_llm", lambda *_: parity_llm())

        assert "parity.extra" in source.skills.activated()
        assert "self.workspace_settings" in source.skills.status()
        saved = SessionOptions.load(workspace)
        assert "parity.extra" in saved.active_skills
        assert "parity.fixture" in saved.active_skills
        assert external in saved.skills_dirs

        # Fresh agents must discover the source and restore activation, without
        # sharing a snapshot or relying on the original host's in-memory config.
        fresh_native = await open_native(workspace)
        created = await reader.new_session(str(workspace))
        fresh_acp = (await reader._sessions.get(created.session_id)).value.agent
        for agent in (fresh_native.agent, fresh_acp):
            assert "parity.extra" in agent.skills.activated()
            assert "nooa.workspace_settings" in agent.skills.activated()

        # Interleave a native slash command with the agent operation. Its
        # configuration predates these writes and must retain other choices.
        await source.workspace_settings.remember_skill("nemo.methodwriting")
        result = await fresh_native.agent._command_registry.get_command("skills").execute(
            ["activate", "nemo.libwriting"]
        )
        assert result.success
        assert {"nemo.methodwriting", "nemo.libwriting"} <= set(
            SessionOptions.load(workspace).active_skills
        )

        await fresh_acp.workspace_settings.forget_skill("parity.extra")
        assert "parity.extra" not in fresh_acp.skills.activated()
        assert "parity.extra" in fresh_native.agent.skills.activated()
        saved = SessionOptions.load(workspace)
        assert "parity.extra" not in saved.active_skills
        assert "parity.extra" in saved.inactive_skills
        assert external in saved.skills_dirs
        await close_native(fresh_native)
        fresh_native = await open_native(workspace)
        created = await reader.new_session(str(workspace))
        another_acp = (await reader._sessions.get(created.session_id)).value.agent
        for agent in (fresh_native.agent, another_acp):
            assert "parity.extra" not in agent.skills.activated()
            assert "parity.extra" in agent.skills.discovered()
    finally:
        await source_adapter.close()
        await reader.close()
        if runner is not None:
            await runner.shutdown()
        if source_native is not None:
            await close_native(source_native)
        if fresh_native is not None:
            await close_native(fresh_native)


@pytest.mark.parametrize(
    "operation, action", [("remember_skill", "activation"), ("forget_skill", "deactivation")]
)
async def test_agent_skill_persistence_reports_save_failure(
    workspace, monkeypatch, operation, action
):
    from nooa_cli.interactive import settings

    result = await open_native(workspace)
    before = (workspace / ".nooa" / "settings.yaml").read_text()
    try:

        def fail(*args, **kwargs):
            raise OSError("fixture write failure")

        monkeypatch.setattr(settings, "write_settings_updates", fail)
        with pytest.raises(RuntimeError, match=f"Could not save skill {action}"):
            await getattr(result.agent.workspace_settings, operation)("parity.fixture")
        assert (workspace / ".nooa" / "settings.yaml").read_text() == before
        with pytest.raises(RuntimeError, match="not found"):
            await result.agent.workspace_settings.remember_skill("nonexistent.skill")
    finally:
        await close_native(result)


async def test_persisting_skills_uses_session_workspace(workspace, tmp_path, monkeypatch):
    from nooa_cli.interactive.options import SessionOptions

    other = tmp_path / "other-workspace"
    other.mkdir()
    monkeypatch.chdir(workspace)
    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    before = (workspace / ".nooa" / "settings.yaml").read_text()
    try:
        created = await adapter.new_session(str(other))
        agent = (await adapter._sessions.get(created.session_id)).value.agent
        await agent.workspace_settings.remember_skill("nemo.methodwriting")
        assert SessionOptions.load(other).active_skills == ["nemo.methodwriting"]
        assert (workspace / ".nooa" / "settings.yaml").read_text() == before
    finally:
        await adapter.close()


@pytest.mark.parametrize("host", ["native", "acp"])
async def test_workspace_settings_memory_reflection_and_status(workspace, host):
    adapter = CodingACPAdapter(parity_llm)
    client = RecordingClient()
    adapter.on_connect(client)
    native_result = await open_native(workspace)
    created = await adapter.new_session(str(workspace))
    acp_agent = (await adapter._sessions.get(created.session_id)).value.agent
    agent = native_result.agent if host == "native" else acp_agent
    fresh_native = None
    try:
        settings = agent.workspace_settings
        assert not hasattr(agent, "persisting_skills")
        with pytest.raises(RuntimeError, match="Memory is not attached"):
            await settings.configure_reflection(True)
        await settings.configure_memory("session")
        await settings.configure_reflection(True)
        status = settings.status()
        assert status["saved"]["memory"] == "session"
        assert status["saved"]["reflection"] is True
        assert status["current"]["memory_attached"] is True
        assert status["current"]["reflection_enabled"] is True
        assert status["current"]["memory"] == "session"
        other = acp_agent if host == "native" else native_result.agent
        assert other.workspace_settings.status()["current"]["memory"] == "off"
        if host == "native":
            reply = await native_result.agent._command_registry.get_command("memory").execute([])
            assert "Memory: local" in reply.outputs[0].content
        else:
            from acp.schema import AgentMessageChunk

            client.updates.clear()
            await adapter.prompt(created.session_id, [text_block("/memory")])
            assert "Memory: local" in "".join(
                u.content.text for u in client.updates if isinstance(u, AgentMessageChunk)
            )
        assert agent.memory._mgr.store.path.endswith("-memory.db")
        # A native command constructed before the agent wrote preferences sees
        # the saved state and applies its new choice through the same operation.
        command = native_result.agent._command_registry.get_command("reflection")
        assert (await command.execute(["off"])).success
        assert settings.status()["saved"]["reflection"] is False
        await settings.configure_memory("project")
        assert agent.memory._mgr.store.path.endswith("memory/memory.sqlite")
        fresh_native = await open_native(workspace)
        fresh = await adapter.new_session(str(workspace))
        fresh_acp = (await adapter._sessions.get(fresh.session_id)).value.agent
        for current in (fresh_native.agent, fresh_acp):
            assert current.workspace_settings.status()["saved"]["memory"] == "project"
            assert current.memory._mgr.store.path.endswith("memory/memory.sqlite")
        previous_model = settings.status()["current"]["model"]
        settings.set_default_model("fixture-model-for-future")
        assert settings.status()["saved"]["default_model"] == "fixture-model-for-future"
        assert settings.status()["current"]["model"] == previous_model
        await settings.configure_memory("off")
        assert not hasattr(agent, "memory")
    finally:
        await adapter.close()
        await close_native(native_result)
        if fresh_native is not None:
            await close_native(fresh_native)


async def test_workspace_settings_remembers_mcp_without_connecting_or_approving(
    workspace, monkeypatch
):
    from nooa_cli.interactive.options import SessionOptions

    result = await open_native(workspace)
    registry = result.agent.mcp
    registry.register(
        "fixture", url="https://example.test/mcp", headers={"Authorization": "Bearer ${TEST_TOKEN}"}
    )
    monkeypatch.setattr(registry, "_approve", lambda *_: pytest.fail("Must not approve"))
    monkeypatch.setattr(registry, "connect", lambda *_: pytest.fail("Must not connect"))
    try:
        settings = result.agent.workspace_settings
        settings.remember_mcp("fixture")
        saved = SessionOptions.load(workspace)
        assert saved.mcp_auto_connect == ["fixture"]
        assert saved.mcp_servers["fixture"]["headers"] == {"Authorization": "Bearer ${TEST_TOKEN}"}
        assert not registry._is_approved("fixture")
        assert "Authorization" not in str(settings.status())
        settings.remember_mcp("fixture", auto_connect=False)
        assert SessionOptions.load(workspace).mcp_auto_connect == []
        settings.forget_mcp("fixture")
        assert "fixture" not in SessionOptions.load(workspace).mcp_servers
        with pytest.raises(ValueError, match="no valid configuration"):
            settings.remember_mcp("not-registered")
    finally:
        await close_native(result)


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


async def test_remembered_mcp_reconnects_in_both_hosts_and_forget_stops_startup(
    workspace, tmp_path
):
    import sys

    script = Path(__file__).parent / "fixtures" / "mcp_probe.py"
    journal = tmp_path / "remembered-mcp.jsonl"
    source = await open_native(workspace)
    fresh_native = None
    adapter = CodingACPAdapter(parity_llm)
    adapter.on_connect(RecordingClient())
    try:
        source.agent.mcp.register(
            "saved_probe", command=sys.executable, args=[str(script), "--journal", str(journal)]
        )
        source.agent.workspace_settings.remember_mcp("saved_probe")
        # Simulate a human having approved this exact harmless fixture. The
        # workspace skill itself must not grant that approval.
        request = source.agent.mcp._approval_request("saved_probe")
        source.agent.mcp._approval_store.approve(request)
        fresh_native = await open_native(workspace)
        await fresh_native.agent._command_registry.auto_connect_mcp()
        created = await adapter.new_session(str(workspace))
        fresh_acp = (await adapter._sessions.get(created.session_id)).value.agent
        for agent in (fresh_native.agent, fresh_acp):
            assert "saved_probe" in agent.mcp.connected()
            assert (
                json.loads(await agent.saved_probe.probe(nonce="remembered"))["nonce"]
                == "remembered"
            )
        fresh_acp.workspace_settings.forget_mcp("saved_probe")
        await close_native(fresh_native)
        fresh_native = await open_native(workspace)
        await fresh_native.agent._command_registry.auto_connect_mcp()
        created = await adapter.new_session(str(workspace))
        next_acp = (await adapter._sessions.get(created.session_id)).value.agent
        for agent in (fresh_native.agent, next_acp):
            assert "saved_probe" not in agent.mcp.connected()
            assert "saved_probe" not in agent.mcp.discovered()
    finally:
        await adapter.close()
        await close_native(source)
        if fresh_native is not None:
            await close_native(fresh_native)
