# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native bootstrap and ACP must restore the same agent, skills and state."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from acp import RequestError, text_block
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
