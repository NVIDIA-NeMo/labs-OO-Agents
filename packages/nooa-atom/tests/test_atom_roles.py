# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One turn method (``handle``); the host's ``turn_method`` selects the role.

The role is the policy the turns enforce and the static ``role`` block of the
prompt prefix (``nooa_atom.agent.roles``). Empty instructions add no block.
"""

import asyncio

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell, reply
from nooa_atom.agent import AtomAgent
from nooa_atom.agent.roles import ROLE_INSTRUCTIONS, role_for
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done, NeedInput
from nooa.storage.json_snapshot import snapshot_from_json, snapshot_to_json
from nooa.unifiedllm import FakeLLMClient

TIMEOUT = 30
WORK = (
    "return Done(explanation='done', result=TaskResult(solution_description='s', "
    "evidence='e', how_to_verify='v', report='R'))"
)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def options(workspace, sessions_dir):
    return SessionOptions(workspace=workspace, agent_spec=ATOM_AGENT, sessions_dir=sessions_dir)


def _agent(tmp_path, **kwargs):
    return AtomAgent(
        llm=FakeLLMClient([]), cwd=tmp_path, libs_dir=tmp_path / "libs", skills_dirs=[], **kwargs
    )


def _role_texts(call):
    messages = [str(message.get("content", "")) for message in call.messages]
    return [(index, text) for index, text in enumerate(messages) if "<role>" in text]


@pytest.mark.parametrize("instructions", [None, "", "   ", " \n\t "])
async def test_empty_instructions_add_no_block(tmp_path, instructions):
    agent = _agent(tmp_path, instructions=instructions)
    try:
        assert "role" not in agent.context
    finally:
        await agent.aclose()


async def test_instructions_are_one_static_prefix_block(tmp_path):
    agent = _agent(tmp_path, instructions="  Report to the parent.  ")
    try:
        assert agent.context["role"] == "Report to the parent."
        # A static prefix block: the rendered system message carries it (see
        # test_the_role_block_is_the_same_prefix_on_every_turn).
    finally:
        await agent.aclose()


async def test_a_restore_keeps_the_hosts_instructions_not_the_saved_ones(tmp_path):
    saved = _agent(tmp_path, instructions="OLD ROLE")
    data = snapshot_to_json(saved)
    await saved.aclose()
    for instructions, expected in ((None, None), ("   ", None), ("NEW ROLE", "NEW ROLE")):
        agent = _agent(tmp_path, instructions=instructions)
        try:
            snapshot_from_json(data, agent)
            assert agent.context.get("role") == "OLD ROLE"  # restore brought the saved copy back
            agent.after_restore()
            assert agent.context.get("role") == expected
        finally:
            await agent.aclose()


def test_each_turn_method_selects_a_policy_and_its_instructions():
    assert role_for("handle") == ("interactive", ROLE_INSTRUCTIONS["interactive"])
    assert role_for("handle_batch") == ("unattended", ROLE_INSTRUCTIONS["unattended"])
    assert role_for("handle_orchestrate") == ("orchestrator", ROLE_INSTRUCTIONS["orchestrator"])
    assert role_for("handle_work") == ("child", ROLE_INSTRUCTIONS["child"])
    for text in ROLE_INSTRUCTIONS.values():
        for old in ("handle_batch", "handle_work", "handle_orchestrate", "delegate", "spawn("):
            assert old not in text


async def test_the_role_block_is_the_same_prefix_on_every_turn(options):
    models = AtomModels()
    models.scripts[None] = [reply("one", "first"), reply("two", "second")]
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        await asyncio.wait_for(root.prompt("first"), TIMEOUT)
        await asyncio.wait_for(root.prompt("second"), TIMEOUT)
        first, second = models.llms[None].calls[:2]
        assert _role_texts(first) == _role_texts(second)
        [(index, text)] = _role_texts(first)
        assert index == 0 and first.messages[0]["role"] == "system"
        assert text.count(ROLE_INSTRUCTIONS["interactive"]) == 1
        assert dict(first.messages[0]) == dict(second.messages[0])  # the whole prefix is stable
    finally:
        await registry.close_all()


async def test_an_unattended_turn_may_not_ask(options):
    models = AtomModels()
    models.scripts[None] = [cell("return NeedInput(question='which one?')"), cell(WORK)]
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options.model_copy(update={"turn_method": "handle_batch"}))
        outcome = await asyncio.wait_for(root.prompt("task"), TIMEOUT)
        assert isinstance(outcome, Done) and outcome.result.report == "R"
        feedback = str(models.llms[None].calls[1].messages)
        assert "Nobody can answer questions" in feedback
    finally:
        await registry.close_all()


async def test_an_orchestrator_done_needs_a_result(options):
    models = AtomModels()
    models.scripts[None] = [cell("return Done(explanation='done')"), cell(WORK)]
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(
            options.model_copy(update={"turn_method": "handle_orchestrate"})
        )
        outcome = await asyncio.wait_for(root.prompt("task"), TIMEOUT)
        assert outcome.result.report == "R"
        assert root._agent._turn_policy == "orchestrator"
        assert ROLE_INSTRUCTIONS["orchestrator"] in str(models.llms[None].calls[0].messages)
    finally:
        await registry.close_all()


async def test_an_interactive_turn_may_ask(options):
    models = AtomModels()
    models.scripts[None] = [cell("return NeedInput(question='which one?')")]
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == NeedInput(
            question="which one?"
        )
    finally:
        await registry.close_all()


async def test_a_one_shot_child_may_not_ask_and_must_report(options):
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "job = self.jobs.launch(Subagent('check').run('look'), name='check')\n"
            "r = await job\n"
            "return Done(explanation=r.report)"
        )
    ]
    models.scripts["check"] = [
        cell("return NeedInput(question='which?')"),
        cell("return Done(explanation='no result')"),
        cell(WORK),
    ]
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="R")
        child_calls = models.llms["check"].calls
        last = "\n".join(str(message.get("content", "")) for message in child_calls[-1].messages)
        asked = last.index("can't answer a one-shot worker")
        assert last.index("A worker turn must end with Done(explanation=...") > asked
        first = "\n".join(str(message.get("content", "")) for message in child_calls[0].messages)
        assert ROLE_INSTRUCTIONS["child"] in first
        [child] = registry.children(root.id)
        assert (child.agent, child.turn_method) == (ATOM_AGENT, "handle_work")
    finally:
        await registry.close_all()


async def test_a_saved_unattended_session_resumes_with_its_role(options):
    """The record keeps the host's turn_method, so the resumed agent gets the same role."""
    models = AtomModels()
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options.model_copy(update={"turn_method": "handle_batch"}))
        root._checkpoint()
        await root.wait_for_checkpoint()
        session_id = root.id
    finally:
        await registry.close_all()
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await registry.load(session_id)
        assert loaded.info.turn_method == "handle_batch"
        assert loaded._agent._turn_policy == "unattended"
        assert loaded._agent.context["role"] == ROLE_INSTRUCTIONS["unattended"]
    finally:
        await registry.close_all()


async def test_custom_agents_keep_their_turn_method(options):
    """Only an agent with ``turn_entry`` runs every turn with it; others run turn_method."""
    from atom_test_agents import ScriptedModels

    models = ScriptedModels()
    models.scripts[None] = [cell(WORK)]
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=models)
    try:
        root = await registry.create(
            options.model_copy(
                update={"agent_spec": "atom_test_agents:BatchAgent", "turn_method": "handle_batch"}
            )
        )
        outcome = await asyncio.wait_for(root.prompt("task"), TIMEOUT)
        assert outcome.result.report == "R"
        assert "handle_batch" in str(models.llms[None].calls[0].messages)
    finally:
        await registry.close_all()


def test_the_private_base_keeps_its_turn_methods():
    from atom_test_agents import BaseAtom
    from nooa_atom.session.session import turn_method_of

    assert turn_method_of(BaseAtom, "handle_work") == "handle_work"
    assert ATOM_SPEC == "atom_test_agents:BaseAtom"


async def test_workspace_is_static_and_only_names_a_distinct_search_root(tmp_path):
    agent = _agent(tmp_path)
    try:
        initial = agent.context["workspace_state"]
        assert agent.context_manager.is_static("workspace_state")
        assert initial == (
            f"Workspace: {tmp_path} — files/repo/new Shells.\n"
            "Python paths use cell cwd, not Shell cwd."
        )
        agent.v.changed = True
        assert agent.context["workspace_state"] == initial
        from nooa_atom.tools.repo_tools import RepoTools

        agent.repo = RepoTools(root=tmp_path / "other", session=agent.shell.session)
        agent._install_context()
        assert "Repository root:" in agent.context["workspace_state"]
        assert str(tmp_path / "other") in agent.context["workspace_state"]
    finally:
        await agent.aclose()


async def test_workspace_shell_and_python_paths_remain_independent(tmp_path, monkeypatch):
    from pathlib import Path

    from nooa_atom.agent.work_shell import Shell

    python_dir = tmp_path / "python"
    workspace = tmp_path / "workspace"
    shell_dir = workspace / "sub"
    python_dir.mkdir()
    shell_dir.mkdir(parents=True)
    for directory, value in [
        (python_dir, "python"),
        (workspace, "workspace"),
        (shell_dir, "shell"),
    ]:
        (directory / "marker").write_text(value)
    monkeypatch.chdir(python_dir)
    agent = _agent(workspace)
    try:
        async with Shell(workspace) as shell:
            await shell.run("cd sub")
            assert (await shell.run("cat marker")).stdout == "shell"
            assert Path("marker").read_text() == "python"
            assert (await agent.files.read("marker")).text == "workspace"
        assert "Python paths use cell cwd, not Shell cwd." in agent.context["workspace_state"]
    finally:
        await agent.aclose()
