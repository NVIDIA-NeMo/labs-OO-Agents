# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The coding agent driven through a Session: turns, results, cleanup and state."""

import asyncio
import logging
import re

import pytest
from coder_test_agents import CODER_SPEC, CoderModels, TrackedLLM, cell, reply
from nooa_coder.session.items import TaskResult
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

from nooa.agentdoc import doc
from nooa.interactive import Done, NeedInput
from nooa.storage.json_snapshot import snapshot_to_json

TIMEOUT = 30


@pytest.fixture
def coder_models():
    return CoderModels()


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
async def coder_registry(sessions_dir, coder_models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=coder_models)
    yield registry
    await registry.close_all()


@pytest.fixture
def coder_options(workspace, sessions_dir):
    return SessionOptions(workspace=workspace, agent_spec=CODER_SPEC, sessions_dir=sessions_dir)


async def test_a_greeting_turn_ends_done_with_a_message(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [reply("Hello! What should we work on?", "greeted the user")]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("hi"), TIMEOUT)
    assert outcome == Done(explanation="greeted the user")
    assert [entry.content for entry in root.transcript() if entry.role == "agent"] == [
        "Hello! What should we work on?"
    ]


async def test_a_question_with_options_ends_the_turn_as_need_input(
    coder_registry, coder_options, coder_models
):
    coder_models.scripts[None] = [
        cell("return_result(NeedInput(question='Which branch?', options=['main', 'dev']))")
    ]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("push it"), TIMEOUT)
    assert outcome == NeedInput(question="Which branch?", options=["main", "dev"])


async def test_a_batch_turn_must_carry_a_task_result(coder_registry, coder_options, coder_models):
    """handle_batch's postcondition sends a Done without a result back to the model."""
    coder_models.scripts["worker"] = [
        cell("return_result(Done(explanation='finished'))"),
        cell(
            "result = TaskResult(solution_description='patched', evidence='tests pass', "
            "how_to_verify='pytest', report='patched the parser')\n"
            "return_result(Done(explanation='finished', result=result))"
        ),
    ]
    options = coder_options.model_copy(update={"name": "worker", "turn_method": "handle_batch"})
    root = await coder_registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("fix the parser"), TIMEOUT)
    assert outcome.result == TaskResult(
        solution_description="patched",
        evidence="tests pass",
        how_to_verify="pytest",
        report="patched the parser",
    )
    second_call = str(coder_models.llms["worker"].calls[1].messages)
    assert "TaskResult" in second_call and "result" in second_call
    assert "return_result validation error" in second_call


async def test_closing_the_session_closes_skills_and_shell_but_not_a_shared_llm(
    sessions_dir, coder_options, workspace
):
    shared = TrackedLLM("shared", [])
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=CoderModels())
    root = await registry.create(coder_options.model_copy(update={"llm": shared}))
    agent = root.agent
    calls: list[str] = []
    real_skills_aclose = agent.skills.aclose
    real_shell_close = agent.shell.close

    async def skills_aclose():
        calls.append("skills")
        await real_skills_aclose()

    async def shell_close():
        calls.append("shell")
        await real_shell_close()

    agent.skills.aclose = skills_aclose
    agent.shell.close = shell_close
    await registry.close_all()
    assert calls == ["skills", "shell"]
    assert shared.closed is False
    assert not hasattr(agent, "close")


async def test_the_model_sees_the_turn_types_but_not_the_port(
    coder_registry, coder_options, coder_models
):
    """Cells can build TaskResult and match ChildResult; self.session is not documented."""
    coder_models.scripts[None] = [
        cell(
            "ok = TaskResult(solution_description='a', evidence='b', how_to_verify='c')\n"
            "names = [ChildResult.__name__, ChildQuestion.__name__, ChildFailed.__name__]\n"
            "return_result(Done(explanation=','.join(names) + ':' + ok.evidence))"
        )
    ]
    root = await coder_registry.create(coder_options)
    outcome = await asyncio.wait_for(root.prompt("check"), TIMEOUT)
    assert outcome == Done(explanation="ChildResult,ChildQuestion,ChildFailed:b")
    for rendered in (doc(type(root.agent)), doc(root.agent)):
        fields = {
            line.split(":", 1)[0].strip()
            for line in rendered.splitlines()
            if re.match(r"^    \w+:", line)
        }
        assert {"shell", "todo", "delegates"} <= fields  # the check sees the fields
        assert "session" not in fields
        assert "session_port_visible" not in fields
        assert "SessionPort" not in rendered


async def test_a_coding_session_round_trips_its_todos_without_skip_warnings(
    coder_registry, coder_options, coder_models, caplog
):
    coder_models.scripts[None] = [
        cell(
            "todo = self.todo.add('Fix the parser')\n"
            "self.todo.comment(todo, 'found the off-by-one')\n"
            "self.v.note = 'kept'\n"
            "return_result(Done(explanation='planned'))"
        )
    ]
    root = await coder_registry.create(coder_options)
    await asyncio.wait_for(root.prompt("plan"), TIMEOUT)
    with caplog.at_level(logging.WARNING):
        snapshot_to_json(root.agent)
    assert "skip" not in caplog.text.lower()
    await root.wait_for_checkpoint()
    session_id = root.id
    await coder_registry.close(session_id)

    loaded = await coder_registry.load(session_id)
    [todo] = loaded.agent.todo.list_todos()
    assert todo.title == "Fix the parser"
    assert [comment.body for comment in todo.comments] == ["found the off-by-one"]
    assert loaded.agent.v.note == "kept"


async def test_the_session_lists_and_runs_the_coders_slash_commands(
    coder_registry, coder_options, workspace
):
    """Session.commands()/invoke_command() reach the coder's CodingSlashCommandRegistry."""
    skills_dir = workspace / ".nooa" / "skills" / "greet"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Say hello\nargument-hint: NAME\n---\n"
        "Say hello to $ARGUMENTS.\n"
    )
    root = await coder_registry.create(coder_options)
    root.agent.slash_commands.add_skills_dir(workspace / ".nooa" / "skills")
    [command] = [c for c in root.commands() if c.name == "greet"]
    assert command.description == "Say hello"
    result = await root.invoke_command("greet", "Ada")
    assert result.text == "Say hello to Ada."
    assert result.output_to_agent is True
