# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the small public examples and check their rendered contracts."""

import asyncio
import inspect
import textwrap

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.agent.work_shell import Shell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.tools.todo_list import Todo, TodoList

from nooa.agentdoc import doc
from nooa.interactive import AgentVars, Done
from nooa.tools.shell_tools import FileWrite


def test_per_todo_state_is_advertised_and_round_trips_without_a_new_field():
    assert "self.todo.get(id).v" in doc(AgentVars)
    assert "self.todo.<id>.v" not in doc(AgentVars)
    todos = TodoList()
    item = todos.add("check the example")
    todos.get(item.id).v.finding = "durable"
    text = doc(TodoList, concise=True)
    assert "v: PersistentVars" in text
    assert "notes:" not in text
    assert "v" not in Todo.model_fields
    saved = item.model_dump()
    assert "v" not in saved and saved["vars"]["finding"] == "durable"
    assert Todo.model_validate(saved).v.finding == "durable"


def test_file_write_documents_real_string_fields():
    text = doc(FileWrite)
    value = FileWrite("file", "written", "diff", "content")
    for name in ("path", "message", "diff", "new_text"):
        assert f"{name}: str" in text
        assert f"{name}: {name}" not in text
        assert isinstance(getattr(value, name), str)
    assert str(value) == "written\ndiff"


async def test_shell_quickstart_executes_as_written(tmp_path):
    description = inspect.getdoc(Shell)
    snippet = description.split("Persistent bash::", 1)[1].lstrip("\n").split("\n\n", 1)[0]
    code = textwrap.dedent(snippet)
    assert len(code.splitlines()) == 3
    assert "self.jobs" not in code and "timeout=600" in code
    (tmp_path / "test_quick.py").write_text("def test_ok():\n    pass\n")
    llm = CellLLM(
        [
            cell(
                code
                + "\nassert r.returncode == 0, r.stdout\n"
                + "return Done(explanation=r.stdout.strip().splitlines()[-1])"
            )
        ]
    )
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    try:
        result = await agent.handle({"user_messages": ["run the quickstart"]})
        assert "1 passed" in result.explanation
        assert len(llm.calls) == 1
    finally:
        await agent.aclose()


@pytest.mark.parametrize("producer", [False, True])
async def test_mesh_wait_guidance_uses_a_real_registered_producer(tmp_path, producer):
    setup = (
        "self.queue_manager.queue('mesh')\n"
        "self.queue_manager.spawn(asyncio.sleep(0.1, result='feedback'), "
        "channel='mesh', label='mesh inbox pump')\n"
        if producer
        else ""
    )
    finish = "assert notification['mesh'] == ['feedback']\n" if producer else ""
    llm = CellLLM(
        [
            cell(setup + "return Waiting(explanation='feedback', on=['mesh'])"),
            cell(finish + "return Done(explanation='done')"),
        ]
    )
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"), agent_factory=create_session_agent
    )
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                llm=llm,
            )
        )
        assert await asyncio.wait_for(root.prompt("wait for feedback"), 30) == Done(
            explanation="done"
        )
        assert len(llm.calls) == 2
        errors = [
            event.error for event in root._agent.events.query(type="PythonOutput") if event.error
        ]
        if producer:
            assert not errors
        else:
            assert "no running job has that name" in str(llm.calls[1].messages)
        # The generic sentence stays; the mesh example belongs to the mesh skill.
        assert 'on=["mesh"]' not in str(llm.calls[0].messages)
        assert "registered queue producers" in str(llm.calls[0].messages)
    finally:
        await registry.close_all()
