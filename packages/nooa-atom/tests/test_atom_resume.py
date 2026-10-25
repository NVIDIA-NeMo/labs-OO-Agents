# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resuming a session an earlier Atom saved: it loads as today's public ``AtomAgent``.

A synthetic earlier Atom session is saved under the standard spec
(``nooa_atom.agent:AtomAgent``), but built from the private base with its
``delegate``/``spawn``/``JobsApi`` surface (``atom_test_agents:BaseAtom``),
the way sessions saved before the switch look on disk. Loading it with the
real agent factory gives the new interface and keeps the history, todos,
``self.v`` and a skill's private vars (a fake mesh identity).
"""

import asyncio

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell, reply
from nooa_atom.agent import AtomAgent
from nooa_atom.agent.base import AtomAgentBase
from nooa_atom.agent.work_manager import JobEvent, WorkManager
from nooa_atom.session.loader import ATOM_AGENT, canonical_agent_spec, load_agent_class
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.storage.snapshot_vars import SnapshotVars

TIMEOUT = 30
FAKE_MESH = {"mesh_handle": "fake-handle", "mesh_token": "fake-token-not-a-secret"}


class EarlierAtom(AtomModels):
    """Builds ``BaseAtom`` (the earlier Atom's surface) whatever the session's spec says."""

    def __call__(self, options, storage):
        return super().__call__(options.model_copy(update={"agent_spec": ATOM_SPEC}), storage)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def options(workspace, sessions_dir):
    return SessionOptions(workspace=workspace, agent_spec=ATOM_AGENT, sessions_dir=sessions_dir)


def _jobs(agent):
    return [item for item in agent._jobs_in.snapshot() if isinstance(item, JobEvent)]


async def _save_earlier_session(sessions_dir, options) -> str:
    """An earlier Atom's session: one turn of history, a todo, vars, a running job."""
    models = EarlierAtom()
    models.scripts[None] = [reply("EARLIER-REPLY", "greeted")]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert type(root._agent).__name__ == "BaseAtom"
        assert await asyncio.wait_for(root.prompt("EARLIER-PROMPT"), TIMEOUT) == Done(
            explanation="greeted"
        )
        agent = root._agent
        todo = agent.todo.add("Keep the parser fix", description="carried over")
        agent.todo.activate(todo)
        agent.v.note = "kept"
        agent._mesh_vars = SnapshotVars(FAKE_MESH)
        agent._jobs_running = {"tests": {"id": "1", "name": "tests", "path": "/root/tests"}}
        agent.context["user_notes"] = "USER-BLOCK"
        root._checkpoint()
        await root.wait_for_checkpoint()
        assert root.info.agent == ATOM_AGENT
        return root.id
    finally:
        await registry.close_all()


async def test_an_earlier_atom_session_resumes_as_the_public_atom_agent(sessions_dir, options):
    session_id = await _save_earlier_session(sessions_dir, options)

    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await registry.load(session_id)
        agent = loaded._agent
        assert loaded.id == session_id and loaded.info.agent == ATOM_AGENT
        assert type(agent) is AtomAgent and isinstance(agent, AtomAgentBase)
        assert isinstance(agent.jobs, WorkManager)
        with pytest.raises(AttributeError, match="Subagent"):
            await agent.delegate("x", "y")

        # The current tools doc and status block, not the earlier Atom's.
        tools = agent.context.get("python_cell_tools")
        text = str(getattr(tools, "content", None) or getattr(tools, "value", None) or tools)
        assert "Subagent" in text and "class JobsApi" not in text
        assert "queues" not in agent.context and "status" in agent.context
        assert "skills" not in agent.context  # the earlier Atom's <skills> block is gone
        assert agent.context["user_notes"] == "USER-BLOCK"

        # History, todos, vars and the fake mesh identity carry over.
        transcript = [(entry.role, entry.content) for entry in loaded.transcript()]
        assert ("user", "EARLIER-PROMPT") in transcript
        assert ("agent", "EARLIER-REPLY") in transcript
        [todo] = agent.todo.list_todos()
        assert (todo.title, agent.todo.active().id) == ("Keep the parser fix", todo.id)
        assert agent.v.note == "kept"
        assert dict(agent._mesh_vars) == FAKE_MESH

        # The earlier Atom's running job is reported lost, once.
        [lost] = _jobs(agent)
        assert (lost.name, lost.kind) == ("tests", "failed") and lost.value.startswith("lost:")
        assert agent._jobs_running == {}
        loaded._checkpoint()
        await loaded.wait_for_checkpoint()
    finally:
        await registry.close_all()

    # A second save and reload keeps the state and reports nothing again.
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        again = await registry.load(session_id)
        agent = again._agent
        assert type(agent) is AtomAgent
        assert [todo.title for todo in agent.todo.list_todos()] == ["Keep the parser fix"]
        assert agent.v.note == "kept" and dict(agent._mesh_vars) == FAKE_MESH
        assert agent.context["user_notes"] == "USER-BLOCK"
        assert "queues" not in agent.context
        assert [event.name for event in _jobs(agent)] == []
    finally:
        await registry.close_all()


async def _restore_events(sessions_dir, session_id) -> list[JobEvent]:
    """Load the session, return the JobEvents it queued, and save it again."""
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await registry.load(session_id)
        agent = loaded._agent
        assert type(agent) is AtomAgent and agent.session is not None
        agent._status()  # the status block once reported the child sessions
        events = _jobs(agent)  # read before a woken turn takes them
        assert [child.name for child in registry.children(session_id)] == ["review"]
        loaded._checkpoint()
        await loaded.wait_for_checkpoint()
        return events
    finally:
        await registry.close_all()


@pytest.mark.parametrize("earlier", [True, False], ids=["earlier_atom", "atom_agent"])
async def test_restores_report_lost_jobs_but_not_child_sessions(sessions_dir, options, earlier):
    """A child session from before a restore is not reported; recorded running jobs are, once."""
    registry = SessionRegistry(
        SessionStore(sessions_dir), agent_factory=EarlierAtom() if earlier else AtomModels()
    )
    try:
        root = await registry.create(options)
        child_options = options.model_copy(update={"name": "review", "kept": True})
        await registry.create(child_options, parent_id=root.id)
        agent = root._agent
        if earlier:  # an earlier Atom's JobsApi record
            agent._jobs_running = {"tests": {"id": "1", "name": "tests", "path": "/root/tests"}}
        else:  # this agent's self.jobs record
            agent._work_records = [{"id": 1, "name": "tests", "kind": "coroutine"}]
        root._checkpoint()
        await root.wait_for_checkpoint()
        session_id = root.id
    finally:
        await registry.close_all()

    [lost] = await _restore_events(sessions_dir, session_id)
    assert (lost.name, lost.kind) == ("tests", "failed") and lost.value.startswith("lost:")
    # Restored again: the child is still saved, and nothing is reported.
    assert await _restore_events(sessions_dir, session_id) == []
    assert await _restore_events(sessions_dir, session_id) == []


async def test_new_atom_sessions_and_their_subagents_use_the_public_atom_agent(
    sessions_dir, options
):
    models = AtomModels()
    work = (
        "return_result(Done(explanation='done', result=TaskResult(solution_description='s', "
        "evidence='e', how_to_verify='v', report='CHILD-REPORT')))"
    )
    models.scripts[None] = [
        cell(
            "job = self.jobs.launch(Subagent('check').run('look around'), name='check')\n"
            "result = await job\n"
            "return_result(Done(explanation=result.report))"
        )
    ]
    models.scripts["check"] = [cell(work)]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert type(root._agent) is AtomAgent
        outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
        assert outcome == Done(explanation="CHILD-REPORT")
        [child] = registry.children(root.id)
        assert child.agent == ATOM_AGENT
        assert [built.agent_spec for built in models.built] == [ATOM_AGENT, ATOM_AGENT]
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "spec",
    [
        "nooa_atom.agent:AtomAgent",
        "nooa_atom.agent.agent:AtomAgent",
        "nooa_atom.agent.unified:UnifiedAtomAgent",  # the spike's spelling
        "nooa_atom.agent:ExperimentalAtomAgent",
        "nooa_cli.coding.agent:CodingAgent",
    ],
)
def test_standard_and_saved_atom_specs_load_the_public_atom_agent(spec):
    assert load_agent_class(spec) is AtomAgent


def test_custom_specs_are_left_alone():
    assert canonical_agent_spec("atom_test_agents:EchoAgent") == "atom_test_agents:EchoAgent"
    assert canonical_agent_spec("./agent.py:AtomAgent") == "./agent.py:AtomAgent"
    assert load_agent_class("atom_test_agents:BaseAtom").__name__ == "BaseAtom"
