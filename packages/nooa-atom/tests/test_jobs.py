# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``self.jobs`` on the Atom agent: Waiting on a job's label or path, and the model-facing docs."""

import asyncio

import pytest
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.conditions import (
    UNATTENDED_FIX,
    WaitingTargetNotRunning,
    check_waiting_targets,
)
from nooa_atom.agent.jobs import JobsApi, JobUpdate

from nooa.interactive import Waiting
from nooa.unifiedllm import FakeLLMClient

TIMEOUT = 15.0


@pytest.fixture
async def agent(tmp_path):
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    yield agent
    await agent.aclose()


def _check(agent, *on: str) -> None:
    check_waiting_targets(agent, Waiting(explanation="w", on=list(on)), fix=UNATTENDED_FIX)


async def test_the_agent_has_jobs(agent):
    assert isinstance(agent.jobs, JobsApi)
    assert "jobs" in agent.queue_manager.channels()
    assert "self.jobs.run(" in (AtomAgent.__doc__ or "")


async def test_waiting_on_a_running_job_by_label_is_accepted(agent):
    await agent.jobs.run("sleep 30", label="tests")
    _check(agent, "tests")
    _check(agent, "/root/tests")
    _check(agent, "jobs")
    with pytest.raises(WaitingTargetNotRunning, match="'other'"):
        _check(agent, "other")
    await agent.jobs.cancel("tests")
    with pytest.raises(WaitingTargetNotRunning):
        _check(agent, "tests")


async def test_waiting_on_a_finished_job_with_its_result_buffered_is_accepted(agent):
    await agent.jobs.run("echo done", label="quick")
    channel = agent.queue_manager.get_channel("jobs")
    for _ in range(int(TIMEOUT / 0.02)):
        if channel.qsize():
            break
        await asyncio.sleep(0.02)
    assert agent.queue_manager.running_handles() == []
    _check(agent, "quick")
    with pytest.raises(WaitingTargetNotRunning):
        _check(agent, "other")
    item = channel.drain()[0]
    assert (isinstance(item, JobUpdate) and item.done) and item.items[0].stdout == "done\n"
    with pytest.raises(WaitingTargetNotRunning):
        _check(agent, "quick")


async def test_jobs_use_the_agent_shell_directory_and_exports(agent, tmp_path):
    (tmp_path / "pkg").mkdir()
    await agent.shell.run("cd pkg; export ATOM_JOBS_VAR=1")
    await agent.jobs.run('pwd; echo "$ATOM_JOBS_VAR"', label="where")
    item = await asyncio.wait_for(agent.queue_manager.get_channel("jobs").get(), TIMEOUT)
    assert item.items[0].stdout.splitlines() == [str(tmp_path / "pkg"), "1"]


async def test_a_resumed_session_reports_jobs_that_did_not_survive(tmp_path, sessions_dir):
    """Jobs running at the last checkpoint come back as one JobsLost event."""
    from atom_test_agents import ATOM_SPEC, AtomModels
    from nooa_atom.agent.jobs import JobsLost
    from nooa_atom.session.options import SessionOptions
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    root = await registry.create(options)
    await root._agent.jobs.run("sleep 30", label="tests", description="full run")
    root._checkpoint()  # what a settled turn does
    await root.wait_for_checkpoint()
    await registry.close_all()  # the process ends; the job dies with it

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await fresh.load(root.id)
        [event] = loaded._agent.event_manager.filter(type="JobsLost")
        assert isinstance(event, JobsLost)
        assert [(j["label"], j["kind"], j["target"], j["description"]) for j in event.jobs] == [
            ("tests", "run", "sleep 30", "full run")
        ]
        assert loaded._agent._jobs_running == {}
        assert loaded._agent.queue_manager.running_handles() == []
    finally:
        await fresh.close_all()


async def test_jobs_lost_is_reported_when_an_older_snapshot_is_restored(tmp_path, sessions_dir):
    """A damaged latest snapshot falls back to the one before; after_restore still runs."""
    import sqlite3
    from contextlib import closing

    from atom_test_agents import ATOM_SPEC, AtomModels
    from nooa_atom.session.options import SessionOptions
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    root = await registry.create(options)
    await root._agent.jobs.run("sleep 30", label="build")
    root._checkpoint()
    await root.wait_for_checkpoint()
    await registry.close_all()
    [path] = sessions_dir.glob(f"{root.id}*.db")
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "INSERT INTO snapshots (snapshot_id, created_at, data) VALUES (?, ?, ?)",
            ("newest", "2099-01-01T00:00:00+00:00", "damaged"),
        )

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await fresh.load(root.id)
        assert loaded._agent.event_manager.filter(type="SnapshotRestoreFailed") or any(
            "older snapshot" in str(getattr(e, "content", "")) for e in loaded.transcript()
        )
        [event] = loaded._agent.event_manager.filter(type="JobsLost")
        assert [job["label"] for job in event.jobs] == ["build"]
    finally:
        await fresh.close_all()


def _queues_expr(agent) -> str:
    return dict(agent.context_manager._raw_items())["queues"].expr


async def test_the_queues_block_does_not_name_the_queue_manager(tmp_path):
    from nooa.storage.snapshot import AgentSnapshot

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert "queue_manager" not in _queues_expr(agent)
        assert agent._queue_status() == agent.queue_manager.status()
        # An older snapshot still carries the old expression; after_restore puts it back.
        snapshot = AgentSnapshot.from_agent(agent)
        for block in snapshot.context:
            if block.key == "queues":
                block.expr = "self.queue_manager.status()"
        restored = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
        snapshot.restore(restored)
        restored.after_restore()
        assert "queue_manager" not in _queues_expr(restored)
        await restored.aclose()
    finally:
        await agent.aclose()
