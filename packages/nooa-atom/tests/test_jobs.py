# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``self.jobs`` on the Atom agent: Waiting on ``jobs:<label>`` and the model-facing docs."""

import asyncio

import pytest
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.conditions import (
    UNATTENDED_FIX,
    WaitingTargetNotRunning,
    check_waiting_targets,
)

from nooa.interactive import Waiting
from nooa.runtime.jobs import JobResult, JobsApi
from nooa.unifiedllm import FakeLLMClient

TIMEOUT = 15.0


@pytest.fixture
async def agent(tmp_path):
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    yield agent
    await agent.aclose()


def _check(agent, *on: str) -> None:
    check_waiting_targets(agent, Waiting(explanation="w", on=list(on)), fix=UNATTENDED_FIX)


async def test_the_agent_has_jobs_and_no_producers(agent):
    assert isinstance(agent.jobs, JobsApi)
    assert not hasattr(agent, "producers")
    assert "jobs" in agent.queue_manager.channels()
    assert "self.jobs.run(" in (AtomAgent.__doc__ or "")


async def test_waiting_on_a_running_job_by_label_is_accepted(agent):
    await agent.jobs.run("sleep 30", label="tests")
    _check(agent, "jobs:tests")
    _check(agent, "jobs")
    with pytest.raises(WaitingTargetNotRunning, match="jobs:other"):
        _check(agent, "jobs:other")
    await agent.jobs.cancel("tests")
    with pytest.raises(WaitingTargetNotRunning):
        _check(agent, "jobs:tests")


async def test_waiting_on_a_finished_job_with_its_result_buffered_is_accepted(agent):
    await agent.jobs.run("echo done", label="quick")
    channel = agent.queue_manager.get_channel("jobs")
    for _ in range(int(TIMEOUT / 0.02)):
        if channel.qsize():
            break
        await asyncio.sleep(0.02)
    assert agent.queue_manager.running_handles() == []
    _check(agent, "jobs:quick")
    with pytest.raises(WaitingTargetNotRunning):
        _check(agent, "jobs:other")
    item = channel.drain()[0]
    assert isinstance(item, JobResult) and item.result.stdout == "done\n"
    with pytest.raises(WaitingTargetNotRunning):
        _check(agent, "jobs:quick")


async def test_jobs_use_the_agent_shell_directory_and_exports(agent, tmp_path):
    (tmp_path / "pkg").mkdir()
    await agent.shell.run("cd pkg; export ATOM_JOBS_VAR=1")
    await agent.jobs.run('pwd; echo "$ATOM_JOBS_VAR"', label="where")
    item = await asyncio.wait_for(agent.queue_manager.get_channel("jobs").get(), TIMEOUT)
    assert item.result.stdout.splitlines() == [str(tmp_path / "pkg"), "1"]
