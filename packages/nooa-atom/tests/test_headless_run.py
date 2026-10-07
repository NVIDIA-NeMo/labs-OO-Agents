# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run_task: one unattended task through the Atom agent's session tree."""

import asyncio

from atom_test_agents import ATOM_SPEC, CellLLM, cell
from nooa_atom import SessionOptions, SessionStore, TaskResult, run_task
from nooa_atom.session.store import sessions_root

from nooa.unifiedllm import LLMResponse, LLMUsage

TIMEOUT = 30

RESULT = (
    "return_result(Done(explanation='solved', result=TaskResult("
    "solution_description='wrote hello.py', evidence='it printed hello', "
    "how_to_verify='python hello.py', report='hello.py prints hello')))"
)

# A job that delivers on its own channel shortly after the turn ends.
WAIT_FOR_JOB = """\
jobs = self.queue_manager.queue("test_jobs")
asyncio.get_running_loop().call_later(0.05, jobs.put, "job finished")
return_result(Waiting(explanation="job running", on=["test_jobs"]))
"""

WAIT_FOREVER = """\
self.queue_manager.queue("test_jobs")
return_result(Waiting(explanation="job running", on=["test_jobs"]))
"""


def _options(tmp_path, *responses: LLMResponse) -> SessionOptions:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, llm=CellLLM(list(responses)))


async def test_a_task_solved_in_one_turn_returns_its_task_result(tmp_path):
    options = _options(tmp_path, cell(RESULT, usage=LLMUsage(input_tokens=120, output_tokens=30)))

    run = await asyncio.wait_for(run_task(options, "Create hello.py"), TIMEOUT)

    assert run.stopped is None
    assert run.done is not None and run.done.explanation == "solved"
    assert run.result == TaskResult(
        solution_description="wrote hello.py",
        evidence="it printed hello",
        how_to_verify="python hello.py",
        report="hello.py prints hello",
    )
    assert run.turns == 1
    assert (run.usage.input_tokens, run.usage.output_tokens) == (120, 30)
    # The session is recorded under the workspace like any other, run unattended.
    info = SessionStore(sessions_root(options.workspace)).get(run.session_id)
    assert info.turn_method == "handle_batch"
    # The root agent's events, in order, for the trajectory export.
    names = [type(event).__name__ for event in run.events]
    assert "LLMResponse" in names


async def test_a_waiting_turn_is_followed_until_done(tmp_path):
    options = _options(tmp_path, cell(WAIT_FOR_JOB), cell(RESULT))

    run = await asyncio.wait_for(run_task(options, "Run the job"), TIMEOUT)

    assert run.stopped is None
    assert run.result is not None and run.result.report == "hello.py prints hello"
    assert run.turns == 2


async def test_the_turn_limit_stops_a_run_that_is_still_waiting(tmp_path):
    options = _options(tmp_path, cell(WAIT_FOREVER))

    run = await asyncio.wait_for(run_task(options, "Run the job", max_turns=1), TIMEOUT)

    assert run.done is None and run.result is None
    assert run.stopped == "turn limit (1) reached while waiting on test_jobs"
    assert run.turns == 1


async def test_the_time_limit_stops_a_run(tmp_path):
    options = _options(tmp_path, cell(WAIT_FOREVER))

    run = await asyncio.wait_for(run_task(options, "Run the job", timeout=0.5), TIMEOUT)

    assert run.done is None
    assert run.stopped == "time limit (0.5 s) reached"


async def test_a_failed_turn_is_reported_not_raised(tmp_path):
    options = _options(tmp_path)  # the strict fake model has no response to give

    run = await asyncio.wait_for(run_task(options, "Do something"), TIMEOUT)

    assert run.done is None
    assert run.stopped is not None and run.stopped.startswith("turn failed: ")


async def test_the_turn_loop_registers_the_tracing_hooks(tmp_path, monkeypatch):
    """The loop runs in a fresh context; without the hooks a traced run has no spans."""
    import nooa.tracing as tracing

    calls = []
    monkeypatch.setattr(tracing, "register_hooks_in_current_context", lambda: calls.append(1))
    options = _options(tmp_path, cell(RESULT))

    run = await asyncio.wait_for(run_task(options, "Create hello.py"), TIMEOUT)

    assert run.done is not None
    assert calls
