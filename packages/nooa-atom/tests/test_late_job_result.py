# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A job outcome delivered after ``Done`` starts a late turn; ``run_task`` still reports cleanly.

OV Gym FU2 case 6: the agent polled a Subagent job until ``status == "done"``,
read ``job.result`` and returned ``Done``. Reading ``result`` does not take the
job's delivery, so its ``JobEvent`` (``output=ExecutionResult(...)`` with the
``_NO_RETURN`` sentinel) started a late turn that the host cancelled. The
runner's ``event.model_dump(mode="json")`` over ``TaskRun.events`` then raised
``PydanticSerializationError: Unable to serialize unknown type: <class 'object'>``
and the done result was lost.
"""

import asyncio

from atom_test_agents import PUBLIC_ATOM_SPEC, CellLLM, cell
from nooa_atom import SessionOptions, run_task

from nooa.events import ExecutionResult

TIMEOUT = 30

POLL_THEN_DONE = """\
async def _review():
    await asyncio.sleep(0.05)
    return "review result"
job = self.jobs.launch(_review(), name="review")
while job.status != "done":
    await asyncio.sleep(0.01)
print(job.result)
return_result(Done(explanation="solved", result=TaskResult(
    solution_description="s", evidence="e", how_to_verify="h", report="r")))
"""

LATE_DONE = "return_result(Done(explanation='late turn'))"


def test_an_execution_result_without_a_return_value_dumps_to_json():
    dumped = ExecutionResult(error=ValueError("bad")).model_dump(mode="json")

    assert dumped["returned_value"] is None
    assert dumped["error"] == "ValueError: bad"
    assert ExecutionResult(returned_value=3).model_dump(mode="json")["returned_value"] == 3


def test_an_execution_result_with_an_unencodable_return_value_dumps_to_json():
    class Opaque:
        def __repr__(self) -> str:
            return "Opaque()"

    dumped = ExecutionResult(returned_value=Opaque()).model_dump(mode="json")

    assert dumped["returned_value"] == "Opaque()"
    assert isinstance(
        ExecutionResult(returned_value=Opaque()).model_dump()["returned_value"], Opaque
    )


def test_an_error_message_with_a_lone_surrogate_dumps_to_json():
    result = ExecutionResult(error=ValueError("bad \udcff name"))

    assert result.model_dump(mode="json")["error"] == "ValueError: bad \\udcff name"
    assert '"error":"ValueError: bad \\\\udcff name"' in result.model_dump_json()


async def test_a_job_delivered_after_done_leaves_the_done_and_json_events(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = CellLLM([cell(POLL_THEN_DONE), cell(LATE_DONE)], strict_exhaustion=False)
    options = SessionOptions(workspace=workspace, agent_spec=PUBLIC_ATOM_SPEC, llm=llm)

    run = await asyncio.wait_for(run_task(options, "Review and finish"), TIMEOUT)

    assert run.stopped is None
    assert run.done is not None and run.done.explanation == "solved"
    assert run.result is not None and run.result.report == "r"
    # What the OV Gym runner (atom_main.py) writes to events.jsonl:
    for event in run.events:
        event.model_dump(mode="json")


async def test_no_turn_starts_after_the_root_is_done(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = CellLLM([cell(POLL_THEN_DONE), cell(LATE_DONE)], strict_exhaustion=False)
    options = SessionOptions(workspace=workspace, agent_spec=PUBLIC_ATOM_SPEC, llm=llm)

    run = await asyncio.wait_for(run_task(options, "Review and finish"), TIMEOUT)

    assert run.done is not None and run.done.explanation == "solved"
    assert run.turns == 1  # the job's queued delivery did not start a late turn
    assert not any(type(event).__name__ == "TurnCancelled" for event in run.events)
