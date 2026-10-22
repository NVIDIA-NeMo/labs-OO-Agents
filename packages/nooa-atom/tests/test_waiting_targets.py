# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive ``handle()`` turns may only wait on something that is running.

``require_running_wait`` checks each name in ``Waiting.on`` against the
queue manager's running jobs and the session's children. A rejected
``Waiting`` is shown to the model as the error from its cell, with a hint
to return ``Done`` or ``NeedInput`` instead; the turn continues.
"""

import asyncio

from atom_test_agents import ATOM_SPEC, AtomModels, cell
from nooa_atom import SessionOptions, SessionStore
from nooa_atom.session.registry import SessionRegistry

TIMEOUT = 30
DONE = "return_result(Done(explanation='done'))"
HINT = (
    "Waiting is only for running jobs or background agents. If the work is finished, "
    "return Done; if you need an answer from the person, return NeedInput."
)
# A stand-in for agent_mesh's inbox pump: a job feeding the "mesh" channel.
MESH_PUMP = (
    "self.queue_manager.queue('mesh')\n"
    "self.v.pump = self.queue_manager.spawn("
    "asyncio.sleep(0.3, result='hello from mesh'), channel='mesh', label='mesh inbox pump')\n"
)


async def _run(tmp_path, sessions_dir, root_cells, workers=None):
    """Run one interactive turn to its Done; return what each model call added, as text."""
    models = AtomModels()
    models.scripts[None] = root_cells
    for name, cells in (workers or {}).items():
        models.scripts[name] = cells
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert root.info.turn_method == "handle"
        outcome = await asyncio.wait_for(root.prompt("hi"), TIMEOUT)
        assert outcome.explanation == "done"
    finally:
        await registry.close_all()
    calls = models.llms[None].calls
    return [
        "\n".join(
            str(message.get("content") or "")
            for message in call.messages[len(calls[index - 1].messages) if index else 0 :]
        )
        for index, call in enumerate(calls)
    ]


def _rejected(text):
    return "WaitingTargetNotRunning" in text or "Running right now:" in text


async def test_waiting_on_mesh_is_accepted_while_the_mesh_pump_runs(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(MESH_PUMP + "return_result(Waiting(explanation='w', on=['mesh']))"),
            cell("assert notification['mesh'] == ['hello from mesh']\n" + DONE),
        ],
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


async def test_waiting_on_mesh_without_a_pump_is_rejected(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell("return_result(Waiting(explanation='w', on=['mesh']))"),
            cell(DONE),
        ],
    )
    assert len(calls) == 2
    assert "Waiting on 'mesh', but no running job or child has that name" in calls[1]
    assert HINT in calls[1]
    assert "Running right now: nothing." in calls[1]


async def test_waiting_on_the_person_is_rejected_with_the_done_or_need_input_hint(
    tmp_path, sessions_dir
):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(MESH_PUMP + "return_result(Waiting(explanation='w', on=['user_messages']))"),
            cell(DONE),
        ],
    )
    assert len(calls) == 2
    assert "Waiting on 'user_messages', but the person is not a job." in calls[1]
    assert HINT in calls[1]
    # What is running is listed, and the person is never among it.
    # The mesh pump is a queue-manager task, not a ``self.jobs`` job: no 'jobs'.
    assert "Running right now: 'mesh', 'mesh inbox pump'." in calls[1]


async def test_waiting_on_delegates_is_accepted_with_a_running_child(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it', name='worker')\n"
                "return_result(Waiting(explanation='w', on=['delegates']))"
            ),
            cell("[item] = notification['delegates']\n" + DONE),
        ],
        {
            "worker": [
                cell(
                    "await asyncio.sleep(0.2)\n"
                    "return_result(Done(explanation='fixed', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h')))"
                )
            ]
        },
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


async def test_waiting_on_delegates_without_a_child_is_rejected(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [cell("return_result(Waiting(explanation='w', on=['delegates']))"), cell(DONE)],
    )
    assert len(calls) == 2
    assert "Waiting on 'delegates', but no child is running" in calls[1]
    assert HINT in calls[1]


async def test_a_finished_or_cancelled_job_is_rejected(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.queue_manager.queue('ci')\n"
                "done_job = self.queue_manager.spawn(asyncio.sleep(0, result='x'), "
                "channel='ci', label='ci-1')\n"
                "await asyncio.sleep(0.05)\n"
                "assert done_job.state != 'running', done_job.state\n"
                "self.queue_manager.get_channel('ci').flush()\n"
                "return_result(Waiting(explanation='w', on=['ci']))"
            ),
            cell(
                "self.queue_manager.queue('build')\n"
                "job = self.queue_manager.spawn(asyncio.Event().wait(), channel='build', "
                "label='build-1')\n"
                "await job.cancel()\n"
                "assert job.state != 'running', job.state\n"
                "return_result(Waiting(explanation='w', on=['build']))"
            ),
            cell(DONE),
        ],
    )
    # The strategy allows three rejected results per turn; this uses two.
    assert len(calls) == 3
    assert "Waiting on 'ci', but no job labelled 'ci' is running: it may" in calls[1]
    assert "Waiting on 'build', but no job labelled 'build' is running" in calls[2]
    for text in calls[1:]:
        assert HINT in text


async def test_waiting_on_jobs_without_a_running_job_is_rejected(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [cell("return_result(Waiting(explanation='w', on=['jobs']))"), cell(DONE)],
    )
    assert len(calls) == 2
    assert "Waiting on 'jobs', but no job is running and nothing is queued" in calls[1]
    assert HINT in calls[1]


async def test_waiting_on_jobs_with_only_a_queue_manager_task_is_rejected(tmp_path, sessions_dir):
    # A queue-manager task on its own channel never delivers on ``jobs``.
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.queue_manager.queue('ci')\n"
                "self.queue_manager.spawn(asyncio.sleep(0.3), channel='ci', label='ci-1')\n"
                "return_result(Waiting(explanation='w', on=['jobs']))"
            ),
            cell(DONE),
        ],
    )
    assert len(calls) == 2
    assert "Waiting on 'jobs', but no job is running and nothing is queued" in calls[1]
    assert HINT in calls[1]


async def test_waiting_on_jobs_is_accepted_while_a_self_jobs_job_runs(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('sleep 0.3; echo hi', label='slow')\n"
                "return_result(Waiting(explanation='w', on=['jobs']))"
            ),
            cell("assert notification['jobs'][0].label == 'slow'\n" + DONE),
        ],
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


async def test_waiting_on_jobs_is_accepted_when_its_result_is_buffered(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('echo hi', label='quick')\n"
                "while self.jobs.get('quick').state == 'running':\n"
                "    await asyncio.sleep(0.02)\n"
                "return_result(Waiting(explanation='w', on=['jobs']))"
            ),
            cell("assert notification['jobs'][0].label == 'quick'\n" + DONE),
        ],
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])
