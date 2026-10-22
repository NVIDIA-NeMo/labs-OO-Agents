# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One view of background work: jobs and child sessions in one status list, one way to wait.

The per-turn ``queues`` block lists running ``self.jobs`` jobs and live
children together under "Background work"; ``Waiting(on=[name])`` takes a
job's label or a child's name, and rejects a name that is both.
"""

import asyncio
import re

from atom_test_agents import ATOM_SPEC, AtomModels, cell
from nooa_atom import SessionOptions, SessionStore
from nooa_atom.session.registry import SessionRegistry

TIMEOUT = 30
DONE = "return_result(Done(explanation='done'))"
CHILD_DONE = (
    "return_result(Done(explanation='ok', result=TaskResult("
    "solution_description='s', evidence='e', how_to_verify='h')))"
)


def _slow_child(seconds: float) -> list:
    return [cell(f"await asyncio.sleep({seconds})\n" + CHILD_DONE)]


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
    return "Running right now:" in text


async def test_the_status_lists_jobs_and_live_children_together(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('sleep 30', label='tests', description='full run before pushing')\n"
                "self.v.fix = await self.spawn('Fix flaky test', 'fix it')\n"
                "await self.v.fix.wait()\n"
                "self.v.review = await self.spawn('Review PR 462', 'review it')\n"
                "await asyncio.sleep(0.3)\n"
                "assert self.child('Review PR 462').id == self.v.review.id\n"
                "try:\n    self.child('nope')\nexcept KeyError as e:\n    print('MISSING<<' + str(e) + '>>')\n"
                "print('STATUS<<' + self._queue_status() + '>>')"
            ),
            cell("await self.jobs.cancel('tests')\n" + DONE),
        ],
        {"Fix flaky test": [cell(CHILD_DONE)], "Review PR 462": _slow_child(5)},
    )
    status = calls[1].split("STATUS<<", 1)[1].split(">>", 1)[0]
    assert "no child named 'nope'; open children: ['Fix flaky test', 'Review PR 462']" in calls[1]
    assert "Background work:\n" in status
    assert re.search(r"  • tests · job · running \d+s · full run before pushing", status)
    assert "  • Review PR 462 · child · running" in status
    assert "  • Fix flaky test · child · retained" in status
    assert status.endswith(
        "\nCancel a job: await self.jobs.cancel('<label>')"
        "\nClose a child: await self.child('<name>').close()"
    )


async def test_waiting_on_a_child_name_or_a_job_label_is_accepted(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'work')\n"
                "return_result(Waiting(explanation='w', on=['Worker']))"
            ),
            cell(
                "assert notification['delegates']\n"
                "await self.jobs.run('sleep 0.2', label='tests')\n"
                "return_result(Waiting(explanation='w', on=['tests']))"
            ),
            cell("assert notification['jobs'][0].label == 'tests'\n" + DONE),
        ],
        {"Worker": _slow_child(0.3)},
    )
    assert len(calls) == 3
    assert not any(_rejected(text) for text in calls[1:])


async def test_a_name_that_is_both_a_job_and_a_child_is_rejected(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('sleep 30', label='review')\n"
                "self.v.child = await self.spawn('review', 'review it')\n"
                "return_result(Waiting(explanation='w', on=['review']))"
            ),
            cell(
                "print(self.v.child.id)\nreturn_result(Waiting(explanation='w', on=['jobs:review']))"
            ),
            cell("await self.jobs.cancel('review')\n" + DONE),
        ],
        {"review": _slow_child(5)},
    )
    assert (
        "Waiting on 'review' is ambiguous: it names both a job and a child. "
        "Use 'jobs:review' for the job, or the child's id '"
    ) in calls[1]
    assert not _rejected(calls[2])  # 'jobs:review' names the job alone


async def test_an_unknown_name_lists_what_is_running(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('sleep 30', label='tests')\n"
                "self.v.child = await self.spawn('Worker', 'work')\n"
                "return_result(Waiting(explanation='w', on=['lint']))"
            ),
            cell("await self.jobs.cancel('tests')\n" + DONE),
        ],
        {"Worker": _slow_child(5)},
    )
    # (Closing the session at the end cancels the child, whose ChildFailed
    # may start more turns: only the first two calls matter.)
    text = calls[1]
    assert "Waiting on 'lint', but no running job or child has that name" in text
    assert "self.jobs.get('lint')" in text and "label='lint'" in text
    running = text.split("Running right now:", 1)[1]
    assert "'tests'" in running and "'Worker'" in running


async def test_a_child_with_an_unread_question_says_so_until_it_is_read(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.spawn('Asker', 'ask which branch')\n"
                "delegates = self.queue_manager.get_channel('delegates')\n"
                "async with asyncio.timeout(10):\n"
                "    while not delegates.qsize():\n"
                "        await asyncio.sleep(0.01)\n"
                "print('ASKED<<' + self._queue_status() + '>>')\n"
                "q = await self.delegates.get()\n"
                "assert isinstance(q, ChildQuestion), q\n"
                "print('READ<<' + self._queue_status() + '>>')"
            ),
            cell(DONE),
        ],
        {"Asker": [cell("return_result(NeedInput(question='Which branch?', options=['main']))")]},
    )
    asked = calls[1].split("ASKED<<", 1)[1].split(">>", 1)[0]
    read = calls[1].split("READ<<", 1)[1].split(">>", 1)[0]
    assert "delegates: 1 pending\n  1. Asker asks: Which branch?\n" in asked
    assert "Background work:\n  • Asker · child · has a question for you" in asked
    # Once read, the question is no longer buffered: the kept child shows as retained.
    assert "has a question for you" not in read
    assert "  • Asker · child · retained" in read
