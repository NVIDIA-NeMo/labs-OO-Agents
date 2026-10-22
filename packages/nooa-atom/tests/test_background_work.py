# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One view of background work: jobs and child sessions in one status list, one way to wait.

The per-turn ``queues`` block lists running ``self.jobs`` jobs and live
children together, by path (``/root/tests``), under "Unread:", "Running:" and "Idle:";
``Waiting(on=[name])`` takes a job's label or a child's name (or path). Jobs
and children share one namespace of live names, so no name is both.
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
                "self.v.fix = await self.spawn('Fix flaky test', 'fix it', name='fix_flaky_test')\n"
                "await self.v.fix.wait()\n"
                "self.v.review = await self.spawn('Review PR 462', 'review it', name='review_pr_462')\n"
                "await asyncio.sleep(0.3)\n"
                "assert self.child('review_pr_462').id == self.v.review.id\n"
                "try:\n    self.child('nope')\nexcept KeyError as e:\n    print('MISSING<<' + str(e) + '>>')\n"
                "print('STATUS<<' + self._queue_status() + '>>')"
            ),
            cell("await self.jobs.cancel('tests')\n" + DONE),
        ],
        {"fix_flaky_test": [cell(CHILD_DONE)], "review_pr_462": _slow_child(5)},
    )
    status = calls[1].split("STATUS<<", 1)[1].split(">>", 1)[0]
    assert "no child named 'nope'; open children: ['fix_flaky_test', 'review_pr_462']" in calls[1]
    assert re.search(
        r"^Running:\n  • /root/tests · job · \d+s · full run before pushing\n"
        r"  • /root/review_pr_462 · child · \d+s\n"
        r"Idle:\n  • /root/fix_flaky_test · child · retained\n",
        status,
    ), status
    assert status.endswith(
        "\nCancel a job: await self.jobs.cancel('<name>')"
        "\nClose a child: await self.child('<name>').close()"
    )


async def test_waiting_on_a_child_name_or_a_job_label_is_accepted(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'work', name='worker')\n"
                "return_result(Waiting(explanation='w', on=['worker']))"
            ),
            cell(
                "assert notification['delegates']\n"
                "await self.jobs.run('sleep 0.2', label='tests')\n"
                "return_result(Waiting(explanation='w', on=['tests']))"
            ),
            cell("assert notification['jobs'][0].label == 'tests'\n" + DONE),
        ],
        {"worker": _slow_child(0.3)},
    )
    assert len(calls) == 3
    assert not any(_rejected(text) for text in calls[1:])


async def test_jobs_and_children_share_one_namespace_of_live_names(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('sleep 30', label='review')\n"
                "try:\n    await self.spawn('Review it', 'review it', name='review')\n"
                "except ValueError as e:\n    print('CHILD<<' + str(e) + '>>')\n"
                "self.v.child = await self.spawn('Lint it', 'lint it', name='lint')\n"
                "try:\n    await self.jobs.run('true', label='lint')\n"
                "except ValueError as e:\n    print('JOB<<' + str(e) + '>>')\n"
                "try:\n    await self.spawn('Lint again', 'lint it', name='lint')\n"
                "except ValueError as e:\n    print('TWICE<<' + str(e) + '>>')\n"
                "try:\n    await self.spawn('Bad', 'x', name='ci-check')\n"
                "except ValueError as e:\n    print('BAD<<' + str(e) + '>>')\n"
                "await self.jobs.cancel('review')\n"
                "await self.v.child.close()\n"
                "await self.jobs.run('true', label='lint')\n"  # free again once closed
                "self.v.again = await self.spawn('Review it', 'review it', name='review')\n"
                "print('PATHS<<' + self.v.again.path + ' ' + repr(self.v.again) + '>>')\n"
                "return_result(Waiting(explanation='w', on=['review']))"
            ),
            cell(DONE),
        ],
        {"review": _slow_child(0.5), "lint": _slow_child(5)},
    )
    text = calls[1]
    assert (
        "CHILD<<name 'review' is already used by the running job /root/review; pick another "
        "name, or cancel the job first (await self.jobs.cancel('review'))>>"
    ) in text
    assert (
        "JOB<<name 'lint' is already used by child /root/lint; pick another label, or close "
        "the child first (await self.child('lint').close())>>"
    ) in text
    assert "TWICE<<name 'lint' is already used by child /root/lint (" in text
    assert "BAD<<child name 'ci-check' is not valid: use a Python identifier" in text
    paths = text.split("PATHS<<", 1)[1].split(">>", 1)[0]
    assert paths.startswith("/root/review ChildRef(name='review', path='/root/review',")
    assert " id=" not in paths
    assert not _rejected(calls[1])


async def test_an_unknown_name_lists_what_is_running(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.jobs.run('sleep 30', label='tests')\n"
                "self.v.child = await self.spawn('Worker', 'work', name='worker')\n"
                "return_result(Waiting(explanation='w', on=['lint']))"
            ),
            cell("await self.jobs.cancel('tests')\n" + DONE),
        ],
        {"worker": _slow_child(5)},
    )
    # (Closing the session at the end cancels the child, whose ChildFailed
    # may start more turns: only the first two calls matter.)
    text = calls[1]
    assert "Waiting on 'lint', but no running job or child has that name" in text
    assert "self.jobs.get('lint')" in text and "label='lint'" in text
    running = text.split("Running right now:", 1)[1]
    assert "'tests'" in running and "'worker'" in running


async def test_a_child_with_an_unread_question_says_so_until_it_is_read(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "await self.spawn('Asker', 'ask which branch', name='asker')\n"
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
        {"asker": [cell("return_result(NeedInput(question='Which branch?'))")]},
    )
    asked = calls[1].split("ASKED<<", 1)[1].split(">>", 1)[0]
    read = calls[1].split("READ<<", 1)[1].split(">>", 1)[0]
    # Unread: only its question line, not under Running/Idle too.
    assert asked == (
        "Unread:\n  • /root/asker asks: Which branch?\n"
        "Close a child: await self.child('<name>').close()"
    )
    # Once read but not answered, it stays under Unread, still not Idle.
    assert read == (
        "Unread:\n  • /root/asker · child · question not answered yet\n"
        "Close a child: await self.child('<name>').close()"
    )


async def test_paths_of_jobs_and_a_grandchild(tmp_path, sessions_dir):
    """Two jobs and a child with its own child: the list shows full paths of your own work only."""
    models = AtomModels()
    root_cells = [
        cell(
            "await self.jobs.run('sleep 30', label='tests', description='full run before pushing')\n"
            "self.jobs.every(60, label='ci_check', description='watch CI on PR 462')\n"
            "self.v.review = await self.spawn('Review PR 462', 'review it', name='review_pr_462')\n"
            "async with asyncio.timeout(10):\n"
            "    while True:\n"
            "        try:\n"
            "            deep = self.child('/root/review_pr_462/verifier')\n"
            "            break\n"
            "        except KeyError:\n"
            "            await asyncio.sleep(0.02)\n"
            "print('DEEP<<' + deep.path + ' ' + deep.status + '>>')\n"
            "assert self.child('/root/review_pr_462').id == self.v.review.id\n"
            "assert self.child(self.v.review.id).name == 'review_pr_462'\n"
            "print('STATUS<<' + self._queue_status() + '>>')\n"
            "return_result(Waiting(explanation='w', on=['/root/review_pr_462/verifier']))"
        ),
        cell(
            "await self.jobs.cancel('tests')\nawait self.jobs.cancel('ci_check')\n"
            "return_result(Done(explanation='done'))"
        ),
    ]
    models.scripts[None] = root_cells
    models.scripts["review_pr_462"] = [
        cell(
            "self.v.verifier = await self.spawn('Verify the fix', 'verify it', name='verifier')\n"
            "return_result(Waiting(explanation='w', on=['verifier']))"
        )
    ]
    models.scripts["verifier"] = _slow_child(30)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        outcome = await asyncio.wait_for(root.prompt("hi"), TIMEOUT)
        assert outcome.explanation == "done"
    finally:
        await registry.close_all()
    calls = models.llms[None].calls

    def added(index):
        new = calls[index].messages[len(calls[index - 1].messages) :]
        return "\n".join(str(message.get("content") or "") for message in new)

    text = added(1)
    assert "DEEP<</root/review_pr_462/verifier running>>" in text
    status = text.split("STATUS<<", 1)[1].split(">>", 1)[0]
    lines = status.splitlines()
    assert lines[0] == "Running:"
    assert re.fullmatch(r"  • /root/tests · job · \d+s · full run before pushing", lines[1])
    assert re.fullmatch(r"  • /root/ci_check · job · \d+s · watch CI on PR 462", lines[2])
    # Its turn's age until the turn ends Waiting on its own child, then "waiting on its own job".
    assert re.fullmatch(
        r"  • /root/review_pr_462 · child · (\d+s|waiting on its own job)", lines[3]
    )
    assert "verifier" not in status  # only your own jobs and children
    rejected = text  # the error from the same cell's return_result
    assert (
        "Waiting on '/root/review_pr_462/verifier', but it is not your child: what it "
        "delivers goes to its parent. Wait on your child 'review_pr_462' instead."
    ) in rejected
    first = str(models.llms["review_pr_462"].calls[0].messages)
    assert "You are /root/review_pr_462, a child of /root. You can ask" in first
    deeper = str(models.llms["verifier"].calls[0].messages)
    assert "You are /root/review_pr_462/verifier, a child of /root/review_pr_462." in deeper


async def test_the_queues_block_reads_unread_running_idle(tmp_path, sessions_dir):
    """The whole block: an unread question and job batches, running jobs and children, idle."""
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "await self.jobs.run('sleep 30', label='tests', description='full test run before pushing')\n"
            "await self.jobs.run('for i in $(seq 1 2000); do echo line $i; sleep 0.02; done',"
            " label='build', items=50)\n"
            "self.jobs.every(600, label='ci_check', description='check PR 480 CI every 10 min')\n"
            "self.v.fix = await self.spawn('Fix flaky test', 'fix it', name='fix_flaky_test')\n"
            "await self.v.fix.wait()\n"
            "self.v.review = await self.spawn('Review PR 480', 'review it', name='review_pr_480')\n"
            "await self.spawn('Pick branch', 'ask', name='pick_branch')\n"
            "delegates = self.queue_manager.get_channel('delegates')\n"
            "jobs = self.queue_manager.get_channel('jobs')\n"
            "async with asyncio.timeout(20):\n"
            "    while not (delegates.qsize() and jobs.qsize() >= 2):\n"
            "        await asyncio.sleep(0.02)\n"
            "    while True:\n"
            "        try:\n"
            "            self.child('/root/review_pr_480/verifier')\n"
            "            break\n"
            "        except KeyError:\n"
            "            await asyncio.sleep(0.02)\n"
            "print('STATUS<<' + self._queue_status() + '>>')"
        ),
        cell(
            "for label in ('tests', 'build', 'ci_check'):\n"
            "    await self.jobs.cancel(label)\n" + DONE
        ),
    ]
    models.scripts["fix_flaky_test"] = [cell(CHILD_DONE)]
    models.scripts["review_pr_480"] = [
        cell(
            "self.v.verifier = await self.spawn('Verify', 'verify it', name='verifier')\n"
            "await asyncio.sleep(30)\n" + CHILD_DONE
        )
    ]
    models.scripts["verifier"] = _slow_child(30)
    models.scripts["pick_branch"] = [
        cell("return_result(NeedInput(question='Which branch should I push to: main or dev?'))")
    ]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        outcome = await asyncio.wait_for(root.prompt("hi"), TIMEOUT)
        assert outcome.explanation == "done"
    finally:
        await registry.close_all()
    calls = models.llms[None].calls
    added = calls[1].messages[len(calls[0].messages) :]
    text = "\n".join(str(message.get("content") or "") for message in added)
    status = text.split("STATUS<<", 1)[1].split(">>", 1)[0]
    print(status)
    lines = status.splitlines()
    assert lines[0] == "Unread:"
    assert lines[1] == "  • /root/pick_branch asks: Which branch should I push to: main or dev?"
    assert re.fullmatch(r"  • /root/build: 50 lines, last: line \d+", lines[2])
    running = lines.index("Running:")
    idle = lines.index("Idle:")
    assert 2 < running < idle
    assert re.fullmatch(
        r"  • /root/tests · job · \d+s · full test run before pushing", lines[running + 1]
    )
    assert lines[running + 2].startswith("  • /root/build · job · ")
    assert re.fullmatch(
        r"  • /root/ci_check · job · \d+s · check PR 480 CI every 10 min", lines[running + 3]
    )
    assert re.fullmatch(r"  • /root/review_pr_480 · child · \d+s", lines[running + 4])
    assert lines[idle:] == [
        "Idle:",
        "  • /root/fix_flaky_test · child · retained",
        "Cancel a job: await self.jobs.cancel('<name>')",
        "Close a child: await self.child('<name>').close()",
    ]
    assert "pick_branch" not in "\n".join(lines[running:])  # its question is under Unread only
    assert "verifier" not in status
