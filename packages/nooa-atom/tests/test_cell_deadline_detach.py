# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A child awaited in a cell outlives the cell deadline as a job; other cancels close it.

Real sessions, fake models. The root's cells get a short deadline (children
keep theirs), so ``await Subagent().run(...)`` in a root cell hits it while
the child still works.
"""

import asyncio

import pytest
from atom_test_agents import AtomModels, cell, work_done
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.port import current_port
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

import nooa.runtime.actor as actor
from nooa.interactive import Done
from nooa.runtime.cell_deadline import run_with_deadline
from nooa.runtime.harness_metrics import get_harness_metrics

UNIFIED = "nooa_atom.agent:AtomAgent"
TIMEOUT = 60
ROOT_DEADLINE = 0.5
DETACHED = (
    "job subagent is still running: "
    'Waiting(on=["subagent"]) for its result, '
    'or await self.jobs.get("subagent").cancel().'
)
# The child signals it started, then works past the root's deadline.
_SLOW_CHILD = "await self.files.write_file('started', 'x')\nawait asyncio.sleep({s})\n"


@pytest.fixture
def models():
    return AtomModels()


@pytest.fixture
async def registry(sessions_dir, models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    yield registry
    await registry.close_all()


@pytest.fixture
def options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=UNIFIED, sessions_dir=sessions_dir)


@pytest.fixture
def root_metrics(monkeypatch):
    """Shorten the root session's cell deadline; collect its cells' harness metrics."""
    seen = []

    async def short_deadline(awaitable, timeout):
        port = current_port.get()
        if port is not None and port.depth == 0:
            seen.append(get_harness_metrics())
            timeout = ROOT_DEADLINE
        return await run_with_deadline(awaitable, timeout)

    monkeypatch.setattr(actor, "run_with_deadline", short_deadline)
    return seen


# The detached job's events: its note (an item) may wake a turn before its result.
_UNTIL_DONE = cell(
    "for event in notification['jobs']:\n"
    "    assert event.name == 'subagent', event\n"
    "    if event.kind == 'item':\n"
    "        assert event.value == 'NOTE: late', event\n"
    "        self.v.note = event.value\n"
    "    else:\n"
    "        assert event.kind == 'done', event\n"
    "        return_result(Done(explanation=event.value.report))\n"
    "return_result(Waiting(explanation='child runs', on=['subagent']))"
)


def _prompt(models, name, call=0):
    return str(models.llms[name].calls[call].messages)


def _status(registry, root):
    return {c.name: c.status for c in registry.children(root.id)}


async def test_a_child_awaited_at_the_deadline_runs_on_as_a_job_and_wakes_waiting(
    registry, options, models, root_metrics
):
    models.scripts[None] = [
        cell("r = await Subagent('slow').run('SLOW')\nreturn_result(Done(explanation='early'))"),
        cell(
            "job = self.jobs.get('subagent')\n"
            "assert job.status == 'running', job\n"
            "return_result(Waiting(explanation='child runs', on=['subagent']))"
        ),
        _UNTIL_DONE,
        _UNTIL_DONE,
    ]
    models.scripts["subagent"] = [
        cell(_SLOW_CHILD.format(s=1.5) + "self.message('NOTE: late')"),
        work_done(report="LATE"),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="LATE")
    feedback = _prompt(models, None, 1)
    assert "timed out after 0.5 seconds" in feedback
    assert DETACHED.replace('"', '\\"') in feedback or DETACHED in feedback
    assert root._agent.v.note == "NOTE: late"  # the child's notes are the job's items
    assert root._agent.jobs.get("subagent").status == "done"
    attrs = root_metrics[0].to_span_attributes()
    assert attrs["harness.atom.child_detached_at_deadline.count"] == 1


async def test_a_detached_child_that_fails_arrives_as_failed(
    registry, options, models, root_metrics
):
    models.scripts[None] = [
        cell("await Subagent().run('SLOW')"),
        cell("return_result(Waiting(explanation='child runs', on=['subagent']))"),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind) == ('subagent', 'failed'), event\n"
            "assert 'subagent subagent failed' in str(event.value), event\n"
            "return_result(Done(explanation='failed as expected'))"
        ),
    ]
    # Its second model call is unscripted: the strict fake model fails the child's turn.
    models.scripts["subagent"] = [cell(_SLOW_CHILD.format(s=1.0))]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="failed as expected")


async def _closed(registry, root, rounds=400):
    """Every child of ``root`` is closed (or closed and unloaded), within a bound."""
    for _ in range(rounds):
        if {c.status for c in registry.children(root.id)} <= {"closed", "on_disk"}:
            return True
        await asyncio.sleep(0.05)
    return False


async def test_cancelling_the_detached_job_closes_the_child(
    registry, options, models, root_metrics
):
    models.scripts[None] = [
        cell("await Subagent().run('ONE')"),
        # A job, so the cancel is not itself bound by the short cell deadline.
        cell(
            "self.jobs.launch(self.jobs.get('subagent').cancel(), name='stop')\n"
            "return_result(Done(explanation='cancelling'))"
        ),
    ]
    models.scripts["subagent"] = [cell(_SLOW_CHILD.format(s=30))]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="cancelling")
    assert await _closed(registry, root)
    assert root._agent.jobs.get("subagent").status == "cancelled"


async def test_session_close_closes_a_detached_child(registry, options, models, root_metrics):
    models.scripts[None] = [
        cell("await Subagent().run('ONE')"),
        cell("return_result(Done(explanation='left running'))"),
    ]
    models.scripts["subagent"] = [cell(_SLOW_CHILD.format(s=30))]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="left running")
    [child] = registry.children(root.id)
    assert child.status != "closed"  # detached, still working
    assert root._agent.jobs.get("subagent").status == "running"
    await registry.close(root.id)
    assert await _closed(registry, root)


async def test_a_turn_cancel_still_closes_the_awaited_child(registry, options, models):
    models.scripts[None] = [cell("await Subagent().run('SLOW')")]
    models.scripts["subagent"] = [cell(_SLOW_CHILD.format(s=30))]
    root = await registry.create(options)
    turn = asyncio.ensure_future(root.prompt("go"))
    workspace = options.workspace
    async with asyncio.timeout(TIMEOUT):
        while not (workspace / "started").exists():
            await asyncio.sleep(0.05)
    assert await root.cancel(by="user")
    outcome = await asyncio.wait_for(turn, TIMEOUT)
    assert type(outcome).__name__ == "TurnCancelledOutcome", outcome
    async with asyncio.timeout(TIMEOUT):
        while _status(registry, root).get("subagent") != "closed":
            await asyncio.sleep(0.05)
    assert root._agent.jobs.get("subagent") is None  # nothing was handed over


async def test_an_awaited_request_job_survives_the_deadline_and_is_named(
    registry, options, models, root_metrics
):
    models.scripts[None] = [
        cell("rev = self.jobs.launch(Subagent(), name='review')\nawait rev.send('SLOW')"),
        cell("return_result(Waiting(explanation='request runs', on=['review#1']))"),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind) == ('review#1', 'done'), event\n"
            "return_result(Done(explanation=event.value.report))"
        ),
    ]
    models.scripts["review"] = [cell(_SLOW_CHILD.format(s=1.5)), work_done(report="KEPT")]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="KEPT")
    feedback = _prompt(models, None, 1)
    assert "job review#1 is still running" in feedback
    attrs = root_metrics[0].to_span_attributes()
    # A kept child's request detached at the deadline counts like a one-shot child.
    assert attrs["harness.atom.child_detached_at_deadline.count"] == 1


async def test_an_awaited_plain_job_at_the_deadline_is_not_a_detached_child(
    registry, options, models, root_metrics
):
    models.scripts[None] = [
        cell("job = self.jobs.launch(asyncio.sleep(1.5, 'SLEPT'), name='nap')\nawait job"),
        cell("return_result(Waiting(explanation='nap runs', on=['nap']))"),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind) == ('nap', 'done'), event\n"
            "return_result(Done(explanation=event.value))"
        ),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="SLEPT")
    assert "job nap is still running" in _prompt(models, None, 1)
    attrs = root_metrics[0].to_span_attributes()
    assert "harness.atom.child_detached_at_deadline.count" not in attrs


# ---- a Shell command awaited at the deadline ----

SHELL_DETACHED = "`sleep 2; echo done` is still running as job sh"


async def test_a_shell_command_awaited_at_the_deadline_runs_on_as_a_job(
    registry, options, models, root_metrics
):
    models.scripts[None] = [
        cell("sh = Shell()\nr = await sh.run('sleep 2; echo done')"),
        cell(
            "try:\n"
            "    await sh.run('echo hi')\n"
            "except RuntimeError as e:\n"
            "    self.v.busy = str(e)\n"
            "return_result(Waiting(explanation='sh runs', on=['sh']))"
        ),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind) == ('sh', 'done'), event\n"
            "r = event.value\n"
            "assert (r.returncode, r.stdout, r.timed_out) == (0, 'done\\n', False), r\n"
            "assert (await sh.run('echo again')).stdout == 'again\\n'\n"
            "return_result(Done(explanation=r.stdout))"
        ),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="done\n")
    feedback = _prompt(models, None, 1)
    assert "timed out after 0.5 seconds" in feedback
    assert SHELL_DETACHED in feedback
    assert 'await self.jobs.get(\\"sh\\").cancel()' in feedback or (
        'await self.jobs.get("sh").cancel()' in feedback
    )
    # The Shell stayed busy while the job ran; the refusal names the job.
    assert "(job sh)" in root._agent.v.busy, root._agent.v.busy
    attrs = root_metrics[0].to_span_attributes()
    assert attrs["harness.atom.shell_detached_at_deadline.count"] == 1
    assert "harness.atom.child_detached_at_deadline.count" not in attrs


async def test_a_queued_shell_command_handed_over_keeps_its_job_name(
    registry, options, models, root_metrics
):
    # Two runs on one Shell, gathered: at the deadline the running one becomes
    # job sh and the queued one sh_2; when sh_2 gets the Shell, it is named busy.
    models.scripts[None] = [
        cell(
            "import asyncio\n"
            "sh = Shell()\n"
            "await asyncio.gather(sh.run('sleep 1.5; echo a'), sh.run('sleep 3; echo b'))"
        ),
        cell("return_result(Waiting(explanation='sh runs', on=['sh']))"),
        cell(
            "try:\n"
            "    await sh.run('echo hi')\n"
            "except RuntimeError as e:\n"
            "    self.v.busy = str(e)\n"
            "return_result(Waiting(explanation='sh_2 runs', on=['sh_2']))"
        ),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.value.stdout) == ('sh_2', 'b\\n'), event\n"
            "return_result(Done(explanation='ok'))"
        ),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="ok")
    assert "(job sh_2)" in root._agent.v.busy, root._agent.v.busy
    attrs = root_metrics[0].to_span_attributes()
    assert attrs["harness.atom.shell_detached_at_deadline.count"] == 2


async def test_a_user_cancel_still_stops_an_awaited_shell_command(registry, options, models):
    models.scripts[None] = [
        cell("await Shell().run('(echo $BASHPID > pid; exec sleep 30)')"),
    ]
    root = await registry.create(options)
    turn = asyncio.ensure_future(root.prompt("go"))
    pidfile = options.workspace / "pid"
    async with asyncio.timeout(TIMEOUT):
        while not (pidfile.exists() and pidfile.read_text().strip()):
            await asyncio.sleep(0.05)
    pid = int(pidfile.read_text())
    assert await root.cancel(by="user")
    outcome = await asyncio.wait_for(turn, TIMEOUT)
    assert type(outcome).__name__ == "TurnCancelledOutcome", outcome
    assert root._agent.jobs.get("sh") is None  # nothing was handed over
    async with asyncio.timeout(TIMEOUT):
        while _alive(pid):
            await asyncio.sleep(0.05)


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False
