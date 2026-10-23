# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The experimental ``UnifiedAtomAgent``: shells and subagents as ``self.jobs`` work.

Real sessions (``SessionRegistry``), fake models: each session's model is
scripted by its name (``None`` for the root), and a ``Subagent`` child is
named after its job.
"""

import asyncio

import pytest
from atom_test_agents import AtomModels, ask, cell, done, work_done
from nooa_atom.agent.work_shell import Shell
from nooa_atom.agent.work_subagent import Question
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done

UNIFIED = "nooa_atom.agent.unified:UnifiedAtomAgent"
TIMEOUT = 60

WORK_RESULT = (
    "return_result(Done(explanation='{e}', result=TaskResult(solution_description='s', "
    "evidence='e', how_to_verify='v', report='{r}')))"
)


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


def _prompt(models, name, call=0):
    return str(models.llms[name].calls[call].messages)


async def _settle(root, rounds=20):
    """Let listeners and follow-up turns run."""
    for _ in range(rounds):
        await asyncio.sleep(0.05)


def _delegates_empty(root):
    channel = root._agent.queue_manager.channels().get("delegates")
    return channel is None or channel.qsize() == 0


async def test_one_shot_subagent_returns_its_task_result_and_notes_are_job_items(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "todo = self.todo.add('fix the parser', description='CHILD-PROMPT: fix it')\n"
            "job = self.jobs.launch(Subagent('fix parser').run(todo), name='fix')\n"
            "result = await job\n"
            "assert isinstance(result, TaskResult)\n"
            "return_result(Done(explanation=result.report))"
        ),
        # The child's note was an item of the awaited job: it arrives as a JobEvent.
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind, event.value) == ('fix', 'item', 'NOTE: halfway')\n"
            "assert 'delegates' not in notification\n"
            "return_result(Done(explanation='saw the note'))"
        ),
    ]
    models.scripts["fix"] = [
        cell("self.message('NOTE: halfway')\n" + WORK_RESULT.format(e="fixed", r="REPORT-1"))
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("fix it"), TIMEOUT) == Done(explanation="REPORT-1")
    await _settle(root)
    assert "CHILD-PROMPT: fix it" in _prompt(models, "fix")
    assert "One-shot task" in _prompt(models, "fix")
    assert len(models.llms[None].calls) == 2  # the note's turn ran
    assert _delegates_empty(root)
    [child] = registry.children(root.id)
    assert (child.name, child.turn_method, child.retained, child.agent) == (
        "fix",
        "handle_work",
        False,
        UNIFIED,
    )


async def test_kept_subagent_question_answer_and_follow_up(registry, options, models):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent('review'), name='review')\n"
            "q = await rev.send('REVIEW-TASK')\n"
            "assert isinstance(q, Question) and q.text == 'SECRET-Q which file?', q\n"
            "summary, status = self.jobs.get('review').summary, self._status()\n"
            "assert 'asked a question, awaiting your answer' in summary, summary\n"
            "assert 'review · asked a question' in status, status\n"
            "assert 'SECRET' not in summary + status\n"
            "r1 = await rev.send('ANSWER-42')\n"
            "r2 = await rev.send('FOLLOW-UP')\n"
            "self.v.reports = [r1.report, r2.report]\n"
            "self.v.names = [j.name for j in self.jobs.list()]\n"
            "return_result(Done(explanation='reviewed'))"
        )
    ]
    models.scripts["review"] = [
        ask("SECRET-Q which file?"),
        cell(
            "assert 'ANSWER-42' in str(notification)\n" + WORK_RESULT.format(e="answered", r="R1")
        ),
        cell("assert 'FOLLOW-UP' in str(notification)\n" + WORK_RESULT.format(e="more", r="R2")),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("review"), TIMEOUT)
    assert outcome == Done(explanation="reviewed")
    agent = root._agent
    assert agent.v.reports == ["R1", "R2"]
    assert agent.v.names == ["review", "review#1", "review#2", "review#3"]
    assert "You can ask your parent questions" in _prompt(models, "review")
    # The kept child stays open after an interactive Done, idle.
    [child] = registry.children(root.id)
    assert child.retained and child.status != "closed"
    assert "idle" in agent.jobs.get("review").summary
    assert _delegates_empty(root)


async def test_background_request_arrives_once_and_waiting_checks_targets(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "req = rev.send('TASK')\n"
            "return_result(Waiting(explanation='review running', on=[req.name]))"
        ),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind) == ('review#1', 'done')\n"
            "assert event.value.report == 'BG'\n"
            "assert self.jobs.get('review#1').result.report == 'BG'\n"
            # The resource is idle now: waiting on it would never end.
            "return_result(Waiting(explanation='more?', on=['review']))"
        ),
        done("finished"),
    ]
    models.scripts["review"] = [work_done("done", report="BG")]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="finished")
    feedback = _prompt(models, None, 2)
    assert "review\\' is an idle resource" in feedback
    assert _delegates_empty(root)


async def test_a_failed_subagent_raises_in_the_cell(registry, options, models):
    models.scripts[None] = [
        cell(
            "try:\n"
            "    await Subagent().run('doomed')\n"
            "except RuntimeError as exc:\n"
            "    self.v.error = str(exc)\n"
            "return_result(Done(explanation='handled'))"
        )
    ]
    models.scripts["subagent"] = []  # the strict fake model fails the child's turn
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="handled")
    assert root._agent.v.error.startswith("subagent subagent failed:")
    assert _delegates_empty(root)


_WAIT_FOR_START = (
    "for _ in range(400):\n"
    "    if (await Shell().run('test -f started')).returncode == 0:\n"
    "        break\n"
    "    await asyncio.sleep(0.05)\n"
    "else:\n"
    "    raise AssertionError('the child never started')\n"
)
_SLOW_CHILD = "await self.files.write_file('started', 'x')\nawait asyncio.sleep(60)\n"


async def test_cancelling_a_request_stops_the_childs_turn_and_it_takes_more_work(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "req = rev.send('SLOW')\n" + _WAIT_FOR_START + "await req.cancel()\n"
            "assert req.status == 'cancelled'\n"
            "await Shell().run('rm started')\n"
            "r = await rev.send('AGAIN')\n"
            "self.v.report = r.report\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["review"] = [cell(_SLOW_CHILD), work_done("again", report="AFTER-CANCEL")]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    assert root._agent.v.report == "AFTER-CANCEL"
    await _settle(root)
    # The cancelled turn's failure went nowhere: no event, no delegates item, no turn.
    assert root._agent._jobs_in.qsize() == 0
    assert len(models.llms[None].calls) == 1
    assert _delegates_empty(root)


async def test_a_child_closed_under_a_request_fails_it(registry, options, models):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "req = rev.send('SLOW')\n" + _WAIT_FOR_START + "[child] = self.children()\n"
            "await self.session.close_child(child.id)\n"
            "try:\n"
            "    await req\n"
            "except RuntimeError as exc:\n"
            "    self.v.error = str(exc)\n"
            "try:\n"
            "    rev.send('more')\n"
            "    await self.jobs.get('review#2')\n"
            "except RuntimeError as exc:\n"
            "    self.v.again = str(exc)\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["review"] = [cell(_SLOW_CHILD)]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    agent = root._agent
    assert "review" in agent.v.error
    assert "launch a new Subagent" in agent.v.again
    assert _delegates_empty(root)


async def test_cancelling_the_resource_closes_the_child_and_session_close_ends_work(
    registry, options, models, tmp_path
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "await rev.send('T1')\n"
            "await rev.cancel()\n"
            "keep = self.jobs.launch(Subagent(), name='keep')\n"
            "await keep.send('T2')\n"
            "sh = self.jobs.launch(Shell(), name='sh')\n"
            "self.v.bash = int((await sh.send('echo $$')).stdout)\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["review"] = [work_done(report="T1")]
    models.scripts["keep"] = [work_done(report="T2")]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    statuses = {c.name: c.status for c in registry.children(root.id)}
    assert statuses["review"] == "closed" and statuses["keep"] != "closed"
    bash = root._agent.v.bash
    await registry.close(root.id)
    statuses = {c.name: c.status for c in registry.children(root.id)}
    assert statuses["keep"] == "closed"
    import os

    with pytest.raises(ProcessLookupError):
        for _ in range(100):
            os.kill(bash, 0)
            await asyncio.sleep(0.05)


async def test_shell_starts_in_the_workspace(registry, options, models):
    models.scripts[None] = [
        cell(
            "assert (await Shell().run('pwd')).stdout.strip() == str(self.cwd)\n"
            "r = await self.jobs.launch(Shell().run('pwd'), name='where')\n"
            "assert r.stdout.strip() == str(self.cwd)\n"
            "sh = self.jobs.launch(Shell(), name='sh')\n"
            "await sh.send('mkdir -p sub && cd sub')\n"
            "self.v.summary = self.jobs.get('sh').summary\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    assert root._agent.v.summary == f"idle · in {options.workspace.resolve() / 'sub'}"
    status = root._agent._status()
    assert f"sh · idle · in {options.workspace.resolve() / 'sub'}" in status
    assert "Cancel: await self.jobs.get('<name>').cancel()" in status


async def test_unattended_done_waits_for_running_work(registry, options, models):
    options = options.model_copy(update={"turn_method": "handle_batch"})
    result = WORK_RESULT.format(e="ok", r="R")
    models.scripts[None] = [
        cell(
            "self.jobs.launch(asyncio.sleep(30), name='nap')\n"
            "self.jobs.launch(Shell(), name='sh')\n" + result
        ),
        cell("await self.jobs.get('nap').cancel()\n" + result),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("task"), TIMEOUT)
    assert outcome.result.report == "R"
    feedback = _prompt(models, None, 1)
    assert "Done while work still runs: \\'nap\\'." in feedback  # the idle shell may stay


async def test_interactive_done_may_leave_background_work(registry, options, models):
    models.scripts[None] = [
        cell(
            "self.jobs.launch(asyncio.sleep(30), name='nap')\n"
            "return_result(Done(explanation='left running'))"
        )
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="left running")
    assert [j.name for j in root._agent.jobs.running()] == ["nap"]
    assert root._agent._work_records == [{"id": 1, "name": "nap", "kind": "await"}]
    status = root._agent._status()
    assert "Running:" in status and "nap · " in status


async def test_legacy_delegation_is_not_offered(registry, options):
    root = await registry.create(options)
    agent = root._agent
    for method in (agent.delegate, agent.spawn):
        with pytest.raises(AttributeError, match="Subagent"):
            await method("x", "y", name="x")


# ---- prompt, visibility, status, restore ----------------------------------------


@pytest.fixture
def _no_installed_skills(monkeypatch, tmp_path):
    monkeypatch.setattr("nooa.skill_registry.entry_points", lambda *, group: [])
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


MODES = {
    "handle": "interactive",
    "handle_batch": "unattended",
    "handle_work": "child",
    "handle_orchestrate": "orchestrator",
}


@pytest.mark.parametrize("method", list(MODES))
async def test_first_call_offers_one_work_vocabulary(
    method, registry, options, models, _no_installed_skills
):
    code = (
        "return_result(Done(explanation='x'))"
        if method == "handle"
        else WORK_RESULT.format(e="x", r="y")
    )
    models.scripts[None] = [cell(code)]
    root = await registry.create(options.model_copy(update={"turn_method": method}))
    await asyncio.wait_for(root.prompt("hello"), TIMEOUT)
    call = models.llms[None].calls[0]
    tools = "\n".join(str(getattr(tool, "description", tool)) for tool in call.tools or ())
    assert "Run commands and child agents as `self.jobs` work" in tools
    messages = "\n".join(str(m.get("content", "")) for m in call.messages)
    rendered = messages + "\n" + tools  # what the model reads: the messages and the tool
    for wanted in (
        "self.jobs.launch",
        "class WorkManager",
        "class Job:",
        "class JobEvent",
        "class Shell",
        "class CommandResult",
        "class Subagent",
        "class Question",
        "class Files",
        "stdout_path",
        'notification["jobs"]',
        "<status ",
        f"(mode: {MODES[method]})",
    ):
        assert wanted in rendered, wanted
    for legacy in (
        "self.delegate",
        "self.spawn",
        "ChildRef",
        "ChildResult",
        "ChildMessage",
        "delegates",
        "self.jobs.run",
        "JobsApi",
        "JobInfo",
        "self.shell",
        "ShellTools",
        "nobody reads ``self.message()``",
        "ActivityShellTools",
        "DepthLimitError",
        "producers",
        "<queues ",
        "def running(",
        "def records(",
        "status_source",
        "doc(self.delegate)",
        "_AgentJobs",
    ):
        assert legacy not in rendered, legacy


def test_turn_docs_are_short_and_name_their_mode():
    from nooa_atom.agent.unified import UnifiedAtomAgent

    for method, mode in (
        ("handle", "interactive"),
        ("handle_batch", "unattended"),
        ("handle_work", "child"),
        ("handle_orchestrate", "orchestrator"),
    ):
        text = getattr(UnifiedAtomAgent, method).__doc__
        assert len(text) <= 300, (method, len(text))
        assert f"(mode: {mode})" in text
        assert f"- {mode}" in UnifiedAtomAgent.__doc__ or mode in UnifiedAtomAgent.__doc__


def test_cells_see_only_the_unified_modules_names():
    import nooa_atom.agent.unified as unified

    from nooa.agentdoc.visibility import filter_module_globals

    names = set(filter_module_globals(unified))
    assert {"Shell", "Subagent", "Job", "JobEvent", "Question", "Todo", "TaskResult"} <= names
    assert not names & {"ChildRef", "ChildResult", "JobsApi", "ActivityShellTools", "AtomAgent"}


async def _agent(tmp_path):
    from nooa_atom.agent.unified import UnifiedAtomAgent

    from nooa.unifiedllm import FakeLLMClient

    return UnifiedAtomAgent(
        llm=FakeLLMClient([]), cwd=tmp_path, libs_dir=tmp_path / "libs", skills_dirs=[]
    )


async def test_status_counts_unread_by_sender_never_by_value(tmp_path):
    from nooa_atom.agent.work_manager import JobEvent

    agent = await _agent(tmp_path)
    try:
        agent._jobs_in.put(JobEvent("tests", 1, "done", "SECRET-OUTPUT"))
        agent._jobs_in.put(JobEvent("review#1", 2, "done", Question("SECRET-Q?")))
        agent._user_messages_in.put("SECRET-USER-TEXT")
        status = agent._status()
        assert "jobs: 2 (from tests, review#1)" in status
        assert "user_messages: 1" in status
        assert "SECRET" not in status
        assert "Read now: await self.user_messages.get() · next turn's notification" in status
        job = agent.jobs.launch(asyncio.sleep(30), name="nap")
        status = agent._status()
        assert "Running:\n  • nap · 0s" in status
        await job.cancel()
        # A shell's running request shows the command, busy, and its last confirmed cwd.
        (tmp_path / "sub").mkdir()
        sh = agent.jobs.launch(Shell(tmp_path), name="sh")
        await sh.send("cd sub")
        req = sh.send("sleep 5")
        for _ in range(100):
            if "running" in sh.summary:
                break
            await asyncio.sleep(0.02)
        status = agent._status()
        assert f"sh#2 · 0s · running $ sleep 5 · was in {tmp_path.resolve() / 'sub'}" in status
        assert "  • sh · " not in status  # listed once, by its request
        await req.cancel()
        assert f"sh · idle · in {tmp_path.resolve() / 'sub'}" in agent._status()
    finally:
        await agent.aclose()


async def test_after_restore_reports_lost_jobs_once(tmp_path):
    agent = await _agent(tmp_path)
    try:
        job = agent.jobs.launch(asyncio.sleep(30), name="nap")
        records = list(agent._work_records)
        assert records == [{"id": 1, "name": "nap", "kind": "await"}]
        await job.cancel()
        assert agent._work_records == []
        agent._work_records = records  # as a snapshot taken while it ran restores it
        agent.after_restore()
        [event] = agent._jobs_in.snapshot()
        assert (event.name, event.kind) == ("nap", "failed")
        assert event.value.startswith("lost:")
        agent.after_restore()
        assert agent._jobs_in.qsize() == 1
        assert "status" in agent.context and "queues" not in agent.context
    finally:
        await agent.aclose()


def test_legacy_atom_agent_is_unchanged():
    """The experiment hides members on its own class only."""
    from nooa_atom.agent.agent import AtomAgent

    from nooa.agentdoc import doc

    legacy = doc(AtomAgent, concise=True)
    for name in ("shell:", "delegate(", "spawn(", "jobs: JobsApi"):
        assert name in legacy, name


async def test_reserved_names_and_snapshots(tmp_path):
    from nooa.storage.json_snapshot import snapshot_to_json

    agent = await _agent(tmp_path)
    try:
        for name in ("jobs", "user_messages", "user", "delegates"):
            with pytest.raises(ValueError, match="names a channel"):
                agent.jobs.launch(asyncio.sleep(1), name=name)
        job = agent.jobs.launch(asyncio.sleep(30), name="nap")
        agent.v.job = job
        agent.v.shell = Shell()
        agent.v.names = ["nap"]
        data = snapshot_to_json(agent)  # live work is left out, not an error
        text = str(data)
        assert "nap" in text and "Job(" not in text and "work_manager" not in text
        await job.cancel()
    finally:
        await agent.aclose()


# ---- Subagent routing, against a fake session port -------------------------------


class _FakeChild:
    def __init__(self, stops: bool) -> None:
        self.stops = stops
        self.cancelled = 0

    async def cancel(self, *, by: str) -> bool:
        self.cancelled += 1
        return self.stops


class _FakeRegistry:
    def __init__(self, child: _FakeChild) -> None:
        self.child = child

    def get(self, child_id: str):
        return self.child


class _FakePort:
    """Enough of a SessionPort for a kept Subagent: one child, its route captured."""

    path = "/root"

    def __init__(self, *, stops: bool) -> None:
        self.child = _FakeChild(stops)
        self._registry = _FakeRegistry(self.child)
        self.route = None
        self.sent: list[str] = []
        self.closed: list[str] = []

    def check_child_name(self, name: str) -> None:
        return None

    async def delegate(self, description, prompt, *, name, model, kept, todo, route):
        from types import SimpleNamespace

        self.route = route
        return SimpleNamespace(id="c1", name=name)

    async def send_child(self, child_id: str, item, *, channel: str):
        self.sent.append(item)

    async def close_child(self, child_id: str) -> None:
        self.closed.append(child_id)


def _failed(error: str):
    from nooa_atom.session.items import ChildFailed, ChildRef

    ref = ChildRef(id="c1", name="review", path="/root/review", depth=1, status="idle")
    return ChildFailed(child=ref, error=error)


async def _armed(port: _FakePort, rounds: int = 200) -> None:
    for _ in range(rounds):
        if port.route is not None and port.route.future is not None:
            return
        await asyncio.sleep(0)
    raise AssertionError("the request never armed its route")


@pytest.fixture
async def fake_work(tmp_path):
    from nooa_atom.agent.work_manager import WorkManager
    from nooa_atom.session.port import current_port

    from nooa.runtime.channels import QueueManager

    outbox = QueueManager().queue("jobs")
    manager = WorkManager(outbox, scratch=tmp_path)

    def start(*, stops: bool):
        from nooa_atom.agent.work_subagent import Subagent

        port = _FakePort(stops=stops)
        token = current_port.set(port)
        try:
            rev = manager.launch(Subagent("review"), name="review")
        finally:
            current_port.reset(token)
        return port, rev

    yield start, outbox
    await manager.aclose()


async def test_a_stale_drop_flag_never_swallows_the_next_requests_failure(fake_work):
    """A cancel that stopped no failing turn must not hide the next request's failure."""
    start, outbox = fake_work
    port, rev = start(stops=True)  # cancel() says it stopped a turn ...
    first = rev.send("T1")
    await _armed(port)
    await first.cancel()  # ... but no failure for that turn ever arrives
    assert first.status == "cancelled" and port.child.cancelled == 1
    second = rev.send("T2")
    await _armed(port)
    assert port.route.drop_failure is False
    port.route.outcome(_failed("real failure"))
    with pytest.raises(RuntimeError, match="real failure"):
        await second
    assert outbox.qsize() == 0


async def test_cancel_after_the_outcome_was_handed_over_drops_it(fake_work):
    """Pinned: the owner's cancel wins over an outcome its request had not read yet.

    The request ends ``cancelled`` with no event; the outcome is not re-delivered
    anywhere; the child stays open and takes the next request.
    """
    from types import SimpleNamespace

    start, outbox = fake_work
    port, rev = start(stops=False)  # the turn had already ended: nothing to stop
    first = rev.send("T1")
    await _armed(port)
    port.route.outcome(SimpleNamespace(report="LATE"))  # handed to the future ...
    await first.cancel()  # ... before the request task resumed
    assert first.status == "cancelled"
    assert outbox.qsize() == 0  # neither the result nor an event
    assert port.route.drop_failure is False and port.closed == []
    second = rev.send("T2")
    await _armed(port)
    port.route.outcome(SimpleNamespace(report="NEXT"))
    assert (await second).report == "NEXT"
    assert port.sent == ["T2"]  # the follow-up went to the same child
