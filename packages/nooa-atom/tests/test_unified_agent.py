# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The experimental ``AtomAgent``: shells and subagents as ``self.jobs`` work.

Real sessions (``SessionRegistry``), fake models: each session's model is
scripted by its name (``None`` for the root), and a ``Subagent`` child is
named after its job.
"""

import asyncio

import pytest
from atom_test_agents import AtomModels, ask, cell, done, work_done
from nooa_atom.agent.roles import ROLE_INSTRUCTIONS
from nooa_atom.agent.work_shell import CommandResult, Shell
from nooa_atom.agent.work_subagent import Question
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.storage.snapshot_vars import SnapshotVars

UNIFIED = "nooa_atom.agent:AtomAgent"
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
    for name in (None, "fix"):
        call = models.llms[name].calls[0]
        rendered = "\n".join(str(m.get("content", "")) for m in call.messages)
        self_doc = rendered.split("<self>", 1)[1].split("</self>", 1)[0]
        assert "write ``self.v.key = value``" in self_doc
        assert "read ``self.v.key``" in self_doc
        assert "delete ``del self.v.key``" in self_doc
        assert "Store serializable values" in self_doc
        assert "get(self, id_or_title: str) -> Todo" in rendered
        assert "Return API documentation as str; display with print(doc(obj))" in rendered
        assert "Print (return None): max_length=items/container" in rendered
        assert "def read(self, name: str)" not in rendered
        [tool] = call.tools
        assert tool.name == "python_cell"
        assert tool.description.count("Blocked imports:") == 1
        assert tool.description.count("Blocked calls by module") == 1
        assert "`subprocess`" in tool.description and "literal dunder names" in tool.description
        assert "Run commands and child agents as `self.jobs` work" in tool.description
        assert "self.shell" not in tool.description and "self.delegate" not in tool.description
        assert "Blocked imports:" not in rendered  # guidance is tool-local, not another block

    # Atom retains a hidden legacy shell for internal file/repo tools; never recommend it.
    from nooa.runtime.code_validator import UnifiedCodeValidator, ValidationContext, ValidationError
    from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES

    with pytest.raises(ValidationError) as error:
        UnifiedCodeValidator().validate(
            "import subprocess",
            ValidationContext(agent=root._agent, blocked_modules=DEFAULT_BLOCKED_MODULES),
        )
    assert "self.shell" not in str(error.value)
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


async def test_a_shell_timeout_delivers_a_result_and_wakes_the_waiting_turn(
    registry, options, models
):
    """Deadline -> JobEvent -> next turn: a command that never exits still ends the wait."""
    models.scripts[None] = [
        cell(
            "import inspect\n"
            "assert inspect.signature(Shell.run).parameters['timeout'].default == 1800.0\n"
            "self.jobs.launch(Shell().run('sleep 30; echo never', timeout=0.5), name='slow')\n"
            "return_result(Waiting(explanation='slow command running', on=['slow']))"
        ),
        cell(
            "[event] = notification['jobs']\n"
            "assert (event.name, event.kind) == ('slow', 'done'), event\n"
            "r = event.value\n"
            "assert isinstance(r, CommandResult), r\n"
            "assert (r.returncode, r.timed_out, r.stdout) == (124, True, ''), r\n"
            "return_result(Done(explanation='timed out as expected'))"
        ),
    ]
    root = await registry.create(options)
    outcome = await asyncio.wait_for(root.prompt("run it"), TIMEOUT)
    assert outcome == Done(explanation="timed out as expected")
    assert len(models.llms[None].calls) == 2  # the timeout's event woke the second turn
    assert root._agent.jobs.get("slow").status == "done"
    assert root._agent.jobs.running() == []


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


async def test_session_close_ends_shells_its_cells_started(registry, options, models):
    """A Shell kept as a cell local (not a job) and its background process end with the session."""
    import os

    models.scripts[None] = [
        cell(
            "sh = Shell()\n"
            "await sh.run('cd / && X=1')\n"
            "self.v.bg = int((await sh.run('sleep 300 & echo $!')).stdout)\n"
            "self.v.bash = int((await sh.run('echo $$')).stdout)\n"
            "self.v.seen = (await sh.run('echo $X $PWD')).stdout\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    agent = root._agent
    assert agent.v.seen == "1 /\n"  # one bash across run() calls
    pids = (agent.v.bg, agent.v.bash)
    os.kill(pids[0], 0)  # still running after the turn: the cell local keeps the shell
    await registry.close(root.id)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            for _ in range(100):
                os.kill(pid, 0)
                await asyncio.sleep(0.05)


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
    [record] = root._agent._work_records
    tag = record.pop("tag")
    assert record == {"id": 1, "name": "nap", "kind": "await", "what": "coroutine"}
    launched = root._agent.event_manager[tag]  # the cell that launched it, code and all
    assert "name='nap'" in launched.arguments["code"]
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
    prefix = str(call.messages[0].get("content", ""))
    assert "<workspace_state>" in prefix
    assert "files/repo/new Shells." in prefix
    assert "Python paths use cell cwd, not Shell cwd." in prefix
    assert "<workspace_state expr=" not in rendered
    assert "persistent vars" not in rendered
    assert prefix.count("class Todo(BaseModel):") == 1
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
        ROLE_INSTRUCTIONS[MODES[method]],
        "# Work: everything that runs is a job on self.jobs.",
        "documented in the python_cell_tools block",
        "class SkillManager",
        "def status(self) -> str:",
        "async def message(self, item",  # Job.message
        "Names are Python identifiers;",
        "class Throttle:",
        "progress=Throttle(seconds=5)",
        "<python_cell_state>\nRetained Python names: none; omitted: 0\n</python_cell_state>",
        "<python_cell_lifetime>",
        "Other generated methods are call-local.",
        "registered queue producers",
        'on=["mesh"]',
    ):
        assert wanted in rendered, wanted
    for legacy in (
        "<skills",
        "## Python cell state",
        "handle_batch",
        "handle_work",
        "handle_orchestrate",
        "includes method inputs",
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
        "def producers(",
        "class Producer",
        "<queues ",
        "def running(",
        "def records(",
        "status_source",
        "doc(self.delegate)",
        "_AgentJobs",
        "Special type indicating",  # the Any pseudo-doc
        "current_dir",
        "print(doc(self.jobs))",
        "async def send(self, command",  # adapters' send(): the manager's protocol
        "async def send(self, task",
        "async def message(self, text",  # Subagent's: the model uses Job.message
    ):
        assert legacy not in rendered, legacy
    for other, text in ROLE_INSTRUCTIONS.items():
        if other != MODES[method]:
            assert text not in rendered, other
    assert rendered.count(ROLE_INSTRUCTIONS[MODES[method]]) == 1
    assert root.info.turn_method == method  # the host's selection is recorded as is
    # The workflow examples live in the tools block only, not in the class doc.
    start = messages.index("<system_prompt")
    system_prompt = messages[start : messages.index("</system_prompt>", start)]
    assert "Shell().run(" not in system_prompt and "python_cell_tools block" in system_prompt
    assert messages.count("# Work: everything that runs is a job on self.jobs.") == 1
    question = rendered[rendered.index("class Question") :]
    question = question[: question.index("\n@dataclass") if "\n@dataclass" in question else 400]
    assert "id:" not in question


def test_one_turn_method_with_a_tiny_doc():
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.session.session import turn_method_of

    assert len(AtomAgent.handle.__doc__) <= 120
    assert "Modes" not in AtomAgent.__doc__ and "mode:" not in AtomAgent.__doc__
    for method in MODES:
        assert turn_method_of(AtomAgent, method) == "handle"
    assert turn_method_of(object(), "handle_batch") == "handle_batch"


async def test_the_other_turn_methods_are_not_entry_points(tmp_path):
    agent = await _agent(tmp_path)
    try:
        for name in ("handle_batch", "handle_work", "handle_orchestrate"):
            with pytest.raises(TypeError, match="one turn method"):
                await getattr(agent, name)({})
    finally:
        await agent.aclose()


def test_cells_see_only_the_unified_modules_names():
    import nooa_atom.agent.agent as unified

    from nooa.agentdoc.visibility import filter_module_globals

    names = set(filter_module_globals(unified))
    assert {"Shell", "Subagent", "Job", "JobEvent", "Question", "Todo", "TaskResult"} <= names
    assert not names & {"ChildRef", "ChildResult", "JobsApi", "ActivityShellTools", "AtomAgentBase"}


async def _agent(tmp_path):
    from nooa_atom.agent.agent import AtomAgent

    from nooa.unifiedllm import FakeLLMClient

    return AtomAgent(
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
        assert records == [{"id": 1, "name": "nap", "kind": "await", "what": "coroutine"}]
        await job.cancel()
        assert agent._work_records == []
        agent._work_records = records  # as a snapshot taken while it ran restores it
        agent.after_restore()
        [event] = agent._jobs_in.snapshot()
        assert (event.name, event.kind) == ("nap", "failed")
        assert event.value == (  # launched outside a cell: no cell pointer
            "lost: the session restarted while it ran (coroutine). Relaunch if still needed."
        )
        agent.after_restore()
        assert agent._jobs_in.qsize() == 1
        assert "status" in agent.context and "queues" not in agent.context
    finally:
        await agent.aclose()


async def test_lost_notice_names_the_launching_cell_and_shell_command(tmp_path):
    from nooa.strategies import codeact

    agent = await _agent(tmp_path)
    try:
        token = codeact._current_cell.set((agent, "23221"))  # as inside cell 23221
        try:
            job = agent.jobs.launch(Shell(tmp_path).run("sleep 5 # " + "x" * 300), name="tests")
            other = codeact._current_cell.set((object(), "7"))  # another agent's cell
            try:
                nap = agent.jobs.launch(asyncio.sleep(30), name="nap")
            finally:
                codeact._current_cell.reset(other)
        finally:
            codeact._current_cell.reset(token)
        for _ in range(100):
            if "cwd" in job.summary:
                break
            await asyncio.sleep(0.02)
        records = list(agent._work_records)
        assert [r.get("tag") for r in records] == ["23221", None]
        assert records[0]["what"] == "shell command" and "cwd" in records[0]["summary"]
        await job.cancel()
        await nap.cancel()
        agent._work_records = records
        agent.after_restore()
        tests, nap_event = agent._jobs_in.snapshot()
        assert tests.value.startswith(
            "lost: the session restarted while it ran (shell command launched in cell 23221: "
            'self.events["23221"]; $ sleep 5 # xxx'
        )
        assert f"… in {tmp_path.resolve()}). Relaunch if still needed." in tests.value
        assert len(tests.value) < 400
        assert nap_event.value == (
            "lost: the session restarted while it ran (coroutine). Relaunch if still needed."
        )
    finally:
        await agent.aclose()


def test_lost_notice_for_subagent_todo_and_bounds():
    from nooa_atom.agent.agent import _lost_notice, _work_label
    from nooa_atom.agent.work_subagent import Subagent
    from nooa_atom.tools.todo_list import Todo

    work = Subagent("fix").run(Todo(id="4861f9a3", title="t"))
    try:
        assert _work_label(work, "await") == {"what": "subagent", "todo": "4861f9a3"}
    finally:
        work.close()
    text = Subagent("fix").run("plain text")
    try:
        assert _work_label(text, "await") == {"what": "subagent"}
    finally:
        text.close()
    assert _work_label(Shell(), "resource") == {"what": "shell resource"}
    notice = _lost_notice({"name": "fix", "kind": "await", "what": "subagent", "todo": "4861f9a3"})
    assert "(subagent; todo 4861f9a3)" in notice
    # Legacy records (id/name/kind only) and junk stay short; odd tags are dropped.
    assert "(iterator)" in _lost_notice({"id": 1, "name": "x", "kind": "iter"})
    huge = _lost_notice(
        {"what": "w" * 500, "tag": "1); drop", "todo": "t" * 500, "summary": "s" * 9999}
    )
    assert "self.events" not in huge and len(huge) < 320
    shell = _lost_notice({"what": "shell command", "summary": "$ pytest -q · cwd /repo"})
    assert shell == (
        "lost: the session restarted while it ran (shell command; $ pytest -q in /repo). "
        "Relaunch if still needed."
    )


def test_public_atom_agent_hides_the_base_surface():
    """The public AtomAgent hides the private base's delegation and JobsApi members."""
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.agent.base import AtomAgentBase

    from nooa.agentdoc import doc

    base = doc(AtomAgentBase, concise=True)
    for name in ("shell:", "delegate(", "spawn(", "jobs: JobsApi"):
        assert name in base, name
    public = doc(AtomAgent, concise=True)
    for name in ("delegate(", "spawn(", "jobs: JobsApi"):
        assert name not in public, name
    assert AtomAgent.__name__ == "AtomAgent" and issubclass(AtomAgent, AtomAgentBase)


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


# ---- Notes to a kept child: rev.message(), beside its requests --------------------


def _child_gate(n):
    """Child code: say it started, then hold its turn until the parent writes ``go<n>``."""
    return (
        f"await self.files.write_file('started{n}', 'x')\n"
        "for _ in range(400):\n"
        f"    if (await Shell().run('test -f go{n}')).returncode == 0:\n"
        "        break\n"
        "    await asyncio.sleep(0.05)\n"
        "else:\n"
        "    raise AssertionError('never released')\n"
    )


def _parent_wait(n):
    """Parent code: wait until the child's turn reached gate ``n``."""
    return (
        "for _ in range(400):\n"
        f"    if (await Shell().run('test -f started{n}')).returncode == 0:\n"
        "        break\n"
        "    await asyncio.sleep(0.05)\n"
        "else:\n"
        "    raise AssertionError('the child never started')\n"
    )


_SEE_ITEM = (
    "[event] = notification['jobs']\n"
    "assert (event.name, event.kind) == ('review', 'item'), event\n"
    "value = event.value\n"
    "self.v.item = value if isinstance(value, str) else getattr(value, 'report', None) or value.text\n"
    "self.v.item_type = type(value).__name__\n"
    "return_result(Done(explanation='saw the item'))"
)


async def test_a_note_reaches_the_running_turn_before_its_request_completes(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "req = rev.send('TASK-1')\n" + _parent_wait(1) + "await rev.message('NOTE-MID')\n"
            "assert req.status == 'running'\n"
            "await Shell().run('touch go1')\n"
            "self.v.report = (await req).report\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["review"] = [
        cell(_child_gate(1) + "print('held')"),
        cell(WORK_RESULT.format(e="done", r="R-1")),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    await _settle(root)
    assert root._agent.v.report == "R-1"
    assert "NOTE-MID" not in _prompt(models, "review", 0)
    assert "NOTE-MID" in _prompt(models, "review", 1)  # read inside the same turn
    assert len(models.llms["review"].calls) == 2  # no extra turn, no extra outcome
    assert root._agent._jobs_in.qsize() == 0 and len(models.llms[None].calls) == 1
    assert _delegates_empty(root)


async def test_a_note_that_misses_the_turn_starts_one_whose_result_is_an_item(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "req = rev.send('TASK-1')\n" + _parent_wait(1) + "await rev.message('NOTE-SPILL')\n"
            "await Shell().run('touch go1')\n"
            "self.v.report = (await req).report\n"
            "return_result(Done(explanation='ok'))"
        ),
        cell(_SEE_ITEM),
    ]
    models.scripts["review"] = [
        # One cell: the turn ends with no model call left to read the note.
        cell(_child_gate(1) + WORK_RESULT.format(e="done", r="R-1")),
        cell("assert 'NOTE-SPILL' in str(notification)\n" + WORK_RESULT.format(e="n", r="R-NOTE")),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    await _settle(root)
    agent = root._agent
    assert agent.v.report == "R-1"  # the request kept its own turn's result
    assert agent.v.item == "R-NOTE"  # the note's turn reported as an item of the resource
    assert _delegates_empty(root)


async def test_a_request_queued_behind_a_note_turn_gets_its_own_result(registry, options, models):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "assert (await rev.send('TASK-1')).report == 'R-1'\n"
            "await rev.message('NOTE-IDLE')\n" + _parent_wait(2) + "req = rev.send('TASK-2')\n"
            "await asyncio.sleep(0.3)\n"
            "await Shell().run('touch go2')\n"
            "self.v.report = (await req).report\n"
            "return_result(Done(explanation='ok'))"
        ),
        cell(_SEE_ITEM),
    ]
    models.scripts["review"] = [
        work_done("one", report="R-1"),
        cell(
            "assert 'NOTE-IDLE' in str(notification)\n"
            + _child_gate(2)
            + WORK_RESULT.format(e="n", r="R-NOTE")
        ),
        cell("assert 'TASK-2' in str(notification)\n" + WORK_RESULT.format(e="two", r="R-2")),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    await _settle(root)
    agent = root._agent
    assert agent.v.report == "R-2"  # not the note turn's R-NOTE
    assert agent.v.item == "R-NOTE"
    assert "TASK-2" not in _prompt(models, "review", 1)  # queued, never steered into it
    assert _delegates_empty(root)


async def test_cancelling_a_queued_request_takes_its_input_back_and_spares_the_note_turn(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "await rev.send('TASK-1')\n"
            "await rev.message('NOTE-C')\n" + _parent_wait(2) + "req = rev.send('TASK-X')\n"
            "await asyncio.sleep(0.3)\n"
            "await req.cancel()\n"
            "assert req.status == 'cancelled'\n"
            "await Shell().run('touch go2')\n"
            "self.v.report = (await rev.send('TASK-3')).report\n"
            "return_result(Done(explanation='ok'))"
        ),
        cell(_SEE_ITEM),
    ]
    models.scripts["review"] = [
        work_done("one", report="R-1"),
        cell(_child_gate(2) + WORK_RESULT.format(e="n", r="R-NOTE")),
        cell(
            "assert 'TASK-X' not in str(notification) and 'TASK-3' in str(notification)\n"
            + WORK_RESULT.format(e="three", r="R-3")
        ),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    await _settle(root)
    agent = root._agent
    assert agent.v.report == "R-3"
    assert agent.v.item == "R-NOTE"  # the note's turn ran to its end
    assert all("TASK-X" not in str(c.messages) for c in models.llms["review"].calls)
    assert _delegates_empty(root)


async def test_note_lifecycle_errors_and_questions_are_answered_with_send(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "errors = []\n"
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "try:\n"
            "    await rev.message('too early')\n"
            "except RuntimeError as exc:\n"
            "    errors.append(str(exc))\n"
            "q = await rev.send('TASK')\n"
            "assert isinstance(q, Question)\n"
            "try:\n"
            "    await rev.message('not an answer')\n"
            "except RuntimeError as exc:\n"
            "    errors.append(str(exc))\n"
            "self.v.report = (await rev.send('ANSWER-7')).report\n"
            "try:\n"
            "    await self.jobs.launch(asyncio.sleep(0), name='nap').message('x')\n"
            "except TypeError as exc:\n"
            "    errors.append(str(exc))\n"
            "await rev.cancel()\n"
            "try:\n"
            "    await rev.message('too late')\n"
            "except RuntimeError as exc:\n"
            "    errors.append(str(exc))\n"
            "self.v.errors = errors\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["review"] = [
        ask("which file?"),
        cell("assert 'ANSWER-7' in str(notification)\n" + WORK_RESULT.format(e="a", r="R-A")),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    agent = root._agent
    assert agent.v.report == "R-A"
    early, asking, plain, late = agent.v.errors
    assert "send() it a task first" in early
    assert "which file?" in asking and "answer it with send()" in asking
    assert "only a launched resource has message()" in plain
    assert "review is" in late
    assert all("not an answer" not in str(c.messages) for c in models.llms["review"].calls)
    assert _delegates_empty(root)


async def test_a_failed_note_turn_is_a_visible_item(registry, options, models):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "await rev.send('TASK')\n"
            "await rev.message('NOTE-DOOM')\n"
            "return_result(Done(explanation='ok'))"
        ),
        cell(_SEE_ITEM),
    ]
    models.scripts["review"] = [work_done(report="R-1")]  # the note's turn has no script: fails
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    for _ in range(1200):  # the model client retries (with backoff) before the turn fails
        if "item_type" in root._agent.vars:
            break
        await asyncio.sleep(0.05)
    agent = root._agent
    assert agent.v.item_type == "str"  # plain text: kept by a snapshot as is
    assert agent.v.item.startswith("subagent review failed:")
    restored = SnapshotVars.from_data(agent.vars.to_json_data())
    assert restored["item"] == agent.v.item
    assert _delegates_empty(root)


async def test_a_request_queued_behind_a_note_turn_that_asks_still_gets_its_result(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent(), name='review')\n"
            "await rev.send('TASK-1')\n"
            "await rev.message('NOTE-Q')\n" + _parent_wait(2) + "req = rev.send('TASK-2')\n"
            "await asyncio.sleep(0.3)\n"
            "await Shell().run('touch go2')\n"
            "self.v.report = (await req).report\n"
            "self.v.summary = self.jobs.get('review').summary\n"
            "return_result(Done(explanation='ok'))"
        ),
        cell(_SEE_ITEM),
    ]
    models.scripts["review"] = [
        work_done("one", report="R-1"),
        cell(_child_gate(2) + "print('held')"),
        ask("NOTE-QUESTION which one?"),  # the note's turn ends with a question
        cell("assert 'TASK-2' in str(notification)\n" + WORK_RESULT.format(e="two", r="R-2")),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    await _settle(root)
    agent = root._agent
    assert agent.v.report == "R-2"  # the queued request was not stranded by the question
    assert (agent.v.item_type, agent.v.item) == ("Question", "NOTE-QUESTION which one?")
    assert "asked a question" not in agent.v.summary  # its own turn ended the question
    assert _delegates_empty(root)


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

    async def send_child(self, child_id: str, item, *, channel: str, steer: bool = True):
        from types import SimpleNamespace

        self.sent.append(item)
        return SimpleNamespace(item_id=f"item{len(self.sent)}")

    def withdraw_child(self, child_id: str, receipt) -> bool:
        return False  # its turn already started

    async def close_child(self, child_id: str) -> None:
        self.closed.append(child_id)


def _failed(error: str):
    from nooa_atom.session.items import ChildFailed, ChildRef

    ref = ChildRef(id="c1", name="review", path="/root/review", depth=1, status="idle")
    return ChildFailed(child=ref, error=error)


async def _armed(port: _FakePort, rounds: int = 200) -> None:
    """Wait for the request, then start the turn that carries its input."""
    for _ in range(rounds):
        route = port.route
        if route is not None and route.future is not None and (route.armed or route.want):
            route.turn_started([route.want] if route.want else [])
            assert route.armed
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


def test_command_output_is_shown_cut_but_kept_whole():
    """Notifications pprint a CommandResult's streams cut; the object keeps them whole."""
    from nooa_atom.agent.work_manager import JobEvent

    from nooa.agentdoc import pformat

    result = CommandResult(0, False, "o" * 9000, "e" * 5000, "/w", "/w/out", None)
    shown = pformat(
        {"jobs": [JobEvent("tests", 1, "done", result)]},
        max_string=50_000,  # the turn's notification limit
        max_length=100,
        max_depth=10,
    )
    assert len(shown) < 4_500  # was 14,202 with both streams in full
    assert "stdout_path='/w/out'" in shown
    assert (len(result.stdout), len(result.stderr)) == (9000, 5000)


async def test_skills_are_documented_statically_and_activation_still_works(tmp_path):
    agent = await _agent(tmp_path)
    try:
        assert "skills" not in agent.context
        tools = str(agent.context["python_cell_tools"])
        assert "class SkillManager" in tools and "def activate(" in tools
        assert "shell" not in agent.skills.status().splitlines()[0]
        print(await agent.skills.activate(["shell"]))
        assert "shell" in agent.skills.status().splitlines()[0]
        assert "skills" not in agent.context  # activation does not bring the block back
        agent.after_restore()
        assert "skills" not in agent.context
    finally:
        await agent.aclose()


async def test_the_cell_state_block_is_one_line(tmp_path):
    from types import SimpleNamespace

    from nooa_atom.agent.agent import AtomCodeAct

    strategy = AtomCodeAct()
    call = SimpleNamespace(
        execution_locals={"r1": 1, "job": "x", "re": __import__("re"), "_hidden": 2},
        session_locals=None,
        bound_parameters=lambda: {"notification": {}},
    )
    text = await strategy.python_cell_state_context(SimpleNamespace(current_call=call))
    assert text == "Cell locals: job (str), notification (dict), r1 (int); imports: re → re"
    empty = SimpleNamespace(execution_locals=None, session_locals=None, bound_parameters=dict)
    assert (
        await strategy.python_cell_state_context(SimpleNamespace(current_call=empty))
        == "Cell locals: none"
    )
