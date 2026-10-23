# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental Atom agent with one work vocabulary: ``self.jobs`` runs shells and subagents.

``UnifiedAtomAgent`` is selectable (``agent_spec="nooa_atom.agent.unified:UnifiedAtomAgent"``)
and leaves ``AtomAgent`` unchanged. It keeps the Atom turn methods and their
return contracts, the workspace tools (files, repo, todos, skills, ``self.v``)
and the session plumbing, and replaces background jobs and child sessions
with the generic ``WorkManager`` and its ``Shell`` and ``Subagent`` work.

Its cells see only this module's names (``UnifiedCodeAct``), not the legacy
names of ``nooa_atom.agent.agent`` that inheritance would otherwise bring in.
"""

from __future__ import annotations

from typing import Annotated, Any

# Cells see this module's names: the model builds Done/TaskResult/Todo, Shell and
# Subagent work, and Context/strategy/PredictStrategy for skill and method writing.
from nooa import Context, hidden, strategy
from nooa.agentdoc import doc, spec
from nooa.interactive import Done, NeedInput, Waiting
from nooa.strategies import CodeActV2, PredictStrategy  # noqa: F401
from nooa.tools import Todo, TodoManager  # noqa: F401
from nooa_atom.agent.work_manager import Job, JobEvent, WorkManager
from nooa_atom.agent.work_shell import CommandResult, Shell
from nooa_atom.agent.work_subagent import Question, Subagent
from nooa_atom.session.items import TaskResult  # noqa: F401
from nooa_atom.tools.repo_tools import RepoTools

with hidden:
    import inspect
    import time
    import types

    from nooa import no_trace
    from nooa.config import CodeActConfig
    from nooa.interactive import InteractiveAgent, require_task_result
    from nooa.runtime.channels import ChannelReader
    from nooa.storage.markers import nosnapshot
    from nooa.strategy_validation import InvariantError
    from nooa.tools.shell_tools import FileWrite, Match
    from nooa_atom.agent import work_shell
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.agent.conditions import PERSON_NAMES, require_result

__all__ = ["Files", "UnifiedAtomAgent"]


class Files:
    """Read and edit workspace files (paths relative to the workspace); run commands with ``Shell``."""

    __nosnapshot__ = True

    def __init__(self, shell: Any) -> None:
        self._shell = shell

    async def read(self, path: str, lines: tuple[int, int] | None = None) -> Match:
        """Read a file, or a 1-indexed inclusive line range, as a ``Match``."""
        return await self._shell.read(path, lines)

    async def replace(self, path: str, old: str, new: str) -> FileWrite:
        """Replace the one occurrence of ``old`` in the file with ``new``."""
        return await self._shell.replace(path, old, new)

    async def replace_match(self, match: Match, new: str) -> FileWrite:
        """Rewrite the lines a ``Match`` covers (from ``read`` or ``self.repo``)."""
        return await self._shell.replace_match(match, new)

    async def write_file(self, path: str, content: str) -> FileWrite:
        """Create or overwrite a file."""
        return await self._shell.write_file(path, content)


@hidden
class _AgentJobs(WorkManager):
    """The agent's ``self.jobs``: also refuses names a ``Waiting`` already means."""

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        super().__init__(agent._jobs_in, **kwargs)
        self._agent = agent

    def launch(self, work: Any, *, name: str | None = None) -> Job:
        if name is not None and (
            name in PERSON_NAMES or name in self._agent.queue_manager.channels()
        ):
            if inspect.iscoroutine(work):
                work.close()
            raise ValueError(f"{name!r} names a channel or the person; pick another job name")
        return super().launch(work, name=name)


@hidden
class UnifiedCodeAct(CodeActV2):
    """CodeActV2 whose cells see the agent's own module only (no inherited legacy names).

    It also makes the agent's workspace the default ``cwd`` of ``Shell()``
    for the turn (and the jobs it launches).
    """

    def _extract_module_context(
        self, agent_module: types.ModuleType, agent: Any | None = None
    ) -> dict[str, Any]:
        context = super()._extract_module_context(agent_module, agent=None)
        if agent is not None:
            self._import_dynamic_classes(agent, agent_module, context)
        return context

    async def python_cell_context(self, runtime: Any) -> str:
        agent_module = inspect.getmodule(type(runtime.agent))
        if agent_module is None:
            return ""
        context = self._extract_module_context(agent_module, agent=runtime.agent)
        return self._render_execution_context_stub(
            context, {agent_module.__name__}, self.config.restrictions.blocked_modules
        )

    def _build_builtins(self, runtime: Any, call: Any) -> dict[str, Any]:
        cwd = getattr(runtime.agent, "cwd", None)
        if cwd is not None:
            work_shell.default_cwd.set(str(cwd))
        return super()._build_builtins(runtime, call)


# ---- turn conditions ---------------------------------------------------------


@hidden
def _active(job: Job) -> bool:
    """Running work that will deliver: not a resource with nothing in flight."""
    if job._kind != "resource":
        return True
    return any(request._status == "running" for request in job._requests)


@hidden
def _deliverable(agent: Any) -> set[str]:
    """Names a ``Waiting`` may name: what will deliver, or already has (unread)."""
    names: set[str] = set()
    jobs: WorkManager = agent.jobs
    for job in jobs.running():
        if _active(job):
            names.add(job.name)
            names.add("jobs")
    channels = agent.queue_manager.channels()
    unread = channels.get("jobs")
    if unread is not None and unread.qsize():
        names.add("jobs")
        for event in unread.snapshot():
            name = getattr(event, "name", None)
            if isinstance(name, str):
                names.update({name, name.split("#", 1)[0]})
    names |= {name for name, channel in channels.items() if channel.qsize() > 0}
    for handle in agent.queue_manager.running_handles():
        names.update(key for key in (handle.name, handle.label) if key)
    return names - PERSON_NAMES


@hidden
def _check_waiting(agent: Any, result: Any, *, fix: str) -> None:
    if not isinstance(result, Waiting):
        return
    ok = _deliverable(agent)
    unmet = [name for name in result.on if name not in ok]
    if not unmet:
        return
    reasons = []
    for name in unmet:
        job = agent.jobs.get(name) if isinstance(name, str) else None
        if name in PERSON_NAMES:
            reasons.append(f"Waiting on {name!r}, but the person is not a job.")
        elif job is not None and job.status == "running":
            reasons.append(
                f"{name!r} is an idle resource: nothing arrives until you send it work "
                f"(req = self.jobs.get({name!r}).send(...)); then wait on the request."
            )
        elif job is not None:
            reasons.append(
                f"{name!r} already {job.status}: read self.jobs.get({name!r}) (.result or .error)."
            )
        else:
            reasons.append(f"Waiting on {name!r}, but no running job has that name.")
    shown = ", ".join(repr(name) for name in sorted(ok)) or "nothing"
    raise InvariantError(
        " ".join(reasons) + f" Waiting is only for work that will deliver. {fix} "
        f"Can deliver now: {shown}."
    )


@hidden
def require_deliverable_wait(agent: Any, result: Any, call: Any) -> None:
    """Interactive turns: ``Waiting.on`` names only work that will deliver."""
    _check_waiting(agent, result, fix="If the work is done, return Done; to ask, NeedInput.")


@hidden
def require_deliverable_wait_unattended(agent: Any, result: Any, call: Any) -> None:
    """Unattended turns: the same check, with the unattended fix."""
    _check_waiting(agent, result, fix="Launch or send the work first, or finish with Done.")


@hidden
def require_settled_work(agent: Any, result: Any, call: Any) -> None:
    """Unattended turns: ``Done`` leaves no running command, task or request behind.

    Idle resources (an open shell or kept subagent with nothing in flight)
    may stay; the session closes them.
    """
    if not isinstance(result, Done):
        return
    live = [job.name for job in agent.jobs.running() if _active(job)]
    if live:
        names = ", ".join(repr(name) for name in live)
        raise InvariantError(
            f"Done while work still runs: {names}. Await it, cancel it "
            "(await self.jobs.get(name).cancel()), or return Waiting(on=[...]) until it "
            "delivers. Idle shells and subagents may stay open."
        )


@hidden
def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


@hidden
def _bounded(text: str, limit: int = 60) -> str:
    flat = "".join(c if c.isprintable() else "?" for c in str(text))
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# Shared repo-tool docs that name the legacy shell; this agent edits through
# self.files and its repo searches start in the workspace.
_DOC_FIXES: Annotated[tuple[tuple[str, str], ...], hidden] = (
    ("self.shell.replace_match()", "self.files.replace_match()"),
    ("as ShellTools ``Match`` anchors", "as ``Match`` anchors (edit with ``self.files``)"),
    ("defaults to the shell's current directory", "defaults to the workspace"),
)


@hidden
def _tools_doc(text: str) -> str:
    """The tools block for this agent: the shared docs, with only advertised paths."""
    for old, new in _DOC_FIXES:
        text = text.replace(old, new)
    return text


_CONTEXT: Annotated[dict[str, Any], hidden] = {
    "state": None,
    "execution_context": None,
    "self": Context(expr="doc(type(self), concise=True)", prefix=True),
}
_NOTIFICATION: Annotated[Any, hidden] = spec(max_string=50_000, max_length=100, max_depth=10)
_REMOVED: Annotated[str, hidden] = (
    "{} is not part of this agent: run a child with Subagent "
    '(await Subagent().run(todo), or self.jobs.launch(Subagent(), name="..."))'
)


class UnifiedAtomAgent(AtomAgent):
    """You are a careful agent working in one local repository.

    Read the repository instructions and the code you touch before editing.
    Make small, focused edits and preserve unrelated changes in the worktree.
    Verify with real commands before you claim anything; say what you did not
    check. Follow the newest steering from your user or parent within
    higher-priority rules. An explicit task authorization (for example to
    install or provision what the task needs) is approval; otherwise ask
    before destructive or outside-the-workspace actions.

    ``self.files`` reads and edits files, ``self.repo`` finds definitions and
    references, ``self.todo`` tracks multi-step work (activate the current
    Todo; comment material findings and checks), and ``self.v`` keeps
    durable facts (cell locals last one method call). Everything else that
    runs is a job on ``self.jobs``::

        r = await Shell().run("pytest -q tests/test_x.py")   # a short command
        self.jobs.launch(Shell().run("pytest -q"), name="tests")   # long: arrives later
        sh = self.jobs.launch(Shell(), name="sh")   # a persistent bash session
        r = await sh.send("cd src && make")
        done = await Subagent("fix parser").run(todo)   # one-shot child -> TaskResult
        rev = self.jobs.launch(Subagent("review"), name="review")   # kept child
        req = rev.send(todo)   # await req -> TaskResult or Question

    A subagent sees only its brief: a Todo whose description carries the
    requirements, constraints and acceptance checks. Verify what it reports;
    run one editing subagent at a time (they share this checkout).

    ``notification`` maps a channel (``user_messages``, ``jobs``,
    ``system_messages``...) to new items; ``jobs`` holds the ``JobEvent``s of
    work nobody awaited. Never predict a pending job's result. End every
    turn with exactly one ``return_result(...)``: ``Done(explanation=...)``,
    ``NeedInput(question=...)`` only when a person must answer, or
    ``Waiting(explanation=..., on=[job names])`` while launched work runs.

    Modes (each turn method names its own):

    - interactive: send brief progress notes and the reply with
      ``self.message()`` (complete Markdown: outcome, evidence, risks), then
      ``Done``; background jobs may keep running.
    - unattended, child, orchestrator: the result is what gets read. End
      with ``Done(explanation=..., result=TaskResult(solution_description=...,
      evidence=..., how_to_verify=..., report=...))``, also when blocked (say
      what blocked you). First await or cancel your running jobs; idle
      shells and subagents may stay. Leave the tree clean.
    - child: your parent's brief is the task; its first line says whether you
      may ask (``NeedInput``). ``self.message()`` reaches your parent as a
      short note: no routine progress notes, the report goes in the result.
    - orchestrator: split the task into Todo briefs for subagents rather than
      implementing it; write runnable acceptance checks first, run them
      yourself (a report is not proof), send failures back with ``send``,
      and loop until the task is met or another round is not worth it.
    """

    jobs: Annotated[WorkManager, nosnapshot]
    files: Annotated[Files, nosnapshot]
    # Small records of the running jobs, saved with the snapshot so a resumed
    # session reports the ones that did not survive (``after_restore``).
    _work_records: Annotated[list, hidden]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The legacy JobsApi never started anything; drop it and its cancel hint.
        self.queue_manager._cancel_hints.pop("jobs", None)
        self._work_records = []
        self.jobs = _AgentJobs(self, on_change=self._record_work)
        self.files = Files(self.shell)
        # Commands run as Shell work; the legacy shell skill stays loaded for
        # self.files and self.repo but is not offered (``/skills`` can still).
        self.skills.registry.deactivate(["nemo.shell"])
        self.context["python_cell_tools"] = Context(
            _tools_doc(
                doc(RepoTools, Files, TodoManager, concise=True)
                + "\n"
                + doc(WorkManager, Job, JobEvent, Shell, CommandResult, Subagent, Question)
            ),
            prefix=True,
        )
        self._status_block()

    @hidden
    @no_trace
    def _status_block(self) -> None:
        """The ``status`` block replaces ``queues`` (its legacy child and job lines)."""
        self.context.pop("queues", None)
        self.context["status"] = Context(expr="self._status()")

    @hidden
    @no_trace
    def _workspace_state_context(self) -> str:
        """Where work happens, without exposing stored values."""
        from html import escape

        def shown(path: object) -> str:
            return escape(_bounded(str(path), 160), quote=False)

        return (
            f"Workspace (Shell() starts here; self.files paths are relative to it): "
            f"{shown(self.cwd)}\n"
            f"Repository root (the boundary for repo searches): {shown(self.repo.root)}\n"
            f"`self.v`: {len(self.vars)} persistent vars — list: `print(dict(self.vars))`"
        )

    @hidden
    @no_trace
    def _record_work(self) -> None:
        self._work_records = self.jobs.records()

    @hidden
    @no_trace
    def _status(self) -> str:
        """Unread counts and senders, read hints, running and idle work, cancel hint."""
        now = time.monotonic()
        running: list[str] = []
        idle: list[str] = []
        for job in self.jobs.running():
            name = _bounded(job.name)
            summary = job.summary
            if job._kind == "resource":
                if not _active(job):
                    idle.append(f"{name} · {summary or 'open'}")
                continue  # its requests are listed instead, with its summary
            owner = job._owner
            if owner is not None and owner._status == "running":
                # The resource's own line (a shell: the command, busy, its last
                # confirmed cwd) says more than the request's.
                summary = owner.summary or summary
            parts = [name, _age(now - job._started), summary]
            running.append(" · ".join(part for part in parts if part))
        hints = {
            "user_messages": "await self.user_messages.get()",
            "jobs": "next turn's notification, or await the job",
        }
        for name in self.queue_manager.channels():
            reader = getattr(self, name, None)
            if name not in hints and isinstance(reader, ChannelReader) and name != "delegates":
                hints[name] = f"await self.{name}.get()"
        status = self.queue_manager.status(more_work=running, idle=idle, read_hints=hints)
        if running or idle:
            status += "\nCancel: await self.jobs.get('<name>').cancel()"
        return status

    # ---- legacy surface, hidden ------------------------------------------------

    @hidden
    async def delegate(self, *args: Any, **kwargs: Any) -> Any:
        raise AttributeError(_REMOVED.format("delegate()"))

    @hidden
    async def spawn(self, *args: Any, **kwargs: Any) -> Any:
        raise AttributeError(_REMOVED.format("spawn()"))

    @hidden
    def children(self) -> Any:
        return super().children()

    @hidden
    def child(self, name: str) -> Any:
        return super().child(name)

    @hidden
    @no_trace
    def get_summarization_status(self) -> dict[str, Any]:
        return super().get_summarization_status()

    def message(self, text: str, *, echo: bool = False) -> None:
        """Send a Markdown message to the user (a child's goes to its parent, cut at 1,000 chars).

        Each call renders as its own block, so make it a complete Markdown
        document. With ``echo=True`` it is also printed for you to see.
        """
        super().message(text, echo=echo)

    @hidden
    def after_restore(self) -> None:
        """Report, once, the jobs that were running when the saved session stopped."""
        from nooa_atom.workspace.options import drop_stale_memory_context

        InteractiveAgent.after_restore(self)
        self._status_block()
        drop_stale_memory_context(self)
        self.skills.after_restore()
        lost, self._work_records = list(self._work_records or []), []
        for record in lost:
            self._jobs_in.put(
                JobEvent(
                    str(record.get("name", "?")),
                    0,
                    "failed",
                    "lost: the session restarted while it ran; launch it again if needed",
                )
            )

    @hidden
    async def aclose(self) -> None:
        """Cancel this session's jobs (shells, subagents), then release the agent."""
        if self.turns.in_turn:
            raise RuntimeError("an executing turn cannot close its own agent")
        try:
            await self.jobs.aclose()
        finally:
            await super().aclose()

    # ---- turns -------------------------------------------------------------------

    @hidden
    @strategy(
        UnifiedCodeAct(
            config=CodeActConfig(cell_timeout=1800.0, postconditions=[require_deliverable_wait])
        ),
        context=_CONTEXT,
    )
    async def handle(
        self, notification: Annotated[dict[str, list[Any]], _NOTIFICATION]
    ) -> Done | NeedInput | Waiting:
        """Interactive turn (mode: interactive): handle the newest request and whatever else
        arrived in ``notification``, keeping the user informed with ``self.message()``."""
        ...

    @hidden
    @strategy(
        UnifiedCodeAct(
            config=CodeActConfig(
                cell_timeout=1800.0,
                postconditions=[
                    require_result,
                    require_deliverable_wait_unattended,
                    require_settled_work,
                ],
            )
        ),
        context=_CONTEXT,
    )
    async def handle_batch(
        self, notification: Annotated[dict[str, list[Any]], _NOTIFICATION]
    ) -> Done | Waiting:
        """Unattended task (mode: unattended): the task is ``notification["user_messages"]``,
        its data ``notification.get("context", [])``. Nobody can answer questions."""
        ...

    @hidden
    @strategy(
        UnifiedCodeAct(
            config=CodeActConfig(
                cell_timeout=1800.0,
                postconditions=[
                    require_result,
                    require_deliverable_wait_unattended,
                    require_settled_work,
                ],
            )
        ),
        context=_CONTEXT,
    )
    async def handle_orchestrate(
        self, notification: Annotated[dict[str, list[Any]], _NOTIFICATION]
    ) -> Done | Waiting:
        """Orchestrator (mode: orchestrator): the task is ``notification["user_messages"]``,
        its data ``notification.get("context", [])``. You own the result; subagents do the work."""
        ...

    @hidden
    @strategy(
        UnifiedCodeAct(
            config=CodeActConfig(
                cell_timeout=1800.0,
                postconditions=[
                    require_task_result,
                    require_deliverable_wait_unattended,
                    require_settled_work,
                ],
            )
        ),
        context=_CONTEXT,
    )
    async def handle_work(
        self, notification: Annotated[dict[str, list[Any]], _NOTIFICATION]
    ) -> Done | NeedInput | Waiting:
        """Child worker (mode: child): your parent's brief, a follow-up or an answer is in
        ``notification["user_messages"]``. Your parent reads only your ``TaskResult``."""
        ...


for _name in ("shell", "delegates", "producers", "system_messages", "cwd"):
    spec(UnifiedAtomAgent, _name, hidden=True)
spec(UnifiedAtomAgent, "jobs", description="Run and track work: see WorkManager")
# Host-facing members of the work types stay out of the model's docs.
for _cls, _members in (
    (WorkManager, ("keep", "max_items", "max_requests")),
    (Job, ("id", "dropped")),
):
    for _member in _members:
        spec(_cls, _member, hidden=True)
for _method in (
    WorkManager.running,
    WorkManager.records,
    WorkManager.aclose,
    JobEvent.status_source,
):
    spec(_method, hidden=True)
spec(JobEvent, "value", description="The item, the result, or the error text")
del _name, _cls, _members, _member, _method
