# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Atom agent: one work vocabulary, ``self.jobs`` runs shells and subagents.

``AtomAgent`` is the agent ``nooa atom`` runs (spec ``nooa_atom.agent:AtomAgent``,
also spelled ``nooa_atom.agent.agent:AtomAgent``). It builds on the private
``AtomAgentBase`` (``nooa_atom.agent.base``): the workspace tools (files,
repo, todos, skills, ``self.v``) and the session plumbing. On top of that it
has its own turn methods, and background jobs and child sessions are the
generic ``WorkManager`` with its ``Shell`` and ``Subagent`` work.

Its cells see this module's names (``AtomCodeAct``), not the base module's
names that inheritance would otherwise bring in. A subclass defined
elsewhere sees this module's names plus its own module's names.

A session saved by an earlier Atom (the ``delegate``/``spawn``/``JobsApi``
interface, or this agent's spike spelling
``nooa_atom.agent.unified:UnifiedAtomAgent``) loads as this class. See
``AtomAgent.after_restore`` for what carries over.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

# Cells see this module's names: the model builds Done/TaskResult/Todo, Shell and
# Subagent work, and Context/strategy/PredictStrategy for skill and method writing.
from nooa import Context, hidden, strategy
from nooa.agentdoc import doc, spec
from nooa.interactive import Done, NeedInput, Waiting
from nooa.strategies import CodeActV2, PredictStrategy  # noqa: F401
from nooa_atom.agent.work_manager import Job, JobEvent, WorkManager
from nooa_atom.agent.work_shell import CommandResult, Shell
from nooa_atom.agent.work_subagent import Question, Subagent
from nooa_atom.session.items import TaskResult  # noqa: F401
from nooa_atom.skills.manager import SkillManager
from nooa_atom.tools.repo_tools import RepoTools
from nooa_atom.tools.todo_list import Todo, TodoList  # noqa: F401

with hidden:
    import inspect
    import sys
    import time
    import types

    from nooa import no_trace
    from nooa.config import CodeActConfig
    from nooa.interactive import InteractiveAgent, require_task_result
    from nooa.runtime.channels import ChannelReader
    from nooa.storage.markers import nosnapshot
    from nooa.strategy_validation import InvariantError
    from nooa.tools import TodoManager
    from nooa.tools.shell_tools import FileWrite, Match
    from nooa_atom.agent import work_shell
    from nooa_atom.agent.base import AtomAgentBase, session_title_request  # noqa: F401
    from nooa_atom.agent.conditions import PERSON_NAMES, require_result
    from nooa_atom.agent.roles import TurnPolicy

__all__ = ["AtomAgent", "Files"]


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


# CodeActV2's python_cell description names the legacy delegation method.
_LEGACY_TOOL_HINT: Annotated[str, hidden] = (
    "If `self` exposes delegation, inspect its documentation with `print(doc(self.delegate))`."
)
_TOOL_HINT: Annotated[str, hidden] = (
    "Run commands and child agents as `self.jobs` work (`Shell`, `Subagent`), "
    "documented in the python_cell_tools block."
)


@hidden
class AtomCodeAct(CodeActV2):
    """CodeActV2 whose cells see this module's names, not the base module's legacy names.

    An ``AtomAgent`` subclass defined in another module (a ``file.py:Class``
    agent) sees this module's names too, and its own module's names win.
    It also makes the agent's workspace the default ``cwd`` of ``Shell()``
    for the turn (and the jobs it launches), and the agent the owner that
    closes the Shells they start.
    """

    def _extract_module_context(
        self, agent_module: types.ModuleType, agent: Any | None = None
    ) -> dict[str, Any]:
        this_module = sys.modules[__name__]
        context = super()._extract_module_context(this_module, agent=None)
        if agent_module is not this_module:
            context.update(super()._extract_module_context(agent_module, agent=None))
        if agent is not None:
            self._import_dynamic_classes(agent, agent_module, context)
        return context

    async def python_cell_context(self, runtime: Any) -> str:
        agent_module = inspect.getmodule(type(runtime.agent))
        if agent_module is None:
            return ""
        context = self._extract_module_context(agent_module, agent=runtime.agent)
        # The stub lists only names worth using; private or hidden helper types
        # (``_AgentJobs`` behind ``self.jobs``) stay bound in cells, unlisted.
        context = {
            name: value
            for name, value in context.items()
            if not name.startswith("_") and getattr(value, "_agentdoc_hidden", False) is not True
        }
        return self._render_execution_context_stub(
            context, {agent_module.__name__, __name__}, self.config.restrictions.blocked_modules
        )

    async def python_cell_state_context(self, runtime: Any) -> str:
        """One line: ``Cell locals: name (type), ...``, then the cell's imports, if any.

        The same inventory as CodeActV2's block (``python_cell_state()`` has
        all of it), without its heading and explanations.
        """
        state = self._cell_state(getattr(runtime, "current_call", None))
        label = self._python_cell_state_label

        def listing(items: list[str], more: str) -> str:
            shown = ", ".join(items[:20])
            return f"{shown} (+{len(items) - 20} more; `{more}`)" if len(items) > 20 else shown

        locals_ = [
            f"{label(name, max_chars=80)} ({label(kind, max_chars=80)})"
            for name, kind in sorted(state["cell_locals"].items())
        ]
        line = "Cell locals: " + (listing(locals_, "print(python_cell_state())") or "none")
        imports = [
            f"{label(name, max_chars=80)} → {label(module, max_chars=80)}"
            for name, module in sorted(state["cell_imports"].items())
        ]
        if imports:
            more = 'print(python_cell_state()["cell_imports"])'
            line += "; imports: " + listing(imports, more)
        return line

    def _build_execute_python_tool(self) -> Any:
        tool = super()._build_execute_python_tool()
        tool.description = tool.description.replace(_LEGACY_TOOL_HINT, _TOOL_HINT)
        return tool

    def _build_builtins(self, runtime: Any, call: Any) -> dict[str, Any]:
        cwd = getattr(runtime.agent, "cwd", None)
        if cwd is not None:
            work_shell.default_cwd.set(str(cwd))
        owner = getattr(runtime.agent, "_shells", None)
        if owner is not None:  # the session closes the Shells its cells start
            work_shell.shell_owner.set(owner)
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


_NO_ONE_TO_ASK: Annotated[str, hidden] = (
    "Nobody can answer questions in an unattended task: decide, state your assumptions in the "
    "TaskResult report, and return Done (or Waiting while launched work runs)."
)
_ONE_TURN_METHOD: Annotated[str, hidden] = (
    "AtomAgent has one turn method, handle(); the session's turn_method selects its role "
    "(nooa_atom.agent.roles)"
)


@hidden
def require_role(agent: Any, result: Any, call: Any) -> None:
    """The postconditions of the agent's role (``agent._turn_policy``), set by the host.

    - interactive: ``Waiting.on`` names only work that will deliver.
    - unattended, orchestrator: no ``NeedInput``; ``Done`` carries a result
      and leaves no running work; ``Waiting`` only on work that will deliver.
    - child: ``Done`` carries a ``TaskResult``; a one-shot child may not ask
      (its session port says so); no running work left at ``Done``;
      ``Waiting`` only on work that will deliver.
    """
    policy = getattr(agent, "_turn_policy", "interactive")
    if policy == "interactive":
        require_deliverable_wait(agent, result, call)
        return
    if policy == "child":
        require_task_result(agent, result, call)
    else:
        if isinstance(result, NeedInput):
            raise InvariantError(_NO_ONE_TO_ASK)
        require_result(agent, result, call)
    require_deliverable_wait_unattended(agent, result, call)
    require_settled_work(agent, result, call)


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


# The one place the work workflows are shown (the tools block, before the work types).
_WORK_EXAMPLES: Annotated[str, hidden] = """\
# Work: everything that runs is a job on self.jobs.
sh = Shell()                                               # one bash per object, started on first use
r = await sh.run("cd src && export X=1")                   # -> CommandResult; cd/variables persist
r = await sh.run("make")                                   # same bash: in src, X set
self.jobs.launch(sh.run("pytest -q"), name="tests")        # long: a JobEvent arrives later
job = self.jobs.launch(Shell(), name="sh")                 # a shell as a job; cancel closes it
r = await job.send("make")                                 # a request -> CommandResult
done = await Subagent("fix parser").run(todo)              # one-shot child -> TaskResult
rev = self.jobs.launch(Subagent("review"), name="review")  # kept child
req = rev.send(todo)                                       # await req -> TaskResult or Question
await rev.message("also check the docs")                   # a note it reads now, even mid-turn
"""

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


class AtomAgent(AtomAgentBase):
    """You are a careful agent working in one local repository.

    Read the repository instructions and the code you touch before editing.
    Make small, focused edits and preserve unrelated changes in the worktree.
    Verify with real commands before you claim anything; say what you did not
    check. Follow the newest steering from your user or parent within
    higher-priority rules. An explicit task authorization (for example to
    install or provision what the task needs) is approval; otherwise ask
    before destructive or outside-the-workspace actions.

    ``self.files`` reads and edits files, ``self.repo`` finds definitions and
    references, ``self.todo`` is your plan for multi-step work (activate the
    item you work on, append findings and checks to its ``comments``, complete
    it), and ``self.v`` keeps
    durable facts (cell locals last one method call). Everything else that
    runs is a job on ``self.jobs``: commands (``Shell``) and child agents
    (``Subagent``); the python_cell_tools block shows each way to run them.

    A subagent sees only its brief: a Todo whose description carries the
    requirements, constraints and acceptance checks. Verify what it reports;
    run one editing subagent at a time (they share this checkout).

    ``notification`` maps a channel (``user_messages``, ``jobs``,
    ``system_messages``...) to new items; ``jobs`` holds the ``JobEvent``s of
    work nobody awaited. Never predict a pending job's result. End every
    turn with exactly one ``return_result(...)``: ``Done(explanation=...)``,
    ``NeedInput(question=...)`` only when someone can answer, or
    ``Waiting(explanation=..., on=[job names])`` while launched work runs.
    The ``role`` block, when present, says who reads your result and how to end.
    """

    jobs: Annotated[WorkManager, nosnapshot]
    files: Annotated[Files, nosnapshot]
    todo: TodoList
    # Small records of the running jobs, saved with the snapshot so a resumed
    # session reports the ones that did not survive (``after_restore``).
    _work_records: Annotated[list, hidden]
    # Set by after_restore until the child sessions from before it are reported.
    _restore_pending: Annotated[bool, hidden, nosnapshot]
    _restored_lost: Annotated[set, hidden, nosnapshot]
    # The Shells this session's cells and jobs started: closed with the session.
    _shells: Annotated[work_shell.ShellOwner, hidden, nosnapshot]

    # The role the host gave this agent (``nooa_atom.agent.roles``): the policy its
    # handle() turns enforce and the instructions of the ``role`` block.
    _turn_policy: Annotated[str, hidden, nosnapshot]
    _role_instructions: Annotated[str | None, hidden, nosnapshot]
    # The session runs every turn with this method, whatever its turn_method.
    turn_entry: ClassVar[str] = "handle"

    def __init__(
        self,
        *args: Any,
        policy: TurnPolicy = "interactive",
        instructions: str | None = None,
        **kwargs: Any,
    ) -> None:
        """``policy`` and ``instructions`` are the role the host gives the agent.

        ``instructions`` become the static ``role`` block of the prompt prefix;
        ``None``, empty or whitespace-only text adds no block.
        """
        super().__init__(*args, **kwargs)
        if policy not in ("interactive", "unattended", "orchestrator", "child"):
            raise ValueError(f"unknown policy {policy!r}")
        self._turn_policy = policy
        text = instructions.strip() if isinstance(instructions, str) else ""
        self._role_instructions = text or None
        # The legacy JobsApi never started anything; drop it and its cancel hint.
        self.queue_manager._cancel_hints.pop("jobs", None)
        self._work_records = []
        self.jobs = _AgentJobs(self, on_change=self._record_work)
        self._shells = work_shell.ShellOwner()
        self.files = Files(self.shell)
        # Atom's own to-do list: membership on the list, content on each item.
        self._install_todo_list()
        # Commands run as Shell work; the legacy shell skill stays loaded for
        # self.files and self.repo but is not offered (``/skills`` can still).
        self.skills.registry.deactivate(["nemo.shell"])
        self._restore_pending = False
        self._restored_lost: set[str] = set()
        self._install_context()

    @hidden
    @no_trace
    def _install_todo_list(self) -> None:
        """Make ``self.todo`` a ``TodoList``, keeping the items of a restored earlier ``TodoManager``."""
        current = getattr(self, "todo", None)
        if isinstance(current, TodoList):
            return
        # An earlier session's TodoManager keeps its items, ids, notes and active item.
        self.todo = TodoList(manager=current) if isinstance(current, TodoManager) else TodoList()
        self.skills.register("nemo.todo", self.todo)

    @hidden
    @no_trace
    def _install_context(self) -> None:
        """Set this agent's own context blocks; ``after_restore`` sets them again.

        Restoring a snapshot brings back the context blocks it saved, an
        earlier Atom's among them: its ``python_cell_tools`` doc of
        ``self.shell``, ``JobsApi`` and ``delegate()``, and its ``queues`` block
        with the child and job hints. These replace them. The ``role`` block is
        the host's instructions for this build (none when they are empty), not
        the saved copy. The ``repository_instructions`` block is set again as
        the dynamic block over this run's startup folder (an earlier Atom saved
        a static copy). The other blocks (a user's own) are left as restored.
        """
        self.context["python_cell_tools"] = Context(
            _tools_doc(
                doc(RepoTools, Files, TodoList, SkillManager, concise=True)
                + "\n"
                + _WORK_EXAMPLES
                + doc(
                    WorkManager,
                    Job,
                    JobEvent,
                    Shell,
                    CommandResult,
                    Subagent,
                    Question,
                    inline_depth=0,
                )
                + "\n"
                + doc(TaskResult, concise=True)
            ),
            prefix=True,
        )
        self.context["todo_status"] = Context(expr="self.todo.status()")
        self.context["workspace_state"] = Context(self._workspace_state_context(), prefix=True)
        self.context["context_usage"] = Context(
            expr="self.context_stats.format() if self.context_stats else ''"
        )
        # Skills are documented in python_cell_tools (SkillManager); the
        # registry's dynamic inventory block is not shown. self.skills.status()
        # gives it on demand.
        self.context.pop("skills", None)
        if self._role_instructions:
            self.context["role"] = Context(self._role_instructions, prefix=True)
        else:
            self.context.pop("role", None)
        self._status_block()
        # AGENTS.md from the startup folder, re-read when a file changes
        # (``AtomAgentBase._repository_instructions``); empty while there is none.
        self.context["repository_instructions"] = Context(
            expr="self._repository_instructions()", prefix=True
        )

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

        text = (
            f"Workspace: {shown(self.cwd)} — files/repo/new Shells.\n"
            "Python paths use cell cwd, not Shell cwd."
        )
        if self.repo.root != self.cwd:
            text += f"\nRepository root: {shown(self.repo.root)} (search boundary)."
        return text

    @hidden
    @no_trace
    def _record_work(self) -> None:
        self._work_records = self.jobs.records()

    @hidden
    @no_trace
    def _status(self) -> str:
        """Unread counts and senders, read hints, running and idle work, cancel hint."""
        self._report_unadopted_children()
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
        """Set up a restored snapshot: current context blocks, and lost work reported once.

        The snapshot may come from this agent or from an earlier Atom (the
        ``delegate``/``spawn``/``JobsApi`` interface). History, todos,
        ``self.v`` and other saved attributes (such as a skill's private vars)
        are kept as restored. This agent's own context blocks are set again
        (``_install_context``), which replaces an earlier Atom's
        ``python_cell_tools`` and ``queues`` blocks. Work does not survive a
        restart, so each job that was running (this agent's ``self.jobs``
        work, or an earlier Atom's ``self.jobs.run()``/``spawn()`` jobs) gets
        one ``failed`` ``JobEvent`` saying it was lost. Child sessions from
        before the restart are not adopted as jobs; once the session port is
        installed, each one that is not closed also gets one notice
        (``_report_unadopted_children``).
        """
        from nooa_atom.workspace.options import drop_stale_memory_context

        InteractiveAgent.after_restore(self)
        self._install_todo_list()
        self._install_context()
        drop_stale_memory_context(self)
        self.skills.after_restore()
        lost, self._work_records = list(self._work_records or []), []
        for record in lost:
            name = str(record.get("name", "?"))
            self._restored_lost.add(name)
            self._jobs_in.put(
                JobEvent(
                    name,
                    0,
                    "failed",
                    "lost: the session restarted while it ran; launch it again if needed",
                )
            )
        # An earlier Atom's JobsApi records (name -> record of each running job).
        legacy, self._jobs_running = dict(self._jobs_running or {}), {}
        for name in legacy:
            self._restored_lost.add(str(name))
            self._jobs_in.put(
                JobEvent(
                    str(name),
                    0,
                    "failed",
                    "lost: a job of the earlier Atom (self.jobs.run/spawn) that stopped when "
                    "the session restarted; run it again as Shell or Subagent work if needed",
                )
            )
        self._restore_pending = True
        self._report_unadopted_children()

    @hidden
    @no_trace
    def _report_unadopted_children(self) -> None:
        """After a restore: one notice for each child session from before it that is not closed.

        Those children (an earlier Atom's ``delegate()``/``spawn()`` children,
        or this agent's ``Subagent`` children) are not jobs of the restored
        session and nothing of theirs is running for it. A child already
        reported as a lost job is not reported again. Runs once, as soon as
        the session port is installed (from ``prepare_tools`` or the
        ``status`` block).
        """
        if not self._restore_pending or self.session is None:
            return
        self._restore_pending = False
        try:
            children = self.session.children()
        except Exception:
            return
        for child in children:
            if child.status == "closed" or child.name in self._restored_lost:
                continue
            self._jobs_in.put(
                JobEvent(
                    child.name,
                    0,
                    "failed",
                    f"not adopted: the child session {child.name!r} ({child.status}) is from "
                    "before the session restarted; it is not a job here and is not running. "
                    "Start new work with Subagent if needed",
                )
            )

    @hidden
    async def prepare_tools(self) -> list[str]:
        """Connect remembered MCP servers; report child sessions a restore did not adopt."""
        self._report_unadopted_children()
        return await super().prepare_tools()

    @hidden
    async def aclose(self) -> None:
        """Cancel this session's jobs, close the Shells it started, then release the agent."""
        if self.turns.in_turn:
            raise RuntimeError("an executing turn cannot close its own agent")
        try:
            try:
                await self.jobs.aclose()
            finally:
                await self._shells.aclose()
        finally:
            await super().aclose()

    # ---- turns -------------------------------------------------------------------

    @hidden
    @strategy(
        AtomCodeAct(config=CodeActConfig(cell_timeout=1800.0, postconditions=[require_role])),
        context=_CONTEXT,
    )
    async def handle(
        self, notification: Annotated[dict[str, list[Any]], _NOTIFICATION]
    ) -> Done | NeedInput | Waiting:
        """Handle what arrived in ``notification``, newest input first, as your role says."""
        ...

    # One turn method: the session's turn_method selects the role, not the method.

    @hidden
    async def handle_batch(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(_ONE_TURN_METHOD)

    @hidden
    async def handle_work(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(_ONE_TURN_METHOD)

    @hidden
    async def handle_orchestrate(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(_ONE_TURN_METHOD)


for _name in ("shell", "delegates", "producers", "system_messages", "cwd"):
    spec(AtomAgent, _name, hidden=True)
spec(AtomAgent, "jobs", description="Run and track work: see WorkManager")
# Host-facing members of the work types stay out of the model's docs.
for _cls, _members in (
    (WorkManager, ("keep", "max_items", "max_requests")),
    (Job, ("id", "dropped")),
):
    for _member in _members:
        spec(_cls, _member, hidden=True)
# The adapters' own send()/message() are the protocol the manager calls; the model
# uses the launched job's (Job.send, Job.message), so only those are documented.
for _method in (
    WorkManager.running,
    WorkManager.records,
    WorkManager.aclose,
    JobEvent.status_source,
    Shell.send,
    Subagent.send,
    Subagent.message,
):
    spec(_method, hidden=True)
spec(Shell, "current_dir", hidden=True)  # the status block shows a session's directory
spec(Question, "id", hidden=True)
del _name, _cls, _members, _member, _method
