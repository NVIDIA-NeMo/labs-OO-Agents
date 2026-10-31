# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Atom agent's private base: the workspace tools, channels and session plumbing.

``AtomAgentBase`` is internal. The public agent is ``nooa_atom.agent:AtomAgent``
(``nooa_atom.agent.agent``), which builds on this class and replaces its
turn methods and the model-facing ``delegate``/``spawn``/``JobsApi`` surface.
It is not offered as a separate agent and has no spec in ``LEGACY_AGENT_SPECS``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from nooa import Context, hidden, no_trace, strategy
from nooa.agentdoc import doc, spec
from nooa.config import CodeActConfig, PredictConfig
from nooa.context_blocks.models import ContextWindowStats
from nooa.interactive import (
    Done,
    InteractiveAgent,
    NeedInput,
    SummarizationConfig,
    Waiting,
    install_summarizer,
)
from nooa.paths import get_project_dir
from nooa.storage.markers import nosnapshot
from nooa.strategies import CodeActV2, PredictStrategy
from nooa.tools import MethodWriting, SkillWriting, Todo, TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_atom.agent.activity import ActivityShellTools
from nooa_atom.agent.instructions import RepositoryInstructions
from nooa_atom.agent.jobs import JobInfo, JobsApi, JobUpdate  # noqa: F401  (visible to cells)
from nooa_atom.agent.slash_commands import SlashCommandRegistry

# Visible to generated cells (cells see this module's globals): the model
# builds TaskResult for unattended turns, matches what arrives on the
# ``delegates`` channel and catches the delegation errors.
from nooa_atom.session.items import (
    CHILD_MESSAGE_CHARS,
    ChildFailed,
    ChildFailedError,
    ChildMessage,
    ChildQuestion,
    ChildRef,
    ChildResult,
    PlanEntry,
    TaskResult,
)
from nooa_atom.session.registry import DepthLimitError
from nooa_atom.skills.manager import SkillManager
from nooa_atom.tools.repo_tools import RepoTools

with hidden:
    import ast
    import re

    from nooa.agents import TokenBudgetSummarizer
    from nooa.events import Error
    from nooa.runtime.channels import Channel, ChannelReader
    from nooa.runtime.harness_metrics import get_harness_metrics
    from nooa.strategies.codeact import TextOnlyResponseAction
    from nooa.strategies.template import TemplateStrategy
    from nooa_atom.agent.conditions import (
        require_result,
        require_running_wait,
        require_running_wait_unattended,
        require_task_result,
    )
    from nooa_atom.session.port import SessionPort, _reject_removed

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

__all__ = [
    "ChildFailed",
    "ChildFailedError",
    "ChildMessage",
    "ChildQuestion",
    "ChildRef",
    "ChildResult",
    "AtomAgentBase",
    "DepthLimitError",
    "Done",
    "NeedInput",
    "TaskResult",
    "Waiting",
    "session_title_request",
]

# CodeActV2 replaces the framework context blocks with a concise self doc: the
# ``state`` dump and the execution-context stub go, and the strategy adds its
# own compact ``python_cell_state`` block.
# Atom cells end the turn with a top-level ``return``; return_result is not
# bound. These reword core CodeAct's model-facing text that names it.
_CORE_FINISH_TEXT: Annotated[str, hidden] = (
    "To finish, call `return_result(value)` inside the cell. It immediately submits a\n"
    "value matching the method's annotated return type."
)
_FINISH_TEXT: Annotated[str, hidden] = (
    "To finish, `return value` at the cell's top level (not inside a def). It\n"
    "immediately submits a value matching the method's annotated return type."
)
_RETURN_HINT: Annotated[str, hidden] = "end the turn with `return Done(...)`"
_COMPLETION_REWRITES: Annotated[tuple[tuple[re.Pattern[str], str], ...], hidden] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"call return_result(?:\(\))? again", "return it again"),
        (
            r"with code `return_result\(value\)`; return_result is a Python builtin,",
            "with a top-level `return value` in the code; `return` is Python,",
        ),
        (r"call return_result\((\w+)\)", r"`return \1`"),
        (r"\bCall it as\b", "Write it as"),
        (r"\bcall it as\b", "write it as"),
        (r"return_result\(\) failed", "The returned value failed"),
        (r"return_result\(result=\[\.\.\.\]\)", "The returned value"),
        (r"return_result\(\) is missing required parameter", "The returned value is missing field"),
        (r"return_result\((\w+)=\.\.\.\)", r"The returned value's \1"),
        (r"return_result\(\)", "`return Done(...)`"),
        # One level of nested parentheses: return_result(Done(...)) -> return Done(...).
        (r"return_result\(((?:[^()]|\([^()]*\))+)\)", r"return \1"),
        (r"return_result validation error", "return validation error"),
        (r"(?<!`)\breturn_result\b(?!`)", "return"),
    )
)


def _completion_text(text: str) -> str:
    """Core text that names ``return_result``, reworded for a top-level ``return``."""
    if "return_result" not in text:
        return text
    for pattern, replacement in _COMPLETION_REWRITES:
        text = pattern.sub(replacement, text)
    return text


@hidden
def _return_result_removed(*args: Any, **kwargs: Any) -> None:
    """Atom's stand-in for core's return_result: a NameError that says what ends a turn."""
    del args, kwargs
    get_harness_metrics().return_result_call()
    raise NameError(f"name 'return_result' is not defined: {_RETURN_HINT}", name="return_result")


def _bare_return_line(code: str) -> int | None:
    """The line of the cell's first top-level ``return`` with no value, if any."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # the runtime reports it
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(node, ast.Return) and node.value is None:
            return node.lineno
        pending.extend(ast.iter_child_nodes(node))
    return None


@hidden
class AtomReturnCodeAct(CodeActV2):
    """CodeActV2 whose cells end the call with a top-level ``return``, not ``return_result``.

    The returned value is validated like ``return_result(value)`` (same
    retries and metrics). It needs the owned execution namespace
    (``_execution_namespace_for``), whose compiler marks executed returns.
    """

    def _build_execute_python_tool(self) -> Any:
        tool = super()._build_execute_python_tool()
        tool.description = tool.description.replace(_CORE_FINISH_TEXT, _FINISH_TEXT)
        return tool

    def _build_builtins(self, runtime: Any, call: Any) -> dict[str, Any]:
        builtins = super()._build_builtins(runtime, call)
        # A top-level ``return`` ends the turn; a call says so (and is counted).
        builtins["return_result"] = _return_result_removed
        return builtins

    def _return_statement_completes(self) -> bool:
        return True

    def _always_available_builtins(self) -> tuple[str, ...]:
        return tuple(
            name for name in super()._always_available_builtins() if name != "return_result()"
        )

    def _mentions_completion(self, arguments: object) -> bool:
        code = arguments.get("code") if isinstance(arguments, dict) else None
        return isinstance(code, str) and "return" in code

    def _completion_text(self, text: str) -> str:
        return _completion_text(text)

    async def _execute_code(self, runtime: Any, code: str, *args: Any, **kwargs: Any) -> Any:
        """Refuse a cell with a bare top-level ``return`` before it runs: it names no result."""
        line = _bare_return_line(code)
        if line is None:
            return await super()._execute_code(runtime, code, *args, **kwargs)
        from nooa.events import ExecutionResult

        error = SyntaxError(
            f"a bare `return` (line {line}) gives the turn no result: return one, "
            "e.g. `return Done(message=..., explanation=...)`"
        )
        return ExecutionResult(stdout="", error=error, defined_methods={})

    @strategy(TemplateStrategy())
    async def _tool_use_reminder(self, runtime: Any, reason: str) -> str:
        """{reason} Call `python_cell(code)`. To finish, `return value` at the cell's top level."""
        ...

    @staticmethod
    def _retry_text_only_response(context: Any) -> Any:
        return TextOnlyResponseAction.retry(
            Error(
                content=(
                    "Your last reply was plain text with no tool call. It was preserved, "
                    "but a bare message cannot end the turn or run code. "
                    f"To finish `{context.call.method_name}`, call `python_cell` with a "
                    "top-level `return value` in the cell. To continue working, "
                    "call `python_cell` with the next computation."
                )
            )
        )


_V2_CONTEXT: Annotated[dict[str, Any], hidden] = {
    "state": None,
    "execution_context": None,
    "self": Context(expr="doc(type(self), concise=True)", prefix=True),
}


class AtomAgentBase(InteractiveAgent):
    """You are a careful agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use the shell for files and commands, the repo
    tools for definitions and references, and todos for multi-step work.
    ``cd`` moves ``self.shell`` and ``self.repo`` together; the repository
    root bounds repo searches.

    Delegate bounded, context-heavy work (exploration, diagnosis, review, an
    independently verifiable change) to a child session with its own history.
    Hand it a Todo whose description carries the task's requirements and its
    rules or constraints: the child sees nothing else of your task.
    ``done = await self.delegate(description, todo, name="fix_parser")`` runs a one-shot
    worker and returns its ``Done``, whose ``result`` is a ``TaskResult`` (read ``result.report``).
    ``child = await self.spawn(description, todo, name="review")`` starts a kept child and returns
    at once; it may ask you questions and takes ``await child.send(...)`` follow-ups
    until ``await child.close()``. Wait on them with ``on=["delegates"]``. Outcomes
    arrive under ``notification["delegates"]``: ``ChildResult`` (``item.result.report``), ``ChildQuestion``
    (reply with ``await item.child.send(...)``), ``ChildFailed`` (``item.error``) or
    ``ChildMessage`` (``item.text``, a progress note). The Todo is a one-way
    brief: the child works on its own copy and its report comes back as the
    ``TaskResult``; update your own Todo after verifying it. Never predict or make up a pending child's
    result, poll, or sleep to wait. Children share this checkout: run one editing
    child at a time; read-only children can run in parallel. Children can delegate
    in turn only down to a fixed depth; past it ``DepthLimitError`` is raised.
    Inspect and verify what a child reports before relying on it.

    For multi-step work, activate the current Todo. Keep its title and description
    aligned with the current understanding, and append comments for material findings,
    decisions, completed steps, and verification—not routine narration. Keep durable
    facts on ``self.v``, task-specific plans and findings on the Todo's ``v``, and
    scratch in cell locals; neither store is a dump. Locals last one method call:
    keep what later turns need in ``self.v`` or the todo list.

    Work until the newest request is complete or genuinely needs user input. Use
    as many execution cells as necessary, inspect each result, and never claim a
    check passed without running it. Run anything that may take more than a
    minute or two (tests, builds) with ``self.jobs.run(...)`` instead of blocking
    a cell. Jobs and children share one set of names.

    ``notification`` maps a channel (``user_messages``, ``system_messages``,
    ``slash_commands``, ``delegates``, ``jobs``) to its new items. In
    interactive turns, send brief progress notes and each answer (complete
    Markdown) with ``self.message()``.

    End every turn with exactly one top-level ``return``: ``Done`` after
    sending the reply with ``self.message()``::

        self.message("Fixed the off-by-one in `parse()`; its tests pass.")
        return Done(explanation="fixed parse() and ran its tests")

    ``NeedInput(question=...)`` only when a person must answer (the host shows
    ``question``; don't also send it); ``Waiting(explanation=..., on=[...])``
    only while a job or child you started runs, ``on`` naming it.
    """

    # Attributes carrying this agent's own tools. The skill registry refuses
    # to let a later skill — a workspace SKILL.md, a client-forwarded MCP
    # server — take one over, which would remove the tool while the model is
    # still told it has it.

    __protected_skill_attrs__ = frozenset(
        {"shell", "repo", "todo", "libs", "skills", "workspace_settings", "session"}
    )

    cwd: Annotated[Path, nosnapshot]
    # Host-driven input channels. These live here rather than on
    # InteractiveAgent because they are host concepts: slash commands
    # are a UI affordance whose registry is in this package, and
    # system_messages carries host-provided system input. A host that sends
    # a command's output to the agent puts it on the slash_commands channel.
    _slash_commands_in: Annotated[Channel, hidden, nosnapshot]
    # The command registry the Session lists and runs (Session.commands(),
    # Session.invoke_command()). create_session_agent() installs the /skills and
    # /mcp controls with set_controls().
    slash_commands: Annotated[SlashCommandRegistry, hidden, nosnapshot]
    # Set by the registry that binds itself to this agent (skills use it).
    _command_registry: Annotated[Any, hidden, nosnapshot]
    _system_messages_in: Annotated[Channel, hidden, nosnapshot]
    system_messages: Annotated[Any, nosnapshot]
    shell: Annotated[ActivityShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]
    todo: TodoManager
    libs: Annotated[SkillWriting, nosnapshot]
    skills: Annotated[SkillManager, nosnapshot]
    _base_shell: Annotated[ShellTools, hidden, nosnapshot]
    _instructions: Annotated[RepositoryInstructions, hidden, nosnapshot]
    _summarizers: Annotated[list[Any], hidden, nosnapshot]
    _delegates_in: Annotated[Any, hidden, nosnapshot]
    delegates: Annotated[ChannelReader, nosnapshot]
    # Background jobs (``self.jobs``): the model-facing view of
    # ``queue_manager.spawn``. Their results arrive on the ``jobs`` channel.
    # JobsApi is a Skill, so it is never snapshotted; its doc line is set
    # with spec() below the class (this module postpones annotations).
    _jobs_in: Annotated[Channel, hidden, nosnapshot]
    jobs: JobsApi
    # name -> small record of each running job, saved with the snapshot so a
    # resumed session can report the jobs that did not survive (JobsApi).
    _jobs_running: Annotated[dict, hidden]
    # The Session port, installed by the session that runs this agent. The
    # model uses the flat delegation methods below, not the port.
    session: Annotated[SessionPort | None, hidden, nosnapshot]
    # Tells install_port() not to document the port to the model.
    session_port_visible: ClassVar[bool] = False

    def __init__(
        self,
        llm: UnifiedLLM | None = None,
        *,
        cwd: str | Path = ".",
        summarization: SummarizationConfig | None = None,
        skills_dirs: list[Path] | None = None,
        libs_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, **kwargs)
        self.session = None
        self._slash_commands_in = self.queue_manager.queue("slash_commands")
        self._system_messages_in = self.queue_manager.queue("system_messages")
        self.system_messages = self._system_messages_in.reader
        self.cwd = Path(cwd).resolve()
        self._base_shell = ShellTools(cwd=str(self.cwd))
        self.shell = ActivityShellTools(self._base_shell, self.event_manager)
        self.repo = RepoTools(root=self.cwd, session=self.shell.session)
        # The Session port finds this channel and does not declare its own.
        self._delegates_in = self.queue_manager.queue("delegates")
        self.delegates = self._delegates_in.reader
        self._jobs_in = self.queue_manager.queue("jobs")
        self._jobs_running = {}
        self.jobs = JobsApi(self)
        self.todo = TodoManager()
        # Libraries live at <project>/.nooa/libs. get_project_dir() resolves
        # that per process, which is what a one-workspace host like the TUI
        # wants. A host serving several workspaces at once must say which
        # project it means, or every session shares one directory — and
        # SkillWriting puts it on sys.path and activates local.*, so that
        # would expose one workspace's agent-authored code to another.

        self.libs = SkillWriting(self, path=libs_dir or get_project_dir("libs"))

        self.skills = SkillManager(self)
        self.skills.register("nemo.shell", self.shell)
        self.skills.register("nemo.repo", self.repo)
        self.skills.register("nemo.todo", self.todo)
        self.skills.register("nemo.libwriting", self.libs)
        self.skills.register("nemo.methodwriting", MethodWriting())
        self.skills.registry.activate(
            ["nemo.shell", "nemo.repo", "nemo.todo", "nemo.libwriting", "nemo.methodwriting"]
        )
        # Installed ``nooa.skills`` entry points are part of the shared host
        # surface. Load them so hosts can expose ``@slash_command`` methods,
        # but leave them inactive until the user opts in with ``/skills``.
        # Memory integration is deferred; do not auto-load its entry point.
        # Also ignore the retired web publisher entry point in older installed
        # package metadata, and producers (core already sets ``self.producers``).
        loaded = set(self.skills.loaded())
        installed = []
        for name in self.skills.discovered():
            attr_name = name.rsplit(".", 1)[-1].replace("-", "_")
            if (
                name in {"nemo.memory", "nemo.web", "nemo.producers"}
                or name in loaded
                or hasattr(self, attr_name)
            ):
                continue
            installed.append(name)
        if installed:
            self.skills.load(installed)
        if skills_dirs:
            self.skills.discover_skills_dirs(skills_dirs)
        self.slash_commands = SlashCommandRegistry(self, skills_dirs=skills_dirs or ())

        self.context["python_cell_tools"] = Context(
            doc(RepoTools, ActivityShellTools, TodoManager, JobsApi, concise=True),
            prefix=True,
        )
        self.context["todo_status"] = Context(expr="self.todo.status()")
        self.context["workspace_state"] = Context(expr="self._workspace_state_context()")
        self.context["context_usage"] = Context(expr="self._context_usage()")
        # Re-read from the startup folder whenever an AGENTS.md changes, is
        # created, or is removed; the block is empty while there is none.
        self._instructions = RepositoryInstructions(self.cwd)
        self.context["repository_instructions"] = Context(
            expr="self._repository_instructions()", prefix=True
        )
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)
        # self.events stays exposed, but the core events_api block is redundant
        # here: context_usage already says how to inspect and collapse events
        # (``self.events.collapse(...)``, ``print(doc(self.events))``).
        self.context_manager.remove_protected("events_api")

        self._summarization = summarization or SummarizationConfig()
        install_summarizer(self._summarization, self)

    @no_trace
    def _context_usage(self) -> str:
        """Context window usage plus the guidance for inspecting and collapsing events.

        The first render comes before any stats exist; it still carries the
        guidance, since the core ``events_api`` block is removed here.
        """
        stats = self.context_stats or ContextWindowStats(context_blocks_count=0, events_count=0)
        return stats.format()

    @no_trace
    def _repository_instructions(self) -> str:
        """Return current AGENTS.md guidance from the startup folder, or ''."""
        return self._instructions.render()

    @no_trace
    def _workspace_state_context(self) -> str:
        """Describe the workspace state without exposing stored values."""
        from html import escape

        def shown(path: object) -> str:
            text = str(path).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
            return escape(text[:159] + "…" if len(text) > 160 else text, quote=False)

        cwd, root = shown(self.shell.cwd), shown(self.repo.root)
        count = len(self.vars)
        return (
            f"Working directory (already active for `self.shell` and `self.repo`; persists across cells and turns): {cwd}\n"
            f"Repository root (the boundary for repo searches): {root}\n"
            "Use relative paths; call `cd` only to intentionally change directories.\n"
            f"`self.v`: {count} persistent vars — list: `print(dict(self.vars))`; "
            "remove one: `del self.v.<name>`; clear all: `self.vars.clear()`"
        )

    @hidden  # the host's [session-title] request names the call; benchmarks forbid it
    async def rename_session(self, title: str) -> str:
        """Set this session's title and return the normalized title.

        Use this when a host system message asks you to title the session. Keep
        the title descriptive and very short (usually 2-5 words). This changes
        session metadata only; do not call it in place of answering the user. A
        title the person chose is kept.
        """
        normalized = " ".join(str(title).strip().strip('"').strip("'").split())[:60]
        if not normalized:
            raise ValueError("Session title cannot be empty")
        await self._port().rename(normalized)
        return normalized

    async def delegate(
        self,
        description: str,
        prompt: str | Todo,
        *,
        name: str,
        brief: str | None = None,
        model: str | None = None,
        **removed: Any,
    ) -> Done:
        """Run a one-shot worker child on ``prompt``, wait for it, and return its ``Done``.

        The child works in this checkout with its own history, cannot ask you
        questions (it decides and states its assumptions), and is closed after
        its result. ``done.result`` is its ``TaskResult`` (read
        ``result.report``). Use it only when you need the report before
        continuing; otherwise use ``spawn()`` (``kept=False`` for a one-shot
        that does not wait). A Todo prompt is a one-way
        brief, as in ``spawn()``: nothing is written to your Todo. Record
        what you verified on it yourself.

        Raises ``ChildFailedError`` when the child's turn fails, and
        ``DepthLimitError`` when a child would exceed the depth limit.

        Args:
            description: Short title (2-6 words) for people.
            prompt: A ``Todo`` carrying the task's requirements, rules and
                acceptance checks, or the full task as text.
            name: Its name (letters, digits, ``_`` or ``-``) such as ``"fix-parser"``, not
                used by a running job or open child; its path is yours plus
                ``/name``.
            brief: Extra instructions sent with the prompt.
            model: Model alias for the child; yours when omitted.
        """
        _reject_removed("delegate", removed)
        child = await _start_child(self, description, prompt, brief, model, name=name, kept=False)
        outcome = await child.wait()
        if not isinstance(outcome, ChildResult):  # a one-shot worker's question is a failure
            raise ChildFailedError(f"Child {child.path} did not return a result")
        return outcome.done

    async def spawn(
        self,
        description: str,
        prompt: str | Todo,
        *,
        name: str,
        brief: str | None = None,
        model: str | None = None,
        kept: bool = True,
        **removed: Any,
    ) -> ChildRef:
        """Start a child on ``prompt`` and return its ``ChildRef`` at once.

        With ``kept=False`` it is a one-shot worker, like ``delegate()`` but
        without waiting: it cannot ask questions, its ``ChildResult`` (or
        ``ChildFailed``) arrives under ``notification["delegates"]``, and it
        closes itself after that, so parallel workers need no ``close()``.

        A kept child's outcomes arrive under ``notification["delegates"]``: a
        ``ChildResult`` (``item.result`` is a ``TaskResult``; ``.report`` the full report), a
        ``ChildQuestion`` (reply with ``await item.child.send(...)``) or a
        ``ChildFailed``, after any ``ChildMessage`` progress notes;
        ``item.child.name`` matches the returned ref (also ``self.child(name)``).
        Send follow-ups with ``await ref.send(text)``; it stays open until
        ``await ref.close()``.

        A ``Todo`` prompt (managed or hand-built) is a one-way brief: the child
        gets its fields and works on its own copy; nothing flows back to your
        Todo. The result comes back as a ``TaskResult``: update your own Todo
        (``self.todo.comment``, ``self.todo.complete``) after verifying it. To
        change a child's task, send the new brief as text or a Todo; the child
        decides how to update its own Todo.

        Args:
            description: Short title (2-6 words) for people.
            prompt: A ``Todo`` carrying the task's requirements, rules and
                acceptance checks, or the full task as text.
            name: Its name (letters, digits, ``_`` or ``-``) such as ``"review-pr"``,
                not used by a running job or open child (a closed child's
                name is free again); its path is yours plus ``/name``.
            brief: Extra instructions sent with the prompt.
            model: Model alias for the child; yours when omitted.
            kept: ``False`` for a one-shot worker that closes itself after its result.
        """
        _reject_removed("spawn", removed)
        return await _start_child(self, description, prompt, brief, model, name=name, kept=kept)

    @hidden
    def receive_delegated_todo(self, todo: Todo) -> None:
        """Called on a child before its first turn: a local copy of its brief becomes its active Todo.

        The copy is this child's own working copy: nothing in it flows back
        to the parent, whose result is the ``TaskResult``. It keeps the
        brief's id (the id named in the child's first message); a Todo with
        that id already in the child's list raises ``ValueError`` rather
        than being replaced. A done brief is added but not activated.
        """
        self.todo.insert_copy(todo)

    def children(self) -> list[ChildRef]:
        """Handles on the child sessions you started, with live status and any unanswered question."""
        return self._port().children()

    def child(self, name: str) -> ChildRef:
        """The child you started with this name, e.g. ``await self.child("review").close()``.

        Also takes its full path (``"/root/review"``) or id. A deeper path
        (``"/root/review/verifier"``) gives a read-only ref: ``status`` and
        ``wait()``; control it through its parent. The open child wins over
        closed ones with the same name. Raises ``KeyError`` if there is none.
        """
        port = self._port()
        if name.startswith("/") and name.count("/") > port.path.count("/") + 1:
            return port.descendant(name)
        children = self.children()
        matches = [c for c in children if name in (c.name, c.path, c.id)]
        open_ = [c for c in matches if c.status != "closed"]
        found = open_ or matches
        if found:
            return found[-1]
        names = sorted({c.name for c in children if c.status != "closed"})
        raise KeyError(f"no child named {name!r}; open children: {names}")

    @hidden
    @no_trace
    def _child_using_name(self, name: str) -> str | None:
        """The path of the open child named ``name``, if any: jobs share the children's names."""
        if self.session is None:
            return None
        for child in self._port().children():
            if child.name == name and child.status != "closed":
                return child.path
        return None

    @hidden
    @no_trace
    def _session_path(self) -> str:
        """This session's path in the tree (``/root`` outside a session)."""
        return self.session.path if self.session is not None else "/root"

    def message(self, text: str, *, echo: bool = False) -> None:
        """Send a Markdown message to the user.

        In a child session the text goes to the parent instead, as a
        ``ChildMessage`` on its ``delegates`` channel: a short progress note
        (cut at 1,000 characters), not the child's result.

        Each call renders as an independent block — so every call must be a
        complete, self-contained Markdown document.  In particular, never split
        a table across calls: the header row and all data rows must be in the
        same ``message()`` call, otherwise the table will not render correctly.

        Args:
            text: Markdown content to send.
            echo: If True, also ``print()`` the text so the LLM can see it
                in the execution output. By default the LLM does NOT see the
                content of ``message()`` calls — only the user does. Use
                ``echo=True`` when you need to reference the sent content in
                subsequent cells.
        """
        super().message(text, echo=echo)
        if len(text) > CHILD_MESSAGE_CHARS and self._is_child_session():
            print(
                f"Note: your parent gets only the first {CHILD_MESSAGE_CHARS:,} characters "
                "of this message. Put the full report in your result (TaskResult.report)."
            )

    @hidden
    @no_trace
    def _is_child_session(self) -> bool:
        """Whether this agent runs as a child session (its messages go to a parent)."""
        port = self.session
        return port is not None and port.info().parent_id is not None

    @hidden
    @no_trace
    def _queue_status(self) -> str:
        """The ``queues`` block: "Unread:" counts, "Read now:", "Running:", "Idle:", then hints."""
        from nooa_atom.agent.background import CLOSE_HINT, READ_HINTS, child_sections

        children = child_sections(self)
        status = self.queue_manager.status(
            more_unread=children["unread"],
            more_work=children["running"],
            idle=children["idle"],
            read_hints=READ_HINTS,
        )
        listed = any(children.values())  # a child is listed (its question counts)
        return f"{status}\n{CLOSE_HINT}" if listed else status

    @hidden
    def after_restore(self) -> None:
        """Called by the session registry after it restores a snapshot into this agent.

        Reports, once, the ``self.jobs`` jobs that were running when the saved
        session stopped (a ``JobsLost`` event); they did not survive.
        """
        from nooa_atom.workspace.options import drop_stale_memory_context

        super().after_restore()
        self.jobs.after_restore()
        drop_stale_memory_context(self)
        self.skills.after_restore()

    @hidden
    def _port(self) -> SessionPort:
        if self.session is None:
            raise RuntimeError("This agent is not running in a session")
        return self.session

    @hidden
    async def prepare_tools(self) -> list[str]:
        """Connect the MCP servers this workspace remembers; return warnings.

        Called through ``Session.prepare_tools()`` by a host before the
        first turn. The ``/skills`` and ``/mcp`` controls come from the
        agent factory.
        """
        from nooa_atom.workspace.options import AtomOptions, connect_session_mcp

        return await connect_session_mcp(self, AtomOptions.load(self.cwd))

    @hidden
    def inherit_tools(self, parent: Any) -> list[str]:
        """Share ``parent``'s connected MCP servers with this child agent; return warnings.

        Called through ``Session.inherit_tools()`` by the registry before a
        child's first turn. The tool objects are the parent's: no new
        connection, and closing this child leaves them to the parent.
        """
        parent_skills = getattr(parent, "skills", None)
        if not isinstance(parent_skills, SkillManager):
            return []
        return self.skills.share_from(parent_skills)

    @hidden
    @no_trace
    def plan(self) -> list[PlanEntry]:
        """The todo list as ACP plan entries, read through ``Session.plan()``.

        A done todo is ``completed``, the active one ``in_progress``, the
        rest ``pending``.
        """
        active = self.todo.active()
        return [
            PlanEntry(
                content=todo.title,
                status=(
                    "completed"
                    if todo.status == "done"
                    else "in_progress"
                    if active is not None and todo.id == active.id
                    else "pending"
                ),
            )
            for todo in self.todo.list_todos()
        ]

    @no_trace
    def get_summarization_status(self) -> dict[str, Any]:
        """Return compact history information for host status displays."""
        tags = self.event_manager.keys()
        summary_tags = [tag for tag in tags if ".." in tag]
        summarizers = getattr(self, "_summarizers", [])
        summarizer = summarizers[0] if summarizers else None
        config = getattr(summarizer, "config", None) if summarizer else None
        pending = getattr(summarizer, "_pending_task", None) if summarizer else None
        automatic = bool(getattr(summarizer, "_automatic_context_budget", False))
        declared_policy = getattr(summarizer, "policy", None) if summarizer else None
        policy = (
            declared_policy
            if isinstance(declared_policy, str) and declared_policy
            else "token_budget"
            if isinstance(summarizer, TokenBudgetSummarizer)
            else "custom"
            if summarizer is not None
            else "none"
        )
        stats = self.context_stats
        return {
            "active_events": len(tags),
            "summary_count": len(summary_tags),
            "summary_tags": summary_tags,
            "has_summarizer": summarizer is not None,
            "policy": policy,
            "current_tokens": getattr(stats, "prompt_tokens", 0) if stats else 0,
            "max_tokens": getattr(config, "max_tokens", 0) if config else 0,
            "threshold_fraction": (
                getattr(summarizer, "_automatic_context_budget_percent", None)
                if automatic
                else None
            ),
            "preserve_recent": getattr(config, "preserve_recent", 0) if config else 0,
            "compaction_pending": pending is not None,
            "compaction_ready": bool(
                pending is not None
                and pending.done()
                and getattr(summarizer, "_pending_summary", None)
                and getattr(summarizer, "_pending_range", None)
            ),
        }

    @hidden
    @no_trace
    def _execution_namespace_for(self, call: Any) -> Any:
        """Each method call gets its own owned namespace (locals last one call).

        Its compiler marks a cell's executed top-level ``return``, which ends
        the turn (``AtomReturnCodeAct``).
        """
        from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace

        return ExecutionNamespace(reserved=RUNTIME_NAMES | frozenset(call.kwargs))

    @hidden
    @strategy(PredictStrategy(PredictConfig(output_serialization="tool_call")))
    async def name_session(self, user_message: str) -> str:
        """Generate an ultra-short 2-5 word session title for the conversation
        opened by the given user message."""
        ...

    @hidden
    @strategy(
        AtomReturnCodeAct(
            config=CodeActConfig(cell_timeout=1800.0, postconditions=[require_running_wait])
        ),
        context=_V2_CONTEXT,
    )
    async def handle(
        self,
        notification: Annotated[
            dict[str, list[Any]], spec(max_string=50_000, max_length=100, max_depth=10)
        ],
    ) -> Done | NeedInput | Waiting:
        """Handle one interactive turn: the newest request and anything else that arrived.

        Do all the work the request needs before ending the turn; run checks
        rather than assume them. End with exactly one top-level ``return``
        (``Done``, ``NeedInput`` or ``Waiting``) as your instructions describe.

        You are event-driven: queues shows running producers and pending
        inputs, while notification contains events delivered for this turn; a
        producer may finish before its result is consumed.

        Use ``self.message()`` regularly to keep the user informed with brief
        progress updates while you work, especially during tool use or longer
        tasks. Updates should explain what you are doing, what you have
        learned, and what comes next.
        """
        ...

    @hidden
    @strategy(
        AtomReturnCodeAct(
            config=CodeActConfig(
                cell_timeout=1800.0,
                postconditions=[require_result, require_running_wait_unattended],
            )
        ),
        context=_V2_CONTEXT,
    )
    async def handle_batch(
        self,
        notification: Annotated[
            dict[str, list[Any]], spec(max_string=50_000, max_length=100, max_depth=10)
        ],
    ) -> Done | Waiting:
        """Work on one task unattended and return a structured result.

        The task is the text in ``notification["user_messages"]``; data sent
        with it is in ``notification.get("context", [])``. Nobody is watching
        and nobody can answer a question, and ``self.message()`` reaches
        nobody: the result is the only thing the requester reads.

        Do the task completely, verify it, then end with::

            return Done(
                explanation="fixed the parser bug",
                result=TaskResult(
                    solution_description="What was done, including modified paths",
                    evidence="Commands run and what they showed",
                    how_to_verify="How someone else can check it",
                    report="The full report for the requester",
                ),
            )

        If something blocks you, still return ``Done`` with a ``TaskResult``
        that says what blocked you and what you tried. Return
        ``Waiting(explanation=..., on=[...])`` only while a job you started
        is still running.
        """
        ...

    @hidden
    @strategy(
        AtomReturnCodeAct(
            config=CodeActConfig(
                cell_timeout=1800.0,
                postconditions=[require_result, require_running_wait_unattended],
            )
        ),
        context=_V2_CONTEXT,
    )
    async def handle_orchestrate(
        self,
        notification: Annotated[
            dict[str, list[Any]], spec(max_string=50_000, max_length=100, max_depth=10)
        ],
    ) -> Done | Waiting:
        """You are a recursive language model agent.

        You work on one task on your own. Nobody is watching and nobody can
        answer a question; the result you return is the only thing the
        requester reads.

        The task is the text in ``notification["user_messages"]``; data sent
        with it is in ``notification.get("context", [])``.

        You are the orchestrator. You decide how the work is split and who does
        each part, and you are responsible for the result. Do not implement the
        work yourself; understanding, writing requirements and measuring are yours.

        Work in this loop:
        1. Understand the request. Read what you need, or have a read-only child
           explore.
        2. Write the requirements: how success will be measured. Put them in a
           Todo (``self.todo.add(title, description=...)``): the goal, the exact
           requirements and constraints from the task, and acceptance checks
           that can be run. When a requirement is best written as code (for
           example a failing test), decide who writes it, and confirm it fails
           before the work starts.
        3. Do the work: hand each Todo to a child with
           ``child = await self.spawn(description, todo, name=...)`` and keep
           ``child`` in ``self.v``. The Todo is all the child sees of the task: put the
           task's requirements and its rules or constraints in it. Children
           share one checkout: run one editing child at a time; read-only work
           can run in parallel. While children run, end the turn with
           ``Waiting(explanation=..., on=["delegates"])``.
        4. Measure: results arrive in ``notification["delegates"]`` (a
           ``ChildResult`` whose ``item.result.report`` is the child's report
           and ``item.result`` its ``TaskResult``, a ``ChildQuestion``: ``await item.child.send(...)``, or
           a ``ChildFailed``; a ``ChildMessage`` (``item.text``) is a progress
           note). Its report is the TaskResult; record what you verified on
           your own Todo.
           Run every acceptance check yourself; a report is
           not proof. Record each result with ``self.todo.comment(todo, ...)``.
        5. Loop: if a requirement is not met, send the child the failing check
           and its output with ``await child.send(...)`` and wait again. Decide
           when another round is not worth it given the time left.

        Finish with ``Done`` and a ``TaskResult`` whose ``evidence`` is what you
        measured in step 4; if requirements are still unmet, say which.
        Before ``Done``, leave the tree clean: no scratch files.
        Leave children open; the host closes them after grading.

            return Done(
                explanation="fixed the parser bug",
                result=TaskResult(
                    solution_description="What was done, including modified paths",
                    evidence="The checks you ran and what they showed",
                    how_to_verify="How someone else can check it",
                    report="The full report for the requester",
                ),
            )

        If something blocks you, still return ``Done`` with a ``TaskResult``
        that says what blocked you and what was tried.
        """
        ...

    @hidden
    @strategy(
        AtomReturnCodeAct(
            config=CodeActConfig(cell_timeout=1800.0, postconditions=[require_task_result])
        ),
        context=_V2_CONTEXT,
    )
    async def handle_work(
        self,
        notification: Annotated[
            dict[str, list[Any]], spec(max_string=50_000, max_length=100, max_depth=10)
        ],
    ) -> Done | NeedInput | Waiting:
        """Work on a task for your parent agent and report back to it.

        You are a child session. Your parent is an agent, not a person, and it
        reads only your result. The task is in ``notification["user_messages"]``;
        its first line says whether you can ask your parent questions (later
        messages there are follow-ups). Your task is the brief; your
        ``TaskResult`` is your report; a Todo in your list is your own
        working copy (nothing in it goes back to your parent).

        Assume you are the only one editing this checkout, and leave it clean:
        remove scratch files before ``Done``.

        End with exactly one top-level ``return``:

        - ``Done`` with a ``TaskResult``, when the task is finished or blocked
          (then say what blocked you and what you tried)::

              return Done(
                  explanation="fixed the parser bug",
                  result=TaskResult(
                      solution_description="What was done, including modified paths",
                      evidence="Commands run and what they showed",
                      how_to_verify="How someone else can check it",
                      report="The full report for your parent",
                  ),
              )

        - ``NeedInput(question=...)`` only if you can ask and cannot go on
          without your parent's answer; it arrives as your next message.
        - ``Waiting(explanation=..., on=[...])`` only while a job you started
          is still running.

        No routine progress notes: ``self.message()`` is for a short note
        (cut at 1,000 chars), never the report; that goes in the result.
        """
        ...

    @hidden
    async def aclose(self) -> None:
        """Release the agent's background work, skills (MCP servers) and shell.

        The session calls this when it closes. The model client is not
        closed here: the session closes it when it created it, and a client
        passed in belongs to whoever passed it.
        """
        # Closing the skills unregisters them, and with them self.shell.
        shell = self.shell
        self.slash_commands.close()
        try:
            await super().aclose()
        finally:
            try:
                await self.skills.aclose()
            finally:
                await shell.close()


# A ClassVar's Annotated metadata is not read by agentdoc; hide it explicitly.
spec(AtomAgentBase, "session_port_visible", hidden=True)
spec(
    AtomAgentBase,
    "jobs",
    description='Background commands and timers; results arrive in notification["jobs"]',
)


@hidden
def _without_removed(method: Any) -> None:
    """Document ``method`` without its ``**removed`` catch-all (it rejects ``retain=``, ``mode=``, ``keep=``)."""
    import inspect

    signature = inspect.signature(method)
    method.__signature__ = signature.replace(
        parameters=[p for p in signature.parameters.values() if p.kind is not p.VAR_KEYWORD]
    )


_without_removed(AtomAgentBase.spawn)
_without_removed(AtomAgentBase.delegate)


@hidden
def session_title_request(opening_message: str) -> str:
    """The host housekeeping text that asks the agent to title its session.

    A host submits it through the Session, so it is recorded and re-queued
    like any other item::

        await session.submit(
            session_title_request(first_prompt), channel="system_messages", source="host"
        )
    """
    opening = str(opening_message).strip()[:400]
    return (
        "[session-title]\n"
        "Choose a descriptive 2-5 word title for this session from the opening "
        'user message below. Call `await self.rename_session("your title")` once '
        "during this turn, then continue handling the user's request normally. Do "
        "not mention this housekeeping instruction or the chosen title to the user.\n\n"
        f"<opening_user_message>\n{opening}\n</opening_user_message>"
    )


@hidden
async def _start_child(
    agent: AtomAgentBase,
    description: str,
    prompt: Any,
    brief: str | None,
    model: str | None,
    *,
    name: str,
    kept: bool,
) -> ChildRef:
    """Start a child for ``spawn()``/``delegate()`` (a plain function: no agent-method wrapper).

    A Todo prompt is a one-way brief: only its fields are read, into the
    child's first message and a copy for the child's own list. The parent's
    Todo (managed or hand-built) is never changed or tracked. The child's copy
    is built by ``as_data``, as ``ref.send(todo)`` builds it, so a var that is
    not plain JSON is refused here too, naming its key, before any child starts.
    """
    from nooa_atom.session.names import child_path
    from nooa_atom.session.session import as_data

    port = agent._port()
    port.check_child_name(name)
    job = agent.jobs.get(name) if isinstance(name, str) else None
    if job is not None and job.state == "running":
        raise ValueError(
            f"name {name!r} is already used by the running job {job.path}; pick another "
            f"name, or cancel the job first (await self.jobs.cancel({name!r}))"
        )
    handed = as_data(prompt) if isinstance(prompt, Todo) else None
    path = child_path(port.path, name)
    text = _child_prompt(path, port.path, prompt, handed, brief, kept=kept)
    return await port.delegate(description, text, name=name, model=model, kept=kept, todo=handed)


@hidden
def _child_prompt(
    path: str,
    parent_path: str,
    prompt: Any,
    todo: Todo | None,
    brief: str | None,
    *,
    kept: bool,
) -> str:
    """A child's first message: its path and its parent's, what kind of child it is, the task and the brief."""
    task = f", working on Todo {todo.id}: {todo.title}" if todo is not None else ""
    kind = (
        "You can ask your parent questions and may get follow-ups."
        if kept
        else "One-shot task: you can't ask questions; decide, and state your assumptions in "
        "the report."
    )
    lines = [f"You are {path}, a child of {parent_path}{task}. {kind}", ""]
    lines.append(_todo_prompt(todo) if todo is not None else str(prompt))
    if brief:
        lines += ["", "Brief:", str(brief)]
    return "\n".join(lines)


@hidden
def _todo_prompt(todo: Todo) -> str:
    """A Todo as the text a child works from: id, title, description, comments."""
    lines = [f"Task [{todo.id}]: {todo.title}"]
    if todo.description:
        lines += ["", todo.description]
    if todo.comments:
        lines += ["", "Notes so far:"]
        # A core TodoComment carries .body; an Atom Todo keeps notes as plain text.
        lines += [f"- {getattr(comment, 'body', comment)}" for comment in todo.comments]
    return "\n".join(lines)
