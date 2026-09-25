# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral interactive coding agent, run by a Session."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from nooa import Context, hidden, no_trace, strategy
from nooa.agentdoc import doc, spec
from nooa.config import CodeActConfig, PredictConfig
from nooa.interactive import (
    Done,
    InteractiveAgent,
    NeedInput,
    SummarizationConfig,
    Waiting,
    install_summarizer,
)
from nooa.paths import get_project_dir
from nooa.skill_registry import SkillRegistry
from nooa.storage.markers import nosnapshot
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.tools import MethodWriting, SkillWriting, Todo, TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_coder.coding.activity import ActivityShellTools
from nooa_coder.coding.instructions import render_agent_instructions
from nooa_coder.coding.slash_commands import CodingSlashCommandRegistry

# Visible to generated cells (cells see this module's globals): the model
# builds TaskResult for unattended turns, matches what arrives on the
# ``delegates`` channel and catches the delegation errors.
from nooa_coder.session.items import (
    ChildFailed,
    ChildFailedError,
    ChildQuestion,
    ChildRef,
    ChildResult,
    TaskResult,
)
from nooa_coder.session.registry import DepthLimitError
from nooa_coder.tools.repo_tools import RepoTools

with hidden:
    from nooa.agents import TokenBudgetSummarizer
    from nooa.runtime.channels import Channel, ChannelReader
    from nooa_coder.coding.conditions import require_result
    from nooa_coder.session.port import SessionPort

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

__all__ = [
    "ChildFailed",
    "ChildFailedError",
    "ChildQuestion",
    "ChildRef",
    "ChildResult",
    "CodingAgent",
    "DepthLimitError",
    "Done",
    "NeedInput",
    "TaskResult",
    "Waiting",
    "session_title_request",
]


class CodingAgent(InteractiveAgent):
    """You are a careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use the shell for files and commands, the repo
    tools for definitions and references, and todos for multi-step work.
    ``cd`` moves ``self.shell`` and ``self.repo`` together; the repository
    root bounds repo searches.

    Delegate bounded, context-heavy work (exploration, diagnosis, review, an
    independently verifiable change) to a child session with its own history:
    ``done = await self.delegate(description, prompt)`` waits for it and returns
    its ``Done``, whose ``result`` is a ``TaskResult`` (read ``result.report``).
    ``await self.spawn(description, prompt)`` starts one and returns at once; when
    nothing else is left, end the turn with
    ``Waiting(explanation=..., on=["delegates"])``. Its outcome arrives in a later
    turn under ``notification["delegates"]``: a ``ChildResult`` (``item.done``), a
    ``ChildQuestion`` from a retained child (``await item.answer(...)``) or a
    ``ChildFailed`` (``item.error``). Never predict or make up a pending child's
    result, poll, or sleep to wait. Children share this checkout: run concurrent
    children only for read-only work and serialize edits. Children can delegate
    in turn only down to a fixed depth; past it ``DepthLimitError`` is raised.
    Inspect and verify what a child reports before relying on it.

    For multi-step work, activate the current Todo. Keep its title and description
    aligned with the current understanding, and append comments for material findings,
    decisions, completed steps, and verification—not routine narration.

    Work until the newest request is complete or genuinely needs user input. Use
    as many execution cells as necessary, inspect each result, and never claim a
    check passed without running it. Send each user-facing answer through
    ``self.message()`` as a complete Markdown document.

    End every turn with exactly one ``return_result(...)``: ``Done`` after
    completing the request (reply with ``self.message()`` first),
    ``NeedInput(question=..., options=[...])`` only when a person must answer,
    and ``Waiting(explanation=..., on=[...])`` only while something you started
    is still running, naming the channel or job it waits on.
    """

    # Attributes carrying this agent's own tools. SkillRegistry refuses to let
    # a later skill — a workspace SKILL.md, a client-forwarded MCP server —
    # take one over, which would remove the tool while the model is still told
    # it has it.

    __protected_skill_attrs__ = frozenset(
        {"shell", "repo", "todo", "libs", "skills", "mcp", "workspace_settings", "session"}
    )

    cwd: Annotated[Path, nosnapshot]
    # Host-driven input channels. These live here rather than on
    # InteractiveAgent because they are coding-host concepts: slash commands
    # are a UI affordance whose registry is in this package, and
    # system_messages carries host-provided system input. A host that sends
    # a command's output to the agent puts it on the slash_commands channel.
    _slash_commands_in: Annotated[Channel, hidden, nosnapshot]
    # The command registry the Session lists and runs (Session.commands(),
    # Session.invoke_command()). create_session_agent() installs the /skills and
    # /mcp controls with set_controls().
    slash_commands: Annotated[CodingSlashCommandRegistry, hidden, nosnapshot]
    # Set by the registry that binds itself to this agent (skills use it).
    _command_registry: Annotated[Any, hidden, nosnapshot]
    _system_messages_in: Annotated[Channel, hidden, nosnapshot]
    system_messages: Annotated[Any, nosnapshot]
    shell: Annotated[ActivityShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]
    todo: TodoManager
    libs: Annotated[SkillWriting, nosnapshot]
    skills: Annotated[SkillRegistry, nosnapshot]
    _base_shell: Annotated[ShellTools, hidden, nosnapshot]
    _summarizers: Annotated[list[Any], hidden, nosnapshot]
    _delegates_in: Annotated[Any, hidden, nosnapshot]
    delegates: Annotated[ChannelReader, nosnapshot]
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
        self.todo = TodoManager()
        # Libraries live at <project>/.nooa/libs. get_project_dir() resolves
        # that per process, which is what a one-workspace host like the TUI
        # wants. A host serving several workspaces at once must say which
        # project it means, or every session shares one directory — and
        # SkillWriting puts it on sys.path and activates local.*, so that
        # would expose one workspace's agent-authored code to another.

        self.libs = SkillWriting(self, path=libs_dir or get_project_dir("libs"))

        self.skills = SkillRegistry(self)
        self.skills.register("nemo.shell", self.shell)
        self.skills.register("nemo.repo", self.repo)
        self.skills.register("nemo.todo", self.todo)
        self.skills.register("nemo.libwriting", self.libs)
        self.skills.register("nemo.methodwriting", MethodWriting())
        self.skills.activate(
            ["nemo.shell", "nemo.repo", "nemo.todo", "nemo.libwriting", "nemo.methodwriting"]
        )
        # Installed ``nooa.skills`` entry points are part of the shared host
        # surface. Load them so hosts can expose ``@slash_command`` methods,
        # but leave them inactive until the user opts in with ``/skills``.
        # Memory integration is deferred; do not auto-load its entry point.
        # Also ignore the retired web publisher entry point in older installed
        # package metadata.
        loaded = set(self.skills.loaded())
        installed = []
        for name in self.skills.discovered():
            attr_name = name.rsplit(".", 1)[-1].replace("-", "_")
            if name in {"nemo.memory", "nemo.web"} or name in loaded or hasattr(self, attr_name):
                continue
            installed.append(name)
        if installed:
            self.skills.load(installed)
        if skills_dirs:
            self.skills.discover_skills_dirs(skills_dirs)
        self.slash_commands = CodingSlashCommandRegistry(self, skills_dirs=skills_dirs or ())

        self.context["python_cell_tools"] = Context(
            doc(RepoTools, ActivityShellTools, concise=True),
            prefix=True,
        )
        self.context["todo_status"] = Context(expr="self.todo.status()")
        self.context["coding_state"] = Context(expr="self._coding_state_context()")
        self.context["context_usage"] = Context(
            expr="self.context_stats.format() if self.context_stats else ''"
        )
        instructions = render_agent_instructions(self.cwd)
        if instructions:
            self.context["repository_instructions"] = Context(instructions, prefix=True)
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)

        self._summarization = summarization or SummarizationConfig()
        install_summarizer(self._summarization, self)

    @no_trace
    def _coding_state_context(self) -> str:
        """Describe coding-specific state without exposing stored values."""
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
            f"`self.v`: {count} persistent vars — inspect: `print(self.v.items())`; "
            "remove one: `del self.v.<name>`; clear all: `self.v.clear()`"
        )

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
        context: Any = None,
        model: str | None = None,
    ) -> Done:
        """Run a child session on ``prompt``, wait for it, and return its ``Done``.

        The child is a new session of this agent with its own history, working
        unattended in this checkout; it is closed after it returns. It must end
        with a ``TaskResult``, so ``done.result.report`` (plus
        ``solution_description``, ``evidence`` and ``how_to_verify``) is its
        report; ``done.explanation`` is its one-line status. Use this only when
        you need the report before continuing; otherwise use ``spawn()``.

        Pass a ``Todo`` as ``prompt`` to hand over that task: the child gets its
        title, description and comments as the prompt, and the report is added
        to the Todo as a comment when the child returns.

        Raises ``ChildFailedError`` when the child's turn fails or it is closed
        first, and ``DepthLimitError`` when a child would exceed the depth
        limit; catch them to report or recover.

        Args:
            description: Short label (2-6 words); the child's name and title.
            prompt: The full task: outcome, scope, whether edits are allowed.
            context: Optional data (a pydantic model or JSON data) sent with it.
            model: Model alias for the child; yours when omitted.
        """
        todo = prompt if isinstance(prompt, Todo) else None
        text = _todo_prompt(todo) if todo is not None else str(prompt)
        child = await self._port().delegate(
            description, text, context=context, model=model, retain=False
        )
        done = await child.wait()
        if todo is not None:
            self.todo.comment(todo, f"Delegated to {description!r}: {_report_text(done)}")
        return done

    async def spawn(
        self,
        description: str,
        prompt: str | Todo,
        *,
        context: Any = None,
        model: str | None = None,
        retain: bool = False,
    ) -> ChildRef:
        """Start a child session on ``prompt`` and return its ``ChildRef`` at once.

        Keep working, and when only the child's outcome is left end the turn
        with ``Waiting(explanation=..., on=["delegates"])``. The outcome arrives
        in a later turn under ``notification["delegates"]`` as a ``ChildResult``
        (``item.done.result`` is a ``TaskResult``), a ``ChildQuestion`` or a
        ``ChildFailed``; ``item.child.id`` matches the returned ref, which you
        can keep in ``self.v``. A ``Todo`` prompt is sent as text; comment on
        the Todo yourself when the result arrives.

        Args:
            description: Short label (2-6 words); the child's name and title.
            prompt: The full task: outcome, scope, whether edits are allowed.
            context: Optional data (a pydantic model or JSON data) sent with it.
            model: Model alias for the child; yours when omitted.
            retain: ``False``: unattended, returns a ``TaskResult`` and closes.
                ``True``: a conversation partner that may ask questions
                (``ChildQuestion``) and takes more messages with
                ``await ref.send(text)``; close it with ``await ref.close()``.
        """
        text = _todo_prompt(prompt) if isinstance(prompt, Todo) else str(prompt)
        return await self._port().delegate(
            description, text, context=context, model=model, retain=retain
        )

    def children(self) -> list[ChildRef]:
        """Handles on the child sessions you started, running and closed."""
        return self._port().children()

    @hidden
    def after_restore(self) -> None:
        """Called by the session registry after it restores a snapshot into this agent."""
        from nooa_coder.workspace.options import drop_stale_memory_context

        drop_stale_memory_context(self)

    @hidden
    def _port(self) -> SessionPort:
        if self.session is None:
            raise RuntimeError("This agent is not running in a session")
        return self.session

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
    @strategy(PredictStrategy(PredictConfig(output_serialization="tool_call")))
    async def name_session(self, user_message: str) -> str:
        """Generate an ultra-short 2-5 word session title for the conversation
        opened by the given user message."""
        ...

    @hidden
    @strategy(CodeActStrategy(config=CodeActConfig(cell_timeout=1800.0)))
    async def handle(self, notification: dict[str, list[Any]]) -> Done | NeedInput | Waiting:
        """Handle one interactive turn: the newest request and anything else that arrived.

        ``notification`` maps a channel name to the items that arrived on it:
        ``"user_messages"`` (text from the person), ``"system_messages"``
        (host housekeeping), ``"slash_commands"`` (command output sent to you)
        and ``"delegates"`` (results of children you spawned). Do all the work
        the request needs before ending the turn; use as many cells as it
        takes, and run checks rather than assume them.

        End with exactly one ``return_result(...)``:

        - ``Done(explanation=...)`` when the request is complete. Send the
          reply with ``self.message()`` first; ``explanation`` is a short
          status line for the host, not the reply::

              self.message("Fixed the off-by-one in `parse()`; the parser tests pass.")
              return_result(Done(explanation="fixed parse() and ran its tests"))

        - ``NeedInput(question=..., options=[...])`` when you cannot go on
          without an answer from the person. The host shows ``question``, so
          do not also send it with ``self.message()``. Use ``options`` for a
          choice; the answer arrives in the next notification.

        - ``Waiting(explanation=..., on=[...])`` when a job you started is
          still running and nothing else is left to do. ``on`` names what
          you wait for, for example ``["jobs"]``; the next turn starts when
          it delivers.

        Python locals live for one method call; when the call returns they
        are gone. Anything you need later goes in ``self.v`` (durable,
        snapshot-backed) or the todo list. Do not rely on a variable from an
        earlier call.
        """
        ...

    @hidden
    @strategy(
        CodeActStrategy(config=CodeActConfig(cell_timeout=1800.0, postconditions=[require_result]))
    )
    async def handle_batch(self, notification: dict[str, list[Any]]) -> Done | Waiting:
        """Work on one task unattended and return a structured result.

        The task is the text in ``notification["user_messages"]``; data sent
        with it is in ``notification.get("context", [])``. Nobody is watching
        and nobody can answer a question, and ``self.message()`` reaches
        nobody: the result is the only thing the requester reads.

        Do the task completely, verify it, then end with::

            return_result(Done(
                explanation="fixed the parser bug",
                result=TaskResult(
                    solution_description="What was done, including modified paths",
                    evidence="Commands run and what they showed",
                    how_to_verify="How someone else can check it",
                    report="The full report for the requester",
                ),
            ))

        If something blocks you, still return ``Done`` with a ``TaskResult``
        that says what blocked you and what you tried. Return
        ``Waiting(explanation=..., on=[...])`` only while a job you started
        is still running.

        Python locals live for one method call; when the call returns they
        are gone. Anything you need later goes in ``self.v`` (durable,
        snapshot-backed) or the todo list. Do not rely on a variable from an
        earlier call.
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
spec(CodingAgent, "session_port_visible", hidden=True)


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
def _todo_prompt(todo: Todo) -> str:
    """A Todo as the text a child works from: title, description, comments."""
    lines = [f"Task: {todo.title}"]
    if todo.description:
        lines += ["", todo.description]
    if todo.comments:
        lines += ["", "Notes so far:"]
        lines += [f"- {comment.body}" for comment in todo.comments]
    return "\n".join(lines)


@hidden
def _report_text(done: Done) -> str:
    """The report in a child's Done: the TaskResult's report, else its summary fields."""
    result = done.result
    if isinstance(result, TaskResult):
        if result.report.strip():
            return result.report
        return (
            f"{result.solution_description}\nEvidence: {result.evidence}\n"
            f"How to verify: {result.how_to_verify}"
        )
    return done.explanation
