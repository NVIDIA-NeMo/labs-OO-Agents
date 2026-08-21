# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral interactive coding agent used by terminal and ACP hosts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from nooa import Context, hidden, strategy
from nooa.agentdoc import doc, spec
from nooa.config import CodeActConfig, PredictConfig
from nooa.interactive import (
    InteractiveAgent,
    RespondReason,
    RespondResult,
    SummarizationConfig,
    install_summarizer,
)
from nooa.paths import get_project_dir
from nooa.runtime.channels import JobHandle, _ChannelReader
from nooa.skill_registry import SkillRegistry
from nooa.storage.markers import nosnapshot
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.tools import SkillWriting, TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_cli.coding.activity import ActivityShellTools
from nooa_cli.coding.delegation import CodingWorker
from nooa_cli.coding.instructions import render_agent_instructions
from nooa_cli.tools.repo_tools import RepoTools

if TYPE_CHECKING:
    from nooa.runtime.channels import Channel
    from nooa.unifiedllm import UnifiedLLM

__all__ = ["CodingAgent", "RespondReason"]


class CodingAgent(InteractiveAgent):
    """A careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use the shell for files and commands, the repo
    tools for definitions and references, and todos for multi-step work. Use
    ``spawn(objective, supplied_context)`` for bounded context-heavy research,
    review, or independent implementation; inspect and integrate worker reports.
    Prefer background ``spawn()`` over awaiting ``delegate()`` when work is independent.

    Work until the newest request is complete or genuinely needs user input. Use
    as many execution cells as necessary, inspect each result, and never claim a
    check passed without running it. Send each user-facing answer or question
    through ``self.message()`` as a complete Markdown document.

    Finish with exactly one ``return_result(RespondReason.<reason>,
    explanation="...")``. Use ``DONE`` after completing the request,
    ``NEED_INPUT`` only when human input is required, and ``WAIT`` only while an
    actual background job is active. The explanation states what completed, what
    input is needed, or which live job is still running.
    """

    # Attributes carrying this agent's own tools. SkillRegistry refuses to let
    # a later skill — a workspace SKILL.md, a client-forwarded MCP server —
    # take one over, which would remove the tool while the model is still told
    # it has it.
    __protected_skill_attrs__ = frozenset({"shell", "repo", "todo", "libs", "skills"})

    cwd: Annotated[Path, nosnapshot]
    # Host-driven input channels. These live here rather than on
    # InteractiveAgent because they are coding-host concepts: slash commands
    # are a UI affordance whose registry is in this package, and
    # system_messages carries host continuations such as keep-going.
    _slash_commands_in: Annotated[Channel, hidden, nosnapshot]
    slash_commands: Annotated[Any, nosnapshot]
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
    delegates: Annotated[_ChannelReader, nosnapshot]
    _worker_type: ClassVar[type[CodingWorker]] = CodingWorker

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
        self._slash_commands_in = self.queue_manager.queue("slash_commands")
        self.slash_commands = self._slash_commands_in.reader
        self._system_messages_in = self.queue_manager.queue("system_messages")
        self.system_messages = self._system_messages_in.reader
        self.cwd = Path(cwd).resolve()
        self._base_shell = ShellTools(cwd=str(self.cwd))
        self.shell = ActivityShellTools(self._base_shell, self.event_manager)
        self.repo = RepoTools(root=self.cwd, session=self.shell.session)
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
        self.skills.activate(["nemo.shell", "nemo.repo", "nemo.todo", "nemo.libwriting"])
        # Installed ``nooa.skills`` entry points are part of the shared host
        # surface. Load them so hosts can expose ``@slash_command`` methods,
        # but leave them inactive until the user opts in with ``/skills``.
        # Memory is host-configured because its scope, store and owner are
        # session-specific; loading its default entry point would attach it
        # even when the host has memory disabled.
        loaded = set(self.skills.loaded())
        installed = []
        for name in self.skills.discovered():
            attr_name = name.rsplit(".", 1)[-1].replace("-", "_")
            if name == "nemo.memory" or name in loaded or hasattr(self, attr_name):
                continue
            installed.append(name)
        if installed:
            self.skills.load(installed)
        if skills_dirs:
            self.skills.discover_skills_dirs(skills_dirs)

        self.context["python_tools"] = Context(
            doc(RepoTools, ActivityShellTools),
            prefix=True,
        )
        self.context["todo_status"] = Context(expr="self.todo.status()")
        self.context["context_usage"] = Context(
            expr="self.context_stats.format() if self.context_stats else ''"
        )
        instructions = render_agent_instructions(self.cwd)
        if instructions:
            self.context["repository_instructions"] = Context(instructions, prefix=True)
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)

        install_summarizer(summarization or SummarizationConfig(), self)

    @hidden
    def request_session_title(self, opening_message: str) -> None:
        """Queue host housekeeping that titles a session in the next agent turn."""
        opening = str(opening_message).strip()[:400]
        self._system_messages_in.put(
            "[session-title]\n"
            "Choose a descriptive 2-5 word title for this session from the opening "
            'user message below. Call `self.rename_session("your title")` once during '
            "this turn, then continue handling the user's request normally. Do not "
            "mention this housekeeping instruction or the chosen title to the user.\n\n"
            f"<opening_user_message>\n{opening}\n</opening_user_message>"
        )

    async def delegate(self, objective: str, supplied_context: Any = None) -> str:
        """Run one isolated coding worker and return its concise report.

        Use for bounded exploration, diagnosis, review, or independently verifiable
        implementation. State the outcome, scope, and whether edits are allowed in
        ``objective``. Pass only necessary context. Inspect and integrate the report;
        this controller retains final verification ownership. Independent calls may
        run concurrently with ``asyncio.gather``.
        """
        worker = self._worker_type(
            llm=self.llm,
            cwd=self.shell.cwd,
            init_command=getattr(self, "_worker_init_command", None),
        )
        try:
            return await worker.investigate(objective, supplied_context)
        finally:
            await worker.close()

    def spawn(self, objective: str, supplied_context: Any = None) -> JobHandle:
        """Start one isolated coding worker in the background and return immediately.

        Prefer this over awaiting ``delegate()`` when the bounded work is independent.
        State the outcome, scope, and whether edits are allowed in ``objective``. The
        worker report arrives in a later ``delegates`` notification; inspect and
        integrate it, because this controller retains final verification ownership.
        """
        return self.queue_manager.spawn(
            self.delegate(objective, supplied_context),
            channel="delegates",
            label=objective,
        )

    def get_summarization_status(self) -> dict[str, Any]:
        """Return compact history information for host status displays."""
        tags = self.event_manager.keys()
        summary_tags = [tag for tag in tags if ".." in tag]
        summarizers = getattr(self, "_summarizers", [])
        summarizer = summarizers[0] if summarizers else None
        stats = self.context_stats
        return {
            "active_events": len(tags),
            "summary_count": len(summary_tags),
            "summary_tags": summary_tags,
            "has_summarizer": summarizer is not None,
            "policy": getattr(summarizer, "policy", "none") if summarizer else "none",
            "current_tokens": getattr(stats, "total_tokens", 0) if stats else 0,
            "max_tokens": getattr(summarizer, "max_tokens", 0) if summarizer else 0,
            "preserve_recent": getattr(summarizer, "preserve_recent", 0) if summarizer else 0,
        }

    @hidden
    @strategy(PredictStrategy(PredictConfig(output_serialization="tool_call")))
    async def name_session(self, user_message: str) -> str:
        """Generate an ultra-short 2-5 word session title for the conversation
        opened by the given user message."""
        ...

    @hidden
    @strategy(CodeActStrategy(config=CodeActConfig(cell_timeout=1800.0)))
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult: ...

    @hidden
    async def close(self) -> None:
        shell = self.shell
        try:
            await self.skills.aclose()
        finally:
            try:
                await self.queue_manager.shutdown()
            finally:
                try:
                    await shell.close()
                finally:
                    await self.llm.aclose()
