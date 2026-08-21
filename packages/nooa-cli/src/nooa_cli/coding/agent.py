# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral interactive coding agent used by terminal and ACP hosts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from nooa import Context, hidden, strategy
from nooa.agentdoc import doc, spec
from nooa.config import CodeActConfig
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
from nooa.strategies import CodeActStrategy
from nooa.tools import SkillWriting, TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_cli.coding.activity import ActivityShellTools
from nooa_cli.coding.delegation import CodingWorker
from nooa_cli.coding.instructions import render_agent_instructions
from nooa_cli.tools.repo_tools import RepoTools

if TYPE_CHECKING:
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

    cwd: Annotated[Path, nosnapshot]
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
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, **kwargs)
        self.cwd = Path(cwd).resolve()
        self._base_shell = ShellTools(cwd=str(self.cwd))
        self.shell = ActivityShellTools(self._base_shell, self.event_manager)
        self.repo = RepoTools(root=self.cwd, session=self.shell.session)
        self._delegates_in = self.queue_manager.queue("delegates")
        self.delegates = self._delegates_in.reader
        self.todo = TodoManager()
        self.libs = SkillWriting(self, path=get_project_dir("libs"))

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
    @strategy(CodeActStrategy(config=CodeActConfig(cell_timeout=1800.0)))
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult: ...

    @hidden
    async def close(self) -> None:
        await self.queue_manager.shutdown()
        await self.shell.close()
        await self.llm.aclose()
