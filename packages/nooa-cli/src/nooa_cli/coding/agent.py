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
from nooa.tools import MethodWriting, SkillWriting, Todo, TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_cli.coding.activity import ActivityShellTools
from nooa_cli.coding.delegation import CodingWorker
from nooa_cli.coding.instructions import render_agent_instructions
from nooa_cli.tools.repo_tools import RepoTools

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

__all__ = ["CodingAgent", "RespondReason"]


class CodingAgent(InteractiveAgent):
    """You are a careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use the shell for files and commands, the repo
    tools for definitions and references, and todos for multi-step work. Use
    ``spawn(objective, supplied_context)`` for bounded context-heavy work. It
    returns immediately; prefer it over awaiting ``delegate()`` when the report is
    not needed before you continue. Reports arrive in later ``delegates``
    notifications under ``notification["delegates"]`` as dictionaries containing
    ``objective`` and ``report``. Inspect and integrate them before final verification.

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
        self.skills.register("nemo.methodwriting", MethodWriting())
        self.skills.activate(
            ["nemo.shell", "nemo.repo", "nemo.todo", "nemo.libwriting", "nemo.methodwriting"]
        )
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

    async def delegate(self, objective: str | Todo, supplied_context: Any = None) -> str:
        """Run one isolated coding worker and return its concise report.

        Pass a :class:`Todo` to make it the worker's task. The worker receives an
        independent task copy and can record comments or variables with ``self.todo``;
        those changes are merged into this agent's Todo before this method returns.
        String objectives retain the existing behavior.

        Use delegation for bounded exploration, diagnosis, review, or independently
        verifiable implementation. Await this only when its report is required before
        continuing; otherwise prefer ``spawn()``. Inspect and integrate the report
        because this controller retains final verification ownership.
        """
        todo_base = self.todo._copy_todo(objective) if isinstance(objective, Todo) else None
        worker_todos = TodoManager._with_todo(todo_base) if todo_base is not None else None
        worker = self._worker_type(
            llm=self.llm,
            cwd=self.shell.cwd,
            init_command=getattr(self, "_worker_init_command", None),
            **({"todo": worker_todos} if worker_todos is not None else {}),
        )
        worker_objective = todo_base.title if todo_base is not None else objective
        worker_context = worker_todos.get(todo_base) if todo_base is not None else supplied_context
        if todo_base is not None and supplied_context is not None:
            worker_context = {"todo": worker_context, "context": supplied_context}
        try:
            report = await worker.investigate(worker_objective, worker_context)
            updated = worker_todos.get(todo_base) if todo_base is not None else None
            if todo_base is not None and updated is None:
                raise RuntimeError(f"delegated todo {todo_base.id!r} disappeared")
        finally:
            await worker.close()
        if todo_base is not None:
            self.todo._merge_todo(updated, base=todo_base)
        return report

    async def _delegation_report(
        self, objective: str | Todo, supplied_context: Any
    ) -> dict[str, str]:
        """Return a correlatable queue item after delegation and Todo merging."""
        objective_text = objective.title if isinstance(objective, Todo) else objective
        result = {
            "objective": objective_text,
            "report": await self.delegate(objective, supplied_context),
        }
        if isinstance(objective, Todo):
            result["todo_id"] = objective.id
        return result

    @staticmethod
    def _delegation_label(objective: str, label: str | None = None, max_length: int = 80) -> str:
        """Return a concise display label without discarding the full objective."""
        source = label if label is not None else objective.splitlines()[0]
        compact = " ".join(source.split())
        if label is None:
            first_sentence, separator, _rest = compact.partition(".")
            compact = f"{first_sentence}." if separator else compact
        if len(compact) <= max_length:
            return compact or "Delegated task"
        return f"{compact[: max_length - 1].rstrip()}…"

    def spawn(
        self,
        objective: str | Todo,
        supplied_context: Any = None,
        *,
        label: str | None = None,
    ) -> JobHandle:
        """Start one isolated coding worker and return immediately.

        Prefer this over awaiting ``delegate()`` when the report is not required before
        continuing. State the outcome, scope, and whether edits are allowed in
        ``objective``. Continue useful controller work while it runs. Its report arrives
        in a later ``delegates`` notification. If the report is the only remaining
        dependency, finish the current turn with ``WAIT``; inspect and integrate the
        later report before final verification. Each notification item is
        ``{"objective": <str>, "report": <str>}``, so concurrent jobs remain identifiable.
        The host displays a short label derived from the objective while the worker and
        notification retain the complete text. Pass ``label`` to override the display
        text without changing the worker objective.
        """
        objective_text = objective.title if isinstance(objective, Todo) else objective
        return self.queue_manager.spawn(
            self._delegation_report(objective, supplied_context),
            channel="delegates",
            label=self._delegation_label(objective_text, label),
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
