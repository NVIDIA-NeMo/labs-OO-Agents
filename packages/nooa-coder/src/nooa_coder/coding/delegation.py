# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable context-isolated coding workers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from nooa import Agent, Context, hidden, strategy
from nooa.agentdoc import doc
from nooa.agents.summarization import SummarizationConfig, install_summarizer
from nooa.config import CodeActConfig
from nooa.storage.markers import nosnapshot
from nooa.strategies import CodeActStrategy
from nooa.tools import TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_coder.coding.activity import ActivityShellTools
from nooa_coder.coding.instructions import render_agent_instructions
from nooa_coder.tools.repo_tools import RepoTools

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM


class CodingWorker(
    Agent,
    context={"context_usage": None, "todo_status": Context(expr="self.todo.status()")},
):
    """You are an isolated software-engineering worker.

    Complete only the bounded objective supplied by the controller. Use the shared
    working tree carefully, report concise evidence, and leave planning, integration,
    and final verification to the controller.
    """

    shell: Annotated[ActivityShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]
    todo: TodoManager
    _base_shell: Annotated[ShellTools, hidden, nosnapshot]

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        cwd: str | Path,
        summarization: SummarizationConfig | None = None,
        init_command: str | None = None,
        todo: TodoManager | None = None,
    ) -> None:
        super().__init__(llm=llm)
        # ActivityShellTools (like CodingAgent's own self.shell) emits
        # FileEdit/TerminalCommand* into this worker's own event_manager —
        # its own isolated stream, not the controller's. A delegating
        # controller observes it live for UI purposes via on_worker_spawned,
        # without any of it becoming part of the controller's own LLM
        # context (workers stay context-isolated by design; only the final
        # report returned by delegate()/spawn() reaches the controller).
        self._base_shell = ShellTools(cwd=str(cwd), init_command=init_command)
        self.shell = ActivityShellTools(self._base_shell, self.event_manager)
        self.repo = RepoTools(root=str(cwd), session=self.shell.session)
        self.todo = todo or TodoManager()
        self.context_manager["python_cell_tools"] = Context(
            doc(RepoTools, ActivityShellTools, TodoManager, concise=True), prefix=True
        )
        instructions = render_agent_instructions(cwd)
        if instructions:
            self.context_manager["repository_instructions"] = Context(instructions, prefix=True)
        install_summarizer(summarization or SummarizationConfig(), self)

    async def close(self) -> None:
        """Drain background summaries and close the shell, leaving the LLM to its owner."""
        try:
            await self.aclose()
        finally:
            await self.shell.close()

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(
                max_retries=6,
            )
        )
    )
    async def investigate(self, objective: str, supplied_context: Any = None) -> str:
        """Complete one bounded coding subtask and return a concise report.

        Read relevant files before drawing conclusions. Make edits only when the
        objective explicitly requests implementation. Report modified paths. Name each
        verification command and its observed outcome; if none ran, state why. For a
        delegated Todo, ``supplied_context`` is either that Todo or a mapping with
        ``"todo"`` and supplemental ``"context"`` entries. In the mapping form, use
        ``todo = supplied_context["todo"]`` for Todo operations and inspect
        ``supplied_context["context"]`` separately. Keep the Todo title and description
        aligned with the current understanding. Record material findings, decisions,
        completed steps, and verification with ``self.todo.comment(todo, ...)``—not
        routine narration—and record task-scoped values with
        ``self.todo.set_var(todo, key, value)``. Return a concise report rather than a
        raw transcript.
        """
        ...
