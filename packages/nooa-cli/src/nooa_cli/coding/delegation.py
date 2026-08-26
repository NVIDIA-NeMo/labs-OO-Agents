# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable context-isolated coding workers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from nooa import Agent, Context, strategy
from nooa.agentdoc import doc
from nooa.config import CodeActConfig
from nooa.interactive import SummarizationConfig, install_summarizer
from nooa.storage.markers import nosnapshot
from nooa.strategies import CodeActStrategy
from nooa.tools import TodoManager
from nooa.tools.shell_tools import ShellTools
from nooa_cli.coding.context_rendering import SafeDelegationPrefill
from nooa_cli.coding.instructions import render_agent_instructions
from nooa_cli.tools.repo_tools import RepoTools

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

    shell: Annotated[ShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]
    todo: TodoManager

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
        self.shell = ShellTools(cwd=str(cwd), init_command=init_command)
        self.repo = RepoTools(root=str(cwd), session=self.shell.session)
        self.todo = todo or TodoManager()
        self.context_manager["python_cell_tools"] = Context(
            doc(RepoTools, ShellTools, TodoManager), prefix=True
        )
        instructions = render_agent_instructions(cwd)
        if instructions:
            self.context_manager["repository_instructions"] = Context(instructions, prefix=True)
        install_summarizer(summarization or SummarizationConfig(), self)

    async def close(self) -> None:
        """Release the worker's shell without closing the controller-owned LLM."""
        await self.shell.close()

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(
                max_retries=6,
                text_only_stop_behavior="synthetic_comment",
                prefill=SafeDelegationPrefill(),
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
