# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable context-isolated coding workers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from nooa import Agent, Context, strategy
from nooa.agentdoc import doc
from nooa.config import CodeActConfig
from nooa.interactive import SummarizationConfig, install_summarizer
from nooa.storage.markers import nosnapshot
from nooa.strategies import CodeActStrategy
from nooa.tools.shell_tools import ShellTools
from nooa_cli.tools.repo_tools import RepoTools

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM


class CodingWorker(Agent, context={"context_usage": None}):
    """Context-isolated worker for one bounded coding investigation.

    The worker receives only its explicit objective and supplied context. It has
    an independent event history but shares the controller's model and working tree.
    """

    shell: Annotated[ShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        cwd: str | Path,
        summarization: SummarizationConfig | None = None,
        init_command: str | None = None,
    ) -> None:
        super().__init__(llm=llm)
        self.shell = ShellTools(cwd=str(cwd), init_command=init_command)
        self.repo = RepoTools(root=str(cwd), session=self.shell.session)
        self.context_manager["python_tools"] = Context(doc(RepoTools, ShellTools), prefix=True)
        install_summarizer(summarization or SummarizationConfig(), self)

    async def close(self) -> None:
        """Release the worker's shell without closing the controller-owned LLM."""
        await self.shell.close()

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(
                max_iterations=120,
                max_retries=6,
                text_only_stop_behavior="synthetic_comment",
            )
        )
    )
    async def investigate(self, objective: str, supplied_context: Any = None) -> str:
        """Complete one bounded coding subtask and return a concise report.

        Objective: {objective}

        Supplied context (any object or collection; possibly empty):
        {supplied_context}

        Read relevant files before drawing conclusions. Make edits only when the
        objective explicitly requests implementation. Run a focused check when
        practical. Return paths, findings or changes, and observed verification;
        do not return a raw transcript.
        """
        ...


class CodingDelegationMixin:
    """Expose explicit, context-isolated coding delegation to a controller agent."""

    llm: Any
    shell: Any
    _worker_type: ClassVar[type[CodingWorker]] = CodingWorker

    async def delegate(self, objective: str, supplied_context: Any = None) -> str:
        """Run one isolated coding worker and return its concise report.

        Use for bounded exploration, diagnosis, review, or independently verifiable
        implementation. State the outcome, scope, and whether edits are allowed in
        ``objective``. ``supplied_context`` may be any useful object or collection,
        including Todo items, paths, matches, structured task data, or text. Pass only
        what is necessary. Inspect and integrate the report; the controller retains
        final verification ownership.
        Independent calls may be run concurrently with ``asyncio.gather``.
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
