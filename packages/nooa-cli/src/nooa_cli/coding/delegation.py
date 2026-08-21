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
from nooa.tools.shell_tools import ShellTools
from nooa_cli.tools.repo_tools import RepoTools

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM


class CodingWorker(Agent, context={"context_usage": None}):
    """You are an isolated software-engineering worker.

    Complete only the bounded objective supplied by the controller. Use the shared
    working tree carefully, report concise evidence, and leave planning, integration,
    and final verification to the controller.
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
                max_retries=6,
                text_only_stop_behavior="synthetic_comment",
            )
        )
    )
    async def investigate(self, objective: str, supplied_context: Any = None) -> str:
        """Complete one bounded coding subtask and return a concise report.

        Read relevant files before drawing conclusions. Make edits only when the
        objective explicitly requests implementation. Report modified paths. Name each
        verification command and its observed outcome; if none ran, state why. Return
        concise findings or changes rather than a raw transcript.
        """
        ...
