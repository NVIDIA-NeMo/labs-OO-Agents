# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOOA coding agent hosted by the ACP adapter."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nooa_cli.tools.repo_tools import RepoTools

from nooa import Context, hidden
from nooa.agentdoc import doc
from nooa.interactive import (
    InteractiveAgent,
    RespondReason,
    RespondResult,
    SummarizationConfig,
    install_summarizer,
)
from nooa.mcp import MCPTool
from nooa.tools import ShellTools, TodoManager

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

__all__ = ["CodingInteractiveAgent", "RespondReason"]


class CodingInteractiveAgent(InteractiveAgent):
    """A careful coding agent working in one local repository."""

    cwd: Path
    shell: ShellTools
    repo: RepoTools
    todo: TodoManager
    mcp: dict[str, MCPTool]

    def __init__(
        self,
        llm: "UnifiedLLM",
        cwd: Path,
        mcp: dict[str, MCPTool] | None = None,
    ) -> None:
        super().__init__(llm=llm)  # type: ignore[no-untyped-call]
        self.cwd = cwd.resolve()
        self.shell = ShellTools(cwd=str(self.cwd))
        self.repo = RepoTools(root=str(self.cwd), session=self.shell.session)
        self.todo = TodoManager()
        self.mcp = mcp or {}
        self.context_manager["python_tools"] = Context(doc(RepoTools, ShellTools), prefix=True)
        self.context_manager["todo"] = Context(doc(TodoManager), prefix=True)
        self.context_manager["todo_status"] = Context(expr="self.todo.status()")
        if self.mcp:
            mcp_docs = "\n\n".join(
                f"self.mcp[{name!r}]:\n{doc(tool)}" for name, tool in self.mcp.items()
            )
            self.context_manager["mcp"] = Context(mcp_docs, prefix=True)
        install_summarizer(SummarizationConfig(), self)

    @hidden
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        """Fulfill the newest software-engineering request in the working directory.

        Inspect repository instructions and relevant code before editing. Use
        ``self.shell`` for files and commands, ``self.repo`` for definitions and
        references, ``self.todo`` for multi-step work, and ``self.mcp`` for MCP
        servers supplied by the client. Preserve unrelated worktree changes and
        avoid destructive commands.

        Complete and verify the work before returning ``RespondReason.DONE``.
        Send each user-facing answer or question through ``self.message()`` as a
        complete Markdown document. Return ``RespondReason.NEED_INPUT`` only when
        human input is required, and ``RespondReason.WAIT`` only for an active
        background job. Always provide a concrete explanation to
        ``return_result()``.
        """
        ...

    @hidden
    async def close(self) -> None:
        await self.queue_manager.shutdown()
        await self.shell.close()
        await self.llm.aclose()
