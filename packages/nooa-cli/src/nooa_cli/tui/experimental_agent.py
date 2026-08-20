# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TUI agent using the experimental single-tool CodeAct strategy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nooa import hidden, strategy
from nooa.config import CodeActConfig
from nooa.interactive import RespondResult
from nooa.strategies.codeact_experimental import CodeActExperimental
from nooa_cli.coding.delegation import CodingWorker
from nooa_cli.tui.agent import TUIAgent


class ExperimentalCodingWorker(CodingWorker):
    """Coding worker whose provider-facing tool surface is only ``python_cell``."""

    @strategy(
        CodeActExperimental(
            config=CodeActConfig(
                max_iterations=120,
                max_retries=6,
                text_only_stop_behavior="synthetic_comment",
            )
        )
    )
    async def investigate(self, objective: str, supplied_context: Any = None) -> str:
        """Complete one bounded coding subtask and return a concise report.

        Read relevant files before drawing conclusions. Make edits only when the
        objective explicitly requests implementation. Run a focused check when
        practical. Return paths, findings or changes, and observed verification;
        do not return a raw transcript.
        """
        ...


class ExperimentalTUIAgent(TUIAgent):
    """Opt-in TUI coding agent backed by :class:`CodeActExperimental`.

    Select it without changing the default TUI behavior::

        nooa tui --agent nooa_cli.tui.experimental_agent:ExperimentalTUIAgent
    """

    _worker_type = ExperimentalCodingWorker

    def __init__(
        self,
        llm=None,
        *,
        cwd: str | Path = ".",
        skills_dirs: list[Path] | None = None,
        **kwargs: Any,
    ) -> None:
        # Custom TUI agents receive ``cwd`` rather than the host's AgentConfig.
        config = kwargs.pop("config", None) or SimpleNamespace(working_dir=cwd)
        super().__init__(llm=llm, config=config, skills_dirs=skills_dirs, **kwargs)

    @hidden
    @strategy(CodeActExperimental(config=CodeActConfig(cell_timeout=1800.0)))
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        """Fulfill the newest coding request delivered in ``notification``.

        Work until the request is complete or genuinely needs user input. Use
        as many small execution cells as necessary and inspect each result
        before proceeding. Never claim a check passed without running it.

        End with exactly one ``return_result(RespondReason.<reason>,
        explanation="...")``. The explanation must say what completed, what
        input is needed, or which live job is still running.
        """
        ...


__all__ = ["ExperimentalCodingWorker", "ExperimentalTUIAgent"]
