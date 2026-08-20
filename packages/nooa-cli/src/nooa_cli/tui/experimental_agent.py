# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TUI agent using the experimental single-tool CodeAct strategy."""

from __future__ import annotations

from typing import Any

from nooa import hidden, strategy
from nooa.config import CodeActConfig
from nooa.interactive import RespondResult
from nooa.strategies.codeact_experimental import CodeActExperimental
from nooa_cli.coding.agent import CodingAgent
from nooa_cli.coding.delegation import CodingWorker


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


class ExperimentalTUIAgent(CodingAgent):
    """Opt-in TUI coding agent backed by :class:`CodeActExperimental`.

    Select it without changing the default TUI behavior::

        nooa tui --agent nooa_cli.tui.experimental_agent:ExperimentalTUIAgent
    """

    _worker_type = ExperimentalCodingWorker

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
