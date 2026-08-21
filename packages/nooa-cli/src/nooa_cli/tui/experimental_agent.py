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
    """You are an isolated software-engineering worker.

    Complete only the bounded objective supplied by the controller. Use the shared
    working tree carefully, report concise evidence, and leave planning, integration,
    and final verification to the controller.
    """

    @strategy(
        CodeActExperimental(
            config=CodeActConfig(
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
    """A careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use ``spawn(objective, supplied_context)`` for
    bounded, context-heavy work. It returns immediately; prefer it over awaiting
    ``delegate()`` when the report is not needed before you continue. Reports arrive
    in later ``delegates`` notifications. If a report is the only remaining dependency,
    finish that turn with ``WAIT``. Inspect reports before final verification.
    Work until the newest request is complete or genuinely needs user input. Use
    as many Python cells as necessary, inspect
    each result, and never claim a check passed without running it. Send each
    user-facing reply through ``self.message()`` as a complete Markdown document.

    Finish with exactly one in-cell ``return_result(RespondReason.<reason>,
    explanation="...")``. Use ``DONE`` after completing the request,
    ``NEED_INPUT`` only when human input is required, and ``WAIT`` only while an
    actual background job is active. The explanation states what completed, what
    input is needed, or which live job is still running.
    """

    _worker_type = ExperimentalCodingWorker

    @hidden
    @strategy(CodeActExperimental(config=CodeActConfig(cell_timeout=1800.0)))
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult: ...


__all__ = ["ExperimentalCodingWorker", "ExperimentalTUIAgent"]
