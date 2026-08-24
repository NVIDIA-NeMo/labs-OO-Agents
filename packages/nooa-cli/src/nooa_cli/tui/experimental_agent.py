# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default TUI agent using the single-tool CodeAct strategy."""

from __future__ import annotations

from typing import Any

from nooa import hidden, strategy
from nooa.config import CodeActConfig
from nooa.interactive import RespondResult
from nooa.strategies import CodeActExperimental
from nooa_cli.coding.agent import CodingAgent
from nooa_cli.coding.context_rendering import SafeDelegationPrefill
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
        ``supplied_context["context"]`` separately. Record findings on that Todo through
        ``self.todo``. Return concise findings or changes rather than a
        raw transcript.
        """
        ...


class ExperimentalTUIAgent(CodingAgent):
    """You are a careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use ``spawn(objective, supplied_context)`` for
    bounded, context-heavy work. It returns immediately; prefer it over awaiting
    ``delegate()`` when the report is not needed before you continue. Run concurrent
    delegates only for read-only work or when each mutating worker has its own isolated
    worktree; otherwise serialize mutations because workers share the current checkout.
    Reports arrive in ``notification["delegates"]`` as dictionaries with ``objective``
    and ``report``.
    If a report is the only remaining dependency, finish that turn with an in-cell
    ``return_result(RespondReason.WAIT, explanation="...")``. Inspect reports before
    final verification.
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
