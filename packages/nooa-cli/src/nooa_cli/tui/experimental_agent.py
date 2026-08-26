# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default TUI agent using the single-tool CodeAct strategy."""

from __future__ import annotations

import datetime  # noqa: F401 — module capability exposed to generated Python cells
import json  # noqa: F401 — module capability exposed to generated Python cells
import re  # noqa: F401 — module capability exposed to generated Python cells
from typing import Any

# Optional data libraries follow the standard InteractiveAgent capability aliases.
try:
    import numpy as np  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import pandas as pd  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import plotly.express as px  # noqa: F401  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import scipy  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import sklearn  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

from nooa import Context, hidden, strategy
from nooa.agentdoc import doc  # noqa: F401 — used by dynamic context expressions
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
    and final verification to the controller. Each ``investigate()`` call gets fresh
    cell locals; reuse them within the call, but do not expect them to survive after
    returning. ``self.shell`` keeps its cwd, so use relative paths and call ``cd`` only
    when intentionally changing directories. When a Todo is supplied, store durable
    task-specific artifacts, findings, and checkpoints on that Todo's ``v`` proxy.
    Plain-string delegations have no Todo-backed durable task state. Do not use
    ``self.v``; keep transient scratch data in cell locals.
    """

    @strategy(
        CodeActExperimental(
            config=CodeActConfig(
                max_retries=6,
                text_only_stop_behavior="synthetic_comment",
                prefill=SafeDelegationPrefill(),
            )
        ),
        context={
            "state": None,
            "execution_context": None,
            "self": Context(expr="doc(type(self), concise=True)", prefix=True),
        },
    )
    async def investigate(self, objective: str, supplied_context: Any = None) -> str:
        """Complete one bounded coding subtask and return a concise report.

        Read relevant files before drawing conclusions. Make edits only when the
        objective explicitly requests implementation. Report modified paths. Name each
        verification command and its observed outcome; if none ran, state why. For a
        delegated Todo, ``supplied_context`` is either that Todo or a mapping with
        ``"todo"`` and supplemental ``"context"`` entries. In the mapping form, use
        ``todo = supplied_context["todo"]`` for Todo operations and inspect
        ``supplied_context["context"]`` separately. When a Todo is present, keep its
        title and description aligned with the current understanding; record material
        findings, decisions, completed steps, and verification with
        ``self.todo.comment(todo, ...)`` (not routine narration), and store durable
        task values on ``todo.v``. For a plain-string objective, ``supplied_context`` is
        optional context rather than a Todo; do not call Todo APIs. Return concise
        findings or changes rather than a raw transcript.
        """
        ...


class ExperimentalTUIAgent(CodingAgent):
    """You are a careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use an RLM-style controller policy: complete requests
    directly when they fit in a few turns. For larger requests, decompose only when
    there are distinct, context-heavy, independently verifiable subtasks; keep tightly
    coupled or small sequential work local. The top-level controller may spawn bounded,
    non-recursive workers for subtasks that benefit from separate context. Use
    ``spawn(objective, supplied_context)`` for
    bounded, context-heavy work. It returns immediately; prefer it over awaiting
    ``delegate()`` when the report is not needed before you continue. Run concurrent
    delegates only for read-only work or when each mutating worker has its own isolated
    worktree; otherwise serialize mutations because workers share the current checkout.
    Reports arrive in a later turn under ``notification["delegates"]`` as dictionaries
    with ``objective`` and ``report``. Never poll a spawned handle with ``state`` or
    ``values``, wait with ``asyncio.sleep()``, call ``self.delegates.get()``, or
    repeatedly inspect queue status. If a report is the only remaining dependency,
    immediately finish that turn with an in-cell
    ``return_result(RespondReason.WAIT, explanation="waiting for <label>")``. The host
    will invoke a new turn when the report arrives. Inspect the report in that
    notification before final verification.
    For multi-step work, activate the current Todo. Keep its title and description
    aligned with the current understanding, and append comments for material findings,
    decisions, completed steps, and verification—not routine narration. Store durable
    cross-task identity, stable environment facts, and long-running coordination on
    ``self.v``. Store task-specific plans, findings, artifacts, and checkpoints on that
    Todo's ``v`` proxy. Keep transient scratch data in cell locals; do not use either
    persistent store as an uncurated dump.
    Each ``handle()`` call gets fresh cell locals; reuse them within the call, but do
    not expect them to survive the turn. ``self.shell`` keeps its cwd across turns, so
    use relative paths and call ``cd`` only when intentionally changing directories.
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
    @strategy(
        CodeActExperimental(config=CodeActConfig(cell_timeout=1800.0)),
        context={
            "state": None,
            "execution_context": None,
            "context_usage": None,
            "self": Context(expr="doc(type(self), concise=True)", prefix=True),
        },
    )
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        """Handle the newest user request and any completed background work."""
        ...


__all__ = ["ExperimentalCodingWorker", "ExperimentalTUIAgent"]
