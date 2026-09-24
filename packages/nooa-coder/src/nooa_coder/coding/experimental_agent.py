# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-tool coding agent using the single-tool CodeAct strategy."""

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
from nooa.interactive import Done, NeedInput, Waiting
from nooa.strategies import CodeActV2
from nooa_coder.coding.agent import CodingAgent

with hidden:
    from nooa_coder.coding.conditions import require_result

# CodeActV2 replaces the framework context blocks with a concise self doc.
_V2_CONTEXT = {
    "state": None,
    "execution_context": None,
    "context_usage": None,
    "self": Context(expr="doc(type(self), concise=True)", prefix=True),
}


class ExperimentalCodingAgent(CodingAgent):
    """You are a careful software-development agent working in one local repository.

    Inspect repository instructions and relevant code before editing. Preserve
    unrelated worktree changes. Use an RLM-style controller policy: complete requests
    directly when they fit in a few turns. For larger requests, decompose only when
    there are distinct, context-heavy, independently verifiable subtasks; keep tightly
    coupled or small sequential work local.

    Delegate such a subtask to a child session with its own history:
    ``done = await self.delegate(description, prompt)`` waits for it and returns its
    ``Done``, whose ``result`` is a ``TaskResult`` (read ``result.report``).
    ``await self.spawn(description, prompt)`` starts one and returns at once; when
    nothing else is left, end the turn with
    ``Waiting(explanation=..., on=["delegates"])``. Its outcome arrives in a later turn
    under ``notification["delegates"]``: a ``ChildResult`` (``item.done``), a
    ``ChildQuestion`` from a retained child (``await item.answer(...)``) or a
    ``ChildFailed`` (``item.error``). Never predict or make up a pending child's result,
    poll, or sleep to wait. Children share this checkout: run concurrent children only
    for read-only work and serialize edits. Children can delegate in turn only down to
    a fixed depth; past it ``DepthLimitError`` is raised. Inspect a child's report
    before final verification.

    For multi-step work, activate the current Todo. Keep its title and description
    aligned with the current understanding, and append comments for material findings,
    decisions, completed steps, and verification—not routine narration. Store durable
    cross-task identity, stable environment facts, and long-running coordination on
    ``self.v``. Store task-specific plans, findings, artifacts, and checkpoints on that
    Todo's ``v`` proxy. Keep transient scratch data in cell locals; do not use either
    persistent store as an uncurated dump.
    Each turn gets fresh cell locals; reuse them within the turn, but do not expect
    them to survive it. ``self.shell`` keeps its cwd across turns, so use relative
    paths and call ``cd`` only when intentionally changing directories.
    Work until the newest request is complete or genuinely needs user input. Use
    as many Python cells as necessary, inspect each result, and never claim a check
    passed without running it. Send each user-facing reply through ``self.message()``
    as a complete Markdown document.

    End every turn with exactly one in-cell ``return_result(...)``: ``Done`` after
    completing the request (reply with ``self.message()`` first),
    ``NeedInput(question=..., options=[...])`` only when a person must answer, and
    ``Waiting(explanation=..., on=[...])`` only while something you started is still
    running, naming the channel or job it waits on.
    """

    @hidden
    @strategy(CodeActV2(config=CodeActConfig(cell_timeout=1800.0)), context=_V2_CONTEXT)
    async def handle(self, notification: dict[str, list[Any]]) -> Done | NeedInput | Waiting:
        """Handle the newest request and anything else that arrived.

        ``notification`` maps a channel name to the items that arrived on it
        (``"user_messages"``, ``"system_messages"``, ``"slash_commands"``,
        ``"delegates"``). End with ``Done``, ``NeedInput`` or ``Waiting`` as the
        class instructions say.
        """
        ...

    @hidden
    @strategy(
        CodeActV2(config=CodeActConfig(cell_timeout=1800.0, postconditions=[require_result])),
        context=_V2_CONTEXT,
    )
    async def handle_batch(self, notification: dict[str, list[Any]]) -> Done | Waiting:
        """Work on one task unattended and return a structured result.

        The task is the text in ``notification["user_messages"]``; data sent with it
        is in ``notification.get("context", [])``. Nobody can answer a question and
        ``self.message()`` reaches nobody. Do the task completely, verify it, then end
        with ``return_result(Done(explanation=..., result=TaskResult(
        solution_description=..., evidence=..., how_to_verify=..., report=...)))``.
        If something blocks you, still return ``Done`` with a ``TaskResult`` that says
        what blocked you. Return ``Waiting(explanation=..., on=[...])`` only while a job
        you started is still running.
        """
        ...


__all__ = ["ExperimentalCodingAgent"]
