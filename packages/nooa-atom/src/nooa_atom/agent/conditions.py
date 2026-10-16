# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy postconditions for the Atom agent's turn methods."""

from typing import Any

from pydantic import ValidationError

from nooa.interactive import Done, NeedInput
from nooa.strategy_validation import InvariantError
from nooa_atom.session.items import TaskResult


def require_result(agent: Any, result: Any, call: Any) -> None:
    """Reject a batch ``Done`` without a ``result``.

    A batch turn is unattended: its ``self.message()`` output reaches at most
    a parent as a ``ChildMessage``, and the parent that delegated it reads
    ``done.result``. The strategy
    turns the ``InvariantError`` into feedback for the model, which then
    returns again with a ``TaskResult``.
    """
    if isinstance(result, Done) and result.result is None:
        raise InvariantError(
            "An unattended turn must end with Done(explanation=..., result=TaskResult(...)): "
            "set solution_description, evidence and how_to_verify, and put the full "
            "report in report. self.message() is no substitute here; the result is "
            "what the requester reads."
        )


ONE_SHOT_QUESTION = (
    "Your parent can't answer a one-shot worker: decide, state your assumptions in "
    "the TaskResult report, and return Done."
)


def require_task_result(agent: Any, result: Any, call: Any) -> None:
    """Reject a worker ``Done`` whose ``result`` is not a ``TaskResult``, and a one-shot question.

    A worker's parent reads ``done.result``; ``self.message()`` text reaches
    it only as progress notes. A dict that validates as a ``TaskResult`` is
    accepted (the parent receives it as one). A one-shot worker
    (``delegate()``) may not ask: its session port says so (``may_ask``).
    """
    if isinstance(result, NeedInput):
        port = getattr(agent, "session", None)
        if port is not None and not getattr(port, "may_ask", True):
            raise InvariantError(ONE_SHOT_QUESTION)
        return
    if not isinstance(result, Done) or isinstance(result.result, TaskResult):
        return
    if isinstance(result.result, dict):
        try:
            TaskResult.model_validate(result.result)
            return
        except ValidationError:
            pass
    raise InvariantError(
        "A worker turn must end with Done(explanation=..., result=TaskResult(...)): "
        "set solution_description, evidence and how_to_verify, and put the full report "
        "in report. Your parent reads only the result; self.message() is no substitute."
    )
