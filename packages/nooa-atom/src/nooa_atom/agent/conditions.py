# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy postconditions for the coding agent's turn methods."""

from typing import Any

from nooa.interactive import Done
from nooa.strategy_validation import InvariantError


def require_result(agent: Any, result: Any, call: Any) -> None:
    """Reject a batch ``Done`` without a ``result``.

    A batch turn is unattended: its ``self.message()`` output reaches nobody,
    and the parent that delegated it reads ``done.result``. The strategy
    turns the ``InvariantError`` into feedback for the model, which then
    returns again with a ``TaskResult``.
    """
    if isinstance(result, Done) and result.result is None:
        raise InvariantError(
            "An unattended turn must end with Done(explanation=..., result=TaskResult(...)): "
            "set solution_description, evidence and how_to_verify, and put the full "
            "report in report. Nobody sees self.message() here; the result is the "
            "only thing the requester reads."
        )
