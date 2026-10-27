# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local observation of actual formatting and capture omissions."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_truncation_reasons: ContextVar[set[str] | None] = ContextVar("truncation_reasons", default=None)


def record_truncation(reason: str) -> None:
    """Record an omission only when a caller is observing this task."""
    reasons = _truncation_reasons.get()
    if reasons is not None:
        reasons.add(reason)


@contextmanager
def observe_truncation() -> Iterator[set[str]]:
    """Collect omissions without changing formatting or traversing values twice."""
    reasons: set[str] = set()
    token = _truncation_reasons.set(reasons)
    try:
        yield reasons
    finally:
        _truncation_reasons.reset(token)
