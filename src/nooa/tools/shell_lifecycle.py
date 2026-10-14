# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invocation-scoped ownership for shells whose services are graded after completion."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nooa.tools._bash_session import BashSession

logger = logging.getLogger(__name__)
_current_scope: ContextVar[BackgroundServiceScope | None] = ContextVar(
    "shell_service_scope", default=None
)


class BackgroundServiceScope:
    """Own sessions until invocation completion; leave their services to the outer owner.

    The caller must arrange final process/sandbox cleanup after verification.
    This policy overrides per-session cleanup preferences inside this scope.
    """

    def __init__(self) -> None:
        self._sessions: dict[int, BashSession] = {}
        self._closed = False

    def adopt(self, session: BashSession) -> None:
        """Include a shell constructed before entering the invocation scope."""
        if self._closed:
            raise RuntimeError("Background service scope has already closed")
        session._keep_background_on_close = True
        self._sessions[id(session)] = session

    async def aclose(self) -> None:
        self._closed = True
        sessions = tuple(self._sessions.values())
        try:
            results = await asyncio.gather(
                *(session.close() for session in sessions), return_exceptions=True
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise errors[0]
        finally:
            self._sessions.clear()


@asynccontextmanager
async def preserve_background_services() -> AsyncIterator[BackgroundServiceScope]:
    """Preserve background jobs from all shells created in this async invocation.

    Child asyncio tasks inherit the policy; concurrent unrelated invocations do
    not. Strong ownership prevents destructors from running on discarded shells.
    At exit, close every owned shell through its preservation path. The external
    harness remains responsible for killing services after grading.
    """
    scope = BackgroundServiceScope()
    token = _current_scope.set(scope)
    failed = False
    try:
        yield scope
    except BaseException:
        failed = True
        raise
    finally:
        _current_scope.reset(token)
        try:
            await scope.aclose()
        except BaseException:
            if not failed:
                raise
            logger.warning("Shell scope cleanup failed during invocation failure", exc_info=True)
