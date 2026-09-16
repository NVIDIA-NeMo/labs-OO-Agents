# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn-status policy shared by native and protocol hosts."""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class LocalTurnPolicy:
    """Apply agent behavior independently of terminal or protocol presentation."""

    def __init__(
        self,
        *,
        emit_output: Callable[[Any], Awaitable[None]],
    ) -> None:
        self._emit_output = emit_output
        self._closed = False
        self._state_lock = threading.Lock()

    async def after_handle(self, agent: Any, result: Any) -> None:
        """Report the completed turn while the host is active."""
        if not self._is_active():
            return
        explanation = getattr(result, "explanation", "")
        logger.debug("[DISPATCHER] handle() returned kind=%r", result.kind)
        if explanation and self._is_active():
            from .policy_events import TurnStatus as StopReasonOutput

            await self._emit_output(StopReasonOutput(result.kind, explanation))

    async def shutdown(self) -> None:
        """Stop emitting turn statuses before host teardown."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

    def _is_active(self) -> bool:
        with self._state_lock:
            return not self._closed
