# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP instrumentation around the shared persistent interactive dispatcher."""

from collections.abc import Coroutine
from contextvars import Context, copy_context
from typing import Any

from nooa_cli.interactive.dispatcher import InteractiveSessionDispatcher as SharedDispatcher

from nooa_acp.execution_tree import ACPExecutionTree


class InteractiveSessionDispatcher(SharedDispatcher):
    """Give each ACP request a fresh tree without creating another dispatch loop."""

    def __init__(self, agent: Any, *, execution_tree: ACPExecutionTree | None = None) -> None:
        self._execution_tree = execution_tree
        self._turn_context: Context | None = None
        self._base_context = copy_context()
        super().__init__(agent, handle_context=self._handle_context if execution_tree else None)

    def _handle_context(self) -> Context:
        if self._turn_context is not None:
            return self._turn_context.copy()

        def capture_background() -> Context:
            assert self._execution_tree is not None
            with self._execution_tree.turn():
                return copy_context()

        return self._base_context.copy().run(capture_background)

    async def _run_active(self, operation: Coroutine[Any, Any, Any]) -> Any:
        if self._execution_tree is None:
            return await super()._run_active(operation)
        if self.active or self._closed:
            operation.close()
            self._ensure_idle()
        with self._execution_tree.turn():
            self._turn_context = copy_context()
            try:
                return await super()._run_active(operation)
            finally:
                self._turn_context = None
