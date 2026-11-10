# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP harness counters: the ``mcp`` extension of :class:`HarnessMetrics`.

Span keys are ``harness.mcp.<name>``. Importing this module registers the
extension; record with ``get_harness_metrics().extension(McpHarnessMetrics)``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from nooa.runtime.harness_metrics import (
    HarnessMetricsExtension,
    SchemaEntry,
    register_harness_metrics_extension,
)

__all__ = ["McpHarnessMetrics"]


@register_harness_metrics_extension
class McpHarnessMetrics(HarnessMetricsExtension):
    """Counters for MCP tool results."""

    namespace: ClassVar[str] = "mcp"

    result_errors: list[str] = Field(default_factory=list)

    span_schema: ClassVar[tuple[SchemaEntry, ...]] = (
        SchemaEntry(
            "result_error.count",
            "MCP results reporting failure in an 'error' field",
            "Tool Usage",
            lambda m: len(m.result_errors),
        ),
        SchemaEntry(
            "result_error.calls",
            "MCP tools whose result had an 'error' field, as <server>.<tool>",
            "Tool Usage",
            lambda m: m.result_errors,
            True,
        ),
    )

    def result_error(self, call: str) -> None:
        """An MCP tool returned a structured result with an ``error`` field (not isError).

        ``call`` is ``<server>.<tool>``; the tool docs say to check for this.
        """
        self._append(self.result_errors, call)
