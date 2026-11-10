# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom's harness counters: an extension of core's HarnessMetrics.

Only Atom records these, so they live here rather than on every agent's
HarnessMetrics. They live in the ``atom`` namespace: span attribute keys are
``harness.atom.<name>`` (Gym/HCB analysis reads them) and
``get_harness_metrics().atom`` is this session's instance. Importing this
module registers them, so ``get_span_schema()`` lists them whenever Atom is in use.

Record with ``atom_metrics().shell_queued()``; outside a metrics session the
call is a no-op, like ``get_harness_metrics()``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from nooa.runtime.harness_metrics import (
    HarnessMetricsExtension,
    SchemaEntry,
    get_harness_metrics,
    register_harness_metrics_extension,
)

__all__ = ["AtomHarnessMetrics", "atom_metrics"]


@register_harness_metrics_extension
class AtomHarnessMetrics(HarnessMetricsExtension):
    """Counters for Atom's Shell, subagents, cell-error hints and Waiting rules."""

    namespace: ClassVar[str] = "atom"

    # A Shell command queued behind another still running in the same Shell
    shell_queued_behind_running: int = 0
    # A Shell command awaited in a cell refused: the Shell runs a launched job's command
    shell_busy_rejected: int = 0
    # A one-shot child awaited in a cell handed to self.jobs at the cell deadline (not closed)
    child_detached_at_deadline: int = 0
    # Waiting on the person (user_messages) rejected; the role whose turn did it
    waiting_on_person_rejections: list[str] = Field(default_factory=list)
    shell_exit_hints: list[int] = Field(default_factory=list)
    shell_cwd_fallbacks: list[str] = Field(default_factory=list)
    name_error_hints: list[str] = Field(default_factory=list)
    mcp_tool_name_hints: list[str] = Field(default_factory=list)
    module_not_found_hints: list[str] = Field(default_factory=list)
    # Reflection messages sent (nooa_atom.session.reflection)
    reflections: int = 0

    span_schema: ClassVar[tuple[SchemaEntry, ...]] = (
        SchemaEntry(
            "shell_queued_behind_running.count",
            "Shell commands queued behind a running one",
            "Tool Usage",
            lambda m: m.shell_queued_behind_running,
        ),
        SchemaEntry(
            "shell_busy_rejected.count",
            "Shell commands refused behind a background job",
            "Tool Usage",
            lambda m: m.shell_busy_rejected,
        ),
        SchemaEntry(
            "child_detached_at_deadline.count",
            "Children awaited in a cell kept running as jobs at the cell deadline",
            "Tool Usage",
            lambda m: m.child_detached_at_deadline,
        ),
        SchemaEntry(
            "waiting_on_person_rejected.count",
            "Waiting on user_messages rejected",
            "Tool Usage",
            lambda m: len(m.waiting_on_person_rejections),
        ),
        SchemaEntry(
            "waiting_on_person_rejected.roles",
            "Roles that waited on the person",
            "Tool Usage",
            lambda m: m.waiting_on_person_rejections,
            True,
        ),
        SchemaEntry(
            "shell_exit_hint.count",
            "Commands that ended their Shell's bash (exit/errexit)",
            "Tool Usage",
            lambda m: len(m.shell_exit_hints),
        ),
        SchemaEntry(
            "shell_exit_hint.codes",
            "Exit statuses of commands that ended their Shell's bash",
            "Tool Usage",
            lambda m: m.shell_exit_hints,
            True,
        ),
        SchemaEntry(
            "shell_cwd_fallback.count",
            "Restarted Shells whose kept cwd was gone",
            "Tool Usage",
            lambda m: len(m.shell_cwd_fallbacks),
        ),
        SchemaEntry(
            "shell_cwd_fallback.details",
            "Gone cwd -> fallback directory",
            "Tool Usage",
            lambda m: m.shell_cwd_fallbacks,
            True,
        ),
        SchemaEntry(
            "name_error_hint.count",
            "NameErrors pointed at python_cell_state",
            "Code Validation",
            lambda m: len(m.name_error_hints),
        ),
        SchemaEntry(
            "name_error_hint.names",
            "Undefined names in hinted NameErrors",
            "Code Validation",
            lambda m: m.name_error_hints,
            True,
        ),
        SchemaEntry(
            "mcp_tool_name_hint.count",
            "MCP tools called bare or as self.<tool>, pointed at their server",
            "Code Validation",
            lambda m: len(m.mcp_tool_name_hints),
        ),
        SchemaEntry(
            "mcp_tool_name_hint.calls",
            "Hinted MCP tools as <server>.<tool>",
            "Code Validation",
            lambda m: m.mcp_tool_name_hints,
            True,
        ),
        SchemaEntry(
            "module_not_found_hint.count",
            "Cell ModuleNotFoundErrors pointed at the task's python3",
            "Code Validation",
            lambda m: len(m.module_not_found_hints),
        ),
        SchemaEntry(
            "module_not_found_hint.modules",
            "Missing modules in hinted ModuleNotFoundErrors",
            "Code Validation",
            lambda m: m.module_not_found_hints,
            True,
        ),
        SchemaEntry(
            "reflection.count",
            "Reflection messages sent",
            "Context Management",
            lambda m: m.reflections,
        ),
    )

    def waiting_on_person_rejected(self, role: str) -> None:
        """Waiting(on=["user_messages"]) was rejected with what to do instead."""
        self._append(self.waiting_on_person_rejections, role)

    def shell_queued(self) -> None:
        """A Shell command was queued behind another still running in the same Shell."""
        self.shell_queued_behind_running += 1

    def shell_busy(self) -> None:
        """A Shell command awaited in a cell was refused behind a launched job's command."""
        self.shell_busy_rejected += 1

    def child_detached(self) -> None:
        """A child awaited in a cell outlived the cell deadline as a job instead of being closed."""
        self.child_detached_at_deadline += 1

    def shell_exit_hint(self, code: int) -> None:
        """A command ended its Shell's bash itself (``exit N``, errexit); the result said so."""
        self._append_item(self.shell_exit_hints, int(code))

    def shell_cwd_fallback(self, detail: str) -> None:
        """A restarted Shell's kept cwd was gone; bash started in its starting directory.

        ``detail`` is ``<gone> -> <fallback>``.
        """
        self._append(self.shell_cwd_fallbacks, detail)

    def name_error_hint(self, name: str) -> None:
        """A cell's NameError got the pointer to the python_cell_state block."""
        self._append(self.name_error_hints, name)

    def mcp_tool_name_hint(self, call: str) -> None:
        """A cell called an MCP tool bare or as ``self.<tool>``; the error named its server.

        ``call`` is ``<server>.<tool>``.
        """
        self._append(self.mcp_tool_name_hints, call)

    def reflection_queued(self) -> None:
        """A reflection message was put on the agent's system_messages queue."""
        self.reflections += 1

    def module_not_found_hint(self, module: str) -> None:
        """A cell's ModuleNotFoundError got the pointer to the task's python3 behind a Shell."""
        self._append(self.module_not_found_hints, module)


def atom_metrics() -> AtomHarnessMetrics:
    """This metrics session's Atom counters, ``get_harness_metrics().atom``.

    A no-op recorder outside a session.
    """
    return get_harness_metrics().extension(AtomHarnessMetrics)
