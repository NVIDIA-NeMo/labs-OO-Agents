# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic interface-behavior metrics extracted from agent trajectories.

This module deliberately scores *observable actions*, not answer quality or hidden
reasoning.  It consumes the ``trajectory.json`` artifact written by the Harbor
runner, so the same validators can compare models, agent variants, and prompt
changes without another model call.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SIGNAL_DESCRIPTIONS: dict[str, str] = {
    "python_cells": "Model-issued execute_python or python_cell calls, excluding framework prefill.",
    "self_references": "Cells that access the runtime agent through self.",
    "persistent_state_uses": "Cells that access self.v.",
    "todo_state_uses": "Cells that access todo-local vars (todo.v or set_var).",
    "todo_creations": "Calls that create a structured todo.",
    "todo_activations": "Calls that activate a structured todo.",
    "todo_comments": "Calls that record a material todo comment.",
    "delegations": "Calls to self.delegate or self.spawn.",
    "parallel_delegations": "Cells using gather with delegation calls.",
    "shell_commands": "Calls to self.shell.run or self.shell.run_stream.",
    "shell_argv_commands": "Shell calls whose command is a literal argv list or tuple.",
    "repo_queries": "Calls to self.repo navigation methods.",
    "user_messages": "Calls to self.message.",
    "completion_calls": "Observed return_result tool calls.",
    "execution_attempts": "Observed PythonOutput execution attempts.",
    "execution_errors": "PythonOutput events with error execution status.",
    "restricted_code_errors": "Python outputs containing a stable validator error code.",
    "path_resolution_errors": "Python outputs containing a structured path-resolution code.",
    "text_only_replies": "Model replies that did not initially use a tool.",
}


RATE_DESCRIPTIONS: dict[str, str] = {
    "self_reference_rate": "Python cells containing at least one self reference",
    "execution_error_rate": "Execution attempts that ended in error",
    "completion_rate": "Whether the trajectory contains a completion call",
}


@dataclass(frozen=True)
class BehaviorReport:
    """Allowlisted aggregate metrics for one trajectory; never contains event payloads."""

    task_id: str
    model: str = "unknown"
    agent_type: str = "unknown"
    change_id: str = "baseline"
    signals: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    schema_version: int = field(default=1, init=False)
    content_policy: str = field(default="aggregate-counts-only", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _CodeSignals(ast.NodeVisitor):
    """Collect interface actions from one executable Python cell."""

    def __init__(self) -> None:
        self.paths: list[tuple[str, ...]] = []
        self.calls: list[tuple[str, ...]] = []
        self.parallel_delegations = 0
        self.shell_argv_calls = 0

    @staticmethod
    def _path(node: ast.AST) -> tuple[str, ...]:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return tuple(reversed(parts))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = self._path(node)
        if path:
            self.paths.append(path)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = self._path(node.func)
        if path:
            self.calls.append(path)
            if path[-1] == "gather":
                delegated_args = sum(
                    1
                    for child in node.args
                    if isinstance(child, ast.Call)
                    and self._path(child.func)[:1] == ("self",)
                    and self._path(child.func)[-1:] in {("delegate",), ("spawn",)}
                )
                if delegated_args >= 2:
                    self.parallel_delegations += 1
            if (
                _is_prefix(path, ("self", "shell"))
                and path[-1] in {"run", "run_stream"}
                and node.args
                and isinstance(node.args[0], (ast.List, ast.Tuple))
            ):
                self.shell_argv_calls += 1
        self.generic_visit(node)


def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("event_type") or event.get("type") or "")


def _is_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def _analyze_code(code: str) -> dict[str, int]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return {}
    visitor = _CodeSignals()
    visitor.visit(tree)
    paths = visitor.paths + visitor.calls
    calls = visitor.calls
    out: dict[str, int] = {}

    if any(path and path[0] == "self" for path in paths):
        out["self_references"] = 1
    if any(_is_prefix(path, ("self", "v")) for path in paths):
        out["persistent_state_uses"] = 1
    if any("todo" in path and (path[-1] == "v" or path[-1] == "set_var") for path in paths):
        out["todo_state_uses"] = 1

    call_metrics = {
        "todo_creations": {"add", "create"},
        "todo_activations": {"activate"},
        "todo_comments": {"comment"},
        "delegations": {"delegate", "spawn"},
        "shell_commands": {"run", "run_stream"},
        "repo_queries": {"symbols", "refs", "find", "search"},
        "user_messages": {"message"},
    }
    for metric, names in call_metrics.items():
        if metric.startswith("todo_"):
            count = sum(1 for path in calls if "todo" in path and path[-1] in names)
        elif metric == "delegations":
            count = sum(1 for path in calls if path[:1] == ("self",) and path[-1] in names)
        elif metric == "shell_commands":
            count = sum(
                1 for path in calls if _is_prefix(path, ("self", "shell")) and path[-1] in names
            )
        elif metric == "repo_queries":
            count = sum(
                1 for path in calls if _is_prefix(path, ("self", "repo")) and path[-1] in names
            )
        else:
            count = sum(1 for path in calls if path == ("self", "message"))
        if count:
            out[metric] = count

    if visitor.shell_argv_calls:
        out["shell_argv_commands"] = visitor.shell_argv_calls
    if visitor.parallel_delegations:
        out["parallel_delegations"] = visitor.parallel_delegations
    return out


def analyze_events(
    events: Iterable[dict[str, Any]],
    *,
    task_id: str = "unknown",
    model: str = "unknown",
    agent_type: str = "unknown",
    change_id: str = "baseline",
) -> BehaviorReport:
    """Analyze already-serialized framework events."""
    signals = dict.fromkeys(SIGNAL_DESCRIPTIONS, 0)
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = _event_type(event)
        metadata = event.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        framework_execution = event.get("synthetic", metadata.get("synthetic")) or event.get(
            "prefill", metadata.get("prefill")
        )
        if event_type == "ToolCallEvent":
            if not isinstance(event.get("name"), str):
                continue
            if event.get("name") == "return_result":
                signals["completion_calls"] += 1
            if event.get("name") not in {"execute_python", "python_cell"} or framework_execution:
                continue
            arguments = event.get("arguments") or {}
            if not isinstance(arguments, dict):
                continue
            code = arguments.get("code", "")
            if not isinstance(code, str):
                continue
            signals["python_cells"] += 1
            for name, count in _analyze_code(code).items():
                signals[name] += count
        elif event_type == "PythonOutput":
            if framework_execution:
                continue
            signals["execution_attempts"] += 1
            status = str(event.get("execution_status", "")).lower()
            is_error = status.endswith("error")
            diagnostic_text = (
                f"{event.get('failure_code', '')}\n{event.get('stdout', '')}\n"
                f"{event.get('stderr', '')}\n{event.get('error', '')}"
            )
            if re.search(r"\[E\d{3}\]", diagnostic_text):
                signals["restricted_code_errors"] += 1
            if re.search(r"\[PATH_[A-Z_]+\]", diagnostic_text):
                signals["path_resolution_errors"] += 1

            if is_error:
                signals["execution_errors"] += 1
        elif event_type == "TextOnlyReply":
            signals["text_only_replies"] += 1

    cells = signals["python_cells"]
    rates = {
        "self_reference_rate": signals["self_references"] / cells if cells else 0.0,
        "execution_error_rate": (
            signals["execution_errors"] / signals["execution_attempts"]
            if signals["execution_attempts"]
            else 0.0
        ),
        "completion_rate": 1.0 if signals["completion_calls"] else 0.0,
    }
    return BehaviorReport(task_id, model, agent_type, change_id, signals, rates)


def analyze_trajectory(
    path: str | Path,
    *,
    task_id: str | None = None,
    model: str = "unknown",
    agent_type: str = "unknown",
    change_id: str = "baseline",
) -> BehaviorReport:
    """Analyze a runner ``trajectory.json`` file."""
    trajectory_path = Path(path)
    raw = json.loads(trajectory_path.read_text())
    if not isinstance(raw, list):
        raise ValueError("trajectory must be a JSON list of serialized events")
    return analyze_events(
        raw,
        task_id=task_id or trajectory_path.parent.name or trajectory_path.stem,
        model=model,
        agent_type=agent_type,
        change_id=change_id,
    )


def aggregate_reports(reports: Iterable[BehaviorReport]) -> list[dict[str, Any]]:
    """Aggregate counts and per-task prevalence by model/agent/change."""
    grouped: dict[tuple[str, str, str], list[BehaviorReport]] = defaultdict(list)
    for report in reports:
        grouped[(report.model, report.agent_type, report.change_id)].append(report)

    rows: list[dict[str, Any]] = []
    for (model, agent_type, change_id), items in sorted(grouped.items()):
        signal_totals = {
            name: sum(item.signals.get(name, 0) for item in items) for name in SIGNAL_DESCRIPTIONS
        }
        prevalence = {
            name: sum(item.signals.get(name, 0) > 0 for item in items) / len(items)
            for name in SIGNAL_DESCRIPTIONS
        }
        rate_means = {
            name: sum(item.rates.get(name, 0.0) for item in items) / len(items)
            for name in sorted({key for item in items for key in item.rates})
        }
        rows.append(
            {
                "model": model,
                "agent_type": agent_type,
                "change_id": change_id,
                "tasks": len(items),
                "signal_totals": signal_totals,
                "task_prevalence": prevalence,
                "mean_rates": rate_means,
            }
        )
    return rows


def load_behavior_report(path: str | Path) -> BehaviorReport:
    """Load one ``behavior.json`` artifact."""
    data = json.loads(Path(path).read_text())
    return BehaviorReport(
        task_id=str(data["task_id"]),
        model=str(data.get("model", "unknown")),
        agent_type=str(data.get("agent_type", "unknown")),
        change_id=str(data.get("change_id", "baseline")),
        signals={str(key): int(value) for key, value in data.get("signals", {}).items()},
        rates={str(key): float(value) for key, value in data.get("rates", {}).items()},
    )


def aggregate_behavior_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load behavior artifacts and aggregate them by model, agent, and change."""
    return aggregate_reports(load_behavior_report(path) for path in paths)
