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
    "python_cells": "Non-synthetic execute_python cells issued by the model.",
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
    "retry_attempts": "Execution attempts explicitly linked to an earlier attempt.",
    "recovered_execution_errors": "Failed attempts followed by a successful linked retry.",
    "restricted_code_errors": "Python outputs containing a stable validator error code.",
    "path_resolution_errors": "Python outputs containing a structured path-resolution code.",
    "recovered_restricted_code_errors": "Restriction failures followed by a successful linked retry.",
    "recovered_path_resolution_errors": "Path failures followed by a successful linked retry.",
    "text_only_replies": "Model replies that did not initially use a tool.",
    "recovered_text_only_replies": "Text-only replies followed by valid tool use.",
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
        self.has_gather = False
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
                self.has_gather = True
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
            count = sum(1 for path in calls if _is_prefix(path, ("self", "shell")) and path[-1] in names)
        elif metric == "repo_queries":
            count = sum(1 for path in calls if _is_prefix(path, ("self", "repo")) and path[-1] in names)
        else:
            count = sum(1 for path in calls if path == ("self", "message"))
        if count:
            out[metric] = count

    if visitor.shell_argv_calls:
        out["shell_argv_commands"] = visitor.shell_argv_calls
    if visitor.has_gather and out.get("delegations", 0) >= 2:
        out["parallel_delegations"] = 1
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
    failed_attempts: dict[str, set[str]] = {}
    recovered_attempts: set[str] = set()
    for event in events:
        event_type = _event_type(event)
        if event_type == "ToolCallEvent":
            metadata = event.get("metadata") or {}
            if event.get("name") == "return_result":
                signals["completion_calls"] += 1
            if event.get("name") != "execute_python" or metadata.get("synthetic"):
                continue
            arguments = event.get("arguments") or {}
            code = arguments.get("code", "")
            if not isinstance(code, str):
                continue
            signals["python_cells"] += 1
            for name, count in _analyze_code(code).items():
                signals[name] += count
        elif event_type == "PythonOutput":
            signals["execution_attempts"] += 1
            status = str(event.get("execution_status", "")).lower()
            is_error = status.endswith("error")
            attempt_id = str(event.get("tool_call_id") or "")
            retry_of = str(event.get("retry_of") or "")
            if retry_of:
                signals["retry_attempts"] += 1

            diagnostic_text = (
                f"{event.get('failure_code', '')}\n{event.get('stdout', '')}\n"
                f"{event.get('stderr', '')}\n{event.get('error', '')}"
            )
            categories: set[str] = set()
            if re.search(r"(?:\[E\d{3}\]|\bE\d{3}\b)", diagnostic_text):
                signals["restricted_code_errors"] += 1
                categories.add("restricted")
            if re.search(r"(?:\[PATH_[A-Z_]+\]|\bPATH_[A-Z_]+\b)", diagnostic_text):
                signals["path_resolution_errors"] += 1
                categories.add("path")

            if is_error:
                signals["execution_errors"] += 1
                if attempt_id:
                    failed_attempts[attempt_id] = categories
            elif retry_of in failed_attempts and retry_of not in recovered_attempts:
                recovered_attempts.add(retry_of)
                signals["recovered_execution_errors"] += 1
                failed_categories = failed_attempts[retry_of]
                if "restricted" in failed_categories:
                    signals["recovered_restricted_code_errors"] += 1
                if "path" in failed_categories:
                    signals["recovered_path_resolution_errors"] += 1
        elif event_type == "TextOnlyReply":
            signals["text_only_replies"] += 1
            if event.get("recovered") is True:
                signals["recovered_text_only_replies"] += 1

    cells = signals["python_cells"]
    text_only = signals["text_only_replies"]
    rates = {
        "self_reference_rate": signals["self_references"] / cells if cells else 0.0,
        "execution_error_rate": (
            signals["execution_errors"] / signals["execution_attempts"]
            if signals["execution_attempts"]
            else 0.0
        ),
        "execution_recovery_rate": (
            signals["recovered_execution_errors"] / signals["execution_errors"]
            if signals["execution_errors"]
            else 0.0
        ),
        "text_only_recovery_rate": (
            signals["recovered_text_only_replies"] / text_only if text_only else 0.0
        ),
        "completion_rate": 1.0 if signals["completion_calls"] else 0.0,
    }
    return BehaviorReport(task_id, model, agent_type, change_id, signals, rates)


def analyze_trajectory(
    path: str | Path,
    *,
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
        task_id=trajectory_path.parent.name or trajectory_path.stem,
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
            name: sum(item.signals.get(name, 0) for item in items)
            for name in SIGNAL_DESCRIPTIONS
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
