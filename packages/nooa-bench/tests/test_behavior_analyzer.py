# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic agent-interface behavior evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa_bench import runner
from nooa_bench.behavior_analyzer import (
    BehaviorReport,
    aggregate_behavior_paths,
    aggregate_reports,
    analyze_events,
    analyze_trajectory,
)
from nooa_bench.change_ledger import load_change_ledger


def _cell(code: str, *, synthetic: bool = False) -> dict:
    return {
        "event_type": "ToolCallEvent",
        "name": "execute_python",
        "arguments": {"code": code},
        "metadata": {"synthetic": synthetic},
    }


def test_ast_signals_cover_core_agent_interface_behaviors() -> None:
    events = [
        _cell(
            """todo = self.todo.add('investigate')
self.todo.activate(todo)
todo.v.notes = {'cause': 'parser'}
self.todo.comment(todo, 'found cause')
self.v.plan = ['inspect', 'fix']
r = await self.shell.run(['pytest', '-q'])
refs = await self.repo.refs('parse')
"""
        ),
        _cell(
            """a, b = await asyncio.gather(
    self.delegate('inspect parser'),
    self.delegate('review tests'),
)
self.message('working')
"""
        ),
        {
            "event_type": "PythonOutput",
            "tool_call_id": "attempt-1",
            "execution_status": "error",
            "failure_code": "E301",
            "stderr": "RestrictedCodeError: [E301] await it",
        },
        {
            "event_type": "PythonOutput",
            "tool_call_id": "attempt-2",
            "execution_status": "complete",
            "retry_of": "attempt-1",
            "stdout": "[PATH_NOT_FOUND] symbols",
        },
        {"event_type": "PythonOutput", "tool_call_id": "attempt-3", "execution_status": "complete"},
        {"event_type": "TextOnlyReply", "recovered": True},
        {"event_type": "ToolCallEvent", "name": "return_result", "arguments": {}},
    ]

    report = analyze_events(
        events, task_id="task-1", model="model-a", agent_type="rlm", change_id="new-prompt"
    )

    assert report.signals == {
        "python_cells": 2,
        "self_references": 2,
        "persistent_state_uses": 1,
        "todo_state_uses": 1,
        "todo_creations": 1,
        "todo_activations": 1,
        "todo_comments": 1,
        "delegations": 2,
        "parallel_delegations": 1,
        "shell_commands": 1,
        "shell_argv_commands": 1,
        "repo_queries": 1,
        "user_messages": 1,
        "completion_calls": 1,
        "execution_attempts": 3,
        "execution_errors": 1,
        "retry_attempts": 1,
        "recovered_execution_errors": 1,
        "restricted_code_errors": 1,
        "path_resolution_errors": 1,
        "recovered_restricted_code_errors": 1,
        "recovered_path_resolution_errors": 0,
        "text_only_replies": 1,
        "recovered_text_only_replies": 1,
    }
    assert report.rates == {
        "self_reference_rate": 1.0,
        "execution_error_rate": 1 / 3,
        "execution_recovery_rate": 1.0,
        "text_only_recovery_rate": 1.0,
        "completion_rate": 1.0,
    }


def test_comments_strings_and_synthetic_cells_do_not_create_false_signals() -> None:
    report = analyze_events(
        [
            _cell("# self.delegate('fake')\ntext = 'self.v and self.todo.add'"),
            _cell("self.delegate('synthetic')", synthetic=True),
            _cell("this is invalid python"),
        ]
    )

    assert report.signals["python_cells"] == 2
    assert report.signals["self_references"] == 0
    assert report.signals["persistent_state_uses"] == 0
    assert report.signals["delegations"] == 0
    assert report.signals["shell_argv_commands"] == 0


def test_trajectory_analysis_and_grouped_aggregation(tmp_path: Path) -> None:
    path = tmp_path / "task-7" / "trajectory.json"
    path.parent.mkdir()
    path.write_text(json.dumps([_cell("self.v.answer = 42"), {"event_type": "ToolCallEvent", "name": "return_result", "arguments": {}}]))

    first = analyze_trajectory(path, model="m", agent_type="bench", change_id="before")
    second = BehaviorReport(
        task_id="task-8",
        model="m",
        agent_type="bench",
        change_id="before",
        signals={**first.signals, "persistent_state_uses": 0, "completion_calls": 0},
        rates={**first.rates, "completion_rate": 0.0},
    )
    rows = aggregate_reports([first, second])

    assert first.task_id == "task-7"
    assert len(rows) == 1
    assert rows[0]["tasks"] == 2
    assert rows[0]["signal_totals"]["persistent_state_uses"] == 1
    assert rows[0]["task_prevalence"]["persistent_state_uses"] == 0.5
    assert rows[0]["mean_rates"]["completion_rate"] == 0.5

    artifact = tmp_path / "behavior.json"
    artifact.write_text(json.dumps(first.to_dict()))
    assert aggregate_behavior_paths([artifact]) == aggregate_reports([first])


def test_change_ledger_is_complete_and_references_known_signals() -> None:
    root = Path(__file__).parents[3]
    ledger = load_change_ledger(root / "evaluations" / "agent-interface-changes.json")
    ids = {change["id"] for change in ledger["changes"]}

    assert "bounded-generic-execution-context" in ids
    assert "safe-default-state-selection" in ids
    assert "deterministic-interface-behavior-ledger" in ids
    assert all(change["deterministic_checks"] for change in ledger["changes"])
    assert all(change["trace_expectations"] for change in ledger["changes"])


def test_runner_writes_behavior_artifact_from_serialized_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: runner artifact -> parser -> deterministic behavior.json."""

    class ToolCallEvent:
        def model_dump(self, mode: str) -> dict:
            assert mode == "json"
            return {
                "name": "execute_python",
                "arguments": {"code": "self.v.note = 'kept'"},
                "metadata": {},
            }

    class ReturnResultEvent:
        def model_dump(self, mode: str) -> dict:
            return {"event_type": "ToolCallEvent", "name": "return_result", "arguments": {}}

    # _write_trajectory uses the concrete class name as event_type. Give the
    # completion event the canonical name without coupling this test to Pydantic.
    ReturnResultEvent.__name__ = "ToolCallEvent"
    agent = SimpleNamespace(event_manager=SimpleNamespace(items=lambda: [("1", ToolCallEvent()), ("2", ReturnResultEvent())]))
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("NOOA_INTERFACE_CHANGE_ID", "prompt-v2")

    runner._write_trajectory(agent)
    runner._write_behavior_report("model-z", "rlm")

    payload = json.loads((tmp_path / "behavior.json").read_text())
    assert payload["model"] == "model-z"
    assert payload["agent_type"] == "rlm"
    assert payload["change_id"] == "prompt-v2"
    assert payload["signals"]["persistent_state_uses"] == 1
    assert payload["signals"]["completion_calls"] == 1


def test_behavior_reporting_is_non_fatal_without_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    runner._write_behavior_report("m", "bench")
    assert not (tmp_path / "behavior.json").exists()


def test_success_after_error_is_not_recovery_without_explicit_link() -> None:
    report = analyze_events(
        [
            {"event_type": "PythonOutput", "tool_call_id": "failed", "execution_status": "error", "failure_code": "E301"},
            {"event_type": "PythonOutput", "tool_call_id": "later", "execution_status": "complete"},
        ]
    )

    assert report.signals["execution_errors"] == 1
    assert report.signals["recovered_execution_errors"] == 0
    assert report.signals["recovered_restricted_code_errors"] == 0


def test_behavior_report_is_content_free_with_sensitive_inputs() -> None:
    secret = "PRIVATE-SENTINEL-DO-NOT-PERSIST"
    report = analyze_events(
        [
            _cell(f"value = {secret!r}; await self.shell.run({secret!r})"),
            {
                "event_type": "PythonOutput",
                "tool_call_id": "failed",
                "execution_status": "error",
                "failure_code": "PATH_NOT_FOUND",
                "stdout": secret,
                "stderr": secret,
                "error": secret,
                "value": {"queue_payload": secret},
            },
            {
                "event_type": "PythonOutput",
                "tool_call_id": "retry",
                "retry_of": "failed",
                "execution_status": "complete",
            },
            {"event_type": "TextOnlyReply", "content": secret, "recovered": True},
        ],
        task_id="task-id",
        model="model-id",
        agent_type="agent-id",
        change_id="change-id",
    )

    payload = report.to_dict()
    serialized = json.dumps(payload)
    assert secret not in serialized
    assert set(payload) == {
        "schema_version",
        "content_policy",
        "task_id",
        "model",
        "agent_type",
        "change_id",
        "signals",
        "rates",
    }
    assert payload["schema_version"] == 1
    assert payload["content_policy"] == "aggregate-counts-only"
    assert all(isinstance(value, int) for value in payload["signals"].values())
    assert all(isinstance(value, float) for value in payload["rates"].values())
    assert payload["signals"]["recovered_path_resolution_errors"] == 1


def test_parallel_delegations_must_be_arguments_of_the_same_gather() -> None:
    report = analyze_events([_cell("""
self.delegate('one')
self.delegate('two')
await asyncio.gather(fetch_a(), fetch_b())
""")])
    assert report.signals["delegations"] == 2
    assert report.signals["parallel_delegations"] == 0


def test_change_ledger_accepts_every_published_rate(tmp_path: Path) -> None:
    from nooa_bench.behavior_analyzer import RATE_DESCRIPTIONS

    ledger = {
        "schema_version": 1,
        "changes": [{
            "id": "all-rates", "status": "implemented", "component": "test",
            "hypothesis": "catalogs agree", "deterministic_checks": ["unit"],
            "trace_expectations": [
                {"signal": name, "direction": "unchanged"} for name in RATE_DESCRIPTIONS
            ],
            "benchmark_slices": ["all"],
        }],
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    assert load_change_ledger(path) == ledger
