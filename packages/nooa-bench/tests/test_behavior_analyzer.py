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
    load_behavior_report,
)

from nooa.runtime.event_manager import EventManager


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
        "text_only_replies": 1,
    }
    assert report.rates == {
        "self_reference_rate": 1.0,
        "execution_error_rate": 1 / 3,
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
    path.write_text(
        json.dumps(
            [
                _cell("self.v.answer = 42"),
                {"event_type": "ToolCallEvent", "name": "return_result", "arguments": {}},
            ]
        )
    )

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


def test_runner_writes_behavior_artifact_from_serialized_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: runner artifact -> parser -> deterministic behavior.json."""

    from nooa.context_blocks import ToolCallEvent

    calls = [
        ToolCallEvent(
            tool_call_id="1", name="execute_python", arguments={"code": "self.v.note = 'kept'"}
        ),
        ToolCallEvent(tool_call_id="2", name="return_result", arguments={}),
    ]
    manager = EventManager()
    for event in calls:
        manager.add(event)
    agent = SimpleNamespace(event_manager=manager)
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("NOOA_INTERFACE_CHANGE_ID", "prompt-v2")
    monkeypatch.setenv("NOOA_TASK_ID", "actual-task-id")

    runner._write_trajectory(agent)
    runner._write_behavior_report("model-z", "rlm")

    payload = json.loads((tmp_path / "behavior.json").read_text())
    assert payload["model"] == "model-z"
    assert payload["agent_type"] == "rlm"
    assert payload["change_id"] == "prompt-v2"
    assert payload["task_id"] == "actual-task-id"
    assert payload["signals"]["persistent_state_uses"] == 1
    assert payload["signals"]["completion_calls"] == 1


def test_behavior_reporting_is_non_fatal_without_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    runner._write_behavior_report("m", "bench")
    assert not (tmp_path / "behavior.json").exists()


def test_recovery_metrics_are_not_reported_without_framework_linkage() -> None:
    report = analyze_events(
        [
            {
                "event_type": "PythonOutput",
                "tool_call_id": "failed",
                "execution_status": "error",
                "failure_code": "E301",
            },
            {"event_type": "PythonOutput", "tool_call_id": "later", "execution_status": "complete"},
        ]
    )

    assert report.signals["execution_errors"] == 1
    assert not any("recover" in key or "retry" in key for key in {*report.signals, *report.rates})


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
    assert payload["schema_version"] == 2
    assert payload["content_policy"] == "aggregate-counts-only"
    assert all(isinstance(value, int) for value in payload["signals"].values())
    assert all(isinstance(value, float) for value in payload["rates"].values())


def test_parallel_delegations_must_be_arguments_of_the_same_gather() -> None:
    report = analyze_events(
        [
            _cell("""
self.delegate('one')
self.delegate('two')
await asyncio.gather(fetch_a(), fetch_b())
""")
        ]
    )
    assert report.signals["delegations"] == 2
    assert report.signals["parallel_delegations"] == 0


@pytest.mark.parametrize("tool_name", ["execute_python", "python_cell"])
def test_model_cells_are_counted_without_framework_prefill(tool_name):
    cell = {**_cell("self.todo.add('task')"), "name": tool_name}
    prefill = {**cell, "metadata": {"prefill": True}}
    synthetic = {**cell, "metadata": {"synthetic": True}}
    report = analyze_events([prefill, cell, synthetic])
    assert report.signals["python_cells"] == 1
    assert report.signals["todo_creations"] == 1


def test_real_export_preserves_only_metric_classification_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nooa.context_blocks import ToolCallEvent
    from nooa.events import PythonOutput, ResultStatus

    events = [
        ToolCallEvent(
            tool_call_id="prefill",
            name="python_cell",
            arguments={"code": "print('inputs')"},
            metadata={"prefill": True, "private_state": "private-sentinel"},
        ),
        ToolCallEvent(
            tool_call_id="synthetic",
            name="python_cell",
            arguments={"code": "print('setup')"},
            metadata={"synthetic": True},
        ),
        ToolCallEvent(
            tool_call_id="model", name="python_cell", arguments={"code": "self.todo.status()"}
        ),
    ]
    for index, metadata in enumerate(({"prefill": True}, {"synthetic": True}, {}, {})):
        events.append(
            PythonOutput(
                tool_call_id=str(index),
                execution_count=index + 1,
                execution_status=ResultStatus.COMPLETE if index == 3 else ResultStatus.ERROR,
                error="[E301] [PATH_NOT_FOUND]" if metadata else "",
                metadata=metadata,
            )
        )
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    manager = EventManager()
    for event in events:
        manager.add(event)
    runner._write_trajectory(SimpleNamespace(event_manager=manager))
    raw = (tmp_path / "trajectory.json").read_text()
    assert "private-sentinel" not in raw
    report = analyze_trajectory(tmp_path / "trajectory.json")
    assert report.signals["python_cells"] == 1
    assert report.rates["self_reference_rate"] == 1.0
    assert report.signals["execution_attempts"] == 2
    assert report.signals["execution_errors"] == 1
    assert report.rates["execution_error_rate"] == 0.5
    assert "restricted_code_errors" not in report.signals
    assert "path_resolution_errors" not in report.signals


@pytest.mark.parametrize("flag", ["prefill", "synthetic"])
def test_output_classification_uses_metadata_fallback_and_explicit_flag_precedence(flag):
    output = {"event_type": "PythonOutput", "execution_status": "error"}
    report = analyze_events(
        [
            {**output, "metadata": {flag: True}},
            {**output, flag: True, "metadata": {flag: False}},
            {**output, flag: False, "metadata": {flag: True}},
            {"event_type": "ToolCallEvent", "name": "return_result", "synthetic": True},
        ]
    )
    assert report.signals["execution_attempts"] == 1
    assert report.signals["execution_errors"] == 1
    assert report.signals["completion_calls"] == 1


@pytest.mark.parametrize("output", ["E501 line too long", "route E101 to bus", "PATH_TO_FILE=/x"])
def test_ordinary_output_does_not_count_as_a_framework_diagnostic(output):
    report = analyze_events([{"event_type": "PythonOutput", "stdout": output}])
    assert "restricted_code_errors" not in report.signals
    assert "path_resolution_errors" not in report.signals


def test_malformed_events_do_not_discard_valid_cells():
    report = analyze_events(
        [
            None,
            "broken",
            [],
            3,
            {"event_type": "ToolCallEvent", "name": []},
            {"event_type": "ToolCallEvent", "name": "python_cell", "arguments": [1]},
            _cell("self.todo.status()"),
        ]
    )
    assert report.signals["python_cells"] == 1


@pytest.mark.parametrize(
    "code,expected",
    [
        ("await asyncio.gather(self.delegate('a'), self.delegate('b'))", 1),
        ("await asyncio.gather(*[self.delegate(x) for x in tasks])", 1),
        ("await asyncio.gather(*(self.delegate(x) for x in tasks))", 1),
        ("jobs = [self.delegate(x) for x in tasks]\nawait asyncio.gather(*jobs)", 1),
        ("await asyncio.gather(self.delegate('a'))", 0),
        ("await other.gather(self.delegate('a'), self.delegate('b'))", 0),
        ("await asyncio.gather(wrap(self.delegate('a')), wrap(self.delegate('b')))", 0),
    ],
)
def test_fanout_source_patterns(code, expected):
    assert analyze_events([_cell(code)]).signals["parallel_delegations"] == expected


def test_same_cell_aliases_and_real_api_names():
    report = analyze_events(
        [
            _cell("""
t = self.todo.add('task')
t.v.note = 'x'
s = self.shell
s.run(['pytest'])
r = self.repo
r.symbols('f')
self.todo.create('not an API')
self.spawn('not an API')
self.repo.find_refs('not an API')
""")
        ]
    )
    assert report.signals["todo_state_uses"] == 1
    assert report.signals["todo_creations"] == 1
    assert report.signals["shell_commands"] == 1
    assert report.signals["shell_argv_commands"] == 1
    assert report.signals["repo_queries"] == 1
    assert report.signals["delegations"] == 0
    report = analyze_events([_cell("t = self.todo.get('x')\nt = other\nt.v.note = 1")])
    assert report.signals["todo_state_uses"] == 0


@pytest.mark.parametrize(
    "patch",
    [
        {"schema_version": 1},
        {"schema_version": 7},
        {"schema_version": True},
        {"content_policy": "raw-content"},
        {"signals": {"unknown": 1}},
        {"rates": {"unknown": 1.0}},
    ],
)
def test_report_loader_rejects_incompatible_artifacts(tmp_path, patch):
    path = tmp_path / "behavior.json"
    path.write_text(json.dumps({**analyze_events([]).to_dict(), **patch}))
    with pytest.raises(ValueError, match="unsupported behavior"):
        load_behavior_report(path)


def test_export_counts_archived_events_and_preserves_real_ids(tmp_path, monkeypatch):
    from nooa.context_blocks import ToolCallEvent

    manager = EventManager()
    events = [
        ToolCallEvent(
            tool_call_id=str(i), name="python_cell", arguments={"code": "self.todo.status()"}
        )
        for i in range(5)
    ]
    tags = [manager.add(event) for event in events]
    manager.collapse(tags[0], tags[2], summary_text="compacted")
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    runner._write_trajectory(SimpleNamespace(event_manager=manager))
    path = tmp_path / "trajectory.json"
    payload = json.loads(path.read_text())
    assert {event.id for event in events} <= {row["event_id"] for row in payload}
    assert analyze_trajectory(path).signals["python_cells"] == 5


def test_trajectory_serialization_failure_is_nonfatal(tmp_path, monkeypatch):
    from nooa.events import PythonOutput

    circular = []
    circular.append(circular)
    manager = EventManager()
    manager.add(
        PythonOutput(
            tool_call_id="c", execution_count=1, execution_status="complete", value=circular
        )
    )
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    runner._write_trajectory(SimpleNamespace(event_manager=manager))
