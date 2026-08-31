# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner
from nooa_cli.commands import import_harbor


def _attrs(items: list[dict[str, Any]]) -> dict[str, object]:
    return {item["key"]: next(iter(item["value"].values())) for item in items}


def _make_trial(tmp_path: Path, verifier_result: dict[str, Any]) -> tuple[Path, Path]:
    job_dir = tmp_path / "harbor-job"
    trial_dir = job_dir / "trial-dir"
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "agent" / "traces").mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"stats": {"evals": {"suite": {}}}}))
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "trial-1",
                "task_name": "task-1",
                "started_at": "2026-08-27T14:00:00Z",
                "finished_at": "2026-08-27T14:05:00Z",
                "config": {"agent": {"model_name": "model-1"}},
            }
        )
    )
    (trial_dir / "verifier" / "result.json").write_text(json.dumps(verifier_result))
    return job_dir, trial_dir


def _trace_body() -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "session.id", "value": {"stringValue": "original-session"}},
                        {"key": "experiment", "value": {"stringValue": "default"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "1" * 32,
                                "spanId": "2" * 16,
                                "name": "agent.run",
                                "startTimeUnixNano": "1",
                                "endTimeUnixNano": "2",
                                "attributes": [],
                            }
                        ]
                    }
                ],
            }
        ]
    }


def test_generic_verifier_result_is_flattened_and_preserved(tmp_path: Path) -> None:
    _job_dir, trial_dir = _make_trial(
        tmp_path,
        {
            "status": "scored",
            "done": True,
            "levels_beaten": 3,
            "max_level": 10,
            "rewards": {"progress": 0.3},
            "details": {"api_calls": 18, "labels": ["not", "a", "column"]},
        },
    )

    meta = import_harbor._trial_meta_from_dir(trial_dir)
    attrs = import_harbor._harbor_resource_attrs(meta, "experiment", "batch")

    assert meta["score"] == 0.3
    assert attrs["eval.passed"] is False
    assert attrs["eval.verifier.status"] == "scored"
    assert attrs["eval.verifier.done"] is True
    assert attrs["eval.verifier.levels_beaten"] == 3
    assert attrs["eval.verifier.rewards.progress"] == 0.3
    assert attrs["eval.verifier.details.api_calls"] == 18
    assert "eval.verifier.details.labels" not in attrs

    body = import_harbor._build_eval_only_body(attrs, meta)
    resource_span = body["resourceSpans"][0]
    span = resource_span["scopeSpans"][0]["spans"][0]
    span_attrs = _attrs(span["attributes"])
    serialized_output = span_attrs["eval.output"]
    assert isinstance(serialized_output, str)
    output = json.loads(serialized_output)

    assert span["name"] == "eval"
    assert span["startTimeUnixNano"] == "1787839200000000000"
    assert span["endTimeUnixNano"] == "1787839500000000000"
    assert output["details"]["labels"] == ["not", "a", "column"]
    assert output["levels_beaten"] == 3


def test_eval_span_ids_are_stable_and_include_job_identity(tmp_path: Path) -> None:
    _job_dir, trial_dir = _make_trial(tmp_path, {"rewards": {"progress": 0.0}})
    meta = import_harbor._trial_meta_from_dir(trial_dir)
    first_attrs = import_harbor._harbor_resource_attrs(meta, "experiment", "batch-1")
    second_attrs = import_harbor._harbor_resource_attrs(meta, "experiment", "batch-2")

    first = import_harbor._build_eval_only_body(first_attrs, meta)
    repeated = import_harbor._build_eval_only_body(first_attrs, meta)
    second = import_harbor._build_eval_only_body(second_attrs, meta)

    def ids(body: dict[str, Any]) -> tuple[str, str]:
        span = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        return span["traceId"], span["spanId"]

    assert meta["score"] == 0.0
    assert ids(first) == ids(repeated)
    assert ids(first) != ids(second)


def test_missing_primary_score_does_not_become_failure(tmp_path: Path) -> None:
    _job_dir, trial_dir = _make_trial(
        tmp_path,
        {
            "status": "scored",
            "rewards": {"levels_beaten": 3},
            "metrics": {"precision": 2, "recall": 3},
        },
    )
    meta = import_harbor._trial_meta_from_dir(trial_dir)
    attrs = import_harbor._harbor_resource_attrs(meta, "experiment", "batch")
    body = import_harbor._build_eval_only_body(attrs, meta)
    span = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span_attrs = _attrs(span["attributes"])

    assert meta["score"] is None
    assert "eval.passed" not in attrs
    assert "eval.passed" not in span_attrs
    assert span["status"]["code"] == 1


def test_trace_discovery_is_content_based_and_layout_independent(tmp_path: Path) -> None:
    job_dir, trial_dir = _make_trial(tmp_path, {"rewards": {"progress": 0.5}})
    nested_dir = trial_dir / "custom" / "deeply" / "nested"
    nested_dir.mkdir(parents=True)
    trace_path = nested_dir / "events.jsonl"
    trace_path.write_text("not-json\n" + json.dumps(_trace_body()) + "\n")

    unrelated = trial_dir / "agent" / "traces" / "messages.jsonl"
    unrelated.write_text(json.dumps({"messages": ["not a trace"]}) + "\n")
    outside_trial = job_dir / "other.jsonl"
    outside_trial.write_text(json.dumps(_trace_body()) + "\n")

    assert import_harbor._find_harbor_traces(job_dir) == [trace_path]
    assert import_harbor._trial_dir_for_trace(trace_path) == trial_dir


def test_regular_import_groups_trial_files_and_adds_one_eval_span(
    tmp_path: Path, monkeypatch
) -> None:
    job_dir, trial_dir = _make_trial(tmp_path, {"rewards": {"progress": 0.5}})
    trace_dir = trial_dir / "custom" / "trace-output"
    trace_dir.mkdir(parents=True)
    for name in ("first.jsonl", "second.jsonl"):
        (trace_dir / name).write_text(json.dumps(_trace_body()) + "\n")

    posted: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(import_harbor, "check_endpoint_reachable", lambda _endpoint: True)
    monkeypatch.setattr(
        import_harbor, "_find_matching_live_session", lambda _endpoint, _meta, _exp: None
    )
    monkeypatch.setattr(import_harbor, "session_exists", lambda _endpoint, _session_id: False)
    monkeypatch.setattr(
        import_harbor,
        "post_traces_batch",
        lambda _endpoint, bodies: posted.append(bodies) or True,
    )

    result = CliRunner().invoke(
        import_harbor.command,
        [str(job_dir), "--endpoint", "http://viewer.invalid"],
    )

    assert result.exit_code == 0, result.output
    assert "2 trace file(s) across 1 Harbor trial(s)" in result.output
    assert "1 imported" in result.output
    spans = [
        span
        for bodies in posted
        for body in bodies
        for resource_span in body["resourceSpans"]
        for scope_span in resource_span["scopeSpans"]
        for span in scope_span["spans"]
    ]
    assert [span["name"] for span in spans].count("eval") == 1
    assert [span["name"] for span in spans].count("agent.run") == 2

    trace_resources = [
        _attrs(resource_span["resource"]["attributes"])
        for bodies in posted[:-1]
        for body in bodies
        for resource_span in body["resourceSpans"]
    ]
    assert all(resource["session.id"] == "trial-1" for resource in trace_resources)
    assert all(resource["experiment"] == "harbor-job" for resource in trace_resources)
    assert all("eval.verifier.rewards.progress" not in resource for resource in trace_resources)
