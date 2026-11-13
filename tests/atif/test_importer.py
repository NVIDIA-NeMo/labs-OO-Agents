# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF -> OTLP import: span layout, reconstructed LLM inputs, and viewer ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, atif_scope, trajectory_to_otlp
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _spans(body: dict) -> list[dict]:
    return body["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(item: dict) -> dict:
    return {a["key"]: next(iter(a["value"].values())) for a in item["attributes"]}


def _by_name(spans: list[dict], name: str) -> list[dict]:
    return [s for s in spans if s["name"] == name]


TRAJECTORY = {
    "schema_version": "ATIF-v1.7",
    "session_id": "s",
    "trajectory_id": "task/r0",
    "agent": {"name": "agent", "version": "1", "model_name": "m"},
    "steps": [
        {"step_id": 1, "source": "system", "message": "sys"},
        {"step_id": 2, "source": "user", "message": [{"type": "text", "text": "task"}]},
        {
            "step_id": 3,
            "source": "agent",
            "message": "",
            "timestamp": "2026-11-12T14:41:55+00:00",
            "reasoning_content": "think",
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "python_cell", "arguments": {"code": "1"}},
                {"tool_call_id": "c2", "function_name": "delegate", "arguments": {"q": "x"}},
            ],
            "metrics": {"prompt_tokens": 10, "completion_tokens": 2},
            "extra": {"duration_ms": 1000},
            "observation": {
                "results": [
                    {"source_call_id": "c1", "content": "ok", "extra": {"duration_ms": 5}},
                    {
                        "source_call_id": "c2",
                        "content": "sub done",
                        "subagent_trajectory_ref": [{"trajectory_id": "child"}],
                    },
                ]
            },
        },
        {"step_id": 4, "source": "user", "message": "stdout: 1"},
        {
            "step_id": 5,
            "source": "agent",
            "message": "done",
            "timestamp": "2026-11-12T14:41:58+00:00",
            "tool_calls": [
                {"tool_call_id": "c3", "function_name": "python_cell", "arguments": {"code": "x"}}
            ],
            "observation": {
                "results": [
                    {"source_call_id": "c3", "content": "boom", "extra": {"error_type": "E"}}
                ]
            },
        },
    ],
    "subagent_trajectories": [
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "child",
            "agent": {"name": "helper", "version": "1"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "x"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "sub done",
                    "timestamp": "2026-11-12T14:41:56+00:00",
                },
            ],
        }
    ],
}


def test_span_tree_and_subagent_nesting():
    spans = _spans(trajectory_to_otlp(TRAJECTORY, "sess", experiment="exp"))
    root = spans[0]
    assert root["name"] == "agent.task/r0" and "parentSpanId" not in root
    assert {s["traceId"] for s in spans} == {root["traceId"]}
    generations = [s for s in spans if s.get("parentSpanId") == root["spanId"]]
    assert [g["name"] for g in generations] == ["generation 1", "generation 2"]
    code = _by_name(spans, "code_execution")
    assert len(code) == 2 and _attrs(code[0])["nooa.viewer.plugin"] == "code_execution"
    assert code[1]["status"]["code"] == 2  # error_type set
    delegate = _by_name(spans, "tool:delegate")[0]
    child = _by_name(spans, "helper.child")[0]
    assert child["parentSpanId"] == delegate["spanId"]
    assert int(delegate["endTimeUnixNano"]) >= int(child["endTimeUnixNano"])
    ids = [s["spanId"] for s in spans]
    assert len(ids) == len(set(ids))


def test_llm_inputs_are_rebuilt_from_history():
    spans = _spans(trajectory_to_otlp(TRAJECTORY, "sess"))
    calls = [_attrs(s) for s in _by_name(spans, "llm_call")]
    first, second = (a for a in calls if a.get("llm.model_name") == "m")
    assert first["llm.reasoning_content"] == "think"
    assert first["llm.token_count.total"] == "12"
    assert first["llm.input_messages.1.message.content"] == "task"
    roles = [second[f"llm.input_messages.{i}.message.role"] for i in range(6)]
    assert roles == ["system", "user", "assistant", "tool", "tool", "user"]
    args = second["llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments"]
    assert json.loads(args) == {"code": "1"}
    assert second["llm.output_messages.0.message.content"] == "done"


def test_root_and_resource_attributes():
    body = trajectory_to_otlp(
        TRAJECTORY, "sess", experiment="exp", resource={"batch_id": "b"}, output="42", failed=True
    )
    resource = _attrs(body["resourceSpans"][0]["resource"])
    assert resource["session.id"] == "sess"
    assert resource["experiment"] == "exp" and resource["batch_id"] == "b"
    root = _spans(body)[0]
    assert _attrs(root)["output.value"] == "42"
    assert _attrs(root)["input.value"] == "task"
    assert root["status"]["code"] == 2


def test_code_tools_are_configurable():
    spans = _spans(trajectory_to_otlp(TRAJECTORY, "sess", code_tools={"delegate"}))
    assert len(_by_name(spans, "code_execution")) == 1
    assert len(_by_name(spans, "tool:python_cell")) == 2


class _CodeActAgent(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def run(self, prompt: str) -> int:
        """Solve the prompt."""
        ...


@pytest.mark.asyncio
async def test_round_trip_from_nooa_export_into_viewer_store(tmp_path: Path, monkeypatch):
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="e1", name="execute_python", arguments=json.dumps({"code": "x = 2"})
                    ),
                    ToolCall(id="r1", name="return_result", arguments=json.dumps({"result": 2})),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 50, "completion_tokens": 10},
            )
        ]
    )
    agent = _CodeActAgent(llm=llm)
    out = tmp_path / "trajectory.json"
    async with atif_scope(agent, path=out, session_id="rt", agent_name="A", agent_version="1"):
        assert await agent.run("compute") == 2

    trajectory = Trajectory.model_validate_json(out.read_text())
    body = trajectory_to_otlp(trajectory, "rt-session")
    spans = _spans(body)
    code = _by_name(spans, "code_execution")
    assert json.loads(_attrs(code[0])["input.value"]) == {"code": "x = 2"}
    assert _by_name(spans, "tool:return_result")

    from nooa.viewer import otlp_store

    monkeypatch.setattr(otlp_store, "DB_PATH", tmp_path / "traces.db")
    otlp_store.init_db()
    otlp_store.ingest(body)
    assert len(otlp_store.get_session_spans("rt-session")) == len(spans)
