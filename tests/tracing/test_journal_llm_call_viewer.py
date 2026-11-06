# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End to end: a non-LiteLLM UnifiedLLM journals its calls; the viewer shows them."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from nooa.llm_types import AssistantText, LLMResponse
from nooa.unifiedllm import UnifiedLLM

JOURNAL_KEY = "nooaJournal"


class FakeGymLLM(UnifiedLLM):
    """Talks to its own server (no LiteLLM), like Gym's GymResponsesLLM."""

    def call(self, messages, tools=None, output_model=None, **kwargs):
        raise NotImplementedError

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        from nooa.tracing import journal_llm_call

        response = LLMResponse(
            parts=(AssistantText(text=f"POLICY REPLY {len(messages)}"),),
            usage={"input_tokens": 12, "output_tokens": 2},
            model_name=self.model,
        )
        journal_llm_call(
            model=self.model,
            messages=messages,
            response=response,
            latency_ms=5,
            extra={"response_id": f"resp_gym_{len(messages)}"},
        )
        return response


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    from nooa.viewer import main as main_module
    from nooa.viewer import otlp_store
    from nooa.viewer.main import app

    monkeypatch.setattr(otlp_store, "DB_PATH", tmp_path / "viewer.db")
    monkeypatch.setattr(otlp_store, "_db", None)
    monkeypatch.setattr(
        main_module, "_write_executor", ThreadPoolExecutor(max_workers=1, thread_name_prefix="t")
    )
    for tls in (otlp_store._read_tls, otlp_store._write_tls):
        if hasattr(tls, "conn"):
            del tls.conn
    with TestClient(app) as client:
        otlp_store._read_tls.conn = otlp_store._db
        otlp_store._write_tls.conn = otlp_store._db
        yield client, otlp_store
    for tls in (otlp_store._read_tls, otlp_store._write_tls):
        if hasattr(tls, "conn"):
            del tls.conn


async def _run_policy_calls(session_id: str, calls: int = 1) -> str:
    """Run ``calls`` policy calls inside one ``generation`` span; return its span id."""
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry import trace

    from nooa.tracing import flush_traces, set_session

    set_session(session_id)
    llm = FakeGymLLM("gym-policy")
    with trace.get_tracer(__name__).start_as_current_span("generation") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value
        )
        messages = [
            {"role": "system", "content": "GYM SYSTEM"},
            {"role": "user", "content": "GYM TASK"},
        ]
        for _ in range(calls):
            response = await llm.acall(messages)
            messages = [*messages, response, {"role": "user", "content": "GYM FOLLOW-UP"}]
        span_id = format(span.get_span_context().span_id, "016x")
    flush_traces()
    return span_id


def _import(client: TestClient, otlp_store, lines: list[str], session_id: str) -> None:
    """What ``nooa import-traces`` does with one file, via the test client."""
    for line in lines:
        body = json.loads(line)
        record = body.get(JOURNAL_KEY)
        if record is not None:
            if record["type"] == "blocks":
                resp = client.post(
                    "/v1/journal/blocks",
                    json=record["blocks"],
                    headers={"X-Session-Id": session_id},
                )
                assert resp.status_code == 200, resp.text
            elif record["type"] == "call":
                resp = client.post(
                    "/v1/journal/calls", json={**record["call"], "session_id": session_id}
                )
                assert resp.status_code == 200, resp.text
            continue
        if body.get("resourceSpans"):
            otlp_store.ingest(body)


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    """OTLP attribute list -> {key: scalar}."""
    out: dict[str, Any] = {}
    for attr in span.get("attributes", []):
        value = attr.get("value", {})
        scalar = next(iter(value.values()), None) if value else None
        out[attr["key"]] = int(scalar) if "intValue" in value else scalar
    return out


def _spans_by_id(spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {span["spanId"]: span for span in spans}


@pytest.mark.asyncio
async def test_journal_file_from_non_litellm_client_reaches_viewer_calls(tmp_path: Path, viewer):
    from nooa.tracing import enable_tracing, exporters

    client, otlp_store = viewer
    traces = tmp_path / "traces"
    enable_tracing(exporters=[exporters.journal_file(traces)])
    session_id = "gym-row-0"
    generation_span_id = await _run_policy_calls(session_id)

    lines = (traces / f"{session_id}.nooa.jsonl").read_text().splitlines()
    _import(client, otlp_store, lines, session_id)

    calls = client.get(f"/api/traces/{session_id}/calls").json()
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gym-policy"
    assert call["extra"] == {"response_id": "resp_gym_2"}
    assert call["tokens"] == {"prompt": 12, "completion": 2}

    # Messages are content-addressed; the session's blocks resolve them.
    blocks = otlp_store.get_session_blocks(session_id)
    inputs = [otlp_store._resolve_message(m, blocks) for m in call["input_skeleton"]]
    outputs = [otlp_store._resolve_message(m, blocks) for m in call["output_messages"]]
    assert inputs == [
        {"role": "system", "content": "GYM SYSTEM"},
        {"role": "user", "content": "GYM TASK"},
    ]
    assert outputs == [{"role": "assistant", "content": "POLICY REPLY 2"}]

    # The call has an LLM span of its own under the generation span: that is the
    # span the viewer puts the messages back onto (generation is a CHAIN span).
    spans = _spans_by_id(otlp_store.reconstruct_full_spans(session_id))
    assert call["span_id"] != generation_span_id
    llm_span = spans[call["span_id"]]
    assert llm_span["name"] == "llm_call"
    assert llm_span["parentSpanId"] == generation_span_id
    attrs = _attrs(llm_span)
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "gym-policy"
    assert attrs["llm.token_count.prompt"] == 12
    assert attrs["llm.input_messages.0.message.content"] == "GYM SYSTEM"
    assert attrs["llm.input_messages.1.message.content"] == "GYM TASK"
    assert attrs["llm.output_messages.0.message.content"] == "POLICY REPLY 2"
    assert "llm.input_messages.0.message.content" not in _attrs(spans[generation_span_id])


@pytest.mark.asyncio
async def test_each_call_in_one_generation_gets_its_own_llm_span(tmp_path: Path, viewer):
    from nooa.tracing import enable_tracing, exporters

    client, otlp_store = viewer
    traces = tmp_path / "traces"
    enable_tracing(exporters=[exporters.journal_file(traces)])
    session_id = "gym-row-1"
    generation_span_id = await _run_policy_calls(session_id, calls=2)

    lines = (traces / f"{session_id}.nooa.jsonl").read_text().splitlines()
    _import(client, otlp_store, lines, session_id)

    calls = sorted(
        client.get(f"/api/traces/{session_id}/calls").json(), key=lambda c: c["ts_start"]
    )
    assert [c["extra"]["response_id"] for c in calls] == ["resp_gym_2", "resp_gym_4"]
    assert len({c["span_id"] for c in calls}) == 2

    spans = _spans_by_id(otlp_store.reconstruct_full_spans(session_id))
    first, second = (_attrs(spans[c["span_id"]]) for c in calls)
    assert all(spans[c["span_id"]]["parentSpanId"] == generation_span_id for c in calls)
    assert first["llm.output_messages.0.message.content"] == "POLICY REPLY 2"
    assert "llm.input_messages.2.message.content" not in first
    assert second["llm.input_messages.2.message.content"] == "POLICY REPLY 2"
    assert second["llm.input_messages.3.message.content"] == "GYM FOLLOW-UP"
    assert second["llm.output_messages.0.message.content"] == "POLICY REPLY 4"


@pytest.mark.asyncio
async def test_a_client_inside_its_own_llm_span_is_linked_to_it(tmp_path: Path, viewer):
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry import trace

    from nooa.tracing import enable_tracing, exporters, flush_traces, set_session

    client, otlp_store = viewer
    traces = tmp_path / "traces"
    enable_tracing(exporters=[exporters.journal_file(traces)])
    session_id = "gym-row-2"
    set_session(session_id)
    llm = FakeGymLLM("gym-policy")
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("generation"), tracer.start_as_current_span("my.llm") as own:
        own.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        )
        await llm.acall([{"role": "user", "content": "GYM TASK"}])
        own_span_id = format(own.get_span_context().span_id, "016x")
    flush_traces()

    lines = (traces / f"{session_id}.nooa.jsonl").read_text().splitlines()
    _import(client, otlp_store, lines, session_id)

    (call,) = client.get(f"/api/traces/{session_id}/calls").json()
    assert call["span_id"] == own_span_id
    spans = _spans_by_id(otlp_store.reconstruct_full_spans(session_id))
    assert [s["name"] for s in spans.values() if s["name"] == "llm_call"] == []
    assert _attrs(spans[own_span_id])["llm.output_messages.0.message.content"] == "POLICY REPLY 1"
