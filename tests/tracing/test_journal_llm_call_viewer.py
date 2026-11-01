# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End to end: a non-LiteLLM UnifiedLLM journals its calls; the viewer shows them."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
            parts=(AssistantText(text="POLICY REPLY"),),
            usage={"input_tokens": 12, "output_tokens": 2},
            model_name=self.model,
        )
        journal_llm_call(
            model=self.model,
            messages=messages,
            response=response,
            latency_ms=5,
            extra={"response_id": "resp_gym_1"},
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


async def _run_policy_call(session_id: str) -> str:
    from opentelemetry import trace

    from nooa.tracing import flush_traces, set_session

    set_session(session_id)
    llm = FakeGymLLM("gym-policy")
    with trace.get_tracer(__name__).start_as_current_span("generation") as span:
        await llm.acall(
            [
                {"role": "system", "content": "GYM SYSTEM"},
                {"role": "user", "content": "GYM TASK"},
            ]
        )
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


@pytest.mark.asyncio
async def test_journal_file_from_non_litellm_client_reaches_viewer_calls(tmp_path: Path, viewer):
    from nooa.tracing import enable_tracing, exporters

    client, otlp_store = viewer
    traces = tmp_path / "traces"
    enable_tracing(exporters=[exporters.journal_file(traces)])
    session_id = "gym-row-0"
    span_id = await _run_policy_call(session_id)

    lines = (traces / f"{session_id}.nooa.jsonl").read_text().splitlines()
    _import(client, otlp_store, lines, session_id)

    calls = client.get(f"/api/traces/{session_id}/calls").json()
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gym-policy"
    assert call["span_id"] == span_id
    assert call["extra"] == {"response_id": "resp_gym_1"}
    assert call["tokens"] == {"prompt": 12, "completion": 2}

    # Messages are content-addressed; the session's blocks resolve them.
    blocks = otlp_store.get_session_blocks(session_id)
    inputs = [otlp_store._resolve_message(m, blocks) for m in call["input_skeleton"]]
    outputs = [otlp_store._resolve_message(m, blocks) for m in call["output_messages"]]
    assert inputs == [
        {"role": "system", "content": "GYM SYSTEM"},
        {"role": "user", "content": "GYM TASK"},
    ]
    assert outputs == [{"role": "assistant", "content": "POLICY REPLY"}]
