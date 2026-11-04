# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""journal_llm_call(): the journal for clients that bypass LiteLLM."""

from __future__ import annotations

import litellm
import pytest

from nooa.events import DebugTrace, Error
from nooa.llm_types import AssistantText, CacheBoundary, LLMResponse, ToolCall
from nooa.tracing import journal_llm_call, set_session
from nooa.tracing._context_sideband import (
    JournalPayload,
    get_journal_payload,
    set_journal_payload,
)
from nooa.tracing._litellm_journal import FileMessageJournalCallback, MessageJournalCallback
from nooa.viewer.otlp_store import _resolve_message


class FakeWriter:
    def __init__(self) -> None:
        self.blocks: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []
        self.block_sessions: list[str] = []

    def append_blocks(self, session_id: str, blocks: list[dict[str, str]]) -> None:
        if not blocks:  # JournalFileWriter.append_blocks writes nothing for []
            return
        self.block_sessions.append(session_id)
        self.blocks.update({b["hash"]: b["content"] for b in blocks})

    def append_call(self, session_id: str, call: dict) -> None:
        self.calls.append((session_id, call))


class FailingWriter(FakeWriter):
    def append_call(self, session_id: str, call: dict) -> None:
        raise OSError("disk full")


@pytest.fixture
def writer(monkeypatch):
    writer = FakeWriter()
    monkeypatch.setattr(litellm, "callbacks", [FileMessageJournalCallback(writer)])
    set_journal_payload(None)
    yield writer
    set_journal_payload(None)


def _resolved(writer: FakeWriter, call: dict, key: str) -> list[dict]:
    return [_resolve_message(m, writer.blocks) for m in call[key]]


def _assistant(text: str, *calls: ToolCall, usage: dict | None = None) -> LLMResponse:
    return LLMResponse(
        parts=(AssistantText(text=text), *calls),
        finish_reason="tool_calls" if calls else "stop",
        usage=usage,
        model_name="gym-policy",
    )


def test_records_call_with_mixed_nooa_messages(writer):
    set_session("journal-api-session")
    replayed = _assistant("calling", ToolCall(id="t1", name="run", arguments='{"x": 1}'))
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": [{"type": "text", "text": "STRUCTURED USER"}]},
        replayed,
        {"role": "tool", "tool_call_id": "t1", "content": "TOOL RESULT"},
        CacheBoundary(),
        DebugTrace(content="never sent"),
        Error(content="RETRY ERROR"),
    ]
    response = _assistant(
        "FINAL ANSWER", usage={"prompt_tokens": 10, "completion_tokens": 3, "cached_tokens": 4}
    )

    journal_llm_call(
        model="gym-policy",
        messages=messages,
        response=response,
        call_id="gym-call-1",
        started_at=100.0,
        latency_ms=250,
        extra={"response_id": "resp_123"},
    )

    assert len(writer.calls) == 1
    session_id, call = writer.calls[0]
    assert session_id == call["session_id"] == "journal-api-session"
    assert call["call_id"] == "gym-call-1"
    assert call["model"] == "gym-policy"
    assert call["ts_start"] == 100.0
    assert call["extra"] == {"response_id": "resp_123"}
    assert call["tokens"] == {"prompt": 10, "completion": 3, "cached": 4}

    inputs = _resolved(writer, call, "input_skeleton")
    assert [m["role"] for m in inputs] == ["system", "user", "assistant", "tool", "user"]
    assert inputs[0]["content"] == "SYSTEM PROMPT"
    assert inputs[1]["content"] == [{"type": "text", "text": "STRUCTURED USER"}]
    assert inputs[2]["content"] == "calling"
    assert inputs[2]["tool_calls"][0]["function"] == {"name": "run", "arguments": '{"x": 1}'}
    assert inputs[3] == {"role": "tool", "tool_call_id": "t1", "content": "TOOL RESULT"}
    assert inputs[4]["content"] == "RETRY ERROR"
    # Large fields travel as content-addressed blocks, not inline.
    assert "parts" in call["input_skeleton"][0] and "content" not in call["input_skeleton"][0]

    outputs = _resolved(writer, call, "output_messages")
    assert outputs == [{"role": "assistant", "content": "FINAL ANSWER"}]


def test_latency_defaults_start_and_failed_call_has_no_output(writer):
    journal_llm_call(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        response=None,
        session_id="explicit",
        ended_at=10.0,
        latency_ms=2000,
    )
    session_id, call = writer.calls[0]
    assert session_id == "explicit"
    assert (call["ts_start"], call["ts_end"]) == (8.0, 10.0)
    assert call["output_messages"] == [] and call["tokens"] is None
    assert call["call_id"].startswith("journal-")


def test_consumes_rendered_context_from_the_runtime(writer):
    payload = JournalPayload(
        skeleton=[{"role": "user", "parts": [{"block_hash": "sha256:abc", "key": "k"}]}],
        blocks={"sha256:abc": "RENDERED BLOCK"},
    )
    set_journal_payload(payload)
    journal_llm_call(
        model="m",
        messages=[{"role": "user", "content": "raw"}],
        response=_assistant("ok"),
        session_id="s",
    )
    call = writer.calls[0][1]
    assert call["input_skeleton"] == payload.skeleton
    assert writer.blocks["sha256:abc"] == "RENDERED BLOCK"
    assert get_journal_payload() is None

    set_journal_payload(payload)
    journal_llm_call(
        model="m",
        messages=[{"role": "user", "content": "raw"}],
        session_id="s",
        use_rendered_context=False,
    )
    assert _resolved(writer, writer.calls[1][1], "input_skeleton") == [
        {"role": "user", "content": "raw"}
    ]


def test_blocks_are_sent_once_per_session(writer):
    for _ in range(2):
        journal_llm_call(model="m", messages=[{"role": "user", "content": "same"}], session_id="s")
    assert len(writer.calls) == 2
    assert writer.block_sessions == ["s"]


def test_noop_without_a_journal_sink(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    payload = JournalPayload(skeleton=[], blocks={})
    set_journal_payload(payload)
    try:
        journal_llm_call(model="m", messages=[{"role": "user", "content": "x"}])
        # Nothing consumed the runtime's sideband either.
        assert get_journal_payload() is payload
    finally:
        set_journal_payload(None)


def test_never_raises(monkeypatch, caplog):
    failing = FailingWriter()
    ok = FakeWriter()
    file_cb = FileMessageJournalCallback(failing)
    file_cb.add_writer(ok)
    monkeypatch.setattr(litellm, "callbacks", [file_cb])
    journal_llm_call(model="m", messages=[{"role": "user", "content": "x"}], session_id="s")
    journal_llm_call(model="m", messages=[object(), 42], response=object(), extra={"o": object()})
    journal_llm_call(model="m", messages=None, response=_assistant("x"))  # type: ignore[arg-type]
    # A failing sink drops its own record; the call itself never raises.
    assert ok.calls == []
    monkeypatch.setattr(litellm, "callbacks", [FileMessageJournalCallback(ok)])
    journal_llm_call(model="m", messages=[object()], response=object(), session_id="s")
    assert ok.calls[-1][1]["input_skeleton"] == [{"unsupported_type": "object"}]


def test_live_viewer_sink_posts_blocks_and_call(monkeypatch):
    import nooa.tracing._litellm_journal as journal_mod

    posted: list[tuple[str, object]] = []

    def fake_post(url, payload, *, session_id="", timeout=15.0, on_success=None):
        posted.append((url, payload))
        if on_success:
            on_success()

    monkeypatch.setattr(journal_mod, "_post_json", fake_post)
    monkeypatch.setattr(litellm, "callbacks", [MessageJournalCallback("http://viewer:5001")])
    journal_llm_call(
        model="gym-policy",
        messages=[{"role": "user", "content": "hello"}],
        response=_assistant("world"),
        session_id="live",
        extra={"response_id": "r1"},
    )
    urls = [url for url, _ in posted]
    assert urls == [
        "http://viewer:5001/v1/journal/blocks",
        "http://viewer:5001/v1/journal/calls",
    ]
    call = posted[1][1]
    assert call["session_id"] == "live" and call["extra"] == {"response_id": "r1"}
    assert {b["content"] for b in posted[0][1]} == {"hello", "world"}


def test_an_explicit_span_id_is_kept(writer):
    journal_llm_call(
        model="m", messages=[{"role": "user", "content": "hi"}], session_id="s", span_id="ab" * 8
    )
    journal_llm_call(model="m", messages=[{"role": "user", "content": "hi"}], session_id="s")
    (_, explicit), (_, default) = writer.calls
    assert explicit["span_id"] == "ab" * 8
    # The default is a span of the call's own (or none without a tracer), never the explicit one.
    assert default.get("span_id") != "ab" * 8


def test_generated_llm_span_takes_the_record_session(writer, monkeypatch):
    """An explicit session_id that differs from the current trace session is used
    for the generated LLM span too, so the viewer can attach the call to it."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nooa.tracing._session_processor import SessionSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", lambda *a, **k: provider.get_tracer("t"))

    set_session("ambient-session")
    journal_llm_call(
        model="m", messages=[{"role": "user", "content": "hi"}], session_id="explicit-session"
    )
    journal_llm_call(model="m", messages=[{"role": "user", "content": "hi"}])

    (s1, explicit), (s2, ambient) = writer.calls
    spans = {format(s.context.span_id, "016x"): s for s in exporter.get_finished_spans()}
    assert s1 == explicit["session_id"] == "explicit-session"
    assert spans[explicit["span_id"]].attributes["session.id"] == "explicit-session"
    assert s2 == ambient["session_id"] == "ambient-session"
    assert spans[ambient["span_id"]].attributes["session.id"] == "ambient-session"


def test_a_failed_call_marks_the_reused_llm_span(writer, monkeypatch):
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode, use_span

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    llm_kind = {SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value}
    for response in (None, _assistant("ok")):
        span = provider.get_tracer("t").start_span("client", attributes=llm_kind)
        with use_span(span, end_on_exit=True):
            journal_llm_call(
                model="m", messages=[{"role": "user", "content": "hi"}], response=response
            )
    failed, ok = exporter.get_finished_spans()
    assert [c["span_id"] for _, c in writer.calls] == [
        format(s.context.span_id, "016x") for s in (failed, ok)
    ]
    assert failed.status.status_code is StatusCode.ERROR
    assert ok.status.status_code is StatusCode.UNSET
