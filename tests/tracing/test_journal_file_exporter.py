# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable journal-file export and import coverage."""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _spans(body: dict):
    for resource_spans in body.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            yield from scope_spans.get("spans", [])


@pytest.mark.asyncio
async def test_journal_file_is_stripped_and_import_reconstructs_messages(tmp_path: Path):
    pytest.importorskip("openinference.instrumentation.litellm")
    import litellm
    from opentelemetry import trace as otel_trace

    from nooa.tracing import enable_tracing, exporters, flush_traces, set_session
    from nooa.tracing._context_sideband import JournalPayload, set_journal_payload

    enable_tracing(exporters=[exporters.journal_file(tmp_path)])
    session_id = "portable-journal"
    set_session(session_id)
    input_text = "PORTABLE_INPUT " + ("repeated " * 100)
    import hashlib

    input_hash = "sha256:" + hashlib.sha256(input_text.encode()).hexdigest()
    payload = JournalPayload(
        skeleton=[{"role": "user", "parts": [{"block_hash": input_hash}]}],
        blocks={input_hash: input_text},
    )
    callback = next(
        callback
        for callback in litellm.callbacks
        if type(callback).__name__ == "FileMessageJournalCallback"
    )
    kwargs = {"litellm_call_id": "portable-call", "model": "gpt-3.5-turbo"}
    messages = [{"role": "user", "content": input_text}]
    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("acompletion") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        set_journal_payload(payload)
        callback.log_pre_api_call("gpt-3.5-turbo", messages, kwargs)
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=messages,
            mock_response="PORTABLE_OUTPUT",
        )
        now = datetime.now(UTC)
        callback.log_success_event(kwargs, response, now, now)
    flush_traces()

    artifact = tmp_path / f"{session_id}.nooa.jsonl"
    assert artifact.exists()
    bodies = [json.loads(line) for line in artifact.read_text().splitlines()]
    records = [body["nooaJournal"] for body in bodies if "nooaJournal" in body]
    assert {record["type"] for record in records} == {"manifest", "blocks", "call"}
    assert sum(input_text in line for line in artifact.read_text().splitlines()) == 1

    otlp_bodies = [body for body in bodies if "resourceSpans" in body]
    llm_spans = [
        span
        for body in otlp_bodies
        for span in _spans(body)
        if any(
            attr.get("key") == "openinference.span.kind"
            and attr.get("value", {}).get("stringValue") == "LLM"
            for attr in span.get("attributes", [])
        )
    ]
    assert llm_spans
    keys = {attr["key"] for attr in llm_spans[0].get("attributes", [])}
    assert not any(key.startswith("llm.input_messages.") for key in keys)
    assert "input.value" not in keys

    from eval_pipeline.headless_backend import HeadlessOtlpBackend

    backend = HeadlessOtlpBackend()
    endpoint = backend.start()
    try:
        from click.testing import CliRunner
        from nooa_cli.commands.import_traces import command

        result = CliRunner().invoke(command, [str(artifact), "--endpoint", endpoint])
        assert result.exit_code == 0, result.output
        urllib.request.urlopen(
            urllib.request.Request(f"{endpoint}/v1/sync", method="POST"), timeout=10
        )
        with urllib.request.urlopen(
            f"{endpoint}/api/trace/export?session_id={session_id}", timeout=10
        ) as response:
            reconstructed = response.read().decode()
        assert input_text in reconstructed
        assert "PORTABLE_OUTPUT" in reconstructed
    finally:
        backend.stop()


def test_harbor_import_posts_journal_with_trial_session(monkeypatch, tmp_path: Path):
    from nooa_cli.commands import import_harbor

    artifact = tmp_path / "trace.nooa.jsonl"
    artifact.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "nooaJournal": {
                            "format": "nooa.message_journal",
                            "version": 1,
                            "type": "call",
                            "session_id": "original",
                            "call": {
                                "call_id": "c1",
                                "session_id": "original",
                                "input_skeleton": [],
                                "output_messages": [],
                            },
                        }
                    }
                ),
                json.dumps({"resourceSpans": []}),
            ]
        )
        + "\n"
    )
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        import_harbor,
        "post_journal_record",
        lambda _endpoint, record, session_id: posted.append((record["type"], session_id)) or True,
    )
    monkeypatch.setattr(import_harbor, "post_traces_batch", lambda _endpoint, _bodies: True)

    imported, errors = import_harbor._import_trace_file(
        "http://viewer",
        artifact,
        {"session.id": "harbor-trial"},
        batch_lines=100,
        batch_bytes=1_000_000,
    )
    assert imported
    assert not errors
    assert posted == [("call", "harbor-trial")]


@pytest.mark.parametrize("file_first", [False, True])
def test_http_and_file_journal_callbacks_coexist_in_either_order(tmp_path: Path, file_first: bool):
    import litellm

    from nooa.tracing import exporters
    from nooa.tracing._litellm_journal import (
        FileMessageJournalCallback,
        MessageJournalCallback,
    )

    factories = [
        lambda: exporters.journal("http://example.invalid/v1/traces"),
        lambda: exporters.journal_file(tmp_path),
    ]
    if file_first:
        factories.reverse()
    created = [factory() for factory in factories]
    try:
        assert sum(type(cb) is MessageJournalCallback for cb in litellm.callbacks) == 1
        assert sum(type(cb) is FileMessageJournalCallback for cb in litellm.callbacks) == 1
    finally:
        for exporter in reversed(created):
            exporter.shutdown()
