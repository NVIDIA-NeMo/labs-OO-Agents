# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace and journal the same public outcome regardless of transport."""

import json
import subprocess
import sys

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nooa.unifiedllm import CompletionClient, RetryConfig


def test_tracing_startup_does_not_load_provider_libraries(tmp_path):
    script = f"""
import sys
from nooa.tracing import enable_tracing, exporters
enable_tracing(exporters=[exporters.journal_file({str(tmp_path)!r})])
assert not {{'litellm', 'openai', 'anthropic'}} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_shared_trace_and_journal_parity(monkeypatch):
    from nooa.tracing import _llm_hooks
    from nooa.tracing._litellm_journal import FileMessageJournalCallback

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_llm_hooks, "_tracer", lambda: provider.get_tracer("test"))
    journals = []

    class Writer:
        def append_blocks(self, session, blocks):
            pass

        def append_call(self, session, call):
            journals.append(call)

    monkeypatch.setattr(_llm_hooks, "callbacks", [FileMessageJournalCallback(Writer())])
    raw = {
        "id": "r1",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "readable",
                    "tool_calls": [
                        {
                            "id": "call__thought__SECRET",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }

    async def respond(self, request):
        return httpx.Response(200, json=raw)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    for transport in ("litellm", "direct"):
        async with CompletionClient(
            "openai/test",
            transport=transport,
            api_key="test",
            api_base="https://models.example/v1",
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        ) as client:
            await client.acall([{"role": "user", "content": "question"}])
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert dict(spans[0].attributes) == dict(spans[1].attributes)
    assert "SECRET" not in json.dumps([dict(s.attributes) for s in spans])
    assert spans[0].attributes["llm.token_count.prompt"] == 10
    assert len(journals) == 2
    for key in ("input_skeleton", "output_messages", "tokens", "model"):
        assert journals[0][key] == journals[1][key]
    assert "SECRET" not in json.dumps(journals)


@pytest.mark.asyncio
async def test_broken_journal_does_not_repeat_or_fail_the_provider_call(monkeypatch, caplog):
    from nooa.tracing import _llm_hooks

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("broken exporter")

    sent = []

    async def respond(self, request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "r1",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    monkeypatch.setattr(_llm_hooks, "callbacks", [Broken()])
    async with CompletionClient("test", transport="direct", api_key="test") as client:
        assert (await client.acall([{"role": "user", "content": "Hello"}])).content == "ok"
    assert len(sent) == 1
    assert "journal callback failed" in caplog.text
