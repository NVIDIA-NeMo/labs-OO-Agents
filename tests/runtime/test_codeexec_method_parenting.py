# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A method invoked from inside generated code (e.g. ``self.submit()`` during
``execute_python``) should nest under the ``code_execution`` span it ran in — not
under the enclosing agent-method span.

Before the fix, ``before_agent_call`` chose its parent solely from ``parent_call_id``
(nearest enclosing *method*), so such calls became siblings of the long ``generation``
span and rendered "at the end" of the loop instead of at their true position.
"""

import json

import pytest

from nooa import Agent
from nooa.runtime.hooks import set_hooks
from nooa.unifiedllm import LLMResponse, ToolCall


class _FakeLLM:
    """Emits one ``execute_python`` call whose code invokes ``self.submit()``
    twice, calls a nested helper, then terminates via ``return_result()``."""

    def __init__(self) -> None:
        self.call_count = 0

    async def acall(self, messages, tools=None, **kwargs):
        self.call_count += 1
        code = "self.submit(1)\nself.submit(2)\nreturn_result({'done': True})\n"
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=f"fake_{self.call_count}",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                ),
            ],
            finish_reason="tool_calls",
            assistant_message={},
        )


class _SubmitAgent(Agent):
    def submit(self, value: int) -> str:
        """Submit a candidate value; calls a helper to format the receipt."""
        return self._format(value)

    def _format(self, value: int) -> str:
        """Format a submission receipt."""
        return f"submitted {value}"

    async def run(self) -> dict:
        """Run the task by submitting candidates from generated code."""
        ...


class _AsyncFakeLLM:
    """Like _FakeLLM but the code uses ``await self.submit(...)`` (the real
    cybergym pattern — submit is an async method awaited from generated code)."""

    def __init__(self) -> None:
        self.call_count = 0

    async def acall(self, messages, tools=None, **kwargs):
        self.call_count += 1
        code = "await self.submit(1)\nawait self.submit(2)\nreturn_result({'done': True})\n"
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=f"fake_{self.call_count}",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                ),
            ],
            finish_reason="tool_calls",
            assistant_message={},
        )


class _AsyncSubmitAgent(Agent):
    async def submit(self, value: int) -> str:
        """Submit a candidate value (async, awaited from generated code)."""
        return f"submitted {value}"

    async def run(self) -> dict:
        """Run the task by awaiting submit() from generated code."""
        ...


@pytest.fixture
def in_memory_spans():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from nooa.tracing import NemoOOAgentsInstrumentor

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    NemoOOAgentsInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    set_hooks(None)


@pytest.mark.asyncio
async def test_method_called_from_code_nests_under_code_execution(in_memory_spans):
    agent = _SubmitAgent(llm=_FakeLLM())
    await agent.run()

    spans = in_memory_spans.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}

    def parent_name(s):
        p = s.parent
        return by_id[p.span_id].name if (p and p.span_id in by_id) else None

    submit_spans = [s for s in spans if s.name == "method.submit"]
    code_execs = [s for s in spans if s.name == "code_execution"]
    helper_spans = [s for s in spans if s.name == "method._format"]

    assert len(submit_spans) == 2, f"expected 2 submit spans, got {len(submit_spans)}"
    assert code_execs, "expected a code_execution span"

    # 1. submit() (called directly from the executing code) nests under code_execution.
    for s in submit_spans:
        assert parent_name(s) == "code_execution", (
            f"submit parented to {parent_name(s)!r}, expected 'code_execution'"
        )

    # 2. The transitive helper call (_format, called by submit) still nests under submit.
    assert helper_spans, "expected _format helper spans"
    for s in helper_spans:
        assert parent_name(s) == "method.submit", (
            f"_format parented to {parent_name(s)!r}, expected 'method.submit'"
        )

    # 3. The agent.parent_call_id attribute is unchanged (still the enclosing method),
    #    so event/ATIF semantics are preserved even though the span parent moved.
    run_span = next(s for s in spans if s.name == "method.run")
    run_call_id = run_span.attributes.get("agent.call_id")
    for s in submit_spans:
        assert s.attributes.get("agent.parent_call_id") == run_call_id, (
            "submit's agent.parent_call_id attribute should still point at run()"
        )


@pytest.mark.asyncio
async def test_async_submit_from_code_nests_under_code_execution(in_memory_spans):
    """The real cybergym pattern: `await self.submit(...)` from generated code."""
    agent = _AsyncSubmitAgent(llm=_AsyncFakeLLM())
    await agent.run()

    spans = in_memory_spans.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}

    def parent_name(s):
        p = s.parent
        return by_id[p.span_id].name if (p and p.span_id in by_id) else None

    submit_spans = [s for s in spans if s.name == "method.submit"]
    assert len(submit_spans) == 2, f"expected 2 submit spans, got {len(submit_spans)}"
    for s in submit_spans:
        assert parent_name(s) == "code_execution", (
            f"async submit parented to {parent_name(s)!r}, expected 'code_execution'"
        )
