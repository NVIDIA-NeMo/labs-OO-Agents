# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenInference semantic-conventions conformance tests.

Locks in that the spans nooa emits stay compatible with the published
OpenInference semantic conventions
(https://arize-ai.github.io/openinference/spec/semantic_conventions.html).

Design:

* All assertions validate against the ``openinference.semconv.trace`` constants
  rather than hardcoded strings, so the suite fails if an emitted attribute
  name/value drifts from the upstream spec.
* Two independent fixtures:
  - ``framework_spans`` runs a small CodeAct ``FakeLLM`` agent through the real
    instrumentation hooks (no network, no ``litellm.acompletion``).  It produces
    the framework spans: AGENT (``method.*``), ``generation`` (CHAIN),
    ``code_execution`` (TOOL), and — depending on the strategy — ``method_call``
    / ``tool_execution`` / ``context_snapshot``.  It produces **no** ``LLM`` span.
  - ``llm_span`` drives ``litellm.acompletion(mock_response=...)`` so the
    ``LiteLLMInstrumentor`` (+ our ``apply_litellm_patch``) emits the real
    ``LLM`` span with ``llm.*`` attributes and ``tool_call.id``.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa import Agent
from nooa.runtime.hooks import set_hooks
from nooa.unifiedllm import LLMResponse, ToolCall

# Set of all valid OpenInference span-kind string values (tracks upstream).
VALID_SPAN_KINDS = {v.value for v in OpenInferenceSpanKindValues}

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND


# ---------------------------------------------------------------------------
# Framework-spans fixture: a CodeAct agent driven by a scripted FakeLLM.
# ---------------------------------------------------------------------------


class _CodeActFakeLLM:
    """Emits one ``execute_python`` call that invokes ``self.lookup()`` then
    terminates via ``return_result()``.  Network-free; never calls litellm."""

    def __init__(self) -> None:
        self.call_count = 0

    async def acall(self, messages, tools=None, **kwargs):
        self.call_count += 1
        code = "x = self.lookup('widget')\nreturn_result({'stock': x})\n"
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.call_count}",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                ),
            ],
            finish_reason="tool_calls",
            assistant_message={},
        )


class _InventoryAgent(Agent):
    def lookup(self, item: str) -> int:
        """Return the stock count for an item."""
        return {"widget": 7}.get(item, 0)

    async def run(self, query: str) -> dict:
        """Answer {query} by looking up inventory from generated code."""
        ...


@pytest.fixture
def framework_spans():
    """Run the CodeAct FakeLLM agent and return the finished spans (in-memory)."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from nooa.tracing import NemoOOAgentsInstrumentor
    from nooa.tracing._session import set_session
    from nooa.tracing._session_processor import SessionSpanProcessor

    exporter = InMemorySpanExporter()
    # OTel's global TracerProvider cannot be reset once set, so reuse it when present
    # (the runtime resolves the hooks' tracer against it); otherwise create one.
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    # SessionSpanProcessor stamps session.id; the in-memory exporter captures spans.
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    NemoOOAgentsInstrumentor().instrument(tracer_provider=provider)
    set_session("conformance-session")
    try:
        yield exporter
    finally:
        # Teardown: clear the registered hooks and the session so they do not leak
        # into later tests (the tracing conftest also resets module state, but be
        # explicit here since this fixture sets a session). The OTel global provider
        # cannot be torn down, so its processors are left in place (harmless — each
        # test reads only its own in-memory exporter).
        set_hooks(None)
        set_session(None)


async def _run_agent(exporter) -> dict:
    """Run the agent and return ``{span_name: [span, ...]}`` grouped by name."""
    agent = _InventoryAgent(llm=_CodeActFakeLLM())
    await agent.run("how many widgets?")
    spans = exporter.get_finished_spans()
    by_name: dict[str, list] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)
    return by_name


def _attr(span, key):
    return span.attributes.get(key)


# ---------------------------------------------------------------------------
# _safe_json_value structure preservation
# ---------------------------------------------------------------------------


def test_safe_json_value_preserves_dict_keys_on_overflow():
    """Even when the serialized input exceeds the cap, the top-level keys survive
    so readers can still extract ``args``/``kwargs``/``code`` by key (the value is
    application/json-tagged, so it must parse back to a dict)."""
    from nooa.tracing._hooks_impl import OpenInferenceHooks

    big = {"args": ["x" * 200_000], "kwargs": {"k": "y" * 200_000}}
    out = OpenInferenceHooks._safe_json_value(big, max_chars=50_000)
    parsed = json.loads(out)  # must be valid JSON
    assert isinstance(parsed, dict), f"overflow output should stay a dict, got {type(parsed)}"
    assert set(parsed) == {"args", "kwargs"}
    assert len(out) <= 50_000  # bounded

    # The common code shape also round-trips by key.
    code_out = OpenInferenceHooks._safe_json_value({"code": "z" * 200_000}, max_chars=50_000)
    assert "code" in json.loads(code_out)


# ---------------------------------------------------------------------------
# Framework span-kind conformance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_framework_span_kinds_are_valid_enum_members(framework_spans):
    """Every emitted ``openinference.span.kind`` is a member of the spec enum."""
    by_name = await _run_agent(framework_spans)
    all_spans = [s for spans in by_name.values() for s in spans]
    assert all_spans, "no spans emitted by the agent run"
    for s in all_spans:
        kind = _attr(s, SPAN_KIND)
        assert kind is not None, f"span {s.name!r} has no {SPAN_KIND}"
        assert kind in VALID_SPAN_KINDS, (
            f"span {s.name!r} kind {kind!r} is not a valid OpenInference span kind "
            f"({sorted(VALID_SPAN_KINDS)})"
        )


@pytest.mark.asyncio
async def test_agent_method_span_is_agent_kind_with_io(framework_spans):
    """AGENT spans carry kind=AGENT + standard input.value/output.value."""
    by_name = await _run_agent(framework_spans)
    run_spans = by_name.get("method.run")
    assert run_spans, f"no method.run span; names={list(by_name)}"
    span = run_spans[0]

    assert _attr(span, SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value

    # input.value must be valid JSON and JSON mime-typed.
    input_value = _attr(span, SpanAttributes.INPUT_VALUE)
    assert input_value is not None, "AGENT span missing input.value"
    parsed = json.loads(input_value)  # must be valid JSON
    assert "args" in parsed and "kwargs" in parsed
    assert _attr(span, SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value

    # output.value present, text mime-typed.
    assert _attr(span, SpanAttributes.OUTPUT_VALUE) is not None
    assert _attr(span, SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.TEXT.value

    # OI-only export: the legacy native I/O attrs are no longer
    # emitted — input.value/output.value are the single canonical representation.
    assert _attr(span, "agent.args") is None
    assert _attr(span, "agent.kwargs") is None
    assert _attr(span, "agent.result") is None


@pytest.mark.asyncio
async def test_generation_span_is_chain_not_llm(framework_spans):
    """Regression guard: the framework ``generation`` span is
    CHAIN, and the FakeLLM run emits NO LLM-kind span (the real LLM span only
    exists when litellm.acompletion runs — see test_llm_span_* below)."""
    by_name = await _run_agent(framework_spans)
    gen_spans = by_name.get("generation")
    assert gen_spans, f"no generation span; names={list(by_name)}"
    for span in gen_spans:
        assert _attr(span, SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value, (
            "generation span must be CHAIN — the nested litellm.acompletion span "
            "is the real LLM call"
        )
        # CHAIN output rendering.
        assert _attr(span, SpanAttributes.OUTPUT_VALUE) is not None

    all_spans = [s for spans in by_name.values() for s in spans]
    llm_spans = [
        s for s in all_spans if _attr(s, SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    ]
    assert not llm_spans, (
        f"FakeLLM run should emit no LLM-kind span, got {[s.name for s in llm_spans]}"
    )


@pytest.mark.asyncio
async def test_code_execution_span_is_tool_with_io(framework_spans):
    """``code_execution`` is a TOOL span with tool.name + standard input/output."""
    by_name = await _run_agent(framework_spans)
    spans = by_name.get("code_execution")
    assert spans, f"no code_execution span; names={list(by_name)}"
    span = spans[0]
    assert _attr(span, SPAN_KIND) == OpenInferenceSpanKindValues.TOOL.value
    assert _attr(span, SpanAttributes.TOOL_NAME) == "python_executor"
    # input.value is JSON {"code": ...}, application/json.
    input_value = _attr(span, SpanAttributes.INPUT_VALUE)
    assert input_value is not None
    assert "code" in json.loads(input_value), "code_execution input.value must be {'code': ...}"
    assert _attr(span, SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert _attr(span, SpanAttributes.OUTPUT_VALUE) is not None
    # output.mime_type is JSON for an ExecutionResult (the value is JSON-encoded
    # stdout/stderr/returned_value), else text/plain — both are valid spec values.
    assert _attr(span, SpanAttributes.OUTPUT_MIME_TYPE) in {
        OpenInferenceMimeTypeValues.JSON.value,
        OpenInferenceMimeTypeValues.TEXT.value,
    }
    # Tool-call identity: function name + JSON args + id (fallback to the
    # execution id when there is no model-provided tool-call id).
    assert _attr(span, ToolCallAttributes.TOOL_CALL_FUNCTION_NAME) == "python_executor"
    args_json = _attr(span, ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON)
    assert "code" in json.loads(args_json), "tool_call.function.arguments must be valid JSON"
    assert _attr(span, ToolCallAttributes.TOOL_CALL_ID), "code_execution missing tool_call.id"
    assert _attr(span, SpanAttributes.TOOL_ID), "code_execution missing tool.id"


@pytest.mark.asyncio
async def test_tool_spans_conformance_when_present(framework_spans):
    """``method_call.*`` / ``tool_execution.*`` spans (if the strategy emits any)
    are TOOL-kind with tool.name and standard I/O attrs."""
    by_name = await _run_agent(framework_spans)
    tool_spans = [
        s
        for name, spans in by_name.items()
        for s in spans
        if name.startswith(("method_call.", "tool_execution."))
    ]
    if not tool_spans:
        pytest.skip("strategy emitted no method_call/tool_execution spans")
    for span in tool_spans:
        assert _attr(span, SPAN_KIND) == OpenInferenceSpanKindValues.TOOL.value
        assert _attr(span, SpanAttributes.TOOL_NAME) is not None
        # input.value is valid JSON (tool args).
        input_value = _attr(span, SpanAttributes.INPUT_VALUE)
        assert input_value is not None
        json.loads(input_value)  # must parse
        # Tool-call identity.
        assert _attr(span, ToolCallAttributes.TOOL_CALL_FUNCTION_NAME) is not None
        args_json = _attr(span, ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON)
        assert args_json is not None
        json.loads(args_json)  # must be valid JSON
        assert _attr(span, ToolCallAttributes.TOOL_CALL_ID), "TOOL span missing tool_call.id"
        assert _attr(span, SpanAttributes.TOOL_ID), "TOOL span missing tool.id"
        # output.value present once the tool/method returns.
        assert _attr(span, SpanAttributes.OUTPUT_VALUE) is not None


@pytest.mark.asyncio
async def test_context_snapshot_is_chain_when_present(framework_spans):
    """``context_snapshot`` spans (emitted when a system message is built) are CHAIN."""
    by_name = await _run_agent(framework_spans)
    spans = by_name.get("context_snapshot")
    if not spans:
        pytest.skip("no context_snapshot span emitted")
    for span in spans:
        assert _attr(span, SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
        # input.value mirrors the system message, text/plain.
        assert _attr(span, SpanAttributes.INPUT_VALUE) is not None
        assert _attr(span, SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.TEXT.value
        # native attr preserved alongside the alias.
        assert _attr(span, "nooa.system_message") is not None


@pytest.mark.asyncio
async def test_framework_span_names_are_stable(framework_spans):
    """The trace_explorer parses turns by span **name** (not kind) — guard the names
    so a rename is caught here."""
    by_name = await _run_agent(framework_spans)
    assert "method.run" in by_name
    assert "generation" in by_name
    assert "code_execution" in by_name


@pytest.mark.asyncio
async def test_session_id_present(framework_spans):
    """``session.id`` (semconv SESSION_ID) is stamped on spans."""
    by_name = await _run_agent(framework_spans)
    run_span = by_name["method.run"][0]
    assert _attr(run_span, SpanAttributes.SESSION_ID) == "conformance-session"


# ---------------------------------------------------------------------------
# LLM-span conformance (delegated to litellm instrumentor + our patch).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_span_conformance():
    """A real ``litellm.acompletion`` produces a spec-conformant ``LLM`` span."""
    pytest.importorskip(
        "openinference.instrumentation.litellm",
        reason="openinference-instrumentation-litellm required for LLM spans",
    )
    import litellm

    from nooa.tracing import enable_tracing, exporters, flush_traces, set_session

    with tempfile.TemporaryDirectory() as tmpdir:
        enable_tracing(exporters=[exporters.jsonl(tmpdir)])
        set_session("conformance-llm")

        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "what is 2+2?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_abc123",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tc_abc123", "content": "4"},
            ],
            mock_response="The answer is 4.",
        )
        flush_traces()

        spans = read_all_otlp_jsonl_spans(tmpdir)
        assert spans, f"no spans written to {tmpdir}"
        llm_spans = [
            s
            for s in spans
            if s["attributes"].get(SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        ]
        assert llm_spans, f"no LLM span; names={[s.get('name') for s in spans]}"
        span = llm_spans[0]
        attrs = span["attributes"]

        # Span name (catches an upstream litellm rename).
        assert span.get("name") == "acompletion", (
            f"expected litellm span name 'acompletion', got {span.get('name')!r}"
        )

        # Core LLM attributes.
        assert isinstance(attrs.get(SpanAttributes.LLM_MODEL_NAME), str)

        # Token counts are integers when present.
        for tok in (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
        ):
            if tok in attrs:
                assert isinstance(attrs[tok], int), f"{tok} must be int, got {attrs[tok]!r}"

        # Cost is stamped (llm.cost.*) — the litellm instrumentor omits it, our
        # patch adds it from litellm's computed cost / gateway headers. gpt-3.5-turbo
        # has known pricing so a positive total is expected here.
        total_cost = attrs.get(SpanAttributes.LLM_COST_TOTAL)
        assert isinstance(total_cost, (int, float)) and total_cost > 0, (
            f"LLM span missing positive llm.cost.total; got {total_cost!r}"
        )

        # Input/output messages present.
        assert any(k.startswith(SpanAttributes.LLM_INPUT_MESSAGES) for k in attrs), (
            "LLM span missing llm.input_messages.*"
        )
        assert any(k.startswith(SpanAttributes.LLM_OUTPUT_MESSAGES) for k in attrs), (
            "LLM span missing llm.output_messages.*"
        )

        # input.value / output.value present.
        assert attrs.get(SpanAttributes.INPUT_VALUE) is not None
        assert attrs.get(SpanAttributes.OUTPUT_VALUE) is not None

        # The tool-call message carries name + arguments + id. Assert the actual
        # dotted semconv keys are present on the span (not just a substring of the
        # serialized blob — ``...function.name``'s last segment "name" appears in
        # unrelated keys like ``llm.model_name`` and would pass vacuously).
        suffix = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME  # "tool_call.function.name"
        assert any(suffix in k for k in attrs), (
            f"LLM span missing a {suffix} message attribute; keys: {sorted(attrs)}"
        )
        # tool_call.id is the patch's contribution — assert it round-tripped.
        flat = json.dumps(attrs)
        assert "tc_abc123" in flat, "tool_call.id not captured on the LLM span (patch regression)"
        # role/content message attributes use the semconv names.
        assert any(k.endswith(MessageAttributes.MESSAGE_ROLE) for k in attrs)
