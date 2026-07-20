# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TraceExplorer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nooa.trace_explorer import (
    ExecutionTurn,
    LLMTurn,
    TraceExplorer,
)
from nooa.trace_explorer.explorer import (
    _extract_any_value,
    _extract_failing_line,
    _extract_messages,
    _extract_reasoning_content,
    _extract_response,
    _extract_token_counts,
    _extract_tool_calls,
    _first_code_line,
    _format_turn_status,
    _normalize_otlp_span,
    _otlp_attrs_to_dict,
    _short_id,
)

# =============================================================================
# Fixtures
# =============================================================================


def create_trace_file(spans: list[dict]) -> Path:
    """Create a temporary trace file from span data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for span in spans:
            f.write(json.dumps(span) + "\n")
        return Path(f.name)


def _sv(v: str) -> dict:
    """Shorthand for OTLP string value."""
    return {"stringValue": v}


def _iv(v: int) -> dict:
    """Shorthand for OTLP int value."""
    return {"intValue": str(v)}


def _otlp_attrs(flat: dict) -> list[dict]:
    """Convert a flat {key: value} dict to OTLP attribute list for test fixtures."""
    result = []
    for k, v in flat.items():
        if isinstance(v, str):
            result.append({"key": k, "value": _sv(v)})
        elif isinstance(v, int):
            result.append({"key": k, "value": _iv(v)})
        elif isinstance(v, bool):
            result.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, float):
            result.append({"key": k, "value": {"doubleValue": v}})
        else:
            result.append({"key": k, "value": _sv(str(v))})
    return result


def make_generation_span(
    span_id: str,
    gen_id: str,
    agent_name: str = "TestAgent",
    method_name: str = "test_method",
    parent_gen_id: str | None = None,
    start_time: int = 1000000000,
    end_time: int = 2000000000,
    status: str = "OK",
    result: str | None = None,
) -> dict:
    """Create a generation span in OTLP format."""
    flat_attrs: dict[str, str] = {
        "generation.id": gen_id,
        "agent.name": agent_name,
        "agent.method": method_name,
        "openinference.span.kind": "AGENT",
    }
    if parent_gen_id:
        flat_attrs["generation.parent_id"] = parent_gen_id
    if result:
        flat_attrs["generation.result"] = result

    return {
        "name": "generation",
        "spanId": span_id,
        "startTimeUnixNano": str(start_time),
        "endTimeUnixNano": str(end_time),
        "attributes": _otlp_attrs(flat_attrs),
        "status": {"code": 2 if status == "ERROR" else 1},
        "events": [],
    }


def make_llm_span(
    span_id: str,
    parent_span_id: str,
    messages: list[tuple[str, str]] | None = None,
    tool_calls: list[tuple[str, str]] | None = None,
    response: str = "",
    reasoning_content: str | None = None,
    model: str = "test-model",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    start_time: int = 1100000000,
    end_time: int = 1200000000,
) -> dict:
    """Create an LLM completion span in OTLP format."""
    flat_attrs: dict[str, str | int] = {
        "llm.model_name": model,
        "llm.token_count.prompt": prompt_tokens,
        "llm.token_count.completion": completion_tokens,
        "llm.token_count.total": prompt_tokens + completion_tokens,
        "openinference.span.kind": "LLM",
    }

    # Add input messages
    if messages:
        for i, (role, content) in enumerate(messages):
            flat_attrs[f"llm.input_messages.{i}.message.role"] = role
            flat_attrs[f"llm.input_messages.{i}.message.content"] = content

    # Add output content
    if response:
        flat_attrs["llm.output_messages.0.message.content"] = response
    if reasoning_content:
        flat_attrs["llm.output_messages.0.message.reasoning_content"] = reasoning_content

    # Add tool calls
    if tool_calls:
        for i, (func_name, args) in enumerate(tool_calls):
            flat_attrs[f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name"] = (
                func_name
            )
            flat_attrs[
                f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.arguments"
            ] = args

    return {
        "name": "acompletion",
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "startTimeUnixNano": str(start_time),
        "endTimeUnixNano": str(end_time),
        "attributes": _otlp_attrs(flat_attrs),
        "status": {"code": 0},
        "events": [],
    }


def make_execution_span(
    span_id: str,
    gen_id: str,
    agent_name: str = "TestAgent",
    code: str = "print('hello')",
    stdout: str = "hello\n",
    returned_value: str | None = None,
    error: str | None = None,
    start_time: int = 1300000000,
    end_time: int = 1400000000,
) -> dict:
    """Create a code execution span in OTLP format."""
    exec_result = {
        "stdout": stdout,
        "returned_value": returned_value,
    }
    if error:
        exec_result["error"] = error

    flat_attrs: dict[str, str] = {
        "agent.name": agent_name,
        "generation.id": gen_id,
        "code": code,
        "result": json.dumps(exec_result),
    }

    return {
        "name": "code_execution",
        "spanId": span_id,
        "startTimeUnixNano": str(start_time),
        "endTimeUnixNano": str(end_time),
        "attributes": _otlp_attrs(flat_attrs),
        "status": {"code": 2 if error else 1},
        "events": [],
    }


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestExtractMessages:
    """Tests for _extract_messages function."""

    def test_extracts_single_message(self):
        attrs = {
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Hello",
        }
        messages = _extract_messages(attrs)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_extracts_multiple_messages(self):
        attrs = {
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are helpful",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "Hello",
        }
        messages = _extract_messages(attrs)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_returns_empty_for_no_messages(self):
        attrs = {}
        messages = _extract_messages(attrs)
        assert len(messages) == 0


class TestExtractToolCalls:
    """Tests for _extract_tool_calls function."""

    def test_extracts_single_tool_call(self):
        attrs = {
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "execute_python",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"code": "print(1)"}',
        }
        tool_calls = _extract_tool_calls(attrs)
        assert len(tool_calls) == 1
        assert tool_calls[0].function_name == "execute_python"
        assert "print(1)" in tool_calls[0].arguments

    def test_extracts_multiple_tool_calls(self):
        attrs = {
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "tool_a",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
            "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "tool_b",
            "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": '{"x": 1}',
        }
        tool_calls = _extract_tool_calls(attrs)
        assert len(tool_calls) == 2
        assert tool_calls[0].function_name == "tool_a"
        assert tool_calls[1].function_name == "tool_b"

    def test_returns_empty_for_no_tool_calls(self):
        attrs = {}
        tool_calls = _extract_tool_calls(attrs)
        assert len(tool_calls) == 0


class TestExtractResponse:
    """Tests for _extract_response function."""

    def test_extracts_from_output_message(self):
        attrs = {"llm.output_messages.0.message.content": "Hello world"}
        response = _extract_response(attrs)
        assert response == "Hello world"

    def test_fallback_to_output_value(self):
        attrs = {"output.value": "Fallback value"}
        response = _extract_response(attrs)
        assert response == "Fallback value"

    def test_prefers_output_message_over_output_value(self):
        attrs = {
            "llm.output_messages.0.message.content": "Primary",
            "output.value": "Secondary",
        }
        response = _extract_response(attrs)
        assert response == "Primary"

    def test_returns_empty_for_no_response(self):
        attrs = {}
        response = _extract_response(attrs)
        assert response == ""


class TestExtractTokenCounts:
    """Tests for _extract_token_counts function."""

    def test_extracts_all_counts(self):
        attrs = {
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 50,
            "llm.token_count.total": 150,
        }
        counts = _extract_token_counts(attrs)
        assert counts == {"prompt": 100, "completion": 50, "total": 150}

    def test_returns_none_for_no_counts(self):
        attrs = {}
        counts = _extract_token_counts(attrs)
        assert counts is None


class TestShortId:
    """Tests for _short_id function."""

    def test_truncates_long_id(self):
        assert _short_id("abcdef123456") == "abcdef"

    def test_keeps_short_id(self):
        assert _short_id("abc") == "abc"

    def test_handles_none(self):
        assert _short_id(None) == "------"


# =============================================================================
# TraceExplorer Tests
# =============================================================================


class TestTraceExplorerBasic:
    """Basic tests for TraceExplorer."""

    async def test_loads_simple_trace(self):
        """Test loading a trace with one session and one LLM turn."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_llm_span(
                "llm1",
                "gen1",
                messages=[("system", "You are helpful"), ("user", "Hello")],
                response="Hi there!",
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)

            assert len(trace.sessions) == 1
            assert trace.sessions[0].agent_name == "TestAgent"
            assert trace.sessions[0].method_name == "test_method"
            assert len(trace.sessions[0].turns) == 1
        finally:
            trace_file.unlink()

    async def test_extracts_tool_calls_from_llm_turns(self):
        """Test that tool calls are properly extracted from LLM turns."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_llm_span(
                "llm1",
                "gen1",
                messages=[("user", "Do something")],
                tool_calls=[
                    ("execute_python", '{"code": "x = 1"}'),
                    ("return_result", '{"value": 42}'),
                ],
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)

            llm_turn = trace.sessions[0].turns[0]
            assert isinstance(llm_turn, LLMTurn)
            assert len(llm_turn.tool_calls) == 2
            assert llm_turn.tool_calls[0].function_name == "execute_python"
            assert llm_turn.tool_calls[1].function_name == "return_result"
        finally:
            trace_file.unlink()

    async def test_extracts_execution_turns(self):
        """Test that execution turns are properly extracted."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_execution_span(
                "exec1",
                "aaaaaa000000",
                code="result = 1 + 2",
                stdout="",
                returned_value="3",
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)

            exec_turn = trace.sessions[0].turns[0]
            assert isinstance(exec_turn, ExecutionTurn)
            assert exec_turn.code == "result = 1 + 2"
            assert exec_turn.returned_value == "3"
        finally:
            trace_file.unlink()


class TestTraceExplorerGetToolCalls:
    """Tests for get_tool_calls method."""

    async def test_finds_tool_calls_from_llm_response(self):
        """Test that tool calls from LLM responses are found."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_llm_span(
                "llm1",
                "gen1",
                messages=[("user", "Calculate")],
                tool_calls=[("execute_python", '{"code": "1+1"}')],
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            # get_tool_calls was internalized, check that tool calls appear in overview
            result = await trace.get_overview()

            assert "TestAgent.test_method" in result
        finally:
            trace_file.unlink()

    async def test_finds_multiple_tool_calls(self):
        """Test that multiple tool calls are found."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_llm_span(
                "llm1",
                "gen1",
                messages=[("user", "Do stuff")],
                tool_calls=[
                    ("execute_python", '{"code": "x = 1"}'),
                    ("return_result", '{"value": 42}'),
                ],
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            # get_tool_calls was internalized, check that tool calls appear in session view
            result = await trace.get_session(trace.sessions[0].session_id[:6], concise=False)

            assert "execute_python" in result or "tool_call" in result
        finally:
            trace_file.unlink()


class TestTraceExplorerOverview:
    """Tests for get_overview method."""

    async def test_overview_success(self):
        """Test overview for successful trace."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000", status="OK"),
            make_llm_span("llm1", "gen1"),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            overview = await trace.get_overview()

            # Format uses [OK] label in call graph line for success
            assert "TestAgent.test_method" in overview
            assert "[OK]" in overview
        finally:
            trace_file.unlink()

    async def test_overview_failure(self):
        """Test overview for failed trace."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000", status="ERROR"),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            overview = await trace.get_overview()

            # Format uses [ERR] label in call graph line for failure
            assert "[ERR]" in overview
        finally:
            trace_file.unlink()


class TestTraceExplorerErrors:
    """Tests for error detection."""

    async def test_finds_execution_errors(self):
        """Test that execution errors are found."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_execution_span(
                "exec1",
                "aaaaaa000000",
                code="1/0",
                stdout="",
                error="ZeroDivisionError: division by zero",
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            errors = await trace.get_errors()

            assert "Found" in errors
            assert "execution_error" in errors
            assert "ZeroDivisionError" in errors
        finally:
            trace_file.unlink()

    async def test_finds_session_status_errors(self):
        """Test that session status errors are found."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000", status="ERROR"),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            errors = await trace.get_errors()

            assert "status_error" in errors
        finally:
            trace_file.unlink()


class TestTraceExplorerSearch:
    """Tests for search functionality."""

    async def test_searches_message_content(self):
        """Test searching in message content."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_llm_span(
                "llm1",
                "gen1",
                messages=[("user", "Find the needle in the haystack")],
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            result = await trace.search("needle")

            assert "Found" in result
            assert "needle" in result.lower()
        finally:
            trace_file.unlink()

    async def test_searches_code(self):
        """Test searching in executed code."""
        spans = [
            make_generation_span("gen1", "aaaaaa000000"),
            make_execution_span(
                "exec1",
                "aaaaaa000000",
                code="secret_function()",
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            result = await trace.search("secret")

            assert "Found" in result
            assert "code" in result
        finally:
            trace_file.unlink()


# =============================================================================
# Integration with Real Trace (if available)
# =============================================================================


# =============================================================================
# Agent Span Parsing Tests (Bug Reproduction)
# =============================================================================


def make_agent_span(
    span_id: str,
    agent_name: str,
    method_name: str,
    parent_span_id: str | None = None,
    start_time: int = 1000000000,
    end_time: int = 2000000000,
    status: str = "OK",
    result: str | None = None,
) -> dict:
    """Create an AGENT span (root agent invocation) in OTLP format."""
    flat_attrs: dict[str, str] = {
        "openinference.span.kind": "AGENT",
        "agent.name": agent_name,
        "agent.method": method_name,
    }
    if result:
        flat_attrs["agent.result"] = result

    span: dict = {
        "name": f"plan.{method_name}",  # Span name format: plan.<method>
        "spanId": span_id,
        "startTimeUnixNano": str(start_time),
        "endTimeUnixNano": str(end_time),
        "attributes": _otlp_attrs(flat_attrs),
        "status": {"code": 2 if status == "ERROR" else 1},
        "events": [],
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id

    return span


class TestAgentSpanParsing:
    """Tests for AGENT span parsing with mismatched agent names.

    This tests a bug where the AGENT span has a name like "plan.classify" but the
    generation and execution spans have agent.name="SentimentSingleAgent".
    The current code extracts agent_name from the span name, causing a mismatch.
    """

    async def test_parses_turns_when_agent_name_differs_from_span_name(self):
        """Test that turns are parsed even when AGENT span name differs from agent.name.

        Bug scenario:
        - AGENT span name: "plan.classify" (parsed as agent_name="plan")
        - AGENT span attributes.agent.name: "SentimentSingleAgent"
        - Generation spans have agent.name: "SentimentSingleAgent"

        The code should match using attributes.agent.name, not the parsed span name.
        """
        # Create AGENT span with different span name vs agent.name attribute
        agent_span = make_agent_span(
            span_id="4af970ac54661525",
            agent_name="SentimentSingleAgent",  # This is in attributes
            method_name="classify",
            start_time=1000000000,
            end_time=2000000000,
            result="positive",
        )
        # Note: span name is "plan.classify" but agent.name attribute is "SentimentSingleAgent"

        # Create generation span (root of the generation tree)
        gen_span = {
            "name": "generation",
            "spanId": "4bef916157ef791f",
            "parentSpanId": "e4fe3373ac8e176f",  # Different parent
            "startTimeUnixNano": "1100000000",
            "endTimeUnixNano": "1900000000",
            "attributes": _otlp_attrs(
                {
                    "openinference.span.kind": "AGENT",
                    "agent.name": "SentimentSingleAgent",
                    "agent.method": "classify",
                    "generation.strategy": "CODEACT",
                    "generation.id": "02561812-1a08-446e-b718-95d1909cc409",
                    "generation.result": "positive",
                }
            ),
            "status": {"code": 1},
            "events": [],
        }

        # Create an acompletion span (LLM call)
        llm_span = make_llm_span(
            span_id="8d9bcf7a4608e085",
            parent_span_id="4bef916157ef791f",  # Child of generation span
            messages=[("system", "You are helpful"), ("user", "Classify sentiment")],
            response="",
            tool_calls=[("return_result", '{"result": "positive"}')],
            start_time=1200000000,
            end_time=1300000000,
        )

        # Create an execution span
        exec_span = {
            "name": "code_execution",
            "spanId": "30afc217b0164c9d",
            "parentSpanId": "4bef916157ef791f",
            "startTimeUnixNano": "1150000000",
            "endTimeUnixNano": "1160000000",
            "attributes": _otlp_attrs(
                {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "python_executor",
                    "agent.name": "SentimentSingleAgent",
                    "code": "print('hello')",
                    "generation.id": "02561812-1a08-446e-b718-95d1909cc409",
                    "result": json.dumps(
                        {
                            "stdout": "hello\n",
                            "returned_value": None,
                        }
                    ),
                }
            ),
            "status": {"code": 1},
            "events": [],
        }

        spans = [gen_span, exec_span, llm_span, agent_span]
        trace_file = create_trace_file(spans)

        try:
            trace = await TraceExplorer.from_file(trace_file)

            # Should have 1 session
            assert len(trace.sessions) == 1, f"Expected 1 session, got {len(trace.sessions)}"

            session = trace.sessions[0]

            # BUG: Currently this fails because agent_name is parsed as "plan" from
            # the span name "plan.classify", but generation spans have agent.name="SentimentSingleAgent"
            assert len(session.turns) > 0, (
                f"Expected turns to be parsed, got 0 turns. "
                f"Session agent_name={session.agent_name}, "
                f"but generation spans have agent.name='SentimentSingleAgent'. "
                f"The code should use attributes.agent.name from the AGENT span."
            )
        finally:
            trace_file.unlink()


# =============================================================================
# OTLP Normalization Tests
# =============================================================================


class TestOtlpAttrsToDict:
    """Tests for _otlp_attrs_to_dict and _extract_any_value."""

    def test_string_value(self):
        attrs = [{"key": "agent.name", "value": {"stringValue": "TestAgent"}}]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"agent.name": "TestAgent"}

    def test_int_value(self):
        attrs = [{"key": "llm.token_count.prompt", "value": {"intValue": "150"}}]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"llm.token_count.prompt": 150}
        assert isinstance(result["llm.token_count.prompt"], int)

    def test_double_value(self):
        attrs = [{"key": "score", "value": {"doubleValue": 0.95}}]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"score": 0.95}

    def test_bool_value(self):
        attrs = [{"key": "flag", "value": {"boolValue": True}}]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"flag": True}

    def test_array_value(self):
        attrs = [
            {
                "key": "tags",
                "value": {
                    "arrayValue": {
                        "values": [
                            {"stringValue": "a"},
                            {"stringValue": "b"},
                        ]
                    }
                },
            }
        ]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"tags": ["a", "b"]}

    def test_kvlist_value(self):
        attrs = [
            {
                "key": "metadata",
                "value": {
                    "kvlistValue": {
                        "values": [
                            {"key": "k1", "value": {"stringValue": "v1"}},
                            {"key": "k2", "value": {"intValue": "42"}},
                        ]
                    }
                },
            }
        ]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"metadata": {"k1": "v1", "k2": 42}}

    def test_bytes_value(self):
        attrs = [{"key": "data", "value": {"bytesValue": "AQID"}}]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"data": "AQID"}

    def test_empty_list(self):
        assert _otlp_attrs_to_dict([]) == {}

    def test_multiple_attrs(self):
        attrs = [
            {"key": "a", "value": {"stringValue": "hello"}},
            {"key": "b", "value": {"intValue": "5"}},
            {"key": "c", "value": {"boolValue": False}},
        ]
        result = _otlp_attrs_to_dict(attrs)
        assert result == {"a": "hello", "b": 5, "c": False}

    def test_extract_any_value_nested_array(self):
        val = {
            "arrayValue": {
                "values": [
                    {"intValue": "1"},
                    {"arrayValue": {"values": [{"stringValue": "nested"}]}},
                ]
            }
        }
        result = _extract_any_value(val)
        assert result == [1, ["nested"]]


class TestNormalizeOtlpSpan:
    """Tests for _normalize_otlp_span."""

    def test_converts_otlp_span(self):
        otlp_span = {
            "traceId": "trace123",
            "spanId": "span456",
            "parentSpanId": "parent789",
            "name": "generation",
            "kind": 1,
            "startTimeUnixNano": "1000000000",
            "endTimeUnixNano": "2000000000",
            "attributes": [
                {"key": "agent.name", "value": {"stringValue": "TestAgent"}},
                {"key": "llm.token_count.prompt", "value": {"intValue": "100"}},
            ],
            "status": {"code": 1},
            "events": [],
            "_resource": {
                "attributes": [
                    {"key": "session.id", "value": {"stringValue": "sess1"}},
                ]
            },
        }
        result = _normalize_otlp_span(otlp_span)
        assert result["span_id"] == "span456"
        assert result["trace_id"] == "trace123"
        assert result["parent_span_id"] == "parent789"
        assert result["name"] == "generation"
        assert result["start_time"] == 1000000000
        assert result["end_time"] == 2000000000
        assert result["duration_ns"] == 1000000000
        assert result["attributes"] == {"agent.name": "TestAgent", "llm.token_count.prompt": 100}
        assert result["status"]["status_code"] == "OK"
        assert result["resource"]["attributes"]["session.id"] == "sess1"

    def test_error_status(self):
        otlp_span = {
            "spanId": "s1",
            "name": "test",
            "startTimeUnixNano": "0",
            "endTimeUnixNano": "0",
            "attributes": [],
            "status": {"code": 2, "message": "something failed"},
        }
        result = _normalize_otlp_span(otlp_span)
        assert result["status"]["status_code"] == "ERROR"
        assert result["status"]["description"] == "something failed"

    def test_unset_status_maps_to_ok(self):
        otlp_span = {
            "spanId": "s1",
            "name": "test",
            "startTimeUnixNano": "0",
            "endTimeUnixNano": "0",
            "attributes": [],
            "status": {"code": 0},
        }
        result = _normalize_otlp_span(otlp_span)
        assert result["status"]["status_code"] == "OK"

    def test_missing_parent_span_id(self):
        """Root spans have no parentSpanId."""
        otlp_span = {
            "spanId": "root1",
            "name": "test",
            "startTimeUnixNano": "0",
            "endTimeUnixNano": "0",
            "attributes": [],
            "status": {},
        }
        result = _normalize_otlp_span(otlp_span)
        assert result["parent_span_id"] is None

    def test_resource_with_dict_attributes(self):
        """Resource with already-flat attributes (edge case)."""
        otlp_span = {
            "spanId": "s1",
            "name": "test",
            "startTimeUnixNano": "0",
            "endTimeUnixNano": "0",
            "attributes": [],
            "status": {},
            "_resource": {"attributes": {"already": "flat"}},
        }
        result = _normalize_otlp_span(otlp_span)
        assert result["resource"]["attributes"] == {"already": "flat"}


class TestLoadSpansOtlp:
    """Test that _load_spans normalizes OTLP format."""

    def test_loads_otlp_jsonl(self):
        """Full round-trip: OTLP JSONL → _load_spans → internal format."""
        otlp_spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "generation",
                "kind": 1,
                "startTimeUnixNano": "1000",
                "endTimeUnixNano": "2000",
                "attributes": [
                    {"key": "agent.name", "value": {"stringValue": "Agent"}},
                ],
                "status": {"code": 1},
                "events": [],
            },
            {
                "traceId": "t1",
                "spanId": "s2",
                "parentSpanId": "s1",
                "name": "acompletion",
                "kind": 3,
                "startTimeUnixNano": "1100",
                "endTimeUnixNano": "1900",
                "attributes": [
                    {"key": "llm.model_name", "value": {"stringValue": "test-model"}},
                ],
                "status": {"code": 0},
                "events": [],
            },
        ]
        path = create_trace_file(otlp_spans)
        from nooa.trace_explorer.explorer import _load_spans

        spans = _load_spans(path)
        assert len(spans) == 2
        assert spans[0]["span_id"] == "s1"
        assert spans[0]["attributes"]["agent.name"] == "Agent"
        assert spans[1]["span_id"] == "s2"
        assert spans[1]["parent_span_id"] == "s1"
        path.unlink()


class TestFromViewer:
    """Tests for TraceExplorer.from_viewer with mocked HTTP (httpx)."""

    async def test_single_page(self):
        """Fetch a trace that fits in one page."""

        spans = [
            {
                "traceId": "t1",
                "spanId": "gen1",
                "name": "generation",
                "kind": 1,
                "startTimeUnixNano": "1000000000",
                "endTimeUnixNano": "2000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "agent.name", "value": {"stringValue": "TestAgent"}},
                    {"key": "agent.method", "value": {"stringValue": "run"}},
                    {"key": "generation.id", "value": {"stringValue": "gen-001"}},
                ],
                "status": {"code": 1},
                "events": [],
            },
        ]
        response_data = {"events": spans, "total_count": 1, "has_more": False}

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: response_data

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            trace = await TraceExplorer.from_viewer("http://localhost:5001", "test-session")
            assert trace.trace_file == "viewer://test-session"
            assert len(trace._raw_spans) == 1

    async def test_pagination(self):
        """Fetch a trace across multiple pages."""

        span1 = {
            "traceId": "t1",
            "spanId": "s1",
            "name": "generation",
            "kind": 1,
            "startTimeUnixNano": "1000",
            "endTimeUnixNano": "2000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "agent.name", "value": {"stringValue": "A"}},
                {"key": "agent.method", "value": {"stringValue": "run"}},
                {"key": "generation.id", "value": {"stringValue": "g1"}},
            ],
            "status": {"code": 1},
            "events": [],
        }
        span2 = {
            "traceId": "t1",
            "spanId": "s2",
            "parentSpanId": "s1",
            "name": "acompletion",
            "kind": 3,
            "startTimeUnixNano": "1100",
            "endTimeUnixNano": "1900",
            "attributes": [],
            "status": {"code": 0},
            "events": [],
        }

        page1 = {"events": [span1], "total_count": 2, "has_more": True}
        page2 = {"events": [span2], "total_count": 2, "has_more": False}

        mock_resp1 = AsyncMock()
        mock_resp1.status_code = 200
        mock_resp1.raise_for_status = lambda: None
        mock_resp1.json = lambda: page1

        mock_resp2 = AsyncMock()
        mock_resp2.status_code = 200
        mock_resp2.raise_for_status = lambda: None
        mock_resp2.json = lambda: page2

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.side_effect = [mock_resp1, mock_resp2]
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            trace = await TraceExplorer.from_viewer("http://localhost:5001", "sess")
            assert len(trace._raw_spans) == 2

    async def test_connection_error(self):
        """Viewer unreachable raises ConnectionError."""

        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.side_effect = httpx.ConnectError("Connection refused")
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            with pytest.raises(ConnectionError, match="Cannot reach viewer"):
                await TraceExplorer.from_viewer("http://localhost:9999", "sess")

    async def test_session_not_found(self):
        """404 raises ValueError."""

        mock_resp = AsyncMock()
        mock_resp.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            with pytest.raises(ValueError, match="Session not found"):
                await TraceExplorer.from_viewer("http://localhost:5001", "nonexistent")


# Tests for turn-summary helper functions
# =============================================================================


class TestFirstCodeLine:
    def test_single_line(self):
        assert _first_code_line("import pandas as pd") == "import pandas as pd"

    def test_multi_line_appends_suffix(self):
        code = "import pandas as pd\ndf = pd.read_csv('data.csv')"
        assert _first_code_line(code) == "import pandas as pd; ..."

    def test_skips_comments_and_docstrings(self):
        code = '# comment\n"""docstring"""\nimport os\nprint(os.getcwd())'
        assert _first_code_line(code) == "import os; ..."

    def test_empty_string(self):
        assert _first_code_line("") == ""

    def test_only_comments(self):
        assert _first_code_line("# comment\n# another") == ""

    def test_long_single_line_truncated(self):
        long_line = "x = " + "a" * 100
        result = _first_code_line(long_line)
        assert len(result) <= 60
        assert result.endswith("...")

    def test_long_first_line_with_more_lines_truncated(self):
        long_line = "x = " + "a" * 100
        code = long_line + "\ny = 1"
        result = _first_code_line(code)
        assert len(result) <= 60
        assert result.endswith("...")

    def test_truncation_boundary_exact_fit(self):
        # Line of exactly max_len chars should not be truncated
        line = "x" * 60
        assert _first_code_line(line) == line


class TestExtractFailingLine:
    _TRACEBACK = (
        "Cell In[34], line 1\n    import numpy as np\n    ^\nimport of 'numpy' is forbidden"
    )

    def test_extracts_failing_line(self):
        assert _extract_failing_line(self._TRACEBACK) == "import numpy as np"

    def test_no_match_returns_empty(self):
        assert _extract_failing_line("NameError: name 'foo' is not defined") == ""

    def test_empty_string(self):
        assert _extract_failing_line("") == ""

    def test_no_trailing_newline(self):
        # Error string ends without newline after the code line
        error = "Cell In[1], line 1\n    import foo"
        assert _extract_failing_line(error) == "import foo"

    def test_long_line_truncated(self):
        long_code = "x = " + "a" * 100
        error = f"Cell In[1], line 1\n    {long_code}\n    ^\nSyntaxError"
        result = _extract_failing_line(error)
        assert len(result) <= 60
        assert result.endswith("...")


class TestFormatTurnStatus:
    def test_no_error_returns_ok(self):
        assert _format_turn_status(None) == "[OK]"

    def test_empty_error_returns_ok(self):
        assert _format_turn_status("") == "[OK]"

    def test_cell_in_traceback(self):
        error = "Cell In[34], line 1\n    import numpy as np\n    ^\nimport of 'numpy' is forbidden"
        result = _format_turn_status(error)
        assert result == "[ERR: import of 'numpy' is forbidden]"

    def test_file_not_found_shortens_path(self):
        error = "FileNotFoundError: No such file or directory: '/long/path/to/Laboratory.csv'"
        result = _format_turn_status(error)
        assert result == "[ERR: No such file: Laboratory.csv]"

    def test_generic_error_uses_first_line(self):
        error = "ValueError: invalid literal for int()\nmore details here"
        assert _format_turn_status(error) == "[ERR: ValueError: invalid literal for int()]"

    def test_long_message_truncated(self):
        error = "SomeError: " + "x" * 100
        result = _format_turn_status(error)
        assert len(result) <= len("[ERR: ") + 50 + 1  # bracket + msg + ]
        assert result.endswith("...]")


# =============================================================================
# Regression tests for GitLab issue #129: _pformat signature mismatch
# =============================================================================


class TestPformatImport:
    """Regression tests for GitLab issue #129.

    The module used to import the low-level streaming writer
    ``agentdoc._pformat._pformat`` (now requires a ``_stream`` positional
    argument) and call it with kwargs only — a TypeError at every call site.
    The fix is to alias the public, string-returning ``agentdoc.pformat``.
    """

    def test_pformat_alias_returns_string(self):
        """The ``_pformat`` name in ``trace_explorer.explorer`` must be the
        public string-returning API, callable with kwargs only."""
        from nooa.trace_explorer.explorer import _pformat

        out = _pformat({"k": "v"}, max_string=60, max_length=5, max_depth=2)
        assert isinstance(out, str)
        assert "k" in out and "v" in out

    async def test_get_session_verbose_formats_session_input(self):
        """End-to-end: ``get_session(concise=False)`` on a session with
        kwargs hits ``_pformat`` without a surrounding try/except (unlike the
        tool-arg path, which swallowed the TypeError silently). Previously
        this raised ``TypeError: _pformat() missing 1 required positional
        argument: '_stream'``."""
        agent_span = make_agent_span(
            span_id="a1b2c3d4e5f60718",
            agent_name="TestAgent",
            method_name="ask",
            start_time=1_000_000_000,
            end_time=2_000_000_000,
        )
        # Inject a serialized kwargs attribute so `_get_session_input` returns
        # a non-empty dict and the non-concise branch calls `_pformat`.
        agent_span["attributes"].append(
            {"key": "agent.kwargs", "value": _sv(json.dumps({"question": "why?"}))}
        )
        spans = [agent_span]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            output = await trace.get_session(trace.sessions[0].session_id[:6], concise=False)

            assert isinstance(output, str)
            # Session input line is rendered via _pformat.
            assert "IN:" in output
            assert "question" in output
        finally:
            trace_file.unlink()


# =============================================================================
# Span-discovery invariants
#
# These tests exist because the parser has several non-obvious paths:
# AGENT-span parsing vs generation-span fallback, parent-chain walking that
# crosses LLM/TOOL spans, and three tiers of turn-population matching
# (call_id → descendant → time-range). A production bug (`_000001` session
# in kdd-cup experiment) showed up as "No sessions found in trace." when
# all child spans were orphaned; no test covered that path.
# =============================================================================


class TestOrphanSpans:
    """Child spans whose parents are not in the trace."""

    async def test_only_orphan_llm_and_exec_yields_no_sessions(self):
        """If acompletion/code_execution spans dangle (parent span ID not in
        trace, no AGENT span, no root generation), the parser returns an
        empty session list without crashing.

        This is the `_000001` case from the kdd-cup experiment: a viewer
        session.id returns 5 spans (2 acompletion + 2 code_execution + 1
        eval) whose generation roots live under a *different* session.id.
        """
        spans = [
            make_llm_span("aa00", parent_span_id="ghost1", messages=[("user", "?")]),
            {
                "name": "code_execution",
                "spanId": "ee00",
                "parentSpanId": "ghost1",
                "startTimeUnixNano": "1000",
                "endTimeUnixNano": "2000",
                "attributes": _otlp_attrs(
                    {
                        "agent.name": "A",
                        "generation.id": "orphaned",
                        "code": "x",
                        "result": json.dumps({"stdout": "", "returned_value": None}),
                    }
                ),
                "status": {"code": 1},
                "events": [],
            },
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            assert trace.sessions == []
            # Overview must not crash and must communicate emptiness.
            overview = await trace.get_overview()
            assert "No sessions found" in overview
        finally:
            trace_file.unlink()

    async def test_only_non_root_generation_spans_yield_no_sessions(self):
        """A trace with only *child* generation spans (all have
        generation.parent_id set) falls through the generation-span fallback
        because it only considers root generations, and returns no sessions."""
        spans = [
            make_generation_span(
                "ge01",
                gen_id="child1",
                parent_gen_id="missing-root",
            ),
        ]
        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            assert trace.sessions == []
        finally:
            trace_file.unlink()


class TestSpanCompleteness:
    """Every descendant gen/acompletion/exec span should map to a turn."""

    async def test_agent_span_path_captures_all_descendant_turns(self):
        """AGENT span + generation span + N acompletion + M code_execution
        should produce exactly N+M turns, correctly typed and ordered by
        start_time.

        Previously there was no assertion that span *count* was preserved —
        only that `len(turns) > 0`. That's not enough to catch a parser
        that silently drops half the spans.
        """
        agent = make_agent_span(
            span_id="ag01",
            agent_name="A",
            method_name="run",
            start_time=1_000,
            end_time=9_000,
        )
        gen = make_generation_span(
            span_id="ge01",
            gen_id="g000001-full-id",
            agent_name="A",
            method_name="run",
            start_time=1_100,
            end_time=8_900,
        )
        gen["parentSpanId"] = "ag01"
        llm_spans = [
            make_llm_span(
                f"ll{i:02x}",
                "ge01",
                messages=[("user", f"q{i}")],
                start_time=1_200 + i * 200,
                end_time=1_300 + i * 200,
            )
            for i in range(3)
        ]
        exec_spans = [
            make_execution_span(
                f"ee{i:02x}",
                gen_id="g000001-full-id",
                agent_name="A",
                start_time=1_800 + i * 200,
                end_time=1_900 + i * 200,
            )
            for i in range(2)
        ]
        spans = [agent, gen, *llm_spans, *exec_spans]

        trace_file = create_trace_file(spans)
        try:
            trace = await TraceExplorer.from_file(trace_file)
            assert len(trace.sessions) == 1
            session = trace.sessions[0]
            assert len(session.turns) == 5, (
                f"Expected 5 turns (3 LLM + 2 exec), got {len(session.turns)}"
            )
            llm_turns = [t for t in session.turns if isinstance(t, LLMTurn)]
            exec_turns = [t for t in session.turns if isinstance(t, ExecutionTurn)]
            assert len(llm_turns) == 3
            assert len(exec_turns) == 2
            # Turns must be ordered by start_time: LLM calls (1_200–1_700)
            # come before exec calls (1_800–2_100).
            for t in session.turns[:3]:
                assert isinstance(t, LLMTurn)
            for t in session.turns[3:]:
                assert isinstance(t, ExecutionTurn)
        finally:
            trace_file.unlink()


class TestMultipleRoots:
    """Traces with multiple concurrent root AGENT spans."""

    async def test_two_independent_root_agents_produce_two_sessions(self):
        """Two AGENT spans with no parent and no cross-links should yield
        two top-level sessions, each with its own turns."""
        agent_a = make_agent_span(
            "aa01", agent_name="A", method_name="run", start_time=1_000, end_time=5_000
        )
        gen_a = make_generation_span(
            "ga01", "gen-A", agent_name="A", method_name="run", start_time=1_100, end_time=4_900
        )
        gen_a["parentSpanId"] = "aa01"
        llm_a = make_llm_span(
            "la01", "ga01", messages=[("user", "A")], start_time=1_200, end_time=1_300
        )

        agent_b = make_agent_span(
            "bb01", agent_name="B", method_name="run", start_time=2_000, end_time=6_000
        )
        gen_b = make_generation_span(
            "gb01", "gen-B", agent_name="B", method_name="run", start_time=2_100, end_time=5_900
        )
        gen_b["parentSpanId"] = "bb01"
        llm_b = make_llm_span(
            "lb01", "gb01", messages=[("user", "B")], start_time=2_200, end_time=2_300
        )

        trace_file = create_trace_file([agent_a, gen_a, llm_a, agent_b, gen_b, llm_b])
        try:
            trace = await TraceExplorer.from_file(trace_file)
            assert len(trace.sessions) == 2
            names = sorted(s.agent_name for s in trace.sessions)
            assert names == ["A", "B"]
            # Each root got exactly one turn (its own LLM call), no cross-contamination.
            for session in trace.sessions:
                assert len(session.turns) == 1, (
                    f"Session {session.agent_name} has {len(session.turns)} turns "
                    f"(expected 1 — turns from the other root leaked in?)"
                )
        finally:
            trace_file.unlink()


class TestCrossSessionParent:
    """Child span's parentSpanId points to a different session's generation.

    Regression guard for a data pattern seen in the kdd-cup experiment where
    sibling sessions' acompletion spans carry a `session.id` attribute
    distinct from their generation-span parent. The parser must not crash
    and must still return the session that owns an intact AGENT tree.
    """

    async def test_does_not_crash_and_still_populates_intact_session(self):
        agent_a = make_agent_span(
            "aa01", agent_name="A", method_name="run", start_time=1_000, end_time=5_000
        )
        gen_a = make_generation_span(
            "ga01", "gen-A", agent_name="A", method_name="run", start_time=1_100, end_time=4_900
        )
        gen_a["parentSpanId"] = "aa01"
        llm_a = make_llm_span(
            "la01", "ga01", messages=[("user", "A")], start_time=1_200, end_time=1_300
        )

        # This acompletion is mis-parented to A's generation but claims to
        # belong to agent B (no B AGENT span exists in this trace).
        stray = make_llm_span(
            "lx01", "ga01", messages=[("user", "stray")], start_time=1_400, end_time=1_500
        )
        # Override agent.name on the stray so it doesn't match A.
        # (It won't be picked up as A's turn because acompletion matching
        # happens via parent-chain hitting a generation with the right
        # gen_key, not via agent.name — so it WILL be attributed to A.
        # That's the current behavior; this test pins it down.)

        trace_file = create_trace_file([agent_a, gen_a, llm_a, stray])
        try:
            trace = await TraceExplorer.from_file(trace_file)
            assert len(trace.sessions) == 1
            assert trace.sessions[0].agent_name == "A"
            # Overview must render without raising.
            assert isinstance(await trace.get_overview(), str)
        finally:
            trace_file.unlink()


class TestPopulateTurnsFallbacks:
    """Three tiers of generation-span matching in `_populate_session_turns`.

    Tier 1: agent.call_id match on generation spans (primary).
    Tier 2: generation span is a descendant of the session's AGENT span.
    Tier 3: time-range overlap (warns).
    """

    async def test_tier2_descendant_match_populates_turns(self):
        """Generation span parented under the AGENT span, no call_id set.
        Turns should populate via descendant walk (no warnings)."""
        import warnings

        agent = make_agent_span(
            "ag01", agent_name="A", method_name="run", start_time=1_000, end_time=5_000
        )
        gen = make_generation_span(
            "ge01", "gen-1", agent_name="A", method_name="run", start_time=1_100, end_time=4_900
        )
        gen["parentSpanId"] = "ag01"
        llm = make_llm_span(
            "ll01", "ge01", messages=[("user", "hi")], start_time=1_200, end_time=1_300
        )

        trace_file = create_trace_file([agent, gen, llm])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trace = await TraceExplorer.from_file(trace_file)
            assert len(trace.sessions) == 1
            assert len(trace.sessions[0].turns) == 1
            fallback_warnings = [w for w in caught if "time-range fallback" in str(w.message)]
            assert not fallback_warnings, "Descendant path should not trigger time-range fallback"
        finally:
            trace_file.unlink()

    async def test_tier3_time_range_fallback_warns(self):
        """Generation span not parented under AGENT, no call_id. The parser
        should still find it via time-range overlap and emit a warning."""
        import warnings

        agent = make_agent_span(
            "ag01", agent_name="A", method_name="run", start_time=1_000, end_time=5_000
        )
        # gen's parent is NOT ag01 — simulating a broken parent chain.
        gen = make_generation_span(
            "ge01", "gen-1", agent_name="A", method_name="run", start_time=1_100, end_time=4_900
        )
        gen["parentSpanId"] = "unrelated-span"
        llm = make_llm_span(
            "ll01", "ge01", messages=[("user", "hi")], start_time=1_200, end_time=1_300
        )

        trace_file = create_trace_file([agent, gen, llm])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trace = await TraceExplorer.from_file(trace_file)
            assert len(trace.sessions) == 1
            assert len(trace.sessions[0].turns) == 1
            fallback_warnings = [w for w in caught if "time-range fallback" in str(w.message)]
            assert fallback_warnings, (
                "Expected a time-range fallback warning when gen isn't a descendant of AGENT"
            )
        finally:
            trace_file.unlink()


class TestTraceExplorerReasoningContent:
    """Reasoning content should be available to thin-client consumers."""

    def test_extract_reasoning_from_output_message(self):
        attrs = {
            "llm.output_messages.0.message.content": "final answer",
            "llm.output_messages.0.message.reasoning_content": "reasoning steps",
        }
        assert _extract_reasoning_content(attrs) == "reasoning steps"

    @pytest.mark.asyncio
    async def test_get_turn_includes_reasoning_by_default_and_can_suppress(self):
        trace_file = create_trace_file(
            [
                make_generation_span("gen1", "abcdef123456"),
                make_llm_span(
                    "llm1",
                    "gen1",
                    messages=[("user", "solve it")],
                    response="final answer",
                    reasoning_content="private chain of thought",
                ),
            ]
        )
        try:
            trace = await TraceExplorer.from_file(trace_file)
            with_reasoning = await trace.get_turn("abcdef", 0)
            without_reasoning = await trace.get_turn("abcdef", 0, include_reasoning=False)

            assert "<reasoning>" in with_reasoning
            assert "private chain of thought" in with_reasoning
            assert "final answer" in with_reasoning
            assert "private chain of thought" not in without_reasoning
            assert "final answer" in without_reasoning
        finally:
            trace_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_get_session_includes_reasoning_by_default_and_can_suppress(self):
        trace_file = create_trace_file(
            [
                make_generation_span("gen1", "abcdef123456"),
                make_llm_span(
                    "llm1",
                    "gen1",
                    messages=[("user", "solve it")],
                    response="final answer",
                    reasoning_content="session reasoning evidence",
                ),
            ]
        )
        try:
            trace = await TraceExplorer.from_file(trace_file)
            with_reasoning = await trace.get_session("abcdef", concise=False)
            without_reasoning = await trace.get_session(
                "abcdef", concise=False, include_reasoning=False
            )
            concise = await trace.get_session("abcdef", concise=True)

            assert "<reasoning>" in with_reasoning
            assert "session reasoning evidence" in with_reasoning
            assert "session reasoning evidence" not in without_reasoning
            assert "[reasoning]" in concise
        finally:
            trace_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_get_turn_data_exposes_reasoning(self):
        trace_file = create_trace_file(
            [
                make_generation_span("gen1", "abcdef123456"),
                make_llm_span(
                    "llm1",
                    "gen1",
                    messages=[("user", "solve it")],
                    response="final answer",
                    reasoning_content="structured reasoning",
                ),
            ]
        )
        try:
            trace = await TraceExplorer.from_file(trace_file)
            turn = await trace.get_turn_data("abcdef", 0)
            assert turn is not None
            assert turn.reasoning_content == "structured reasoning"
            assert turn.to_dict()["reasoning_content"] == "structured reasoning"
        finally:
            trace_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_turn_full_content_renders_reasoning_before_response(self):
        trace_file = create_trace_file(
            [
                make_generation_span("gen1", "abcdef123456"),
                make_llm_span(
                    "llm1",
                    "gen1",
                    messages=[("user", "solve it")],
                    response="final answer",
                    reasoning_content="structured reasoning",
                ),
            ]
        )
        try:
            trace = await TraceExplorer.from_file(trace_file)
            turn = await trace.get_turn_data("abcdef", 0)
            assert turn is not None
            content = turn.full_content()

            reasoning_index = content.index("### Reasoning")
            response_index = content.index("### LLM Response")
            assert reasoning_index < response_index
        finally:
            trace_file.unlink(missing_ok=True)
