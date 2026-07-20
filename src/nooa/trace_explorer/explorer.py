# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Navigable trace structure for agent-driven analysis.

Provides a programmatic interface for exploring agent traces.
Supports loading from OTLP .jsonl files or from a viewer API.

Usage:
    from nooa.trace_explorer import TraceExplorer

    # From file (OTLP format)
    trace = await TraceExplorer.from_file("path/to/trace.jsonl")

    # From viewer API
    trace = await TraceExplorer.from_viewer("http://localhost:5001", "session-id")

    print(await trace.get_overview())

    for session in await trace.get_session_list():
        print(session)
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa.trace_explorer.client import TraceExplorerClient

from nooa.agentdoc import pformat as _pformat

# =============================================================================
# Module Configuration
# =============================================================================

_quiet_mode: ContextVar[bool] = ContextVar("_quiet_mode", default=False)
_root_generation_index: ContextVar[int | None] = ContextVar("_root_generation_index", default=None)
_cli_mode: ContextVar[bool] = ContextVar("_cli_mode", default=False)


def set_quiet_mode(quiet: bool) -> None:
    """Enable or disable warning suppression.

    When quiet mode is enabled, parser warnings and other non-critical
    messages are suppressed. Errors are still reported.

    Args:
        quiet: True to suppress warnings, False to show them.
    """
    _quiet_mode.set(quiet)


def set_root_generation_index(index: int | None) -> None:
    """Set which root generation to use when multiple exist.

    Args:
        index: 0-based index into the sorted root generation IDs, or None for default (first).
    """
    _root_generation_index.set(index)


def get_root_generation_index() -> int | None:
    """Return the current root generation selection index."""
    return _root_generation_index.get()


def get_quiet_mode() -> bool:
    """Return the current quiet mode setting."""
    return _quiet_mode.get()


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_prefill_inputs(content: str) -> str | None:
    """Extract clean input arguments from prefill XML format.

    Prefill format looks like:
        <execute_python expr="self.events[3].content" tool_call_id="prefill_xxx">
        Execution successful.
        Stdout:
        Call: async def method(self, arg1: str, arg2: list) -> Result

        arg1 (str):
        'value1'

        arg2 (list):
        [1, 2, 3]

        Return type: Result { ... }
        </execute_python>

    Returns the clean arguments section or None if not prefill format.
    """
    try:
        if "<execute_python" not in content or "Stdout:" not in content:
            return None

        # Try regex extraction first (more robust to format variations)
        match = re.search(
            r"Stdout:\s*\n(.*?)(?:</execute_python>|\Z)",
            content,
            re.DOTALL,
        )
        if not match:
            # Fallback to string-based extraction
            stdout_start = content.find("Stdout:")
            if stdout_start == -1:
                return None
            stdout_content = content[stdout_start + len("Stdout:") :].strip()
            if "</execute_python>" in stdout_content:
                stdout_content = stdout_content[: stdout_content.find("</execute_python>")].strip()
        else:
            stdout_content = match.group(1).strip()

        # Skip the "Call:" line and get to arguments
        lines = stdout_content.split("\n")
        result_lines = []
        skip_call_line = True

        for line in lines:
            if skip_call_line and line.strip().startswith("Call:"):
                skip_call_line = False
                continue
            if skip_call_line:
                continue
            # Stop at "Return type:" line
            if line.strip().startswith("Return type:"):
                break
            result_lines.append(line)

        # Clean up leading/trailing empty lines
        result = "\n".join(result_lines).strip()
        return result if result else None
    except Exception:
        # Don't fail on malformed input - just return None
        return None


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class ToolCall:
    """A tool/function call made by the LLM."""

    function_name: str
    arguments: str  # JSON string of arguments
    tool_call_id: str = ""  # Tool call ID


@dataclass
class LLMMessage:
    """A single message in an LLM conversation."""

    role: str  # system, user, assistant, tool
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # For assistant messages
    tool_call_id: str = ""  # For tool messages


@dataclass
class ToolDefinition:
    """A tool definition available to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters


@dataclass
class LLMTurn:
    """An LLM generation turn with full message history and response."""

    session_id: str  # Short 6-char ID from generation.id
    messages: list[LLMMessage]  # Full conversation history up to this point
    response: str  # LLM's generated code/response
    model: str
    token_counts: dict[str, int] | None = None
    duration_ms: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # Tool calls in response
    reasoning_content: str = ""  # Hidden/model reasoning emitted alongside response
    span_id: str = ""  # Span ID of the acompletion span
    provider: str = ""  # LLM provider (e.g., "openai", "anthropic")
    invocation_parameters: dict[str, Any] = field(default_factory=dict)  # temperature, etc.
    tools: list[ToolDefinition] = field(default_factory=list)  # Available tools


@dataclass
class ExecutionTurn:
    """A code execution result."""

    code: str
    stdout: str
    error: str | None
    returned_value: Any
    duration_ms: float | None = None
    execution_id: str = ""  # Short 6-char ID from execution.id
    generation_id: str = ""  # Short 6-char ID from generation.id (for correlation)
    error_type: str | None = None  # Error type (e.g., "_ReturnResultSignal", "NameError")
    span_id: str = ""  # Span ID for reference
    tool_call_id: str = ""  # Tool call ID (e.g., "call_xxx" or "prefill_xxx")


@dataclass
class AgentSession:
    """A complete agent method invocation session.

    Represents an agent invocation with:
    - Identity: session_id, agent_name, method_name
    - Tree structure: parent_session_id, children, depth
    - Content: turns (LLM messages and code executions)
    - Metadata: start_time, end_time, result, status
    """

    session_id: str  # Short 6-char ID (from span_id)
    agent_name: str
    method_name: str
    parent_session_id: str | None
    depth: int = 0  # Nesting depth in call tree (0 = root)
    turns: list[LLMTurn | ExecutionTurn] = field(default_factory=list)
    children: list[AgentSession] = field(default_factory=list)
    start_time: int = 0  # nanoseconds
    end_time: int = 0  # nanoseconds
    result: Any = None
    status: str = "OK"
    span_id: str = ""  # Full span_id for correlation
    method_signature: str = ""  # e.g., "handle_user_message(user_message: str) -> UserResponse"
    docstring: str = ""  # First line of docstring
    file_path: str = ""  # Relative path to agent file
    args: list[Any] = field(default_factory=list)  # Positional arguments from span
    kwargs: dict[str, Any] = field(default_factory=dict)  # Keyword arguments from span
    error_message: str | None = None  # Error message from span attributes (error.message)
    strategy: str | None = None  # Strategy used (e.g., STRUCTURED_OUTPUT, CODE_ACT)
    call_id: str = ""  # agent.call_id from AGENT span (for generation span correlation)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) / 1_000_000

    @property
    def full_name(self) -> str:
        """Full method name like 'RouterTestWrapper.process'."""
        return f"{self.agent_name}.{self.method_name}"

    def get_error_turns(self) -> list[ExecutionTurn]:
        """Get all turns that have errors."""
        return [t for t in self.turns if isinstance(t, ExecutionTurn) and t.error]

    def get_llm_turns(self) -> list[LLMTurn]:
        """Get all LLM turns."""
        return [t for t in self.turns if isinstance(t, LLMTurn)]

    def get_execution_turns(self) -> list[ExecutionTurn]:
        """Get all execution turns."""
        return [t for t in self.turns if isinstance(t, ExecutionTurn)]


# =============================================================================
# Parsing Helpers
# =============================================================================


def _short_id(full_id: str | None) -> str:
    """Get a short 6-char ID for display."""
    if not full_id:
        return "------"
    return full_id[:6]


# ---------------------------------------------------------------------------
# OpenInference-first attribute reads
#
# Spans now carry OpenInference-standard I/O attrs (``input.value`` /
# ``output.value``) as the canonical representation, with the legacy
# nooa-native attrs (``agent.result``, ``code``, ``tool.arguments``,
# ``generation.result``, ``agent.args``/``kwargs``, ``result``, …) kept for a
# deprecation window. These helpers prefer the OI name and fall back to the
# native name(s), so the explorer renders BOTH new OI-only traces and old
# native-only traces.
# ---------------------------------------------------------------------------


def _io_value(attrs: dict[str, Any], oi_key: str, *native_keys: str) -> Any:
    """Return ``attrs[oi_key]`` if present, else the first present native key.

    Uses ``is not None`` (not truthiness) so a deliberately-empty value such as
    ``output.value == ""`` is preserved rather than skipped.
    """
    v = attrs.get(oi_key)
    if v is not None:
        return v
    for k in native_keys:
        v = attrs.get(k)
        if v is not None:
            return v
    return None


def _io_json_field(
    attrs: dict[str, Any], oi_key: str, field_name: str, *native_keys: str, default: Any = None
) -> Any:
    """Prefer ``json.loads(attrs[oi_key])[field_name]``, else first native key.

    The OI I/O value for tool/method/agent/code spans is a JSON object — e.g.
    agent/method inputs are ``{"args": [...], "kwargs": {...}}`` and code-exec
    input is ``{"code": "..."}``. This extracts one field from that object,
    falling back to the legacy flat native attr(s) for old traces.
    """
    raw = attrs.get(oi_key)
    if raw is not None:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict) and field_name in parsed:
                return parsed[field_name]
        except (json.JSONDecodeError, TypeError):
            pass
    for k in native_keys:
        v = attrs.get(k)
        if v is not None:
            return v
    return default


def _extract_any_value(value_obj: dict[str, Any]) -> Any:
    """Extract a scalar from an OTLP AnyValue dict."""
    if "stringValue" in value_obj:
        return value_obj["stringValue"]
    if "intValue" in value_obj:
        return int(value_obj["intValue"])
    if "doubleValue" in value_obj:
        return float(value_obj["doubleValue"])
    if "boolValue" in value_obj:
        return value_obj["boolValue"]
    if "bytesValue" in value_obj:
        return value_obj["bytesValue"]
    if "arrayValue" in value_obj:
        return [_extract_any_value(v) for v in value_obj["arrayValue"].get("values", [])]
    if "kvlistValue" in value_obj:
        return {
            kv["key"]: _extract_any_value(kv.get("value", {}))
            for kv in value_obj["kvlistValue"].get("values", [])
        }
    return None


def _otlp_attrs_to_dict(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OTLP attribute array [{key, value}] to a flat dict."""
    result: dict[str, Any] = {}
    for attr in attrs:
        key = attr.get("key", "")
        value_obj = attr.get("value", {})
        if "stringValue" in value_obj:
            result[key] = value_obj["stringValue"]
        elif "intValue" in value_obj:
            result[key] = int(value_obj["intValue"])
        elif "doubleValue" in value_obj:
            result[key] = float(value_obj["doubleValue"])
        elif "boolValue" in value_obj:
            result[key] = value_obj["boolValue"]
        elif "arrayValue" in value_obj:
            result[key] = [_extract_any_value(v) for v in value_obj["arrayValue"].get("values", [])]
        elif "kvlistValue" in value_obj:
            result[key] = {
                kv["key"]: _extract_any_value(kv.get("value", {}))
                for kv in value_obj["kvlistValue"].get("values", [])
            }
        elif "bytesValue" in value_obj:
            result[key] = value_obj["bytesValue"]
    return result


def _first_code_line(code: str, max_len: int = 60) -> str:
    """Return the first non-comment, non-docstring line of *code*, truncated.

    Appends ``; ...`` when there are additional meaningful lines so the caller
    can see at a glance that the cell contained more than just that one line.
    """
    lines = code.split("\n")
    first: str = ""
    has_more = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""'):
            continue
        if not first:
            first = stripped
        else:
            has_more = True
            break
    if not first:
        return ""
    suffix = "; ..." if has_more else ""
    budget = max_len - len(suffix)
    if len(first) > budget:
        return first[: budget - 3] + "..." + suffix
    return first + suffix


def _extract_failing_line(error: str) -> str:
    """Extract the code line that caused the error, if the error carries one.

    Handles the IPython-style traceback emitted by the code validator::

        Cell In[34], line 1
            import numpy as np
            ^
        import of 'numpy' is forbidden...

    Returns the stripped code line, or ``""`` if the pattern is not found.
    """
    m = re.search(r"Cell In\[.*?\], line \d+\n\s+(.+?)(?:\n|$)", error)
    if m:
        line = m.group(1).strip()
        return line[:57] + "..." if len(line) > 60 else line
    return ""


def _format_turn_status(error: str | None) -> str:
    """Return a status label: ``[OK]`` or ``[ERR: <brief message>]``."""
    if not error:
        return "[OK]"
    # For Cell In traceback errors, the real message follows the caret line.
    m = re.search(r"Cell In\[.*?\], line \d+\n\s+.+?\n\s+\^\n(.+?)(?:\n|$)", error)
    if m:
        msg = m.group(1).strip()
    else:
        first_line = error.split("\n")[0].strip()
        # FileNotFoundError: shorten the path to just the filename
        path_m = re.search(r"No such file or directory: '(.+?)'", first_line)
        if path_m:
            msg = f"No such file: {Path(path_m.group(1)).name}"
        else:
            msg = first_line
    if len(msg) > 50:
        msg = msg[:47] + "..."
    return f"[ERR: {msg}]"


def _normalize_otlp_span(span: dict[str, Any]) -> dict[str, Any]:
    """Convert an OTLP span to internal flat-attribute format."""
    status = span.get("status", {})
    code = status.get("code", 0)
    # Map OTLP status: 0 (UNSET) and 1 (OK) both map to "OK" for compat
    status_map = {0: "OK", 1: "OK", 2: "ERROR"}

    start_ns = int(span.get("startTimeUnixNano", 0))
    end_ns = int(span.get("endTimeUnixNano", 0))

    resource = span.get("_resource", {})
    if isinstance(resource.get("attributes"), list):
        resource = {"attributes": _otlp_attrs_to_dict(resource["attributes"])}

    return {
        "span_id": span.get("spanId", ""),
        "trace_id": span.get("traceId", ""),
        "parent_span_id": span.get("parentSpanId"),
        "name": span.get("name", ""),
        "kind": span.get("kind"),
        "start_time": start_ns,
        "end_time": end_ns,
        "duration_ns": end_ns - start_ns if end_ns and start_ns else 0,
        "attributes": _otlp_attrs_to_dict(span.get("attributes", [])),
        "events": span.get("events", []),
        "status": {
            "status_code": status_map.get(code, "OK"),
            "description": status.get("message"),
        },
        "resource": resource,
    }


def _load_spans(trace_path: str | Path) -> list[dict[str, Any]]:
    """Load all spans from a .jsonl file and normalize to internal format.

    Handles two formats:
    - OTLP TracesData envelope: ``{"resourceSpans": [...]}`` (one per line)
    - Flat span objects: ``{"span_id": ..., "name": ..., ...}`` (one per line)

    Parse errors are logged to stderr but don't stop loading.
    """
    spans = []
    parse_errors = 0
    with open(trace_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    raw = json.loads(line)
                    if "resourceSpans" in raw:
                        # OTLP TracesData envelope — unwrap to individual spans
                        for rs in raw.get("resourceSpans", []):
                            resource = rs.get("resource", {})
                            for ss in rs.get("scopeSpans", []):
                                for span in ss.get("spans", []):
                                    span["_resource"] = resource
                                    spans.append(_normalize_otlp_span(span))
                    else:
                        # Legacy flat span format
                        spans.append(_normalize_otlp_span(raw))
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if not _quiet_mode.get() and parse_errors <= 3:  # Limit noise
                        print(f"Warning: Parse error at line {line_num}: {e}", file=sys.stderr)
    if not _quiet_mode.get() and parse_errors > 3:
        print(f"Warning: {parse_errors} total JSON parse errors in trace file", file=sys.stderr)
    return spans


def _build_span_index(spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build an index of spans by span_id.

    Note: If duplicate span_ids exist, later spans overwrite earlier ones.
    A warning is emitted for duplicates.
    """
    index: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for span in spans:
        span_id = span.get("span_id")
        if not span_id:
            continue
        if span_id in index:
            duplicates += 1
        index[span_id] = span

    if duplicates > 0 and not _quiet_mode.get():
        import warnings

        warnings.warn(f"Found {duplicates} duplicate span_id(s) in trace", stacklevel=2)

    return index


def _extract_messages(attrs: dict[str, Any]) -> list[LLMMessage]:
    """Extract ALL input messages from span attributes.

    The attributes contain flattened message data like:
        llm.input_messages.0.message.role = "system"
        llm.input_messages.0.message.content = "..."
        llm.input_messages.2.message.tool_calls.0.tool_call.function.name = "..."

    Handles non-contiguous indices (e.g., starting from index 4).
    """
    # Find all message indices present in attributes
    indices = set()
    for key in attrs:
        if key.startswith("llm.input_messages.") and ".message.role" in key:
            # Extract the index from the key
            try:
                idx_str = key.split(".")[2]
                indices.add(int(idx_str))
            except (IndexError, ValueError):
                continue

    if not indices:
        return []

    # Extract messages in order
    messages = []
    for i in sorted(indices):
        role_key = f"llm.input_messages.{i}.message.role"
        content_key = f"llm.input_messages.{i}.message.content"
        tool_call_id_key = f"llm.input_messages.{i}.message.tool_call_id"

        role = attrs.get(role_key, "unknown")
        content = attrs.get(content_key) or ""
        tool_call_id = attrs.get(tool_call_id_key) or ""

        # Extract tool_calls for assistant messages
        tool_calls = []
        if role == "assistant":
            # Find tool call indices for this message
            tc_prefix = f"llm.input_messages.{i}.message.tool_calls."
            tc_indices = set()
            for key in attrs:
                if key.startswith(tc_prefix) and ".tool_call.function.name" in key:
                    try:
                        tc_idx = int(key.split(".")[5])
                        tc_indices.add(tc_idx)
                    except (IndexError, ValueError):
                        continue

            for tc_idx in sorted(tc_indices):
                name_key = f"{tc_prefix}{tc_idx}.tool_call.function.name"
                args_key = f"{tc_prefix}{tc_idx}.tool_call.function.arguments"
                id_key = f"{tc_prefix}{tc_idx}.tool_call.id"

                tc_name = attrs.get(name_key, "")
                tc_args = attrs.get(args_key, "")
                tc_id = attrs.get(id_key, "")

                if tc_name:
                    tool_calls.append(
                        ToolCall(
                            function_name=tc_name,
                            arguments=tc_args,
                            tool_call_id=tc_id,
                        )
                    )

        messages.append(
            LLMMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    return messages


def _extract_tool_calls(attrs: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from LLM output messages.

    The attributes contain flattened tool call data like:
        llm.output_messages.0.message.tool_calls.0.tool_call.function.name = "execute_python"
        llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments = "{...}"
        llm.output_messages.0.message.tool_calls.0.tool_call.id = "call_xxx"

    Note: Iterates through all output messages (0, 1, 2...) not just the first one.
    """
    tool_calls = []

    # Iterate through output messages (most traces have only message 0, but handle multiple)
    msg_idx = 0
    max_empty_messages = 2  # Stop after 2 consecutive empty messages
    empty_count = 0

    while empty_count < max_empty_messages:
        found_in_message = False
        tc_idx = 0

        while True:
            name_key = (
                f"llm.output_messages.{msg_idx}.message.tool_calls.{tc_idx}.tool_call.function.name"
            )
            args_key = f"llm.output_messages.{msg_idx}.message.tool_calls.{tc_idx}.tool_call.function.arguments"
            id_key = f"llm.output_messages.{msg_idx}.message.tool_calls.{tc_idx}.tool_call.id"

            if name_key not in attrs:
                break

            found_in_message = True
            tool_calls.append(
                ToolCall(
                    function_name=attrs.get(name_key, "unknown"),
                    arguments=attrs.get(args_key, "{}"),
                    tool_call_id=attrs.get(id_key, ""),
                )
            )
            tc_idx += 1

        if found_in_message:
            empty_count = 0
        else:
            empty_count += 1

        msg_idx += 1

    return tool_calls


def _extract_response(attrs: dict[str, Any]) -> str:
    """Extract the LLM response from span attributes."""
    # Try the first output message content
    content = attrs.get("llm.output_messages.0.message.content", "")
    if content:
        return content
    # Fallback to output.value if available
    return attrs.get("output.value", "")


def _extract_reasoning_content(attrs: dict[str, Any]) -> str:
    """Extract model reasoning from span attributes."""
    reasoning_parts: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value and value not in reasoning_parts:
            reasoning_parts.append(value)

    add(attrs.get("llm.reasoning_content"))

    msg_idx = 0
    max_empty_messages = 2
    empty_count = 0
    while empty_count < max_empty_messages:
        found = False
        for suffix in (
            "message.reasoning_content",
            "message.additional_kwargs.reasoning_content",
            "message.provider_specific_fields.reasoning_content",
        ):
            key = f"llm.output_messages.{msg_idx}.{suffix}"
            if key in attrs:
                found = True
                add(attrs.get(key))
        if f"llm.output_messages.{msg_idx}.message.content" in attrs:
            found = True
        empty_count = 0 if found else empty_count + 1
        msg_idx += 1

    output_value = attrs.get("output.value")
    if isinstance(output_value, str) and "reasoning_content" in output_value:
        try:
            parsed = json.loads(output_value)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            add(parsed.get("reasoning_content"))
            choices = parsed.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        message = choice.get("message")
                        if isinstance(message, dict):
                            add(message.get("reasoning_content"))

    return "\n".join(reasoning_parts)


def _extract_token_counts(attrs: dict[str, Any]) -> dict[str, int] | None:
    """Extract token usage from span attributes."""
    prompt = attrs.get("llm.token_count.prompt")
    completion = attrs.get("llm.token_count.completion")
    total = attrs.get("llm.token_count.total")

    if prompt is not None or completion is not None or total is not None:
        return {
            "prompt": prompt or 0,
            "completion": completion or 0,
            "total": total or 0,
        }
    return None


def _extract_invocation_parameters(attrs: dict[str, Any]) -> dict[str, Any]:
    """Extract invocation parameters (temperature, etc.) from span attributes."""
    params_str = attrs.get("llm.invocation_parameters", "")
    if params_str:
        try:
            return json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _extract_tools(attrs: dict[str, Any]) -> list[ToolDefinition]:
    """Extract tool definitions from span attributes.

    The attributes contain flattened tool data like:
        llm.tools.0.tool.json_schema = '{"type": "function", "function": {...}}'
    """
    tools = []
    idx = 0

    while True:
        schema_key = f"llm.tools.{idx}.tool.json_schema"
        if schema_key not in attrs:
            break

        schema_str = attrs.get(schema_key, "")
        if schema_str:
            try:
                schema = json.loads(schema_str)
                # OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
                if schema.get("type") == "function" and "function" in schema:
                    func = schema["function"]
                    tools.append(
                        ToolDefinition(
                            name=func.get("name", ""),
                            description=func.get("description", ""),
                            parameters=func.get("parameters", {}),
                        )
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        idx += 1

    return tools


def _parse_execution_result(result_str: str) -> tuple[str, Any, str | None]:
    """Parse a JSON-encoded ExecutionResult.

    Returns:
        Tuple of (stdout, returned_value, error_message)
    """
    if not result_str:
        return "", None, None

    try:
        result = json.loads(result_str)
        stdout = result.get("stdout", "")
        returned = result.get("returned_value")
        error = result.get("error")
        return stdout, returned, error
    except json.JSONDecodeError:
        return result_str, None, None


def _get_all_sessions(sessions: list[AgentSession]) -> list[AgentSession]:
    """Flatten session tree into a list."""
    all_sessions = []
    for session in sessions:
        all_sessions.append(session)
        all_sessions.extend(_get_all_sessions(session.children))
    return all_sessions


def _parse_trace_from_spans(spans: list[dict[str, Any]]) -> list[AgentSession]:
    """Parse loaded spans into structured AgentSession objects.

    Builds a complete tree from AGENT spans with:
    - Parent-child relationships via parent_span_id
    - Depth calculated from tree structure
    - Content (turns) populated from associated generation/acompletion/execution spans

    Returns a list of root-level sessions (those with no parent AGENT span).
    """
    if not spans:
        return []

    span_index = _build_span_index(spans)

    # Step 1: Find all AGENT spans and build the tree structure.
    # Generation spans (kind=CHAIN, span name "generation") represent internal runs
    # within a session, not top-level sessions. They are excluded here both by the
    # kind check and by carrying a "generation.id" attribute (AGENT spans don't).
    agent_spans = [
        s
        for s in spans
        if s.get("attributes", {}).get("openinference.span.kind") == "AGENT"
        and not s.get("attributes", {}).get("generation.id")
    ]

    if not agent_spans:
        # Fallback to generation-based parsing if no AGENT spans
        return _parse_trace_from_generation_spans(spans, span_index)

    # Build session objects from AGENT spans
    sessions_by_span_id: dict[str, AgentSession] = {}

    for agent_span in agent_spans:
        span_id = agent_span.get("span_id", agent_span.get("context", {}).get("span_id", ""))
        if not span_id:
            continue

        attrs = agent_span.get("attributes", {})

        # Get agent name - prefer attributes.agent.name over parsing span name
        # This handles cases where span name is "plan.method" but actual agent is different
        agent_name = attrs.get("agent.name")
        method_name = attrs.get("agent.method")

        # Fall back to parsing from span name if attributes missing
        if not agent_name or not method_name:
            name = agent_span.get("name", "Unknown.unknown")
            if "." in name:
                parsed_agent, parsed_method = name.rsplit(".", 1)
                agent_name = agent_name or parsed_agent
                method_name = method_name or parsed_method
            else:
                agent_name = agent_name or name
                method_name = method_name or "unknown"

        parent_span_id = agent_span.get("parent_span_id")
        status = "OK"
        if agent_span.get("status", {}).get("status_code") == "ERROR":
            status = "ERROR"

        # Extract args, kwargs, and result from span attributes (OI-first, native
        # fallback). New traces carry input as ``input.value`` =
        # ``{"args": [...], "kwargs": {...}}`` and output as ``output.value``; old
        # traces carry ``agent.args``/``agent.kwargs``/``agent.result``.
        # These may be stored as JSON strings, so parse them downstream.
        # NOTE: the OI branch returns a pre-parsed list/dict (from input.value JSON)
        # while the native branch returns a JSON string — both are handled by the
        # ``isinstance(..., str)`` guard in the parsing block below. Keep that guard.
        span_args_raw = _io_json_field(attrs, "input.value", "args", "agent.args", default="[]")
        span_kwargs_raw = _io_json_field(
            attrs, "input.value", "kwargs", "agent.kwargs", default="{}"
        )
        span_result_raw = _io_value(attrs, "output.value", "agent.result")

        try:
            span_args = (
                json.loads(span_args_raw) if isinstance(span_args_raw, str) else span_args_raw
            )
        except (json.JSONDecodeError, TypeError):
            span_args = []

        try:
            span_kwargs = (
                json.loads(span_kwargs_raw) if isinstance(span_kwargs_raw, str) else span_kwargs_raw
            )
        except (json.JSONDecodeError, TypeError):
            span_kwargs = {}

        # Result might be JSON or plain string
        span_result = span_result_raw
        if isinstance(span_result_raw, str):
            try:
                span_result = json.loads(span_result_raw)
            except (json.JSONDecodeError, TypeError):
                span_result = span_result_raw

        session = AgentSession(
            session_id=_short_id(span_id),
            agent_name=agent_name,
            method_name=method_name,
            parent_session_id=None,  # Will be set later
            depth=0,  # Will be calculated later
            start_time=agent_span.get("start_time", 0),
            end_time=agent_span.get("end_time", 0),
            result=span_result,
            status=status,
            span_id=span_id,
            method_signature=attrs.get("agent.method_signature", ""),
            docstring=attrs.get("agent.docstring", ""),
            file_path=attrs.get("agent.file_path", ""),
            args=span_args if isinstance(span_args, list) else [],
            kwargs=span_kwargs if isinstance(span_kwargs, dict) else {},
            error_message=attrs.get("error.message"),
            strategy=attrs.get("agent.strategy.name"),
            call_id=attrs.get("agent.call_id", ""),
        )
        sessions_by_span_id[span_id] = session

    # Step 2: Build parent-child relationships and calculate depth
    root_sessions: list[AgentSession] = []

    for span_id, session in sessions_by_span_id.items():
        agent_span = span_index.get(span_id)
        if not agent_span:
            continue

        parent_span_id = agent_span.get("parent_span_id")

        # Find parent AGENT span (may be indirect through LLM/TOOL spans)
        # Use visited set to prevent infinite loops from malformed span hierarchies
        parent_session = None
        current_parent_id = parent_span_id
        visited: set[str] = set()
        while current_parent_id and current_parent_id not in visited:
            visited.add(current_parent_id)
            if current_parent_id in sessions_by_span_id:
                parent_session = sessions_by_span_id[current_parent_id]
                break
            parent_span = span_index.get(current_parent_id)
            if not parent_span:
                break
            current_parent_id = parent_span.get("parent_span_id")

        if parent_session:
            session.parent_session_id = parent_session.session_id
            parent_session.children.append(session)
        else:
            root_sessions.append(session)

    # Step 3: Calculate depth via BFS from roots
    def set_depths(sessions: list[AgentSession], depth: int) -> None:
        for session in sessions:
            session.depth = depth
            set_depths(session.children, depth + 1)

    set_depths(root_sessions, 0)

    # Step 4: Populate turns for each session from associated spans
    # Must populate ALL sessions, not just roots - child sessions have their own turns
    all_sessions = _get_all_sessions(root_sessions)
    for session in all_sessions:
        _populate_session_turns(session, spans, span_index)

    # Sort by start_time
    root_sessions.sort(key=lambda s: s.start_time)
    return root_sessions


def _parse_trace_from_generation_spans(
    spans: list[dict[str, Any]], span_index: dict[str, dict]
) -> list[AgentSession]:
    """Fallback parser using generation spans when no AGENT spans exist."""
    # Group generation spans by their generation.id (first 6 chars)
    gen_id_to_spans: dict[str, list[dict]] = {}

    for span in spans:
        if span["name"] == "generation":
            attrs = span.get("attributes", {})
            gen_id = attrs.get("generation.id", "")
            parent_gen_id = attrs.get("generation.parent_id")

            # Only consider root generation spans (no parent generation)
            if not parent_gen_id:
                key = _short_id(gen_id)
                if key not in gen_id_to_spans:
                    gen_id_to_spans[key] = []
                gen_id_to_spans[key].append(span)

    if not gen_id_to_spans:
        return []

    sessions: list[AgentSession] = []

    for gen_key, gen_spans in gen_id_to_spans.items():
        first_span = gen_spans[0]
        attrs = first_span.get("attributes", {})
        agent_name = attrs.get("agent.name", "Unknown")
        method_name = attrs.get("agent.method", "Unknown")

        all_gen_spans = [
            s
            for s in spans
            if s["name"] == "generation"
            and _short_id(s.get("attributes", {}).get("generation.id", "")) == gen_key
        ]
        start_time = min(s.get("start_time", 0) for s in all_gen_spans)
        end_time = max(s.get("end_time", 0) for s in all_gen_spans)

        final_span = max(all_gen_spans, key=lambda s: s.get("end_time", 0))
        status = "ERROR" if final_span.get("status", {}).get("status_code") == "ERROR" else "OK"

        session = AgentSession(
            session_id=gen_key,
            agent_name=agent_name,
            method_name=method_name,
            parent_session_id=None,
            depth=0,
            start_time=start_time,
            end_time=end_time,
            result=_io_value(final_span.get("attributes", {}), "output.value", "generation.result"),
            status=status,
            span_id=first_span.get("span_id", ""),  # Capture span_id for correlation
        )

        _populate_session_turns_from_generation(session, gen_key, spans, span_index)
        sessions.append(session)

    sessions.sort(key=lambda s: s.start_time)
    return sessions


def _populate_session_turns(
    session: AgentSession, spans: list[dict[str, Any]], span_index: dict[str, dict]
) -> None:
    """Populate turns for a session from associated spans."""
    # Find generation spans that are descendants of this session's span
    session_span_id = session.span_id

    # Build set of all descendant span IDs
    descendant_ids: set[str] = set()
    to_check = [session_span_id]
    while to_check:
        current_id = to_check.pop()
        for s in spans:
            if s.get("parent_span_id") == current_id:
                child_id = s.get("span_id", "")
                if child_id and child_id not in descendant_ids:
                    descendant_ids.add(child_id)
                    to_check.append(child_id)

    # Primary match: generation spans that stamp agent.call_id matching this session
    # This handles sub-agents whose generation spans are parented to a caller's span
    # rather than to the sub-agent's own AGENT span (cross-session parent links).
    gen_spans = []
    if session.call_id:
        gen_spans = [
            s
            for s in spans
            if s["name"] == "generation"
            and s.get("attributes", {}).get("agent.call_id") == session.call_id
            and s.get("attributes", {}).get("agent.name") == session.agent_name
            and s.get("attributes", {}).get("agent.method") == session.method_name
        ]

    # Fallback: find generation spans that are descendants of this session's span
    if not gen_spans:
        gen_spans = [
            s
            for s in spans
            if s["name"] == "generation"
            and s.get("span_id", "") in descendant_ids
            and s.get("attributes", {}).get("agent.name") == session.agent_name
            and s.get("attributes", {}).get("agent.method") == session.method_name
        ]
    # Final fallback: if no descendants found via parent links, try time-range matching
    # Note: This fallback is less reliable if sessions have overlapping time ranges
    used_time_fallback = False
    if not gen_spans:
        gen_spans = [
            s
            for s in spans
            if s["name"] == "generation"
            and s.get("attributes", {}).get("agent.name") == session.agent_name
            and s.get("attributes", {}).get("agent.method") == session.method_name
            and session.start_time <= s.get("start_time", 0) <= session.end_time
        ]
        if gen_spans:
            used_time_fallback = True

    if not gen_spans:
        return

    # Get the root generation ID (no parent_id) from THIS session's generations
    root_gen_ids = set()
    for gs in gen_spans:
        attrs = gs.get("attributes", {})
        if not attrs.get("generation.parent_id"):
            root_gen_ids.add(_short_id(attrs.get("generation.id", "")))

    if not root_gen_ids:
        return

    # Choose root generation for this session
    # Note: Multiple roots might indicate concurrent executions or parsing ambiguity
    sorted_root_ids = sorted(root_gen_ids)
    selected_index = 0
    _rgi = _root_generation_index.get()
    if _rgi is not None:
        if 0 <= _rgi < len(sorted_root_ids):
            selected_index = _rgi
        elif not _quiet_mode.get():
            import warnings

            warnings.warn(
                f"Session {session.session_id}: Requested root generation index "
                f"{_rgi} out of range (0..{len(sorted_root_ids) - 1}); "
                f"using 0",
                stacklevel=2,
            )

    gen_key = sorted_root_ids[selected_index]
    if not _quiet_mode.get() and (len(root_gen_ids) > 1 or used_time_fallback):
        import warnings

        if used_time_fallback:
            warnings.warn(
                f"Session {session.session_id}: Used time-range fallback to find "
                f"{len(gen_spans)} generation spans (may be inaccurate)",
                stacklevel=2,
            )
        if len(root_gen_ids) > 1:
            warnings.warn(
                f"Session {session.session_id}: Found {len(root_gen_ids)} root generations, using first",
                stacklevel=2,
            )
    _populate_session_turns_from_generation(session, gen_key, spans, span_index)


def _populate_session_turns_from_generation(
    session: AgentSession,
    gen_key: str,
    spans: list[dict[str, Any]],
    span_index: dict[str, dict],
) -> None:
    """Populate turns from generation/acompletion/execution spans."""
    # Find all generation spans for this session
    all_gen_spans = [
        s
        for s in spans
        if s["name"] == "generation"
        and _short_id(s.get("attributes", {}).get("generation.id", "")) == gen_key
    ]

    session_gen_ids = {
        _short_id(s.get("attributes", {}).get("generation.id", "")) for s in all_gen_spans
    }

    turn_spans: list[tuple[str, dict, str]] = []  # (type, span, generation_id)

    for span in spans:
        span_name = span["name"]
        span_attrs = span.get("attributes", {})

        if span_name == "acompletion":
            parent_id = span.get("parent_span_id")
            while parent_id:
                parent_span = span_index.get(parent_id)
                if not parent_span:
                    break
                if parent_span["name"] == "generation":
                    p_attrs = parent_span.get("attributes", {})
                    parent_gen_id = p_attrs.get("generation.id", "")
                    if _short_id(parent_gen_id) == gen_key:
                        # Store the generation.id from the parent span
                        turn_spans.append(("llm", span, parent_gen_id))
                        break
                parent_id = parent_span.get("parent_span_id")

        elif span_name == "code_execution":
            if span_attrs.get("agent.name") == session.agent_name:
                exec_gen_id = span_attrs.get("generation.id", "")
                if _short_id(exec_gen_id) == gen_key or _short_id(exec_gen_id) in session_gen_ids:
                    turn_spans.append(("exec", span, exec_gen_id))

    turn_spans.sort(key=lambda x: x[1].get("start_time", 0))

    for span_type, span, gen_id in turn_spans:
        attrs = span.get("attributes", {})
        duration_ns = span.get("duration_ns", 0)

        if span_type == "llm":
            turn = LLMTurn(
                session_id=_short_id(gen_id),
                messages=_extract_messages(attrs),
                response=_extract_response(attrs),
                model=attrs.get("llm.model_name", "unknown"),
                token_counts=_extract_token_counts(attrs),
                duration_ms=duration_ns / 1_000_000 if duration_ns else None,
                tool_calls=_extract_tool_calls(attrs),
                reasoning_content=_extract_reasoning_content(attrs),
                span_id=span.get("span_id", ""),
                provider=attrs.get("llm.provider", ""),
                invocation_parameters=_extract_invocation_parameters(attrs),
                tools=_extract_tools(attrs),
            )
            session.turns.append(turn)
        else:
            # OI-first: code-exec output is ``output.value`` (same JSON as the
            # legacy ``result`` attr); fall back to ``result`` for old traces.
            result_str = _io_value(attrs, "output.value", "result") or ""
            stdout, returned, error = _parse_execution_result(result_str)
            status_obj = span.get("status", {})
            error_msg = error or attrs.get("error.message") or status_obj.get("description")
            error_type = attrs.get("error.type")  # Capture error type (e.g., "_ReturnResultSignal")

            turn = ExecutionTurn(
                # OI-first: code lives in ``input.value`` = {"code": ...}; fall
                # back to the legacy flat ``code`` attr for old traces.
                code=_io_json_field(attrs, "input.value", "code", "code", default=""),
                stdout=stdout,
                error=error_msg,
                returned_value=returned,
                duration_ms=duration_ns / 1_000_000 if duration_ns else None,
                execution_id=_short_id(attrs.get("execution.id", "")),
                generation_id=_short_id(gen_id),
                error_type=error_type,
                span_id=span.get("span_id", ""),
                tool_call_id=attrs.get("tool_call_id", ""),
            )
            session.turns.append(turn)


# =============================================================================
# Summary Data Structures (Agent-friendly views)
# =============================================================================


@dataclass
class SessionSummary:
    """Summary of a single agent session."""

    session_id: str
    agent_name: str
    method_name: str
    status: str
    turn_count: int
    llm_turns: int
    execution_turns: int
    duration_ms: float
    parent_session_id: str | None
    has_children: bool
    result_preview: str | None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __str__(self) -> str:
        status_label = "[OK]" if self.status == "OK" else "[ERR]"
        parent_info = f" (child of {self.parent_session_id})" if self.parent_session_id else ""
        return (
            f"[{self.session_id}] {self.agent_name}.{self.method_name}(){parent_info} "
            f"{status_label} - {self.turn_count} turns ({self.llm_turns} LLM, {self.execution_turns} exec) "
            f"in {self.duration_ms:.1f}ms"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "method_name": self.method_name,
            "status": self.status,
            "turn_count": self.turn_count,
            "llm_turns": self.llm_turns,
            "execution_turns": self.execution_turns,
            "duration_ms": self.duration_ms,
            "parent_session_id": self.parent_session_id,
            "has_children": self.has_children,
            "result_preview": self.result_preview,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


@dataclass
class TurnInfo:
    """Detailed info about a single turn."""

    session_id: str
    turn_index: int
    turn_type: str  # "llm" or "execution"

    # For LLM turns
    messages: list[dict] | None = None  # [{role, content}]
    response: str | None = None
    model: str | None = None
    token_counts: dict[str, int] | None = None
    tool_calls: list[dict] | None = None  # [{function_name, arguments}]
    reasoning_content: str | None = None

    # For execution turns
    code: str | None = None
    stdout: str | None = None
    error: str | None = None
    error_type: str | None = None  # Error type (e.g., "NameError", "_ReturnResultSignal")
    returned_value: Any = None

    duration_ms: float | None = None

    def __str__(self) -> str:
        if self.turn_type == "llm":
            msg_count = len(self.messages) if self.messages else 0
            response_preview = (
                (self.response[:100] + "...")
                if self.response and len(self.response) > 100
                else self.response
            )
            tool_info = f", {len(self.tool_calls)} tool calls" if self.tool_calls else ""
            return (
                f"Turn {self.turn_index} [LLM]: {msg_count} input messages{tool_info}\n"
                f"Response preview: {response_preview}"
            )
        else:
            status = self.error_type if self.error_type else ("ERROR" if self.error else "OK")
            code_preview = (
                (self.code[:80] + "...") if self.code and len(self.code) > 80 else self.code
            )
            return f"Turn {self.turn_index} [EXEC]: {status}\nCode: {code_preview}"

    def full_content(self) -> str:
        """Return full content for deep analysis."""
        if self.turn_type == "llm":
            lines = ["## LLM Turn", ""]
            lines.append(f"**Model**: {self.model}")
            if self.token_counts:
                lines.append(f"**Tokens**: {self.token_counts}")
            lines.append("")
            lines.append("### Input Messages")
            for i, msg in enumerate(self.messages or []):
                lines.append(f"**[{i}] {msg.get('role', 'unknown').upper()}**")
                lines.append("```")
                lines.append(msg.get("content", ""))
                lines.append("```")
                lines.append("")
            if self.reasoning_content:
                lines.append("### Reasoning")
                lines.append("```")
                lines.append(self.reasoning_content)
                lines.append("```")
                lines.append("")
            lines.append("### LLM Response")
            lines.append("```python")
            lines.append(self.response or "")
            lines.append("```")
            if self.tool_calls:
                lines.append("")
                lines.append("### Tool Calls")
                for tc in self.tool_calls:
                    lines.append(f"- **{tc.get('function_name', 'unknown')}**")
                    lines.append(f"  ```json\n  {tc.get('arguments', '{}')}\n  ```")
            return "\n".join(lines)
        else:
            lines = ["## Execution Turn", ""]
            lines.append("### Code Executed")
            lines.append("```python")
            lines.append(self.code or "")
            lines.append("```")
            if self.stdout:
                lines.append("")
                lines.append("### stdout")
                lines.append("```")
                lines.append(self.stdout)
                lines.append("```")
            if self.error:
                lines.append("")
                error_header = "### ERROR"
                if self.error_type:
                    error_header += f" ({self.error_type})"
                lines.append(error_header)
                lines.append("```")
                lines.append(self.error)
                lines.append("```")
            if self.returned_value is not None:
                lines.append("")
                lines.append("### Returned Value")
                lines.append("```json")
                lines.append(json.dumps(self.returned_value, indent=2, default=str))
                lines.append("```")
            return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a structured dict representation."""
        data: dict[str, Any] = {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "turn_type": self.turn_type,
            "duration_ms": self.duration_ms,
        }
        if self.turn_type == "llm":
            data.update(
                {
                    "messages": self.messages or [],
                    "response": self.response,
                    "model": self.model,
                    "token_counts": self.token_counts,
                    "tool_calls": self.tool_calls or [],
                    "reasoning_content": self.reasoning_content,
                }
            )
        else:
            data.update(
                {
                    "code": self.code,
                    "stdout": self.stdout,
                    "error": self.error,
                    "error_type": self.error_type,
                    "returned_value": self.returned_value,
                }
            )
        return data


@dataclass
class SessionData:
    """Structured session data for programmatic use."""

    session: SessionSummary
    turns: list[TurnInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass
class ErrorInfo:
    """Information about an error in the trace."""

    session_id: str
    turn_index: int | None
    error_category: str  # "execution_error", "status_error", "validation_error"
    error_message: str
    context: str  # Brief context about where error occurred
    error_type: str | None = None  # Exception type (e.g., "_ReturnResultSignal", "NameError")
    # Additional fields for validation errors
    span_id: str | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None

    def __str__(self) -> str:
        if self.error_category == "validation_error":
            # Format for validation errors: show span_id and tool name
            span_label = self.span_id[:6] if self.span_id else "?"
            return f"[{span_label}] {self.tool_name}: {self.error_type} - {self.error_message}"
        turn_info = f", turn {self.turn_index}" if self.turn_index is not None else ""
        # Show error_type if available, otherwise fall back to error_category
        type_label = self.error_type if self.error_type else self.error_category
        return f"[{self.session_id}{turn_info}] {type_label}: {self.error_message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "context": self.context,
            "error_type": self.error_type,
            "span_id": self.span_id,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
        }


@dataclass
class SearchResult:
    """Result of searching trace content."""

    session_id: str
    turn_index: int
    turn_type: str
    location: str  # "response", "message", "code", "stdout", "error"
    match_text: str  # The matched text with context

    def __str__(self) -> str:
        return (
            f"[{self.session_id}, turn {self.turn_index}] {self.location}: ...{self.match_text}..."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "turn_type": self.turn_type,
            "location": self.location,
            "match_text": self.match_text,
        }


@dataclass
class SearchMatches:
    """Structured search results for programmatic use."""

    pattern: str
    match_count: int
    matches: list[SearchResult]
    by_location: dict[str, int]

    def __str__(self) -> str:
        if not self.matches:
            return f"No matches found for pattern: {self.pattern}"
        lines = [f"Found {self.match_count} match(es) for '{self.pattern}':", ""]
        lines.append("By location:")
        for loc, count in sorted(self.by_location.items(), key=lambda x: -x[1]):
            lines.append(f"  • {loc}: {count}")
        lines.append("")
        lines.append("Matches:")
        for match in self.matches[:10]:  # Show first 10
            lines.append(f"  • {match}")
        if len(self.matches) > 10:
            lines.append(f"  ... and {len(self.matches) - 10} more")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        """Allow `if search_results:` to check if any matches found."""
        return self.match_count > 0


@dataclass
class TimelineEvent:
    """A single event in the execution timeline."""

    time_ns: int
    span_id: str
    event_type: str  # "AGENT_START", "LLM", "EXEC", "AGENT_END"
    summary: str

    def __str__(self) -> str:
        return f"[{self.event_type}] {self.span_id}: {self.summary}"


@dataclass
class TimelineData:
    """Structured timeline of trace execution."""

    total_events: int
    max_events: int
    events: list[TimelineEvent]

    def __str__(self) -> str:
        lines = [f"Timeline: {len(self.events)} of {self.total_events} events", ""]
        for event in self.events:
            lines.append(f"  {event}")
        if self.total_events > self.max_events:
            lines.append(f"  ... and {self.total_events - self.max_events} more events")
        return "\n".join(lines)


@dataclass
class OverviewStats:
    """Statistics from trace overview."""

    duration_ms: float
    session_count: int
    turn_count: int
    runtime_errors: int
    eval_passed: bool | None  # None if no eval data

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "session_count": self.session_count,
            "turn_count": self.turn_count,
            "runtime_errors": self.runtime_errors,
            "eval_passed": self.eval_passed,
        }


@dataclass
class RootSessionInfo:
    """Info about the root/main session."""

    agent_name: str
    method_name: str
    session_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "method_name": self.method_name,
            "session_id": self.session_id,
        }


@dataclass
class OverviewData:
    """Structured overview of the trace."""

    trace_file: str
    root: RootSessionInfo
    stats: OverviewStats
    sessions: list[SessionSummary]
    eval_result: EvalContextData | None
    benchmark_context: str | None
    call_graph: list[dict[str, Any]] | dict[str, Any]  # Hierarchical agent call graph

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_file": self.trace_file,
            "root": self.root.to_dict(),
            "stats": self.stats.to_dict(),
            "sessions": [s.to_dict() for s in self.sessions],
            "eval_result": self.eval_result.to_dict() if self.eval_result else None,
            "benchmark_context": self.benchmark_context,
            "call_graph": self.call_graph,
        }


@dataclass
class ScoreDetail:
    """Detail for a single scorer in evaluation."""

    score: float
    passed: bool
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
        }


@dataclass
class EvalContextData:
    """Evaluation context data extracted from trace spans.

    This replaces the dict-based eval_result for type safety.
    """

    test_id: str | None = None
    passed: bool | None = None
    weighted_score: float | None = None
    model: str | None = None
    agent_class: str | None = None
    method: str | None = None
    scores: dict[str, ScoreDetail] = field(default_factory=dict)
    input: Any = None
    expected: Any = None  # expected_output from task
    error: str | None = None

    # Additional eval context attributes that may be present
    task_description: str | None = None
    output: Any = None
    trajectory: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalContextData:
        """Construct EvalContextData from a dictionary, handling nested structures.

        Args:
            data: Dictionary containing eval context data. Can be nested with
                  'eval_result' key or flat.

        Returns:
            Properly constructed EvalContextData with nested ScoreDetail objects.
        """
        # Extract from nested structure if present (e.g., from OTel spans)
        # But if 'test_id' is at top level, it's already flat (e.g., from .noo-eval.jsonl)
        if "eval_result" in data and "test_id" not in data:
            data = data["eval_result"]

        # Get valid field names
        from dataclasses import fields as dataclass_fields

        valid_fields = {f.name for f in dataclass_fields(cls)}

        # Filter to valid fields, excluding scores (we'll handle it separately)
        filtered_data = {k: v for k, v in data.items() if k in valid_fields and k != "scores"}

        # Handle scores - convert nested dicts to ScoreDetail objects
        if "scores" in data and isinstance(data["scores"], dict):
            scores_dict = {}
            for name, score_data in data["scores"].items():
                if isinstance(score_data, ScoreDetail):
                    # Already a ScoreDetail
                    scores_dict[name] = score_data
                elif isinstance(score_data, dict):
                    # Convert dict to ScoreDetail
                    scores_dict[name] = ScoreDetail(
                        score=score_data.get("score", 0.0),
                        passed=score_data.get("passed", False),
                        reasoning=score_data.get("reasoning"),
                    )
            filtered_data["scores"] = scores_dict

        # Normalize actual_output: unpack nested structure into separate fields
        actual_output = data.get("actual_output")
        if actual_output is not None:
            if isinstance(actual_output, dict):
                # Extract trajectory if present
                if "trajectory" in actual_output and filtered_data.get("trajectory") is None:
                    filtered_data["trajectory"] = actual_output["trajectory"]

                # Extract output from nested structure
                if filtered_data.get("output") is None:
                    if "output" in actual_output:
                        nested_output = actual_output["output"]
                        if isinstance(nested_output, dict):
                            filtered_data["output"] = (
                                nested_output.get("result")
                                or nested_output.get("response")
                                or nested_output
                            )
                        else:
                            filtered_data["output"] = nested_output
                    elif "result" in actual_output:
                        filtered_data["output"] = actual_output["result"]
                    elif "response" in actual_output:
                        filtered_data["output"] = actual_output["response"]
            elif filtered_data.get("output") is None:
                filtered_data["output"] = actual_output

        # Also normalize if 'output' itself is a nested dict (e.g., from eval files)
        # This handles the case where the eval file has output={'trajectory': ..., 'output': ...}
        output_val = filtered_data.get("output")
        if isinstance(output_val, dict) and ("trajectory" in output_val or "output" in output_val):
            # Extract trajectory if present and not already set
            if "trajectory" in output_val and filtered_data.get("trajectory") is None:
                filtered_data["trajectory"] = output_val["trajectory"]

            # Extract the actual output from nested structure
            if "output" in output_val:
                nested_output = output_val["output"]
                if isinstance(nested_output, dict):
                    filtered_data["output"] = (
                        nested_output.get("result")
                        or nested_output.get("response")
                        or nested_output
                    )
                else:
                    filtered_data["output"] = nested_output
            elif "result" in output_val:
                filtered_data["output"] = output_val["result"]
            elif "response" in output_val:
                filtered_data["output"] = output_val["response"]

        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "passed": self.passed,
            "weighted_score": self.weighted_score,
            "model": self.model,
            "agent_class": self.agent_class,
            "method": self.method,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "input": self.input,
            "expected": self.expected,
            "error": self.error,
            "task_description": self.task_description,
            "output": self.output,
            "trajectory": self.trajectory,
        }


# =============================================================================
# TraceExplorer Class
# =============================================================================


class TraceExplorer:
    """Programmatic interface for exploring .jsonl files.

    Designed for agent-driven analysis - an AI agent uses these methods to
    navigate large agent traces and perform root cause analysis.

    # TraceExplorer API Guide

    ## Quick Start
    Use these methods to investigate agent traces and find root causes.

    ## Truncation Format
    Values are formatted with `pprint(max_length=N, max_string=M, max_depth=D)`.
    Truncated content shows: `'text'+42` (42 more chars), `[...]+5` (5 more items).

    ## Core Methods

    ### 1. get_overview(concise=True)
    Start here. Shows the call graph with inputs, outputs, and errors.
    - concise=True: Compact view with truncated I/O (~500 chars max)
    - concise=False: Full details with 10-20x more content

    ### 2. get_session(session_id, concise=True)
    Drill into a specific session to see all turns.
    - session_id: 6-character ID from call graph (e.g., '278a10')
    - concise=True: Turn summaries only (one line per turn)
    - concise=False: Full turn content with messages and tool calls
    - Note: A session can have 0 turns if the agent method only orchestrates
      child sessions or if no generation/execution spans are linked.

    ### 2b. get_session_list()
    Get a list of all sessions as structured summaries.

    ### 3. get_turn(session_id, turn_index)
    Get full details for a specific turn with self-documenting headers.
    - **LLM turns**: Shows context window → LLM output → execution result
    - **Execution turns**: Includes LLM context from the related turn:
      - Standard flow: Context from preceding LLM turn (turn N-1)
      - Prefill flow: Context from following LLM turn (turn N+1)
    - Headers explain what type of turn and where context came from
    - Use this to understand why the LLM made a specific decision

    ### 4. get_errors()
    List all errors in the trace with context.
    - Shows error chain for failed sessions
    - Points to specific turns where errors occurred

    ### 5. get_eval_context()
    See evaluation inputs, expected outputs, and scorer results.
    - Use this to understand why a trace failed evaluation
    - Requires trace to have an 'eval' span with eval.* attributes
    - Returns help message if no eval data available

    ### 6. search(pattern)
    Find occurrences of a pattern across all trace content.
    - Searches messages, code, stdout, and responses
    - Returns matches with session/turn location

    ### 7. compare(other_trace) / TraceExplorer.diff(trace1, trace2)
    Compare two traces for regression analysis.
    - Side-by-side call graph comparison
    - Finds first divergence point
    - Detects prompt expression path differences
    - Useful for MR vs main branch comparison

    ### 8. get_timeline(max_events=50)
    Chronological timeline of events across all sessions.
    - Shows spans in order with timestamps
    - Useful for understanding execution flow

    ### 9. find_first_error()
    Jump directly to the first error in the trace.
    - Shows error details with navigation hints
    - Saves time when debugging failures

    ### 10. get_raw_span(span_id)
    ### 11. *_data() structured outputs
    Use `get_overview_data()`, `get_session_data()`, `get_turn_data()`,
    `get_errors_data()`, `get_eval_context_data()`, `search_data()`,
    `get_timeline_data()`, `find_first_error_data()`, and `diff_data()`
    for programmatic JSON-friendly output.

    Access raw span data as JSON.
    - Useful for debugging trace structure
    - Access attributes not exposed by high-level API

    ## Navigation Pattern

    1. Start with `get_overview()` to see the big picture
    2. Find a failed session in the call graph
    3. Use `get_session(id)` to see turn-by-turn execution
    4. Use `get_turn(id, n)` to see exact LLM context at turn n
    5. Use `get_eval_context()` if it's an eval failure
    6. Use `get_errors()` to see all errors at once
    7. Use `compare(other)` for regression analysis

    ## Status Labels
    - [OK] - Session/turn completed successfully
    - [ERR] - Session/turn had a runtime error
    - [PASS] - Evaluation passed
    - [FAIL] - Evaluation failed
    """

    def __init__(
        self,
        sessions: list[AgentSession],
        trace_file: str,
        eval_result: EvalContextData | None = None,
        benchmark_context: str | None = None,
        raw_spans: list[dict[str, Any]] | None = None,
        viewer_url: str | None = None,
        viewer_session_id: str | None = None,
    ):
        self._sessions = sessions
        self._trace_file = trace_file
        self._eval_result = eval_result
        self._benchmark_context = benchmark_context
        self._raw_spans = raw_spans or []
        self._viewer_url = viewer_url  # set when loaded via from_viewer
        # Viewer session id that loaded this trace; needed to emit valid
        # viewer-mode CLI nav hints (the CLI requires --session-id with --viewer).
        self._viewer_session_id = viewer_session_id

        # Build flat session index for quick lookup
        # This is the unified tree structure with depth and parent-child relationships
        self._all_sessions = _get_all_sessions(sessions)
        self._session_by_id = {s.session_id: s for s in self._all_sessions}

    def _find_session(self, session_id: str) -> AgentSession | None:
        """Find a session by ID (supports both short 6-char and full IDs)."""
        # Try exact match first
        if session_id in self._session_by_id:
            return self._session_by_id[session_id]

        # Try prefix match (for 6-char short IDs)
        for sid, session in self._session_by_id.items():
            if sid.startswith(session_id) or session_id.startswith(sid[:6]):
                return session

        return None

    def _source_prefix(self) -> str:
        """Return the base trace-explorer CLI invocation for this trace source.

        In viewer mode the CLI requires ``--session-id`` alongside ``--viewer``
        (see ``_async_main``), so the hint includes the session id that loaded
        this trace to stay valid against the parser's own argument validation.
        """
        if self._viewer_url:
            prefix = f"trace-explorer --viewer {shlex.quote(self._viewer_url)}"
            if self._viewer_session_id:
                prefix += f" --session-id {shlex.quote(self._viewer_session_id)}"
            return prefix
        return f"trace-explorer {shlex.quote(str(self._trace_file))}"

    def _nav_hint(self, session_id: str) -> str:
        """Return a navigation hint appropriate for the current context.

        In CLI mode: emits a trace-explorer shell command.
        In API mode: emits a Python method call.
        """
        if not _cli_mode.get():
            return f"await self.get_session('{session_id}')"
        return f"{self._source_prefix()} -s '{session_id}'"

    def _nav_hint_turn(self, session_id: str, turn_idx: int) -> str:
        """Drill-into-turn hint, context-aware (CLI vs API)."""
        if not _cli_mode.get():
            return f"await self.get_turn('{session_id}', {turn_idx})"
        return f"{self._source_prefix()} -s '{session_id}' -t {turn_idx}"

    def _nav_hint_cmd(self, method: str, flag: str) -> str:
        """Hint for a top-level command (e.g. get_errors / --errors).

        method: the Python call expression, e.g. 'get_errors()' (already includes parens)
        flag: the CLI flag, e.g. '--errors'
        """
        if not _cli_mode.get():
            return f"await self.{method}"
        return f"{self._source_prefix()} {flag}"

    # =========================================================================
    # Properties (for internal access - LLM should use get_* methods instead)
    # =========================================================================

    @property
    def sessions(self) -> list[AgentSession]:
        """Access sessions (prefer get_overview() or get_session_list() for analysis)."""
        return self._sessions

    @property
    def trace_file(self) -> str:
        """Path to the trace file."""
        return self._trace_file

    @property
    def trace_path(self) -> str:
        """Alias for trace_file (for compatibility)."""
        return self._trace_file

    @property
    def eval_result(self) -> EvalContextData | None:
        """Eval result data (prefer get_eval_context() for analysis)."""
        return self._eval_result

    @property
    def benchmark_context(self) -> str | None:
        """Additional benchmark context."""
        return self._benchmark_context

    async def help(self) -> str:
        """Return the usage guide from the class docstring."""
        import inspect

        return inspect.cleandoc(self.__doc__ or "No documentation available.")

    async def get_session_list(self) -> list[SessionSummary]:
        """Return a list of session summaries (root-first, including children)."""
        return [self._build_session_summary(s) for s in self._all_sessions]

    async def get_span_id(self, session_id: str, turn_index: int) -> str | None:
        """Get the span ID for a specific turn.

        Args:
            session_id: The 6-character session ID
            turn_index: The turn index (0, 1, 2, ...)

        Returns:
            The full hex span ID (e.g., "c2f8ead1ae87273f"), or None if not found
        """
        session = self._find_session(session_id)
        if not session:
            return None

        if turn_index < 0 or turn_index >= len(session.turns):
            return None

        return session.turns[turn_index].span_id or None

    @classmethod
    async def from_file(
        cls,
        trace_path: str | Path,
        eval_result: EvalContextData | None = None,
        benchmark_context: str | None = None,
        root_generation_index: int | None = None,
    ) -> TraceExplorer:
        """Load a trace from a .jsonl file.

        Reads the file once and builds a unified session tree with:
        - Parent-child relationships
        - Depth at each level
        - Full span_id for correlation
        - Turns with messages and code executions

        Raises:
            ValueError: If the file is an eval-results file (e.g. .noo-eval.jsonl)
                or otherwise does not contain OTLP execution spans.
        """
        trace_path = Path(trace_path)

        # Reject eval-result files by name. Current runs write ``.noo-eval.jsonl``
        # (see eval_pipeline.experiment_writer); legacy runs used ``.006eval.``.
        # Both contain evaluation records, not agent execution traces.
        if ".noo-eval." in trace_path.name or ".006eval." in trace_path.name:
            raise ValueError(
                f"Cannot load eval file as trace: {trace_path.name}\n"
                "Use .jsonl trace files instead. Eval files contain evaluation "
                "results, not agent execution traces."
            )

        import asyncio

        def _load():
            # Load spans once
            raw_spans = _load_spans(trace_path)

            # Structural guard for eval files that slipped past the name check
            # (e.g. renamed). Eval records are JSONL objects tagged with a
            # top-level ``_type`` of metadata/result/completion (see
            # eval_pipeline.eval_types) and carry no OTLP span identity, so a
            # normalized "span" from such a file has neither a name nor a span_id.
            if raw_spans and not raw_spans[0]["name"] and not raw_spans[0]["span_id"]:
                raise ValueError(
                    f"File does not appear to be a valid trace file: {trace_path.name}\n"
                    "Expected JSONL with OTLP span objects (each with a 'name' and "
                    "'span_id'). This may be an eval results file or another format."
                )

            # Build unified session tree from spans
            prev_root_index = get_root_generation_index()
            try:
                if root_generation_index is not None:
                    set_root_generation_index(root_generation_index)
                sessions = _parse_trace_from_spans(raw_spans)
            finally:
                set_root_generation_index(prev_root_index)

            # Extract eval result from trace file if not provided
            er = eval_result
            if er is None:
                er = cls._extract_eval_from_spans(raw_spans)

            return cls(sessions, str(trace_path), er, benchmark_context, raw_spans)

        return await asyncio.to_thread(_load)

    @classmethod
    def from_otlp_spans(
        cls,
        otlp_spans: list[dict[str, Any]],
        *,
        trace_file: str,
        viewer_url: str | None = None,
        extract_eval: bool = True,
        quiet: bool = True,
    ) -> TraceExplorer:
        """Build a TraceExplorer from raw OTLP span dicts.

        Encapsulates OTLP span normalization, trace parsing, and (optionally)
        eval-context extraction. Warning suppression is controlled per call via
        ``quiet`` and scoped with a ContextVar token, so callers never toggle the
        process-wide quiet mode themselves — avoiding a race where concurrent
        request handlers could silence or unsilence each other's work.

        This is a synchronous, CPU-bound builder; callers running inside an event
        loop should wrap it in ``asyncio.to_thread``.

        Args:
            otlp_spans: Raw OTLP span dicts (as returned by the viewer store).
            trace_file: Identifier for the trace source (e.g. ``viewer://<id>``).
            viewer_url: Base viewer URL, if the spans came from a viewer.
            extract_eval: Whether to extract eval-context data from the spans.
            quiet: Suppress parser warnings for the duration of this build.
        """
        raw_spans = [_normalize_otlp_span(s) for s in otlp_spans]

        token = _quiet_mode.set(quiet)
        try:
            sessions = _parse_trace_from_spans(raw_spans)
            eval_result = cls._extract_eval_from_spans(raw_spans) if extract_eval else None
        finally:
            _quiet_mode.reset(token)

        return cls(
            sessions=sessions,
            trace_file=trace_file,
            eval_result=eval_result,
            raw_spans=raw_spans,
            viewer_url=viewer_url,
        )

    @classmethod
    async def from_viewer(
        cls,
        base_url: str,
        session_id: str,
        eval_result: EvalContextData | None = None,
        root_generation_index: int | None = None,
    ) -> TraceExplorer:
        """Load a trace from the viewer API.

        Fetches all spans for the given session via paginated API calls
        and builds a TraceExplorer instance.

        Args:
            base_url: Viewer base URL (e.g., "http://localhost:5001")
            session_id: Session ID to load
            eval_result: Optional pre-computed eval result
            root_generation_index: Select Nth root generation (0-based)

        Raises:
            ConnectionError: If the viewer is unreachable
            ValueError: If the session is not found or response is invalid
        """
        import asyncio
        import urllib.parse

        import httpx

        base_url = base_url.rstrip("/")
        encoded_sid = urllib.parse.quote(session_id, safe="")
        all_spans: list[dict[str, Any]] = []
        offset = 0
        page_size = 500

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                url = f"{base_url}/api/trace?session_id={encoded_sid}&limit={page_size}&offset={offset}"
                try:
                    resp = await client.get(url)
                    if resp.status_code == 404:
                        raise ValueError(f"Session not found: {session_id}")
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.ConnectError as e:
                    raise ConnectionError(f"Cannot reach viewer at {base_url}: {e}") from e
                except httpx.HTTPStatusError as e:
                    raise ValueError(
                        f"Viewer returned HTTP {e.response.status_code}: {e.response.reason_phrase}"
                    ) from e

                raw_spans = data.get("events", [])
                all_spans.extend(_normalize_otlp_span(s) for s in raw_spans)

                if not data.get("has_more", False):
                    break
                offset += page_size

        if not all_spans:
            raise ValueError(f"No spans found for session: {session_id}")

        # Offload CPU-bound parsing to a thread
        def _build():
            prev_root_index = get_root_generation_index()
            try:
                if root_generation_index is not None:
                    set_root_generation_index(root_generation_index)
                sessions = _parse_trace_from_spans(all_spans)
            finally:
                set_root_generation_index(prev_root_index)

            er = eval_result
            if er is None:
                er = cls._extract_eval_from_spans(all_spans)

            return cls(
                sessions=sessions,
                trace_file=f"viewer://{session_id}",
                eval_result=er,
                raw_spans=all_spans,
                viewer_url=base_url,
                viewer_session_id=session_id,
            )

        return await asyncio.to_thread(_build)

    @classmethod
    async def load_experiment_sessions(
        cls,
        base_url: str,
        experiment_id: str,
        root_generation_index: int | None = None,
    ) -> dict[str, TraceExplorer]:
        """Bulk-load all sessions from an experiment in a single HTTP request.

        Uses ``GET /api/experiment/{id}/traces`` to fetch all spans in one call,
        avoiding the N+1 request problem of calling ``from_viewer()`` per session.

        Spans are grouped by OTel ``trace_id`` so that sub-agent spans stored
        under a different viewer ``session.id`` (but sharing the same OTel trace)
        are merged into one TraceExplorer per test case. If a session's spans
        carry no ``trace_id``, that session is kept as its own group.

        Args:
            base_url: Viewer base URL (e.g., "http://localhost:5001")
            experiment_id: Experiment name / ID
            root_generation_index: Select Nth root generation (0-based)

        Returns:
            Dict mapping viewer session_id → TraceExplorer, one entry per
            distinct OTel trace. The key is the session_id of the first viewer
            session encountered for each trace (by viewer ordering, typically
            the root agent session).

        Raises:
            ConnectionError: If the viewer is unreachable
            ValueError: If the experiment is not found, has no sessions, or the
                response cannot be parsed
        """
        import asyncio
        import urllib.parse

        import httpx

        base_url = base_url.rstrip("/")
        encoded_exp = urllib.parse.quote(experiment_id, safe="")
        url = f"{base_url}/api/experiment/{encoded_exp}/traces"

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                resp = await client.get(url)
                if resp.status_code == 404:
                    raise ValueError(f"Experiment not found: {experiment_id}")
                resp.raise_for_status()
                data = resp.json()
            except httpx.ConnectError as e:
                raise ConnectionError(f"Cannot reach viewer at {base_url}: {e}") from e
            except httpx.HTTPStatusError as e:
                raise ValueError(
                    f"Viewer returned HTTP {e.response.status_code}: {e.response.reason_phrase}"
                ) from e
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Viewer returned invalid JSON for experiment {experiment_id}: {e}"
                ) from e

        raw_sessions = data.get("sessions", [])
        if not raw_sessions:
            raise ValueError(f"No sessions found for experiment: {experiment_id}")

        def _build():
            # Group all spans by trace_id so sub-agent spans (different viewer session.id
            # but same OTel trace) are merged into one TraceExplorer per test case.
            trace_id_to_spans: dict[str, list[dict[str, Any]]] = {}
            trace_id_to_primary_session: dict[str, str] = {}

            for viewer_session in raw_sessions:
                session_id = viewer_session.get("session_id", "")
                raw_spans = viewer_session.get("spans", [])
                normalized = [_normalize_otlp_span(s) for s in raw_spans]

                trace_id = ""
                for span in normalized:
                    tid = span.get("trace_id", "")
                    if tid:
                        trace_id = tid
                        break

                if not trace_id:
                    trace_id = session_id

                if trace_id not in trace_id_to_spans:
                    trace_id_to_spans[trace_id] = []
                    trace_id_to_primary_session[trace_id] = session_id

                trace_id_to_spans[trace_id].extend(normalized)

            # Build one TraceExplorer per trace_id group
            result: dict[str, TraceExplorer] = {}
            prev_root_index = get_root_generation_index()
            try:
                if root_generation_index is not None:
                    set_root_generation_index(root_generation_index)

                for trace_id, spans in trace_id_to_spans.items():
                    primary_session_id = trace_id_to_primary_session[trace_id]
                    sessions = _parse_trace_from_spans(spans)
                    eval_result = cls._extract_eval_from_spans(spans)
                    explorer = cls(
                        sessions=sessions,
                        trace_file=f"viewer://{primary_session_id}",
                        eval_result=eval_result,
                        raw_spans=spans,
                        viewer_url=base_url,
                        viewer_session_id=primary_session_id,
                    )
                    result[primary_session_id] = explorer
            finally:
                set_root_generation_index(prev_root_index)

            return result

        return await asyncio.to_thread(_build)

    @staticmethod
    def _extract_eval_from_spans(raw_spans: list[dict]) -> EvalContextData | None:
        """Extract eval result from an 'eval' span in the trace.

        Looks for a span with name='eval' and extracts attributes like:
        - eval.test_id -> test_id
        - eval.passed -> passed
        - eval.scorer.X.score -> scores[X].score
        - eval.scorer.X.passed -> scores[X].passed
        - eval.scorer.X.reasoning -> scores[X].reasoning
        - eval.expected_output -> expected
        - eval.actual_output -> output
        - eval.input -> input
        - eval.task_description -> task_description
        """

        def _normalize_output(eval_data: EvalContextData, value: Any) -> Any:
            if isinstance(value, dict):
                # Check if there's a nested "output" dict with "response" or "result"
                output_dict = value.get("output")
                if output_dict and isinstance(output_dict, dict):
                    eval_data.output = (
                        output_dict.get("response") or output_dict.get("result") or output_dict
                    )
                else:
                    # Fallback to top-level keys
                    eval_data.output = (
                        value.get("response") or value.get("result") or value.get("output")
                    )
                eval_data.trajectory = value.get("trajectory", None)
            else:
                eval_data.output = value

        for span in raw_spans:
            if span.get("name") != "eval":
                continue

            attrs = span.get("attributes", {})
            if not attrs:
                continue

            # Build eval context data
            eval_data = EvalContextData()
            actual_output_raw: Any = None  # Track actual_output for normalization

            # Extract top-level eval attributes
            for key, value in attrs.items():
                if key == "eval.test_id":
                    eval_data.test_id = value
                elif key == "eval.passed":
                    eval_data.passed = value
                elif key == "eval.weighted_score":
                    eval_data.weighted_score = value
                elif key == "eval.model":
                    eval_data.model = value
                elif key == "eval.agent_class":
                    eval_data.agent_class = value
                elif key == "eval.method":
                    eval_data.method = value
                elif key == "eval.expected_output" or key == "eval.expected":
                    try:
                        eval_data.expected = json.loads(value)
                    except json.JSONDecodeError:
                        eval_data.expected = value
                elif key == "eval.actual_output" or key == "eval.output":
                    # Parse actual_output for later normalization into output
                    try:
                        actual_output_raw = json.loads(value)
                    except json.JSONDecodeError:
                        actual_output_raw = value
                    _normalize_output(eval_data, actual_output_raw)
                elif key == "eval.input" or key == "eval.input_messages":
                    try:
                        eval_data.input = json.loads(value)
                    except json.JSONDecodeError:
                        eval_data.input = value
                elif key == "eval.task_description":
                    eval_data.task_description = value

            # Extract scorer data
            scorer_data: dict[str, dict[str, Any]] = {}
            for key, value in attrs.items():
                if key.startswith("eval.scorer."):
                    # Parse: eval.scorer.<scorer_name>.<field>
                    parts = key.split(".")
                    if len(parts) >= 4:
                        scorer_name = parts[2]
                        field = parts[3]
                        if scorer_name not in scorer_data:
                            scorer_data[scorer_name] = {}
                        scorer_data[scorer_name][field] = value

            # Convert scorer dicts to ScoreDetail objects
            for scorer_name, fields in scorer_data.items():
                eval_data.scores[scorer_name] = ScoreDetail(
                    score=fields.get("score", 0.0),
                    passed=fields.get("passed", False),
                    reasoning=fields.get("reasoning"),
                )

            # Only return if we found at least some eval data
            if eval_data.test_id or eval_data.passed is not None:
                return eval_data

        return None

    # =========================================================================
    # Overview Methods
    # =========================================================================

    async def get_overview(self, *, concise: bool = True) -> str:
        """Get a high-level summary of the trace.

        Args:
            concise: If True, use compact formatting with smart truncation. If False, show full details.

        Returns a text summary showing:
        - Status (PASSED/FAILED) in title
        - Duration, session count, turn count, error count
        - Call graph with IN/OUT for each method
        - Eval result summary for root
        - Navigation hints for drilling down
        """
        if not self.sessions:
            return "No sessions found in trace."

        root = self.sessions[0]
        lines = []

        # Title with pass/fail status
        status_icon = ""
        if self.eval_result:
            passed = self.eval_result.passed
            if passed is not None:
                status_icon = "  [PASSED]" if passed else "  [FAILED]"

        # Use test_id from eval if available, otherwise use agent.method name
        title = f"{root.agent_name}.{root.method_name}()"
        if self.eval_result:
            test_id = self.eval_result.test_id
            if test_id:
                title = f"{test_id} - {root.agent_name}.{root.method_name}()"

        lines.append(f"# {title}{status_icon}")
        lines.append("")

        # Stats line
        total_duration = sum(s.duration_ms for s in self._all_sessions)
        total_turns = sum(len(s.turns) for s in self._all_sessions)
        runtime_errors = sum(1 for s in self._all_sessions if s.status != "OK")
        session_count = len(self._all_sessions)

        # Always use consistent format with 1 decimal place
        if concise:
            duration_str = (
                f"{total_duration / 1000:.1f}s"
                if total_duration >= 1000
                else f"{total_duration:.1f}ms"
            )
        else:
            duration_str = f"{total_duration:.1f}ms"

        # Build stats parts
        stats_parts = [
            f"Duration: {duration_str}",
            f"Sessions: {session_count}",
            f"Turns: {total_turns}",
        ]

        # Show runtime errors if any
        if runtime_errors > 0:
            stats_parts.append(f"Runtime Errors: {runtime_errors}")

        # Show eval status if available
        if self.eval_result:
            eval_passed = self.eval_result.passed
            if eval_passed is not None:
                eval_status = "PASSED" if eval_passed else "FAILED"
                stats_parts.append(f"Eval: {eval_status}")

        lines.append(" | ".join(stats_parts))
        lines.append("")

        # Call graph with IN/OUT
        lines.append("## Call Graph")
        lines.extend(self._format_call_graph(concise=concise))
        lines.append("")

        # Navigation hints
        lines.append("## Navigation")
        if concise:
            lines.append(
                f"→ {self._nav_hint_cmd('get_overview(concise=False)', '-v')} - Full I/O details"
            )
        lines.append(f"→ {self._nav_hint(root.session_id[:6])} - Root session details")
        if self.eval_result:
            lines.append(
                f"→ {self._nav_hint_cmd('get_eval_context()', '--eval')} - Full eval comparison"
            )
        if runtime_errors > 0:
            lines.append(f"→ {self._nav_hint_cmd('get_errors()', '--errors')} - All errors")

        return "\n".join(lines)

    async def get_overview_data(self) -> OverviewData | None:
        """Structured overview data for programmatic use.

        Returns:
            OverviewData with trace overview, or None if no sessions found.
        """
        if not self.sessions:
            return None

        root = self.sessions[0]
        total_duration = sum(s.duration_ms for s in self._all_sessions)
        total_turns = sum(len(s.turns) for s in self._all_sessions)
        runtime_errors = sum(1 for s in self._all_sessions if s.status != "OK")

        return OverviewData(
            trace_file=self.trace_file,
            root=RootSessionInfo(
                agent_name=root.agent_name,
                method_name=root.method_name,
                session_id=root.session_id,
            ),
            stats=OverviewStats(
                duration_ms=total_duration,
                session_count=len(self._all_sessions),
                turn_count=total_turns,
                runtime_errors=runtime_errors,
                eval_passed=self.eval_result.passed if self.eval_result else None,
            ),
            call_graph=self._build_call_graph_data(),
            sessions=[self._build_session_summary(s) for s in self._all_sessions],
            eval_result=self.eval_result,
            benchmark_context=self.benchmark_context,
        )

    def _get_session_error(self, session: AgentSession) -> str | None:
        """Extract error message from a session."""
        # Check span-level error message first (most reliable for generation errors)
        if session.error_message:
            return session.error_message

        # Check if result contains an error message
        if isinstance(session.result, dict) and "error" in session.result:
            return session.result["error"]

        # Check execution turns for errors
        error_turns = session.get_error_turns()
        if error_turns:
            return error_turns[-1].error

        # Look for error in ancestor sessions that mentions this method
        for ancestor in self._all_sessions:
            if isinstance(ancestor.result, dict) and "error" in ancestor.result:
                ancestor_error = ancestor.result["error"]
                if session.method_name in ancestor_error:
                    return ancestor_error

        return None

    def _get_descendants(self, session: AgentSession) -> list[AgentSession]:
        """Get all descendants of a session."""
        descendants = []
        for child in session.children:
            descendants.append(child)
            descendants.extend(self._get_descendants(child))
        return descendants

    def _build_call_graph_data(self) -> list[dict[str, Any]]:
        """Build a structured call graph list for JSON output."""
        return [
            {
                "session_id": s.session_id,
                "full_name": s.full_name,
                "depth": s.depth,
                "status": s.status,
                "turn_count": len(s.turns),
                "duration_ms": s.duration_ms,
                "parent_session_id": s.parent_session_id,
            }
            for s in self._all_sessions
        ]

    def _format_call_graph(self, concise: bool = True) -> list[str]:
        """Format sessions as a tree-style call graph with IN/OUT details.

        Args:
            concise: If True, use compact formatting. If False, show full details with indentation.

        Produces output like (concise=True):
            RouterTestWrapper.process [45bdba] ───────────────────  6279.0ms [ERR]
              IN:  user_message='...', values=[1, 2, 3, 4, 5]
              OUT: {'agents_called': ['Transformer', 'Validator'], 'results': {dict: 2 items}}
              EVAL: [FAIL] 0.0 - Transformer output mismatch
            ├── TransformerSubAgent.transform [c4f60c] ───────────  1307.0ms [OK]
            │   IN:  [1, 2, 3, 4, 5], format='CSV and validate'
            │   OUT: 'Unsupported format: CSV and validate'  ⚠️
            └── ValidatorSubAgent.validate [973b70] ──────────────  2280.0ms [OK]
                IN:  [1, 2, 3, 4, 5]
                OUT: {'all_positive': True, ...}
        """
        lines = []

        # Build parent->children map using explicit parent_session_id
        children_map: dict[str | None, list[AgentSession]] = {}
        for session in self._all_sessions:
            parent = session.parent_session_id
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(session)

        # Sort children by start_time for chronological order
        for children in children_map.values():
            children.sort(key=lambda s: s.start_time)

        def render_session(
            session: AgentSession, prefix: str, is_last: bool, is_root: bool
        ) -> None:
            """Render a single session and its children recursively."""
            # Determine status - check both session.status and result['success'] if available
            is_ok = session.status == "OK"
            if is_ok and isinstance(session.result, dict):
                is_ok = session.result.get("success", True)
            status_label = "[OK]" if is_ok else "[ERR]"
            duration = f"{session.duration_ms:.1f}ms"
            turn_count = len(session.turns)

            # Build the session line
            method = f"{session.agent_name}.{session.method_name}"
            label = f"{method} [{session.session_id[:6]}]"

            # Tree connector
            if is_root:
                connector = ""
            elif is_last:
                connector = "└── "
            else:
                connector = "├── "

            # Calculate padding for alignment (target ~60 chars before duration)
            base_len = len(prefix) + len(connector) + len(label)
            min_dashes = 3
            target_width = 55
            dash_count = max(min_dashes, target_width - base_len)
            dashes = "─" * dash_count

            # Format: "method [id] ─── 5t  1234.0ms [OK]"
            line = (
                f"{prefix}{connector}{label} {dashes} {turn_count:>2}t {duration:>8} {status_label}"
            )
            lines.append(line)

            # Format input/output with proper tree continuation
            # Use box-drawing │ for tree continuation, plain indent for last/root
            has_children = (
                session.session_id in children_map and len(children_map[session.session_id]) > 0
            )
            if is_root and has_children:
                # Root with children: use │ to show tree continues
                io_prefix = "│ "
            elif is_root:
                # Root without children: just indent
                io_prefix = "  "
            elif is_last:
                # Last child: no continuation line
                io_prefix = "  "
            else:
                # Non-last child: show continuation
                io_prefix = "│ "
            indent = prefix + io_prefix

            # Input
            input_data = self._get_session_input(session)
            if input_data is not None:
                if concise:
                    # Concise: format as param=value, param2=value2 (no dict braces)
                    input_str = self._format_params_compact(input_data, max_str=60)
                else:
                    # Verbose: show much more detail (~10-20x more tokens)
                    input_str = _pformat(input_data, max_string=1200, max_length=100, max_depth=6)
                    input_str = " ".join(input_str.split())
                lines.append(f"{indent}IN:  {input_str}")

            # Output
            output_data = session.result
            if output_data is not None:
                output_str = self._format_value_smart(output_data, concise=concise)
                lines.append(f"{indent}OUT: {output_str}")

            # Error info for failed sessions
            if session.status != "OK":
                error_msg = self._get_session_error(session)
                # Show error if we found one
                if error_msg:
                    if concise and len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    elif not concise and len(error_msg) > 1000:
                        # Verbose: show up to 1000 chars (10x concise limit)
                        error_msg = error_msg[:1000] + "..."
                    lines.append(f"{indent}ERR: {error_msg}")

            # Eval info (only for root)
            if is_root and self.eval_result:
                passed = self.eval_result.passed
                if passed is not None:
                    eval_label = "[PASS]" if passed else "[FAIL]"
                    scores = self.eval_result.scores

                    # Find the most relevant scorer to show:
                    # - If overall failed, show first failing scorer's name + reasoning
                    # - Otherwise show first passing scorer
                    score_summary = ""
                    if scores:
                        target_scorer_name = None
                        target_scorer = None

                        if not passed:
                            # Find first failed scorer
                            for name, scorer_data in scores.items():
                                if not scorer_data.passed:
                                    target_scorer_name = name
                                    target_scorer = scorer_data
                                    break

                        if target_scorer is None:
                            # Use first scorer
                            target_scorer_name = next(iter(scores.keys())) if scores else None
                            target_scorer = (
                                scores.get(target_scorer_name) if target_scorer_name else None
                            )

                        if target_scorer:
                            score_val = target_scorer.score
                            reasoning = target_scorer.reasoning or ""

                            if concise and reasoning:
                                # Concise: show reasoning only if short and clear
                                # Skip if it's a long JSON dump or starts with complex output
                                if len(reasoning) < 80 and not reasoning.startswith(("{", "[")):
                                    score_summary = (
                                        f"{score_val} ({target_scorer_name}) - {reasoning}"
                                    )
                                else:
                                    # Too long/complex, just show score
                                    score_summary = f"{score_val} ({target_scorer_name})"
                            elif reasoning and not concise:
                                # Verbose: show more reasoning (up to 800 chars, ~10x concise limit)
                                if len(reasoning) > 800:
                                    reasoning = reasoning[:800] + "..."
                                score_summary = f"{score_val} ({target_scorer_name})\n{indent}        {reasoning}"
                            else:
                                score_summary = f"{score_val} ({target_scorer_name})"

                    lines.append(f"{indent}EVAL: {eval_label} {score_summary}")

            # Get children of this session
            children = children_map.get(session.session_id, [])
            for i, child in enumerate(children):
                child_is_last = i == len(children) - 1
                # Update prefix for children
                if is_root:
                    child_prefix = ""
                elif is_last:
                    child_prefix = prefix + "    "
                else:
                    child_prefix = prefix + "│   "
                render_session(child, child_prefix, child_is_last, is_root=False)

        # Render root sessions (those without a parent)
        root_sessions = children_map.get(None, [])
        for i, root in enumerate(root_sessions):
            is_last = i == len(root_sessions) - 1
            render_session(root, "", is_last, is_root=True)

        return lines

    def _get_session_input(self, session: AgentSession) -> dict[str, Any] | None:
        """Extract input data from session's args and kwargs.

        Parses method_signature to get real parameter names instead of arg0, arg1.

        Returns:
            Combined args and kwargs as a dict with proper param names, or None if no input.
        """
        if not session.args and not session.kwargs:
            return None

        # Parse method_signature to get parameter names
        # Format: "method_name(param1: type, param2: type, ...) -> ReturnType"
        param_names = self._parse_param_names(session.method_signature)

        # Combine args and kwargs
        result = {}

        # Add positional args with proper names
        for i, arg in enumerate(session.args):
            if i == 0 and arg is not None:
                # Skip 'self' argument if it looks like an object
                if hasattr(arg, "__dict__"):
                    continue
            # Use parsed param name if available, else fallback to arg{i}
            if i < len(param_names):
                param_name = param_names[i]
            else:
                param_name = f"arg{i}"
            result[param_name] = arg

        # Add kwargs (these already have proper names)
        result.update(session.kwargs)

        return result if result else None

    def _parse_param_names(self, method_signature: str) -> list[str]:
        """Parse parameter names from a method signature.

        Examples:
            "find_rules(question: str) -> dict" -> ["question"]
            "compute_answer(q: str, g: str, h: str = '') -> R" -> ["q", "g", "h"]
        """
        if not method_signature:
            return []

        # Find the part between ( and )
        paren_start = method_signature.find("(")
        paren_end = method_signature.rfind(")")
        if paren_start == -1 or paren_end == -1 or paren_end <= paren_start:
            return []

        params_str = method_signature[paren_start + 1 : paren_end]
        if not params_str.strip():
            return []

        # Split by comma, but be careful of nested types like dict[str, int]
        params = []
        depth = 0
        current = ""
        for char in params_str:
            if char in "[{(":
                depth += 1
                current += char
            elif char in "]})":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                params.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            params.append(current.strip())

        # Extract just the parameter name (before : or =)
        names = []
        for param in params:
            # Skip 'self' parameter
            param = param.strip()
            if param == "self" or param.startswith("self,"):
                continue
            # Extract name before : or =
            name = param.split(":")[0].split("=")[0].strip()
            if name and name != "self":
                names.append(name)

        return names

    def _format_params_compact(self, params: dict[str, Any], max_str: int = 60) -> str:
        """Format parameters compactly as param=value, param2=value2.

        More readable than {'param': value} dict format.

        TODO: Simplify when _pformat supports __class__ handling natively.
        """
        if not params:
            return ""

        parts = []
        for name, value in params.items():
            # Format the value
            if isinstance(value, str):
                # Truncate long strings
                if len(value) > max_str:
                    val_str = repr(value[:max_str]) + f"+{len(value) - max_str}"
                else:
                    val_str = repr(value)
            elif isinstance(value, dict):
                # Handle Pydantic-style dicts with __class__
                if "__class__" in value:
                    class_name = value.get("__class__", "")
                    fields = {k: v for k, v in value.items() if k != "__class__"}
                    if len(fields) <= 2:
                        inner_parts = []
                        for k, v in fields.items():
                            v_str = _pformat(v, max_string=40, max_length=3, max_depth=1)
                            v_str = " ".join(v_str.split())
                            if len(v_str) > 50:
                                v_str = v_str[:50] + "..."
                            inner_parts.append(f"{k}={v_str}")
                        val_str = f"{class_name}({', '.join(inner_parts)})"
                    else:
                        val_str = f"{class_name}(...)"
                # Show dict keys only for compactness
                elif len(value) <= 3:
                    val_str = _pformat(value, max_string=max_str, max_length=5, max_depth=2)
                    val_str = " ".join(val_str.split())
                else:
                    keys = list(value.keys())[:3]
                    val_str = (
                        "{" + ", ".join(repr(k) for k in keys) + f", ... +{len(value) - 3}" + "}"
                    )
            elif isinstance(value, (list, tuple)):
                val_str = _pformat(value, max_string=max_str, max_length=5, max_depth=2)
                val_str = " ".join(val_str.split())
            else:
                val_str = _pformat(value, max_string=max_str, max_length=5, max_depth=2)
                val_str = " ".join(val_str.split())

            parts.append(f"{name}={val_str}")

        return ", ".join(parts)

    def _format_value_smart(self, value: Any, *, concise: bool = True) -> str:
        """Format a value smartly, handling Pydantic-style dicts with __class__.

        If dict has '__class__' key, format as: ClassName(key=val, ...)
        Otherwise use standard _pformat.

        TODO: Remove this custom handling when _pformat is updated to natively
        handle __class__ keys in Pydantic-serialized dicts. See agentdoc issue #XX.
        """
        if isinstance(value, dict) and "__class__" in value:
            # Extract class name and format nicely
            class_name = value.get("__class__", "")
            # Filter out __class__ for display
            fields = {k: v for k, v in value.items() if k != "__class__"}

            if concise:
                # Format as ClassName(key=val, key2=val2)
                parts = []
                for k, v in fields.items():
                    if isinstance(v, str):
                        if len(v) > 60:
                            v_str = repr(v[:60]) + f"+{len(v) - 60}"
                        else:
                            v_str = repr(v)
                    elif isinstance(v, dict):
                        if "__class__" in v:
                            # Nested Pydantic: just show class name
                            v_str = v.get("__class__", "{...}")
                        elif len(v) <= 2:
                            v_str = _pformat(v, max_string=40, max_length=3, max_depth=1)
                            v_str = " ".join(v_str.split())
                        else:
                            v_str = "{...}"
                    elif isinstance(v, (list, tuple)):
                        if len(v) <= 2:
                            v_str = _pformat(v, max_string=40, max_length=3, max_depth=1)
                            v_str = " ".join(v_str.split())
                        else:
                            v_str = f"[{len(v)} items]"
                    else:
                        v_str = _pformat(v, max_string=60, max_length=3, max_depth=1)
                        v_str = " ".join(v_str.split())
                    parts.append(f"{k}={v_str}")
                return f"{class_name}({', '.join(parts)})"
            else:
                # Verbose: show full content (~10-20x limits) but still use class name prefix
                fields_str = _pformat(fields, max_string=1200, max_length=100, max_depth=6)
                fields_str = " ".join(fields_str.split())
                return f"{class_name}({fields_str})"

        # Non-Pydantic values: use standard formatting
        if concise:
            result = _pformat(value, max_string=80, max_length=5, max_depth=2)
        else:
            # Verbose: ~10-20x truncation limits
            result = _pformat(value, max_string=1200, max_length=100, max_depth=6)
        return " ".join(result.split())

    def _get_session_output_preview(self, session: AgentSession, max_len: int = 80) -> str | None:
        """Extract output preview from session's return_result call or last execution."""
        # First, look for return_result tool calls in raw spans within this session's time range
        for span in self._raw_spans:
            if span.get("name", "").startswith("tool_execution.return_result"):
                span_start = span.get("start_time", 0)
                if session.start_time <= span_start <= session.end_time:
                    # Found a return_result call within this session (OI-first)
                    args = _io_value(span.get("attributes", {}), "input.value", "tool.arguments")
                    if args:
                        try:
                            args_dict = json.loads(args)
                            result = args_dict.get("result", "")
                            if result:
                                if len(result) > max_len:
                                    return result[:max_len] + "..."
                                return result
                        except (json.JSONDecodeError, TypeError):
                            pass

        # Fallback: look at execution turns for errors or meaningful output
        for turn in reversed(session.turns):
            if isinstance(turn, ExecutionTurn):
                if turn.error:
                    return f"ERROR: {turn.error.split(chr(10))[0][:max_len]}"
        return None

    # =========================================================================
    # Session Navigation
    # =========================================================================

    def _build_session_summary(self, session: AgentSession) -> SessionSummary:
        """Build summary for a single session."""
        llm_count = sum(1 for t in session.turns if isinstance(t, LLMTurn))
        exec_count = sum(1 for t in session.turns if isinstance(t, ExecutionTurn))

        # Aggregate token counts across all LLM turns
        total_prompt = 0
        total_completion = 0
        for t in session.turns:
            if isinstance(t, LLMTurn) and t.token_counts:
                total_prompt += t.token_counts.get("prompt", 0)
                total_completion += t.token_counts.get("completion", 0)

        result_preview = None
        if session.result:
            result_str = str(session.result)
            result_preview = result_str[:80] + "..." if len(result_str) > 80 else result_str

        return SessionSummary(
            session_id=session.session_id,
            agent_name=session.agent_name,
            method_name=session.method_name,
            status=session.status,
            turn_count=len(session.turns),
            llm_turns=llm_count,
            execution_turns=exec_count,
            duration_ms=session.duration_ms,
            parent_session_id=session.parent_session_id,
            has_children=len(session.children) > 0,
            result_preview=result_preview,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
        )

    async def get_session(
        self, session_id: str, *, concise: bool = False, include_reasoning: bool = True
    ) -> str:
        """Get detailed execution info about a specific session.

        Shows how a method executed: turns, tool calls, reasoning, errors.
        Answer: "How did this method work? What decisions were made?"

        Args:
            session_id: The 6-character session ID
            concise: If True, truncate long content. If False, show full details.

        Returns:
            Detailed session info including all turns with reasoning
        """
        session = self._find_session(session_id)
        if not session:
            return (
                f"Session '{session_id}' not found. Use get_overview() to see available sessions."
            )

        lines = []

        # Context header - what trace is this from?
        root = self.sessions[0] if self.sessions else None
        trace_name = ""
        if self.eval_result:
            trace_name = self.eval_result.test_id or ""
        if not trace_name and root:
            trace_name = f"{root.agent_name}.{root.method_name}()"

        # Overall trace status
        trace_status = ""
        if self.eval_result:
            passed = self.eval_result.passed
            if passed is not None:
                trace_status = "[PASSED]" if passed else "[FAILED]"

        # Build context header
        lines.append(f"# Trace: {trace_name} {trace_status}")
        if session.parent_session_id:
            lines.append(f"# Context: Child of session [{session.parent_session_id}]")
        else:
            lines.append("# Context: Root session")
        lines.append("")

        # Status
        is_ok = session.status == "OK"
        if is_ok and isinstance(session.result, dict):
            is_ok = session.result.get("success", True)
        status_label = "[OK]" if is_ok else "[ERR]"

        # Counts
        total_turns = len(session.turns)
        duration = f"{session.duration_ms:.1f}ms"

        # Build header line matching call graph format:
        # "Agent.method [id] (STRATEGY) ────────  5t  1234.0ms [OK]"
        method = f"{session.agent_name}.{session.method_name}"
        strategy_label = f" ({session.strategy})" if session.strategy else ""
        label = f"{method} [{session.session_id[:6]}]{strategy_label}"
        target_width = 55
        dash_count = max(3, target_width - len(label))
        dashes = "─" * dash_count

        lines.append(f"{label} {dashes} {total_turns:>2}t {duration:>8} {status_label}")

        # Input line (same as call graph IN:)
        input_data = self._get_session_input(session)
        if input_data:
            if concise:
                input_str = self._format_params_compact(input_data, max_str=80)
            else:
                input_str = _pformat(input_data, max_string=800, max_length=50, max_depth=5)
                input_str = " ".join(input_str.split())  # Collapse to single line
            lines.append(f"IN:  {input_str}")

        # Output line (same as call graph OUT:)
        if session.result is not None:
            output_str = self._format_value_smart(session.result, concise=concise)
            lines.append(f"OUT: {output_str}")

        # Error info for failed sessions
        if session.status != "OK":
            error_msg = self._get_session_error(session)
            if error_msg:
                error_msg = _pformat(error_msg, max_string=100 if concise else 500)
                lines.append(f"ERR: {error_msg}")

        lines.append("")

        # Execution section - turn-by-turn details with XML structure
        lines.extend(
            self._format_session_execution(
                session, concise=concise, include_reasoning=include_reasoning
            )
        )

        # Navigation
        lines.append("")
        lines.append("## Navigation")
        if session.turns:
            lines.append(
                f"→ {self._nav_hint_turn(session.session_id[:6], 0)} - Full LLM context at turn 0"
            )
            if not is_ok:
                # Find first error turn
                for i, turn in enumerate(session.turns):
                    if isinstance(turn, ExecutionTurn) and turn.error:
                        lines.append(
                            f"→ {self._nav_hint_turn(session.session_id[:6], i)} - Error details"
                        )
                        break

        return "\n".join(lines)

    async def get_session_data(self, session_id: str) -> SessionData:
        """Structured session data for programmatic use.

        Returns:
            SessionData with session summary and turns.

        Raises:
            ValueError: If the session is not found.
        """
        session = self._find_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")

        summary = self._build_session_summary(session)
        turns = [
            self._build_turn_info(session.session_id, i, t) for i, t in enumerate(session.turns)
        ]
        return SessionData(session=summary, turns=turns)

    def _format_session_turns_summary(
        self, session: AgentSession, *, include_reasoning: bool = True
    ) -> list[str]:
        """Format session turns as one-line summaries (concise mode).

        Format: Turn N: [TYPE] code_preview → STATUS (duration)
        """
        lines = []
        lines.append(f"## Turns ({len(session.turns)} total)")

        turn_num = 0
        i = 0
        while i < len(session.turns):
            turn = session.turns[i]
            next_turn = session.turns[i + 1] if i + 1 < len(session.turns) else None

            if isinstance(turn, LLMTurn):
                exec_turn = next_turn if isinstance(next_turn, ExecutionTurn) else None
                duration = (turn.duration_ms or 0) + (
                    (exec_turn.duration_ms or 0) if exec_turn else 0
                )
                error = exec_turn.error if exec_turn else None
                status = _format_turn_status(error)
                reasoning_marker = (
                    " [reasoning]" if include_reasoning and turn.reasoning_content else ""
                )

                # Determine turn type based on tool_call_id pattern
                if exec_turn and exec_turn.tool_call_id.startswith("prefill"):
                    turn_type = "PREFILL"
                else:
                    turn_type = "EXECUTE"

                # Get code preview from the tool call
                code_preview = ""
                if turn.tool_calls:
                    tc = turn.tool_calls[0]
                    if tc.function_name == "execute_python":
                        try:
                            args = json.loads(tc.arguments)
                            code = args.get("code", "") if isinstance(args, dict) else ""
                            # When the turn errored, prefer the failing line extracted from
                            # the error message (e.g. "Cell In[N], line M\n    <code>").
                            # Fall back to the first meaningful code line.
                            if error:
                                code_preview = _extract_failing_line(error) or _first_code_line(
                                    code
                                )
                            else:
                                code_preview = _first_code_line(code)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif tc.function_name == "return_result":
                        code_preview = "return_result(...)"
                    else:
                        code_preview = f"{tc.function_name}(...)"
                elif turn.response and turn.response.strip():
                    # Text response (no tool call)
                    resp = turn.response.strip().split("\n")[0]
                    if len(resp) > 50:
                        code_preview = resp[:47] + "..."
                    else:
                        code_preview = resp

                # Token info for the summary line
                tok_str = ""
                if turn.token_counts:
                    p = turn.token_counts.get("prompt", 0)
                    c = turn.token_counts.get("completion", 0)
                    tok_str = f", {p}→{c}tok"

                lines.append(
                    f"  Turn {turn_num}: [{turn_type}]{reasoning_marker} {code_preview} → {status} ({duration:.0f}ms{tok_str})"
                )

                turn_num += 1
                if exec_turn:
                    i += 2
                else:
                    i += 1
            else:
                # Standalone execution turn (rare)
                exec_turn = turn
                error = exec_turn.error
                status = _format_turn_status(error)
                if error:
                    code_preview = _extract_failing_line(error) or _first_code_line(
                        exec_turn.code or ""
                    )
                else:
                    code_preview = _first_code_line(exec_turn.code or "")
                lines.append(f"  Turn {turn_num}: [EXEC] {code_preview} → {status}")
                turn_num += 1
                i += 1

        return lines

    def _format_session_execution(
        self,
        session: AgentSession,
        *,
        concise: bool = True,
        include_reasoning: bool = True,
    ) -> list[str]:
        """Format session turns as XML mirroring OpenAI message format.

        Args:
            session: The session to format
            concise: If True, show one-line summaries per turn. If False, show full XML content.

        Uses actual roles from messages and function names from tool calls.
        """
        lines = []

        # Concise mode: show one-line summaries per turn
        if concise:
            return self._format_session_turns_summary(session, include_reasoning=include_reasoning)

        max_len = 5000  # Full content in verbose mode

        def trunc(text: str) -> str:
            if len(text) > max_len:
                return text[:max_len] + f"... (+{len(text) - max_len} chars)"
            return text

        def indent(content: str, prefix: str) -> list[str]:
            return [f"{prefix}{line}" for line in content.split("\n")]

        def format_args(args_json: str, tool_name: str) -> str:
            """Parse and format tool arguments. Extract code for readability."""
            try:
                args = json.loads(args_json)
                # Extract code for execute_python (common case, makes output readable)
                if tool_name == "execute_python" and isinstance(args, dict) and "code" in args:
                    return args["code"]
                return _pformat(args, max_string=200 if concise else 5000)
            except (json.JSONDecodeError, TypeError):
                return args_json

        def format_exec_output(exec_turn: ExecutionTurn) -> str | None:
            """Get execution output: error, stdout, or returned_value."""
            if exec_turn.error:
                return exec_turn.error
            if exec_turn.stdout and exec_turn.stdout.strip():
                return exec_turn.stdout.strip()
            if exec_turn.returned_value is not None:
                return _pformat(exec_turn.returned_value, max_string=200 if concise else 5000)
            return None

        turn_num = 0
        shown_messages = False
        i = 0
        while i < len(session.turns):
            turn = session.turns[i]
            next_turn = session.turns[i + 1] if i + 1 < len(session.turns) else None

            if isinstance(turn, LLMTurn):
                exec_turn = next_turn if isinstance(next_turn, ExecutionTurn) else None
                duration = (turn.duration_ms or 0) + (
                    (exec_turn.duration_ms or 0) if exec_turn else 0
                )
                # Turn status: whether this turn's execution had an error (not the overall session status)
                turn_status = "[ERR]" if (exec_turn and exec_turn.error) else "[OK]"

                # Token counts from the LLM generation
                tokens_str = ""
                if turn.token_counts:
                    prompt = turn.token_counts.get("prompt", 0)
                    completion = turn.token_counts.get("completion", 0)
                    tokens_str = f' tokens="{prompt}→{completion}"'

                lines.append(
                    f'<turn n="{turn_num}" duration="{duration:.1f}ms"{tokens_str} status="{turn_status}">'
                )

                # Show messages once (first LLMTurn) - skip system messages
                if turn.messages and not shown_messages:
                    shown_messages = True
                    for msg in turn.messages:
                        # Skip system messages and empty messages without tool_calls
                        if msg.role == "system":
                            continue
                        if not msg.content and not msg.tool_calls:
                            continue

                        # Tool response messages get special formatting
                        if msg.role == "tool":
                            id_attr = f' id="{msg.tool_call_id}"' if msg.tool_call_id else ""
                            lines.append(f"  <tool_response{id_attr}>")
                            if msg.content:
                                lines.extend(indent(trunc(msg.content.strip()), "    "))
                            lines.append("  </tool_response>")
                        else:
                            lines.append(f"  <{msg.role}>")
                            if msg.content:
                                lines.extend(indent(trunc(msg.content.strip()), "    "))

                            # Show tool_calls for assistant messages
                            for tc in msg.tool_calls:
                                id_attr = f' id="{tc.tool_call_id}"' if tc.tool_call_id else ""
                                lines.append(f'    <tool_call name="{tc.function_name}"{id_attr}>')
                                lines.extend(
                                    indent(
                                        trunc(format_args(tc.arguments, tc.function_name)), "      "
                                    )
                                )
                                lines.append("    </tool_call>")

                            lines.append(f"  </{msg.role}>")

                # Assistant response (from this turn's LLM call)
                has_response = turn.response and turn.response.strip()
                has_reasoning = include_reasoning and bool(turn.reasoning_content.strip())
                has_tool_calls = bool(turn.tool_calls)

                if has_response or has_reasoning or has_tool_calls:
                    lines.append("  <assistant>")

                    if has_reasoning:
                        lines.append("    <reasoning>")
                        lines.extend(indent(trunc(turn.reasoning_content.strip()), "      "))
                        lines.append("    </reasoning>")

                    # Text response (if any)
                    if has_response:
                        lines.extend(indent(trunc(turn.response.strip()), "    "))

                    # Tool calls with actual function names
                    # Use exec_turn's tool_call_id if available (LLM output doesn't have it)
                    for i_tc, tc in enumerate(turn.tool_calls):
                        # Get ID from tc if available, otherwise from exec_turn (for single tool call)
                        tc_id = tc.tool_call_id
                        if not tc_id and exec_turn and i_tc == 0:
                            tc_id = exec_turn.tool_call_id
                        id_attr = f' id="{tc_id}"' if tc_id else ""
                        lines.append(f'    <tool_call name="{tc.function_name}"{id_attr}>')
                        lines.extend(
                            indent(trunc(format_args(tc.arguments, tc.function_name)), "      ")
                        )
                        lines.append("    </tool_call>")

                    lines.append("  </assistant>")

                # Execution result as tool_response
                if exec_turn:
                    output = format_exec_output(exec_turn)
                    if output:
                        exec_status = "[ERR]" if exec_turn.error else "[OK]"
                        id_attr = (
                            f' id="{exec_turn.tool_call_id}"' if exec_turn.tool_call_id else ""
                        )
                        lines.append(f'  <tool_response{id_attr} status="{exec_status}">')
                        lines.extend(indent(trunc(output), "    "))
                        lines.append("  </tool_response>")

                lines.append("</turn>")
                lines.append("")
                turn_num += 1
                i += 2 if exec_turn else 1

            elif isinstance(turn, ExecutionTurn):
                # Skip if followed by LLMTurn (prefill - output in LLMTurn messages)
                if isinstance(next_turn, LLMTurn):
                    i += 1
                    continue

                # Standalone execution turn
                output = format_exec_output(turn)
                if output:
                    duration = f"{turn.duration_ms:.0f}ms" if turn.duration_ms else ""
                    status = "error" if turn.error else "ok"
                    lines.append(f'<turn n="{turn_num}" duration="{duration}" status="{status}">')
                    lines.extend(indent(trunc(output), "  "))
                    lines.append("</turn>")
                    lines.append("")
                turn_num += 1
                i += 1

        return lines

    # =========================================================================
    # Turn Navigation
    # =========================================================================

    async def get_turn(
        self, session_id: str, turn_index: int, *, include_reasoning: bool = True
    ) -> str:
        """Get full context window, LLM response, and execution output for a specific turn.

        Shows exactly what the LLM saw and produced. Answer: "What was the exact
        context? Why did it make this decision?"

        Format: [CONTEXT] → [LLM OUTPUT] → [EXECUTION RESULT]

        Args:
            session_id: The 6-character session ID
            turn_index: 0-based index of the turn

        Returns:
            Full turn content including all messages/code/output
        """
        session = self._find_session(session_id)
        if not session:
            return (
                f"Session '{session_id}' not found. Use get_overview() to see available sessions."
            )

        if turn_index < 0 or turn_index >= len(session.turns):
            return f"Turn index {turn_index} out of range (session has {len(session.turns)} turns)"

        turn = session.turns[turn_index]

        if isinstance(turn, LLMTurn):
            return self._format_llm_turn_full(
                session, turn_index, turn, include_reasoning=include_reasoning
            )
        else:
            return self._format_exec_turn_full(session, turn_index, turn)

    async def get_turn_data(self, session_id: str, turn_index: int) -> TurnInfo | None:
        """Structured turn data for programmatic use.

        Returns:
            TurnInfo with turn details, or None if session/turn not found.
        """
        session = self._find_session(session_id)
        if not session:
            return None

        if turn_index < 0 or turn_index >= len(session.turns):
            return None

        turn = session.turns[turn_index]
        return self._build_turn_info(session.session_id, turn_index, turn)

    def _format_llm_turn_full(
        self,
        session: AgentSession,
        turn_index: int,
        turn: LLMTurn,
        *,
        include_reasoning: bool = True,
    ) -> str:
        """Format an LLM turn with full context window using XML-style tags.

        Format: [CONTEXT] → [LLM OUTPUT] → [EXECUTION RESULT]
        """
        max_len = 10000  # Full content always

        def trunc(text: str) -> str:
            if len(text) > max_len:
                return text[:max_len] + f"... (+{len(text) - max_len} chars)"
            return text

        def indent(content: str, prefix: str) -> list[str]:
            return [f"{prefix}{line}" for line in content.split("\n")]

        def format_args(args_json: str, tool_name: str) -> str:
            """Format tool arguments, extracting code for readability."""
            try:
                args = json.loads(args_json)
                if tool_name == "execute_python" and isinstance(args, dict) and "code" in args:
                    return args["code"]
                return _pformat(args, max_string=5000)
            except (json.JSONDecodeError, TypeError):
                return args_json

        lines = []

        # Self-documenting header
        lines.append(f"# Turn {turn_index}: LLM Turn")
        lines.append("#")
        lines.append("# Shows: Context Window → LLM Output → Execution Result (if any)")
        lines.append("")

        # Section header
        lines.append("## LLM Context Window")

        # Header with metadata
        tokens_str = ""
        if turn.token_counts:
            prompt = turn.token_counts.get("prompt", 0)
            completion = turn.token_counts.get("completion", 0)
            tokens_str = f" tokens={prompt}→{completion}"

        duration_str = f" duration={turn.duration_ms:.0f}ms" if turn.duration_ms else ""
        model_str = f' model="{turn.model}"' if turn.model else ""
        lines.append(f'<llm_turn n="{turn_index}"{model_str}{duration_str}{tokens_str}>')

        # Provider and params as attributes
        if turn.provider:
            lines.append(f"  provider: {turn.provider}")

        if turn.invocation_parameters:
            key_params = {
                k: v
                for k, v in turn.invocation_parameters.items()
                if k not in ("tools", "model", "api_base")
            }
            if key_params:
                params_str = ", ".join(f"{k}={v}" for k, v in list(key_params.items())[:5])
                if len(params_str) > 80:
                    params_str = params_str[:80] + "..."
                lines.append(f"  params: {params_str}")

        if turn.tools:
            tool_names = [t.name for t in turn.tools]
            lines.append(f"  tools: {', '.join(tool_names)}")

        lines.append("")

        # Messages (context window) - using same format as get_session
        for msg in turn.messages:
            if not msg.content and not msg.tool_calls:
                continue

            # Tool response messages get special formatting
            if msg.role == "tool":
                id_attr = f' id="{msg.tool_call_id}"' if msg.tool_call_id else ""
                lines.append(f"  <tool_response{id_attr}>")
                if msg.content:
                    lines.extend(indent(trunc(msg.content.strip()), "    "))
                lines.append("  </tool_response>")
            else:
                lines.append(f"  <{msg.role}>")
                if msg.content:
                    lines.extend(indent(trunc(msg.content.strip()), "    "))

                for tc in msg.tool_calls:
                    id_attr = f' id="{tc.tool_call_id}"' if tc.tool_call_id else ""
                    lines.append(f'    <tool_call name="{tc.function_name}"{id_attr}>')
                    lines.extend(
                        indent(trunc(format_args(tc.arguments, tc.function_name)), "      ")
                    )
                    lines.append("    </tool_call>")

                lines.append(f"  </{msg.role}>")

        lines.append("")
        lines.append("  <!-- LLM OUTPUT -->")

        # Get the following exec_turn if any (for tool_call_id)
        next_turn = session.turns[turn_index + 1] if turn_index + 1 < len(session.turns) else None
        exec_turn = next_turn if isinstance(next_turn, ExecutionTurn) else None

        # LLM response
        has_response = turn.response and turn.response.strip()
        has_reasoning = include_reasoning and bool(turn.reasoning_content.strip())
        has_tool_calls = bool(turn.tool_calls)

        if has_response or has_reasoning or has_tool_calls:
            lines.append("  <assistant>")

            if has_reasoning:
                lines.append("    <reasoning>")
                lines.extend(indent(trunc(turn.reasoning_content.strip()), "      "))
                lines.append("    </reasoning>")

            if has_response:
                lines.extend(indent(trunc(turn.response.strip()), "    "))

            for i_tc, tc in enumerate(turn.tool_calls):
                # Get ID from tc if available, otherwise from exec_turn (for single tool call)
                tc_id = tc.tool_call_id
                if not tc_id and exec_turn and i_tc == 0:
                    tc_id = exec_turn.tool_call_id
                id_attr = f' id="{tc_id}"' if tc_id else ""
                lines.append(f'    <tool_call name="{tc.function_name}"{id_attr}>')
                lines.extend(indent(trunc(format_args(tc.arguments, tc.function_name)), "      "))
                lines.append("    </tool_call>")

            lines.append("  </assistant>")

        # Include execution output if the next turn is an ExecutionTurn
        if exec_turn:
            lines.append("")
            lines.append("  <!-- EXECUTION RESULT -->")
            exec_status = "[ERR]" if exec_turn.error else "[OK]"
            id_attr = f' id="{exec_turn.tool_call_id}"' if exec_turn.tool_call_id else ""
            lines.append(f'  <tool_response{id_attr} status="{exec_status}">')

            if exec_turn.error:
                lines.extend(indent(trunc(exec_turn.error), "    "))
            elif exec_turn.stdout and exec_turn.stdout.strip():
                lines.extend(indent(trunc(exec_turn.stdout.strip()), "    "))
            elif exec_turn.returned_value is not None:
                val_str = _pformat(exec_turn.returned_value, max_string=5000)
                lines.append(f"    Returned: {val_str}")

            lines.append("  </tool_response>")

        lines.append("</llm_turn>")
        return "\n".join(lines)

    def _format_exec_turn_full(
        self, session: AgentSession, turn_index: int, turn: ExecutionTurn
    ) -> str:
        """Format an execution turn with full details using XML-style tags.

        Also includes the LLM context that led to this execution so the agent can
        understand why this code was generated. Looks for:
        1. Preceding LLM turn (standard flow)
        2. Following LLM turn (prefill flow where execution comes first)
        """
        max_len = 10000  # Full content always

        def trunc(text: str) -> str:
            if len(text) > max_len:
                return text[:max_len] + f"... (+{len(text) - max_len} chars)"
            return text

        def indent(content: str, prefix: str) -> list[str]:
            return [f"{prefix}{line}" for line in content.split("\n")]

        lines = []

        # Find the LLM turn that provides context for this execution
        # First try preceding turn (standard flow), then following (prefill flow)
        context_llm_turn: LLMTurn | None = None
        context_turn_idx = None
        context_source = ""

        # Look for preceding LLM turn first
        for i in range(turn_index - 1, -1, -1):
            t = session.turns[i]
            if isinstance(t, LLMTurn):
                context_llm_turn = t
                context_turn_idx = i
                context_source = "preceding"
                break

        # If no preceding turn, look for following LLM turn (prefill case)
        if not context_llm_turn:
            for i in range(turn_index + 1, len(session.turns)):
                t = session.turns[i]
                if isinstance(t, LLMTurn):
                    context_llm_turn = t
                    context_turn_idx = i
                    context_source = "following"
                    break

        # Add self-documenting header
        lines.append(f"# Turn {turn_index}: Execution Turn")
        lines.append("#")
        if context_llm_turn and context_turn_idx is not None:
            if context_source == "preceding":
                lines.append(
                    f"# This is an execution turn. Including LLM context and tool call from turn {context_turn_idx}."
                )
            else:
                # Prefill case - LLM turn comes after but contains the context
                lines.append("# This is a prefill execution (code runs before LLM completes).")
                lines.append(
                    f"# Including LLM context from turn {context_turn_idx} (system prompt + task)."
                )
        else:
            lines.append("# This is an execution turn. No associated LLM turn found.")
        lines.append("")

        # If we found an LLM turn with context, show relevant messages
        if context_llm_turn:
            lines.append(f"## LLM Context (from turn {context_turn_idx})")

            # For prefill flow, only show the initial context (system + user task)
            # not the execution results that came from this turn
            messages_to_show = context_llm_turn.messages
            if context_source.startswith("following"):
                # Only show system and first user message for prefill
                messages_to_show = [
                    m
                    for m in context_llm_turn.messages
                    if m.role in ("system", "user")
                    and "<execute_python" not in (m.content or "")
                    and "<tool_result" not in (m.content or "")
                ]

            for msg in messages_to_show:
                if not msg.content and not msg.tool_calls:
                    continue

                if msg.role == "tool":
                    id_attr = f' id="{msg.tool_call_id}"' if msg.tool_call_id else ""
                    lines.append(f"<tool_response{id_attr}>")
                    if msg.content:
                        lines.extend(indent(trunc(msg.content.strip()), "  "))
                    lines.append("</tool_response>")
                else:
                    lines.append(f"<{msg.role}>")
                    if msg.content:
                        lines.extend(indent(trunc(msg.content.strip()), "  "))

                    for tc in msg.tool_calls:
                        id_attr = f' id="{tc.tool_call_id}"' if tc.tool_call_id else ""
                        lines.append(f'  <tool_call name="{tc.function_name}"{id_attr}>')
                        try:
                            args = json.loads(tc.arguments)
                            if (
                                tc.function_name == "execute_python"
                                and isinstance(args, dict)
                                and "code" in args
                            ):
                                lines.extend(indent(trunc(args["code"]), "    "))
                            else:
                                lines.extend(indent(trunc(_pformat(args, max_string=5000)), "    "))
                        except (json.JSONDecodeError, TypeError):
                            lines.extend(indent(trunc(tc.arguments), "    "))
                        lines.append("  </tool_call>")

                    lines.append(f"</{msg.role}>")

            lines.append("")

        # Now show the execution turn itself
        lines.append(f"## Execution (turn {turn_index})")
        status = "[ERR]" if turn.error else "[OK]"
        duration_str = f" duration={turn.duration_ms:.1f}ms" if turn.duration_ms else ""

        lines.append(f'<exec_turn n="{turn_index}"{duration_str} status="{status}">')

        # Code executed (tool call)
        if turn.code:
            id_attr = f' id="{turn.tool_call_id}"' if turn.tool_call_id else ""
            lines.append(f'  <tool_call name="execute_python"{id_attr}>')
            lines.extend(indent(trunc(turn.code), "    "))
            lines.append("  </tool_call>")

        # Output (tool response)
        if turn.stdout or turn.returned_value is not None or turn.error:
            id_attr = f' id="{turn.tool_call_id}"' if turn.tool_call_id else ""
            exec_status = "[ERR]" if turn.error else "[OK]"
            lines.append(f'  <tool_response{id_attr} status="{exec_status}">')

            if turn.stdout:
                lines.extend(indent(trunc(turn.stdout.strip()), "    "))

            if turn.returned_value is not None:
                ret_str = _pformat(
                    turn.returned_value,
                    max_string=5000,
                    max_length=100,
                    max_depth=6,
                )
                lines.append(f"    Returned: {ret_str}")

            if turn.error:
                lines.extend(indent(trunc(turn.error), "    "))

            lines.append("  </tool_response>")

        lines.append("</exec_turn>")
        return "\n".join(lines)

    def _build_turn_info(
        self,
        session_id: str,
        turn_index: int,
        turn: LLMTurn | ExecutionTurn,
    ) -> TurnInfo:
        """Build TurnInfo from a turn object."""
        if isinstance(turn, LLMTurn):
            return TurnInfo(
                session_id=session_id,
                turn_index=turn_index,
                turn_type="llm",
                messages=[{"role": m.role, "content": m.content} for m in turn.messages],
                response=turn.response,
                reasoning_content=turn.reasoning_content or None,
                model=turn.model,
                token_counts=turn.token_counts,
                tool_calls=[
                    {"function_name": tc.function_name, "arguments": tc.arguments}
                    for tc in turn.tool_calls
                ],
                duration_ms=turn.duration_ms,
            )
        else:
            return TurnInfo(
                session_id=session_id,
                turn_index=turn_index,
                turn_type="execution",
                code=turn.code,
                stdout=turn.stdout,
                error=turn.error,
                error_type=turn.error_type,
                returned_value=turn.returned_value,
                duration_ms=turn.duration_ms,
            )

    # =========================================================================
    # Error Navigation
    # =========================================================================

    async def get_errors(self) -> str:
        """Get all errors found in the trace.

        Returns:
            List of all errors with context and navigation hints
        """
        errors = self._find_errors()
        if not errors:
            return "No errors found in trace."

        lines = [f"Found {len(errors)} error(s):", ""]

        for error in errors:
            session_id = (
                error.session_id[:6]
                if error.session_id
                else error.span_id[:6]
                if error.span_id
                else "?"
            )
            turn_info = f", turn {error.turn_index}" if error.turn_index is not None else ""
            err_type = error.error_type or error.error_category
            lines.append(f"  [{session_id}{turn_info}] {err_type}")
            # Show up to 300 chars of error message
            msg = error.error_message[:300]
            if len(error.error_message) > 300:
                msg += "..."
            lines.append(f"    {msg}")
            if error.context:
                lines.append(f"    Context: {error.context[:100]}")
            lines.append("")

        # Navigation hint
        first_sid = errors[0].session_id if errors else None
        lines.append("")
        if first_sid:
            lines.append(f"→ Drill into a session:  {self._nav_hint(first_sid)}")
        else:
            lines.append(f"→ Drill into a session:  {self._nav_hint('<session_id>')}")

        return "\n".join(lines)

    async def get_errors_data(self) -> dict[str, Any]:
        """Structured error list for programmatic use."""
        errors = self._find_errors()
        return {"count": len(errors), "errors": [e.to_dict() for e in errors]}

    # Alias for backwards compatibility
    # Alias removed: get_all_errors = get_errors

    # Control flow signals that should not be reported as errors
    # These are exceptions used internally for control flow, not actual failures
    CONTROL_FLOW_SIGNALS = frozenset(
        {
            "_ReturnResultSignal",  # Normal return_result() completion
            "_DelegateSignal",  # Agent delegation control flow
            "_CancelSignal",  # Cancellation control flow
            "StopIteration",  # Standard Python iterator exhaustion
            "GeneratorExit",  # Generator cleanup
        }
    )

    def _find_errors(self) -> list[ErrorInfo]:
        """Find all errors in the trace.

        Note: Control flow signals (like _ReturnResultSignal) are filtered out
        since they represent normal execution flow, not actual errors.

        Includes:
        - Session status errors (session ended with non-OK status)
        - Execution errors (code raised an exception)
        - Validation errors (tool received invalid arguments)
        """
        errors = []

        for session in self._all_sessions:
            # Check session status
            if session.status != "OK":
                # Get actual error message from session
                error_msg = self._get_session_error(session)
                if error_msg:
                    # Include error chain info if this session has failed children
                    child_errors = [
                        f"{c.agent_name}.{c.method_name}"
                        for c in session.children
                        if c.status != "OK"
                    ]
                    if child_errors:
                        error_msg = f"{error_msg} (caused by: {', '.join(child_errors)})"
                else:
                    error_msg = f"Session ended with status: {session.status}"

                errors.append(
                    ErrorInfo(
                        session_id=session.session_id,
                        turn_index=None,
                        error_category="status_error",
                        error_message=error_msg,
                        context=f"{session.agent_name}.{session.method_name}()",
                    )
                )

            # Check execution turns for errors (excluding control flow signals)
            for i, turn in enumerate(session.turns):
                if isinstance(turn, ExecutionTurn) and turn.error:
                    # Skip control flow signals - they're not real errors
                    if turn.error_type in self.CONTROL_FLOW_SIGNALS:
                        continue
                    errors.append(
                        ErrorInfo(
                            session_id=session.session_id,
                            turn_index=i,
                            error_category="execution_error",
                            error_message=turn.error[:500],
                            context=f"Code: {turn.code[:100]}..." if turn.code else "Unknown code",
                            error_type=turn.error_type,  # Include actual error type
                        )
                    )

        # Check raw spans for validation errors (tool argument validation failures)
        for span in self._raw_spans:
            name = span.get("name", "")
            attrs = span.get("attributes", {})

            # Look for tool execution spans with errors
            if name.startswith("tool_execution") and attrs.get("error.type"):
                span_id = span.get("span_id", "unknown")
                errors.append(
                    ErrorInfo(
                        session_id="",  # Validation errors don't have a session context
                        turn_index=None,
                        error_category="validation_error",
                        error_message=attrs.get("error.message", "")[:500],
                        context=f"Tool: {attrs.get('tool.name', name)}",
                        error_type=attrs.get("error.type"),
                        span_id=span_id,
                        tool_name=attrs.get("tool.name", name),
                        tool_arguments=_io_value(attrs, "input.value", "tool.arguments") or "{}",
                    )
                )

        return errors

    # =========================================================================
    # Result Access
    # =========================================================================

    async def get_eval_context(self, concise: bool = True) -> str:
        """Get evaluation context if available.

        Returns input, expected output, actual output, and scores from the
        eval result. This data comes from an 'eval' span in the trace file.

        **Note:** If this returns "No evaluation result", the trace file
        doesn't include eval metadata. Ensure your evaluation runner adds
        a span with name='eval' and these attributes:

        Required attributes:
        - eval.passed (bool): Whether the eval passed
        - eval.test_id (str): Test identifier

        Optional attributes:
        - eval.weighted_score (float): Overall score
        - eval.model (str): Model used
        - eval.agent_class (str): Agent class name
        - eval.method (str): Method name

        Scorer attributes (for detailed scoring):
        - eval.scorer.<name>.score (float): Score from scorer
        - eval.scorer.<name>.passed (bool): Whether scorer passed
        - eval.scorer.<name>.reasoning (str): Scorer's reasoning

        Example instrumentation:
        ```python
        with tracer.start_span("eval") as span:
            span.set_attribute("eval.passed", False)
            span.set_attribute("eval.test_id", "router_002")
            span.set_attribute("eval.scorer.output_check.passed", False)
            span.set_attribute("eval.scorer.output_check.reasoning", "...")
        ```

        Returns:
            Formatted eval context, or message if no eval data available.
        """
        if not self.eval_result:
            return (
                "No evaluation result provided.\n\n"
                "To enable eval context, the trace file must contain an 'eval' span "
                "with attributes like eval.passed, eval.test_id, etc.\n"
                "See `help(TraceExplorer.get_eval_context)` for details."
            )

        lines = ["# Evaluation Context", ""]

        if self.eval_result.input is not None:
            lines.append("## Input")
            lines.append("```json")
            if concise and len(json.dumps(self.eval_result.input, indent=2, default=str)) > 1000:
                lines.append(
                    json.dumps(self.eval_result.input, indent=2, default=str)[:1000] + "..."
                )
            else:
                lines.append(json.dumps(self.eval_result.input, indent=2, default=str))
            lines.append("```")
            lines.append("")

        if self.eval_result.expected is not None:
            lines.append("## Expected Output")
            lines.append("```json")
            if concise and len(json.dumps(self.eval_result.expected, indent=2, default=str)) > 1000:
                lines.append(
                    json.dumps(self.eval_result.expected, indent=2, default=str)[:1000] + "..."
                )
            else:
                lines.append(json.dumps(self.eval_result.expected, indent=2, default=str))
            lines.append("```")
            lines.append("")

        # Show actual output
        if self.eval_result.output is not None:
            lines.append("## Actual Output")
            lines.append("```json")
            output_str = json.dumps(self.eval_result.output, indent=2, default=str)
            if concise and len(output_str) > 1000:
                lines.append(output_str[:1000] + "...")
            else:
                lines.append(output_str)
            lines.append("```")
            lines.append("")

        if self.eval_result.passed is not None:
            status = "PASSED" if self.eval_result.passed else "FAILED"
            lines.append(f"## Result: {status}")

        if self.eval_result.scores:
            lines.append("")
            lines.append("## Scorer Results")
            for scorer_name, score_detail in self.eval_result.scores.items():
                passed_label = "[PASS]" if score_detail.passed else "[FAIL]"
                reason = score_detail.reasoning or ""
                lines.append(f"- **{scorer_name}** {passed_label}: {reason}")

        if self.eval_result.error:
            lines.append("")
            lines.append("## Error")
            lines.append(f"```\n{self.eval_result.error}\n```")

        return "\n".join(lines)

    async def get_eval_context_data(self) -> dict[str, Any]:
        """Structured eval context data."""
        if not self.eval_result:
            return {"error": "No evaluation result provided."}
        return {
            "eval_result": self.eval_result,
            "benchmark_context": self.benchmark_context,
        }

    # =========================================================================
    # Harness Telemetry
    # =========================================================================

    async def get_harness_telemetry(self, session_id: str | None = None) -> str:
        """Show harness telemetry for a session or all sessions.

        Extracts harness.* span attributes from generation spans and
        formats them as a readable table.  The display is driven by
        ``HarnessMetrics.span_attribute_schema()`` so it stays in sync
        with the metrics model automatically.
        """
        data = await self.get_harness_telemetry_data(session_id)
        if not data or "error" in data:
            return (
                data.get("error", "No harness telemetry found.")
                if data
                else "No harness telemetry found."
            )

        lines: list[str] = []
        title = data.get("title", "Harness Telemetry")
        lines.append(title)
        lines.append("\u2500" * len(title))
        lines.append("")

        metrics = data.get("metrics", {})
        if not metrics:
            lines.append("  (no harness interventions recorded)")
            return "\n".join(lines)

        # Use the schema to drive display (single source of truth)
        from nooa.runtime.harness_metrics import get_span_schema

        schema = get_span_schema()

        # Group schema entries by category
        categories: dict[str, list[tuple[str, str, bool]]] = {}
        for entry in schema:
            cat = entry.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((entry.key, entry.label, entry.is_detail))

        for cat_name, items in categories.items():
            cat_has_data = False
            cat_lines: list[str] = []
            for key, label, is_detail in items:
                value = metrics.get(key, 0 if not is_detail else [])
                if is_detail:
                    continue  # Show details inline with their count
                if value:
                    cat_has_data = True
                    # Find matching detail key
                    detail_str = ""
                    for dk, _, did in items:
                        if did and dk in metrics:
                            details = metrics[dk]
                            if isinstance(details, list) and details:
                                detail_str = f"  [{', '.join(str(d) for d in details[:5])}]"
                                if len(details) > 5:
                                    detail_str = (
                                        detail_str[:-1] + f", ... +{len(details) - 5} more]"
                                    )
                            break
                    cat_lines.append(f"  {label + ':':<35} {value}{detail_str}")

            if cat_has_data:
                lines.append(f"  {cat_name}")
                lines.extend(cat_lines)
                lines.append("")

        prefill = data.get("prefill_type", "")
        if prefill:
            lines.append(f"  Prefill: {prefill}")
            lines.append("")

        return "\n".join(lines)

    async def get_harness_telemetry_data(self, session_id: str | None = None) -> dict[str, Any]:
        """Structured harness telemetry data.

        Extracts all harness.* attributes from generation spans.
        Returns a dict with title, metrics (merged from all generation spans), and prefill_type.
        """
        # Determine which spans to search
        if session_id:
            session = self._find_session(session_id)
            if not session:
                return {"error": f"Session not found: {session_id}"}
            title = f"Harness Telemetry for session {session_id}"
            gen_spans = [
                s
                for s in self._raw_spans
                if s.get("name") == "generation"
                and s.get("attributes", {}).get("agent.call_id") == session.call_id
            ]
        else:
            title = "Harness Telemetry (all sessions)"
            gen_spans = [s for s in self._raw_spans if s.get("name") == "generation"]

        if not gen_spans:
            return {"title": title, "metrics": {}, "prefill_type": ""}

        # Merge harness.* attributes from all generation spans
        merged: dict[str, Any] = {}
        prefill_type = ""
        for span in gen_spans:
            attrs = span.get("attributes", {})
            for key, value in attrs.items():
                if not key.startswith("harness."):
                    continue
                if key == "harness.prefill_type":
                    prefill_type = value
                    continue

                if key in merged:
                    existing = merged[key]
                    if isinstance(existing, int) and isinstance(value, int):
                        merged[key] = existing + value
                    elif isinstance(existing, list) and isinstance(value, list):
                        merged[key] = existing + value
                else:
                    merged[key] = value

        return {"title": title, "metrics": merged, "prefill_type": prefill_type}

    # =========================================================================
    # Search
    # =========================================================================

    async def search(self, pattern: str, *, concise: bool = True) -> str:
        """Search for a pattern across all trace content.

        Args:
            pattern: Regex pattern to search for
            concise: If True, show one-line summaries; otherwise show details

        Returns:
            List of matches with context
        """
        results = self._search(pattern)
        if not results:
            return f"No matches found for pattern: {pattern}"

        # Group by location for summary
        loc_counts: dict[str, int] = {}
        for r in results:
            loc_counts[r.location] = loc_counts.get(r.location, 0) + 1

        lines = [f"Found {len(results)} match(es) for '{pattern}':", ""]

        # Summary
        lines.append("Summary:")
        for loc, count in sorted(loc_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  • {loc}: {count}")
        lines.append("")

        # Results
        lines.append("Matches:")
        max_results = 20
        for result in results[:max_results]:
            match_text = result.match_text.replace("\n", " ").strip()
            # Format match text with ellipsis to show context around match
            if concise:
                if len(match_text) > 60:
                    match_text = "..." + match_text[:54] + "..."
                else:
                    match_text = "..." + match_text + "..."
                lines.append(
                    f"  • [{result.session_id[:6]} t{result.turn_index}] {result.location}: {match_text}"
                )
            else:
                if len(match_text) > 100:
                    match_text = "..." + match_text[:94] + "..."
                else:
                    match_text = "..." + match_text + "..."
                lines.append(
                    f"  [{result.session_id[:6]}, turn {result.turn_index}] {result.location}"
                )
                lines.append(f"    {match_text}")

        if len(results) > max_results:
            lines.append(f"  ... and {len(results) - max_results} more matches")

        # Navigation hint
        first_sid = results[0].session_id if results else None
        lines.append("")
        if first_sid:
            lines.append(f"→ Drill into a session:  {self._nav_hint(first_sid)}")
        else:
            lines.append(f"→ Drill into a session:  {self._nav_hint('<session_id>')}")

        return "\n".join(lines)

    async def search_data(self, pattern: str) -> SearchMatches:
        """Structured search results for programmatic use.

        Returns:
            SearchMatches with:
            - pattern: The search pattern
            - match_count: Number of matches found
            - matches: List of SearchResult objects
            - by_location: Dict mapping location to count
        """
        results = self._search(pattern)
        loc_counts: dict[str, int] = {}
        for r in results:
            loc_counts[r.location] = loc_counts.get(r.location, 0) + 1
        return SearchMatches(
            pattern=pattern,
            match_count=len(results),
            matches=results,
            by_location=loc_counts,
        )

    # Alias for backwards compatibility
    # Alias removed: search_content = search

    def _search(self, pattern: str) -> list[SearchResult]:
        """Search all trace content for a pattern."""
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return results

        for session in self._all_sessions:
            for i, turn in enumerate(session.turns):
                if isinstance(turn, LLMTurn):
                    # Search messages
                    for msg in turn.messages:
                        for match in regex.finditer(msg.content):
                            start = max(0, match.start() - 30)
                            end = min(len(msg.content), match.end() + 30)
                            results.append(
                                SearchResult(
                                    session_id=session.session_id,
                                    turn_index=i,
                                    turn_type="llm",
                                    location=f"message[{msg.role}]",
                                    match_text=msg.content[start:end],
                                )
                            )

                    # Search response
                    if turn.response:
                        for match in regex.finditer(turn.response):
                            start = max(0, match.start() - 30)
                            end = min(len(turn.response), match.end() + 30)
                            results.append(
                                SearchResult(
                                    session_id=session.session_id,
                                    turn_index=i,
                                    turn_type="llm",
                                    location="response",
                                    match_text=turn.response[start:end],
                                )
                            )
                else:
                    # Search code
                    if turn.code:
                        for match in regex.finditer(turn.code):
                            start = max(0, match.start() - 30)
                            end = min(len(turn.code), match.end() + 30)
                            results.append(
                                SearchResult(
                                    session_id=session.session_id,
                                    turn_index=i,
                                    turn_type="execution",
                                    location="code",
                                    match_text=turn.code[start:end],
                                )
                            )

                    # Search stdout
                    if turn.stdout:
                        for match in regex.finditer(turn.stdout):
                            start = max(0, match.start() - 30)
                            end = min(len(turn.stdout), match.end() + 30)
                            results.append(
                                SearchResult(
                                    session_id=session.session_id,
                                    turn_index=i,
                                    turn_type="execution",
                                    location="stdout",
                                    match_text=turn.stdout[start:end],
                                )
                            )

                    # Search error
                    if turn.error:
                        for match in regex.finditer(turn.error):
                            start = max(0, match.start() - 30)
                            end = min(len(turn.error), match.end() + 30)
                            results.append(
                                SearchResult(
                                    session_id=session.session_id,
                                    turn_index=i,
                                    turn_type="execution",
                                    location="error",
                                    match_text=turn.error[start:end],
                                )
                            )

        return results

    async def get_turn_context(
        self,
        session_id: str,
        turn_index: int,
        max_length: int | None = None,
        include_system: bool = False,
    ) -> str:
        """Get the full context text for a specific turn.

        Builds the complete context window from messages and tool calls in the turn.
        By default, excludes system messages to show just user/assistant conversation.

        Args:
            session_id: Session ID (6-character short ID or full ID)
            turn_index: Turn index (0-based)
            max_length: Optional max length to truncate (for preview)
            include_system: If True, include system messages (default: False)

        Returns:
            Full context text, or empty string if turn not found

        Example:
            # Get just user/assistant messages
            context = await trace.get_turn_context("abc123", 5)

            # Include system prompt too
            full_context = await trace.get_turn_context("abc123", 5, include_system=True)
        """
        session = self._find_session(session_id)
        if not session:
            return ""

        if turn_index < 0 or turn_index >= len(session.turns):
            return ""

        turn = session.turns[turn_index]
        if not isinstance(turn, LLMTurn):
            return ""

        # Build the full context string from messages (excluding system by default)
        context_parts = []
        for msg in turn.messages:
            # Skip system messages unless explicitly requested
            if not include_system and msg.role == "system":
                continue
            if msg.content:
                context_parts.append(
                    msg.content[:max_length] + "... [truncated]" if max_length else msg.content
                )
            # Also include tool call arguments
            for tc in msg.tool_calls:
                if tc.arguments:
                    context_parts.append(
                        tc.arguments[:max_length] + "..." if max_length else tc.arguments
                    )

        context_text = "\n".join(context_parts)

        return context_text

    async def search_in_turn_context(
        self,
        session_id: str,
        turn_index: int,
        pattern: str,
        max_matches: int = 10,
        include_system: bool = False,
    ) -> list[SearchResult]:
        """Search for a pattern in a specific turn's context.

        Returns matches with context around each match, useful for finding
        specific information that was present (or missing) at a given turn.

        Args:
            session_id: Session ID (6-character short ID or full ID)
            turn_index: Turn index (0-based)
            pattern: Regex pattern to search for
            max_matches: Maximum number of matches to return

        Returns:
            List of SearchResult objects

        Example:
            matches = await trace.search_in_turn_context("abc123", 5, "TypeError")
            for match in matches:
                print(f"Found: {match['match']} at position {match['start']}")
        """
        context_text = await self.get_turn_context(
            session_id, turn_index, include_system=include_system
        )
        if not context_text:
            return []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []

        matches = []
        for match_obj in regex.finditer(context_text):
            if len(matches) >= max_matches:
                break

            start = match_obj.start()
            end = match_obj.end()
            match_text = match_obj.group()

            # Get context around the match (50 chars before/after)
            context_before = context_text[max(0, start - 50) : start]
            context_after = context_text[end : min(len(context_text), end + 50)]

            # Include context in match_text for better readability
            match_with_context = f"{context_before}{match_text}{context_after}"
            matches.append(
                SearchResult(
                    session_id=session_id,
                    turn_index=turn_index,
                    turn_type="llm",
                    location="context",
                    match_text=match_with_context,
                )
            )

        return matches

    # =========================================================================
    # Recursion Analysis (via Session Tree)
    # =========================================================================

    @property
    def agent_count(self) -> int:
        """Total number of agent sessions (including nested)."""
        return len(self._all_sessions)

    @property
    def max_agent_depth(self) -> int:
        """Maximum depth of agent nesting."""
        if not self._all_sessions:
            return 0
        return max(s.depth for s in self._all_sessions)

    def _get_method_counts(self) -> dict[str, int]:
        """Internal: Get invocation count for each agent method."""
        counts: dict[str, int] = {}
        for session in self._all_sessions:
            name = session.full_name
            counts[name] = counts.get(name, 0) + 1
        return counts

    async def get_method_counts(self) -> dict[str, int]:
        """Get invocation count for each agent method.

        Returns:
            Dict mapping "AgentName.method_name" to invocation count
        """
        return self._get_method_counts()

    async def get_recursion_pattern(self) -> str:
        """Identify the recursion pattern (self-recursion, mutual, etc.).

        Returns:
            Description string like "self-recursion (Agent.method)" or "mutual recursion (A <-> B)"
        """
        method_counts = await self.get_method_counts()
        unique_methods = len(method_counts)

        if unique_methods == 0:
            return "no agent calls"
        elif unique_methods == 1:
            method = list(method_counts.keys())[0]
            return f"self-recursion ({method})"
        elif unique_methods == 2 and self.agent_count >= 4:
            methods = list(method_counts.keys())
            return f"mutual recursion ({methods[0]} <-> {methods[1]})"
        else:
            return f"{unique_methods} methods involved"

    # Backward compat alias
    _get_recursion_pattern = get_recursion_pattern

    # =========================================================================
    # Navigation Helpers
    # =========================================================================

    def _find_first_error(self) -> tuple[AgentSession, int, ExecutionTurn] | None:
        """Internal: Find the first error in the trace."""
        # Sort sessions by start_time to find chronologically first error
        for session in sorted(self._all_sessions, key=lambda s: s.start_time):
            for i, turn in enumerate(session.turns):
                if isinstance(turn, ExecutionTurn) and turn.error:
                    return (session, i, turn)
        return None

    # =========================================================================
    # Timeline View
    # =========================================================================

    def _get_timeline(self, max_events: int = 50) -> str:
        """Internal: Get a chronological timeline of events across all sessions.

        Prefer get_overview() for call graph view.
        """
        events: list[tuple[int, str, str, str]] = []  # (time, span_id, type, summary)

        for session in self._all_sessions:
            # Session start
            events.append(
                (
                    session.start_time,
                    session.session_id,
                    "AGENT_START",
                    f"{session.full_name} (depth {session.depth})",
                )
            )

            # Turns within session
            for i, turn in enumerate(session.turns):
                if isinstance(turn, LLMTurn):
                    # Show tool names instead of just count
                    if turn.tool_calls:
                        tool_names = [tc.function_name for tc in turn.tool_calls]
                        tool_info = f" → {', '.join(tool_names)}"
                    else:
                        tool_info = ""
                    summary = f"LLM {len(turn.messages)} msgs{tool_info}"
                    # Estimate time from session start + turn order
                    t = session.start_time + (i * 1000000)  # rough ordering
                    # Use turn's span_id for unique identification
                    turn_id = _short_id(turn.span_id) if turn.span_id else turn.session_id
                    events.append((t, turn_id, "LLM", summary))
                else:
                    status = "ERR" if turn.error else "OK"
                    summary = f"Exec [{status}]"
                    if turn.error:
                        summary += f": {turn.error[:50]}..."
                    elif turn.returned_value is not None:
                        val_preview = str(turn.returned_value)[:40]
                        summary += f" returned: {val_preview}"
                    t = session.start_time + (i * 1000000)
                    # Use turn's own span_id if available, otherwise fall back to execution_id
                    turn_id = (
                        _short_id(turn.span_id)
                        if turn.span_id
                        else (
                            turn.execution_id
                            if turn.execution_id != "------"
                            else session.session_id
                        )
                    )
                    events.append((t, turn_id, "EXEC", summary))

            # Session end
            events.append(
                (
                    session.end_time,
                    session.session_id,
                    "AGENT_END",
                    f"{session.status}",
                )
            )

        # Sort by time
        events.sort(key=lambda x: x[0])

        # Format output
        lines = ["# Timeline", ""]
        for _, sid, etype, summary in events[:max_events]:
            lines.append(f"[{sid}] {etype}: {summary}")

        if len(events) > max_events:
            lines.append(f"... ({len(events) - max_events} more events)")

        # Navigation hint
        lines.append("")
        lines.append(
            f"→ {self._nav_hint_cmd('get_span(span_id)', '--raw <span_id>')} to see full details"
        )

        return "\n".join(lines)

    # =========================================================================
    # Public Timeline & Error Navigation
    # =========================================================================

    async def get_timeline(self, max_events: int = 50) -> str:
        """Get a chronological timeline of events across all sessions.

        Shows spans in order with timestamps and durations.
        Useful for understanding execution flow in long traces.

        Args:
            max_events: Maximum number of events to show (default 50).

        Returns:
            Formatted timeline of events.
        """
        return self._get_timeline(max_events)

    async def get_timeline_data(self, max_events: int = 50) -> TimelineData:
        """Structured timeline events for programmatic use.

        Returns:
            TimelineData with chronological events from the trace.
        """
        raw_events: list[tuple[int, str, str, str]] = []

        for session in self._all_sessions:
            raw_events.append(
                (
                    session.start_time,
                    session.session_id,
                    "AGENT_START",
                    f"{session.full_name} (depth {session.depth})",
                )
            )

            for i, turn in enumerate(session.turns):
                if isinstance(turn, LLMTurn):
                    tool_names = (
                        [tc.function_name for tc in turn.tool_calls] if turn.tool_calls else []
                    )
                    summary = f"LLM {len(turn.messages)} msgs"
                    if tool_names:
                        summary += f" → {', '.join(tool_names)}"
                    t = session.start_time + (i * 1000000)
                    turn_id = _short_id(turn.span_id) if turn.span_id else turn.session_id
                    raw_events.append((t, turn_id, "LLM", summary))
                else:
                    status = "ERR" if turn.error else "OK"
                    summary = f"Exec [{status}]"
                    if turn.error:
                        summary += f": {turn.error[:50]}..."
                    elif turn.returned_value is not None:
                        summary += f" returned: {str(turn.returned_value)[:40]}"
                    t = session.start_time + (i * 1000000)
                    turn_id = (
                        _short_id(turn.span_id)
                        if turn.span_id
                        else (
                            turn.execution_id
                            if turn.execution_id != "------"
                            else session.session_id
                        )
                    )
                    raw_events.append((t, turn_id, "EXEC", summary))

            raw_events.append(
                (session.end_time, session.session_id, "AGENT_END", f"{session.status}")
            )

        raw_events.sort(key=lambda x: x[0])

        events = [
            TimelineEvent(time_ns=time_ns, span_id=span_id, event_type=kind, summary=summary)
            for time_ns, span_id, kind, summary in raw_events[:max_events]
        ]

        return TimelineData(
            total_events=len(raw_events),
            max_events=max_events,
            events=events,
        )

    async def find_first_error(self) -> str:
        """Navigate to the first error in the trace.

        Returns formatted output for the turn where the first error occurred,
        along with navigation hints.

        Returns:
            Formatted error details or "No errors found" message.
        """
        result = self._find_first_error()
        if not result:
            return "No errors found in trace."

        session, turn_idx, turn = result

        lines = [
            "# First Error",
            "",
            f"**Location:** Session `{session.session_id}` ({session.full_name}), Turn {turn_idx}",
            f"**Error Type:** {turn.error_type or 'Unknown'}",
            "",
            "## Error Message",
            "```",
            turn.error or "No error message",
            "```",
            "",
            "## Code That Caused Error",
            "```python",
            turn.code,
            "```",
            "",
            "## Navigation",
            f"→ {self._nav_hint_turn(session.session_id, turn_idx)} for full turn context",
            f"→ {self._nav_hint(session.session_id)} for all turns in this session",
        ]

        return "\n".join(lines)

    async def find_first_error_data(self) -> dict[str, Any]:
        """Structured first error info for programmatic use."""
        result = self._find_first_error()
        if not result:
            return {"error": "No errors found in trace."}
        session, turn_idx, turn = result
        return {
            "session_id": session.session_id,
            "session_name": session.full_name,
            "turn_index": turn_idx,
            "error_type": turn.error_type,
            "error_message": turn.error,
            "code": turn.code,
        }

    # =========================================================================
    # Raw Span Access
    # =========================================================================

    async def find_span(self, span_id: str, *, json_output: bool = False) -> str:
        """Find a span by ID and show it with navigation breadcrumbs.

        Searches all sessions for a turn matching the span ID. Shows the turn
        with context about where it is in the trace (session, position, total
        turns) so an agent can navigate up/down.

        Args:
            span_id: Full span_id or prefix to search for.
            json_output: If True, return structured JSON.

        Returns:
            Turn details with navigation breadcrumbs, or raw span if not
            associated with a turn.
        """
        # Search sessions for the span
        for session in self._all_sessions:
            for i, turn in enumerate(session.turns):
                if turn.span_id and (turn.span_id == span_id or turn.span_id.startswith(span_id)):
                    total = len(session.turns)
                    sid = _short_id(session.session_id)
                    header = (
                        f"# Span {span_id[:8]} → session {sid} "
                        f"({session.agent_name}.{session.method_name}), "
                        f"turn {i} of {total}\n"
                    )
                    nav = "\n## Navigation\n"
                    nav += f"→ trace-explorer ... --session {sid}              # all turns in this session\n"
                    if i > 0:
                        nav += f"→ trace-explorer ... --session {sid} --turn {i - 1}  # previous turn\n"
                    if i < total - 1:
                        nav += f"→ trace-explorer ... --session {sid} --turn {i + 1}  # next turn\n"

                    if json_output:
                        data = await self.get_turn_data(session.session_id, i)
                        result = {
                            "span_id": span_id,
                            "session_id": session.session_id,
                            "session_short_id": sid,
                            "agent": f"{session.agent_name}.{session.method_name}",
                            "turn_index": i,
                            "total_turns": total,
                            "turn": data,
                        }
                        return json.dumps(result, indent=2, default=str)

                    return header + await self.get_turn(session.session_id, i) + nav

        # Fallback: check raw spans
        raw = await self.get_raw_span(span_id)
        if "not found" in raw.lower():
            return f"Span {span_id} not found in any session or raw spans."
        return f"# Span {span_id[:8]} (raw, not associated with a turn)\n{raw}"

    async def get_raw_span(self, span_id: str) -> str:
        """Get raw span data as formatted JSON.

        Useful for debugging trace structure or accessing attributes
        not exposed by the high-level API.

        Args:
            span_id: Full span_id or 6-char prefix.

        Returns:
            JSON-formatted span with all attributes, or error message.
        """
        if not self._raw_spans:
            return "No raw spans available (trace was loaded without span data)."

        # Build index if not exists
        if not hasattr(self, "_span_index"):
            self._span_index = {
                s.get("span_id", ""): s for s in self._raw_spans if s.get("span_id")
            }

        # Find span by full or partial ID
        matched_span = None
        matched_id = None

        for sid, span in self._span_index.items():
            if sid == span_id:
                matched_span = span
                matched_id = sid
                break
            if sid.startswith(span_id) or span_id.startswith(sid[:6]):
                matched_span = span
                matched_id = sid
                break

        if not matched_span or matched_id is None:
            # List available spans for debugging
            available = list(self._span_index.keys())[:10]
            hint = f"Available span IDs (first 10): {[s[:6] for s in available]}"
            return f"No span found matching '{span_id}'\n{hint}"

        # Format as indented JSON
        lines = [
            f"# Raw Span: {matched_id[:6]}",
            f"Full ID: {matched_id}",
            "",
            "```json",
            json.dumps(matched_span, indent=2, default=str),
            "```",
        ]

        return "\n".join(lines)

    async def get_raw_span_data(self, span_id: str) -> dict[str, Any]:
        """Structured raw span data (dict) or error."""
        if not self._raw_spans:
            return {"error": "No raw spans available (trace was loaded without span data)."}

        if not hasattr(self, "_span_index"):
            self._span_index = {
                s.get("span_id", ""): s for s in self._raw_spans if s.get("span_id")
            }

        for sid, span in self._span_index.items():
            if sid == span_id or sid.startswith(span_id) or span_id.startswith(sid[:6]):
                return span

        available = list(self._span_index.keys())[:10]
        return {
            "error": f"No span found matching '{span_id}'",
            "available": [s[:6] for s in available],
        }

    async def get_raw_spans(self, session_id: str) -> str:
        """Get all raw spans for a session.

        Returns spans in chronological order with their relationships.

        Args:
            session_id: Session ID (6-char or full).

        Returns:
            JSON-formatted list of spans for the session.
        """
        session = self._find_session(session_id)
        if not session:
            return f"Session not found: {session_id}"

        if not self._raw_spans:
            return "No raw spans available."

        # Find all spans related to this session
        session_span_id = session.span_id

        # Collect spans that are the session span or descendants
        related_spans = []
        for span in self._raw_spans:
            sid = span.get("span_id", "")
            parent_id = span.get("parent_span_id", "")

            # Include the session span itself
            if sid == session_span_id:
                related_spans.append(span)
            # Include spans that are children of the session span
            elif parent_id == session_span_id:
                related_spans.append(span)
            # Include spans that reference this session (generation/execution)
            elif sid in [t.span_id for t in session.turns if hasattr(t, "span_id")]:
                related_spans.append(span)

        if not related_spans:
            return f"No raw spans found for session {session_id}"

        # Sort by start_time
        related_spans.sort(key=lambda s: s.get("start_time", 0))

        lines = [
            f"# Raw Spans for Session {session.session_id}",
            f"Session: {session.full_name}",
            f"Found {len(related_spans)} spans",
            "",
        ]

        for span in related_spans:
            sid = span.get("span_id", "")[:6]
            name = span.get("name", "unnamed")
            kind = span.get("attributes", {}).get("openinference.span.kind", "")
            lines.append(f"## {sid} - {name} ({kind})")
            lines.append("```json")
            lines.append(json.dumps(span, indent=2, default=str))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Trace Comparison / Diff
    # =========================================================================

    async def compare(self, other: TraceExplorer) -> str:
        """Compare this trace with another trace.

        Useful for regression analysis - comparing a failing MR trace
        with a passing main branch trace.

        Compares:
        - Session count and structure
        - Call graph (agents called, order)
        - Turn-by-turn prompt content differences
        - First point of divergence

        Args:
            other: Another TraceExplorer instance to compare against.

        Returns:
            Formatted diff report.
        """
        return await TraceExplorer.diff(self, other)

    async def compare_data(self, other: TraceExplorer) -> dict[str, Any]:
        """Structured comparison data for programmatic diffing."""
        return await TraceExplorer.diff_data(self, other)

    @classmethod
    async def diff(cls, trace1: TraceExplorer, trace2: TraceExplorer) -> str:
        """Compare two traces and show differences.

        Args:
            trace1: First trace (typically the one being tested/MR).
            trace2: Second trace (typically the baseline/main).

        Returns:
            Formatted diff report showing:
            - Summary table with metrics
            - Call graph comparison
            - First divergence point
            - Prompt differences if any
        """
        lines = ["# Trace Comparison", ""]

        # File names
        file1 = Path(trace1.trace_file).name
        file2 = Path(trace2.trace_file).name

        # Get all sessions for comparison
        sessions1 = trace1._all_sessions
        sessions2 = trace2._all_sessions

        # Calculate totals
        total_turns1 = sum(len(s.turns) for s in sessions1)
        total_turns2 = sum(len(s.turns) for s in sessions2)

        # Determine status
        status1 = "ERROR" if any(s.status == "ERROR" for s in sessions1) else "OK"
        status2 = "ERROR" if any(s.status == "ERROR" for s in sessions2) else "OK"

        # Eval status if available
        eval_status1 = "N/A"
        eval_status2 = "N/A"
        if trace1.eval_result:
            eval_status1 = "PASS" if trace1.eval_result.passed else "FAIL"
        if trace2.eval_result:
            eval_status2 = "PASS" if trace2.eval_result.passed else "FAIL"

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Trace 1 | Trace 2 |")
        lines.append("|--------|---------|---------|")
        lines.append(f"| File | {file1} | {file2} |")
        lines.append(f"| Sessions | {len(sessions1)} | {len(sessions2)} |")
        lines.append(f"| Total Turns | {total_turns1} | {total_turns2} |")
        lines.append(f"| Status | {status1} | {status2} |")
        lines.append(f"| Eval | {eval_status1} | {eval_status2} |")
        lines.append("")

        # Call graph comparison
        lines.append("## Call Graphs")
        lines.append("")

        # Build call graph summaries
        def _build_call_graph_lines(sessions: list[AgentSession], prefix: str = "") -> list[str]:
            result = []
            for session in sessions:
                status_mark = "[ERR]" if session.status == "ERROR" else "[OK]"
                turn_count = len(session.turns)
                duration = session.duration_ms
                indent = "  " * session.depth
                result.append(
                    f"{indent}├─ {session.full_name} ({turn_count}t, {duration:.0f}ms) {status_mark}"
                )
            return result

        graph1 = _build_call_graph_lines(sessions1)
        graph2 = _build_call_graph_lines(sessions2)

        lines.append("### Trace 1")
        lines.append("```")
        lines.extend(graph1 if graph1 else ["(no sessions)"])
        lines.append("```")
        lines.append("")

        lines.append("### Trace 2")
        lines.append("```")
        lines.extend(graph2 if graph2 else ["(no sessions)"])
        lines.append("```")
        lines.append("")

        # Find first divergence
        lines.append("## Divergence Analysis")
        lines.append("")

        # Match sessions by method name and position
        matched_pairs: list[tuple[AgentSession | None, AgentSession | None]] = []

        # Group by full_name
        by_name1: dict[str, list[AgentSession]] = {}
        by_name2: dict[str, list[AgentSession]] = {}

        for s in sessions1:
            by_name1.setdefault(s.full_name, []).append(s)
        for s in sessions2:
            by_name2.setdefault(s.full_name, []).append(s)

        all_names = set(by_name1.keys()) | set(by_name2.keys())

        for name in sorted(all_names):
            list1 = by_name1.get(name, [])
            list2 = by_name2.get(name, [])
            max_len = max(len(list1), len(list2))

            for i in range(max_len):
                s1 = list1[i] if i < len(list1) else None
                s2 = list2[i] if i < len(list2) else None
                matched_pairs.append((s1, s2))

        # Find first difference
        divergence_found = False
        for s1, s2 in matched_pairs:
            if s1 is None and s2 is not None:
                lines.append(
                    f"**Missing in Trace 1:** `{s2.full_name}` (session `{s2.session_id}`)"
                )
                divergence_found = True
                break
            elif s2 is None and s1 is not None:
                lines.append(
                    f"**Missing in Trace 2:** `{s1.full_name}` (session `{s1.session_id}`)"
                )
                divergence_found = True
                break
            elif s1 is not None and s2 is not None:
                # Compare turn counts
                if len(s1.turns) != len(s2.turns):
                    lines.append(f"**Turn Count Difference:** `{s1.full_name}`")
                    lines.append(f"- Trace 1: {len(s1.turns)} turns (session `{s1.session_id}`)")
                    lines.append(f"- Trace 2: {len(s2.turns)} turns (session `{s2.session_id}`)")
                    divergence_found = True
                    break

                # Compare turn-by-turn
                for i, (t1, t2) in enumerate(zip(s1.turns, s2.turns, strict=False)):
                    if type(t1) is not type(t2):
                        lines.append(f"**Turn Type Difference:** `{s1.full_name}` turn {i}")
                        lines.append(f"- Trace 1: {type(t1).__name__}")
                        lines.append(f"- Trace 2: {type(t2).__name__}")
                        divergence_found = True
                        break

                    if isinstance(t1, LLMTurn) and isinstance(t2, LLMTurn):
                        # Compare prompts for expression path differences
                        prompt_diffs = cls._find_prompt_diffs(t1, t2)
                        if prompt_diffs:
                            lines.append(f"**Prompt Difference:** `{s1.full_name}` turn {i}")
                            lines.append("")
                            for diff_desc in prompt_diffs[:3]:  # Limit to first 3
                                lines.append(f"- {diff_desc}")
                            lines.append("")
                            lines.append(
                                "→ Compare with "
                                f"`get_turn('{s1.session_id}', {i})` vs "
                                f"`get_turn('{s2.session_id}', {i})`"
                            )
                            divergence_found = True
                            break

                        # Compare tool calls
                        tc1 = [tc.function_name for tc in t1.tool_calls]
                        tc2 = [tc.function_name for tc in t2.tool_calls]
                        if tc1 != tc2:
                            lines.append(f"**Tool Call Difference:** `{s1.full_name}` turn {i}")
                            lines.append(f"- Trace 1: {tc1}")
                            lines.append(f"- Trace 2: {tc2}")
                            divergence_found = True
                            break

                    elif isinstance(t1, ExecutionTurn) and isinstance(t2, ExecutionTurn):
                        # Compare execution status
                        if (t1.error is not None) != (t2.error is not None):
                            lines.append(
                                f"**Execution Status Difference:** `{s1.full_name}` turn {i}"
                            )
                            lines.append(f"- Trace 1: {'ERROR' if t1.error else 'OK'}")
                            lines.append(f"- Trace 2: {'ERROR' if t2.error else 'OK'}")
                            divergence_found = True
                            break

                if divergence_found:
                    break

        if not divergence_found:
            if len(sessions1) == len(sessions2) and total_turns1 == total_turns2:
                lines.append("No structural differences found. Traces appear equivalent.")
            else:
                lines.append(
                    "Session/turn structures differ but no specific divergence point identified."
                )

        lines.append("")

        # Prompt expression differences (even if divergence found earlier)
        prompt_diffs = cls._collect_prompt_expr_diffs(matched_pairs)
        if prompt_diffs:
            lines.append("## Prompt Expression Differences")
            lines.append("")
            for item in prompt_diffs[:10]:
                lines.append(f"- `{item['session']}` turn {item['turn_index']}: {item['diff']}")
            if len(prompt_diffs) > 10:
                lines.append(f"... ({len(prompt_diffs) - 10} more)")
            lines.append("")

        # Navigation hints
        lines.append("## Next Steps")
        lines.append("")
        if sessions1:
            lines.append(f"- Trace 1: `get_session('{sessions1[0].session_id}')`")
        if sessions2:
            lines.append(f"- Trace 2: `get_session('{sessions2[0].session_id}')`")

        return "\n".join(lines)

    @classmethod
    async def diff_data(cls, trace1: TraceExplorer, trace2: TraceExplorer) -> dict[str, Any]:
        """Structured diff data for programmatic consumption."""
        sessions1 = trace1._all_sessions
        sessions2 = trace2._all_sessions
        total_turns1 = sum(len(s.turns) for s in sessions1)
        total_turns2 = sum(len(s.turns) for s in sessions2)
        status1 = "ERROR" if any(s.status == "ERROR" for s in sessions1) else "OK"
        status2 = "ERROR" if any(s.status == "ERROR" for s in sessions2) else "OK"
        eval_status1 = (
            "PASS"
            if trace1.eval_result and trace1.eval_result.passed
            else "FAIL"
            if trace1.eval_result
            else "N/A"
        )
        eval_status2 = (
            "PASS"
            if trace2.eval_result and trace2.eval_result.passed
            else "FAIL"
            if trace2.eval_result
            else "N/A"
        )

        def _call_graph_data(sessions: list[AgentSession]) -> list[dict[str, Any]]:
            return [
                {
                    "session_id": s.session_id,
                    "full_name": s.full_name,
                    "depth": s.depth,
                    "status": s.status,
                    "turn_count": len(s.turns),
                    "duration_ms": s.duration_ms,
                }
                for s in sessions
            ]

        # Build matched pairs for prompt diffs
        matched_pairs: list[tuple[AgentSession | None, AgentSession | None]] = []
        by_name1: dict[str, list[AgentSession]] = {}
        by_name2: dict[str, list[AgentSession]] = {}
        for s in sessions1:
            by_name1.setdefault(s.full_name, []).append(s)
        for s in sessions2:
            by_name2.setdefault(s.full_name, []).append(s)
        all_names = set(by_name1.keys()) | set(by_name2.keys())
        for name in sorted(all_names):
            list1 = by_name1.get(name, [])
            list2 = by_name2.get(name, [])
            max_len = max(len(list1), len(list2))
            for i in range(max_len):
                s1 = list1[i] if i < len(list1) else None
                s2 = list2[i] if i < len(list2) else None
                matched_pairs.append((s1, s2))

        prompt_diffs = cls._collect_prompt_expr_diffs(matched_pairs)

        return {
            "summary": {
                "file_1": Path(trace1.trace_file).name,
                "file_2": Path(trace2.trace_file).name,
                "sessions_1": len(sessions1),
                "sessions_2": len(sessions2),
                "total_turns_1": total_turns1,
                "total_turns_2": total_turns2,
                "status_1": status1,
                "status_2": status2,
                "eval_1": eval_status1,
                "eval_2": eval_status2,
            },
            "call_graphs": {
                "trace_1": _call_graph_data(sessions1),
                "trace_2": _call_graph_data(sessions2),
            },
            "prompt_expression_differences": prompt_diffs,
        }

    @classmethod
    def _find_prompt_diffs(cls, turn1: LLMTurn, turn2: LLMTurn) -> list[str]:
        """Find differences in prompt content between two LLM turns.

        Looks for:
        - Different expression paths in XML tags (e.g., expr="self.events[0].prompt")
        - Missing sections
        - Content differences

        Returns list of difference descriptions.
        """
        diffs = []

        # Extract XML expr attributes from both prompts
        def extract_expr_paths(messages: list[LLMMessage]) -> dict[str, set[str]]:
            paths: dict[str, set[str]] = {}
            for msg in messages:
                for match in re.finditer(r"<(\w+)[^>]*\bexpr=\"([^\"]+)\"", msg.content, re.DOTALL):
                    tag = match.group(1)
                    expr = match.group(2)
                    paths.setdefault(tag, set()).add(expr)
            return paths

        paths1 = extract_expr_paths(turn1.messages)
        paths2 = extract_expr_paths(turn2.messages)

        # Compare paths
        all_tags = set(paths1.keys()) | set(paths2.keys())

        for tag in sorted(all_tags):
            p1 = paths1.get(tag, set())
            p2 = paths2.get(tag, set())

            if p1 and p2:
                if p1 != p2:
                    diffs.append(f"`<{tag}>` expr differs: {sorted(p1)} vs {sorted(p2)}")
            elif p1 and not p2:
                diffs.append(f"`<{tag}>` present in Trace 1 only")
            elif p2 and not p1:
                diffs.append(f"`<{tag}>` present in Trace 2 only")

        # Compare message counts
        if len(turn1.messages) != len(turn2.messages):
            diffs.append(f"Message count: {len(turn1.messages)} vs {len(turn2.messages)}")

        return diffs

    @classmethod
    def _collect_prompt_expr_diffs(
        cls,
        matched_pairs: list[tuple[AgentSession | None, AgentSession | None]],
    ) -> list[dict[str, Any]]:
        """Collect prompt expr differences across matched sessions."""
        diffs: list[dict[str, Any]] = []
        for s1, s2 in matched_pairs:
            if not s1 or not s2:
                continue
            for i, (t1, t2) in enumerate(zip(s1.turns, s2.turns, strict=False)):
                if isinstance(t1, LLMTurn) and isinstance(t2, LLMTurn):
                    prompt_diffs = cls._find_prompt_diffs(t1, t2)
                    for diff in prompt_diffs:
                        diffs.append(
                            {
                                "session": s1.full_name,
                                "session_id_1": s1.session_id,
                                "session_id_2": s2.session_id,
                                "turn_index": i,
                                "diff": diff,
                            }
                        )
        return diffs

    # =========================================================================
    # Removed Methods (Historical Notes)
    # =========================================================================
    # what_went_wrong() - removed, use get_overview(concise=True) + get_errors() instead


# =============================================================================
# CLI
# =============================================================================


async def _handle_experiment_errors(
    base_url: str, experiment_id: str, *, failed_only: bool = True
) -> None:
    """Load each session in an experiment and print errors across all traces."""
    import urllib.parse

    import httpx

    base_url = base_url.rstrip("/")
    encoded_eid = urllib.parse.quote(experiment_id, safe="")

    async with httpx.AsyncClient(timeout=30) as _client:
        try:
            _resp = await _client.get(f"{base_url}/api/eval/experiment/{encoded_eid}/tests")
            if _resp.status_code == 404:
                print(f"Error: Experiment not found: {experiment_id}", file=sys.stderr)
                sys.exit(1)
            _resp.raise_for_status()
            tests_data = _resp.json()
        except httpx.ConnectError as e:
            print(f"Error: Cannot reach viewer at {base_url}: {e}", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error: HTTP {e.response.status_code} from viewer", file=sys.stderr)
            sys.exit(1)
    tests = tests_data.get("tests", [])
    if failed_only:
        tests = [t for t in tests if t.get("passed") == False]  # noqa: E712 — explicit False, not None/missing

    if not tests:
        print("No failed tests found." if failed_only else "No tests found.")
        return

    print(f"# Errors across experiment: {experiment_id}")
    print(f"# Checking {len(tests)} {'failed ' if failed_only else ''}session(s)...\n")

    any_errors = False
    load_errors = 0
    for t in tests:
        sid = t.get("session_id", "")
        name = t.get("display_name") or t.get("test_name") or t.get("test_id", sid)
        if not sid:
            continue
        try:
            trace = await TraceExplorer.from_viewer(base_url, sid)
        except (ValueError, ConnectionError) as e:
            load_errors += 1
            print(f"## {name}\n  Error loading trace: {e}\n")
            continue

        errors_data = await trace.get_errors_data()
        if errors_data["count"] > 0:
            any_errors = True
            print(f"## {name}")
            print(
                f"   trace-explorer --viewer {shlex.quote(base_url)} --session-id {shlex.quote(sid)}"
            )
            print(await trace.get_errors())
            print()

    if not any_errors:
        if load_errors:
            print(f"No errors found ({load_errors} session(s) failed to load).")
        else:
            print("No errors found in any session.")


async def _handle_experiment_search(base_url: str, experiment_id: str, pattern: str) -> None:
    """Search for a pattern across all sessions in an experiment."""
    import urllib.parse

    import httpx

    base_url = base_url.rstrip("/")
    encoded_eid = urllib.parse.quote(experiment_id, safe="")

    async with httpx.AsyncClient(timeout=30) as _client:
        try:
            _resp = await _client.get(f"{base_url}/api/eval/experiment/{encoded_eid}/tests")
            if _resp.status_code == 404:
                print(f"Error: Experiment not found: {experiment_id}", file=sys.stderr)
                sys.exit(1)
            _resp.raise_for_status()
            tests_data = _resp.json()
        except httpx.ConnectError as e:
            print(f"Error: Cannot reach viewer at {base_url}: {e}", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error: HTTP {e.response.status_code} from viewer", file=sys.stderr)
            sys.exit(1)
    tests = tests_data.get("tests", [])
    if not tests:
        print("No tests found.")
        return

    print(f"# Search '{pattern}' across experiment: {experiment_id}")
    print(f"# Checking {len(tests)} session(s)...\n")

    total_matches = 0
    sessions_with_matches = 0
    load_errors = 0
    for t in tests:
        sid = t.get("session_id", "")
        name = t.get("display_name") or t.get("test_name") or t.get("test_id", sid)
        if not sid:
            continue
        try:
            trace = await TraceExplorer.from_viewer(base_url, sid)
        except (ValueError, ConnectionError) as e:
            load_errors += 1
            print(f"## {name}\n  Error loading trace: {e}\n")
            continue

        data = await trace.search_data(pattern)
        if data.match_count > 0:
            total_matches += data.match_count
            sessions_with_matches += 1
            print(f"## {name}  ({data.match_count} match(es))")
            cmd = (
                f"   trace-explorer --viewer {shlex.quote(base_url)}"
                f" --session-id {shlex.quote(sid)} --search {shlex.quote(pattern)}"
            )
            print(cmd)
            # Print concise match list without nav hint
            for result in data.matches[:10]:
                match_text = result.match_text.replace("\n", " ").strip()
                if len(match_text) > 80:
                    match_text = match_text[:77] + "..."
                print(
                    f"  [{result.session_id} t{result.turn_index}] {result.location}: ...{match_text}..."
                )
            if data.match_count > 10:
                print(f"  ... and {data.match_count - 10} more matches")
            print()

    if total_matches == 0:
        suffix = f" ({load_errors} session(s) failed to load)" if load_errors else ""
        print(f"No matches found for '{pattern}' in any session{suffix}.")
    else:
        print(f"Total: {total_matches} match(es) across {sessions_with_matches} session(s).")


async def _handle_experiment_failures(base_url: str, experiment_id: str) -> None:
    """Show eval failure reasons (wrong answer, wrong schema) across all failed sessions.

    Complements --errors (which shows Python exceptions) by surfacing the scorer's
    failure reason and the agent's actual output — useful for wrong-answer failures
    that don't produce any exception.
    """
    import urllib.parse

    import httpx

    base_url = base_url.rstrip("/")
    encoded_eid = urllib.parse.quote(experiment_id, safe="")

    async with httpx.AsyncClient(timeout=30) as _client:
        try:
            _resp = await _client.get(f"{base_url}/api/eval/experiment/{encoded_eid}/tests")
            if _resp.status_code == 404:
                print(f"Error: Experiment not found: {experiment_id}", file=sys.stderr)
                sys.exit(1)
            _resp.raise_for_status()
            tests_data = _resp.json()
        except httpx.ConnectError as e:
            print(f"Error: Cannot reach viewer at {base_url}: {e}", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error: HTTP {e.response.status_code} from viewer", file=sys.stderr)
            sys.exit(1)
    tests = tests_data.get("tests", [])
    failed_tests = [t for t in tests if t.get("passed") == False]  # noqa: E712

    if not failed_tests:
        print("No failed tests found.")
        return

    print(f"# Eval failures: {experiment_id}")
    print(f"# {len(failed_tests)} failed session(s)\n")

    load_errors = 0
    for t in failed_tests:
        sid = t.get("session_id", "")
        name = t.get("display_name") or t.get("test_name") or t.get("test_id", sid)
        if not sid:
            continue
        try:
            trace = await TraceExplorer.from_viewer(base_url, sid)
        except (ValueError, ConnectionError) as e:
            load_errors += 1
            print(f"## {name}\n  Error loading trace: {e}\n")
            continue

        er = trace.eval_result
        if er is None:
            print(f"## {name}\n  (no eval result in trace)\n")
            continue

        score = er.weighted_score if er.weighted_score is not None else 0.0

        # Find failure reason from first failing scorer, fall back to error field
        reason = ""
        scorer_name = ""
        if er.scores:
            for sname, sdata in er.scores.items():
                if not sdata.passed:
                    scorer_name = sname
                    reason = sdata.reasoning or ""
                    break
            if not reason:
                sname, sdata = next(iter(er.scores.items()))
                scorer_name, reason = sname, sdata.reasoning or ""
        if not reason and er.error:
            reason = er.error

        # Format actual output: show in full if short, truncate if long
        output_str = ""
        if er.output is not None:
            raw = str(er.output).strip()
            lines = raw.splitlines()
            if len(lines) <= 4:
                output_str = raw
            else:
                output_str = "\n".join(lines[:4]) + f"\n  ... ({len(lines)} lines total)"

        print(f"## {name}  [score={score:.2f}]")
        print(f"   trace-explorer --viewer {shlex.quote(base_url)} --session-id {shlex.quote(sid)}")
        print(
            f"   trace-explorer --viewer {shlex.quote(base_url)} --session-id {shlex.quote(sid)} --eval"
        )
        if reason:
            if len(reason) > 200:
                reason = reason[:197] + "..."
            scorer_label = f" ({scorer_name})" if scorer_name else ""
            print(f"   Reason{scorer_label}: {reason}")
        if output_str:
            output_lines = output_str.splitlines()
            print(f"   Output: {output_lines[0]}")
            for line in output_lines[1:]:
                print(f"           {line}")
        print()

    if load_errors:
        print(f"({load_errors} session(s) failed to load)")


async def _handle_experiment(
    base_url: str, experiment_id: str, *, json_output: bool = False
) -> None:
    """Fetch and display experiment summary from the viewer API."""
    import urllib.parse

    import httpx

    base_url = base_url.rstrip("/")
    encoded_eid = urllib.parse.quote(experiment_id, safe="")

    # Fetch summary
    async with httpx.AsyncClient(timeout=30) as _client:
        try:
            _resp = await _client.get(f"{base_url}/api/eval/experiment/{encoded_eid}/summary")
            if _resp.status_code == 404:
                print(f"Error: Experiment not found: {experiment_id}", file=sys.stderr)
                sys.exit(1)
            _resp.raise_for_status()
            summary = _resp.json()
        except httpx.ConnectError as e:
            print(f"Error: Cannot reach viewer at {base_url}: {e}", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error: HTTP {e.response.status_code} from viewer", file=sys.stderr)
            sys.exit(1)
    # Fetch test results
    tests_data: dict = {"tests": []}
    async with httpx.AsyncClient(timeout=30) as _client:
        try:
            _resp = await _client.get(f"{base_url}/api/eval/experiment/{encoded_eid}/tests")
            _resp.raise_for_status()
            tests_data = _resp.json()
        except httpx.HTTPError:
            pass  # Fall back to empty tests

    tests = tests_data.get("tests", [])

    if json_output:
        print(json.dumps({"summary": summary, "tests": tests}, indent=2, default=str))
        return

    # Format summary
    overall = summary.get("overall", {})
    total = overall.get("total", 0)
    passed = overall.get("passed", 0)
    failed = total - passed
    rate = overall.get("success_rate", 0)
    avg_score = overall.get("avg_score", 0)

    print(f"# Experiment: {experiment_id}")
    print()
    print(
        f"Total: {total} | Passed: {passed} | Failed: {failed} | Success rate: {rate:.0%} | Avg score: {avg_score:.2f}"
    )

    # By model
    by_model = summary.get("by_model", {})
    if by_model:
        print()
        print("## By Model")
        for model, stats in sorted(by_model.items()):
            m_total = stats.get("total", 0)
            m_passed = stats.get("passed", 0)
            m_rate = m_passed / m_total if m_total else 0
            print(f"  {model}: {m_passed}/{m_total} ({m_rate:.0%})")

    # By test type
    by_type = summary.get("by_test_type", {})
    if by_type:
        print()
        print("## By Test Type")
        for ttype, stats in sorted(by_type.items()):
            t_total = stats.get("total", 0)
            t_passed = stats.get("passed", 0)
            t_rate = t_passed / t_total if t_total else 0
            print(f"  {ttype}: {t_passed}/{t_total} ({t_rate:.0%})")

    # Failed tests
    failed_tests = [t for t in tests if t.get("passed") == False]  # noqa: E712 — explicit False, not None/missing
    if failed_tests:
        print()
        print(f"## Failed Tests ({len(failed_tests)})")
        for t in failed_tests[:20]:  # Show max 20
            sid = t.get("session_id", "?")
            name = t.get("display_name") or t.get("test_name") or t.get("test_id", "?")
            score = t.get("score")
            score_str = f" score={score:.2f}" if score is not None else ""
            print(f"  {name}{score_str}")
            if sid and sid != "?":
                print(f"    trace-explorer --viewer {base_url} --session-id '{sid}'")
        if len(failed_tests) > 20:
            print(f"  ... and {len(failed_tests) - 20} more")

    # Navigation
    print()
    print("## Navigation")
    print("→ Drill into a trace:")
    print(f"  trace-explorer --viewer {base_url} --session-id '<session_id>'")
    print("→ Show this as JSON:")
    print(f"  trace-explorer --viewer {base_url} --experiment '{experiment_id}' --json")


def _handle_install_skill() -> None:
    """Install the trace-explorer Claude Code skill to ~/.claude/skills/.

    Idempotent: prints 'already up to date' if the installed skill matches
    the current version, 'updated' if it was changed.
    """
    import hashlib
    import shutil

    skill_src = Path(__file__).parent / "skill" / "SKILL.md"
    if not skill_src.exists():
        print(f"Error: Skill file not found: {skill_src}", file=sys.stderr)
        sys.exit(1)

    skill_dir = Path.home() / ".claude" / "skills" / "trace-explorer"
    skill_dir.mkdir(parents=True, exist_ok=True)
    dest = skill_dir / "SKILL.md"

    src_hash = hashlib.md5(skill_src.read_bytes()).hexdigest()
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == src_hash:
        print(f"trace-explorer skill already up to date ({dest})")
        return

    action = "Updated" if dest.exists() else "Installed"
    shutil.copy2(skill_src, dest)
    print(f"{action} trace-explorer skill: {dest}")


def main() -> None:
    """Command-line interface for trace_explorer."""
    import asyncio

    asyncio.run(_async_main())


async def _try_thin_client(viewer_url: str, session_id: str) -> TraceExplorerClient | None:
    """Try to connect via thin-client endpoints; return None if unavailable."""
    import httpx

    from nooa.trace_explorer.client import TraceExplorerClient

    base = viewer_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{base}/api/explorer/summary",
                params={"session_id": session_id},
            )
            if resp.status_code == 200:
                return TraceExplorerClient(viewer_url, session_id)
    except httpx.RequestError:
        pass
    return None


async def _async_main() -> None:
    """Async implementation of the CLI."""
    import argparse

    _cli_mode.set(True)

    parser = argparse.ArgumentParser(
        prog="trace-explorer",
        description="Explore agent traces from .jsonl files or a viewer API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (file mode):
  trace-explorer trace.jsonl                    # Show overview
  trace-explorer trace.jsonl --verbose          # Show full details
  trace-explorer trace.jsonl --session abc123   # Show session details
  trace-explorer trace.jsonl --session abc123 --turn 0  # Show turn details
  trace-explorer trace.jsonl --errors           # Show all errors
  trace-explorer trace.jsonl --search "pattern" # Search for pattern
  trace-explorer trace.jsonl --diff other.jsonl # Compare two traces
  trace-explorer trace.jsonl --json             # Structured JSON output
  trace-explorer trace.jsonl --root-generation 1  # Select root generation
  trace-explorer trace.jsonl --timeline         # Show event timeline
  trace-explorer trace.jsonl --first-error      # Jump to first error
  trace-explorer trace.jsonl --raw abc123       # Show raw span data
  trace-explorer trace.jsonl --api-help         # Show API guide

Examples (viewer mode):
  trace-explorer --viewer http://localhost:5001 --session-id abc123
  trace-explorer --viewer http://localhost:5001 --session-id abc123 --errors
  trace-explorer --viewer http://localhost:5001 --session-id abc123 --session abc123 --turn 0
  trace-explorer --viewer http://localhost:5001 --session-id abc123 --span-id deadbeef

Examples (experiment mode):
  trace-explorer --viewer http://localhost:5001 --experiment my_eval_run
  trace-explorer --viewer http://localhost:5001 --experiment my_eval_run --errors    # Python exceptions
  trace-explorer --viewer http://localhost:5001 --experiment my_eval_run --failures  # Wrong answers
  trace-explorer --viewer http://localhost:5001 --experiment my_eval_run --search "task_dir"
  trace-explorer --viewer http://localhost:5001 --experiment my_eval_run --json
        """,
    )

    # Data source (mutually exclusive: file or viewer)
    parser.add_argument("trace_file", nargs="?", help="Path to .jsonl trace file")
    parser.add_argument(
        "--viewer", metavar="URL", help="Viewer base URL (e.g., http://localhost:5001)"
    )
    parser.add_argument(
        "--session-id", metavar="ID", help="Session ID to load from viewer (requires --viewer)"
    )
    parser.add_argument(
        "--span-id",
        metavar="ID",
        help="Jump to a specific span (requires --viewer and --session-id)",
    )
    parser.add_argument(
        "--experiment",
        metavar="ID",
        help="Show experiment summary with test results (requires --viewer)",
    )
    parser.add_argument(
        "--install-skill",
        action="store_true",
        help="Install Claude Code skill to ~/.claude/skills/",
    )

    # Display options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full details (concise=False)"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress parser warnings")
    parser.add_argument("--session", "-s", metavar="ID", help="Show details for session ID")
    parser.add_argument(
        "--turn", "-t", type=int, metavar="N", help="Show turn N (requires --session)"
    )
    parser.add_argument("--errors", "-e", action="store_true", help="Show all errors")
    parser.add_argument(
        "--failures",
        action="store_true",
        help="Show eval failure reasons across all failed sessions (requires --experiment)",
    )
    parser.add_argument("--eval", action="store_true", help="Show evaluation context")
    parser.add_argument("--search", metavar="PATTERN", help="Search for regex pattern")
    parser.add_argument("--diff", metavar="OTHER", help="Compare with another trace file")
    parser.add_argument("--timeline", action="store_true", help="Show chronological event timeline")
    parser.add_argument("--first-error", action="store_true", help="Jump to the first error")
    parser.add_argument("--raw", metavar="SPAN_ID", help="Show raw span data for given span ID")
    parser.add_argument("--api-help", action="store_true", help="Show API guide")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    parser.add_argument(
        "--root-generation",
        type=int,
        metavar="N",
        help="Select Nth root generation (0-based) when multiple roots exist",
    )
    parser.add_argument(
        "--harness",
        action="store_true",
        help="Show harness telemetry (model-helping patterns tracked via OTLP spans)",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Hide model reasoning_content in session and turn output",
    )

    args = parser.parse_args()

    # Handle --api-help and --install-skill without requiring trace file
    if args.api_help:
        print(TraceExplorer.__doc__ or "No documentation available.")
        return

    if args.install_skill:
        _handle_install_skill()
        return

    # Validate data source arguments
    if args.viewer:
        if not args.session_id and not args.experiment:
            print("Error: --viewer requires --session-id or --experiment", file=sys.stderr)
            sys.exit(1)
        if args.trace_file:
            print("Error: Cannot specify both trace_file and --viewer", file=sys.stderr)
            sys.exit(1)
    elif args.session_id or args.span_id or args.experiment:
        print("Error: --session-id, --span-id, and --experiment require --viewer", file=sys.stderr)
        sys.exit(1)
    elif not args.trace_file:
        parser.print_help()
        sys.exit(1)

    # Set quiet mode before loading trace
    if args.quiet:
        set_quiet_mode(True)

    # Handle --experiment: show experiment summary (no trace loading needed)
    if args.experiment:
        exclusive = [args.errors, args.failures, bool(args.search)]
        if sum(exclusive) > 1:
            print(
                "Error: --errors, --failures, and --search are mutually exclusive with --experiment",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.errors:
            await _handle_experiment_errors(args.viewer, args.experiment)
        elif args.failures:
            await _handle_experiment_failures(args.viewer, args.experiment)
        elif args.search:
            await _handle_experiment_search(args.viewer, args.experiment, args.search)
        else:
            await _handle_experiment(args.viewer, args.experiment, json_output=args.json)
        return

    # Load trace from file or viewer
    trace: Any
    try:
        if args.viewer:
            # Try thin-client path first (server-side execution, much faster)
            if (
                not args.json
                and not args.diff
                and not args.raw
                and not args.harness
                and not args.root_generation
            ):
                trace = await _try_thin_client(args.viewer, args.session_id)
            else:
                trace = None
            # Fall back to full loading if thin client unavailable
            if trace is None:
                trace = await TraceExplorer.from_viewer(
                    args.viewer,
                    args.session_id,
                    root_generation_index=args.root_generation,
                )
        else:
            trace = await TraceExplorer.from_file(
                args.trace_file,
                root_generation_index=args.root_generation,
            )
    except FileNotFoundError:
        print(f"Error: File not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle commands
    def _print_json(data: Any) -> None:
        print(json.dumps(data, indent=2, default=str))

    # --span-id takes priority: jump directly to that span
    if args.span_id:
        print(await trace.find_span(args.span_id, json_output=args.json))
        return

    if args.diff:
        # Load second trace for comparison
        try:
            trace2 = await TraceExplorer.from_file(
                args.diff, root_generation_index=args.root_generation
            )
        except FileNotFoundError:
            print(f"Error: File not found: {args.diff}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            _print_json(await trace.compare_data(trace2))
        else:
            print(await trace.compare(trace2))
    elif args.raw:
        if args.json:
            _print_json(await trace.get_raw_span_data(args.raw))
        else:
            print(await trace.get_raw_span(args.raw))
    elif args.timeline:
        if args.json:
            _print_json(await trace.get_timeline_data())
        else:
            print(await trace.get_timeline())
    elif args.first_error:
        if args.json:
            _print_json(await trace.find_first_error_data())
        else:
            print(await trace.find_first_error())
    elif args.turn is not None:
        if not args.session:
            print("Error: --turn requires --session", file=sys.stderr)
            sys.exit(1)
        if args.json:
            turn = await trace.get_turn_data(args.session, args.turn)
            _print_json(turn.to_dict() if turn else None)
        else:
            print(
                await trace.get_turn(
                    args.session, args.turn, include_reasoning=not args.no_reasoning
                )
            )
    elif args.session:
        if args.json:
            _print_json((await trace.get_session_data(args.session)).to_dict())
        else:
            print(
                await trace.get_session(
                    args.session, concise=not args.verbose, include_reasoning=not args.no_reasoning
                )
            )
    elif args.errors:
        if args.json:
            _print_json(await trace.get_errors_data())
        else:
            print(await trace.get_errors())
    elif args.eval:
        if args.json:
            _print_json(await trace.get_eval_context_data())
        else:
            print(await trace.get_eval_context())
    elif args.harness:
        session_id = args.session if args.session else None
        if args.json:
            _print_json(await trace.get_harness_telemetry_data(session_id))
        else:
            print(await trace.get_harness_telemetry(session_id))
    elif args.search:
        if args.json:
            _print_json(await trace.search_data(args.search))
        else:
            print(await trace.search(args.search, concise=not args.verbose))
    else:
        # Default: show overview
        if args.json:
            overview = await trace.get_overview_data()
            _print_json(overview.to_dict() if overview else None)
        else:
            print(await trace.get_overview(concise=not args.verbose))


if __name__ == "__main__":
    main()
