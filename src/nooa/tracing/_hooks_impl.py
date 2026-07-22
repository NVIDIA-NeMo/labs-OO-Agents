# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry hooks implementation following OpenInference conventions."""

import contextlib
import difflib
import inspect
import json
import os
import time
import traceback
from contextvars import ContextVar
from typing import Any

from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from nooa.agentdoc import TruncatingStringIO, truncating_pformat

# Context variable for per-async-context active span tracking
# This prevents context leakage during concurrent execution (e.g., parallel eval samples)
_context_active_spans: ContextVar[dict[str, Span] | None] = ContextVar(
    "context_active_spans", default=None
)

# Tracks the code_execution span currently running in THIS async task, paired with the
# call_id of the agent whose generated code is executing. Lets a method invoked directly
# from that code (e.g. ``self.submit()`` inside ``execute_python``) nest under the
# execution span rather than the enclosing agent-method span — reflecting that the call
# happened inside that execution. Per-task ContextVar, so it is concurrency-safe under
# asyncio.gather (each task gets its own value; child tasks inherit at spawn time).
_current_code_execution: ContextVar[tuple[Span, str | None] | None] = ContextVar(
    "current_code_execution", default=None
)


def _get_active_spans() -> dict[str, Span]:
    """Get the active spans dict for the current async context."""
    spans = _context_active_spans.get()
    if spans is None:
        spans = {}
        _context_active_spans.set(spans)
    return spans


def end_active_spans(reason: str = "timeout") -> int:
    """End all active (un-ended) spans with an error status.

    Call this before shutdown_traces() when a task is cancelled or timed out,
    so that OTel SDK can export the spans. Without ending them first,
    force_flush/shutdown silently drops un-ended spans.

    Returns the number of spans that were ended.
    """
    spans = _context_active_spans.get()
    if not spans:
        return 0
    count = 0
    # End in reverse order (most recent/child first) to preserve parent-child ordering
    for span_id in list(reversed(list(spans.keys()))):
        span = spans.pop(span_id, None)
        if span is None:
            continue
        try:
            span.set_status(Status(StatusCode.ERROR, reason))
            span.set_attribute(
                "error.type", "Timeout" if "timeout" in reason.lower() else "Cancelled"
            )
            span.set_attribute("error.message", reason)
            span.end(end_time=time.time_ns())
            count += 1
        except Exception:
            pass  # Span may already be ended or invalid
    return count


_ERROR_MESSAGE_LIMIT = 5_000
_TRACE_MAX_DEPTH = 16  # prevent stack overflow on deeply nested objects
_TRACE_MAX_LENGTH = 20  # keep trace attr serialization bounded on large containers
_TRACE_MAX_STRING = 2_000  # avoid materializing multi-MB strings in trace attrs


def _error_message(exception: BaseException) -> str:
    """Return str(exception) capped via the shared truncating IO path.

    Exception messages are unbounded — a custom exception can embed repr()
    of a large object.  Span attributes with MB-sized values blow up OTLP
    payloads and trace storage.
    """
    stream = TruncatingStringIO(limit=_ERROR_MESSAGE_LIMIT)
    stream.write(str(exception))
    return stream.getvalue()


def _exception_stacktrace(exception: BaseException) -> str:
    """Return the formatted traceback for ``exception`` capped like error.message."""
    stream = TruncatingStringIO(limit=_ERROR_MESSAGE_LIMIT)
    traceback.print_exception(type(exception), exception, exception.__traceback__, file=stream)
    return stream.getvalue()


def _record_error(span: Span, exception: BaseException) -> None:
    """Attach common error attributes, including the traceback, to ``span``."""
    message = _error_message(exception)
    exception_type = type(exception).__name__
    span.set_status(Status(StatusCode.ERROR, str(exception)))
    span.set_attribute("error.type", exception_type)
    span.set_attribute("error.message", message)
    span.set_attribute("exception.type", exception_type)
    span.set_attribute("exception.message", message)
    span.set_attribute("exception.stacktrace", _exception_stacktrace(exception))


# OpenInference span kinds.
#
# Values are sourced from ``OpenInferenceSpanKindValues`` (the
# openinference-semantic-conventions package) rather than hardcoded literals so
# the strings we emit track upstream automatically. ``SpanAttributes`` /
# ``OpenInferenceMimeTypeValues`` constants are likewise used at every
# ``set_attribute`` site for the spec-defined attribute names.
class SpanKind:
    """OpenInference span kinds (values from the semantic-conventions enum)."""

    AGENT = OpenInferenceSpanKindValues.AGENT.value
    LLM = OpenInferenceSpanKindValues.LLM.value
    CHAIN = OpenInferenceSpanKindValues.CHAIN.value
    TOOL = OpenInferenceSpanKindValues.TOOL.value
    RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value
    EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
    RERANKER = OpenInferenceSpanKindValues.RERANKER.value
    GUARDRAIL = OpenInferenceSpanKindValues.GUARDRAIL.value
    EVALUATOR = OpenInferenceSpanKindValues.EVALUATOR.value


# Viewer plugin hint attribute — tells the trace viewer which rendering plugin to use.
# The attribute name is an arbitrary convention shared by exporters and the viewer.
VIEWER_PLUGIN_ATTR = "nooa.viewer.plugin"


class ViewerPlugin:
    """Valid values for the nooa.viewer.plugin span attribute.

    Each value corresponds to a rendering plugin in the trace viewer.
    When set on a span, the viewer uses this value (instead of the span name)
    to select the plugin that renders the span.
    """

    METHOD = "method"
    GENERATION = "generation"
    CODE_EXECUTION = "code_execution"
    TOOL_EXECUTION = "tool_execution"


class OpenInferenceHooks:
    """OpenTelemetry-based instrumentation hooks with OpenInference conventions.

    Implements the nooa InstrumentationHooks protocol.
    """

    def __init__(self, tracer: trace.Tracer, service_name: str = "nooa"):
        """Initialize OTel hooks.

        Args:
            tracer: OTel tracer to use for creating spans
            service_name: Service name for span metadata
        """
        self.tracer = tracer
        self.service_name = service_name

        # State for context_snapshot diffing (keyed by agent call_id)
        self._prev_system_messages: dict[str, str] = {}
        self._turn_counters: dict[str, int] = {}

        # Note: Active spans are tracked per async context via _get_active_spans()
        # to prevent context leakage during concurrent execution

    def before_agent_call(
        self,
        agent: Any,
        method_name: str,
        args: tuple,
        kwargs: dict,
        call_id: str,
        parent_call_id: str | None,
        **extra_kwargs: Any,
    ) -> Any:
        """Create AGENT span for ellipsis method execution."""
        agent_name = type(agent).__name__

        # DEFENSIVE: Initialize spans dict unconditionally to ensure early context establishment
        # This pattern prevents potential timing issues with ContextVar inheritance in async contexts
        spans_dict = _get_active_spans()
        parent_span = spans_dict.get(parent_call_id) if parent_call_id else None

        # If this method was invoked directly from an agent's executing code (e.g.
        # self.submit() inside execute_python), nest it under the active code_execution
        # span instead of the enclosing agent-method span — so it appears where it ran.
        # Only when the call is a DIRECT child of the executing agent (parent_call_id ==
        # the code's owning call_id); transitive calls (method -> method) are unchanged.
        active_exec = _current_code_execution.get()
        if (
            active_exec is not None
            and active_exec[1] is not None
            and active_exec[1] == parent_call_id
        ):
            parent_span = active_exec[0]

        context = trace.set_span_in_context(parent_span) if parent_span else None

        # Create span
        span = self.tracer.start_span(
            name=f"method.{method_name}",
            context=context,
            start_time=time.time_ns(),
        )

        # Set OpenInference attributes
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SpanKind.AGENT)
        span.set_attribute(VIEWER_PLUGIN_ATTR, ViewerPlugin.METHOD)
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.method", method_name)
        span.set_attribute("agent.call_id", call_id)
        if parent_call_id:
            span.set_attribute("agent.parent_call_id", parent_call_id)

        # Extract method signature, docstring, and file path
        try:
            method = getattr(agent, method_name, None)
            if method:
                # Get signature
                sig = inspect.signature(method)
                span.set_attribute("agent.method_signature", f"{method_name}{sig}")

                # Get first line of docstring
                doc = inspect.getdoc(method)
                if doc:
                    first_line = doc.split("\n")[0].strip()
                    span.set_attribute("agent.docstring", first_line)

                # Get agent file path (relative to cwd)
                try:
                    agent_file = inspect.getfile(agent.__class__)
                    cwd = os.getcwd()
                    rel_path = os.path.relpath(agent_file, cwd)
                    span.set_attribute("agent.file_path", rel_path)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

        # Emit the agent-method input as the OpenInference-standard ``input.value``
        # (JSON of args+kwargs). This is the single canonical representation — the
        # trace viewer/explorer read ``input.value`` (with a native fallback for
        # older traces that used ``agent.args``/``agent.kwargs``).
        try:
            span.set_attribute(
                SpanAttributes.INPUT_VALUE,
                self._safe_json_value({"args": args, "kwargs": kwargs}),
            )
            span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
            )
        except Exception:
            pass

        # Automatically add any extra kwargs as span attributes
        for key, value in extra_kwargs.items():
            if value is not None:
                with contextlib.suppress(Exception):
                    span.set_attribute(
                        f"agent.{key}",
                        str(value) if not isinstance(value, (str, int, float, bool)) else value,
                    )

        # Track span
        _get_active_spans()[call_id] = span

        return {
            "span": span,
            "call_id": call_id,
            **extra_kwargs,
            "start_time": time.time(),
        }

    def after_agent_call(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        **kwargs: Any,
    ) -> None:
        """Complete AGENT span."""
        if not context:
            return

        span: Span = context.get("span")
        call_id = context.get("call_id")

        if not span:
            return

        # Set result/error
        if exception:
            _record_error(span, exception)
        else:
            span.set_status(Status(StatusCode.OK))
            with contextlib.suppress(Exception):
                # OpenInference-standard output is the single canonical representation
                # (the legacy ``agent.result`` attr is no longer emitted).
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, self._safe_serialize(result))
                span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
                )

        # End span
        span.end(end_time=time.time_ns())

        # Clean up context_snapshot state for this agent call
        self._prev_system_messages.pop(call_id, None)
        self._turn_counters.pop(call_id, None)

        # Remove from tracking
        if call_id and call_id in _get_active_spans():
            del _get_active_spans()[call_id]

    def before_generation(
        self,
        agent: Any,
        method_name: str,
        strategy: str,
        generation_id: str,
        parent_generation_id: str | None,
        agent_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create the CHAIN span for a generation session.

        This is an orchestration step (one strategy "turn"), NOT the provider
        call itself — it carries no ``llm.*`` attributes. The real ``LLM`` span
        is the nested ``litellm.acompletion`` span emitted by
        ``openinference-instrumentation-litellm``. Marking this ``CHAIN`` (rather
        than ``LLM``) keeps OpenInference backends from expecting
        ``llm.model_name`` / ``llm.input_messages`` here.
        """
        from opentelemetry import context

        agent_name = type(agent).__name__

        # Get parent context: prefer the AGENT span for this specific call so that
        # sub-agent generation spans are correctly parented to the sub-agent's AGENT
        # span rather than to the caller's generation span.
        parent_span = _get_active_spans().get(agent_call_id) if agent_call_id else None
        if not parent_span:
            parent_span = (
                _get_active_spans().get(parent_generation_id) if parent_generation_id else None
            )

        parent_context = trace.set_span_in_context(parent_span) if parent_span else None

        # Create span
        span = self.tracer.start_span(
            name="generation",
            context=parent_context,
            start_time=time.time_ns(),
        )

        # Set OpenInference attributes. CHAIN (not LLM) — the nested
        # litellm.acompletion span is the real LLM call.
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SpanKind.CHAIN)
        span.set_attribute(VIEWER_PLUGIN_ATTR, ViewerPlugin.GENERATION)
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.method", method_name)
        span.set_attribute("generation.strategy", strategy)
        span.set_attribute("generation.id", generation_id)
        if parent_generation_id:
            span.set_attribute("generation.parent_id", parent_generation_id)
        if agent_call_id:
            span.set_attribute("agent.call_id", agent_call_id)

        # Automatically add any extra kwargs as span attributes
        for key, value in kwargs.items():
            if value is not None:
                with contextlib.suppress(Exception):
                    span.set_attribute(
                        f"generation.{key}",
                        str(value) if not isinstance(value, (str, int, float, bool)) else value,
                    )

        # Track span
        _get_active_spans()[generation_id] = span

        # CRITICAL: Attach span to context so litellm instrumentor sees it
        token = context.attach(trace.set_span_in_context(span))

        return {
            "span": span,
            "generation_id": generation_id,
            **kwargs,
            "start_time": time.time(),
            "context_token": token,  # Store token to detach later
        }

    def after_generation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        """Complete generation span."""
        from opentelemetry import context as otel_context

        if not context:
            return

        span: Span = context.get("span")
        context_token = context.get("context_token")

        if not span:
            return

        # Set result/error
        if exception:
            _record_error(span, exception)
        else:
            span.set_status(Status(StatusCode.OK))
            # Capture the generation result as the OpenInference-standard output
            # (the legacy ``generation.result`` attr is no longer emitted). The
            # ``result.type`` metadata has no OI equivalent and is retained.
            try:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, self._safe_serialize(result))
                span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
                )
                span.set_attribute("result.type", type(result).__name__ if result else "None")
            except Exception:
                pass

        # End span
        span.end(end_time=time.time_ns())

        # Detach context
        if context_token:
            otel_context.detach(context_token)

        # Remove from tracking
        if generation_id and generation_id in _get_active_spans():
            del _get_active_spans()[generation_id]

    def before_code_execution(
        self,
        agent: Any,
        code: str,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create TOOL span for code execution."""
        agent_name = type(agent).__name__

        # Find parent span - prefer generation span if generation_id provided
        parent_span = None
        if generation_id and generation_id in _get_active_spans():
            parent_span = _get_active_spans()[generation_id]
        else:
            # Fallback: most recent span
            for span in reversed(list(_get_active_spans().values())):
                parent_span = span
                break

        context = trace.set_span_in_context(parent_span) if parent_span else None

        # Create span
        span = self.tracer.start_span(
            name="code_execution",
            context=context,
            start_time=time.time_ns(),
        )

        # Set OpenInference attributes
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SpanKind.TOOL)
        span.set_attribute(VIEWER_PLUGIN_ATTR, ViewerPlugin.CODE_EXECUTION)
        span.set_attribute("tool.name", "python_executor")
        span.set_attribute("agent.name", agent_name)
        capped_code = code[:10000]  # Limit code length
        span.set_attribute("code.length", len(code))
        span.set_attribute("execution.id", execution_id)
        # OpenInference-standard input: JSON {"code": ...} is the single
        # canonical representation of the executed code (the legacy flat ``code``
        # attr is no longer emitted; ``code.length`` metadata is retained).
        code_args_json = self._safe_json_value({"code": capped_code})
        span.set_attribute(SpanAttributes.INPUT_VALUE, code_args_json)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        # Tool-call identity. No model tool-call id here, so fall back to
        # the execution id.
        self._set_tool_call_attrs(
            span,
            function_name="python_executor",
            arguments_json=code_args_json,
            tool_call_id=kwargs.get("tool_call_id") or execution_id,
        )

        # Add generation_id for correlation with LLM turns
        if generation_id:
            span.set_attribute("generation.id", generation_id)

        # Automatically add any extra kwargs as span attributes
        for key, value in kwargs.items():
            if value is not None:
                attr_name = f"tool.{key}" if not key.startswith("tool") else key
                with contextlib.suppress(Exception):
                    span.set_attribute(
                        attr_name,
                        str(value) if not isinstance(value, (str, int, float, bool)) else value,
                    )

        # Track span
        _get_active_spans()[execution_id] = span

        # Mark this as the active code execution for the current task, paired with the
        # call_id of the agent whose code is running (top of the agent call stack — pushed
        # by the method wrapper before the body runs). Methods invoked directly from this
        # code nest under this span (see before_agent_call). Reset in after_code_execution.
        exec_token = None
        try:
            from nooa.runtime.context_vars import _get_agent_call_stack

            stack = _get_agent_call_stack()
            owning_call_id = stack[-1] if stack else None
            exec_token = _current_code_execution.set((span, owning_call_id))
        except Exception:
            exec_token = None

        return {
            "span": span,
            "execution_id": execution_id,
            "generation_id": generation_id,
            "exec_token": exec_token,
            **kwargs,  # Pass through all extra context
            "start_time": time.time(),
        }

    def after_code_execution(
        self,
        agent: Any,
        code: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        """Complete code execution span."""
        if not context:
            return

        # Restore the previous active-execution marker (supports nested execute_python).
        exec_token = context.get("exec_token")
        if exec_token is not None:
            with contextlib.suppress(Exception):
                _current_code_execution.reset(exec_token)

        span: Span = context.get("span")

        if not span:
            return

        # Set result/error
        if exception:
            _record_error(span, exception)
        else:
            span.set_status(Status(StatusCode.OK))
            try:
                # JSON-encoded {stdout, stderr, returned_value} — gives the
                # ATIF exporter a parseable observation source. Falls back to
                # the generic repr serializer when the
                # result doesn't look like an ExecutionResult.
                if any(hasattr(result, attr) for attr in ("stdout", "stderr", "returned_value")):
                    result_str = self._safe_serialize_execution_result(result)
                    out_mime = OpenInferenceMimeTypeValues.JSON.value
                else:
                    result_str = self._safe_serialize(result)
                    out_mime = OpenInferenceMimeTypeValues.TEXT.value
                # OpenInference-standard output is the single canonical representation
                # (the legacy ``result`` attr is no longer emitted; ``result.type``
                # metadata has no OI equivalent and is retained).
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, result_str)
                span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, out_mime)
                span.set_attribute("result.type", type(result).__name__ if result else "None")
            except Exception:
                pass

        # End span
        span.end(end_time=time.time_ns())

        # Remove from tracking
        if execution_id and execution_id in _get_active_spans():
            del _get_active_spans()[execution_id]

    def before_method_invocation(
        self,
        agent: Any,
        method_name: str,
        args: tuple,
        kwargs: dict,
        invocation_id: str,
        **extra_kwargs: Any,
    ) -> Any:
        """Create TOOL span for generated method invocation."""
        agent_name = type(agent).__name__

        # Find parent span (most recent generation or agent span)
        parent_span = None
        for span in reversed(list(_get_active_spans().values())):
            parent_span = span
            break

        context = trace.set_span_in_context(parent_span) if parent_span else None

        # Create span
        span = self.tracer.start_span(
            name=f"method_call.{method_name}",
            context=context,
            start_time=time.time_ns(),
        )

        # Set OpenInference attributes
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SpanKind.TOOL)
        span.set_attribute(VIEWER_PLUGIN_ATTR, ViewerPlugin.METHOD)
        span.set_attribute("tool.name", f"generated_method:{method_name}")
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("method.name", method_name)
        span.set_attribute("invocation.id", invocation_id)

        # Emit method-call input as the OpenInference-standard input.value (the
        # legacy ``method.args``/``method.kwargs`` attrs are no longer emitted).
        try:
            args_json = self._safe_json_value({"args": args, "kwargs": kwargs})
            span.set_attribute(SpanAttributes.INPUT_VALUE, args_json)
            span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
            )
            # Tool-call identity; fall back to the invocation id when the model
            # provided no tool-call id.
            self._set_tool_call_attrs(
                span,
                function_name=method_name,
                arguments_json=args_json,
                tool_call_id=extra_kwargs.get("tool_call_id") or invocation_id,
            )
        except Exception:
            pass

        # Automatically add any extra kwargs as span attributes
        for key, value in extra_kwargs.items():
            if value is not None:
                with contextlib.suppress(Exception):
                    span.set_attribute(
                        f"method.{key}",
                        str(value) if not isinstance(value, (str, int, float, bool)) else value,
                    )

        # Track span
        _get_active_spans()[invocation_id] = span

        return {
            "span": span,
            "invocation_id": invocation_id,
            **extra_kwargs,
            "start_time": time.time(),
        }

    def after_method_invocation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        invocation_id: str,
        **kwargs: Any,
    ) -> None:
        """Complete method invocation span."""
        if not context:
            return

        span: Span = context.get("span")

        if not span:
            return

        # Set result/error
        if exception:
            _record_error(span, exception)
        else:
            span.set_status(Status(StatusCode.OK))
            try:
                # OpenInference-standard output is canonical (the legacy
                # ``method.result`` attr is no longer emitted).
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, self._safe_serialize(result))
                span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
                )
                span.set_attribute("result.type", type(result).__name__ if result else "None")
            except Exception:
                pass

        # End span
        span.end(end_time=time.time_ns())

        # Remove from tracking
        if invocation_id and invocation_id in _get_active_spans():
            del _get_active_spans()[invocation_id]

    def before_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create TOOL span for tool execution (e.g., return_result)."""
        agent_name = type(agent).__name__

        # Find parent span - prefer generation span if generation_id provided
        parent_span = None
        if generation_id and generation_id in _get_active_spans():
            parent_span = _get_active_spans()[generation_id]
        else:
            # Fallback: most recent span
            for span in reversed(list(_get_active_spans().values())):
                parent_span = span
                break

        context = trace.set_span_in_context(parent_span) if parent_span else None

        # Create span with tool-specific name
        span = self.tracer.start_span(
            name=f"tool_execution.{tool_name}",
            context=context,
            start_time=time.time_ns(),
        )

        # Set OpenInference attributes
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SpanKind.TOOL)
        span.set_attribute(VIEWER_PLUGIN_ATTR, ViewerPlugin.TOOL_EXECUTION)
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("execution.id", execution_id)

        # Emit tool arguments as the OpenInference-standard input.value (valid JSON);
        # the legacy ``tool.arguments`` attr is no longer emitted.
        with contextlib.suppress(Exception):
            arguments_json = self._safe_json_value(arguments)
            span.set_attribute(SpanAttributes.INPUT_VALUE, arguments_json)
            span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
            )
            # Tool-call identity. Prefer the model-provided
            # tool_call_id; fall back to execution_id (#12).
            self._set_tool_call_attrs(
                span,
                function_name=tool_name,
                arguments_json=arguments_json,
                tool_call_id=kwargs.get("tool_call_id") or execution_id,
            )

        # Add generation_id for correlation with LLM turns
        if generation_id:
            span.set_attribute("generation.id", generation_id)

        # Automatically add any extra kwargs as span attributes
        for key, value in kwargs.items():
            if value is not None:
                attr_name = f"tool.{key}" if not key.startswith("tool") else key
                with contextlib.suppress(Exception):
                    span.set_attribute(
                        attr_name,
                        str(value) if not isinstance(value, (str, int, float, bool)) else value,
                    )

        # Track span
        _get_active_spans()[execution_id] = span

        return {
            "span": span,
            "execution_id": execution_id,
            "generation_id": generation_id,
            **kwargs,  # Pass through all extra context
            "start_time": time.time(),
        }

    def after_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict,
        result: Any,
        exception: Exception | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        """Complete tool execution span."""
        if not context:
            return

        span: Span = context.get("span")

        if not span:
            return

        # Set result/error
        if exception:
            _record_error(span, exception)
        else:
            span.set_status(Status(StatusCode.OK))
            try:
                # OpenInference-standard output is canonical (the legacy
                # ``tool.result`` attr is no longer emitted).
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, self._safe_serialize(result))
                span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
                )
                span.set_attribute("result.type", type(result).__name__ if result else "None")
            except Exception:
                pass

        # End span
        span.end(end_time=time.time_ns())

        # Remove from tracking
        if execution_id and execution_id in _get_active_spans():
            del _get_active_spans()[execution_id]

    def on_messages_built(
        self,
        agent: Any,
        method_name: str,
        messages: list[dict[str, Any]],
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        """Create a context_snapshot span with the system message (full or diff)."""
        # Extract system message
        if not messages or messages[0].get("role") != "system":
            return
        system_content = messages[0].get("content", "")
        if not system_content:
            return

        # Find current agent call_id from active spans
        call_id = self._find_current_call_id()
        if not call_id:
            return

        turn_index = self._turn_counters.get(call_id, 0)
        self._turn_counters[call_id] = turn_index + 1
        prev = self._prev_system_messages.get(call_id)

        if prev is None:
            # First generation in this agent call — full snapshot
            content = system_content
            is_diff = False
        elif prev == system_content:
            # No change — skip span entirely
            self._prev_system_messages[call_id] = system_content
            return
        else:
            # Changed — unified diff
            content = "\n".join(
                difflib.unified_diff(
                    prev.splitlines(),
                    system_content.splitlines(),
                    fromfile="previous",
                    tofile="current",
                    lineterm="",
                )
            )
            is_diff = True

        self._prev_system_messages[call_id] = system_content

        # Create child span under current generation span (or any active span)
        parent_span = _get_active_spans().get(generation_id)
        if not parent_span:
            for span in reversed(list(_get_active_spans().values())):
                parent_span = span
                break

        parent_context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            "context_snapshot",
            context=parent_context,
            start_time=time.time_ns(),
        )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SpanKind.CHAIN)
        span.set_attribute("nooa.system_message", content)
        span.set_attribute("nooa.system_message.is_diff", is_diff)
        span.set_attribute("nooa.system_message.turn_index", turn_index)
        # OpenInference-standard input: the system message (full or diff), text/plain.
        # Reuses ``content`` so the native attr and input.value match.
        span.set_attribute(SpanAttributes.INPUT_VALUE, content)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value)
        span.end(end_time=time.time_ns())

    def _find_current_call_id(self) -> str | None:
        """Find the current agent call_id from active spans."""
        active = _get_active_spans()
        # Check which active span keys have tracking state
        for key in reversed(list(active.keys())):
            if key in self._prev_system_messages or key in self._turn_counters:
                return key
        # Fallback: first active span key (likely the agent call)
        if active:
            return next(iter(active))
        return None

    @staticmethod
    def _safe_serialize_execution_result(result: Any, max_chars: int = 50_000) -> str:
        """Serialize an ExecutionResult to JSON, capping returned_value *before* encoding.

        Unlike post-JSON slicing (``json_str[:50000]``), this approach always
        produces valid JSON because truncation happens on the Python value, not
        on the serialised string.
        """
        data: dict[str, Any] = {}
        if hasattr(result, "stdout"):
            data["stdout"] = truncating_pformat(
                result.stdout,
                max_chars=max_chars,
                max_depth=_TRACE_MAX_DEPTH,
                max_length=_TRACE_MAX_LENGTH,
                max_string=_TRACE_MAX_STRING,
            )
        if hasattr(result, "stderr"):
            data["stderr"] = truncating_pformat(
                result.stderr,
                max_chars=max_chars,
                max_depth=_TRACE_MAX_DEPTH,
                max_length=_TRACE_MAX_LENGTH,
                max_string=_TRACE_MAX_STRING,
            )
        if hasattr(result, "returned_value"):
            rv = result.returned_value
            # Check for _NO_RETURN sentinel or None
            if rv is None or (hasattr(rv, "__class__") and rv.__class__.__name__ == "object"):
                data["returned_value"] = None
            else:
                data["returned_value"] = truncating_pformat(
                    rv,
                    max_chars=max_chars,
                    max_depth=_TRACE_MAX_DEPTH,
                    max_length=_TRACE_MAX_LENGTH,
                    max_string=_TRACE_MAX_STRING,
                )
        return json.dumps(data)

    @staticmethod
    def _safe_serialize(obj: Any, max_chars: int = 50_000) -> str:
        """Serialize an object to string for trace span attributes.

        Delegates to ``truncating_pformat`` which handles all types (primitives,
        Pydantic models, dicts, lists) and enforces a hard ``max_chars``
        cap with head+tail truncation.

        Trace span attributes are NOT visible to the agent — the agent sees
        rendered, truncated context blocks.  These attributes exist only for
        human inspection in the trace viewer, so a 50 K cap is generous.
        """
        try:
            return truncating_pformat(
                obj,
                max_chars=max_chars,
                max_depth=_TRACE_MAX_DEPTH,
                max_length=_TRACE_MAX_LENGTH,
                max_string=_TRACE_MAX_STRING,
            )
        except Exception:
            return "<unserializable>"

    @staticmethod
    def _safe_json_value(obj: Any, max_chars: int = 50_000) -> str:
        """Serialize ``obj`` to a **valid JSON** string for ``input.value`` /
        ``output.value`` attributes tagged ``application/json``.

        Non-JSON-serializable leaves fall back to their truncated pformat repr
        (via ``_safe_serialize``) so the result is always valid JSON.

        On overflow (encoded length > ``max_chars``) the **top-level dict structure
        is preserved** — each value is re-serialized with a smaller budget — so a
        ``{"args": …, "kwargs": …}`` / ``{"code": …}`` input still parses back to a
        dict with those keys (readers like ``_io_json_field`` and the viewer extract
        fields by key). Only a non-dict, or a dict that still overflows after
        per-value bounding, collapses to a JSON string literal as a last resort.
        """
        try:
            encoded = json.dumps(
                obj,
                default=lambda o: OpenInferenceHooks._safe_serialize(o, max_chars),
            )
            if len(encoded) <= max_chars:
                return encoded
            # Overflow: keep the top-level keys so field extraction still works,
            # bounding each value rather than collapsing the whole object. The
            # per-value budget leaves headroom for ``_safe_serialize``'s head+tail
            # truncation (~2x the budget) plus JSON-escaping overhead.
            if isinstance(obj, dict):
                per_value = max(256, max_chars // (4 * max(1, len(obj))))
                shaped = json.dumps(
                    {
                        str(k): OpenInferenceHooks._safe_serialize(v, per_value)
                        for k, v in obj.items()
                    }
                )
                if len(shaped) <= max_chars:
                    return shaped
        except Exception:
            pass
        # Last resort: a JSON string literal wrapping the bounded human-readable repr.
        return json.dumps(OpenInferenceHooks._safe_serialize(obj, max_chars))

    @staticmethod
    def _set_tool_call_attrs(
        span: Span,
        *,
        function_name: str,
        arguments_json: str,
        tool_call_id: str,
    ) -> None:
        """Emit the OpenInference flat tool-call identity attrs on a TOOL span.

        * ``tool.id`` + ``tool_call.id`` — the call's identity. ``tool_call_id``
          should be the model-provided id when one exists, falling back to the
          framework ``execution_id`` / ``invocation_id`` so inline framework tools
          always have a stable id.
        * ``tool_call.function.name`` + ``tool_call.function.arguments`` (valid
          JSON) — mirror how an LLM span records a tool call, so OpenInference
          backends render these framework TOOL spans as calls.

        ``arguments_json`` must already be a valid-JSON string (use
        :meth:`_safe_json_value`).
        """
        with contextlib.suppress(Exception):
            span.set_attribute(SpanAttributes.TOOL_ID, tool_call_id)
            span.set_attribute(ToolCallAttributes.TOOL_CALL_ID, tool_call_id)
            span.set_attribute(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME, function_name)
            span.set_attribute(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON, arguments_json)
