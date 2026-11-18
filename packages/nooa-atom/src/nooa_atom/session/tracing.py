# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace a child session's turns in its parent's trace.

Every session's turn loop runs in a fresh ``contextvars.Context`` (see
``Session.start``), so the tracing state kept in context variables (the
tracing hooks, the trace session id and the current span, both in the
OTel context) is not there. Roots are reconnected by their host
(``acp.server._trace_as``, ``hosts.headless._trace_turns``); a child is
reconnected here, for every host: the parent's OTel context is captured
when the child is created (inside the parent's ``delegate()`` or
``spawn()`` call, so its current span is that call's span; when a child is
reopened from a cell, that cell's ``code_execution`` span) and attached in
the child's loop context, with the tracing hooks. A persisted child
loaded directly by a host (``SessionRegistry.load`` with no ``prepare``)
while its parent is live has no calling cell: it is traced in the tree's
trace session instead, as its own top-level trace (``trace_in_tree``).
Only OTel state is
copied; the child's context stays fresh otherwise (no call stack,
generation state or scoped blocks of the parent).

The child's agent is also named after the session in the trace
(``nooa.tracing.set_agent_trace_name``): its spans carry the session name
as ``agent.name`` (``Multiply.handle_work``), so in the viewer a child
looks like a separate agent, as a sub-agent called from a cell does. The
name belongs to that agent instance only; other agents keep their class
names.

Tracing is optional: without ``nooa.tracing`` (or OpenTelemetry) this
does nothing.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nooa_atom.session.session import Session

# Span attributes that say which child session a span belongs to.
SESSION_ID_ATTR = "nooa.atom.session.id"
SESSION_NAME_ATTR = "nooa.atom.session.name"
PARENT_ID_ATTR = "nooa.atom.session.parent_id"
DEPTH_ATTR = "nooa.atom.session.depth"


def _attributes(session: "Session") -> dict[str, str | int]:
    attributes: dict[str, str | int] = {
        SESSION_ID_ATTR: session.id,
        SESSION_NAME_ATTR: session.name or "",
        DEPTH_ATTR: session.depth,
    }
    if session.parent_id is not None:
        attributes[PARENT_ID_ATTR] = session.parent_id
    return attributes


def trace_under_caller(session: "Session") -> None:
    """Make ``session``'s turns continue the caller's trace, under the caller's span.

    Call it in the caller's context (the parent's cell) before the session
    starts: the OTel context is captured now and attached by a loop context
    hook. Spans started in the session's turns carry the caller's trace
    session (``session.id``, the root's) and the ``nooa.atom.session.*``
    attributes of this session, and its agent's spans are named after the
    session. A kept child's later turns run in the same loop context, so
    they nest under the span current at creation too (the ``spawn()`` call).
    """
    try:
        from nooa.tracing import (
            attach_trace_context,
            capture_trace_context,
            set_agent_trace_name,
        )
    except ImportError:
        return
    captured = capture_trace_context()
    if session.name:
        set_agent_trace_name(session._agent, session.name)
    attributes = _attributes(session)

    def enter() -> None:
        attach_trace_context(captured, attributes)

    session.add_loop_context_hook(enter)


def trace_in_tree(session: "Session", trace_session_id: str) -> None:
    """Trace ``session``'s turns in trace session ``trace_session_id``, under no span.

    For a child opened with no calling cell (a host's direct
    ``SessionRegistry.load``): its spans carry ``session.id`` =
    ``trace_session_id`` (the root's id), the ``nooa.atom.session.*``
    attributes and the session name as agent name, and form their own
    top-level trace. The context is built from a fresh OTel context, so
    nothing of the loading caller (its span or session) is captured.
    """
    try:
        from openinference.semconv.trace import SpanAttributes
        from opentelemetry import context as otel_context

        from nooa.tracing import attach_trace_context, set_agent_trace_name
    except ImportError:
        return
    captured = otel_context.set_value(
        SpanAttributes.SESSION_ID, trace_session_id, otel_context.Context()
    )
    if session.name:
        set_agent_trace_name(session._agent, session.name)
    attributes = _attributes(session)

    def enter() -> None:
        attach_trace_context(captured, attributes)

    session.add_loop_context_hook(enter)
