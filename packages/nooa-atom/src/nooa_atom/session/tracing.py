# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace a child session's turns in its parent's trace.

Every session's turn loop runs in a fresh ``contextvars.Context`` (see
``Session.start``), so the tracing state kept in context variables (the
tracing hooks, the trace session id and the current span, both in the
OTel context) is not there. Roots are reconnected by their host
(``acp.server._trace_as``, ``hosts.headless._trace_turns``); a child is
reconnected here, for every host: the parent's OTel context is captured
when the child is created (in the parent's cell, so its current span is
that cell's ``code_execution`` span) and attached in the child's loop
context, with the tracing hooks. Only OTel state is copied; the child's
context stays fresh otherwise (no call stack, generation state or scoped
blocks of the parent).

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


def trace_under_caller(session: "Session") -> None:
    """Make ``session``'s turns continue the caller's trace, under the caller's span.

    Call it in the caller's context (the parent's cell) before the session
    starts: the OTel context is captured now and attached by a loop context
    hook. Spans started in the session's turns carry the caller's trace
    session (``session.id``, the root's) and the ``nooa.atom.session.*``
    attributes of this session. A kept child's later turns run in the same
    loop context, so they nest under the span current at creation too.
    """
    try:
        from nooa.tracing import attach_trace_context, capture_trace_context
    except ImportError:
        return
    captured = capture_trace_context()
    attributes: dict[str, str | int] = {
        SESSION_ID_ATTR: session.id,
        SESSION_NAME_ATTR: session.name or "",
        DEPTH_ATTR: session.depth,
    }
    if session.parent_id is not None:
        attributes[PARENT_ID_ATTR] = session.parent_id

    def enter() -> None:
        attach_trace_context(captured, attributes)

    session.add_loop_context_hook(enter)
