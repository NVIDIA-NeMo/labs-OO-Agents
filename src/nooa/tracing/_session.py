# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-id propagation, backed by the OpenTelemetry context.

``set_session()`` writes ``session.id`` into the current OTel context.
Two things then read from that one source of truth:

1. **OpenInference instrumentations** (``OITracer`` used by
   ``LiteLLMInstrumentor`` and others) — at span creation, they call
   ``get_attributes_from_context()`` and merge the value into the
   span's initial attribute dict. This is what makes
   litellm's ``acompletion`` spans land in the correct session.
2. **Our own hooks** — :class:`SessionSpanProcessor` reads the same
   OTel context at ``on_start`` and stamps ``session.id`` on spans
   created by the plain SDK tracer (framework hooks like
   ``method.respond``, ``generation``, ``code_execution``,
   ``context_snapshot``) which otherwise have no OpenInference layer
   to consult the context on their behalf.

OTel context is a per-task ``ContextVar`` under the hood, so each
asyncio task gets its own value. Concurrent multi-session callers
(``--parallel N`` eval runs, multiple agents in one process) keep
per-task attribution without any extra bookkeeping.
"""

from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context

_KEY = SpanAttributes.SESSION_ID


def set_session(session_id: str | None) -> None:
    """Set the session ID for the current async context.

    All spans created in this context will carry ``session.id`` as a
    span attribute. Writes to the OpenTelemetry context so OpenInference
    instrumentations pick it up automatically at span creation, and
    our :class:`SessionSpanProcessor` picks it up at ``on_start``.

    The returned ``attach()`` token is intentionally not kept — the
    value persists for the rest of this task, same semantics as a
    bare ``ContextVar.set()``. Pass ``None`` to clear.
    """
    value = "" if session_id is None else session_id
    new_ctx = otel_context.set_value(_KEY, value, otel_context.get_current())
    otel_context.attach(new_ctx)


def get_session() -> str | None:
    """Get the current session ID for this async context (``None`` if unset)."""
    value = otel_context.get_value(_KEY)
    if not value:
        return None
    return str(value)
