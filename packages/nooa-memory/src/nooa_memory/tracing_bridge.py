# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge memory events into tracing spans.

Memory monitoring events are ``RUNTIME_EVENT`` + never recorded — invisible in
traces. This bridge subscribes to them and re-emits each as a **span event on
the currently active span** (``memory.written`` / ``memory.recalled`` /
``memory.injected`` / ``memory.reflected``), so memory activity shows up inside
the very ``execute_python``/method spans where it happened. Every event carries
``memory.db_path`` + ``memory.owner`` — that is how the web viewer discovers
which memory DB belongs to a trace session.

Conversely, :func:`current_trace_ref` hands the active span id back to the
access log (``AccessRecord.trace_ref``) — the two stores cross-link.

Everything degrades to a no-op when ``opentelemetry`` is not installed or no
span is recording.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from opentelemetry import trace as _ot_trace
except ImportError:  # pragma: no cover - exercised only without the tracing extra
    _ot_trace = None

if TYPE_CHECKING:
    from nooa_memory.manager import MemoryManager

# monitoring event class name -> span event name
_EVENT_NAMES = {
    "MemoryWritten": "memory.written",
    "MemoryRecalled": "memory.recalled",
    "MemoryInjected": "memory.injected",
    "ReflectionStarted": "memory.reflection_started",
    "ReflectionCompleted": "memory.reflected",
}


def current_trace_ref() -> str | None:
    """The active span id as a hex string, or None when not tracing."""
    if _ot_trace is None:
        return None
    ctx = _ot_trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.span_id, "016x")


def _attr_safe(value: Any) -> Any | None:
    """Coerce an event field into an OTel-legal attribute value (or drop it)."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def install_tracing_bridge(manager: MemoryManager) -> list[Callable[[], None]]:
    """Subscribe the span-event handlers; returns unsubscribe closures."""
    if _ot_trace is None:
        return []
    db_path = manager.store.path
    base_attrs: dict[str, Any] = {
        "memory.db_path": str(Path(db_path).resolve()) if db_path != ":memory:" else db_path,
        "memory.owner": manager.owner,
    }

    def _handler_for(span_event_name: str) -> Callable[[object], None]:
        def handler(event: object) -> None:
            span = _ot_trace.get_current_span()
            if not span.get_span_context().is_valid or not span.is_recording():
                return
            attrs = dict(base_attrs)
            for field, value in getattr(event, "__dict__", {}).items():
                if field.startswith("_"):
                    continue
                safe = _attr_safe(value)
                if safe is not None:
                    attrs[f"memory.{field}"] = safe
            span.add_event(span_event_name, attributes=attrs)

        return handler

    em = manager.agent.event_manager
    return [em.on(cls_name, _handler_for(name)) for cls_name, name in _EVENT_NAMES.items()]
