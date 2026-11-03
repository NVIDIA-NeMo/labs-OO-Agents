# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record an LLM call in the message journal without going through LiteLLM.

The journal sinks (:class:`~nooa.tracing._litellm_journal.MessageJournalCallback`
for the live viewer, :class:`~nooa.tracing._litellm_journal.FileMessageJournalCallback`
for ``exporters.journal_file``) are LiteLLM callbacks, so a client that talks
to its model server directly (for example a ``UnifiedLLM`` subclass that posts
over aiohttp) never produces a journal record.  :func:`journal_llm_call` lets
such a client hand over one finished call; it is normalized exactly like the
LiteLLM path and written to every installed sink.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)

# Roles that the runtime never sends to a model (mirrors ``build_journal_payload``).
_SKIPPED_ROLES = ("metadata", "runtime_event")

# Name of the LLM span created for a journaled call that has no span of its own.
_LLM_SPAN_NAME = "llm_call"


def journal_llm_call(
    *,
    model: str,
    messages: Sequence[Any],
    response: Any = None,
    session_id: str | None = None,
    call_id: str | None = None,
    started_at: Any = None,
    ended_at: Any = None,
    latency_ms: float | None = None,
    span_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    use_rendered_context: bool = True,
) -> None:
    """Record one finished LLM call in every installed message-journal sink.

    For clients that bypass LiteLLM (whose callbacks feed the journal).
    Call it once per call, after the response arrived (or failed: pass
    ``response=None`` to record the prompt with no output).

    Args:
        model: Model name recorded on the call.
        messages: The message list exactly as the client received it: dicts,
            ``LLMResponse`` (replayed assistant turns), ``CacheBoundary`` and
            other NOOA events are all accepted; metadata entries are skipped.
        response: The ``LLMResponse`` (or a provider response with
            ``.choices`` / ``.output``); ``None`` for a failed call.
        session_id: Journal session; defaults to the current NOOA trace
            session (:func:`nooa.tracing.get_session`).
        call_id: Unique id of the call record; generated when omitted.
        started_at, ended_at: Unix seconds or datetimes. ``ended_at``
            defaults to now, ``started_at`` to ``ended_at - latency_ms``.
        latency_ms: Call duration, used when ``started_at`` is omitted.
        span_id: 16-hex span id the call belongs to.  By default the call
            gets an ``LLM`` span of its own under the current OTel span (the
            ``generation`` span inside a NOOA turn), which is what the viewer
            shows the messages on; a caller already inside an ``LLM`` span is
            linked to that span.
        extra: JSON-able values stored on the call record as ``extra``
            (e.g. ``{"response_id": ...}`` to join with external rollouts).
        use_rendered_context: Consume the skeleton the NOOA runtime published
            for this turn (block-level dedup, identical to the LiteLLM path)
            instead of normalizing ``messages``; falls back to ``messages``
            when none is pending.

    Best effort: never raises (failures are logged at debug level) and does
    nothing when no journal exporter is installed.
    """
    try:
        sinks = _installed_sinks()
        if not sinks:
            return
        _record(
            sinks,
            model=model,
            messages=messages,
            response=response,
            session_id=session_id,
            call_id=call_id,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            span_id=span_id,
            extra=extra,
            use_rendered_context=use_rendered_context,
        )
    except Exception as exc:
        log.debug("journal_llm_call failed: %s", exc, exc_info=True)


def _installed_sinks() -> list[Any]:
    """The journal callbacks the exporters registered, each with a live sink."""
    try:
        import litellm
    except ImportError:
        return []
    from nooa.tracing._litellm_journal import (
        _INSTALL_LOCK,
        FileMessageJournalCallback,
        MessageJournalCallback,
    )

    with _INSTALL_LOCK:
        callbacks = list(litellm.callbacks)
    sinks = []
    for cb in callbacks:
        if isinstance(cb, FileMessageJournalCallback):
            if cb.has_writers():
                sinks.append(cb)
        elif isinstance(cb, MessageJournalCallback) and cb.has_destinations():
            sinks.append(cb)
    return sinks


def _record(
    sinks: list[Any],
    *,
    model: str,
    messages: Sequence[Any],
    response: Any,
    session_id: str | None,
    call_id: str | None,
    started_at: Any,
    ended_at: Any,
    latency_ms: float | None,
    span_id: str | None,
    extra: Mapping[str, Any] | None,
    use_rendered_context: bool,
) -> None:
    from nooa.tracing._context_sideband import get_journal_payload, set_journal_payload
    from nooa.tracing._litellm_journal import _to_ts
    from nooa.tracing._secret_scrubber import scrub_value
    from nooa.tracing._session import get_session

    session = session_id or get_session() or "unknown"

    input_blocks: dict[str, str] = {}
    payload = get_journal_payload() if use_rendered_context else None
    if payload is not None:
        set_journal_payload(None)  # consume, as log_pre_api_call does
        input_skeleton = payload.skeleton
        input_blocks.update(payload.blocks)
    else:
        input_skeleton = []
        for msg in messages or ():
            entry = _input_message(msg)
            if entry is not None:
                input_skeleton.append(_skeleton(entry, input_blocks))

    output_blocks: dict[str, str] = {}
    output_messages = [_skeleton(m, output_blocks) for m in _output_messages(response)]

    end = _to_ts(ended_at) if ended_at is not None else time.time()
    if started_at is not None:
        start = _to_ts(started_at)
    elif latency_ms is not None:
        start = end - float(latency_ms) / 1000.0
    else:
        start = end

    if not model and response is not None:
        model = str(getattr(response, "model_name", "") or "")
    record: dict[str, Any] = {
        "call_id": call_id or f"journal-{uuid.uuid4().hex}",
        "session_id": session,
        "model": model or "",
        "ts_start": start,
        "ts_end": end,
        "input_skeleton": input_skeleton,
        "output_messages": output_messages,
        "tokens": _tokens(response),
    }
    span = span_id or _llm_span_id(
        model=record["model"],
        start=start,
        end=end,
        tokens=record["tokens"],
        failed=response is None,
        session=session if session != "unknown" else None,
    )
    if span:
        record["span_id"] = span
    if extra:
        scrubbed, _ = scrub_value(json.loads(json.dumps(dict(extra), default=str)))
        record["extra"] = scrubbed

    blocks = {**input_blocks, **output_blocks}
    for sink in sinks:
        try:
            sink._send_new_blocks(session, blocks)
            sink._send_call(session, record)
        except Exception as exc:
            log.debug("journal_llm_call: sink %r failed: %s", sink, exc, exc_info=True)


def _llm_span_id(
    *,
    model: str,
    start: float,
    end: float,
    tokens: Mapping[str, int] | None,
    failed: bool,
    session: str | None = None,
) -> str | None:
    """The span this call is linked to, as 16 hex chars; ``None`` without a tracer.

    The viewer puts journal messages back only onto spans whose
    ``openinference.span.kind`` is ``LLM`` (``reconstruct_full_spans``) and maps
    one call per span.  NOOA's ``generation`` span is a ``CHAIN`` span, so a
    call linked to it is stored but never shown, and two calls in one
    generation would collide.  So, like the LiteLLM instrumentor, each call
    gets an ``LLM`` span of its own: a child of the current span covering the
    call's duration.  A caller already inside an ``LLM`` span (its own
    instrumentation) is linked to that span instead.

    ``session`` is the session the call record is stored under: the span is
    started in that session's scope, so its ``session.id`` (stamped at span
    start) matches the record even when the caller passed an explicit
    ``session_id`` that differs from the current trace session.
    """
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode

    current = otel_trace.get_current_span()
    ctx = current.get_span_context()
    kind = (getattr(current, "attributes", None) or {}).get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
    if ctx.is_valid and kind == OpenInferenceSpanKindValues.LLM.value:
        return format(ctx.span_id, "016x")

    attributes: dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_MODEL_NAME: model,
    }
    if tokens:
        attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = tokens["prompt"]
        attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = tokens["completion"]
        if "cached" in tokens:
            attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = tokens["cached"]
    from nooa.tracing._session import session_scope

    scope = session_scope(session) if session else contextlib.nullcontext()
    with scope:
        span = otel_trace.get_tracer(__name__).start_span(
            _LLM_SPAN_NAME, start_time=int(start * 1e9), attributes=attributes
        )
    if failed:
        span.set_status(Status(StatusCode.ERROR, "no response"))
    span.end(end_time=int(max(end, start) * 1e9))
    span_ctx = span.get_span_context()
    return format(span_ctx.span_id, "016x") if span_ctx.is_valid else None


def _role_of(msg: Any) -> str | None:
    role = msg.get("role") if isinstance(msg, Mapping) else getattr(msg, "event_role", None)
    if role is None:
        role = getattr(msg, "role", None)
    if role is None:
        return None
    return str(getattr(role, "value", role))


def _input_message(msg: Any) -> dict | None:
    """One client message as a scrubbed OpenAI-shape dict; ``None`` to skip."""
    from nooa.llm_types import CacheBoundary
    from nooa.tracing._litellm_journal import _safe_msg_to_dict

    if msg is None or isinstance(msg, CacheBoundary):
        return None
    role = _role_of(msg)
    if role in _SKIPPED_ROLES:
        return None
    if isinstance(msg, Mapping):
        # LLMResponse is a Mapping over its public assistant message.
        raw: Any = dict(msg)
    elif hasattr(msg, "public_message"):
        raw = msg.public_message()
    elif hasattr(msg, "event_role") and hasattr(msg, "model_dump"):
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            content = json.dumps(msg.model_dump(mode="json", exclude={"event_type"}), default=str)
        raw = {"role": role or "user", "content": content}
    else:
        raw = msg
    return _safe_msg_to_dict(raw)


def _output_messages(response: Any) -> list[dict]:
    from nooa.tracing._litellm_journal import _extract_output_msgs, _safe_msg_to_dict

    if response is None:
        return []
    if isinstance(response, Mapping):
        return [_safe_msg_to_dict(dict(response))]
    return [_safe_msg_to_dict(m) for m in _extract_output_msgs(response)]


def _skeleton(msg: dict, blocks: dict[str, str]) -> dict:
    """Content-address a message; structured (list) content stays inline."""
    from nooa.tracing._litellm_journal import _skeleton_dict_message

    content = msg.get("content")
    if content is None or isinstance(content, str):
        return _skeleton_dict_message(msg, blocks)
    entry = _skeleton_dict_message({k: v for k, v in msg.items() if k != "content"}, blocks)
    entry["content"] = content
    return entry


def _tokens(response: Any) -> dict[str, int] | None:
    from nooa.llm_types import LLMUsage

    usage = LLMUsage.from_provider(getattr(response, "usage", None))
    if usage is None:
        return None
    tokens = {"prompt": usage.input_tokens, "completion": usage.output_tokens}
    if usage.cached_input_tokens:
        tokens["cached"] = usage.cached_input_tokens
    return tokens
