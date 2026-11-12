# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK-free logical tracing; concrete journal rendering is an optional binding.

Initial binding is None: a custom adapter or tracing-disabled caller still gets
one logical span (a no-op without an OTel provider) and neutral settlement. The
tracing journal module binds its own renderer on import, idempotently on repeated
binding. Reload replaces the single renderer for future invocations; active
invocations retain their entry-time callable. No callback registration in SDKs,
per-call registry, transient IDs, sinks or concrete imports live here.
"""

import time

_journal_renderer = None


def _bind_journal_renderer(renderer):
    global _journal_renderer
    if not callable(renderer):
        raise TypeError("A journal renderer must be callable")
    if _journal_renderer is not renderer:
        _journal_renderer = renderer


class _LogicalJournal:
    """SDK-free logical span, independent of renderer/sink installation.

    Capture one terminal renderer at entry. Enabling/reloading tracing affects
    future calls only; a terminal worker never consults another call's binding.
    """

    def __init__(self, logical_id, model, messages):
        from opentelemetry import trace

        self._writer = _journal_renderer
        self.logical_id = logical_id
        self.model = model
        self.messages = messages
        self.effective_messages = None
        self.start = time.time()
        self.span = trace.get_tracer(__name__).start_span(
            "llm_call",
            attributes={
                "openinference.span.kind": "LLM",
                "llm.model_name": model,
                "nooa.llm.owner": "logical",
                "nooa.llm.logical_id": logical_id,
            },
        )
        try:
            self.scope = trace.use_span(
                self.span, end_on_exit=False, record_exception=False, set_status_on_exception=False
            )
            self.scope.__enter__()
        except BaseException:
            self.span.end()
            raise

    def describe_request(self, *, messages=None, model=None):
        if messages is not None and messages != self.messages:
            self.effective_messages = messages
        if isinstance(model, str) and model:
            self.model = model
            self.span.set_attribute("llm.model_name", model)

    def finish(self, snapshot, response):
        from opentelemetry.trace import Status, StatusCode

        end = time.time()
        try:
            totals = dict(snapshot.totals)
            names = {
                "input_tokens": "prompt",
                "output_tokens": "completion",
                "cached_input_tokens": "cached",
            }
            tokens = {
                target: totals[source].known
                for source, target in names.items()
                if totals[source].known is not None
            }
            # Keep the journal's original prompt/completion pair-or-null schema;
            # field-specific partial subtotals remain available in additive extra.
            journal_tokens = tokens if "prompt" in tokens and "completion" in tokens else None
            extra = {
                "nooa_llm_owner": "logical",
                "logical_id": snapshot.logical_id,
                "outcome": snapshot.outcome,
                "failure_kind": snapshot.failure_kind,
                "attempt_count": snapshot.attempt_count,
                "dropped_attempts": snapshot.dropped_attempts,
                "accounting": {
                    name: {"known": total.known, "complete": total.complete}
                    for name, total in snapshot.totals
                },
                "cost_sources": list(snapshot.cost_sources),
                "attempts": [
                    {
                        "index": attempt.attempt_index,
                        "provider_entered": attempt.provider_entered,
                        "admission_known": attempt.admission_known,
                        "response_received": attempt.response_received,
                        "hidden_retries": attempt.hidden_retries,
                        **(
                            {"failure_kind": attempt.failure_kind}
                            if attempt.failure_kind is not None
                            else {}
                        ),
                        **(
                            {"status_code": attempt.status_code}
                            if attempt.status_code is not None
                            else {}
                        ),
                        **(
                            {"retry_after": attempt.retry_after}
                            if attempt.retry_after is not None
                            else {}
                        ),
                    }
                    for attempt in snapshot.attempts
                ],
            }
            for name, total in snapshot.totals:
                if total.known is not None:
                    self.span.set_attribute("nooa.llm.paid." + name, total.known)
                self.span.set_attribute("nooa.llm.complete." + name, total.complete)
            for source, target in names.items():
                if totals[source].known is not None:
                    self.span.set_attribute("llm.token_count." + target, totals[source].known)
            for attempt in extra["attempts"]:
                self.span.add_event("llm.attempt", attributes=attempt)
            if snapshot.outcome != "success":
                self.span.set_status(
                    Status(StatusCode.ERROR, snapshot.failure_kind or snapshot.outcome)
                )
            if self._writer is not None:
                self._writer(self, snapshot, response, extra, journal_tokens, end)
        finally:
            try:
                self.scope.__exit__(None, None, None)
            finally:
                self.span.end(end_time=int(end * 1e9))
