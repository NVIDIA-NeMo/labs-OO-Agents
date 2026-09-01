# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JournalExporter: spans (sans messages) + content-addressed message sideband.

Combines two responsibilities into a single exporter:

1. Exports spans to the local viewer with ``llm.input_messages.*`` /
   ``llm.output_messages.*`` attributes **stripped** — the viewer reconstructs
   full messages from the journal instead.
2. Installs a litellm ``CustomLogger`` callback that intercepts message lists
   *before* each LLM call and posts only new (delta) messages to the viewer's
   journal endpoints, reducing per-call storage from O(N) to O(delta).

Multiple ``JournalExporter`` instances share a single
``MessageJournalCallback`` -- each adds its base URL to the callback's
destination list rather than installing a separate ``litellm.callbacks``
entry.  litellm only delivers ``log_success_event`` to one callback per
class, so a per-exporter callback would silently drop the call record on
every destination after the first.

Usage::

    enable_tracing(exporters=[exporters.journal()])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from nooa.tracing._litellm_journal import MessageJournalCallback


class JournalExporter(SpanExporter):
    """SpanExporter: stripped OTLP spans + content-addressed message journal.

    Delegates span export to an internal :class:`OtlpJsonHttpExporter` with
    ``strip_llm_messages=True`` so message attributes are omitted from the
    wire payload.  The litellm callback handles message delivery separately.
    """

    def __init__(self, base_url: str) -> None:
        from nooa.tracing._otlp_http_exporter import (
            OtlpJsonHttpExporter,
        )

        self._base_url = base_url.rstrip("/")
        self._span_exporter = OtlpJsonHttpExporter(
            endpoint=f"{self._base_url}/v1/traces",
            strip_llm_messages=True,
        )
        self._callback = self._install()

    def _install(self) -> MessageJournalCallback:
        """Register this exporter's base URL on the shared journal callback.

        If a :class:`MessageJournalCallback` is already in
        ``litellm.callbacks`` we add our base URL to its destination list
        rather than installing a second instance.  Returns the callback
        so :meth:`shutdown` can deregister.

        Holds a process-wide lock across the scan-and-append so two
        threads constructing exporters for the same URL converge on a
        single shared callback (otherwise both would scan, see no
        existing callback, and append two separate instances -- the
        ``_Destination`` refcount only helps when the *same* callback
        is shared).
        """
        import litellm

        from nooa.tracing._litellm_journal import (
            _INSTALL_LOCK,
            MessageJournalCallback,
        )

        with _INSTALL_LOCK:
            for cb in litellm.callbacks:
                # FileMessageJournalCallback subclasses MessageJournalCallback
                # to reuse normalization, but is a different sink.  Match the
                # exact class so exporter construction order cannot mix them.
                if isinstance(cb, MessageJournalCallback) and type(cb) is MessageJournalCallback:
                    cb.add_destination(self._base_url)
                    return cb

            cb = MessageJournalCallback(self._base_url)
            litellm.callbacks.append(cb)
            return cb

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans with LLM message attributes stripped."""
        return self._span_exporter.export(spans)

    def shutdown(self) -> None:
        """Drop this exporter's base URL from the shared callback.

        If we held the last destination, also drop the callback from
        ``litellm.callbacks`` entirely.  Other exporters' destinations on
        the same callback are left intact.

        Holds the same lock ``_install`` does so a concurrent
        ``enable_tracing`` for the same URL can't read ``litellm.callbacks``
        and our refcount in inconsistent states (e.g., see refcount
        about-to-hit-zero, decide to install a fresh callback, while we're
        still in the middle of the rewrite).
        """
        import litellm

        from nooa.tracing._litellm_journal import _INSTALL_LOCK, flush_pending

        # Join the journal callback's daemon-thread POSTs before tearing
        # down. OTel's built-in atexit hook calls ``TracerProvider.shutdown``
        # → ``BatchSpanProcessor.shutdown`` → ``self.shutdown``; without this
        # join, short-lived processes exit with in-flight journal POSTs
        # killed mid-request, so the viewer receives spans with no message
        # content (issue #168).
        flush_pending(timeout=30.0)

        with _INSTALL_LOCK:
            self._callback.remove_destination(self._base_url)
            if not self._callback.has_destinations():
                litellm.callbacks = [c for c in litellm.callbacks if c is not self._callback]

        self._span_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # Flush both halves of the wire:
        #   1. OTel span batch exporter (HTTP POST /v1/traces)
        #   2. Journal callback's fire-and-forget POSTs (/v1/journal/*)
        # Without (2) the eval pipeline's process-exit race truncates
        # in-flight POSTs at the receiver -- the call record never lands
        # and the viewer download can't reconstruct messages for that span.
        from nooa.tracing._litellm_journal import flush_pending

        flush_pending(timeout=timeout_millis / 1000.0)
        return self._span_exporter.force_flush(timeout_millis)

    def describe(self) -> str:
        return f"journal:{self._base_url}"
