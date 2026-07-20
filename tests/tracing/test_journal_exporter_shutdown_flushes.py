# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issue #168: ``JournalExporter.shutdown`` must flush in-flight journal POSTs.

The OTel SDK already registers ``TracerProvider.shutdown`` with ``atexit``,
which fans out to every ``BatchSpanProcessor.shutdown`` and then to the
exporter's ``shutdown``.  For ``JournalExporter`` the wire has two halves:

  1. OTel span batch (HTTP POST /v1/traces) — drained by
     ``BatchSpanProcessor.shutdown`` before our exporter's ``shutdown`` runs.
  2. Journal callback's daemon-thread POSTs (/v1/journal/calls,
     /v1/journal/blocks) — these are *not* drained unless someone calls
     ``flush_pending``.

Pre-fix, ``JournalExporter.shutdown`` only delegated to the inner span
exporter and never joined the journal threads, so short-lived processes
(e.g. an LLM-generated ``main.py`` invoked once per eval sample) exited
before the call records landed; the viewer received OTLP spans with no
message content.

This test pins ``shutdown`` to the same join semantics as ``force_flush``.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch


def test_shutdown_joins_pending_journal_posts():
    """``JournalExporter.shutdown`` must not return until in-flight
    journal POSTs have run their HTTP request to completion.

    Without this, OTel's atexit hook fires ``TracerProvider.shutdown``
    → ``JournalExporter.shutdown`` → returns immediately → process exits
    → daemon-thread POSTs are killed mid-flight."""
    from nooa.tracing._journal_exporter import JournalExporter
    from nooa.tracing._litellm_journal import _post_json

    exporter = JournalExporter("http://example.invalid")

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_post(*_args, **_kwargs):
        started.set()
        release.wait(timeout=10)
        finished.set()
        # Return a context-manager-able fake so the worker doesn't blow up
        # on the ``with urlopen(...) as r`` of ``_post_json``.
        fake = MagicMock()
        fake.__enter__ = lambda self: MagicMock(status=200)
        fake.__exit__ = lambda *a: False
        return fake

    with patch(
        "nooa.tracing._litellm_journal.urllib.request.urlopen",
        side_effect=slow_post,
    ):
        # Dispatch a journal POST; daemon thread blocks inside urlopen.
        _post_json("http://example.invalid/v1/journal/calls", {"call_id": "x"})
        assert started.wait(timeout=2), "POST thread didn't start"

        # Run shutdown() in a side thread so we can observe whether it
        # blocks until the slow POST completes.
        shutdown_done = threading.Event()

        def _shutdown():
            exporter.shutdown()
            shutdown_done.set()

        threading.Thread(target=_shutdown, daemon=True).start()

        # If shutdown returns before we release the POST, the bug is back:
        # the daemon thread is still inside urlopen when the process would
        # otherwise exit.
        assert not shutdown_done.wait(timeout=0.3), (
            "JournalExporter.shutdown returned while a journal POST was "
            "still in flight — issue #168 regression."
        )

        # Release the POST; shutdown should now return promptly.
        release.set()
        assert shutdown_done.wait(timeout=5), (
            "JournalExporter.shutdown did not return after the journal POST completed"
        )
        assert finished.is_set()
