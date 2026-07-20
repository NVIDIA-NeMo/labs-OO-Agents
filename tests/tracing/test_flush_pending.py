# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``flush_pending`` -- the journal callback's daemon-thread join.

The eval pipeline's process-exit race truncated journal POSTs at the
receiver because ``_post_json`` dispatches into daemon threads and the
calling process can exit before they finish.  ``JournalExporter.force_flush``
calls ``flush_pending`` to block on those threads.  This module covers
the join behaviour directly, including the append-vs-start race that
the original implementation got wrong.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from nooa.tracing._litellm_journal import (
    _PENDING_THREADS,
    _post_json,
    flush_pending,
)


def test_flush_pending_blocks_until_post_completes():
    """``flush_pending`` must not return until in-flight POSTs have run
    their HTTP request to completion.  Otherwise the eval-pipeline
    process-exit race comes back."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_post(*_args, **_kwargs):
        started.set()
        release.wait(timeout=10)
        finished.set()

    with patch(
        "nooa.tracing._litellm_journal.urllib.request.urlopen",
        side_effect=slow_post,
    ):
        _post_json("http://example.invalid/v1/journal/calls", {"call_id": "x"})
        assert started.wait(timeout=2), "POST thread didn't start"

        # Run flush_pending in another thread so we can observe it block.
        flushed = threading.Event()

        def _flush():
            flush_pending(timeout=10)
            flushed.set()

        flush_thread = threading.Thread(target=_flush, daemon=True)
        flush_thread.start()
        # If it returned immediately, the test infra is broken.
        assert not flushed.wait(timeout=0.2), (
            "flush_pending returned before the POST thread finished"
        )

        release.set()
        assert flushed.wait(timeout=5), "flush_pending didn't return after POST done"
        assert finished.is_set()


def test_flush_pending_handles_thread_added_during_flush():
    """A POST dispatched *while* ``flush_pending`` is mid-join must still
    be waited on -- otherwise back-to-back ``log_success_event`` calls
    drop the second batch.

    Coordination:
      1. Dispatch the first POST; it blocks inside ``urlopen`` until
         released.
      2. Start ``flush_pending`` in a side thread; it enters its first
         iteration and is joining the first POST.
      3. Dispatch a second POST — appended to ``_PENDING_THREADS`` while
         flush is mid-join.
      4. Release the first POST.
      5. flush returns only after the second POST has *also* run; both
         ``"p1"`` and ``"p2"`` are in ``seen``.
    """
    from unittest.mock import MagicMock

    seen: list[str] = []
    started_post1 = threading.Event()
    release_post1 = threading.Event()

    fake_response = MagicMock()
    fake_response.__enter__ = lambda self: MagicMock(status=200)
    fake_response.__exit__ = lambda *a: False

    call_count = [0]

    def dispatcher(*_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            started_post1.set()
            release_post1.wait(timeout=5)
            seen.append("p1")
        else:
            seen.append("p2")
        return fake_response

    with patch(
        "nooa.tracing._litellm_journal.urllib.request.urlopen",
        side_effect=dispatcher,
    ):
        _post_json("http://example.invalid/one", {"x": 1})
        assert started_post1.wait(timeout=2), "first POST didn't start"

        flush_done = threading.Event()

        def _flush():
            flush_pending(timeout=5)
            flush_done.set()

        threading.Thread(target=_flush, daemon=True).start()
        # Let flush enter its first iteration and start joining the
        # first POST (still blocked on release_post1).
        time.sleep(0.05)

        # Append the second POST while flush is mid-join.
        _post_json("http://example.invalid/two", {"x": 2})

        # Release the first POST so it can finish; flush must then loop
        # and pick up the second one.
        release_post1.set()
        assert flush_done.wait(timeout=5), "flush_pending didn't return"

    assert "p1" in seen
    # The key assertion: flush_pending waited for the second POST too.
    # If it returned after the first iteration only, ``seen`` would
    # lack ``"p2"`` and the append-during-flush bug would slip through.
    assert "p2" in seen, f"flush_pending returned before the second POST completed; seen={seen!r}"


def test_pending_threads_self_evict_when_post_returns():
    """``_PENDING_THREADS`` must not grow without bound: the worker thread
    discards itself from the set when it returns, so a long-running
    process that never calls ``flush_pending`` doesn't leak."""

    def fast_post(*_args, **_kwargs):
        return None

    with patch(
        "nooa.tracing._litellm_journal.urllib.request.urlopen",
        side_effect=fast_post,
    ):
        for i in range(5):
            _post_json(f"http://example.invalid/{i}", {"i": i})

        # Wait for all threads to drain.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _PENDING_THREADS:
            time.sleep(0.02)
        assert not _PENDING_THREADS, (
            f"_PENDING_THREADS leaked entries after completion: {_PENDING_THREADS!r}"
        )
