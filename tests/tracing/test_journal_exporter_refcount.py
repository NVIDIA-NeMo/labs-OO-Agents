# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for shared-destination refcount in JournalExporter.

Two ``JournalExporter`` instances pointed at the same base URL share a
single ``MessageJournalCallback`` destination -- the refcount makes
``shutdown`` safe so the first exporter to go away doesn't kill the URL
the second still owns.  Without this, hot-reload / repeated
``enable_tracing`` cycles silently lose journal delivery on whichever
sink survived.
"""

from __future__ import annotations

import litellm

from nooa.tracing._journal_exporter import JournalExporter
from nooa.tracing._litellm_journal import MessageJournalCallback


def _journal_callbacks() -> list[MessageJournalCallback]:
    return [cb for cb in litellm.callbacks if isinstance(cb, MessageJournalCallback)]


def test_two_exporters_same_url_share_one_callback_with_refcount_two():
    a = JournalExporter("http://shared.invalid")
    try:
        b = JournalExporter("http://shared.invalid")
        try:
            cbs = _journal_callbacks()
            assert len(cbs) == 1, (
                f"two exporters at the same URL should share one callback, got {cbs!r}"
            )
            (cb,) = cbs
            (dest,) = cb._destinations
            assert dest.refcount == 2, (
                f"each add_destination must bump refcount; got {dest.refcount}"
            )
        finally:
            b.shutdown()
    finally:
        a.shutdown()


def test_first_shutdown_keeps_destination_alive_for_second_exporter():
    a = JournalExporter("http://shared.invalid")
    b = JournalExporter("http://shared.invalid")

    a.shutdown()  # first shutdown should NOT kill the destination

    cbs = _journal_callbacks()
    assert len(cbs) == 1, "callback removed before all referencing exporters shut down"
    (cb,) = cbs
    assert cb.base_urls == ["http://shared.invalid"], (
        f"second exporter should still own the URL, got {cb.base_urls!r}"
    )

    # Sanity: the second shutdown drops the destination + the callback.
    b.shutdown()
    assert _journal_callbacks() == [], "callback should be removed at refcount=0"


def test_distinct_urls_each_get_own_destination():
    a = JournalExporter("http://a.invalid")
    try:
        b = JournalExporter("http://b.invalid")
        try:
            (cb,) = _journal_callbacks()
            assert sorted(cb.base_urls) == [
                "http://a.invalid",
                "http://b.invalid",
            ]
            for dest in cb._destinations:
                assert dest.refcount == 1
        finally:
            b.shutdown()
    finally:
        a.shutdown()
    assert _journal_callbacks() == []
