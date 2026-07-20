# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Meta-test: ``conftest._reset_tracing_module_state`` actually drops
``MessageJournalCallback`` instances from every litellm callback list.

Background.  ``litellm.function_setup`` copies anything in
``litellm.callbacks`` into ``input_callback`` / ``success_callback`` /
``failure_callback`` / ``_async_success_callback`` /
``_async_failure_callback`` on the first call.  If a previous test left
a callback there and the conftest only cleared ``litellm.callbacks``,
the leaked callback would keep firing against whatever recorder the
old test had spun up -- silently breaking subsequent tests with
"recorder got an unexpected POST" or "session_id from a stranger".

The conftest's ``_strip`` helper is the contract that prevents this.
This test pins it.  If a litellm version drift adds a *new* callback
list, this test fails in a way that says "go look at conftest".
"""

from __future__ import annotations

import litellm

from nooa.tracing._litellm_journal import MessageJournalCallback

# The list of litellm callback lists the conftest is responsible for
# stripping.  If this drifts from what ``conftest._reset_tracing_module_state``
# strips, one side or the other is buggy -- this test surfaces the
# discrepancy explicitly.
_LITELLM_CALLBACK_LISTS = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


def test_conftest_strip_removes_journal_callback_from_every_litellm_list():
    """Plant a ``MessageJournalCallback`` instance into every list, run
    the conftest reset, assert all six lists are clean."""
    from tests.integration.conftest import _reset_tracing_module_state

    cb = MessageJournalCallback("http://meta-test.invalid")
    for name in _LITELLM_CALLBACK_LISTS:
        getattr(litellm, name).append(cb)

    # Sanity: planted in all of them.
    for name in _LITELLM_CALLBACK_LISTS:
        assert any(isinstance(c, MessageJournalCallback) for c in getattr(litellm, name)), (
            f"setup precondition: list {name!r} should contain the planted callback"
        )

    _reset_tracing_module_state()

    leaked = [
        name
        for name in _LITELLM_CALLBACK_LISTS
        if any(isinstance(c, MessageJournalCallback) for c in getattr(litellm, name))
    ]
    assert not leaked, (
        f"_reset_tracing_module_state must strip MessageJournalCallback from "
        f"every litellm callback list; still present in: {leaked!r}.  Update "
        f"the conftest's ``_strip`` helper to cover any new lists."
    )


def test_conftest_strip_leaves_other_callbacks_alone():
    """The strip is type-targeted: callbacks that aren't
    ``MessageJournalCallback`` instances must survive the reset.  Otherwise
    it would clobber unrelated callbacks owned by the test suite or
    third-party plugins."""
    from litellm.integrations.custom_logger import CustomLogger

    from tests.integration.conftest import _reset_tracing_module_state

    class _Sentinel(CustomLogger):
        pass

    keep = _Sentinel()
    drop = MessageJournalCallback("http://meta-test.invalid")
    try:
        litellm.callbacks.append(keep)
        litellm.callbacks.append(drop)

        _reset_tracing_module_state()

        assert keep in litellm.callbacks, "non-journal CustomLogger was incorrectly stripped"
        assert drop not in litellm.callbacks
    finally:
        # Tidy up: if the strip-under-test is buggy and leaks ``drop``
        # into later tests, this finally also clears it.  Without the
        # ``drop`` cleanup a regression in ``_strip`` would silently
        # corrupt the callback list for downstream tests instead of
        # failing here.
        if keep in litellm.callbacks:
            litellm.callbacks.remove(keep)
        if drop in litellm.callbacks:
            litellm.callbacks.remove(drop)
