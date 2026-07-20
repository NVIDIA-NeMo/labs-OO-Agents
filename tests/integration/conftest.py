# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for integration tests."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

# Make tests/tracing/otlp_test_helpers.py importable from integration tests.
sys.path.insert(0, str(Path(__file__).parent.parent / "tracing"))


def _reset_tracing_module_state() -> None:
    """Reset tracing module-level singletons between tests.

    Mirrors ``tests/tracing/conftest.py``.  Without this, tests that call
    ``enable_tracing`` repeatedly leave stale exporters / litellm callbacks
    behind, and one test's journal callback POSTs into another test's viewer.
    """
    import nooa.tracing as module
    from nooa.tracing._session import set_session

    if module._provider is not None:
        with contextlib.suppress(Exception):
            module._provider.shutdown()

    module._enabled = False
    module._provider = None
    module._probe_failed = False
    module._hooks = None

    set_session(None)

    with contextlib.suppress(ImportError):
        from nooa.runtime.hooks import set_hooks

        set_hooks(None)

    with contextlib.suppress(ImportError):
        from nooa.tracing._hooks_impl import _context_active_spans

        _context_active_spans.set(None)

    # Drop any MessageJournalCallback instances left in litellm's callback
    # lists by a previous test.  ``function_setup`` copies anything in
    # ``litellm.callbacks`` into ``success_callback`` /
    # ``_async_success_callback`` / etc on the first call -- those copies
    # outlive the original list and would let a stale callback keep firing
    # against a recorder that has since been torn down.
    with contextlib.suppress(ImportError):
        import litellm

        from nooa.tracing._litellm_journal import MessageJournalCallback

        def _strip(lst: list) -> list:
            return [cb for cb in lst if not isinstance(cb, MessageJournalCallback)]

        litellm.callbacks = _strip(litellm.callbacks)
        litellm.input_callback = _strip(litellm.input_callback)
        litellm.success_callback = _strip(litellm.success_callback)
        litellm.failure_callback = _strip(litellm.failure_callback)
        litellm._async_success_callback = _strip(litellm._async_success_callback)
        litellm._async_failure_callback = _strip(litellm._async_failure_callback)


@pytest.fixture(autouse=True)
def auto_reset_tracing_state():
    _reset_tracing_module_state()
    yield
    _reset_tracing_module_state()
