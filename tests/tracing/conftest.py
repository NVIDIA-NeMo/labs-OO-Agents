# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures for tracing tests.

Ensures proper isolation between tests by resetting module-level state.
"""

import contextlib
import sys
from pathlib import Path

import pytest

# Ensure the tests directory is importable (for otlp_test_helpers)
sys.path.insert(0, str(Path(__file__).parent))


def reset_tracing_module_state():
    """Reset module-level tracing state.

    Necessary because enable_tracing() uses module-level flags for idempotency.
    Tests need to reset this to verify behaviour in isolation.
    """
    import nooa.tracing as module
    from nooa.tracing._session import set_session

    # Shutdown provider if any
    if module._provider is not None:
        with contextlib.suppress(Exception):
            module._provider.shutdown()

    # Reset module-level flags
    module._enabled = False
    module._provider = None
    module._probe_failed = False
    module._hooks = None

    # Clear session in the OTel context
    set_session(None)

    # Reset instrumentation hooks ContextVar (set by enable_tracing / set_hooks)
    with contextlib.suppress(ImportError):
        from nooa.runtime.hooks import set_hooks

        set_hooks(None)

    # Reset _context_active_spans ContextVar.
    # sync tests (e.g. test_context_snapshot.py) call before_agent_call()
    # outside of any asyncio Task, which sets _context_active_spans in the
    # pytest main-thread context.  Without this reset, subsequent tests that
    # use asyncio.gather() inherit the shared dict and fail because all child
    # tasks see the same non-None value instead of getting their own fresh dict.
    with contextlib.suppress(ImportError):
        from nooa.tracing._hooks_impl import _context_active_spans

        _context_active_spans.set(None)

    # Drop any MessageJournalCallback instances from every litellm
    # callback list.  ``function_setup`` copies anything in
    # ``litellm.callbacks`` into ``success_callback`` /
    # ``_async_success_callback`` / etc on the first call -- those copies
    # outlive a test that only cleared ``litellm.callbacks`` and would
    # let a stale callback keep firing against a recorder that has since
    # been torn down.  Mirrors ``tests/integration/conftest.py``; the
    # ``test_conftest_reset.py`` meta-test pins the contract.
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
    """Automatically reset tracing state before AND after each test."""
    reset_tracing_module_state()
    yield
    # also clean up after — the last test in this package must not leave
    # _context_active_spans set in the main-thread context for other test files.
    reset_tracing_module_state()


@pytest.fixture
def clean_module_state():
    """Explicit fixture for backwards compat — the autouse fixture handles it."""
    yield
