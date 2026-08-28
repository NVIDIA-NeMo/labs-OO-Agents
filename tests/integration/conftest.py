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
    ``enable_tracing`` repeatedly leave stale exporters / any_llm callbacks
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

    # Clear native journal sinks left by a previous test.
    with contextlib.suppress(ImportError):
        from nooa.tracing._llm_journal import _SINKS, _SINKS_LOCK

        with _SINKS_LOCK:
            _SINKS.clear()


@pytest.fixture(autouse=True)
def auto_reset_tracing_state():
    _reset_tracing_module_state()
    yield
    _reset_tracing_module_state()
