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
    from nooa.tracing import _llm_hooks

    _llm_hooks.callbacks.clear()

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


@pytest.fixture
def isolated_gate_tracing(auto_reset_tracing_state, monkeypatch):
    """Do not discover a viewer or export live gate prompts/replies externally."""
    import nooa.tracing as tracing

    # Unsetting OTLP_ENDPOINT alone still probes the default local viewer.
    monkeypatch.setattr(tracing, "_default_exporters", lambda: [])
    tracing.enable_tracing(exporters=[])
    # OTel's process-global provider can outlive the module reset. Reconfigure
    # explicitly to remove processors retained from an earlier test as well.
    tracing.enable_tracing(exporters=[])


@pytest.fixture
def mock_model_client(monkeypatch):
    """Real UnifiedLLM/SDK dispatch with only the model's HTTP pool mocked."""
    import httpx

    from nooa.unifiedllm import CompletionClient
    from nooa.unifiedllm.direct import DirectTransport
    from nooa.unifiedllm.unifiedllm import _ClientHttp

    def make(reply, transport="direct"):
        def respond(request):
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": reply},
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )

        mock = httpx.MockTransport(respond)
        monkeypatch.setattr(
            DirectTransport, "_http_settings", staticmethod(lambda config: {"transport": mock})
        )
        monkeypatch.setattr(
            _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": mock})
        )
        return CompletionClient(
            "openai/test", transport=transport, api_key="test", api_base="https://models.example/v1"
        )

    return make
