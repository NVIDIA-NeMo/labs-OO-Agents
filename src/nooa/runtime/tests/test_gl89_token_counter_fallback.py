# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for char_approximate_token_counter and chars→tokens estimation.

gl-89 history: an earlier revision raised a RuntimeError when token limits were
set but the LLM had no count_tokens. That requirement has since been removed:
the runtime now sizes eviction with a provider-calibrated chars→tokens ratio
(prompt_tokens / total_chars from the last response, defaulting to ~4 chars per
token). No LLM token counter is required, and there is no RuntimeError.
``char_approximate_token_counter`` remains a public utility (the same 4-chars
heuristic) that users may still attach to an LLM as ``count_tokens``.
"""

from unittest.mock import Mock

import pytest


class TestCharApproximateTokenCounter:
    """char_approximate_token_counter is a public explicit-opt-in utility."""

    def test_is_importable_from_nooa(self):
        from nooa import char_approximate_token_counter

        assert callable(char_approximate_token_counter)

    def test_is_importable_from_token_counter_module(self):
        from nooa.token_counter import char_approximate_token_counter

        assert callable(char_approximate_token_counter)

    def test_formula_is_len_div_4(self):
        from nooa import char_approximate_token_counter

        assert char_approximate_token_counter("") == 0
        assert char_approximate_token_counter("xxxx") == 1
        assert char_approximate_token_counter("x" * 100) == 25
        assert char_approximate_token_counter("x" * 1000) == 250

    def test_returns_non_negative_int(self):
        from nooa import char_approximate_token_counter

        for text in ["", "x", "hello world", "a" * 10_000]:
            result = char_approximate_token_counter(text)
            assert isinstance(result, int)
            assert result >= 0

    def test_can_be_attached_as_count_tokens(self):
        """Users can attach it to their LLM as count_tokens."""
        from nooa import char_approximate_token_counter

        llm = Mock()
        llm.count_tokens = char_approximate_token_counter
        assert llm.count_tokens("x" * 40) == 10


class TestCalibratedFallback:
    """Runtime needs no LLM count_tokens: eviction is sized with a
    provider-calibrated chars→tokens ratio, not the LLM tokenizer."""

    def test_actor_has_no_get_token_counter(self):
        """_get_token_counter was removed long ago — no per-call counter lookup."""
        import nooa.runtime.actor as actor_mod

        assert not hasattr(actor_mod, "_get_token_counter")

    def test_actor_does_not_require_an_llm_token_counter(self):
        """The old RuntimeError demanding an LLM count_tokens is gone; the
        runtime defines a cold-start chars→tokens ratio instead."""
        import inspect

        import nooa.runtime.actor as actor_mod

        source = inspect.getsource(actor_mod)
        assert "has no count_tokens method" not in source
        assert "_DEFAULT_TOKENS_PER_CHAR" in source

    @pytest.mark.asyncio
    async def test_build_messages_without_llm_counter_does_not_raise(self):
        """An LLM that exposes a context_window but no usable token counter
        still builds messages — eviction uses the calibrated ratio."""
        from nooa import Agent
        from nooa.events import Message
        from nooa.runtime.actor import _current_llm_var
        from nooa.unifiedllm import FakeLLMClient

        class _NoCounterLLM(FakeLLMClient):
            @property
            def context_window(self):  # type: ignore[override]
                return 200_000

            # No count_tokens override — the runtime must not need one.

        llm = _NoCounterLLM()

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for _ in range(5):
            agent.event_manager.add(Message(content="hello"))

        method = type(agent).respond
        token = _current_llm_var.set(llm)
        try:
            messages = await agent.runtime._build_messages(
                method, call_args=(agent, "hi"), call_kwargs={}
            )
        finally:
            _current_llm_var.reset(token)
        assert messages is not None
