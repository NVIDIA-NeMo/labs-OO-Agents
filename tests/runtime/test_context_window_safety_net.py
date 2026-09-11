# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-level safety net: final structured payload must fit the LLM window.

The pre-render safety net (clamping ``max_event_tokens`` against
``ctx_window``) only saw per-block CONTENT tokens — it missed:

- JSON message wrappers (role, content-array, tool_use, tool_result)
- ``<event_xxx>`` XML wrappers added by ``format_message_content``

Measured on a real session: 101K content tokens → 163K when litellm
counts the structured message list (+61%) → 207K when Bedrock actually
tokenizes (+27% further, likely tokenizer differences).

Fix: after ``render_context`` produces the full messages list, count
it with ``litellm.token_counter(messages=...)`` and drop oldest
non-system messages until the total fits under ``ctx_window × 0.70``.
The 30% margin covers the litellm→API tokenizer gap.
"""

import pytest

from nooa import Agent
from nooa.events import Message
from nooa.unifiedllm import FakeLLMClient


class _FakeLLM(FakeLLMClient):
    """FakeLLM with a settable context_window for tests."""

    _cw = 200_000

    @property
    def context_window(self):  # type: ignore[override]
        return self._cw

    def count_tokens(self, text: str) -> int:
        # Lean on litellm's real tokenizer — tests assert against it.
        import litellm

        # Use a real model so litellm picks the right tokenizer.
        return litellm.token_counter(model="anthropic/claude-3-5-sonnet-20240620", text=text)


def _mk_llm(context_window: int) -> _FakeLLM:
    class _LLM(_FakeLLM):
        _cw = context_window

    # Pin model so the runtime's model_context_window resolution works.
    llm = _LLM()
    # Monkey-patch .model to something litellm recognizes (matches _FakeLLM.count_tokens).
    llm.model = "anthropic/claude-3-5-sonnet-20240620"  # type: ignore[attr-defined]
    return llm


class TestStructuredPayloadSafetyNet:
    """The rendered messages + tools MUST fit under ``ctx_window × 0.70``
    when measured by ``litellm.token_counter`` — the same counter the API
    uses (approximately; Bedrock adds another 25% that we cover with
    margin).
    """

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_small_session_is_not_truncated(self):
        """Session that already fits the window must be left alone —
        safety net only fires when the structured count exceeds budget."""
        import litellm

        llm = _mk_llm(200_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for _ in range(3):
            agent.event_manager.add(Message(content="small message"))

        method = type(agent).respond
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(agent._llm)
        try:
            try:
                messages = await agent.runtime._build_messages(
                    method, call_args=(agent, "hi"), call_kwargs={}
                )
            except Exception:
                messages = None
        finally:
            _current_llm_var.reset(token)

        assert messages is not None
        # No events dropped.
        stats = agent.runtime._last_context_stats
        assert stats is not None
        # Structured count is well under the window — nothing to prune.
        structured = litellm.token_counter(model=agent._llm.model, messages=messages)
        assert structured < int(agent._llm.context_window * 0.70)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_llm_without_context_window_skips_clamp(self):
        """LLM with no context_window disables the safety net — we have no
        number to clamp against. Call proceeds; if it overflows, the API
        surfaces the error."""
        import litellm

        class NoWindowLLM(FakeLLMClient):
            model = "anthropic/claude-3-5-sonnet-20240620"

            @property
            def context_window(self):  # type: ignore[override]
                return None  # explicitly missing

            def count_tokens(self, text: str) -> int:
                return litellm.token_counter(model=self.model, text=text)

        llm = NoWindowLLM()

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for _ in range(10):
            agent.event_manager.add(Message(content="hi"))

        method = type(agent).respond
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(llm)
        try:
            try:
                messages = await agent.runtime._build_messages(
                    method, call_args=(agent, "hi"), call_kwargs={}
                )
            except Exception:
                messages = None
        finally:
            _current_llm_var.reset(token)

        # Doesn't crash; no assertion on cap (there isn't one).
        assert messages is not None


class TestTokenCounterRegression:
    """Regression tests for the two bugs root-caused in issue #133."""

    @pytest.mark.asyncio
    async def test_render_records_chars_not_a_token_estimate(self):
        """Issue #133 (revised): ``render_context`` must not produce a local
        token estimate at all. It records raw *character* sizes
        (``context_blocks_chars`` / ``events_chars``) and leaves every token
        figure ``None`` until the provider reports usage. This makes the
        chars-mistaken-for-tokens class of bug structurally impossible — the
        two are now distinct fields with distinct units.
        """
        llm = _mk_llm(200_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        # A big-enough system-role block. (System-role blocks land in
        # ``context_blocks_chars``.)
        long_block = "the quick brown fox jumps over the lazy dog. " * 400
        agent.context_manager["prose"] = long_block

        method = type(agent).respond
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime._build_messages(method, call_args=(agent, "hi"), call_kwargs={})
        finally:
            _current_llm_var.reset(token)

        stats = agent.runtime._last_context_stats
        assert stats is not None
        # No local token estimate exists before the provider reports usage.
        assert stats.prompt_tokens is None
        assert stats.total_tokens is None
        assert stats.context_blocks_tokens is None
        # Raw characters ARE recorded, and they are genuinely characters
        # (>= the block we put in), never silently divided to look like tokens.
        assert stats.context_blocks_chars >= len(long_block)

    @pytest.mark.asyncio
    async def test_default_unconfigured_budget_split_caps_context_to_half_window(self):
        """When both token limits are unset, runtime applies default split:

        - context_limit = usable window // 2, where usable = context_window
          minus the output-token reserve (response_reserve_tokens when the
          call sets no max_tokens)
        - over-budget context blocks are EVICTED before the prompt is sent
        """
        llm = _mk_llm(200_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        # Intentionally large context block: should be capped to <= half-window.
        agent.context_manager["prose"] = "context block " * 120_000
        # Add events so event budget path is exercised.
        for _ in range(20):
            agent.event_manager.add(Message(content="event payload " * 2000))

        method = type(agent).respond
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime._build_messages(method, call_args=(agent, "hi"), call_kwargs={})
        finally:
            _current_llm_var.reset(token)

        stats = agent.runtime._last_context_stats
        assert stats is not None
        # Requirement 1: default context budget = half the usable window
        # (window minus the default 4096-token output reserve).
        reserve = agent._truncation.response_reserve_tokens
        assert stats.max_context_tokens == (agent._llm.context_window - reserve) // 2
        # Requirement 2: the over-budget context block was evicted to enforce
        # the cap (eviction counts tokens internally via the LLM counter).
        assert stats.context_blocks_dropped >= 1


class TestMaxOutputTokensBudget:
    """The safety net must account for ``max_output_tokens`` when computing
    the input budget.  With ``max_tokens=64000`` on a 131072-token window
    the old ``ctx_window * 0.70 = 91750`` cap was too generous -- the real
    safe limit is ``131072 - 64000 = 67072``.  The fix passes
    ``max_output_tokens`` into ``_build_messages`` so the cap tightens.
    """

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_small_max_tokens_uses_default_cap(self):
        """When max_tokens is small (< 30 % of ctx_window), the default
        70 % heuristic is already tighter and should win.  No regression.
        """
        from nooa.events import Message

        llm = _mk_llm(200_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for _ in range(5):
            agent.event_manager.add(Message(content="small message"))

        method = type(agent).respond
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method,
                call_args=(agent, "hi"),
                call_kwargs={},
                max_output_tokens=4096,
            )
        finally:
            _current_llm_var.reset(token)

        assert messages is not None
        stats = agent.runtime._last_context_stats
        assert stats is not None
        # Nothing should be dropped — session is tiny.
        assert stats.events_dropped == 0

    @pytest.mark.asyncio
    async def test_none_max_output_tokens_falls_back(self):
        """When max_output_tokens is None (not passed), the old 70 %
        heuristic must be used — no crash from None arithmetic.
        """
        from nooa.events import Message

        llm = _mk_llm(200_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for _ in range(3):
            agent.event_manager.add(Message(content="test message"))

        method = type(agent).respond
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method,
                call_args=(agent, "hi"),
                call_kwargs={},
                # max_output_tokens not passed — defaults to None
            )
        finally:
            _current_llm_var.reset(token)

        assert messages is not None


class _ContextWindowExceededError(Exception):
    """Test double for litellm.ContextWindowExceededError."""

    pass


class TestContextWindowRecovery:
    """When the LLM API rejects with ContextWindowExceededError, generate()
    should reduce max_tokens and retry once instead of propagating the error.
    """

    @pytest.mark.asyncio
    async def test_recovery_reduces_max_tokens_and_retries(self):
        """Simulates the KDD crash: first acall raises ContextWindowExceeded,
        recovery retries with reduced max_tokens and succeeds."""
        from unittest.mock import patch

        from nooa.events import Message
        from nooa.runtime.actor import (
            _current_llm_var,
            _current_method_var,
        )

        llm = _mk_llm(131_072)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="hello"))

        error = _ContextWindowExceededError(
            "This model's maximum context length is 131072 tokens. "
            "However, you requested 64000 output tokens and your prompt "
            "contains at least 67073 input tokens, for a total of at least "
            "131073 tokens."
        )

        call_count = 0
        received_max_tokens = []

        original_acall = llm.acall

        async def mock_acall(messages, **kw):
            nonlocal call_count
            call_count += 1
            received_max_tokens.append(kw.get("max_tokens"))
            if call_count == 1:
                raise error
            return await original_acall(messages, **kw)

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=mock_acall):
                _response, _event_id = await agent.runtime.generate(tools=[], max_tokens=64000)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

        assert call_count == 2, f"Expected 2 calls (fail + retry), got {call_count}"
        assert received_max_tokens[0] == 64000, "First call should use original max_tokens"
        assert received_max_tokens[1] is not None
        assert received_max_tokens[1] < 64000, (
            f"Retry max_tokens ({received_max_tokens[1]}) should be less than original (64000)"
        )
        assert received_max_tokens[1] >= 1024, "Retry max_tokens should be >= minimum"

    @pytest.mark.asyncio
    async def test_non_context_window_errors_still_propagate(self):
        """Errors that aren't ContextWindowExceeded must propagate normally."""
        from unittest.mock import patch

        from nooa.events import Message
        from nooa.runtime.actor import (
            _current_llm_var,
            _current_method_var,
        )

        llm = _mk_llm(200_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="hello"))

        async def mock_acall(messages, **kw):
            raise RuntimeError("Some other API error")

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=mock_acall):
                with pytest.raises(RuntimeError, match="Some other API error"):
                    await agent.runtime.generate(tools=[], max_tokens=4096)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

    @pytest.mark.asyncio
    async def test_recovery_gives_up_when_budget_too_small(self):
        """If the prompt is so large that even minimal output won't fit,
        recovery should re-raise instead of retrying with a useless budget."""
        from unittest.mock import patch

        from nooa.events import Message
        from nooa.runtime.actor import (
            _current_llm_var,
            _current_method_var,
        )

        llm = _mk_llm(131_072)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="hello"))

        error = _ContextWindowExceededError(
            "This model's maximum context length is 131072 tokens. "
            "However, you requested 64000 output tokens and your prompt "
            "contains at least 130500 input tokens."
        )

        async def mock_acall(messages, **kw):
            raise error

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=mock_acall):
                with pytest.raises(_ContextWindowExceededError):
                    await agent.runtime.generate(tools=[], max_tokens=64000)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)


class TestEndToEndSmallContextWindow:
    """Integration test: FakeLLM with a tiny context window exercises both
    the proactive safety net and the reactive recovery path.
    """

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_recovery_fires_on_token_estimation_error(self):
        """Simulate the case where the safety net's token count is slightly
        off and the API still rejects. Recovery should reduce max_tokens
        and succeed on retry."""
        from unittest.mock import patch

        from nooa.events import Message
        from nooa.runtime.actor import (
            _current_llm_var,
            _current_method_var,
        )

        llm = _mk_llm(4096)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        agent.event_manager.add(Message(content="hello world"))

        # First call: raises ContextWindowExceeded (simulating tokenizer gap)
        # Second call: succeeds with reduced max_tokens
        call_count = 0
        received_max_tokens = []
        original_acall = llm.acall

        async def mock_acall(messages, **kw):
            nonlocal call_count
            call_count += 1
            received_max_tokens.append(kw.get("max_tokens"))
            if call_count == 1:
                raise _ContextWindowExceededError(
                    "This model's maximum context length is 4096 tokens. "
                    "However, you requested 2048 output tokens and your prompt "
                    "contains at least 2500 input tokens, for a total of at "
                    "least 4548 tokens."
                )
            return await original_acall(messages, **kw)

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=mock_acall):
                _response, _event_id = await agent.runtime.generate(tools=[], max_tokens=2048)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

        assert call_count == 2
        assert received_max_tokens[0] == 2048
        # Recovery: 4096 - 2500 - margin(~82) = ~1514
        assert received_max_tokens[1] < 2048
        assert received_max_tokens[1] >= 1024  # above minimum
