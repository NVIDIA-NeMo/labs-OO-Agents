# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for L4 context eviction.

Inline tests use FakeLLMClient (no API calls). Tests marked @pytest.mark.integration
require a real LLM and should run in nightly CI only.
"""

import pytest

from nooa import Agent
from nooa.config.truncation_config import TruncationConfig
from nooa.context_blocks import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import OpenAIProviderFormatter, XMLBlockFormatter
from nooa.context_blocks.renderer import render_context
from nooa.events import PythonOutput
from nooa.runtime.actor import _current_llm_var
from nooa.runtime.harness_metrics import harness_metrics_session
from nooa.unifiedllm import FakeLLMClient

# ── Helpers ──────────────────────────────────────────────────────────────


class _FakeLLM(FakeLLMClient):
    """FakeLLM with a settable context_window for tests."""

    _cw = 4096

    @property
    def context_window(self):  # type: ignore[override]
        return self._cw

    def count_tokens(self, text: str) -> int:
        import litellm

        return litellm.token_counter(model="anthropic/claude-3-5-sonnet-20240620", text=text)


def _mk_agent(context_window: int = 4096, max_context_tokens: int | None = None) -> Agent:
    """Create a test agent with configurable window and context budget."""

    class _LLM(_FakeLLM):
        _cw = context_window

    llm = _LLM()
    llm.model = "anthropic/claude-3-5-sonnet-20240620"  # type: ignore[attr-defined]

    if max_context_tokens is not None:
        tc = TruncationConfig(max_context_tokens=max_context_tokens)

        class A(Agent, llm=llm, truncation=tc):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
    else:

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
    return agent


def _count_words(s: str) -> int:
    """Simple word-count token approximation for tests."""
    return len(s.split())


# ── Context Block EVICTION ───────────────────────────────────────────────


class TestContextBlockEviction:
    """Context blocks over budget are marked EVICTED in-place."""

    def test_over_budget_block_gets_evicted_marker(self):
        """Non-static blocks exceeding context_limit get EVICTED content."""
        blocks = [
            ResolvedBlock(
                key="system_prompt",
                content="You are helpful.",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="big_block",
                content="data " * 200,
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            ),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=50,
            count_tokens=_count_words,
        )

        assert result.stats.context_blocks_dropped > 0
        # EVICTED marker present in output
        output_str = str(result.output)
        assert "EVICTED" in output_str

    def test_static_blocks_never_evicted(self):
        """Blocks with static=True survive regardless of budget."""
        blocks = [
            ResolvedBlock(
                key="immutable",
                content="x " * 200,
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=50,
            count_tokens=_count_words,
        )

        assert result.stats.context_blocks_dropped == 0

    def test_multiple_blocks_evicted_newest_first(self):
        """When multiple blocks exceed budget, newest are evicted first
        (eviction works from the end to preserve oldest/most-stable blocks)."""
        blocks = [
            ResolvedBlock(
                key=f"block_{i}",
                content=f"content_{i} " * 50,
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            )
            for i in range(5)
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=100,
            count_tokens=_count_words,
        )

        assert result.stats.context_blocks_dropped >= 3
        # Oldest blocks survive (eviction removes from end)
        output_str = str(result.output)
        assert "content_0" in output_str

    def test_render_context_does_not_touch_runtime_metrics(self):
        """render_context is a framework-agnostic leaf: it reports evictions via
        stats and never reaches into the runtime's HarnessMetrics (issue #330).

        The eviction count is surfaced on ``stats.context_blocks_dropped``; the
        runtime is responsible for folding that into
        ``context_limits_blocks_evicted``.
        """
        blocks = [
            ResolvedBlock(
                key=f"block_{i}",
                content="y " * 100,
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            )
            for i in range(5)
        ]

        with harness_metrics_session() as hm:
            result = render_context(
                blocks,
                block_formatter=XMLBlockFormatter(),
                provider_formatter=OpenAIProviderFormatter(),
                context_limit=200,
                count_tokens=_count_words,
            )
            # Evictions are reported via stats ...
            assert result.stats.context_blocks_dropped > 0
            # ... but the leaf library must NOT mutate the runtime metric.
            assert hm.context_limits_blocks_evicted == 0

    @pytest.mark.asyncio
    async def test_harness_metrics_track_eviction_end_to_end(self):
        """context_limits_blocks_evicted is populated on eviction via the runtime.

        Drives the actor's ``_build_messages`` (the real caller of
        ``render_context``) so the eviction count flows from the renderer's
        stats into the runtime-owned HarnessMetrics — the value must match
        ``stats.context_blocks_dropped``.
        """
        agent = _mk_agent(context_window=4096, max_context_tokens=50)
        # Oversized, non-static context block guarantees eviction under the budget.
        agent.context["big_block"] = "data " * 500

        tok = _current_llm_var.set(agent._llm)
        try:
            with harness_metrics_session() as hm:
                await agent.runtime._build_messages(agent.respond, ("hi",))
                stats = agent.runtime._last_context_stats
                assert stats is not None
                assert stats.context_blocks_dropped > 0
                assert hm.context_limits_blocks_evicted == stats.context_blocks_dropped
        finally:
            _current_llm_var.reset(tok)


# ── pformat(unquote_strings=True) ───────────────────────────────────────


class TestUnquoteStrings:
    """Context blocks render strings verbatim (no quotes) via unquote_strings."""

    def test_short_string_no_quotes(self):
        from nooa.agentdoc import pformat

        assert pformat("Hello world", unquote_strings=True) == "Hello world"

    def test_long_string_gets_truncation_marker(self):
        from nooa.agentdoc import pformat

        result = pformat("a" * 1000, unquote_strings=True, max_string=100)
        assert result.startswith("str(len=1000")
        assert "[:50]=" in result

    def test_non_string_unaffected(self):
        from nooa.agentdoc import pformat

        assert pformat([1, 2, 3], unquote_strings=True) == pformat([1, 2, 3])

    def test_multiline_string_verbatim(self):
        from nooa.agentdoc import pformat

        text = "line1\nline2\nline3"
        result = pformat(text, unquote_strings=True)
        assert result == text

    def test_triple_quote_string_preserved(self):
        """String consisting of triple quotes (''') is not deleted by unquote logic."""
        from nooa.agentdoc import pformat

        # Edge case: input is literally 3 single-quote chars
        result = pformat("'''", unquote_strings=True)
        assert result == "'''", f"Expected '''' but got {repr(result)}"

    def test_string_with_embedded_triple_quotes(self):
        """String containing triple quotes inside is preserved."""
        from nooa.agentdoc import pformat

        text = "before''' after"
        result = pformat(text, unquote_strings=True)
        assert result == text


# ── Post-render event collapse ───────────────────────────────────────────


class TestPostRenderEventCollapse:
    """When rendered payload exceeds context_window, events are collapsed."""

    @pytest.mark.asyncio
    async def test_overflow_passes_all_messages_through(self):
        """Without proactive clamping, all events pass through _build_messages.

        The API will reject if context is too large, and the recovery path
        in generate() handles archival.
        """
        agent = _mk_agent(context_window=4096)

        # Fill with many events (will exceed 4096 token window)
        for i in range(20):
            tc_id = f"tc_{i}"
            agent.event_manager.add(
                ToolCallEvent(
                    tool_call_id=tc_id,
                    name="execute_python",
                    arguments={"code": "x " * 100},
                    result=ToolResult(
                        tool_call_id=tc_id,
                        content="done",
                        result_status=ResultStatus.COMPLETE,
                    ),
                )
            )
            agent.event_manager.add(
                PythonOutput(
                    tool_call_id=tc_id,
                    execution_count=i,
                    stdout="y " * 100,
                    stderr="",
                    execution_status=ResultStatus.COMPLETE,
                )
            )

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method,
                call_args=(agent, "hi"),
                call_kwargs={},
                max_output_tokens=2048,
            )
        finally:
            _current_llm_var.reset(token)

        assert messages is not None
        # No clamping — all events should be in the message list
        active = len(list(agent.event_manager.keys()))
        assert active == 40, f"All 40 events should still be active, got {active}"


# ── Default budget split ─────────────────────────────────────────────────


class TestDefaultBudgetSplit:
    """When max_context_tokens is unset, context budget = usable window // 2,
    where usable = context_window − output-token reserve (the call's
    max_tokens, else TruncationConfig.response_reserve_tokens)."""

    @pytest.mark.asyncio
    async def test_half_window_cap_evicts_large_context(self):
        """Large context blocks are evicted when they exceed half the usable window.

        Window 8192 − 2048 reserved for output = 6144 usable → 3072 budget.
        Eviction sizes blocks with the chars→tokens ratio (cold-start ~1/4),
        so the block must exceed 3072 tokens ≈ 12,288 chars to be evicted.
        """
        agent = _mk_agent(context_window=8192)

        # ~40,000 chars ≈ 10,000 tokens at the cold-start ratio — well over
        # the 3072-token budget.
        agent.context["huge"] = "z " * 20000

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime._build_messages(
                method,
                call_args=(agent, "hi"),
                call_kwargs={},
                max_output_tokens=2048,
            )
        finally:
            _current_llm_var.reset(token)

        stats = agent.runtime._last_context_stats
        assert stats is not None
        # Eviction fired: the over-budget context block was dropped.
        assert stats.context_blocks_dropped > 0

    @pytest.mark.asyncio
    async def test_eviction_consults_the_calibrated_ratio(self):
        """Eviction sizing reads the live _tokens_per_char: a block that would
        be evicted at the cold-start ratio survives once the ratio is lower."""
        # No explicit max_tokens → default 4096 output reserve applies:
        # usable = 8192 − 4096 = 4096 → budget = 2048 tokens.
        agent = _mk_agent(context_window=8192)

        # 24,000 chars. At the cold-start ratio 0.25 → 6,000 tokens > 2048
        # (would evict). At a calibrated ratio of 0.08 → 1,920 tokens < 2048.
        agent.context["mid"] = "z " * 12000
        agent.runtime._tokens_per_char = 0.08

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime._build_messages(method, call_args=(agent, "hi"), call_kwargs={})
        finally:
            _current_llm_var.reset(token)

        stats = agent.runtime._last_context_stats
        assert stats is not None
        # Lower ratio → block fits → no eviction. Proves the ratio is consulted.
        assert stats.context_blocks_dropped == 0

    @pytest.mark.asyncio
    async def test_output_reserve_shrinks_default_budget(self):
        """The call's max_tokens is reserved out of the window before the
        half split: a block that fits half the RAW window is still evicted
        when the completion budget leaves less usable room."""
        agent = _mk_agent(context_window=8192)

        # 12,000 chars ≈ 3,000 tokens at the cold-start ratio. Under the old
        # raw half-window budget (4096) it would fit; with max_tokens=4096 the
        # usable window is 4096 → budget 2048 → evicted.
        agent.context["mid"] = "z " * 6000

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime._build_messages(
                method,
                call_args=(agent, "hi"),
                call_kwargs={},
                max_output_tokens=4096,
            )
        finally:
            _current_llm_var.reset(token)

        stats = agent.runtime._last_context_stats
        assert stats is not None
        assert stats.reserved_output_tokens == 4096
        assert stats.context_blocks_dropped > 0


# ── Nightly integration tests (real LLM) ────────────────────────────────


@pytest.mark.integration
class TestNightlyRealLLM:
    """Tests that exercise real LLM context window limits.

    Requires a real API key and network access. Run with:
        pytest -m integration tests/integration/test_l4_eviction_e2e.py
    """

    @pytest.mark.asyncio
    async def test_real_context_overflow_recovery(self):
        """Agent recovers gracefully when hitting real context window limits."""
        # This test would:
        # 1. Use a small-window model (e.g. gpt-4o-mini with 128k)
        # 2. Fill context to >90% capacity
        # 3. Verify the agent can still respond (collapse fires)
        # 4. Verify no 400/context_length_exceeded errors
        pytest.skip("Requires real API key — run in nightly CI")
