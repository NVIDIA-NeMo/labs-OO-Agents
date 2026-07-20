# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ContextWindowStats.

Token usage is provider-reported only: ``render_context()`` records
structural facts (block/event counts and raw character sizes) and leaves
``prompt_tokens=None``. The runtime writes the authoritative ``prompt_tokens``
back after a successful call; the per-category breakdown is then attributed
from that total by character share.
"""

from nooa.context_blocks.formatter import OpenAIProviderFormatter, XMLBlockFormatter
from nooa.context_blocks.models import (
    BlockMetadata,
    ContextWindowStats,
    ResolvedBlock,
    Role,
)
from nooa.context_blocks.renderer import RenderResult, render_context


class TestContextWindowStatsBasic:
    """Basic stats computation tests."""

    def test_stats_returned_as_render_result(self):
        """render_context() returns a RenderResult with output and stats."""
        result = render_context(
            [],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        )
        assert isinstance(result, RenderResult)
        assert isinstance(result.stats, ContextWindowStats)

    def test_render_leaves_prompt_tokens_none(self):
        """render_context never estimates tokens — prompt_tokens stays None."""
        blocks = [ResolvedBlock(key="a", content="AAA")]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).stats
        assert stats.prompt_tokens is None
        assert stats.total_tokens is None
        assert stats.context_blocks_tokens is None
        assert stats.events_tokens is None

    def test_context_blocks_counted(self):
        """System blocks counted in context_blocks_count; chars recorded."""
        blocks = [
            ResolvedBlock(key="a", content="AAA"),
            ResolvedBlock(key="b", content="BBB"),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).stats
        assert stats.context_blocks_count == 2
        assert stats.context_blocks_chars == 6  # len("AAA") + len("BBB")

    def test_events_counted(self):
        """Message blocks counted in events_count; chars recorded."""
        blocks = [
            ResolvedBlock(key="e1", content="hello", role=Role.USER),
            ResolvedBlock(key="e2", content="world", role=Role.ASSISTANT),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).stats
        assert stats.events_count == 2
        assert stats.events_chars == 10  # len("hello") + len("world")

    def test_empty_blocks(self):
        """Stats for empty input are all zero / None."""
        stats = render_context(
            [],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).stats
        assert stats.context_blocks_count == 0
        assert stats.context_blocks_chars == 0
        assert stats.events_count == 0
        assert stats.events_chars == 0
        assert stats.total_tokens is None


class TestProviderTokenAttribution:
    """Once prompt_tokens is known, the breakdown is attributed by char share."""

    def test_total_tokens_is_provider_value(self):
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            context_blocks_chars=600,
            events_chars=400,
            prompt_tokens=5_000,
        )
        assert stats.total_tokens == 5_000

    def test_breakdown_sums_to_total(self):
        stats = ContextWindowStats(
            context_blocks_count=2,
            events_count=3,
            context_blocks_chars=600,
            events_chars=400,
            prompt_tokens=5_000,
        )
        # 600 / 1000 of 5000 = 3000 context, remainder 2000 events
        assert stats.context_blocks_tokens == 3_000
        assert stats.events_tokens == 2_000
        assert stats.context_blocks_tokens + stats.events_tokens == stats.total_tokens

    def test_breakdown_zero_chars(self):
        """No characters anywhere → context attributed 0, events get the rest."""
        stats = ContextWindowStats(
            context_blocks_count=0,
            events_count=0,
            context_blocks_chars=0,
            events_chars=0,
            prompt_tokens=42,
        )
        assert stats.context_blocks_tokens == 0
        assert stats.events_tokens == 42

    def test_breakdown_rounds_and_sums_exactly_on_fractional_split(self):
        """A non-clean char split still sums exactly to prompt_tokens —
        events is the remainder, so rounding can never lose/gain a token."""
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            context_blocks_chars=333,
            events_chars=667,
            prompt_tokens=100,
        )
        # 333/1000 * 100 = 33.3 → 33; events = 100 - 33 = 67
        assert stats.context_blocks_tokens == 33
        assert stats.events_tokens == 67
        assert stats.context_blocks_tokens + stats.events_tokens == 100

    def test_removed_token_kwargs_are_rejected(self):
        """extra='forbid': a stale caller passing the removed estimate fields
        fails loudly instead of silently yielding None."""
        import pytest
        from pydantic import ValidationError

        for bad_kwarg in (
            "total_tokens",
            "context_blocks_tokens",
            "events_tokens",
            "max_event_tokens",
        ):
            with pytest.raises(ValidationError):
                ContextWindowStats(
                    context_blocks_count=1,
                    events_count=1,
                    **{bad_kwarg: 123},
                )

    def test_overall_utilization(self):
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            context_blocks_chars=1,
            events_chars=1,
            prompt_tokens=50_000,
            model_context_window=200_000,
        )
        assert stats.overall_utilization == 0.25

    def test_effective_window_subtracts_output_reserve(self):
        """The usable window is the model window minus the output reserve."""
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            prompt_tokens=50_000,
            model_context_window=200_000,
            reserved_output_tokens=64_000,
        )
        assert stats.effective_window == 136_000
        assert stats.overall_utilization == 50_000 / 136_000

    def test_effective_window_without_reserve_is_raw_window(self):
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            prompt_tokens=50_000,
            model_context_window=200_000,
        )
        assert stats.effective_window == 200_000

    def test_utilization_can_exceed_one_when_prompt_eats_the_reserve(self):
        """prompt 150k with 64k reserved out of a 200k window: the next call
        at full max_tokens would be rejected — utilization reads >100%."""
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            prompt_tokens=150_000,
            model_context_window=200_000,
            reserved_output_tokens=64_000,
        )
        assert stats.overall_utilization is not None
        assert stats.overall_utilization > 1.0

    def test_overall_utilization_none_before_response(self):
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            model_context_window=200_000,
        )
        assert stats.overall_utilization is None


class TestContextWindowStatsTruncation:
    """Tests for dropped block/event tracking (eviction still uses count_fn)."""

    def test_context_blocks_dropped_on_truncation(self):
        blocks = [
            ResolvedBlock(key="small", content="#" * 100),
            ResolvedBlock(key="large", content="$" * 5000),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=1000,
            count_tokens=len,
        ).stats
        assert stats.context_blocks_dropped == 1
        # "small" (100) survives; "large" (5000) was evicted to a short label
        assert stats.context_blocks_chars < 1000

    def test_context_blocks_dropped_multiple(self):
        blocks = [
            ResolvedBlock(key="tiny", content="#" * 50),
            ResolvedBlock(key="medium", content="$" * 3000),
            ResolvedBlock(key="huge", content="@" * 8000),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=500,
            count_tokens=len,
        ).stats
        assert stats.context_blocks_dropped == 2

    def test_no_drops_when_under_budget(self):
        blocks = [
            ResolvedBlock(key="a", content="hello"),
            ResolvedBlock(key="b", content="world"),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=10000,
            count_tokens=len,
        ).stats
        assert stats.context_blocks_dropped == 0
        assert stats.events_dropped == 0

    def test_all_context_blocks_dropped(self):
        blocks = [
            ResolvedBlock(key="a", content="x" * 5000),
            ResolvedBlock(key="b", content="y" * 5000),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=1,
            count_tokens=len,
        ).stats
        assert stats.context_blocks_dropped == 2
        # Blocks are retained in-place and labeled EVICTED
        assert stats.context_blocks_count == 2

    def test_context_blocks_dropped_with_user_blocks(self):
        """User blocks (from self.context) are dropped first."""
        blocks = [
            ResolvedBlock(key="framework", content="x" * 100),
            ResolvedBlock(
                key="user_data",
                content="y" * 5000,
                metadata=BlockMetadata(user_block=True),
            ),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=500,
            count_tokens=len,
        ).stats
        assert stats.context_blocks_dropped == 1


class TestContextWindowStatsToolCallEvents:
    """ToolCallEvent blocks have content="" and contribute 0 chars."""

    def test_tool_call_event_counted_in_events_count_but_zero_chars(self):
        from nooa.context_blocks.events import ToolCallEvent, ToolResult

        event = ToolCallEvent(
            tool_call_id="tc_1",
            name="execute_python",
            arguments={"code": "1+1"},
            result=ToolResult(tool_call_id="tc_1", content="2"),
        )
        blocks = [
            ResolvedBlock(key="msg", content="hello", role=Role.USER),
            ResolvedBlock(key="tc", content="", role=Role.ASSISTANT, event=event),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).stats
        assert stats.events_count == 2
        assert stats.events_chars == len("hello")  # ToolCallEvent contributes 0


class TestContextWindowStatsEdgeCases:
    """Edge cases and regression tests."""

    def test_user_block_named_truncation_notice_no_false_positive(self):
        blocks = [
            ResolvedBlock(key="a", content="hello"),
            ResolvedBlock(key="truncation_notice", content="user data"),
        ]
        stats = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=10000,
            count_tokens=len,
        ).stats
        assert stats.context_blocks_dropped == 0
        assert stats.context_blocks_count == 2

    def test_render_result_destructuring(self):
        from nooa.context_blocks.models import RenderedMessage

        output, stats, messages = render_context(
            [ResolvedBlock(key="sys", content="hello")],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        )
        assert isinstance(output, list)
        assert isinstance(stats, ContextWindowStats)
        assert all(isinstance(m, RenderedMessage) for m in messages)

    def test_exports_from_context_blocks_package(self):
        from nooa.context_blocks import ContextWindowStats as CWS
        from nooa.context_blocks import RenderResult as RR

        assert CWS is ContextWindowStats
        assert RR is RenderResult

    def test_exports_from_nooa_package(self):
        from nooa import ContextWindowStats as CWS

        assert CWS is ContextWindowStats


class TestContextWindowStatsFormat:
    """Tests for the format() context block output."""

    def test_format_awaiting_first_response(self):
        """Before the first provider response, format() says so — no numbers."""
        stats = ContextWindowStats(
            context_blocks_count=3,
            events_count=8,
            context_blocks_chars=2000,
            events_chars=800,
        )
        text = stats.format()
        assert "awaiting first model response" in text
        assert "%" not in text
        assert "self.events.collapse(start_tag, end_tag, summary_text=...)" in text

    def test_format_no_window(self):
        """With provider tokens but no model window, shows the total only."""
        stats = ContextWindowStats(
            context_blocks_count=3,
            events_count=8,
            context_blocks_chars=500,
            events_chars=200,
            prompt_tokens=700,
        )
        text = stats.format()
        assert "Context usage: 700 tokens [provider-reported]" in text
        assert "3 blocks" in text
        assert "8 events" in text
        assert "%" not in text  # no window → no percentage

    def test_format_model_context_window(self):
        """With a model window, the header shows the percentage."""
        stats = ContextWindowStats(
            context_blocks_count=6,
            events_count=18,
            context_blocks_chars=8_200,
            events_chars=4_250,
            prompt_tokens=12_450,
            model_context_window=200_000,
        )
        text = stats.format()
        assert "Context usage: 12,450 / 200,000 tokens (6.2%) [provider-reported]" in text
        # Breakdown is attributed (prefixed ~), only the header carries a %
        assert "Context blocks: ~" in text
        assert "Events:" in text
        assert text.count("%") == 1

    def test_format_warning_and_guidance_when_hot(self):
        """Above 80% utilization the warning appears; guidance is always present."""
        stats = ContextWindowStats(
            context_blocks_count=5,
            events_count=10,
            context_blocks_chars=29_000,
            events_chars=1_000,
            prompt_tokens=185_000,
            model_context_window=200_000,
        )
        text = stats.format()
        assert "Context is nearly full" in text
        assert "self.events.collapse(start_tag, end_tag, summary_text=...)" in text

    def test_format_shows_output_reserve_and_usable_denominator(self):
        """With a reserve, the header shows the usable window and the reserve."""
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            context_blocks_chars=100,
            events_chars=100,
            prompt_tokens=12_450,
            model_context_window=200_000,
            reserved_output_tokens=64_000,
        )
        text = stats.format()
        assert "12,450 / 136,000 usable tokens" in text
        assert "64,000 of the 200,000-token window reserved for output" in text
        # pct is against the usable window: 12450/136000 = 9.2%
        assert "(9.2%)" in text

    def test_format_warning_uses_usable_window(self):
        """The nearly-full warning fires at >80% of the USABLE window even when
        the raw-window fraction is well under 80%."""
        stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=1,
            context_blocks_chars=100,
            events_chars=100,
            prompt_tokens=120_000,  # 60% of raw 200k; 88% of usable 136k
            model_context_window=200_000,
            reserved_output_tokens=64_000,
        )
        text = stats.format()
        assert "Context is nearly full" in text
        assert "doc(self.events)" in text
        assert "self.context" in text
        assert "ContextApi" in text

    def test_format_cleanup_guidance_when_dropped(self):
        """Cleanup guidance remains visible when blocks or events were dropped."""
        stats = ContextWindowStats(
            context_blocks_count=2,
            events_count=5,
            context_blocks_chars=500,
            events_chars=200,
            prompt_tokens=700,
            model_context_window=200_000,
            context_blocks_dropped=1,
        )
        text = stats.format()
        assert "1 EVICTED" in text
        assert "self.events.collapse(start_tag, end_tag, summary_text=...)" in text

    def test_format_dropped_counts_shown(self):
        """Dropped counts appear in the breakdown lines."""
        stats = ContextWindowStats(
            context_blocks_count=2,
            events_count=5,
            context_blocks_chars=500,
            events_chars=200,
            prompt_tokens=700,
            context_blocks_dropped=3,
            events_dropped=7,
        )
        text = stats.format()
        assert "3 EVICTED" in text
        assert "7 dropped" in text


class TestContextWindowStatsFrozen:
    """ContextWindowStats is immutable."""

    def test_stats_are_frozen(self):
        """Cannot mutate stats after creation."""
        import pytest
        from pydantic import ValidationError

        stats = render_context(
            [],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).stats
        with pytest.raises(ValidationError):
            stats.prompt_tokens = 999
