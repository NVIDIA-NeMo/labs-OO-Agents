# SPDX-License-Identifier: Apache-2.0
"""Context rendering records structural facts and exposes no token estimates."""

import pytest
from pydantic import ValidationError

from nooa.context_blocks.formatter import OpenAIProviderFormatter, XMLBlockFormatter
from nooa.context_blocks.models import ContextWindowStats, ResolvedBlock, Role
from nooa.context_blocks.renderer import RenderResult, render_context


def _render(blocks):
    return render_context(
        blocks, block_formatter=XMLBlockFormatter(), provider_formatter=OpenAIProviderFormatter()
    )


def test_render_returns_structural_character_counts_only():
    result = _render(
        [
            ResolvedBlock(key="system", content="abc"),
            ResolvedBlock(key="event", content="hello", role=Role.USER),
        ]
    )
    assert isinstance(result, RenderResult)
    assert result.stats == ContextWindowStats(
        context_blocks_count=1, events_count=1, context_blocks_chars=3, events_chars=5
    )


def test_context_stats_format_is_structural():
    stats = ContextWindowStats(
        context_blocks_count=2, events_count=3, context_blocks_chars=12, events_chars=34
    )
    assert stats.format() == "Context contains 2 blocks (12 chars) and 3 events (34 chars)."


@pytest.mark.parametrize(
    "removed",
    [
        "prompt_tokens",
        "total_tokens",
        "context_blocks_tokens",
        "events_tokens",
        "max_event_tokens",
        "max_context_tokens",
        "model_context_window",
        "reserved_output_tokens",
    ],
)
def test_removed_token_fields_are_rejected(removed):
    with pytest.raises(ValidationError):
        ContextWindowStats(context_blocks_count=0, events_count=0, **{removed: 1})
