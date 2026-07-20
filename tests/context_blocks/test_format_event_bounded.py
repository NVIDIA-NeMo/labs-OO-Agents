# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests verifying format_event renders events safely.

Block-level head/tail string truncation has been removed. ``format_event``
uses structural bounds from ``event_format`` only; it must not rely on
``max_chars`` / ``TruncatingStringIO`` for event internals.
"""

from nooa.context_blocks.events import EventBase
from nooa.context_blocks.formatter import XMLBlockFormatter


class BigValueEvent(EventBase):
    """Minimal EventBase subclass carrying an arbitrary Python value."""

    value: object

    model_config = {"arbitrary_types_allowed": True}


class TestFormatEventBounded:
    def setup_method(self):
        self.fmt = XMLBlockFormatter()

    def test_small_event_unchanged(self):
        """A normal event is formatted without a truncation notice."""
        event = BigValueEvent(value=[1, 2, 3])
        result = self.fmt.format_event(event)
        assert "1" in result
        assert "truncation" not in result.lower()

    def test_large_list_value_bounded_with_event_format(self):
        """Large structured values are bounded by event_format structural knobs."""
        from nooa.config.truncation_config import FormatConfig

        event = BigValueEvent(value=list(range(1_000_000)))
        result = self.fmt.format_event(
            event,
            event_format=FormatConfig(max_string=500, max_length=50, max_depth=4),
        )
        assert "list(len=1000000" in result
        assert len(result) < 5_000

    def test_large_string_value_uses_marker_family(self):
        """A string field NESTED inside a structured event gets the marker
        family treatment when ``event_format`` provides a ``max_string`` bound.
        Without ``event_format``, hardcoded fallbacks are gone — the caller is
        expected to pass an explicit FormatConfig."""
        from nooa.config.truncation_config import FormatConfig

        event = BigValueEvent(value="x" * 2_000_000)
        result = self.fmt.format_event(
            event,
            event_format=FormatConfig(max_string=500, max_length=50, max_depth=4),
        )
        # Marker family kicks in for the nested string field
        assert "str(len=2000000" in result
        # And the result is bounded well below the original 2 MB
        assert len(result) < 2000

    def test_default_no_cap(self):
        """Default ``format_event`` call has no max_chars; non-string content
        renders fully (per-value bounds come from spec() / cfg.events.* upstream)."""
        event = BigValueEvent(value=[1, 2, 3, 4, 5])
        result = self.fmt.format_event(event)
        for n in (1, 2, 3, 4, 5):
            assert str(n) in result
