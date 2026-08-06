# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Total-size bound on rendered values (#96).

``FormatConfig`` bounds rendering *per value* — ``max_string``, ``max_length``,
``max_depth``. Those limits multiply rather than compose, so respecting all three
does not bound the total: at the ``event_format`` defaults
``max_length ** max_depth`` is 200**5 leaf slots at up to 10,000 chars each.

These tests assert the total bound holds, and are written so they fail without
``max_render_chars``: every fixture stays *inside* the per-value limits, so only
a total cap can keep the output small.

**Scope of the bound.** ``max_render_chars`` bounds the rendered *output*, not
peak allocation while rendering. ``TruncatingStringIO`` discards overflow as it
arrives, but the renderer still walks the whole structure and allocates each
chunk before the sink drops it — measured, a value that renders to 72 MB
uncapped still peaks around 65 MB with the cap applied. What the cap removes is
the multi-megabyte string entering the trajectory and being re-sent every turn,
which is the recurring cost. Bounding allocation as well means aborting the
traversal once the budget is spent, which is a change inside the renderer rather
than in config, and is left as follow-up.
"""

from __future__ import annotations

import pytest

from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG, TruncationConfig
from nooa.strategies.codeact_errors import _pformat as error_pformat
from nooa.strategies.generated_code import _pformat as event_pformat


def nest(depth: int, width: int, leaf: str):
    """A structure that stays within max_depth/max_length but fans out."""
    if depth == 0:
        return leaf
    return [nest(depth - 1, width, leaf) for _ in range(width)]


@pytest.fixture
def wide_value():
    """Depth 4, width 30 — inside event_format's max_depth=5 and max_length=200."""
    ef = DEFAULT_TRUNCATION_CONFIG.event_format
    assert ef.max_depth is not None and 4 < ef.max_depth
    assert ef.max_length is not None and 30 < ef.max_length
    return nest(4, 30, "x" * 10)


class TestDefaults:
    def test_default_is_set_and_mirrors_max_stdout(self):
        tc = DEFAULT_TRUNCATION_CONFIG
        assert tc.max_render_chars == 50_000
        # Same budget as the analogous raw-text path.
        assert tc.max_render_chars == tc.capture.max_stdout

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="max_render_chars must be > 0 or None"):
            TruncationConfig(max_render_chars=0)
        with pytest.raises(ValueError, match="max_render_chars must be > 0 or None"):
            TruncationConfig(max_render_chars=-1)

    def test_none_means_unlimited(self, wide_value):
        tc = TruncationConfig(max_render_chars=None)
        assert len(event_pformat(wide_value, tc)) > 1_000_000

    def test_merge_with_carries_the_field(self):
        merged = DEFAULT_TRUNCATION_CONFIG.merge_with(TruncationConfig(max_render_chars=123))
        assert merged.max_render_chars == 123
        # Untouched fields survive the merge.
        assert merged.event_format.max_string == DEFAULT_TRUNCATION_CONFIG.event_format.max_string


class TestEventRenderIsBounded:
    def test_event_render_respects_the_total_cap(self, wide_value):
        tc = DEFAULT_TRUNCATION_CONFIG
        rendered = event_pformat(wide_value, tc)
        assert tc.max_render_chars is not None
        # TruncatingStringIO appends a truncation notice, so allow modest overhead
        # rather than asserting an exact equality that the notice would break.
        assert len(rendered) < tc.max_render_chars * 2

    def test_error_render_respects_the_total_cap(self, wide_value):
        tc = DEFAULT_TRUNCATION_CONFIG
        rendered = error_pformat(wide_value, tc)
        assert tc.max_render_chars is not None
        assert len(rendered) < tc.max_render_chars * 2

    def test_truncation_is_visible_not_silent(self, wide_value):
        """The model must be able to tell content was elided."""
        rendered = event_pformat(wide_value, DEFAULT_TRUNCATION_CONFIG)
        lowered = rendered.lower()
        assert "truncat" in lowered or "not shown" in lowered

    def test_cap_holds_as_the_structure_grows(self):
        """Output size is decided by the cap, not by the value's shape.

        Two fixtures an order of magnitude apart in leaf count must render to
        about the same size. This is the property that matters for the recurring
        cost: ``event_format`` renders every turn for the rest of the run, so an
        output that scales with the value scales the whole remaining run.
        """
        tc = DEFAULT_TRUNCATION_CONFIG
        smaller = len(event_pformat(nest(3, 30, "x" * 10), tc))
        larger = len(event_pformat(nest(4, 30, "x" * 10), tc))
        assert tc.max_render_chars is not None
        assert smaller < tc.max_render_chars * 2
        assert larger < tc.max_render_chars * 2
        # ~27k leaves vs ~810k leaves, yet the rendered sizes stay comparable.
        assert abs(larger - smaller) < tc.max_render_chars

    def test_small_values_are_untouched(self):
        """The cap must not disturb ordinary rendering."""
        tc = DEFAULT_TRUNCATION_CONFIG
        for value in ({"a": 1, "b": [1, 2, 3]}, [1, 2, 3], {"nested": {"x": "y"}}):
            rendered = event_pformat(value, tc)
            assert "truncat" not in rendered.lower()
            assert "not shown" not in rendered.lower()

    def test_strings_still_pass_through_verbatim(self):
        """truncating_pformat returns str inputs unchanged, as pformat did."""
        text = "plain string, not a structure"
        assert event_pformat(text, DEFAULT_TRUNCATION_CONFIG) == text


class TestContextBlocksStayUnlimited:
    def test_context_block_format_is_still_unbounded(self):
        """Context blocks are author-curated and documented as rendering in full.

        The cap is applied at the event/error render helpers, not to
        context_block_format, so this stays deliberately unlimited.
        """
        cbf = DEFAULT_TRUNCATION_CONFIG.context_block_format
        assert cbf.max_string is None
        assert cbf.max_length is None
        assert cbf.max_depth is None
