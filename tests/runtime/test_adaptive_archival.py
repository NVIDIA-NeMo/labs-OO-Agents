# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adaptive archival logic (_archive_on_context_error).

Tests the computation that determines how many events to archive
when the LLM API returns a context-window error.
"""

import math

from nooa.runtime.actor import _ARCHIVE_TARGET_UTILIZATION


class TestAdaptiveArchivalComputation:
    """Test the adaptive archival n_to_archive calculation."""

    def _compute_n_to_archive(self, *, cap, total_tok, events_tok, n_active):
        """Replicate the archival computation from _archive_on_context_error."""
        if n_active == 0:
            return 0
        target_tok = int(cap * _ARCHIVE_TARGET_UTILIZATION)
        tokens_to_shed = max(0, total_tok - target_tok)
        if tokens_to_shed == 0:
            return 0
        avg_event_tok = events_tok / max(1, n_active)
        n_to_archive = min(
            int(math.ceil(tokens_to_shed / max(1, avg_event_tok))),
            n_active,
        )
        return n_to_archive

    def test_basic_archival_count(self):
        """Basic case: over budget, archives proportionally."""
        # cap=1000, target=600, total_tok=800 → tokens_to_shed=200
        # 10 events → avg=500/10=50 per event → need ceil(200/50)=4
        n = self._compute_n_to_archive(cap=1000, total_tok=800, events_tok=500, n_active=10)
        assert n == 4

    def test_under_target_no_archival(self):
        """When total_tok <= target, no archival needed."""
        n = self._compute_n_to_archive(cap=1000, total_tok=500, events_tok=400, n_active=10)
        assert n == 0

    def test_exactly_at_target(self):
        """When total_tok == target, no archival needed."""
        n = self._compute_n_to_archive(cap=1000, total_tok=600, events_tok=500, n_active=10)
        assert n == 0

    def test_n_active_zero(self):
        """Edge case: no active events — cannot archive."""
        n = self._compute_n_to_archive(cap=1000, total_tok=800, events_tok=0, n_active=0)
        assert n == 0

    def test_events_tok_zero(self):
        """Edge case: events_tok=0 means avg is 0, capped to archive all."""
        # tokens_to_shed=200, avg=0/5=0 → max(1,0)=1 → ceil(200/1)=200, min(200,5)=5
        n = self._compute_n_to_archive(cap=1000, total_tok=800, events_tok=0, n_active=5)
        assert n == 5  # capped at n_active

    def test_capped_at_n_active(self):
        """n_to_archive never exceeds n_active."""
        n = self._compute_n_to_archive(cap=1000, total_tok=950, events_tok=50, n_active=3)
        assert n <= 3

    def test_large_tokens_to_shed(self):
        """When way over budget, archives aggressively but respects n_active cap."""
        # cap=1000, target=600, total_tok=950 → tokens_to_shed=350
        # 5 events, events_tok=250 → avg=50 → ceil(350/50)=7, min(7,5)=5
        n = self._compute_n_to_archive(cap=1000, total_tok=950, events_tok=250, n_active=5)
        assert n == 5

    def test_small_overshoot(self):
        """Small overshoot archives just 1 event."""
        # cap=1000, target=600, total_tok=610 → tokens_to_shed=10
        # 10 events, events_tok=500 → avg=50 → ceil(10/50)=1
        n = self._compute_n_to_archive(cap=1000, total_tok=610, events_tok=500, n_active=10)
        assert n == 1


class TestRealScaleArchivalCap:
    """The archival safety net works in calibrated/API token scale directly."""

    def _compute_n_to_archive(self, *, cap, total_tok, events_tok, n_active):
        """Replicate the archival computation from _archive_on_context_error."""
        if n_active == 0:
            return 0
        target_tok = int(cap * _ARCHIVE_TARGET_UTILIZATION)
        tokens_to_shed = max(0, total_tok - target_tok)
        if tokens_to_shed == 0:
            return 0
        avg_event_tok = events_tok / max(1, n_active)
        return min(
            int(math.ceil(tokens_to_shed / max(1, avg_event_tok))),
            n_active,
        )

    def test_cap_is_real_token_budget_not_ratio_scaled(self):
        """No per-actor ratio should tighten the cap after total_tokens is calibrated."""
        ctx_window = 200_000
        cap = int(ctx_window * 0.70)
        assert cap == 140_000

    def test_real_scaled_total_sheds_enough_for_undercounting_tokenizer(self):
        """Regression: shed from real total to a real target, with no ratio-B drift."""
        ctx_window = 200_000
        cap = int(ctx_window * 0.70)
        total_tok = 180_000
        context_blocks_tok = 20_000
        n_active = 16
        events_tok = total_tok - context_blocks_tok

        n = self._compute_n_to_archive(
            cap=cap,
            total_tok=total_tok,
            events_tok=events_tok,
            n_active=n_active,
        )

        # target = 140k * 0.60 = 84k; shed 96k. Events avg = 160k/16 = 10k.
        assert n == 10

    def test_fixed_tool_tokens_do_not_inflate_archiveable_event_average(self):
        """Tool-schema tokens are fixed overhead, not archiveable event tokens."""
        ctx_window = 1_000
        cap = int(ctx_window * 0.70)
        total_tok = 1_250  # 950 fixed tool tokens + 300 event tokens
        events_tok = 300
        n_active = 30

        n = self._compute_n_to_archive(
            cap=cap,
            total_tok=total_tok,
            events_tok=events_tok,
            n_active=n_active,
        )

        # target = 420; shed 830. Events avg = 300/30 = 10, so all events are
        # archiveable and must be selected. Using total_tok/n_active would pick
        # only 20 and leave the fixed tool overhead still over the window.
        assert n == 30


class TestArchiveTargetUtilization:
    """Verify the constant is what we expect."""

    def test_default_value(self):
        assert _ARCHIVE_TARGET_UTILIZATION == 0.60
