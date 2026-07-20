# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scorers operating without trace data.

Verifies that ModeSelectionScorer and LLMMethodologyScorer return score=0.0
when no trace_file is available.
"""

from unittest.mock import MagicMock

import pytest

from eval_pipeline.models import ScoringContext
from eval_pipeline.scoring import (
    LLMMethodologyScorer,
    ModeSelectionScorer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(trace=None, session_id=None, **kwargs) -> ScoringContext:
    return ScoringContext(
        task_id="t1",
        input=((), {}),
        expected="ok",
        actual="ok",
        trace=trace,
        session_id=session_id,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ModeSelectionScorer: no data → score 0.0
# ---------------------------------------------------------------------------


class TestModeSelectionScorerWithoutTrace:
    def test_no_trace_returns_zero(self):
        """No trace file → score 0.0 (nothing to evaluate)."""
        scorer = ModeSelectionScorer(expected="code")
        ctx = _ctx(trace=None)
        result = scorer.score(ctx)
        assert result.score == 0.0

    def test_no_trace_internal_returns_zero(self):
        """No data → 0.0 regardless of expected mode."""
        scorer = ModeSelectionScorer(expected="internal")
        ctx = _ctx(trace=None)
        result = scorer.score(ctx)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# LLMMethodologyScorer: no data → score 0.0
# ---------------------------------------------------------------------------


class TestLLMMethodologyScorerWithoutTrace:
    def _make_scorer(self) -> LLMMethodologyScorer:
        """Create a scorer with a mocked LLM client."""
        mock_llm = MagicMock()
        scorer = LLMMethodologyScorer.__new__(LLMMethodologyScorer)
        scorer._llm = mock_llm
        scorer._rubric = "Did the agent execute real code?\n\nExecutions:\n{executions}"
        scorer._skip_prefill = False
        return scorer

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_trace(self):
        """Should return score=0.0 when there is nothing to evaluate."""
        scorer = self._make_scorer()
        ctx = _ctx(trace=None)
        result = await scorer.score(ctx)
        assert result.score == 0.0
