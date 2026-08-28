# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue 180 and summarizer input sizing.

`TokenBudgetSummarizer.summarize()` must accept large rendered history —
its purpose is to compress oversized history. Before this fix the default
PredictConfig.max_param_chars=200_000 rejected large histories, causing the
summarizer to silently fail with a WARNING and leaving history uncompressed
(catch-22).

Do not add a method-wide TruncationConfig override here: issue 243 showed
that strategy-level truncation re-renders unrelated call history for the
child summarizer prompt and can itself cause prompt-too-long failures.
"""

from nooa.agents import SummarizationAgent
from nooa.config.strategy_config import PredictConfig
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.predict import PredictStrategy


def test_summarize_strategy_override_present():
    """The @strategy decorator stores _strategy_override on the inner func.

    The wrapper preserves access via functools.wraps' __wrapped__.
    ``None`` means the parameter-size guard is disabled — the summarizer's
    contract is "accept arbitrarily large input."
    """
    inner = SummarizationAgent.summarize.__wrapped__
    override = inner._strategy_override
    assert isinstance(override, PredictStrategy)
    assert override.config.max_param_chars is None


def test_summarize_has_no_method_wide_truncation_override():
    """The summarizer must not override truncation for the whole child prompt.

    `max_param_chars=None` keeps the explicit `history_markdown` argument from
    being rejected before the call. A method-level TruncationConfig is broader:
    it also re-renders all context events inherited by the child agent. That
    caused issue 243's prompt-too-long failure, so the summarizer must leave
    method-wide truncation unset.
    """
    assert SummarizationAgent.summarize._strategy_truncation is None


def _summarizer_call(history_markdown: str) -> CurrentCall:
    """Build a CurrentCall mirroring SummarizationAgent.summarize's signature."""
    return CurrentCall(
        id="test-id",
        method_name="summarize",
        decorator="agent",
        signature="(self, history_markdown: str, target_chars: int)",
        args=(history_markdown, 1000),
        kwargs={},
    )


def test_assert_param_sizes_noop_when_limit_is_none():
    """``max_param_chars=None`` disables the guard entirely.

    The reproduction from issue 180: a 1 M-char history must not be rejected
    when the summarizer's PredictConfig sets ``max_param_chars=None``. With
    the default 200 K limit this raised ValueError and caused the summarizer
    to silently abort.
    """
    strategy = PredictStrategy(PredictConfig(max_param_chars=None))
    call = _summarizer_call("x" * 1_000_000)
    strategy._assert_param_sizes(call)  # must not raise


def test_assert_param_sizes_still_fires_when_limit_is_set():
    """Non-summarizer callers keep the safety guard.

    The decorator-scoped ``None`` for the summarizer must NOT change the
    behavior of ordinary PredictStrategy callers — the global default is
    still 200 K.
    """
    import pytest

    strategy = PredictStrategy(PredictConfig(max_param_chars=1000))
    call = _summarizer_call("x" * 5_000)
    with pytest.raises(ValueError, match="exceeding max_param_chars"):
        strategy._assert_param_sizes(call)


# ---------------------------------------------------------------------------
# Total-input token cap (session d2a3557e: summarizer's own summarize() call was
# handed a ~1.09M-token render and 400'd "prompt is too long" in a retry loop).
# ---------------------------------------------------------------------------
