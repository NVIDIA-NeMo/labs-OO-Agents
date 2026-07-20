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


def test_render_range_caps_total_input_to_model_budget():
    """_render_range_to_markdown bounds its TOTAL output to the summarizer model
    budget, head-dropping the oldest events with a marker — so a huge range can't
    blow past the model context window on the summarize() call."""
    from nooa import Agent
    from nooa.agents import SummarizationAgent
    from nooa.events import Message
    from nooa.unifiedllm import FakeLLMClient

    # Small, known window so the per-token budget (0.7*window) is easy to exceed.
    llm = FakeLLMClient()
    llm._context_window = 2000  # 0.7*2000 = 1400 token budget

    class A(Agent, llm=llm):
        async def chat(self, m: str) -> str:
            """Chat about {m}."""
            ...

    agent = A()
    # Many events; each renders to a few hundred chars -> total >> 1400 tokens.
    for i in range(60):
        agent.event_manager.add(Message(content=f"event {i}: " + ("lorem ipsum " * 40)))

    summarizer = SummarizationAgent(agent)
    tags = agent.event_manager.keys()
    rendered = summarizer._render_range_to_markdown(tags[0], tags[-1])

    counter = summarizer._input_token_counter()
    budget = summarizer._input_token_budget()
    assert budget == 1400
    # The rendered input fits the budget (with a small marker allowance).
    assert counter(rendered) <= budget + counter("[... older event(s) omitted ...]") + 32
    # And it announced the head-drop rather than silently truncating.
    assert "older event(s) omitted" in rendered
    # The NEWEST event survived (most relevant to a resume summary).
    assert "event 59" in rendered


def test_render_range_no_cap_when_window_unknown():
    """No model window -> no cap (don't wipe input on a misconfig; the API error
    path is the backstop)."""
    from nooa import Agent
    from nooa.agents import SummarizationAgent
    from nooa.events import Message
    from nooa.unifiedllm import FakeLLMClient

    llm = FakeLLMClient()
    llm._context_window = 0  # unknown/disabled

    class A(Agent, llm=llm):
        async def chat(self, m: str) -> str:
            """Chat about {m}."""
            ...

    agent = A()
    for i in range(5):
        agent.event_manager.add(Message(content=f"event {i}"))

    summarizer = SummarizationAgent(agent)
    assert summarizer._input_token_budget() is None
    tags = agent.event_manager.keys()
    rendered = summarizer._render_range_to_markdown(tags[0], tags[-1])
    assert "older event(s) omitted" not in rendered
    assert "event 0" in rendered and "event 4" in rendered
