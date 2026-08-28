# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for summarization agents.

Tests the SummarizationAgent base class and its implementations:
- TokenBudgetSummarizer
- MethodSummarizer
"""

import pytest

from nooa import Agent
from nooa.agents import MethodSummarizer, SummarizationAgent
from nooa.config.summarizer_config import MethodSummarizerConfig
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.events import AfterTurn, Message
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_llm():
    """Create a fake LLM for testing."""
    return FakeLLMClient()


@pytest.fixture
def test_agent(fake_llm):
    """Create a simple test agent for summarizer attachment."""

    class SimpleAgent(Agent, llm=fake_llm):
        pass

    return SimpleAgent()


# =============================================================================
# SummarizationAgent Base Class Tests
# =============================================================================


class TestSummarizationAgentBase:
    """Tests for SummarizationAgent base class."""

    def test_init_attaches_to_agent(self, test_agent):
        """Summarizer attaches to agent's history and inherits LLM."""
        summarizer = SummarizationAgent(test_agent)

        assert summarizer.target_event_manager is test_agent.event_manager
        assert summarizer._llm is test_agent._llm
        assert summarizer._pending_task is None
        assert summarizer._pending_summary is None

    def test_init_subscribes_to_events(self, test_agent):
        """Summarizer subscribes to before_turn and after_turn on init."""
        summarizer = SummarizationAgent(test_agent)

        # Verify subscriptions exist
        assert summarizer._unsub_before is not None
        assert summarizer._unsub_after is not None

    def test_uninstall_clears_subscriptions(self, test_agent):
        """_uninstall() clears subscriptions and cancels pending tasks."""
        summarizer = SummarizationAgent(test_agent)
        summarizer._uninstall()

        assert summarizer._unsub_before is None
        assert summarizer._unsub_after is None

    def test_default_should_summarize_returns_false(self, test_agent):
        """Base class _should_summarize returns False."""
        summarizer = SummarizationAgent(test_agent)
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        assert summarizer._should_summarize(event) is False

    def test_default_compute_range_returns_none(self, test_agent):
        """Base class _compute_range returns None."""
        summarizer = SummarizationAgent(test_agent)
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        assert summarizer._compute_range(event) is None

    def test_get_events_in_range(self, test_agent):
        """_get_events_in_range returns events within range."""
        # Add some events to agent's history
        test_agent.event_manager.add(Message(content="Message 1"))
        test_agent.event_manager.add(Message(content="Message 2"))
        test_agent.event_manager.add(Message(content="Message 3"))

        summarizer = SummarizationAgent(test_agent)
        events = summarizer._get_events_in_range("1", "3")

        # Should return 3 events
        assert len(events) == 3
        assert events[0][0] == "1"
        assert events[-1][0] == "3"

    def test_render_range_to_markdown_uses_parent_event_format(self, fake_llm):
        """Summarizer source markdown preserves the parent agent's event bounds."""

        class BoundedAgent(
            Agent,
            llm=fake_llm,
            truncation=TruncationConfig(
                event_format=FormatConfig(max_string=25, max_length=10, max_depth=5)
            ),
        ):
            pass

        agent = BoundedAgent()
        agent.event_manager.add(Message(content="x" * 200))
        summarizer = SummarizationAgent(agent)

        rendered = summarizer._render_range_to_markdown("1", "1")

        assert "str(len=200" in rendered
        assert "x" * 100 not in rendered

    def test_summarize_has_no_method_wide_truncation_override(self):
        """A large history parameter must not unbound-render unrelated context events."""
        method = getattr(SummarizationAgent.summarize, "__func__", SummarizationAgent.summarize)
        assert getattr(method, "_strategy_truncation", None) is None


# =============================================================================
# TokenBudgetSummarizer Tests
# =============================================================================


# =============================================================================
# MethodSummarizer Tests
# =============================================================================


class TestMethodSummarizer:
    """Tests for MethodSummarizer."""

    def test_default_config(self, test_agent):
        """Default configuration values."""
        summarizer = MethodSummarizer(test_agent)
        assert summarizer.config.min_events == 3
        assert summarizer.config.exclude_root is True

    def test_custom_config(self, test_agent):
        """Custom configuration via config object."""
        summarizer = MethodSummarizer(
            test_agent, config=MethodSummarizerConfig(min_events=5, exclude_root=False)
        )
        assert summarizer.config.min_events == 5
        assert summarizer.config.exclude_root is False

    def test_should_summarize_on_final(self, test_agent):
        """Should summarize when is_final=True (non-root)."""
        summarizer = MethodSummarizer(test_agent)

        # Non-root call (turn_number > 1 or not final earlier)
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id="parent-gen",
            turn_number=2,
            is_final=True,
            success=True,
        )
        assert summarizer._should_summarize(event) is True

    def test_should_not_summarize_non_final(self, test_agent):
        """Should not summarize when is_final=False."""
        summarizer = MethodSummarizer(test_agent)

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=False,
            success=True,
        )
        assert summarizer._should_summarize(event) is False

    def test_should_not_summarize_root_by_default(self, test_agent):
        """Should not summarize root calls by default (exclude_root=True)."""
        summarizer = MethodSummarizer(test_agent)

        # Root call: turn_number=1 and is_final=True
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        assert summarizer._should_summarize(event) is False

    def test_should_summarize_root_when_allowed(self, test_agent):
        """Should summarize root calls when exclude_root=False."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(exclude_root=False))

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        assert summarizer._should_summarize(event) is True

    def test_compute_range_returns_none_when_too_few_events(self, test_agent):
        """Returns None when fewer events match the call_id than min_events."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=10))

        msg = Message(content="Message 1")
        msg.metadata["call_id"] = "call-abc"
        test_agent.event_manager.add(msg)

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        event.metadata["call_id"] = "call-abc"
        result = summarizer._compute_range(event)
        assert result is None

    def test_compute_range_returns_none_when_no_call_id(self, test_agent):
        """Returns None when AfterTurn has no call_id in metadata."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=1))

        test_agent.event_manager.add(Message(content="Message 1"))

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        # No call_id in metadata
        result = summarizer._compute_range(event)
        assert result is None


class TestMethodSummarizerComputeRangeScenarios:
    """Scenario-based tests for MethodSummarizer._compute_range.

    These tests verify that call_id based range computation correctly handles
    the key scenarios: simple calls, nested calls, and interleaved events.
    """

    def _after_turn(self, call_id: str) -> AfterTurn:
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        event.metadata["call_id"] = call_id
        return event

    def _msg(self, content: str, call_id: str) -> Message:
        msg = Message(content=content)
        msg.metadata["call_id"] = call_id
        return msg

    def test_simple_single_method_call(self, test_agent):
        """Scenario: agent.analyze("data") — 3 turns, all same call_id.

        call_id=C1
          Turn 1: Task event, LLM output, execution result
          Turn 2: LLM output, execution result
          Turn 3: return_result
        """
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=2))
        em = test_agent.event_manager

        em.add(self._msg("Task: analyze data", "C1"))  # tag=1
        em.add(self._msg("LLM turn 1 output", "C1"))  # tag=2
        em.add(self._msg("Execution result 1", "C1"))  # tag=3
        em.add(self._msg("LLM turn 2 output", "C1"))  # tag=4
        em.add(self._msg("Final result", "C1"))  # tag=5

        result = summarizer._compute_range(self._after_turn("C1"))
        assert result == ("1", "5")

    def test_nested_method_call_included_in_range(self, test_agent):
        """Scenario: report() calls analyze() internally.

        call_id=C1 (report)
          Event: "starting report"
          call_id=C2 (analyze, called by LLM code)
            Event: "analyzing data"
            Event: "analysis result"
          Event: "report complete"
        """
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=2))
        em = test_agent.event_manager

        em.add(self._msg("starting report", "C1"))  # tag=1
        em.add(self._msg("analyzing data", "C2"))  # tag=2 (child)
        em.add(self._msg("analysis result", "C2"))  # tag=3 (child)
        em.add(self._msg("report complete", "C1"))  # tag=4

        result = summarizer._compute_range(self._after_turn("C1"))
        assert result == ("1", "4")
        # Tags 2 and 3 (child events) are inside this range

    def test_child_call_has_own_range(self, test_agent):
        """The child's call_id can also be summarized independently."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=2))
        em = test_agent.event_manager

        em.add(self._msg("parent start", "C1"))  # tag=1
        em.add(self._msg("child work 1", "C2"))  # tag=2
        em.add(self._msg("child work 2", "C2"))  # tag=3
        em.add(self._msg("parent end", "C1"))  # tag=4

        # Child range
        result = summarizer._compute_range(self._after_turn("C2"))
        assert result == ("2", "3")

    def test_multiple_nested_children(self, test_agent):
        """Scenario: method calls two different sub-methods.

        call_id=C1 (orchestrator)
          Event: "start"
          call_id=C2 (first tool call)
            Event: "tool 1 result"
          call_id=C3 (second tool call)
            Event: "tool 2 result"
          Event: "done"
        """
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=2))
        em = test_agent.event_manager

        em.add(self._msg("start", "C1"))  # tag=1
        em.add(self._msg("tool 1 result", "C2"))  # tag=2
        em.add(self._msg("tool 2 result", "C3"))  # tag=3
        em.add(self._msg("done", "C1"))  # tag=4

        result = summarizer._compute_range(self._after_turn("C1"))
        assert result == ("1", "4")
        # Includes C2 and C3 events by chronological position

    def test_unrelated_events_before_and_after(self, test_agent):
        """Events from other call_ids outside our range are excluded."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=2))
        em = test_agent.event_manager

        em.add(self._msg("unrelated before", "C0"))  # tag=1
        em.add(self._msg("our start", "C1"))  # tag=2
        em.add(self._msg("our end", "C1"))  # tag=3
        em.add(self._msg("unrelated after", "C0"))  # tag=4

        result = summarizer._compute_range(self._after_turn("C1"))
        assert result == ("2", "3")
        # Tags 1 and 4 are outside the range

    def test_events_without_call_id_not_matched(self, test_agent):
        """Events without call_id metadata don't count toward min_events."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=3))
        em = test_agent.event_manager

        em.add(Message(content="no call_id"))  # tag=1
        em.add(self._msg("has call_id 1", "C1"))  # tag=2
        em.add(Message(content="no call_id again"))  # tag=3
        em.add(self._msg("has call_id 2", "C1"))  # tag=4

        # Only 2 events match C1 (tags 2, 4), but min_events=3
        result = summarizer._compute_range(self._after_turn("C1"))
        assert result is None

    def test_deeply_nested_calls(self, test_agent):
        """Three levels deep: C1 → C2 → C3."""
        summarizer = MethodSummarizer(test_agent, config=MethodSummarizerConfig(min_events=2))
        em = test_agent.event_manager

        em.add(self._msg("level 1 start", "C1"))  # tag=1
        em.add(self._msg("level 2 start", "C2"))  # tag=2
        em.add(self._msg("level 3 work", "C3"))  # tag=3
        em.add(self._msg("level 2 end", "C2"))  # tag=4
        em.add(self._msg("level 1 end", "C1"))  # tag=5

        # Top level includes everything
        assert summarizer._compute_range(self._after_turn("C1")) == ("1", "5")
        # Mid level includes its child
        assert summarizer._compute_range(self._after_turn("C2")) == ("2", "4")
        # Leaf level just its own events
        result = summarizer._compute_range(self._after_turn("C3"))
        assert result is None  # Only 1 event, min_events=2


# =============================================================================
# Agent Integration Tests
# =============================================================================


class TestAgentSummarizerIntegration:
    """Tests for Agent + Summarizer integration."""

    def test_agent_standalone_without_summarizer(self, fake_llm):
        """Agent works fine without summarizer - they're decoupled."""

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # No _summarizers attribute on agent - they're decoupled
        assert not hasattr(agent, "_summarizers")

    def test_method_summarizer_install_with_config(self, test_agent):
        """MethodSummarizer.install() accepts config= keyword argument."""
        s = MethodSummarizer.install(test_agent, config=MethodSummarizerConfig(min_events=5))
        assert s.config.min_events == 5

    def test_method_summarizer_install_rejects_flat_kwargs(self, test_agent):
        """MethodSummarizer.install() raises TypeError on flat config kwargs."""
        with pytest.raises(TypeError):
            MethodSummarizer.install(test_agent, min_events=5)


# =============================================================================
# Async Integration Tests
# =============================================================================


# =============================================================================
# Tracing opt-out — issue #192
# =============================================================================
class TestSummarizerNoTrace:
    """Regression test: @hidden summarizer helpers must opt out of tracing.

    Private helpers fire on every turn and used to drown out useful spans.
    The fix decorates them with @no_trace so only ``summarize()`` produces a span.
    """

    @pytest.mark.parametrize("cls", [SummarizationAgent, MethodSummarizer])
    def test_hidden_helpers_have_no_trace(self, cls):
        """Every @hidden method on a summarizer class must also be @no_trace."""
        import inspect

        from nooa.agentdoc._visibility import is_hidden_method

        hidden_methods = [
            (name, attr)
            for name, attr in cls.__dict__.items()
            if inspect.isfunction(attr) and is_hidden_method(attr)
        ]
        assert hidden_methods, f"No @hidden methods found on {cls.__name__} — test stale?"

        for name, attr in hidden_methods:
            # @no_trace causes the metaclass to skip wrapping entirely, so
            # cls.__dict__[name] IS the original function (no _original
            # indirection).
            assert getattr(attr, "_no_trace", False) is True, (
                f"{cls.__name__}.{name} is @hidden but missing @no_trace — "
                "private helpers should not produce trace spans (issue #192)"
            )

    def test_summarize_remains_traced(self):
        """The generation method ``summarize()`` must still produce a span."""
        summarize = SummarizationAgent.summarize
        # summarize() goes through @strategy, which wraps; the original is on _original.
        original = getattr(summarize, "_original", summarize)
        assert getattr(original, "_no_trace", False) is False, (
            "summarize() must not be marked @no_trace — it is the only summarizer "
            "method that carries real LLM-call signal"
        )
        # Runtime flag on the wrapper itself (mutable list, set by @strategy/no_trace machinery).
        tracing_enabled = getattr(summarize, "_tracing_enabled", None)
        if tracing_enabled is not None:
            assert tracing_enabled[0] is True, "summarize() wrapper has tracing disabled"
