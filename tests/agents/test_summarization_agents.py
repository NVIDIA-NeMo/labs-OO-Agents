# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for summarization agents.

Tests the SummarizationAgent base class and its implementations:
- TokenBudgetSummarizer
- MethodSummarizer
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nooa import Agent
from nooa.agents import MethodSummarizer, SummarizationAgent, TokenBudgetSummarizer
from nooa.config.summarizer_config import MethodSummarizerConfig, TokenBudgetConfig
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.events import AfterTurn, Message
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
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


class TestTokenBudgetSummarizer:
    """Tests for TokenBudgetSummarizer."""

    def test_default_config(self, test_agent):
        """Default configuration values."""
        summarizer = TokenBudgetSummarizer(test_agent)
        assert summarizer.config.max_tokens == 100_000
        assert summarizer.config.preserve_recent == 10

    def test_custom_config(self, test_agent):
        """Custom configuration via config object."""
        summarizer = TokenBudgetSummarizer(
            test_agent, config=TokenBudgetConfig(max_tokens=50_000, preserve_recent=5)
        )
        assert summarizer.config.max_tokens == 50_000
        assert summarizer.config.preserve_recent == 5

    def test_should_summarize_under_budget(self, test_agent):
        """Should not summarize when under budget."""
        # Add a few small events (well under 100k tokens)
        for i in range(5):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent)  # Default 100k budget

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

    def test_should_summarize_over_budget(self, test_agent):
        """Should summarize when this runtime's provider-reported prompt count is over budget."""
        test_agent.runtime._last_prompt_tokens_actual = 500
        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=100))

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

    def test_should_not_summarize_from_estimate_when_actual_under_budget(self, test_agent):
        """A local estimate alone does not trigger summarization; actual usage is authoritative."""
        from nooa import ContextWindowStats

        test_agent.runtime._last_prompt_tokens_actual = 600
        test_agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=0,
            events_count=20,
            context_blocks_chars=0,
            events_chars=1500,
            prompt_tokens=1500,
        )
        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=1000))
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        assert summarizer._should_summarize(event) is False

    def test_should_not_summarize_without_provider_actual(self, test_agent):
        """No provider actual means no token-budget summarization trigger."""
        from nooa import ContextWindowStats

        test_agent.runtime._last_prompt_tokens_actual = None
        test_agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=0,
            events_count=20,
            context_blocks_chars=0,
            events_chars=1500,
            prompt_tokens=1500,
        )
        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=1000))
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        assert summarizer._should_summarize(event) is False

    def test_compute_range_preserves_recent(self, test_agent):
        """Compute range preserves recent events."""
        # Add 5 events: tags will be "1", "2", "3", "4", "5"
        for i in range(5):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(preserve_recent=2))

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        result = summarizer._compute_range(event)

        # With 5 tags and preserve_recent=2, should summarize "1" to "3"
        # (preserve "4" and "5")
        assert result == ("1", "3")

    def test_compute_range_returns_none_when_too_few_events(self, test_agent):
        """Returns None when not enough events to summarize."""
        # Add only 2 events
        test_agent.event_manager.add(Message(content="Message 1"))
        test_agent.event_manager.add(Message(content="Message 2"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(preserve_recent=5))

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        result = summarizer._compute_range(event)
        assert result is None


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

    def test_summarizer_attaches_to_agent(self, test_agent):
        """Summarizer attaches to agent via constructor."""
        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Summarizer should be wired up
        assert summarizer.target_event_manager is test_agent.event_manager
        assert summarizer._llm is test_agent._llm

    def test_summarizer_inherits_llm_from_agent(self, test_agent, fake_llm):
        """Summarizer inherits LLM from agent if not explicitly set."""
        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Summarizer should inherit agent's LLM
        assert summarizer._llm is fake_llm

    def test_summarizer_uses_own_llm(self, test_agent, fake_llm):
        """Summarizer uses its own LLM if explicitly set."""
        summarizer_llm = FakeLLMClient()

        summarizer = TokenBudgetSummarizer(
            test_agent, llm=summarizer_llm, config=TokenBudgetConfig(max_tokens=50_000)
        )

        # Summarizer should keep its own LLM
        assert summarizer._llm is summarizer_llm
        assert summarizer._llm is not fake_llm

    def test_agent_standalone_without_summarizer(self, fake_llm):
        """Agent works fine without summarizer - they're decoupled."""

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # No _summarizers attribute on agent - they're decoupled
        assert not hasattr(agent, "_summarizers")

    def test_install_class_method_stores_on_agent(self, test_agent):
        """install() class method stores summarizer on agent for lifetime management."""
        # Use install() instead of constructor
        summarizer = TokenBudgetSummarizer.install(
            test_agent, config=TokenBudgetConfig(max_tokens=50_000)
        )

        # Summarizer should be wired up
        assert summarizer.target_event_manager is test_agent.event_manager
        assert summarizer._llm is test_agent._llm

        # Summarizer should be stored on agent (no need to keep reference)
        assert hasattr(test_agent, "_summarizers")
        assert summarizer in test_agent._summarizers

    def test_install_can_be_fire_and_forget(self, test_agent):
        """install() doesn't require keeping a reference - stored on agent."""
        # Fire and forget - don't capture return value
        TokenBudgetSummarizer.install(test_agent, config=TokenBudgetConfig(max_tokens=80_000))

        # Agent should have the summarizer attached
        assert hasattr(test_agent, "_summarizers")
        assert len(test_agent._summarizers) == 1
        assert isinstance(test_agent._summarizers[0], TokenBudgetSummarizer)
        assert test_agent._summarizers[0].config.max_tokens == 80_000

    def test_token_budget_install_with_config(self, test_agent):
        """install() accepts config= keyword argument."""
        s = TokenBudgetSummarizer.install(test_agent, config=TokenBudgetConfig(max_tokens=80_000))
        assert s.config.max_tokens == 80_000

    def test_token_budget_install_rejects_flat_kwargs(self, test_agent):
        """install() raises TypeError on flat config kwargs."""
        with pytest.raises(TypeError):
            TokenBudgetSummarizer.install(test_agent, max_tokens=80_000)

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


class TestSummarizationAsyncIntegration:
    """Async integration tests for the full summarization flow."""

    @pytest.mark.asyncio
    async def test_schedule_summarization_creates_background_task(self, test_agent):
        """_schedule_summarization creates a background task."""
        # Add events to summarize
        for i in range(5):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Mock the summarize method to return a fixed summary
        summarizer.summarize = AsyncMock(return_value="Mocked summary of messages 1-3")

        # Manually trigger scheduling
        summarizer._schedule_summarization("1", "3")

        # Task should be created
        assert summarizer._pending_task is not None
        assert summarizer._pending_range == ("1", "3")

        # Wait for task to complete
        await summarizer._pending_task

        # Summary should be pending application
        assert summarizer._pending_summary is not None
        assert summarizer._pending_summary == "Mocked summary of messages 1-3"

    @pytest.mark.asyncio
    async def test_apply_pending_summary_collapses_history(self, test_agent):
        """_apply_pending_summary collapses events in target history."""
        # Add events to summarize
        for i in range(10):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        # Create summarizer that will produce a mock summary
        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Manually set pending state (simulating completed background task)
        summarizer._pending_range = ("1", "5")
        summarizer._pending_summary = "Summary of messages 1-5"
        summarizer._pending_task = asyncio.create_task(asyncio.sleep(0))  # Completed task
        await summarizer._pending_task

        # Apply the summary
        summarizer._apply_pending_summary()

        # History should have a collapsed range
        active_tags = test_agent.event_manager.keys()
        assert "1..5" in active_tags
        # Original individual tags should be gone from active list
        assert "1" not in active_tags
        assert "2" not in active_tags
        assert "5" not in active_tags
        # But later events should remain
        assert "6" in active_tags

    @pytest.mark.asyncio
    async def test_after_turn_triggers_summarization_when_over_budget(self, test_agent):
        """_handle_after_turn schedules summarization when over token budget."""
        # Add enough events to have something to summarize
        for _ in range(20):
            test_agent.event_manager.add(Message(content="x" * 100))

        # Simulate this runtime's last successful call reporting an over-budget prompt.
        test_agent.runtime._last_prompt_tokens_actual = 1000

        # Very low budget to trigger summarization
        summarizer = TokenBudgetSummarizer(
            test_agent, config=TokenBudgetConfig(max_tokens=50, preserve_recent=5)
        )

        # Mock the summarize method
        summarizer.summarize = AsyncMock(return_value="Summary")

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )

        # Trigger after_turn handler
        summarizer._handle_after_turn(event)

        # Should have scheduled summarization
        assert summarizer._pending_task is not None
        assert summarizer._pending_range is not None

        # Wait for task
        await summarizer._pending_task

    @pytest.mark.asyncio
    async def test_before_turn_applies_pending_summary(self, test_agent):
        """_handle_before_turn applies any pending summary."""
        # Add events
        for i in range(10):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Set up pending state
        summarizer._pending_range = ("1", "5")
        summarizer._pending_summary = "Summary text"
        summarizer._pending_task = asyncio.create_task(asyncio.sleep(0))
        await summarizer._pending_task

        # Create before_turn event
        from nooa.events import BeforeTurn

        event = BeforeTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=2,
        )

        # Trigger before_turn handler
        summarizer._handle_before_turn(event)

        # Summary should have been applied
        assert summarizer._pending_task is None
        assert summarizer._pending_summary is None
        assert "1..5" in test_agent.event_manager.keys()

    @pytest.mark.asyncio
    async def test_end_to_end_summarization_flow(self, fake_llm):
        """Full flow: add events → trigger summarization → verify collapse."""

        # Create agent with events
        class SimpleAgent(Agent, llm=fake_llm):
            pass

        agent = SimpleAgent()

        # Add 20 events with substantial content
        for i in range(20):
            agent.event_manager.add(Message(content=f"Message {i}: " + "x" * 50))

        initial_tag_count = len(agent.event_manager.keys())
        assert initial_tag_count == 20

        # Simulate this runtime's last successful call reporting an over-budget prompt.
        agent.runtime._last_prompt_tokens_actual = 1000

        # Create summarizer with low threshold to trigger
        summarizer = TokenBudgetSummarizer(
            agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=5)
        )

        # Mock the summarize method to return a fixed summary
        summarizer.summarize = AsyncMock(return_value="Summary of old messages")

        # Simulate after_turn event
        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )

        # This should trigger background summarization
        summarizer._handle_after_turn(event)

        # Wait for background task
        if summarizer._pending_task:
            await summarizer._pending_task

        # Apply pending summary (normally happens on next before_turn)
        summarizer._apply_pending_summary()

        # Verify history was collapsed
        final_tags = agent.event_manager.keys()
        assert len(final_tags) < initial_tag_count

        # Should have a summary tag
        summary_tags = [t for t in final_tags if ".." in t]
        assert len(summary_tags) >= 1

        # Recent events should be preserved
        assert str(initial_tag_count) in final_tags  # Last event "20"

    @pytest.mark.asyncio
    async def test_concurrent_summarization_requests_are_deduplicated(self, test_agent):
        """Multiple after_turn events don't create multiple tasks."""
        # Add events
        for _ in range(20):
            test_agent.event_manager.add(Message(content="x" * 100))

        summarizer = TokenBudgetSummarizer(
            test_agent, config=TokenBudgetConfig(max_tokens=50, preserve_recent=5)
        )

        # Mock the summarize method
        summarizer.summarize = AsyncMock(return_value="Summary")

        event = AfterTurn(
            method_name="test",
            strategy="CODEACT",
            generation_id="gen-123",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )

        # Trigger multiple times
        summarizer._handle_after_turn(event)
        first_task = summarizer._pending_task

        summarizer._handle_after_turn(event)
        second_task = summarizer._pending_task

        # Should be the same task (not a new one)
        assert first_task is second_task

        # Clean up
        if first_task:
            await first_task

    @pytest.mark.asyncio
    async def test_summarizer_clears_own_history_after_summarization(self, test_agent):
        """Summarizer's own history is cleared after each summarize() call."""
        for i in range(10):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Mock the summarize method
        summarizer.summarize = AsyncMock(return_value="Summary")

        # Add something to summarizer's history
        summarizer.event_manager.add(Message(content="Internal"))
        assert len(summarizer.event_manager.values()) == 1

        # Run summarization
        summarizer._schedule_summarization("1", "5")
        await summarizer._pending_task

        # Summarizer's history should be cleared
        assert len(summarizer.event_manager.values()) == 0

    @pytest.mark.asyncio
    async def test_after_turn_does_not_apply_pending_summary(self, test_agent):
        """AfterTurn must NOT collapse events. Application is the job of
        BeforeTurn alone — separating the two phases keeps
        ``_should_summarize`` reading stats that match the event list the
        LLM just saw, eliminating the cascade.
        """
        for i in range(10):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Completed pending summary waiting to be applied.
        summarizer._pending_range = ("1", "5")
        summarizer._pending_summary = "Summary"
        summarizer._pending_task = asyncio.create_task(asyncio.sleep(0))
        await summarizer._pending_task

        after_turn = AfterTurn(
            method_name="t",
            strategy="CODEACT",
            generation_id="g",
            parent_generation_id=None,
            turn_number=1,
            is_final=False,
            success=True,
        )
        summarizer._handle_after_turn(after_turn)

        # Pending state is untouched; nothing collapsed.
        assert summarizer._pending_task is not None
        assert summarizer._pending_summary == "Summary"
        assert not any(".." in t for t in test_agent.event_manager.keys())

    @pytest.mark.asyncio
    async def test_after_turn_skips_while_pending_task_is_done_but_unapplied(self, test_agent):
        """Dedup guard: a done-but-unapplied pending task must block
        scheduling a new summarization. Without this, a new schedule call
        would overwrite ``_pending_summary`` and the already-computed
        summary would be lost.
        """
        from nooa import ContextWindowStats

        for _ in range(20):
            test_agent.event_manager.add(Message(content="x" * 100))

        test_agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=0,
            events_count=20,
            context_blocks_chars=0,
            events_chars=200_000,
            prompt_tokens=200_000,
        )
        test_agent.runtime._last_prompt_tokens_actual = 200_000

        summarizer = TokenBudgetSummarizer(
            test_agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=5)
        )
        summarizer.summarize = AsyncMock(return_value="Fresh Summary")

        # Pretend a prior turn scheduled a summarization that already
        # completed. The task is done, the result is sitting in
        # _pending_summary, waiting for BeforeTurn to apply it.
        summarizer._pending_range = ("1", "10")
        summarizer._pending_summary = "Prior Summary"
        summarizer._pending_task = asyncio.create_task(asyncio.sleep(0))
        await summarizer._pending_task
        assert summarizer._pending_task.done()

        after_turn = AfterTurn(
            method_name="t",
            strategy="CODEACT",
            generation_id="g",
            parent_generation_id=None,
            turn_number=1,
            is_final=False,
            success=True,
        )
        summarizer._handle_after_turn(after_turn)

        # Pending state preserved; nothing rescheduled.
        assert summarizer._pending_summary == "Prior Summary"
        summarizer.summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_cascade_across_turns(self, test_agent):
        """End-to-end: AfterTurn schedules → BeforeTurn applies → AfterTurn
        on fresh stats does NOT re-schedule.

        Pins the design invariant: applying at a turn boundary is what
        guarantees ``_should_summarize`` on the next AfterTurn reads
        post-collapse stats.
        """
        from nooa import ContextWindowStats
        from nooa.events import BeforeTurn

        for _ in range(20):
            test_agent.event_manager.add(Message(content="x" * 100))

        # Simulate the runtime having shipped an over-budget prompt.
        test_agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=0,
            events_count=20,
            context_blocks_chars=0,
            events_chars=200_000,
            prompt_tokens=200_000,
        )
        test_agent.runtime._last_prompt_tokens_actual = 200_000

        summarizer = TokenBudgetSummarizer(
            test_agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=5)
        )
        summarizer.summarize = AsyncMock(return_value="Summary")

        after_turn = AfterTurn(
            method_name="t",
            strategy="CODEACT",
            generation_id="g",
            parent_generation_id=None,
            turn_number=1,
            is_final=False,
            success=True,
        )
        before_turn = BeforeTurn(
            method_name="t",
            strategy="CODEACT",
            generation_id="g",
            parent_generation_id=None,
            turn_number=2,
        )

        # Turn 1 AfterTurn — over budget → schedules.
        summarizer._handle_after_turn(after_turn)
        assert summarizer._pending_task is not None
        await summarizer._pending_task

        # Turn 2 BeforeTurn — applies the completed summary.
        summarizer._handle_before_turn(before_turn)
        test_agent.runtime._last_prompt_tokens_actual = None
        active = test_agent.event_manager.keys()
        assert any(".." in t for t in active), "BeforeTurn should apply the summary"
        assert summarizer._pending_task is None

        # Simulate the next _build_messages() having re-rendered with the
        # post-collapse event list — total_tokens drops below budget.
        test_agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=0,
            events_count=6,
            context_blocks_chars=0,
            events_chars=50,
            prompt_tokens=50,
        )

        # Turn 2 AfterTurn — fresh stats say under budget, no reschedule.
        summarizer._handle_after_turn(after_turn)
        assert summarizer._pending_task is None
        assert summarizer.summarize.call_count == 1, (
            "summarize() should only have been invoked once across the whole flow"
        )

    @pytest.mark.asyncio
    async def test_uninstall_cancels_pending_task(self, test_agent):
        """_uninstall() cancels any pending summarization task."""
        for i in range(10):
            test_agent.event_manager.add(Message(content=f"Message {i}"))

        summarizer = TokenBudgetSummarizer(test_agent, config=TokenBudgetConfig(max_tokens=50_000))

        # Start a summarization
        summarizer._schedule_summarization("1", "5")
        task = summarizer._pending_task
        assert task is not None

        # Uninstall should cancel
        summarizer._uninstall()

        # Task should be cancelled
        assert summarizer._pending_task is None
        # Give a moment for cancellation to propagate
        await asyncio.sleep(0.01)
        assert task.cancelled() or task.done()


# =============================================================================
# Tracing opt-out — issue #192
# =============================================================================
class TestSummarizerNoTrace:
    """Regression test: @hidden summarizer helpers must opt out of tracing.

    Private helpers fire on every turn and used to drown out useful spans.
    The fix decorates them with @no_trace so only ``summarize()`` produces a span.
    """

    @pytest.mark.parametrize("cls", [SummarizationAgent, TokenBudgetSummarizer, MethodSummarizer])
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
