# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarization agents for NVIDIA OO Agents.

This module provides agent-based event summarization following the NVIDIA OO Agents pattern.
Summarizers are proper agents that subscribe to a parent's event manager and use
LLM-generated code to produce summaries.

Example:
    from nooa import Agent
    from nooa.agents import TokenBudgetSummarizer

    class MyAgent(Agent, llm=my_llm):
        async def chat(self, message: str) -> str:
            '''Process a chat message.'''
            ...

    # Create agent and install summarizer
    agent = MyAgent()
    from nooa.config.summarizer_config import TokenBudgetConfig
    TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=80_000))
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from nooa.agent import Agent
from nooa.agentdoc import hidden
from nooa.config.strategy_config import PredictConfig
from nooa.decorators import strategy
from nooa.metaclass import no_trace
from nooa.strategies import PredictStrategy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nooa.config.summarizer_config import MethodSummarizerConfig, TokenBudgetConfig
    from nooa.events import AfterTurn, EventBase
    from nooa.runtime.event_manager import EventManager


class SummarizationAgent(Agent):
    """Base class for summarization subagents.

    Summarizers subscribe to a parent agent's event manager and produce
    summaries when triggered. The actual summarization logic is LLM-generated.

    Subclasses define:
    - When to summarize (override _should_summarize)
    - What range to summarize (override _compute_range)

    The base class handles:
    - Event subscription to target_event_manager
    - Background task management
    - Applying summaries at safe boundaries (before_turn)
    - Clearing own events after each summarize() call (stateless)

    Configuration uses Annotated types for self-documentation:
        max_tokens: Annotated[int, "Description"] = 100_000
    """

    # Event manager whose events will be summarized. Wired automatically by the parent Agent.
    target_event_manager: Annotated["EventManager | None", hidden] = None

    # Parent agent. Used to read runtime-computed context stats for accurate
    # token counting. Hidden from the LLM.
    _target_agent: Annotated["Agent | None", hidden] = None

    # Config must be set by subclasses (TokenBudgetSummarizer, MethodSummarizer set self.config
    # in __init__). The base class provides this sentinel to prevent AttributeError if a
    # subclass forgets to set it or if the base is instantiated directly.
    config: Annotated[Any, hidden] = None

    # Background task state
    _pending_task: Annotated[asyncio.Task[Any] | None, hidden]
    _pending_range: Annotated[tuple[str, str] | None, hidden]
    _pending_summary: Annotated[str | None, hidden]
    _unsub_before: Annotated[Callable[[], None] | None, hidden]
    _unsub_after: Annotated[Callable[[], None] | None, hidden]

    @classmethod
    def install(cls, agent: Agent, **kwargs: Any) -> "SummarizationAgent":
        """Install this summarizer on an agent.

        The summarizer is stored on the agent, so you don't need to keep
        a reference to it - its lifetime is tied to the agent's lifetime.

        Args:
            agent: Agent to attach to. The summarizer:
                   - Inherits LLM from agent (unless explicitly overridden)
                   - Attaches to agent's event manager automatically
            **kwargs: Passed to constructor (use config= on subclasses)

        Returns:
            The installed summarizer (usually not needed)

        Example:
            from nooa.config.summarizer_config import TokenBudgetConfig
            agent = MyAgent(llm=my_llm)
            TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=80_000))
        """
        summarizer = cls(agent, **kwargs)

        # Store on agent to prevent GC - lifetime tied to agent
        if not hasattr(agent, "_summarizers"):
            agent._summarizers = []  # type: ignore[attr-defined]
        agent._summarizers.append(summarizer)  # type: ignore[attr-defined]

        return summarizer

    def __init__(self, agent: Agent, **kwargs: Any) -> None:
        """Initialize summarization agent attached to a parent agent.

        Prefer using the `install()` class method instead of direct construction.

        Args:
            agent: Parent agent to attach to. The summarizer:
                   - Inherits LLM from agent (unless explicitly overridden)
                   - Attaches to agent's event manager automatically
            **kwargs: Passed through to Agent.__init__ (e.g. llm=)
        """
        # Inherit LLM from parent agent unless explicitly provided
        kwargs.setdefault("llm", agent.llm)
        self.target_event_manager = agent.event_manager
        self._target_agent = agent

        # Extract annotated class attributes from kwargs (max_tokens, preserve_recent, etc.)
        # Skip descriptors (e.g. Agent's ``llm`` property): they are not summarizer
        # config fields and must pass through to ``Agent.__init__`` (which resolves
        # ``llm`` itself). Assigning to a read-only property would also raise.
        for name in list(kwargs.keys()):
            if hasattr(self.__class__, name) and not isinstance(
                getattr(type(self), name, None), property
            ):
                setattr(self, name, kwargs.pop(name))

        super().__init__(**kwargs)

        # Background task state
        self._pending_task: asyncio.Task[Any] | None = None
        self._pending_range: tuple[str, str] | None = None
        self._pending_summary: str | None = None

        # Unsubscribe functions for cleanup
        self._unsub_before: Callable[[], None] | None = None
        self._unsub_after: Callable[[], None] | None = None

        # Install event subscriptions
        self._install()

    @hidden
    @no_trace
    def _install(self) -> None:
        """Subscribe to target event manager.

        Called automatically if target_event_manager is set at creation,
        or by parent Agent after wiring target_event_manager.
        """
        if self.target_event_manager is None:
            raise ValueError("Cannot install: target_event_manager is None")

        self._unsub_before = self.target_event_manager.on("BeforeTurn", self._handle_before_turn)
        self._unsub_after = self.target_event_manager.on("AfterTurn", self._handle_after_turn)

    @hidden
    @no_trace
    def _uninstall(self) -> None:
        """Unsubscribe from target event manager and cancel pending tasks."""
        if self._unsub_before:
            self._unsub_before()
            self._unsub_before = None
        if self._unsub_after:
            self._unsub_after()
            self._unsub_after = None
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    @hidden
    @no_trace
    def _handle_before_turn(self, event: "EventBase") -> None:
        """Apply any pending summary before the next turn."""
        self._apply_pending_summary()

    @hidden
    @no_trace
    def _handle_after_turn(self, event: "EventBase") -> None:
        """Check whether to schedule a summarization.

        Never applies pending summaries — that's ``_handle_before_turn``'s
        job. Keeping the two phases separate means ``_should_summarize``
        always reads ``agent.context_stats`` from the most recent
        ``_build_messages()`` call, which reflects the actual event list
        the LLM just saw. Applying a collapse inside this same handler (as
        earlier versions did) left stats stale for the immediate
        ``_should_summarize`` call and produced the cascade pattern where
        a fresh summary was followed seconds later by another over a
        handful of newly-added events.

        Dedup guard skips while ANY pending state exists — running OR
        done-but-not-yet-applied. Without this, a task that finished mid
        -turn would be overwritten by a new schedule call, losing the
        waiting-to-apply summary.
        """
        from nooa.events import AfterTurn as _AfterTurn

        if self._pending_task is not None:
            return

        if not isinstance(event, _AfterTurn):
            return

        if self._should_summarize(event):
            range_result = self._compute_range(event)
            if range_result is not None:
                start_tag, end_tag = range_result
                self._schedule_summarization(start_tag, end_tag)

    # -------------------------------------------------------------------------
    # Override in subclasses
    # -------------------------------------------------------------------------

    @hidden
    @no_trace
    def _should_summarize(self, event: "AfterTurn") -> bool:
        """Decide if summarization is needed.

        Override in subclasses to implement policy logic.

        Args:
            event: The AfterTurn that triggered this check

        Returns:
            True if summarization should be scheduled
        """
        return False

    @hidden
    @no_trace
    def _compute_range(self, event: "AfterTurn") -> tuple[str, str] | None:
        """Compute the range of events to summarize.

        Override in subclasses to implement policy logic.

        Args:
            event: The AfterTurn that triggered this check

        Returns:
            (start_tag, end_tag) tuple, or None to skip summarization
        """
        return None

    # -------------------------------------------------------------------------
    # LLM-generated summarization
    # -------------------------------------------------------------------------

    # The source document is rendered explicitly by _render_range_to_markdown().
    # Do not use a method-level unbounded TruncationConfig here: that would also
    # re-render unrelated context events in this generation with unbounded event_format.
    @strategy(PredictStrategy(PredictConfig(max_param_chars=None)))
    async def summarize(self, history_markdown: str, target_chars: int) -> str:
        """Summarize the `history_markdown` parameter into approximately {target_chars} characters.

        Structure your summary as:

        **Summary:**

        **Topics:** [2-4 bullet points]

        **Outcomes:** [Decisions, conclusions - preserve exact numbers/terms]

        **State:** [Current status, pending items]

        KEEP: Decisions, numbers, technical terms, conclusions
        COMPRESS: Discussions (themes only), Q&A (essence only)
        OMIT: Greetings, process details, resolved errors

        Target length: ~{target_chars} characters
        """
        # NOTE: Don't include {history_markdown} inline in the docstring - the
        # strategy already shows all parameters in the "Input parameters" section.
        # Including it here would duplicate the content and waste tokens.
        ...

    # -------------------------------------------------------------------------
    # Background task management
    # -------------------------------------------------------------------------

    @hidden
    @no_trace
    def _schedule_summarization(self, start_tag: str, end_tag: str) -> None:
        """Schedule async summarization as a background task."""
        if self.target_event_manager is None:
            return

        self._pending_range = (start_tag, end_tag)

        # Render events to markdown for LLM consumption
        history_markdown = self._render_range_to_markdown(start_tag, end_tag)

        logger.debug(
            f"Scheduling summarization: {start_tag} -> {end_tag} ({len(history_markdown)} chars)"
        )

        # Schedule the async summarization
        self._pending_task = asyncio.create_task(
            self._run_summarization(history_markdown, start_tag, end_tag)
        )

    @hidden
    @no_trace
    async def _run_summarization(self, history_markdown: str, start_tag: str, end_tag: str) -> None:
        """Run summarization and store result for later application."""
        try:
            summary = await self.summarize(history_markdown, self.config.target_chars)
            self._pending_summary = summary
            logger.debug(
                f"Summarization complete: {start_tag} -> {end_tag} (summary: {len(summary)} chars)"
            )
        except Exception as e:
            # Log error but don't crash - summarization is best-effort
            logger.warning(f"Summarization failed: {e}")
            self._pending_summary = None
        finally:
            # Clear our own events to stay stateless
            self.event_manager.clear()

    @hidden
    @no_trace
    def _apply_pending_summary(self) -> None:
        """Apply a completed summary to the target event manager.

        Called only from ``_handle_before_turn``. Running exclusively at
        the turn boundary guarantees the next ``_build_messages()`` sees
        the collapsed event list, so the subsequent ``_should_summarize``
        check at AfterTurn reads fresh stats — no cascade from stale
        cached totals possible.
        """
        if self._pending_task is None:
            return

        if not self._pending_task.done():
            return

        # Get the result (summary is stored in _pending_summary by _run_summarization)
        if self._pending_summary is not None and self._pending_range is not None:
            start_tag, end_tag = self._pending_range
            try:
                assert self.target_event_manager is not None
                self.target_event_manager.collapse(start_tag, end_tag, self._pending_summary)
                logger.debug(f"Applied summary: {start_tag} -> {end_tag}")
            except Exception as e:
                logger.warning(f"Failed to apply summary: {e}")

        # Clear pending state
        self._pending_task = None
        self._pending_range = None
        self._pending_summary = None

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    @hidden
    @no_trace
    def _get_events_in_range(self, start_tag: str, end_tag: str) -> list[tuple[str, "EventBase"]]:
        """Get events between start_tag and end_tag (inclusive).

        Internal ``@hidden`` helper for the agent's own Python methods to
        investigate raw events. Not part of the generated-code surface (the
        agent runs single-shot PredictStrategy, which executes no code).

        Args:
            start_tag: First event tag in the range
            end_tag: Last event tag in the range

        Returns:
            List of (tag, event) tuples
        """
        if self.target_event_manager is None:
            return []

        result = []
        # keys() returns tags in chronological (insertion) order
        tags = self.target_event_manager.keys()

        in_range = False
        for tag in tags:
            if tag == start_tag:
                in_range = True
            if in_range:
                event = self.target_event_manager[tag]
                result.append((tag, event))
            if tag == end_tag:
                break

        return result

    @hidden
    @no_trace
    def _render_range_to_markdown(self, start_tag: str, end_tag: str) -> str:
        """Render events in range to markdown for LLM consumption.

        Uses context-blocks MarkdownBlockFormatter for consistent rendering
        with proper truncation and pformat handling.

        Args:
            start_tag: First event tag in the range
            end_tag: Last event tag in the range

        Returns:
            Markdown-formatted events section
        """
        if self.target_event_manager is None:
            return ""

        events = self._get_events_in_range(start_tag, end_tag)

        if not events:
            return ""

        from nooa.context_blocks import (
            BlockMetadata,
            ResolvedBlock,
            Role,
            format_message_content,
        )
        from nooa.context_blocks.utils import truncating_pformat

        event_format = None
        if self._target_agent is not None:
            target_truncation = getattr(self._target_agent, "_truncation", None)
            if target_truncation is not None:
                event_format = target_truncation.event_format
        format_kwargs = event_format.model_dump() if event_format is not None else {}

        rendered: list[str] = []
        for tag, event in events:
            event_role = getattr(event, "_role", Role.USER)
            body = truncating_pformat(event, **format_kwargs)
            block = ResolvedBlock(
                key=f"event_{tag}",
                content=body,
                role=event_role,
                metadata=BlockMetadata(expr=f'self.events["{tag}"]', tag=str(tag)),
            )
            rendered.append(format_message_content(block, "markdown"))

        # Total-input cap. Per-event ``event_format`` bounds each event's fields,
        # but the SUM across a long range can still exceed the summarizer's own
        # model context window — the summarize() call then 400s ("prompt is too
        # long") and compaction can never make progress. Keep the NEWEST events
        # (most relevant to a resume summary) under a token budget and head-drop
        # the oldest with an explicit marker, so the single summarize() call
        # always fits. See _input_token_budget for the budget derivation.
        budget = self._input_token_budget()
        if budget is not None:
            count = self._input_token_counter()
            kept: list[str] = []
            used = 0
            dropped = 0
            sep = 4  # "\n\n" between parts, counted approximately
            # Walk newest -> oldest so the most recent survive the cap.
            for part in reversed(rendered):
                cost = count(part) + sep
                if used + cost > budget and kept:
                    dropped = len(rendered) - len(kept)
                    break
                kept.append(part)
                used += cost
            if dropped:
                kept.reverse()
                marker = (
                    f"[... {dropped} older event(s) omitted to fit the summarizer's "
                    f"input budget; summarize what remains ...]"
                )
                return "\n\n".join([marker, *kept])
            # No drop: kept is newest-first; restore chronological order.
            rendered = list(reversed(kept))

        return "\n\n".join(rendered)

    @hidden
    @no_trace
    def _input_token_counter(self) -> "Callable[[str], int]":
        """Token counter for sizing the summarizer's own input.

        Prefer the summarizer LLM's ``count_tokens``; fall back to the shared
        char-approximate counter so the cap still applies when no counter is set.
        """
        llm = getattr(self, "_llm", None)
        counter = getattr(llm, "count_tokens", None)
        if callable(counter):
            return counter
        from nooa.token_counter import char_approximate_token_counter

        return char_approximate_token_counter

    @hidden
    @no_trace
    def _input_token_budget(self) -> int | None:
        """Max tokens for the rendered summarization input (history_markdown).

        ~70% of the summarizer model's context window, leaving headroom for the
        summarize() method's own prompt scaffolding (docstring, instructions,
        target_chars) and the completion. ``None`` (no cap) when the model
        window can't be determined — never wipe the input on a misconfig; the
        API error path is still the backstop."""
        llm = getattr(self, "_llm", None)
        for attr in ("context_window", "context_limit"):
            window = getattr(llm, attr, None)
            if isinstance(window, int) and window > 0:
                return int(window * 0.7)
        return None


# =============================================================================
# Helper Functions
# =============================================================================
def context_budget(llm: Any, percent: float = 0.8, fallback: int = 100_000) -> int:
    """Calculate token budget as percentage of the LLM's context window.

    Args:
        llm: LLM instance with ``context_window`` attribute (``context_limit``
             also accepted for historical callers)
        percent: Fraction of context to use (0.0-1.0), default 0.8 (80%)
        fallback: Value returned when the LLM doesn't expose either attribute

    Returns:
        Token budget as integer.

    Example:
        TokenBudgetSummarizer.install(
            agent, config=TokenBudgetConfig(max_tokens=context_budget(my_llm, 0.8))
        )
    """
    if percent <= 0:
        raise ValueError("percent must be > 0")

    # ``context_window`` is the UnifiedLLM convention. ``context_limit`` was
    # the originally-documented attribute name but no shipped LLM client sets
    # it — falling back keeps the helper useful for any custom wrapper that
    # does. Treat non-positive limits as unavailable; returning 0 silently
    # disables useful token-budget summarization.
    for attr in ("context_window", "context_limit"):
        limit = getattr(llm, attr, None)
        if limit is not None and limit > 0:
            return int(limit * percent)
    return fallback


# =============================================================================
# Example Summarizers (Good Defaults)
# =============================================================================
class TokenBudgetSummarizer(SummarizationAgent):
    """Summarize when event count exceeds token budget.

    Trigger: Event tokens > config.max_tokens
    Action: Summarize oldest events, preserve N most recent

    Example:
        from nooa.config.summarizer_config import TokenBudgetConfig
        # Absolute limit
        TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=80_000))

        # Percentage of LLM context
        TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=context_budget(my_llm, 0.8)))
    """

    @classmethod
    def install(
        cls, agent: Agent, *, config: "TokenBudgetConfig | None" = None, **kwargs: Any
    ) -> SummarizationAgent:
        """Install with a TokenBudgetConfig.

        Args:
            agent: Agent to attach to.
            config: TokenBudgetConfig instance. Use TokenBudgetConfig(field=value) to override.
            **kwargs: Only 'llm' is allowed; all other flat kwargs raise TypeError.
        """
        unknown = set(kwargs) - {"llm"}
        if unknown:
            raise TypeError(
                f"TokenBudgetSummarizer.install() got unexpected keyword arguments: "
                f"{sorted(unknown)}. Use config=TokenBudgetConfig(...) instead."
            )
        return super().install(agent, config=config, **kwargs)

    def __init__(self, agent: Agent, **kwargs: Any) -> None:
        from nooa.config.summarizer_config import TokenBudgetConfig as _TBC

        config = kwargs.pop("config", None)
        self.config = config or _TBC()
        super().__init__(agent, **kwargs)

    @hidden
    @no_trace
    def _should_summarize(self, event: "AfterTurn") -> bool:
        """Trigger only from provider-reported prompt tokens."""
        agent = self._target_agent
        if agent is None:
            return False

        try:
            actual = agent.runtime.last_prompt_tokens_actual
        except Exception:
            logger.warning(
                "TokenBudgetSummarizer: failed to read API-reported token count", exc_info=True
            )
            return False

        return actual is not None and actual > self.config.max_tokens

    @hidden
    @no_trace
    def _compute_range(self, event: "AfterTurn") -> tuple[str, str] | None:
        """Summarize oldest events, preserving recent ones."""
        if self.target_event_manager is None:
            return None

        tags = self.target_event_manager.keys()
        if len(tags) <= self.config.preserve_recent:
            return None

        # Summarize from oldest to (len - preserve_recent - 1)
        start_tag = tags[0]
        end_tag = tags[-(self.config.preserve_recent + 1)]
        return (start_tag, end_tag)


class MethodSummarizer(SummarizationAgent):
    """Summarize after each method call completes.

    Trigger: event.is_final == True (method completed)
    Action: Summarize all events from that method's generation_id

    Example:
        from nooa.config.summarizer_config import MethodSummarizerConfig
        agent = MyAgent(llm=my_llm)
        MethodSummarizer.install(agent, config=MethodSummarizerConfig(min_events=5))
    """

    @classmethod
    def install(
        cls, agent: Agent, *, config: "MethodSummarizerConfig | None" = None, **kwargs: Any
    ) -> SummarizationAgent:
        """Install with a MethodSummarizerConfig.

        Args:
            agent: Agent to attach to.
            config: MethodSummarizerConfig instance.
            **kwargs: Only 'llm' is allowed; all other flat kwargs raise TypeError.
        """
        unknown = set(kwargs) - {"llm"}
        if unknown:
            raise TypeError(
                f"MethodSummarizer.install() got unexpected keyword arguments: "
                f"{sorted(unknown)}. Use config=MethodSummarizerConfig(...) instead."
            )
        return super().install(agent, config=config, **kwargs)

    def __init__(self, agent: Agent, **kwargs: Any) -> None:
        from nooa.config.summarizer_config import MethodSummarizerConfig as _MSC

        config = kwargs.pop("config", None)
        self.config = config or _MSC()
        super().__init__(agent, **kwargs)

    @hidden
    @no_trace
    def _should_summarize(self, event: "AfterTurn") -> bool:
        """Trigger on method completion."""
        if not event.is_final:
            return False

        # Optionally exclude root calls
        if self.config.exclude_root and self._is_root_call(event):
            return False

        return True

    @hidden
    @no_trace
    def _compute_range(self, event: "AfterTurn") -> tuple[str, str] | None:
        """Summarize all events from this method invocation (including children).

        Uses call_id from event metadata to find the range. Child method calls
        (nested call_ids) are included because they fall chronologically between
        the first and last event with the parent call_id.
        """
        if self.target_event_manager is None:
            return None

        call_id = event.metadata.get("call_id")
        if call_id is None:
            return None

        # Find first and last tags with this call_id.
        # Events from child calls (different call_ids) are interleaved
        # chronologically, so the range naturally includes them.
        first_tag: str | None = None
        last_tag: str | None = None
        match_count = 0
        for tag in self.target_event_manager.keys():
            evt = self.target_event_manager[tag]
            if evt.metadata.get("call_id") == call_id:
                if first_tag is None:
                    first_tag = tag
                last_tag = tag
                match_count += 1

        if match_count < self.config.min_events:
            return None

        return (first_tag, last_tag)  # type: ignore[return-value]

    @hidden
    @no_trace
    def _is_root_call(self, event: "AfterTurn") -> bool:
        """Check if this is a root (top-level) method call."""
        # Root calls typically don't have a parent generation context
        # This is a heuristic - could be refined based on actual context
        return event.turn_number == 1 and event.is_final
