# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context_builder.py — the 6-phase context pipeline.

Covers:
- _apply_overrides (batched override logic)
- Each pipeline phase in isolation
- Full build_context() pipeline
- BuildResult structure (blocks + resolved_cache separation)
- Error handling (DynamicContext expression failures shown inline)
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from nooa.context_blocks import DynamicContext, ResolvedBlock, Role
from nooa.context_blocks.events import ToolCallEvent, ToolResult, UserEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(key: str, content: str = "content", role: Role = Role.SYSTEM) -> ResolvedBlock:
    """Shorthand for creating a ResolvedBlock in tests."""
    return ResolvedBlock(key=key, content=content, role=role)


async def _identity_resolve(key: str, value: str | DynamicContext) -> str:
    """Resolve function that returns the value as-is (for static) or expr (for DynamicContext)."""
    if isinstance(value, DynamicContext):
        return f"<resolved:{value.expr}>"
    return value


def _make_context_manager(
    blocks: dict[str, Any] | None = None,
    protected_blocks: dict[str, Any] | None = None,
) -> Any:
    """Create a minimal ContextManager object for testing."""
    from nooa.runtime.context_manager import ContextManager

    cm = ContextManager()
    if protected_blocks:
        for key, value in protected_blocks.items():
            if isinstance(value, DynamicContext):
                cm.set_static_protected(key, expr=value.expr)
            else:
                cm.set_static_protected(key, value)
    if blocks:
        for key, value in blocks.items():
            if isinstance(value, DynamicContext):
                cm.set_dynamic(key, value.expr)
            else:
                cm[key] = value
    return cm


def _make_event_manager(events: list | None = None) -> MagicMock:
    """Create a mock event manager that returns the given events from values()."""
    em = MagicMock()
    em.filter.return_value = events or []
    em.values.return_value = events or []
    return em


# ---------------------------------------------------------------------------
# Tests: _apply_overrides (shared override logic)
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    """Tests for _apply_overrides — shared by strategy, decorator, and scoped phases."""

    @pytest.mark.asyncio
    async def test_static_override_appends(self):
        from nooa.runtime.context_builder import _apply_overrides

        blocks = [_block("a")]
        result = await _apply_overrides(
            blocks, {"b": "content_b"}, _identity_resolve, static_expr=lambda k: f"test.{k}"
        )
        assert len(result) == 2
        assert result[-1].key == "b"
        assert result[-1].metadata.expr == "test.b"

    @pytest.mark.asyncio
    async def test_dynamic_override_uses_expr(self):
        from nooa.runtime.context_builder import _apply_overrides

        blocks = [_block("a")]
        result = await _apply_overrides(
            blocks,
            {"x": DynamicContext("self.x()")},
            _identity_resolve,
            static_expr=lambda k: "unused",
        )
        assert result[-1].metadata.expr == "self.x()"

    @pytest.mark.asyncio
    async def test_none_removes(self):
        from nooa.runtime.context_builder import _apply_overrides

        blocks = [_block("a"), _block("b")]
        result = await _apply_overrides(
            blocks, {"a": None}, _identity_resolve, static_expr=lambda k: k
        )
        assert [b.key for b in result] == ["b"]

    @pytest.mark.asyncio
    async def test_replaces_existing_key(self):
        from nooa.runtime.context_builder import _apply_overrides

        blocks = [_block("a", "old")]
        result = await _apply_overrides(
            blocks, {"a": "new"}, _identity_resolve, static_expr=lambda k: k
        )
        assert len(result) == 1
        assert result[0].content == "new"

    @pytest.mark.asyncio
    async def test_does_not_mutate_input(self):
        from nooa.runtime.context_builder import _apply_overrides

        original = [_block("a")]
        original_copy = list(original)
        await _apply_overrides(original, {"b": "new"}, _identity_resolve, static_expr=lambda k: k)
        assert original == original_copy


# ---------------------------------------------------------------------------
# Tests: BuildResult
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_stores_blocks_and_cache(self):
        from nooa.runtime.context_builder import BuildResult

        blocks = [_block("a")]
        cache = {"x": "resolved_x"}
        result = BuildResult(blocks=blocks, resolved_cache=cache)

        assert result.blocks is blocks
        assert result.resolved_cache is cache


# ---------------------------------------------------------------------------
# Tests: Phase 1 — Protected (framework) blocks via context_manager
# ---------------------------------------------------------------------------


def _get_framework_context_manager():
    """Get a ContextManager with default framework blocks registered as protected."""
    return _make_context_manager(
        protected_blocks={
            "system_prompt": DynamicContext("self._system_prompt()"),
            "self": DynamicContext("doc(type(self))"),
            "state": DynamicContext("pformat(self, max_length=50, max_string=500, max_depth=4)"),
        }
    )


class TestProtectedFrameworkBlocks:
    """Tests for protected (framework) blocks emitted by _phase_persistent_blocks."""

    @pytest.mark.asyncio
    async def test_adds_framework_blocks(self):
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _get_framework_context_manager()
        result, _ = await _phase_persistent_blocks([], cm, resolve_fn=_identity_resolve)

        keys = [b.key for b in result]
        assert "system_prompt" in keys
        assert "self" in keys

    @pytest.mark.asyncio
    async def test_framework_blocks_not_user_blocks(self):
        """Protected blocks should have user_block=False so they survive truncation."""
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _get_framework_context_manager()
        result, _ = await _phase_persistent_blocks([], cm, resolve_fn=_identity_resolve)

        for block in result:
            assert block.metadata is not None
            assert block.metadata.user_block is False, f"{block.key} should not be a user_block"

    @pytest.mark.asyncio
    async def test_empty_resolved_content_still_renders(self):
        """Empty string is valid content — blocks are included."""
        from nooa.runtime.context_builder import _phase_persistent_blocks

        async def empty_resolve(key: str, value: str | DynamicContext) -> str:
            return ""

        cm = _get_framework_context_manager()
        result, _ = await _phase_persistent_blocks([], cm, resolve_fn=empty_resolve)
        assert len(result) > 0
        assert all(b.content == "" for b in result)

    @pytest.mark.asyncio
    async def test_does_not_mutate_input(self):
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _get_framework_context_manager()
        original: list[ResolvedBlock] = []
        await _phase_persistent_blocks(original, cm, resolve_fn=_identity_resolve)
        assert len(original) == 0


# ---------------------------------------------------------------------------
# Tests: Phase 2 — Persistent blocks
# ---------------------------------------------------------------------------


class TestPhasePersistentBlocks:
    @pytest.mark.asyncio
    async def test_adds_static_blocks(self):
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _make_context_manager({"notes": "My notes"})
        blocks, cache = await _phase_persistent_blocks([], cm, _identity_resolve)

        assert len(blocks) == 1
        assert blocks[0].key == "notes"
        assert blocks[0].content == "My notes"
        # Static values are NOT in the resolved cache (read directly from _blocks)
        assert cache == {}

    @pytest.mark.asyncio
    async def test_adds_dynamic_blocks(self):
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _make_context_manager({"status": DynamicContext("self.get_status()")})
        blocks, cache = await _phase_persistent_blocks([], cm, _identity_resolve)

        assert len(blocks) == 1
        assert blocks[0].key == "status"
        assert "resolved:" in blocks[0].content
        assert blocks[0].metadata.expr == "self.get_status()"

    @pytest.mark.asyncio
    async def test_pprints_non_string_values(self):
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _make_context_manager({"data": {"key": "value"}})
        blocks, cache = await _phase_persistent_blocks([], cm, _identity_resolve)

        assert len(blocks) == 1
        assert "key" in blocks[0].content
        assert "value" in blocks[0].content

    @pytest.mark.asyncio
    async def test_empty_string_renders_as_block(self):
        """Empty strings are valid content — only None is skipped."""
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _make_context_manager({"empty": ""})
        blocks, cache = await _phase_persistent_blocks([], cm, _identity_resolve)

        assert len(blocks) == 1
        assert blocks[0].key == "empty"
        assert blocks[0].content == ""

    @pytest.mark.asyncio
    async def test_none_dynamic_values_render_as_none(self):
        """DynamicContext expressions that resolve to None render as 'None' (not skipped)."""
        from nooa.runtime.context_builder import _phase_persistent_blocks

        async def resolve_to_none(key, value):
            return None

        cm = _make_context_manager({"gone": DynamicContext("self.missing")})
        blocks, cache = await _phase_persistent_blocks([], cm, resolve_to_none)

        assert len(blocks) == 1
        assert blocks[0].key == "gone"
        assert blocks[0].content == "None"

    @pytest.mark.asyncio
    async def test_resolved_cache_returned_separately(self):
        from nooa.runtime.context_builder import _phase_persistent_blocks

        cm = _make_context_manager({"a": "val_a", "b": "val_b"})
        blocks, cache = await _phase_persistent_blocks([], cm, _identity_resolve)

        # Static values are NOT in cache (read directly from _blocks)
        assert cache == {}
        # Verify context_manager was NOT mutated (static values live in _blocks)
        assert cm["a"] == "val_a"


# ---------------------------------------------------------------------------
# Tests: Phase 3 — Strategy overrides
# ---------------------------------------------------------------------------


class TestPhaseStrategyOverrides:
    @pytest.mark.asyncio
    async def test_no_strategy_returns_unchanged(self):
        from nooa.runtime.context_builder import _phase_strategy_overrides

        blocks = [_block("a")]
        result = await _phase_strategy_overrides(blocks, None, _identity_resolve)
        assert result is blocks

    @pytest.mark.asyncio
    async def test_replaces_existing_block(self):
        from nooa.runtime.context_builder import _phase_strategy_overrides

        blocks = [_block("prompt", "original")]
        strategy = MagicMock()
        strategy.get_block_overrides.return_value = {"prompt": "overridden"}

        result = await _phase_strategy_overrides(blocks, strategy, _identity_resolve)
        assert len(result) == 1
        assert result[0].content == "overridden"

    @pytest.mark.asyncio
    async def test_appends_new_block(self):
        from nooa.runtime.context_builder import _phase_strategy_overrides

        blocks = [_block("a")]
        strategy = MagicMock()
        strategy.get_block_overrides.return_value = {"new_block": "content"}

        result = await _phase_strategy_overrides(blocks, strategy, _identity_resolve)
        assert len(result) == 2
        assert result[-1].key == "new_block"

    @pytest.mark.asyncio
    async def test_none_removes_block(self):
        from nooa.runtime.context_builder import _phase_strategy_overrides

        blocks = [_block("a"), _block("remove_me")]
        strategy = MagicMock()
        strategy.get_block_overrides.return_value = {"remove_me": None}

        result = await _phase_strategy_overrides(blocks, strategy, _identity_resolve)
        assert len(result) == 1
        assert result[0].key == "a"

    @pytest.mark.asyncio
    async def test_dynamic_override_resolved(self):
        """Strategy overrides with DynamicContext values are resolved and get correct metadata."""
        from nooa.runtime.context_builder import _phase_strategy_overrides

        blocks = [_block("prompt", "original")]
        strategy = MagicMock()
        strategy.get_block_overrides.return_value = {
            "prompt": DynamicContext("strategy.strategy_instructions(runtime)")
        }

        result = await _phase_strategy_overrides(blocks, strategy, _identity_resolve)
        assert len(result) == 1
        assert result[0].content == "<resolved:strategy.strategy_instructions(runtime)>"
        assert result[0].metadata.expr == "strategy.strategy_instructions(runtime)"


# ---------------------------------------------------------------------------
# Tests: Phase 3b — Decorator context
# ---------------------------------------------------------------------------


class TestPhaseDecoratorContext:
    @pytest.mark.asyncio
    async def test_no_decorator_context_returns_unchanged(self):
        from nooa.runtime.context_builder import _phase_decorator_context

        blocks = [_block("a")]
        result = await _phase_decorator_context(blocks, None, _identity_resolve)
        assert result is blocks

    @pytest.mark.asyncio
    async def test_applies_decorator_overrides(self):
        from nooa.runtime.context_builder import _phase_decorator_context

        blocks = [_block("a")]
        result = await _phase_decorator_context(
            blocks, {"focus": "security analysis"}, _identity_resolve
        )
        assert len(result) == 2
        assert result[-1].key == "focus"
        assert result[-1].content == "security analysis"

    @pytest.mark.asyncio
    async def test_none_removes_block(self):
        from nooa.runtime.context_builder import _phase_decorator_context

        blocks = [_block("a")]
        result = await _phase_decorator_context(blocks, {"a": None}, _identity_resolve)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_dynamic_values_resolved(self):
        """DynamicContext values in decorator context are resolved and get correct metadata."""
        from nooa.runtime.context_builder import _phase_decorator_context

        blocks = [_block("a")]
        result = await _phase_decorator_context(
            blocks, {"focus": DynamicContext("self.get_focus()")}, _identity_resolve
        )
        assert len(result) == 2
        assert result[-1].key == "focus"
        assert result[-1].content == "<resolved:self.get_focus()>"
        assert result[-1].metadata.expr == "self.get_focus()"


# ---------------------------------------------------------------------------
# Tests: Phase 4 — Scoped blocks
# ---------------------------------------------------------------------------


class TestPhaseScopedBlocks:
    @pytest.mark.asyncio
    async def test_no_scoped_blocks_returns_unchanged(self):
        from nooa.runtime.context_builder import _phase_scoped_blocks

        blocks = [_block("a")]
        result = await _phase_scoped_blocks(blocks, None, _identity_resolve)
        assert result is blocks

    @pytest.mark.asyncio
    async def test_applies_scoped_overrides(self):
        from nooa.runtime.context_builder import _phase_scoped_blocks

        blocks = [_block("a")]
        result = await _phase_scoped_blocks(blocks, {"focus": "perf testing"}, _identity_resolve)
        assert len(result) == 2
        assert result[-1].key == "focus"
        assert result[-1].content == "perf testing"

    @pytest.mark.asyncio
    async def test_none_removes_block(self):
        from nooa.runtime.context_builder import _phase_scoped_blocks

        blocks = [_block("a")]
        result = await _phase_scoped_blocks(blocks, {"a": None}, _identity_resolve)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Phase 5 — Events
# ---------------------------------------------------------------------------


class TestPhaseEvents:
    def test_empty_events(self):
        from nooa.runtime.context_builder import _phase_events

        em = _make_event_manager([])
        result = _phase_events([], em)
        assert result == []

    def test_user_event(self):
        from nooa.runtime.context_builder import _phase_events

        event = UserEvent(content="Hello", tag="1")
        em = _make_event_manager([event])
        result = _phase_events([], em)

        assert len(result) == 1
        assert result[0].key == "event_1"
        assert result[0].role == Role.USER
        assert result[0].content == ""  # deferred — serialized at render time
        assert result[0].event is event  # raw event carried through

    def test_tool_call_event(self):
        from nooa.runtime.context_builder import _phase_events

        event = ToolCallEvent(
            tool_call_id="tc_1",
            name="search",
            arguments={"query": "test"},
            tag="2",
            result=ToolResult(tool_call_id="tc_1", content="found it"),
        )
        em = _make_event_manager([event])
        result = _phase_events([], em)

        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        assert result[0].content == ""
        assert result[0].event is event
        assert isinstance(result[0].event, ToolCallEvent)
        assert result[0].event.name == "search"
        assert result[0].event.result is not None
        assert result[0].event.result.content == "found it"

    def test_tool_call_without_result(self):
        from nooa.runtime.context_builder import _phase_events

        event = ToolCallEvent(
            tool_call_id="tc_1",
            name="search",
            arguments={"query": "test"},
            tag="3",
        )
        em = _make_event_manager([event])
        result = _phase_events([], em)

        assert result[0].event is event
        assert isinstance(result[0].event, ToolCallEvent)
        assert result[0].event.result is None

    def test_does_not_mutate_input_blocks(self):
        from nooa.runtime.context_builder import _phase_events

        original = [_block("existing")]
        original_copy = list(original)
        event = UserEvent(content="Hi", tag="1")
        em = _make_event_manager([event])
        result = _phase_events(original, em)

        assert len(original) == 1
        assert original == original_copy
        assert len(result) == 2

    def test_preserves_existing_blocks(self):
        from nooa.runtime.context_builder import _phase_events

        existing = [_block("system_prompt", "You are helpful.")]
        event = UserEvent(content="Hi", tag="1")
        em = _make_event_manager([event])
        result = _phase_events(existing, em)

        assert result[0].key == "system_prompt"
        assert result[1].key == "event_1"

    def test_tag_none_falls_back_to_id(self):
        """When event.tag is None, event.id is used as the key suffix."""
        from nooa.runtime.context_builder import _phase_events

        event = UserEvent(content="Hello")
        event.tag = None
        event.id = "evt_42"
        em = _make_event_manager([event])
        result = _phase_events([], em)

        assert len(result) == 1
        assert result[0].key == "event_evt_42"
        assert result[0].metadata.tag == "evt_42"

    def test_non_tool_event_carries_original_event(self):
        """Non-tool events also carry the original event on block.event."""
        from nooa.runtime.context_builder import _phase_events

        event = UserEvent(content="Hello world", tag="1")
        em = _make_event_manager([event])
        result = _phase_events([], em)

        assert len(result) == 1
        assert result[0].event is event
        assert result[0].content == ""  # Deferred — serialized at render time

    def test_current_call_query_keeps_task_event(self):
        """EventQuery.current_call() must keep the task so LLM gets system + task.

        When decorator uses ScopedContext(events=EventQuery.current_call()),
        _phase_events filters by current_call_id. The task (user message) for
        this call must be included so the prompt has at least system + task.
        """
        from nooa.events import Task
        from nooa.runtime.context_builder import _phase_events
        from nooa.runtime.event_query import EventQuery

        current_id = "call-xyz-456"
        task = Task(prompt="Classify sentiment of: Great product!")
        task.metadata["call_id"] = current_id
        task.tag = "1"
        em = _make_event_manager([task])

        result = _phase_events(
            [],
            em,
            decorator_event_query=EventQuery.current_call(),
            current_call_id=current_id,
        )
        assert len(result) >= 1, "must include task so LLM gets system + task message"
        assert result[0].role == Role.USER
        assert result[0].content == ""  # deferred — serialized at render time
        assert result[0].event is task  # raw event carried through


# ---------------------------------------------------------------------------
# Tests: Phase 2 — DynamicContext resolution failure
# ---------------------------------------------------------------------------


class TestPhasePersistentBlocksErrors:
    @pytest.mark.asyncio
    async def test_dynamic_resolution_failure_displays_inline(self):
        """When resolve_fn returns an error string, it appears in the block content.

        In production, the resolve function catches exceptions and formats them
        as "ExceptionType: message" (no traceback, since the source is a short
        expression). The builder renders these as normal block content so the
        LLM can see and diagnose the problem.
        """
        from nooa.runtime.context_builder import _phase_persistent_blocks

        async def error_showing_resolve(key: str, value: str | DynamicContext) -> str:
            if isinstance(value, DynamicContext):
                # Simulate IPython-style error formatting (matching format_error_for_llm)
                return "NameError: name 'broken' is not defined"
            return value

        cm = _make_context_manager({"status": DynamicContext("self.broken()")})
        blocks, cache = await _phase_persistent_blocks([], cm, error_showing_resolve)

        assert len(blocks) == 1
        assert blocks[0].key == "status"
        assert "NameError:" in blocks[0].content


# ---------------------------------------------------------------------------
# Tests: Full pipeline (build_context)
# ---------------------------------------------------------------------------


class TestBuildContext:
    @pytest.mark.asyncio
    async def test_returns_build_result(self):
        from nooa.runtime.context_builder import BuildResult, build_context

        cm = _make_context_manager({"notes": "My notes"})
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
        )

        assert isinstance(result, BuildResult)
        assert isinstance(result.blocks, list)
        assert isinstance(result.resolved_cache, dict)

    @pytest.mark.asyncio
    async def test_includes_framework_and_persistent_blocks(self):
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager(
            blocks={"notes": "My notes"},
            protected_blocks={
                "system_prompt": DynamicContext("self._system_prompt()"),
                "self": DynamicContext("doc(type(self))"),
            },
        )
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
        )

        keys = [b.key for b in result.blocks]
        assert "system_prompt" in keys
        assert "self" in keys
        assert "notes" in keys

    @pytest.mark.asyncio
    async def test_does_not_mutate_context_manager(self):
        """build_context must not call _update_resolved — that's the caller's job."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"notes": "My notes"})
        original_cache = dict(cm._dynamic_cache)
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
        )

        # Static blocks don't go into resolved_cache (read directly from _blocks)
        assert result.resolved_cache == {}
        # ContextApi._dynamic_cache should not have been updated by build_context
        assert cm._dynamic_cache == original_cache

    @pytest.mark.asyncio
    async def test_strategy_overrides_applied(self):
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"notes": "original notes"})
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"notes": "overridden notes"}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
        )

        notes_blocks = [b for b in result.blocks if b.key == "notes"]
        assert len(notes_blocks) == 1
        assert notes_blocks[0].content == "overridden notes"

    @pytest.mark.asyncio
    async def test_events_included(self):
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager()
        event = UserEvent(content="Hello", tag="1")
        em = _make_event_manager([event])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
        )

        event_blocks = [b for b in result.blocks if b.key.startswith("event_")]
        assert len(event_blocks) == 1
        assert event_blocks[0].content == ""  # deferred — serialized at render time
        assert event_blocks[0].event is event  # raw event carried through

    @pytest.mark.asyncio
    async def test_decorator_context_applied(self):
        """build_context applies decorator_context overrides."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"notes": "original"})
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
            decorator_context={"focus": "security", "notes": "overridden"},
        )

        keys = [b.key for b in result.blocks]
        assert "focus" in keys
        notes = [b for b in result.blocks if b.key == "notes"]
        assert len(notes) == 1
        assert notes[0].content == "overridden"

    @pytest.mark.asyncio
    async def test_strategy_overrides_framework_block(self):
        """Strategy overrides can replace protected blocks (e.g. system_prompt).

        Verifies no duplicate keys: strategy phase replaces the persistent block in-place.
        """
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager(
            protected_blocks={
                "system_prompt": DynamicContext("self._system_prompt()"),
            },
        )
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"system_prompt": "custom prompt"}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
        )

        prompt_blocks = [b for b in result.blocks if b.key == "system_prompt"]
        assert len(prompt_blocks) == 1, "Expected exactly one system_prompt block (no duplicates)"
        assert prompt_blocks[0].content == "custom prompt"

    @pytest.mark.asyncio
    async def test_scoped_overrides_persistent_block(self):
        """Scoped blocks (Phase 4) can override persistent blocks (Phase 2).

        Verifies no duplicate keys across phases.
        """
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"notes": "original"})
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
            scoped_context={"notes": "scoped override"},
        )

        notes_blocks = [b for b in result.blocks if b.key == "notes"]
        assert len(notes_blocks) == 1, "Expected exactly one notes block (no duplicates)"
        assert notes_blocks[0].content == "scoped override"

    @pytest.mark.asyncio
    async def test_keyword_only_args(self):
        """build_context uses keyword-only args — positional call should fail."""
        from nooa.runtime.context_builder import build_context

        with pytest.raises(TypeError):
            await build_context(
                _make_context_manager(),  # type: ignore[reportCallIssue]
                _make_event_manager(),
                None,
                _identity_resolve,
            )


# ---------------------------------------------------------------------------
# Tests: Override phase interactions (#11)
# ---------------------------------------------------------------------------


class TestOverridePhaseInteractions:
    """Comprehensive tests for override priority and interaction across phases.

    Pipeline order (later phases override earlier):
    1. Persistent blocks (protected framework blocks + user blocks)
    2. Strategy overrides (strategy.get_block_overrides())
    3. Decorator context (@strategy(context={...}))
    4. Scoped blocks (with scoped_blocks(...))

    These tests verify:
    - Override priority is correct (later wins)
    - No duplicate keys across phases
    - Complex multi-phase override scenarios
    - Remove (None) semantics work correctly
    """

    @pytest.mark.asyncio
    async def test_persistent_coexists_with_framework(self):
        """User blocks coexist with protected framework blocks in context_manager.

        Protected blocks can't be overwritten by the LLM-facing API.
        User blocks add new keys alongside them.
        """
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager(
            blocks={"custom_key": "custom value"},
            protected_blocks={
                "system_prompt": DynamicContext("self._system_prompt()"),
            },
        )
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
        )

        keys = [b.key for b in result.blocks]
        assert "system_prompt" in keys, "Protected block present"
        assert "custom_key" in keys, "User block present"

    @pytest.mark.asyncio
    async def test_strategy_overrides_persistent(self):
        """Strategy overrides (Phase 3) override persistent blocks (Phase 2)."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"notes": "persistent notes"})
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"notes": "strategy notes"}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
        )

        notes_blocks = [b for b in result.blocks if b.key == "notes"]
        assert len(notes_blocks) == 1, "No duplicates"
        assert notes_blocks[0].content == "strategy notes"

    @pytest.mark.asyncio
    async def test_decorator_overrides_strategy(self):
        """Decorator context (Phase 4) overrides strategy (Phase 3)."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager()
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"focus": "strategy focus"}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
            decorator_context={"focus": "decorator focus"},
        )

        focus_blocks = [b for b in result.blocks if b.key == "focus"]
        assert len(focus_blocks) == 1, "No duplicates"
        assert focus_blocks[0].content == "decorator focus"

    @pytest.mark.asyncio
    async def test_scoped_overrides_decorator(self):
        """Scoped blocks (Phase 5) override decorator context (Phase 4)."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager()
        em = _make_event_manager([])
        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=None,
            resolve_fn=_identity_resolve,
            decorator_context={"task": "decorator task"},
            scoped_context={"task": "scoped task"},
        )

        task_blocks = [b for b in result.blocks if b.key == "task"]
        assert len(task_blocks) == 1, "No duplicates"
        assert task_blocks[0].content == "scoped task"

    @pytest.mark.asyncio
    async def test_full_cascade_override(self):
        """Test a single block overridden by all phases in sequence.

        Start: framework "focus" block
        Phase 2: persistent overrides it
        Phase 3: strategy overrides it
        Phase 4: decorator overrides it
        Phase 5: scoped overrides it
        Result: scoped wins
        """
        from nooa.runtime.context_builder import build_context

        # Mock framework blocks with a "focus" block
        async def resolve_with_focus(key: str, value: str | DynamicContext) -> str:
            if key == "focus" and isinstance(value, DynamicContext):
                return "framework focus"
            return await _identity_resolve(key, value)

        cm = _make_context_manager({"focus": "persistent focus"})
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"focus": "strategy focus"}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=resolve_with_focus,
            decorator_context={"focus": "decorator focus"},
            scoped_context={"focus": "scoped focus"},
        )

        focus_blocks = [b for b in result.blocks if b.key == "focus"]
        assert len(focus_blocks) == 1, "Exactly one focus block (no duplicates)"
        assert focus_blocks[0].content == "scoped focus", "Scoped (Phase 5) wins"

    @pytest.mark.asyncio
    async def test_remove_semantics_across_phases(self):
        """Setting a block to None removes it, even if defined in earlier phases."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"notes": "persistent notes", "task": "persistent task"})
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"notes": None}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
            decorator_context={"task": None},
        )

        keys = [b.key for b in result.blocks]
        assert "notes" not in keys, "Strategy removed notes"
        assert "task" not in keys, "Decorator removed task"

    @pytest.mark.asyncio
    async def test_disabled_keys_suppress_all_block_sources(self):
        """Disabled keys are omitted from persistent, strategy, decorator, and scoped phases."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager(
            {"notes": "persistent notes", "task": "persistent task"},
            protected_blocks={"self": DynamicContext("doc(type(self))")},
        )
        cm.disable("self", "notes", "strategy_prompt", "decorator_key", "scoped_key")
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.get_block_overrides.return_value = {
            "strategy_prompt": "strategy prompt",
            "notes": "strategy notes",
        }

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
            decorator_context={"decorator_key": "decorator", "task": "decorator task"},
            scoped_context={"scoped_key": "scoped"},
        )

        by_key = {b.key: b.content for b in result.blocks}
        assert "self" not in by_key
        assert "notes" not in by_key
        assert "strategy_prompt" not in by_key
        assert "decorator_key" not in by_key
        assert "scoped_key" not in by_key
        assert by_key["task"] == "decorator task"

    @pytest.mark.asyncio
    async def test_multiple_independent_overrides(self):
        """Each phase can add new blocks that don't conflict."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"persistent_key": "persistent"})
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"strategy_key": "strategy"}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
            decorator_context={"decorator_key": "decorator"},
            scoped_context={"scoped_key": "scoped"},
        )

        keys = [b.key for b in result.blocks]
        assert "persistent_key" in keys
        assert "strategy_key" in keys
        assert "decorator_key" in keys
        assert "scoped_key" in keys

    @pytest.mark.asyncio
    async def test_dynamic_overrides_in_phases(self):
        """DynamicContext values work correctly across all phases."""
        from nooa.runtime.context_builder import build_context

        cm = _make_context_manager({"a": DynamicContext("self.dynamic_a()")})
        em = _make_event_manager([])
        strategy = MagicMock()
        strategy.wants_framework_block.return_value = True
        strategy.get_block_overrides.return_value = {"b": DynamicContext("self.dynamic_b()")}

        result = await build_context(
            context_manager=cm,
            event_manager=em,
            strategy=strategy,
            resolve_fn=_identity_resolve,
            decorator_context={"c": DynamicContext("self.dynamic_c()")},
            scoped_context={"d": DynamicContext("self.dynamic_d()")},
        )

        # Verify all DynamicContext blocks were resolved
        a_blocks = [b for b in result.blocks if b.key == "a"]
        b_blocks = [b for b in result.blocks if b.key == "b"]
        c_blocks = [b for b in result.blocks if b.key == "c"]
        d_blocks = [b for b in result.blocks if b.key == "d"]

        assert a_blocks[0].content == "<resolved:self.dynamic_a()>"
        assert b_blocks[0].content == "<resolved:self.dynamic_b()>"
        assert c_blocks[0].content == "<resolved:self.dynamic_c()>"
        assert d_blocks[0].content == "<resolved:self.dynamic_d()>"


# ---------------------------------------------------------------------------
# Phase 7 and Phase 8 (decorator events and scoped events injection) have been removed.
# Events are now filtered directly in _phase_events() using EventQuery with 4-level priority:
# 1. Runtime (event_manager.set_event_query()) - highest
# 2. Scoped (with ScopedContext(events=...)) - high
# 3. Decorator (@strategy(ScopedContext(events=...))) - medium
# 4. Agent (agent-level default) - low
# 5. No filter (show all events) - default
#
# See tests/runtime/test_context_integration.py for EventQuery integration tests.
