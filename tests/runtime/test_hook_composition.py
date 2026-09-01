# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for composable, task-local instrumentation hooks."""

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from nooa.runtime.hooks import (
    CompositeInstrumentationHooks,
    call_after_hook,
    call_before_hook,
    compose_hooks,
    get_hooks,
    hooks_scope,
    set_hooks,
)


class RecordingHooks:
    def __init__(self, name: str, calls: list[tuple[Any, ...]]) -> None:
        self.name = name
        self.calls = calls

    def before_generation(self, **kwargs: Any) -> str:
        context = f"{self.name}-context"
        self.calls.append((self.name, "before", kwargs["generation_id"]))
        return context

    def after_generation(self, *, context: Any, **kwargs: Any) -> None:
        self.calls.append((self.name, "after", context, kwargs["generation_id"]))

    def on_messages_built(self, **kwargs: Any) -> None:
        self.calls.append((self.name, "messages", kwargs["generation_id"]))


class FailingHooks(RecordingHooks):
    def before_generation(self, **kwargs: Any) -> str:
        raise RuntimeError("before failed")

    def after_generation(self, *, context: Any, **kwargs: Any) -> None:
        raise RuntimeError("after failed")

    def on_messages_built(self, **kwargs: Any) -> None:
        raise RuntimeError("messages failed")


@pytest.fixture(autouse=True)
def _reset_hooks():
    set_hooks(None)
    yield
    set_hooks(None)


def test_composite_preserves_order_and_pairs_child_contexts() -> None:
    calls: list[tuple[Any, ...]] = []
    first = RecordingHooks("first", calls)
    second = RecordingHooks("second", calls)
    set_hooks(CompositeInstrumentationHooks(first, second))  # type: ignore[arg-type]

    context = call_before_hook(
        "before_generation",
        agent=None,
        method_name="method",
        strategy="strategy",
        generation_id="generation-1",
        parent_generation_id=None,
    )
    call_after_hook(
        "after_generation",
        context,
        agent=None,
        method_name="method",
        result="result",
        exception=None,
        generation_id="generation-1",
    )

    assert calls == [
        ("first", "before", "generation-1"),
        ("second", "before", "generation-1"),
        ("first", "after", "first-context", "generation-1"),
        ("second", "after", "second-context", "generation-1"),
    ]


def test_failing_child_does_not_block_other_observers() -> None:
    calls: list[tuple[Any, ...]] = []
    failing = FailingHooks("failing", calls)
    healthy = RecordingHooks("healthy", calls)
    hooks = CompositeInstrumentationHooks(failing, healthy)  # type: ignore[arg-type]
    set_hooks(hooks)

    context = call_before_hook(
        "before_generation",
        agent=None,
        method_name="method",
        strategy="strategy",
        generation_id="generation-1",
        parent_generation_id=None,
    )
    call_after_hook(
        "after_generation",
        context,
        agent=None,
        method_name="method",
        result="result",
        exception=None,
        generation_id="generation-1",
    )
    hooks.on_messages_built(
        agent=None, method_name="method", messages=[], generation_id="generation-1"
    )

    assert context.children == (None, "healthy-context")
    assert calls == [
        ("healthy", "before", "generation-1"),
        ("healthy", "after", "healthy-context", "generation-1"),
        ("healthy", "messages", "generation-1"),
    ]


def test_compose_hooks_flattens_and_ignores_none() -> None:
    calls: list[tuple[Any, ...]] = []
    first = RecordingHooks("first", calls)
    second = RecordingHooks("second", calls)

    assert compose_hooks(None) is None
    assert compose_hooks(first) is first  # type: ignore[arg-type]
    combined = compose_hooks(CompositeInstrumentationHooks(first), None, second)  # type: ignore[arg-type]

    assert isinstance(combined, CompositeInstrumentationHooks)
    assert combined.hooks == (first, second)


def test_hooks_scope_restores_nested_scopes_after_exception() -> None:
    calls: list[tuple[Any, ...]] = []
    original = RecordingHooks("original", calls)
    outer = RecordingHooks("outer", calls)
    inner = RecordingHooks("inner", calls)
    set_hooks(original)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="boom"):
        with hooks_scope(outer):  # type: ignore[arg-type]
            outer_active = get_hooks()
            assert isinstance(outer_active, CompositeInstrumentationHooks)
            assert outer_active.hooks == (original, outer)
            with hooks_scope(inner, compose=False):  # type: ignore[arg-type]
                assert get_hooks() is inner
                raise RuntimeError("boom")

    assert get_hooks() is original


@pytest.mark.asyncio
async def test_hooks_scope_is_isolated_between_concurrent_tasks() -> None:
    calls: list[tuple[Any, ...]] = []
    first = RecordingHooks("first", calls)
    second = RecordingHooks("second", calls)
    ready = asyncio.Event()
    entered = 0

    async def observe(hooks: RecordingHooks) -> tuple[Any, Any]:
        nonlocal entered
        with hooks_scope(hooks, compose=False):  # type: ignore[arg-type]
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            active = get_hooks()
            await asyncio.sleep(0)
            return active, get_hooks()

    first_result, second_result = await asyncio.gather(observe(first), observe(second))

    assert first_result == (first, first)
    assert second_result == (second, second)
    assert get_hooks() is None


def test_after_uses_originating_composite_when_active_hooks_change() -> None:
    calls: list[tuple[Any, ...]] = []
    original = RecordingHooks("original", calls)
    replacement = RecordingHooks("replacement", calls)
    set_hooks(CompositeInstrumentationHooks(original))  # type: ignore[arg-type]
    context = call_before_hook(
        "before_generation",
        agent=None,
        method_name="method",
        strategy="strategy",
        generation_id="generation-1",
        parent_generation_id=None,
    )

    set_hooks(replacement)  # type: ignore[arg-type]
    call_after_hook(
        "after_generation",
        context,
        agent=None,
        method_name="method",
        result="result",
        exception=None,
        generation_id="generation-1",
    )

    assert calls == [
        ("original", "before", "generation-1"),
        ("original", "after", "original-context", "generation-1"),
    ]


def test_composite_reactivates_each_child_agent_call_context() -> None:
    events: list[tuple[str, str, Any]] = []

    class ActivatingHooks(RecordingHooks):
        def before_agent_call(self, **kwargs: Any) -> str:
            return f"{self.name}-{kwargs['call_id']}-context"

        @contextmanager
        def activate_agent_call(self, context: Any):
            events.append((self.name, "enter", context))
            try:
                yield
            finally:
                events.append((self.name, "exit", context))

    first = ActivatingHooks("first", [])
    second = ActivatingHooks("second", [])
    hooks = CompositeInstrumentationHooks(first, second)  # type: ignore[arg-type]
    context = hooks.before_agent_call(
        agent=None,
        method_name="stream",
        args=(),
        kwargs={},
        call_id="call-1",
        parent_call_id=None,
    )

    with hooks.activate_agent_call(context):
        events.append(("body", "active", None))

    assert events == [
        ("first", "enter", context.children[0]),
        ("second", "enter", context.children[1]),
        ("body", "active", None),
        ("second", "exit", context.children[1]),
        ("first", "exit", context.children[0]),
    ]


def test_after_uses_originating_single_hook_when_active_hook_changes() -> None:
    calls: list[tuple[Any, ...]] = []
    original = RecordingHooks("original", calls)
    replacement = RecordingHooks("replacement", calls)
    set_hooks(original)  # type: ignore[arg-type]
    context = call_before_hook(
        "before_generation",
        agent=None,
        method_name="method",
        strategy="strategy",
        generation_id="generation-1",
        parent_generation_id=None,
    )

    set_hooks(replacement)  # type: ignore[arg-type]
    call_after_hook(
        "after_generation",
        context,
        agent=None,
        method_name="method",
        result="result",
        exception=None,
        generation_id="generation-1",
    )

    assert calls == [
        ("original", "before", "generation-1"),
        ("original", "after", "original-context", "generation-1"),
    ]


@pytest.mark.parametrize("phase", ["before", "after", "messages"])
def test_cancelling_child_does_not_block_other_observers(phase: str) -> None:
    calls: list[tuple[Any, ...]] = []

    class CancellingHooks(RecordingHooks):
        def before_generation(self, **kwargs: Any) -> str:
            if phase == "before":
                raise asyncio.CancelledError
            return super().before_generation(**kwargs)

        def after_generation(self, *, context: Any, **kwargs: Any) -> None:
            if phase == "after":
                raise asyncio.CancelledError
            super().after_generation(context=context, **kwargs)

        def on_messages_built(self, **kwargs: Any) -> None:
            if phase == "messages":
                raise asyncio.CancelledError
            super().on_messages_built(**kwargs)

    cancelling = CancellingHooks("cancelling", calls)
    healthy = RecordingHooks("healthy", calls)
    hooks = CompositeInstrumentationHooks(cancelling, healthy)  # type: ignore[arg-type]
    set_hooks(hooks)

    context = call_before_hook(
        "before_generation",
        agent=None,
        method_name="method",
        strategy="strategy",
        generation_id="generation-1",
        parent_generation_id=None,
    )
    call_after_hook(
        "after_generation",
        context,
        agent=None,
        method_name="method",
        result="result",
        exception=None,
        generation_id="generation-1",
    )
    hooks.on_messages_built(
        agent=None, method_name="method", messages=[], generation_id="generation-1"
    )

    assert ("healthy", "before", "generation-1") in calls
    assert ("healthy", "after", "healthy-context", "generation-1") in calls
    assert ("healthy", "messages", "generation-1") in calls


@pytest.mark.parametrize(
    ("before_name", "after_name", "before_kwargs", "after_kwargs"),
    [
        (
            "before_agent_call",
            "after_agent_call",
            {
                "agent": None,
                "method_name": "method",
                "args": (),
                "kwargs": {},
                "call_id": "call",
                "parent_call_id": None,
            },
            {"agent": None, "method_name": "method", "result": None, "exception": None},
        ),
        (
            "before_code_execution",
            "after_code_execution",
            {"agent": None, "code": "pass", "execution_id": "execution", "generation_id": None},
            {
                "agent": None,
                "code": "pass",
                "result": None,
                "exception": None,
                "execution_id": "execution",
            },
        ),
        (
            "before_method_invocation",
            "after_method_invocation",
            {
                "agent": None,
                "method_name": "method",
                "args": (),
                "kwargs": {},
                "invocation_id": "invocation",
            },
            {
                "agent": None,
                "method_name": "method",
                "result": None,
                "exception": None,
                "invocation_id": "invocation",
            },
        ),
        (
            "before_tool_execution",
            "after_tool_execution",
            {
                "agent": None,
                "tool_name": "tool",
                "arguments": {},
                "execution_id": "execution",
                "generation_id": None,
            },
            {
                "agent": None,
                "tool_name": "tool",
                "arguments": {},
                "result": None,
                "exception": None,
                "execution_id": "execution",
            },
        ),
    ],
)
def test_all_composite_callback_pairs_forward_their_own_context(
    before_name: str,
    after_name: str,
    before_kwargs: dict[str, Any],
    after_kwargs: dict[str, Any],
) -> None:
    first = MagicMock()
    second = MagicMock()
    getattr(first, before_name).return_value = "first-context"
    getattr(second, before_name).return_value = "second-context"
    composite = CompositeInstrumentationHooks(first, second)

    context = getattr(composite, before_name)(**before_kwargs)
    getattr(composite, after_name)(context=context, **after_kwargs)

    getattr(first, before_name).assert_called_once_with(**before_kwargs)
    getattr(second, before_name).assert_called_once_with(**before_kwargs)
    getattr(first, after_name).assert_called_once_with(context="first-context", **after_kwargs)
    getattr(second, after_name).assert_called_once_with(context="second-context", **after_kwargs)
