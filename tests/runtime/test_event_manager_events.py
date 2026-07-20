# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventManager event bus pattern."""

from unittest.mock import MagicMock

from nooa import Agent
from nooa.events import LLMOutput, Task
from nooa.runtime.event_manager import EventManager
from nooa.runtime.events import EventsApi
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


class _TestAgent(Agent, llm=_LLM):
    pass


def _format_events_for_test(events: list, *, last_n: int | None = None) -> list[dict]:
    """Test helper: format events to OpenAI message format."""
    if last_n:
        events = events[-last_n:]

    result = []
    for event in events:
        # Get role from class attribute
        role = event._role.value

        # Extract content from known content fields
        content = ""
        if hasattr(event, "content"):
            content = event.content
        elif hasattr(event, "prompt"):
            content = event.prompt
        if not isinstance(content, str):
            content = str(content) if content else ""

        tag = event.metadata.get("tag")
        if tag:
            content = f"[{tag}] {content}"

        result.append({"role": role, "content": content})

    return result


class TestEventManagerAdd:
    """Tests for EventManager.add() method."""

    def test_add_records_event_by_default(self):
        """add() should record event to history by default."""
        manager = EventManager()
        event = Task(prompt="Hello")

        manager.add(event)

        assert len(manager) == 1
        assert manager.values()[0].prompt == "Hello"

    def test_add_with_record_false_skips_storage(self):
        """add(record=False) should not store in history."""
        manager = EventManager()
        event = Task(prompt="Ephemeral")

        manager.add(event, record=False)

        assert len(manager) == 0

    def test_add_emits_to_handlers(self):
        """add() should emit to registered handlers."""
        manager = EventManager()
        handler = MagicMock()
        manager.on("Task", handler)

        event = Task(prompt="Hello")
        manager.add(event)

        handler.assert_called_once()
        call_args = handler.call_args[0][0]
        assert call_args.prompt == "Hello"

    def test_add_emits_even_when_not_recorded(self):
        """add(record=False) should still emit to handlers."""
        manager = EventManager()
        handler = MagicMock()
        manager.on("Task", handler)

        event = Task(prompt="Ephemeral")
        manager.add(event, record=False)

        handler.assert_called_once()

    def test_add_with_metadata(self):
        """Events preserve metadata set after construction."""
        manager = EventManager()
        event = Task(prompt="Response")
        event.metadata["source"] = "llm"
        event.metadata["call_id"] = "call_123"

        manager.add(event)

        assert manager.values()[0].metadata.get("source") == "llm"
        assert manager.values()[0].metadata.get("call_id") == "call_123"


class TestEventManagerOn:
    """Tests for EventManager.on() method."""

    def test_on_registers_handler(self):
        """on() should register handler for event type."""
        manager = EventManager()
        handler = MagicMock()

        manager.on("Task", handler)
        manager.add(Task(prompt="Test"))

        handler.assert_called_once()

    def test_on_multiple_handlers(self):
        """on() should support multiple handlers for same event type."""
        manager = EventManager()
        handler1 = MagicMock()
        handler2 = MagicMock()

        manager.on("Task", handler1)
        manager.on("Task", handler2)
        manager.add(Task(prompt="Test"))

        handler1.assert_called_once()
        handler2.assert_called_once()

    def test_on_different_event_types(self):
        """on() should dispatch to correct handlers based on event_type."""
        manager = EventManager()
        task_handler = MagicMock()
        llm_output_handler = MagicMock()

        # Register handlers by event_type
        manager.on("Task", task_handler)
        manager.on("LLMOutput", llm_output_handler)

        manager.add(Task(prompt="Question"))
        manager.add(LLMOutput(content="Answer"))  # Uses LLMOutput via alias

        task_handler.assert_called_once()
        llm_output_handler.assert_called_once()

    def test_on_returns_unsubscribe_function(self):
        """on() should return function to unsubscribe."""
        manager = EventManager()
        handler = MagicMock()

        unsubscribe = manager.on("Task", handler)
        unsubscribe()

        manager.add(Task(prompt="Test"))
        handler.assert_not_called()

    def test_on_wildcard_receives_all_events(self):
        """on('*') should receive all event types."""
        manager = EventManager()
        handler = MagicMock()

        manager.on("*", handler)

        manager.add(Task(prompt="Task"))
        manager.add(LLMOutput(content="Response"))

        assert handler.call_count == 2

    def test_collapse_emits_summary_to_handlers(self):
        """collapse() should notify ``Summary`` subscribers just like add()
        does. The TUI relies on this to surface a live line whenever a
        summarization or truncation is applied — without it, the summarizer
        is silent and pathological cascades go unnoticed.
        """
        manager = EventManager()
        handler = MagicMock()
        manager.on("Summary", handler)

        # Populate with a few events so collapse has something to fold.
        for i in range(5):
            manager.add(Task(prompt=f"Task {i}"))

        summary_tag = manager.collapse("1", "3", summary_text="Summary of 1-3")

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.summary_tag == summary_tag == "1..3"
        assert event.summary_text == "Summary of 1-3"

    def test_collapse_without_summary_text_still_emits(self):
        """Truncation (summary_text=None) should also fire the ``Summary``
        handler — callers that watch for all collapses (e.g. the TUI
        renderer's truncated-vs-summarized branches) need the signal."""
        manager = EventManager()
        handler = MagicMock()
        manager.on("Summary", handler)

        for i in range(3):
            manager.add(Task(prompt=f"Task {i}"))

        manager.collapse("1", "2")  # truncation, no summary_text

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.summary_text is None


class TestEventManagerQuery:
    """Tests for EventManager query methods."""

    def test_filter_returns_last_n_events(self):
        """filter(limit=n) should return last n events."""
        manager = EventManager()
        for i in range(5):
            manager.add(Task(prompt=f"Message {i}"))

        recent = manager.filter(limit=3)
        assert len(recent) == 3
        assert recent[0].prompt == "Message 2"
        assert recent[2].prompt == "Message 4"

    def test_filter_returns_all_if_less_than_limit(self):
        """filter(limit=n) should return all if fewer than n events."""
        manager = EventManager()
        manager.add(Task(prompt="Only one"))

        recent = manager.filter(limit=10)
        assert len(recent) == 1

    def test_filter_by_call_returns_events_with_call_id(self):
        """filter(call_id=...) should return events with that call_id."""
        manager = EventManager()

        event1 = Task(prompt="Call 1")
        event1.metadata["call_id"] = "call_1"
        manager.add(event1)

        event2 = LLMOutput(content="Response 1")
        event2.metadata["call_id"] = "call_1"
        manager.add(event2)

        event3 = Task(prompt="Call 2")
        event3.metadata["call_id"] = "call_2"
        manager.add(event3)

        call_1_events = manager.filter(call_id="call_1")
        assert len(call_1_events) == 2
        assert all(e.metadata.get("call_id") == "call_1" for e in call_1_events)


class TestCallIdAutoInjection:
    """Tests for automatic call_id injection from the agent call stack ContextVar."""

    def test_add_auto_injects_call_id_nested_stacks(self):
        """Nested call stacks should inject the innermost call_id."""
        from nooa.runtime.context_vars import _pop_agent_call_id, _push_agent_call_id

        manager = EventManager()

        _push_agent_call_id("outer-call")
        try:
            manager.add(Task(prompt="Outer event"))

            _push_agent_call_id("inner-call")
            try:
                manager.add(Task(prompt="Inner event"))
            finally:
                _pop_agent_call_id()

            # After inner pops, we're back to outer
            manager.add(Task(prompt="Back to outer"))
        finally:
            _pop_agent_call_id()

        outer_events = manager.filter(call_id="outer-call")
        assert len(outer_events) == 2
        assert outer_events[0].prompt == "Outer event"
        assert outer_events[1].prompt == "Back to outer"

        inner_events = manager.filter(call_id="inner-call")
        assert len(inner_events) == 1
        assert inner_events[0].prompt == "Inner event"

    def test_add_auto_injects_call_id_on_unrecorded_events(self):
        """call_id should be injected even for record=False events."""
        from nooa.runtime.context_vars import _pop_agent_call_id, _push_agent_call_id

        manager = EventManager()
        _push_agent_call_id("call-xyz")
        try:
            event = Task(prompt="Not recorded")
            manager.add(event, record=False)
            # call_id is injected before the record decision
            assert event.metadata["call_id"] == "call-xyz"
        finally:
            _pop_agent_call_id()


class TestEventsViewCallId:
    """Tests that the Events view delegates call_id filtering to EventManager."""

    def test_events_view_filter_by_call_id(self):
        """Events.filter(call_id=...) delegates to EventManager."""
        from nooa.runtime.context_vars import _pop_agent_call_id, _push_agent_call_id

        agent = _TestAgent()

        _push_agent_call_id("call-aaa")
        try:
            agent.event_manager.add(Task(prompt="Task A"))
        finally:
            _pop_agent_call_id()

        _push_agent_call_id("call-bbb")
        try:
            agent.event_manager.add(Task(prompt="Task B"))
        finally:
            _pop_agent_call_id()

        events = EventsApi(agent)

        results_a = events.query(call_id="call-aaa")
        assert len(results_a) == 1
        assert results_a[0].prompt == "Task A"

        results_b = events.query(call_id="call-bbb")
        assert len(results_b) == 1
        assert results_b[0].prompt == "Task B"

        # No match
        results_none = events.query(call_id="nonexistent")
        assert len(results_none) == 0


class TestPrefillCallId:
    """Tests that prefill code generation includes call_id."""

    def test_inspect_inputs_prefill_includes_call_signature(self):
        """InspectInputsPrefill generates code with an inspection comment and call signature."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.prefill import InspectInputsPrefill

        call = CurrentCall(
            id="test-call-1",
            method_name="analyze",
            decorator="plan",
            signature="(self, data: str) -> str",
            docstring="Analyze data.",
            args=(),
            kwargs={"data": "test"},
        )

        prefill = InspectInputsPrefill()
        code = prefill.get_code(call)

        assert code is not None
        assert 'print(f"Task: analyze()")' in code
        assert "Inspecting inputs for analyze()" in code

    def test_inspect_inputs_prefill_no_params_returns_none(self):
        """InspectInputsPrefill returns None when no kwargs."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.prefill import InspectInputsPrefill

        call = CurrentCall(
            id="test-call-2",
            method_name="run",
            decorator="plan",
            signature="(self) -> None",
            docstring="Run.",
            args=(),
            kwargs={},
        )

        prefill = InspectInputsPrefill()
        assert prefill.get_code(call) is None


class TestOpenAIProviderFormatter:
    """Tests for OpenAI message formatting (via helper function)."""

    def test_format_events_basic(self):
        """format_events() converts events to OpenAI format."""
        manager = EventManager()
        manager.add(Task(prompt="Hello"))
        manager.add(LLMOutput(content="Hi there"))

        messages = _format_events_for_test(manager.values())
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_format_events_with_tag(self):
        """Events with tag get [TAG] prefix in OpenAI format."""
        manager = EventManager()
        event = Task(prompt="Error occurred")
        event.metadata["tag"] = "SYSTEM"
        manager.add(event)

        messages = _format_events_for_test(manager.values())
        assert messages[0]["content"] == "[SYSTEM] Error occurred"

    def test_format_events_last_n(self):
        """format_events(last_n=N) returns only last N."""
        manager = EventManager()
        for i in range(5):
            manager.add(Task(prompt=f"Message {i}"))

        messages = _format_events_for_test(manager.values(), last_n=2)
        assert len(messages) == 2
        assert messages[0]["content"] == "Message 3"
        assert messages[1]["content"] == "Message 4"
