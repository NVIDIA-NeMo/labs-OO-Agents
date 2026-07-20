# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventManager clean API."""

import pytest

from nooa.events import Error, Feedback, LLMOutput, Task
from nooa.runtime.event_backend import InMemoryBackend, _tag_max_num
from nooa.runtime.event_manager import EventManager


def _format_events_for_test(events: list, *, last_n: int | None = None) -> list[dict]:
    """Test helper: format events to OpenAI message format.

    Extracts content from events using the new field-based approach.
    """
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

        # Apply tag prefix if present in metadata
        tag = event.metadata.get("tag")
        if tag:
            content = f"[{tag}] {content}"

        result.append({"role": role, "content": content})

    return result


def test_basic_conversation_flow():
    """Test basic task/assistant conversation."""
    hm = EventManager()

    # Add task
    hm.add(Task(prompt="Write a function to add two numbers"))

    # Add LLM response
    hm.add(LLMOutput(content="I'll write that for you"))

    # Convert to OpenAI format via formatter
    messages = _format_events_for_test(hm.values())

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert "add two numbers" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "I'll write that for you"


def test_error_feedback_flow():
    """Test error → retry flow."""
    hm = EventManager()

    # Task
    hm.add(Task(prompt="Generate code"))

    # Assistant response (with error)
    hm.add(LLMOutput(content="def foo(): syntax error"))

    # Error feedback
    hm.add(Error(content="SyntaxError: invalid syntax"))

    # Retry response
    hm.add(LLMOutput(content="def foo(): pass"))

    messages = _format_events_for_test(hm.values())
    assert len(messages) == 4
    assert messages[2]["role"] == "user"  # Error is user message
    assert "SyntaxError" in messages[2]["content"]


def test_execution_feedback_flow():
    """Test execution feedback flow."""
    hm = EventManager()

    # Task
    hm.add(Task(prompt="Solve the problem"))

    # Assistant code
    hm.add(LLMOutput(content="print('exploring')"))

    # Execution feedback
    hm.add(Feedback(content="Output:\n```\nexploring\n```\nDefine `solve` to complete."))

    messages = _format_events_for_test(hm.values())
    assert len(messages) == 3
    assert "exploring" in messages[2]["content"]
    assert "Define `solve`" in messages[2]["content"]


def test_tagged_messages():
    """Test that tags are prepended to message content."""
    hm = EventManager()

    # Add a tagged message
    event = Task(prompt="Test content")
    event.metadata["tag"] = "SYSTEM"
    hm.add(event)

    messages = _format_events_for_test(hm.values())

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "[SYSTEM] Test content"


def test_events_property():
    """Test events property returns copy."""
    hm = EventManager()
    hm.add(Task(prompt="Test"))

    events = hm.values()
    events.clear()  # Modifying copy

    # Original still has event
    assert len(hm) == 1


def test_len():
    """Test __len__ returns event count."""
    hm = EventManager()
    assert len(hm) == 0

    hm.add(Task(prompt="One"))
    assert len(hm) == 1

    hm.add(LLMOutput(content="Two"))
    assert len(hm) == 2


def test_getitem_by_tag():
    """Test indexing by string tag in the history."""
    hm = EventManager()
    hm.add(Task(prompt="First"))  # tag="1"
    hm.add(Task(prompt="Second"))  # tag="2"

    # Tags start at "1", not "0"
    assert hm["1"].prompt == "First"
    assert hm["2"].prompt == "Second"

    # Also verify active_tags
    assert hm.keys() == ["1", "2"]


def test_getitem_by_tag_after_remove():
    """Test that __getitem__ uses string tags that remain stable after remove.

    This is critical for LLM code like `self.events["5"]` to work correctly
    when events have been removed via history.remove(). String tags provide
    stable identifiers that don't shift like list indices.
    """
    hm = EventManager()

    # Add three events: tags will be "1", "2", "3"
    hm.add(Task(prompt="First"))  # tag="1"
    hm.add(Task(prompt="Second"))  # tag="2"
    hm.add(Task(prompt="Third"))  # tag="3"

    # Remove the middle event (tag="2")
    second_event = hm.values()[1]
    hm.remove(second_event.id)

    # List now has 2 items
    assert len(hm) == 2

    # String tags are stable - "1" and "3" still work
    assert hm["1"].prompt == "First"
    assert hm["3"].prompt == "Third"

    # Accessing removed tag raises KeyError

    with pytest.raises(KeyError, match="No event with tag or id '2'"):
        _ = hm["2"]


def test_clear():
    """Test clear() removes all events and resets the tag counter."""
    hm = EventManager()
    hm.add(Task(prompt="Test"))
    assert len(hm) == 1

    hm.clear()
    assert len(hm) == 0
    # Tag counter (now on the backend) must restart from 1 after clear
    tag = hm.add(Task(prompt="After clear"))
    assert tag == "1"


# === _tag_max_num and max_tag_num tests ===


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("1", 1),
        ("42", 42),
        ("2..40", 40),
        ("1..1", 1),
        ("notanumber", 0),
        ("abc..def", 0),
    ],
)
def test_tag_max_num(tag, expected):
    assert _tag_max_num(tag) == expected


def test_in_memory_backend_max_tag_num_empty():
    backend = InMemoryBackend()
    assert backend.max_tag_num() == 0


def test_in_memory_backend_max_tag_num_simple_tags():
    from nooa.events import EventBase

    backend = InMemoryBackend()
    backend.store("1", EventBase())
    backend.store("5", EventBase())
    backend.store("3", EventBase())
    assert backend.max_tag_num() == 5


def test_in_memory_backend_max_tag_num_with_range_tag():
    from nooa.events import EventBase

    backend = InMemoryBackend()
    backend.store("1", EventBase())
    backend.store("2..40", EventBase())  # range tag — end value is 40
    backend.store("41", EventBase())
    assert backend.max_tag_num() == 41


def test_event_manager_init_syncs_from_prepopulated_backend():
    """A backend that already has events must hand out tag = max+1 next."""
    from nooa.events import EventBase

    backend = InMemoryBackend()
    backend.store("1", EventBase())
    backend.store("2", EventBase())
    backend.store("3", EventBase())

    em = EventManager(backend=backend)
    assert em.add(Task(prompt="next")) == "4"


def test_event_manager_init_empty_backend_starts_at_one():
    em = EventManager()
    tag = em.add(Task(prompt="first"))
    assert tag == "1"


def test_format_events_last_n():
    """Test formatter with last_n parameter."""
    hm = EventManager()
    for i in range(5):
        hm.add(Task(prompt=f"Message {i}"))

    messages = _format_events_for_test(hm.values(), last_n=2)
    assert len(messages) == 2
    assert messages[0]["content"] == "Message 3"
    assert messages[1]["content"] == "Message 4"


def test_filter_by_query_basic():
    """Test basic keyword filter."""
    hm = EventManager()
    hm.add(Task(prompt="Find the database schema"))
    hm.add(Task(prompt="Query the user table"))
    hm.add(LLMOutput(content="Here is the schema information"))

    # Filter for "schema" should return 2 events
    results = hm.filter(query="schema")
    assert len(results) == 2

    # Case-insensitive
    results = hm.filter(query="DATABASE")
    assert len(results) == 1
    assert "database" in results[0].prompt


def test_filter_with_limit():
    """Test filter with limit parameter."""
    hm = EventManager()
    hm.add(Task(prompt="Query 1"))
    hm.add(Task(prompt="Query 2"))
    hm.add(Task(prompt="Query 3"))

    # Limit results
    results = hm.filter(query="Query", limit=2)
    assert len(results) == 2


def test_filter_no_results():
    """Test filter with no matches."""
    hm = EventManager()
    hm.add(Task(prompt="Hello world"))

    results = hm.filter(query="nonexistent")
    assert len(results) == 0


# === Collapse/Summary Tests ===


def test_collapse_basic():
    """Test basic collapse of a range of events."""
    hm = EventManager()

    # Add 5 events: tags "1", "2", "3", "4", "5"
    for i in range(5):
        hm.add(Task(prompt=f"Event {i + 1}"))

    assert hm.keys() == ["1", "2", "3", "4", "5"]

    # Collapse events 2-4
    summary_tag = hm.collapse("2", "4", summary_text="Events 2-4 summarized")

    assert summary_tag == "2..4"
    assert hm.keys() == ["1", "2..4", "5"]

    # Original events still accessible by tag
    assert hm["2"].prompt == "Event 2"
    assert hm["3"].prompt == "Event 3"
    assert hm["4"].prompt == "Event 4"

    # Summary event accessible
    summary = hm["2..4"]
    assert summary.summary_text == "Events 2-4 summarized"
    assert summary.replaced_range == (2, 4)


def test_collapse_truncation():
    """Test collapse without summary (truncation mode)."""
    hm = EventManager()

    for i in range(3):
        hm.add(Task(prompt=f"Event {i + 1}"))

    # Collapse without summary_text = truncation
    summary_tag = hm.collapse("1", "2")

    assert summary_tag == "1..2"
    summary = hm[summary_tag]
    assert summary.summary_text is None  # No summary = truncation


def test_collapse_nested_flattens():
    """Test that collapsing over an existing summary flattens the range."""
    hm = EventManager()

    # Add 10 events
    for i in range(10):
        hm.add(Task(prompt=f"Event {i + 1}"))

    # First collapse: 2-4
    hm.collapse("2", "4")
    assert hm.keys() == ["1", "2..4", "5", "6", "7", "8", "9", "10"]

    # Second collapse: 1 through 6 (includes the summary 2..4)
    summary_tag = hm.collapse("1", "6")

    # Should flatten to 1..6, not 1..2..4..6
    assert summary_tag == "1..6"
    assert hm.keys() == ["1..6", "7", "8", "9", "10"]


def test_collapse_empty_range_raises():
    """Test that collapsing an invalid range raises ValueError."""

    hm = EventManager()
    hm.add(Task(prompt="Event 1"))

    with pytest.raises(ValueError, match="No active events in range"):
        hm.collapse("5", "10")  # No events in this range


def test_items_returns_tag_event_pairs():
    """Test that items() returns (tag, event) pairs."""
    hm = EventManager()
    hm.add(Task(prompt="First"))
    hm.add(Task(prompt="Second"))

    pairs = hm.items()
    assert len(pairs) == 2
    assert pairs[0] == ("1", hm["1"])
    assert pairs[1] == ("2", hm["2"])


def test_values_returns_events_only():
    """Test that values() returns just events without tags."""
    hm = EventManager()
    hm.add(Task(prompt="First"))
    hm.add(Task(prompt="Second"))

    events = hm.values()
    assert len(events) == 2
    assert events[0].prompt == "First"
    assert events[1].prompt == "Second"


# === Tag Validation Tests ===


def test_validate_tag_valid_simple():
    """Test _validate_tag accepts valid simple numeric tags."""
    hm = EventManager()

    # These should not raise
    hm._validate_tag("1")
    hm._validate_tag("42")
    hm._validate_tag("100")


def test_validate_tag_valid_range():
    """Test _validate_tag accepts valid range tags."""
    hm = EventManager()

    # These should not raise
    hm._validate_tag("1..5")
    hm._validate_tag("10..100")
    hm._validate_tag("1..1")  # Same start and end is valid


def test_validate_tag_invalid_non_numeric():
    """Test _validate_tag rejects non-numeric tags."""

    hm = EventManager()

    with pytest.raises(ValueError, match="must be a numeric string"):
        hm._validate_tag("abc")

    with pytest.raises(ValueError, match="must be a numeric string"):
        hm._validate_tag("event_1")


def test_validate_tag_invalid_range_format():
    """Test _validate_tag rejects malformed range tags."""

    hm = EventManager()

    with pytest.raises(ValueError, match="start and end must be integers"):
        hm._validate_tag("a..b")

    with pytest.raises(ValueError, match="start and end must be integers"):
        hm._validate_tag("1..abc")


def test_validate_tag_invalid_range_order():
    """Test _validate_tag rejects ranges where start > end."""

    hm = EventManager()

    with pytest.raises(ValueError, match="start .* must be <= end"):
        hm._validate_tag("10..5")


def test_parse_tag_start_defensive():
    """Test _parse_tag_start raises clear error for invalid tags."""

    hm = EventManager()

    with pytest.raises(ValueError, match="Invalid tag format"):
        hm._parse_tag_start("invalid")


def test_parse_tag_end_defensive():
    """Test _parse_tag_end raises clear error for invalid tags."""

    hm = EventManager()

    with pytest.raises(ValueError, match="Invalid tag format"):
        hm._parse_tag_end("invalid")


# === BeforeTurn / AfterTurn Symmetry Tests ===


def test_before_turn_event_first_turn():
    """Test BeforeTurn can identify first turn (turn_number=1)."""
    from nooa.events import BeforeTurn

    # First turn
    event = BeforeTurn(
        method_name="solve",
        strategy="codeact",
        generation_id="gen-123",
        turn_number=1,
    )

    assert event.turn_number == 1
    # First turn is like "before method" - can check turn_number == 1
    is_first_turn = event.turn_number == 1
    assert is_first_turn is True


def test_after_turn_event_intermediate():
    """Test AfterTurn for intermediate turns (is_final=False)."""
    from nooa.events import AfterTurn

    # Intermediate turn - method continues
    event = AfterTurn(
        method_name="solve",
        strategy="codeact",
        generation_id="gen-123",
        turn_number=1,
        is_final=False,
    )

    assert event.turn_number == 1
    assert event.is_final is False
    assert event.success is None  # Not set for intermediate turns


def test_after_turn_event_final_success():
    """Test AfterTurn for final turn with successful return."""
    from nooa.events import AfterTurn

    # Final turn - method completed successfully
    event = AfterTurn(
        method_name="solve",
        strategy="codeact",
        generation_id="gen-123",
        turn_number=5,
        is_final=True,
        success=True,
    )

    assert event.turn_number == 5
    assert event.is_final is True
    assert event.success is True
    assert event.exception_type is None

    # Can check if this is the final successful turn
    is_successful_completion = event.is_final and event.success
    assert is_successful_completion is True


def test_after_turn_event_final_failure():
    """Test AfterTurn for final turn with exception."""
    from nooa.events import AfterTurn

    # Final turn - method failed with exception
    event = AfterTurn(
        method_name="solve",
        strategy="codeact",
        generation_id="gen-123",
        turn_number=3,
        is_final=True,
        success=False,
        exception_type="MaxTurnsExceeded",
    )

    assert event.turn_number == 3
    assert event.is_final is True
    assert event.success is False
    assert event.exception_type == "MaxTurnsExceeded"

    # Can check if this is a failed completion
    is_failed_completion = event.is_final and not event.success
    assert is_failed_completion is True


def test_turn_events_symmetry():
    """Test that BeforeTurn and AfterTurn have symmetric fields."""
    from nooa.events import AfterTurn, BeforeTurn

    # Create matching before/after events for the same turn
    before = BeforeTurn(
        method_name="analyze",
        strategy="pure_python",
        generation_id="gen-456",
        parent_generation_id="gen-parent",
        turn_number=2,
    )

    after = AfterTurn(
        method_name="analyze",
        strategy="pure_python",
        generation_id="gen-456",
        parent_generation_id="gen-parent",
        turn_number=2,
        is_final=False,
    )

    # Both have the same core fields
    assert before.method_name == after.method_name
    assert before.strategy == after.strategy
    assert before.generation_id == after.generation_id
    assert before.parent_generation_id == after.parent_generation_id
    assert before.turn_number == after.turn_number

    # AfterTurn has additional final-turn fields
    assert hasattr(after, "is_final")
    assert hasattr(after, "success")
    assert hasattr(after, "exception_type")


# === Backend Protocol Tests ===


def test_custom_backend_can_be_injected():
    """Test that a custom backend can be provided to EventManager."""
    from nooa.runtime.event_backend import InMemoryBackend

    # Create a custom backend instance
    backend = InMemoryBackend()

    # Inject it into EventManager
    hm = EventManager(backend=backend)

    # Add events
    hm.add(Task(prompt="Test event"))

    # Verify it's stored in our backend
    assert len(backend) == 1
    assert backend.active_tags() == ["1"]


def test_backend_protocol_is_runtime_checkable():
    """Test that EventBackend protocol is runtime checkable."""
    from nooa.runtime.event_backend import EventBackend, InMemoryBackend

    backend = InMemoryBackend()
    assert isinstance(backend, EventBackend)


# === Additional Edge Case Tests ===


def test_turn_events_not_recorded_in_history():
    """Verify turn events are not recorded in event_manager when record=False."""
    from nooa.events import BeforeTurn

    hm = EventManager()

    before = BeforeTurn(
        method_name="test",
        strategy="test",
        generation_id="gen-1",
        turn_number=1,
    )
    hm.add(before, record=False)

    # Not recorded in history
    assert len(hm) == 0


def test_inmemory_backend_update_merges_metadata():
    """Verify metadata update merges rather than replaces."""
    from nooa.runtime.event_backend import InMemoryBackend

    backend = InMemoryBackend()
    event = Task(prompt="test")
    event.metadata = {"a": 1}
    backend.store("1", event)

    # Update with additional metadata
    backend.update("1", metadata={"b": 2})

    # Should merge, not replace
    assert backend.get("1").metadata == {"a": 1, "b": 2}


def test_inmemory_backend_get_by_id_not_found():
    """Verify get_by_id returns None for non-existent event."""
    from nooa.runtime.event_backend import InMemoryBackend

    backend = InMemoryBackend()
    backend.store("1", Task(prompt="test"))

    # Non-existent UUID
    assert backend.get_by_id("non-existent-uuid") is None


def test_inmemory_backend_set_status_transitions():
    """Verify set_status correctly transitions between states."""
    from nooa.runtime.event_backend import InMemoryBackend

    backend = InMemoryBackend()
    event = Task(prompt="test")
    backend.store("1", event)

    # Initial state is active
    assert backend.get("1").status == "active"

    # Transition to archived
    result = backend.set_status("1", "archived")
    assert result is True
    assert backend.get("1").status == "archived"

    # Transition back to active
    result = backend.set_status("1", "active")
    assert result is True
    assert backend.get("1").status == "active"

    # Non-existent tag returns False
    result = backend.set_status("999", "archived")
    assert result is False


# === __contains__ Tests ===


def test_contains_by_tag():
    """Test __contains__ finds events by their string tag."""
    hm = EventManager()
    hm.add(Task(prompt="First"))  # tag="1"
    hm.add(Task(prompt="Second"))  # tag="2"

    assert "1" in hm
    assert "2" in hm
    assert "3" not in hm


def test_contains_by_uuid():
    """Test __contains__ finds events by their UUID."""
    hm = EventManager()
    event = Task(prompt="Test")
    hm.add(event)

    assert event.id in hm
    assert "nonexistent-uuid" not in hm


def test_contains_after_remove():
    """Test __contains__ returns False for removed events."""
    hm = EventManager()
    hm.add(Task(prompt="First"))  # tag="1"
    hm.add(Task(prompt="Second"))  # tag="2"

    hm.remove("1")

    assert "1" not in hm
    assert "2" in hm


def test_contains_summary_tag():
    """Test __contains__ works with summary range tags after collapse."""
    hm = EventManager()
    for i in range(5):
        hm.add(Task(prompt=f"Event {i + 1}"))

    hm.collapse("2", "4", summary_text="Summary of 2-4")

    assert "2..4" in hm
    assert "1" in hm
    assert "5" in hm


# === update() Public API Tests ===


def test_update_by_tag():
    """Test update() modifies event fields by tag."""
    hm = EventManager()
    hm.add(Task(prompt="Original"))

    result = hm.update("1", metadata={"updated": True})

    assert result is True
    assert hm["1"].metadata["updated"] is True


def test_update_by_uuid():
    """Test update() modifies event fields by UUID."""
    hm = EventManager()
    event = Task(prompt="Original")
    hm.add(event)

    result = hm.update(event.id, metadata={"source": "test"})

    assert result is True
    assert hm["1"].metadata["source"] == "test"


def test_update_nonexistent_returns_false():
    """Test update() returns False for non-existent key."""
    hm = EventManager()
    hm.add(Task(prompt="Test"))

    result = hm.update("999", metadata={"x": 1})

    assert result is False


def test_update_merges_metadata():
    """Test update() merges metadata rather than replacing."""
    hm = EventManager()
    event = Task(prompt="Test")
    event.metadata["original"] = True
    hm.add(event)

    hm.update("1", metadata={"added": True})

    assert hm["1"].metadata["original"] is True
    assert hm["1"].metadata["added"] is True


# === remove() Dedicated Tests ===


def test_remove_by_tag():
    """Test remove() by string tag."""
    hm = EventManager()
    hm.add(Task(prompt="First"))  # tag="1"
    hm.add(Task(prompt="Second"))  # tag="2"
    hm.add(Task(prompt="Third"))  # tag="3"

    result = hm.remove("2")

    assert result is True
    assert len(hm) == 2
    assert hm.keys() == ["1", "3"]


def test_remove_by_uuid():
    """Test remove() by UUID."""
    hm = EventManager()
    event = Task(prompt="To remove")
    hm.add(event)
    hm.add(Task(prompt="Keep"))

    result = hm.remove(event.id)

    assert result is True
    assert len(hm) == 1
    assert hm["2"].prompt == "Keep"


def test_remove_nonexistent_returns_false():
    """Test remove() returns False for non-existent key."""
    hm = EventManager()
    hm.add(Task(prompt="Test"))

    assert hm.remove("999") is False
    assert hm.remove("nonexistent-uuid") is False
    assert len(hm) == 1


def test_remove_returns_true_on_success():
    """Test remove() return value is True when event is found and removed."""
    hm = EventManager()
    hm.add(Task(prompt="Test"))

    assert hm.remove("1") is True
    assert len(hm) == 0


# === keys() Dedicated Tests ===


def test_keys_order_preserved():
    """Test keys() returns tags in insertion order."""
    hm = EventManager()
    hm.add(Task(prompt="First"))
    hm.add(Task(prompt="Second"))
    hm.add(Task(prompt="Third"))

    assert hm.keys() == ["1", "2", "3"]


def test_keys_after_remove():
    """Test keys() reflects removal without shifting other tags."""
    hm = EventManager()
    hm.add(Task(prompt="First"))
    hm.add(Task(prompt="Second"))
    hm.add(Task(prompt="Third"))

    hm.remove("2")

    assert hm.keys() == ["1", "3"]


def test_keys_after_collapse():
    """Test keys() shows summary tag replacing collapsed range."""
    hm = EventManager()
    for i in range(5):
        hm.add(Task(prompt=f"Event {i + 1}"))

    hm.collapse("2", "4")

    assert hm.keys() == ["1", "2..4", "5"]


def test_keys_empty():
    """Test keys() returns empty list for empty EventManager."""
    hm = EventManager()
    assert hm.keys() == []


# === Emit / Unsubscribe Edge Cases ===


def test_on_unsubscribe_idempotent():
    """Calling unsubscribe twice does not raise."""
    hm = EventManager()
    unsub = hm.on("Task", lambda e: None)
    unsub()
    unsub()  # second call should be a no-op


def test_on_handler_self_unsubscribe_during_emit():
    """A handler that unsubscribes itself during emit doesn't crash the loop."""
    hm = EventManager()
    received = []
    unsub = None

    def self_removing(event):
        received.append(event)
        unsub()

    unsub = hm.on("Task", self_removing)
    hm.add(Task(prompt="trigger"))

    assert len(received) == 1
    # Handler removed itself — second add should NOT call it
    hm.add(Task(prompt="second"))
    assert len(received) == 1


def test_on_handler_adds_new_handler_during_emit():
    """A handler that adds another handler during emit doesn't affect the current emit."""
    hm = EventManager()
    second_received = []

    def adder(event):
        # Add a new handler mid-emit
        hm.on("Task", lambda e: second_received.append(e))

    hm.on("Task", adder)
    hm.add(Task(prompt="first"))

    # The new handler was added but should NOT have seen 'first'
    assert len(second_received) == 0

    # Next emit should see it
    hm.add(Task(prompt="second"))
    assert len(second_received) == 1


if __name__ == "__main__":
    test_basic_conversation_flow()
    test_error_feedback_flow()
    test_execution_feedback_flow()
    test_tagged_messages()
    test_events_property()
    test_len()
    test_clear()
    test_format_events_last_n()
    test_filter_by_query_basic()
    test_filter_with_limit()
    test_filter_no_results()
    test_collapse_basic()
    test_collapse_truncation()
    test_collapse_nested_flattens()
    test_items_returns_tag_event_pairs()
    test_values_returns_events_only()
    test_validate_tag_valid_simple()
    test_validate_tag_valid_range()
    test_validate_tag_invalid_non_numeric()
    test_turn_events_not_recorded_in_history()
    test_inmemory_backend_update_merges_metadata()
    test_inmemory_backend_get_by_id_not_found()
    test_inmemory_backend_set_status_transitions()
    test_contains_by_tag()
    test_contains_by_uuid()
    test_contains_after_remove()
    test_contains_summary_tag()
    test_update_by_tag()
    test_update_by_uuid()
    test_update_nonexistent_returns_false()
    test_update_merges_metadata()
    test_remove_by_tag()
    test_remove_by_uuid()
    test_remove_nonexistent_returns_false()
    test_remove_returns_true_on_success()
    test_keys_order_preserved()
    test_keys_after_remove()
    test_keys_after_collapse()
    test_keys_empty()
    print("All tests passed!")


def test_emit_handler_exception_does_not_skip_others():
    """A handler that raises must not prevent subsequent handlers from firing."""
    hm = EventManager()
    calls = []

    def bad_handler(event):
        raise RuntimeError("boom")

    def good_handler(event):
        calls.append("good")

    hm.on("Task", bad_handler)
    hm.on("Task", good_handler)

    hm.add(Task(prompt="test"))
    assert "good" in calls  # good_handler must still fire


def test_set_backend_preserves_subscribers():
    """Pins the bug fix: subscribers stay attached when the backend is swapped.

    Regression for the silent-preview bug — when ``/clear`` swapped the
    agent's storage, the old EventManager went away with it and any
    handler subscribed at startup (e.g. the TUI's AgentEventRenderer)
    pointed at a dead manager. The structural fix moves backend swap
    onto a stable EventManager so subscribers keep firing across the
    swap.
    """
    hm = EventManager(backend=InMemoryBackend())
    received = []
    hm.on("Task", lambda e: received.append(e.prompt))

    hm.add(Task(prompt="before-swap"))

    new_backend = InMemoryBackend()
    hm.set_backend(new_backend)

    hm.add(Task(prompt="after-swap"))

    assert received == ["before-swap", "after-swap"]
    # The post-swap event landed in the new backend; the pre-swap one didn't.
    pre_swap_in_new = any(
        getattr(e, "prompt", None) == "before-swap" for e in new_backend.all_events()
    )
    post_swap_in_new = any(
        getattr(e, "prompt", None) == "after-swap" for e in new_backend.all_events()
    )
    assert post_swap_in_new and not pre_swap_in_new


def test_set_backend_resets_tag_allocation_to_new_backend():
    """After set_backend, tag allocation tracks the new backend's high-water mark."""
    em = EventManager(backend=InMemoryBackend())
    em.add(Task(prompt="a"))  # tag "1"
    em.add(Task(prompt="b"))  # tag "2"

    # Swap to a backend that already has events through tag "5".
    populated = InMemoryBackend()
    for tag in ("1", "2", "3", "4", "5"):
        ev = Task(prompt=f"existing-{tag}")
        ev.tag = tag
        populated.store(tag, ev)

    em.set_backend(populated)
    new_tag = em.add(Task(prompt="new"))
    # Must skip past the existing tags rather than colliding.
    assert int(new_tag) > 5


def test_multiple_managers_share_backend_without_tag_collision():
    """Two EventManagers writing through the same backend never reuse a tag.

    This is the invariant that lets the TUI's SessionManager keep a
    light EventManager bound to the storage backend while the agent
    uses its own stable EventManager — both safe because tag allocation
    lives on the backend.
    """
    backend = InMemoryBackend()
    em1 = EventManager(backend=backend)
    em2 = EventManager(backend=backend)

    tags = [
        em1.add(Task(prompt="a")),
        em2.add(Task(prompt="b")),
        em1.add(Task(prompt="c")),
        em2.add(Task(prompt="d")),
    ]
    assert len(set(tags)) == len(tags)
    assert tags == ["1", "2", "3", "4"]


def test_direct_store_keeps_tag_counter_coherent():
    """A caller writing a high tag directly via ``backend.store()`` must
    not collide with the next ``allocate_next_tag()``.

    This pins the counter-staleness fix: ``store()`` updates
    ``_next_tag_num`` so any subsequent allocation skips past the
    just-stored tag, regardless of whether the tag came via
    ``allocate_next_tag()`` or a direct ``store()`` (e.g. snapshot
    re-hydration, tests, or any caller that pre-assigns a tag).
    """
    from nooa.events import EventBase

    backend = InMemoryBackend()
    em = EventManager(backend=backend)
    em.add(Task(prompt="a"))  # tag "1"

    pre_existing = EventBase()
    pre_existing.tag = "99"
    backend.store("99", pre_existing)

    assert em.add(Task(prompt="b")) == "100"


def test_clear_after_set_backend_clears_current_backend_only():
    """``em.clear()`` after ``set_backend()`` clears the new backend, not the old.

    Pins that swap-then-clear doesn't accidentally wipe the original
    storage and that fresh allocation restarts from "1" on the new one.
    """
    old_backend = InMemoryBackend()
    em = EventManager(backend=old_backend)
    em.add(Task(prompt="old1"))
    em.add(Task(prompt="old2"))

    new_backend = InMemoryBackend()
    em.set_backend(new_backend)
    em.add(Task(prompt="new1"))

    em.clear()

    # New backend was cleared; old backend untouched.
    assert len(new_backend) == 0
    assert len(old_backend) == 2
    # New backend's allocator restarts from "1" after clear.
    assert em.add(Task(prompt="fresh")) == "1"


def test_collapse_then_add_continues_tag_progression():
    """``add()`` after ``collapse()`` must allocate a tag past the collapsed range.

    Collapse stores a range tag (e.g. ``"2..4"``) via ``backend.store()``.
    The store coherence fix relies on ``_tag_max_num("2..4") = 4`` so the
    counter doesn't go backwards. This test pins that progression — if
    range-tag parsing in ``_tag_max_num`` ever regresses, collapse would
    silently corrupt the counter.
    """
    em = EventManager()
    for i in range(5):
        em.add(Task(prompt=f"e{i + 1}"))  # tags "1".."5", counter at 6
    em.collapse("2", "4", summary_text="s")  # stores "2..4"; counter still 6

    # Next allocated tag must be 6 — beyond both the original tag "5"
    # and the range tag "2..4"'s end value of 4.
    assert em.add(Task(prompt="next")) == "6"


def test_collapse_does_not_rewind_counter_when_range_end_below_max():
    """A ``collapse()`` whose range ends below the current high-water
    mark must not pull the counter backwards via the store coherence
    update.

    With 5 events stored (counter=6), collapsing "2..3" stores a tag
    whose ``_tag_max_num`` is 3. The coherence fix uses ``max(_next, …)``
    so the counter stays at 6, not 4.
    """
    em = EventManager()
    for i in range(5):
        em.add(Task(prompt=f"e{i + 1}"))
    em.collapse("2", "3")
    assert em.add(Task(prompt="after")) == "6"
