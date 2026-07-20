# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ActorRuntime generation ID methods."""

import pytest

from nooa.runtime.actor import (
    ActorRuntime,
    _generation_id_stack_var,
    _pop_generation_id,
    _push_generation_id,
)


class MockAgent:
    """Minimal mock agent for testing ActorRuntime."""

    def __init__(self):
        self.event_manager = None  # Will be set by ActorRuntime


@pytest.fixture
def clean_generation_stack():
    """Ensure generation ID stack is empty before and after each test."""
    # Clear before test
    _generation_id_stack_var.set(())
    yield
    # Clear after test
    _generation_id_stack_var.set(())


class TestActorRuntimeGenerationIds:
    """Tests for get_generation_id() and get_parent_generation_id()."""

    def test_get_generation_id_empty_stack_returns_none(self, clean_generation_stack):
        """get_generation_id() returns None when stack is empty."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)

        assert runtime._generation_id_stack == ()
        assert runtime.get_generation_id() is None

    def test_get_generation_id_single_item_returns_it(self, clean_generation_stack):
        """get_generation_id() returns the only item when stack has one."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)
        _push_generation_id("gen-1")

        assert runtime.get_generation_id() == "gen-1"

    def test_get_generation_id_returns_innermost(self, clean_generation_stack):
        """get_generation_id() returns the innermost (last) generation ID."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)
        _push_generation_id("gen-root")
        _push_generation_id("gen-child")
        _push_generation_id("gen-grandchild")

        assert runtime.get_generation_id() == "gen-grandchild"

    def test_get_parent_generation_id_empty_stack_returns_none(self, clean_generation_stack):
        """get_parent_generation_id() returns None when stack is empty."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)

        assert runtime._generation_id_stack == ()
        assert runtime.get_parent_generation_id() is None

    def test_get_parent_generation_id_single_item_returns_none(self, clean_generation_stack):
        """get_parent_generation_id() returns None when only one item (root)."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)
        _push_generation_id("gen-root")

        assert runtime.get_parent_generation_id() is None

    def test_get_parent_generation_id_two_items_returns_first(self, clean_generation_stack):
        """get_parent_generation_id() returns parent when two items."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)
        _push_generation_id("gen-parent")
        _push_generation_id("gen-child")

        assert runtime.get_parent_generation_id() == "gen-parent"

    def test_get_parent_generation_id_returns_second_to_last(self, clean_generation_stack):
        """get_parent_generation_id() returns second-to-last generation ID."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)
        _push_generation_id("gen-root")
        _push_generation_id("gen-child")
        _push_generation_id("gen-grandchild")

        assert runtime.get_parent_generation_id() == "gen-child"

    def test_generation_ids_consistent_after_push_pop(self, clean_generation_stack):
        """Generation IDs are consistent after stack push/pop operations."""
        agent = MockAgent()
        runtime = ActorRuntime(agent)

        # Empty stack
        assert runtime.get_generation_id() is None
        assert runtime.get_parent_generation_id() is None

        # Push first
        _push_generation_id("gen-1")
        assert runtime.get_generation_id() == "gen-1"
        assert runtime.get_parent_generation_id() is None

        # Push second (nested call)
        _push_generation_id("gen-2")
        assert runtime.get_generation_id() == "gen-2"
        assert runtime.get_parent_generation_id() == "gen-1"

        # Push third (deeper nesting)
        _push_generation_id("gen-3")
        assert runtime.get_generation_id() == "gen-3"
        assert runtime.get_parent_generation_id() == "gen-2"

        # Pop (return from nested)
        _pop_generation_id()
        assert runtime.get_generation_id() == "gen-2"
        assert runtime.get_parent_generation_id() == "gen-1"

        # Pop again
        _pop_generation_id()
        assert runtime.get_generation_id() == "gen-1"
        assert runtime.get_parent_generation_id() is None

        # Pop last
        _pop_generation_id()
        assert runtime.get_generation_id() is None
        assert runtime.get_parent_generation_id() is None
