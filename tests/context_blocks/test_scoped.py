# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ScopedContext context manager."""

from nooa.context_blocks.models import DynamicContext
from nooa.context_blocks.scoped import ScopedContext, _scoped_blocks_var


class TestScopedBlocks:
    """Tests for ScopedContext context manager."""

    def test_sets_and_cleans_up(self):
        """Blocks are visible inside scope and gone after."""
        assert _scoped_blocks_var.get() is None

        with ScopedContext(context={"focus": "security"}):
            val = _scoped_blocks_var.get()
            assert val is not None
            assert val["focus"] == "security"

        assert _scoped_blocks_var.get() is None

    def test_nesting_inherits_parent(self):
        """Inner scope inherits blocks from parent scope."""
        with ScopedContext(context={"a": "1"}):
            with ScopedContext(context={"b": "2"}):
                val = _scoped_blocks_var.get()
                assert val is not None
                assert val["a"] == "1"
                assert val["b"] == "2"

            # After inner scope, only "a" remains
            val = _scoped_blocks_var.get()
            assert val is not None
            assert val["a"] == "1"
            assert "b" not in val

    def test_nesting_overrides_parent(self):
        """Inner scope can override parent blocks."""
        with ScopedContext(context={"mode": "fast"}):
            with ScopedContext(context={"mode": "slow"}):
                val = _scoped_blocks_var.get()
                assert val is not None
                assert val["mode"] == "slow"

            # Parent value restored
            val = _scoped_blocks_var.get()
            assert val is not None
            assert val["mode"] == "fast"

    def test_none_removes_block(self):
        """Setting a block to None removes it within the scope."""
        with ScopedContext(context={"focus": "security", "mode": "strict"}):
            with ScopedContext(context={"focus": None}):
                val = _scoped_blocks_var.get()
                assert val is not None
                # None is stored as-is (runtime interprets it as removal)
                assert val["focus"] is None
                assert val["mode"] == "strict"

            # Parent value restored
            val = _scoped_blocks_var.get()
            assert val is not None
            assert val["focus"] == "security"

    def test_dynamic_value(self):
        """DynamicContext values are stored as-is for later resolution."""
        dyn = DynamicContext("self.context['notes']")
        with ScopedContext(context={"notes": dyn}):
            val = _scoped_blocks_var.get()
            assert val is not None
            assert isinstance(val["notes"], DynamicContext)
            assert val["notes"].expr == "self.context['notes']"

    def test_empty_context(self):
        """Empty context dict is valid (no-op scope)."""
        with ScopedContext(context={}):
            val = _scoped_blocks_var.get()
            assert val == {}

    def test_none_context(self):
        """None context is valid (no-op scope)."""
        with ScopedContext(context=None):
            val = _scoped_blocks_var.get()
            assert val == {}

    def test_importable_from_package(self):
        """ScopedContext is importable from the top-level package."""
        from nooa.context_blocks import ScopedContext as SC

        assert SC is ScopedContext
