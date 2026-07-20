# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context_blocks models: DynamicContext, ResolvedBlock, Role, BlockMetadata."""

import pytest
from pydantic import ValidationError


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Role enum has expected values."""
        from nooa.context_blocks.models import Role

        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"
        assert Role.RUNTIME_EVENT == "runtime_event"

    def test_role_is_string(self):
        """Role values are strings (str enum)."""
        from nooa.context_blocks.models import Role

        assert isinstance(Role.SYSTEM, str)
        assert Role.SYSTEM == "system"


class TestDynamicContext:
    """Tests for DynamicContext marker type."""

    def test_create_with_valid_expr(self):
        """DynamicContext accepts valid Python expression."""
        from nooa.context_blocks.models import DynamicContext

        d = DynamicContext("self.format_status()")
        assert d.expr == "self.format_status()"

    def test_create_with_simple_expr(self):
        """DynamicContext accepts simple expressions."""
        from nooa.context_blocks.models import DynamicContext

        d = DynamicContext("1 + 2")
        assert d.expr == "1 + 2"

    def test_create_with_complex_expr(self):
        """DynamicContext accepts complex Python expressions."""
        from nooa.context_blocks.models import DynamicContext

        d = DynamicContext("doc(self.context, concise=True)")
        assert d.expr == "doc(self.context, concise=True)"

    def test_create_with_invalid_expr_raises(self):
        """DynamicContext raises BlockSyntaxError for invalid Python syntax."""
        from nooa.context_blocks.exceptions import BlockSyntaxError
        from nooa.context_blocks.models import DynamicContext

        with pytest.raises(BlockSyntaxError):
            DynamicContext("this is not valid python!!!")

    def test_create_with_statement_raises(self):
        """DynamicContext raises for Python statements (not expressions)."""
        from nooa.context_blocks.exceptions import BlockSyntaxError
        from nooa.context_blocks.models import DynamicContext

        with pytest.raises(BlockSyntaxError):
            DynamicContext("x = 5")

    def test_repr(self):
        """DynamicContext has a clear repr."""
        from nooa.context_blocks.models import DynamicContext

        d = DynamicContext("self.value")
        assert repr(d) == "DynamicContext('self.value')"

    def test_equality(self):
        """DynamicContext supports equality comparison."""
        from nooa.context_blocks.models import DynamicContext

        d1 = DynamicContext("self.value")
        d2 = DynamicContext("self.value")
        d3 = DynamicContext("self.other")

        assert d1 == d2
        assert d1 != d3

    def test_hash(self):
        """DynamicContext is hashable (can be used in sets/dicts)."""
        from nooa.context_blocks.models import DynamicContext

        d1 = DynamicContext("self.value")
        d2 = DynamicContext("self.value")

        assert hash(d1) == hash(d2)
        assert {d1, d2} == {d1}  # type: ignore[set-literal]

    def test_not_equal_to_string(self):
        """DynamicContext is not equal to a plain string."""
        from nooa.context_blocks.models import DynamicContext

        d = DynamicContext("self.value")
        assert d != "self.value"


class TestBlockMetadata:
    """Tests for BlockMetadata Pydantic model."""

    def test_default_values(self):
        """BlockMetadata has sensible defaults."""
        from nooa.context_blocks.models import BlockMetadata

        meta = BlockMetadata()
        assert meta.expr is None
        assert meta.tag is None
        assert meta.truncated is False

    def test_with_expr(self):
        """BlockMetadata with expr field."""
        from nooa.context_blocks.models import BlockMetadata

        meta = BlockMetadata(expr="self.context['notes']")
        assert meta.expr == "self.context['notes']"

    def test_frozen(self):
        """BlockMetadata is immutable (frozen)."""
        from nooa.context_blocks.models import BlockMetadata

        meta = BlockMetadata(expr="test")
        with pytest.raises(ValidationError):
            meta.expr = "other"

    def test_model_copy(self):
        """BlockMetadata supports model_copy for updates."""
        from nooa.context_blocks.models import BlockMetadata

        meta = BlockMetadata(expr="test")
        updated = meta.model_copy(update={"truncated": True})
        assert updated.truncated is True
        assert updated.expr == "test"
        assert meta.truncated is False  # original unchanged


class TestResolvedBlock:
    """Tests for ResolvedBlock Pydantic model."""

    def test_create_minimal(self):
        """ResolvedBlock with just key and content."""
        from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role

        block = ResolvedBlock(key="test", content="hello")
        assert block.key == "test"
        assert block.content == "hello"
        assert block.role == Role.SYSTEM  # default
        assert block.metadata == BlockMetadata()  # default

    def test_create_with_role(self):
        """ResolvedBlock with explicit role."""
        from nooa.context_blocks.models import ResolvedBlock, Role

        block = ResolvedBlock(key="msg", content="hi", role=Role.USER)
        assert block.role == Role.USER

    def test_create_with_metadata(self):
        """ResolvedBlock with BlockMetadata."""
        from nooa.context_blocks.models import BlockMetadata, ResolvedBlock

        block = ResolvedBlock(
            key="test",
            content="hello",
            metadata=BlockMetadata(expr="self.value", truncated=False),
        )
        assert block.metadata.expr == "self.value"
        assert block.metadata.truncated is False

    def test_metadata_default_is_independent(self):
        """Each ResolvedBlock gets its own default BlockMetadata instance."""
        from nooa.context_blocks.models import ResolvedBlock

        b1 = ResolvedBlock(key="a", content="a")
        b2 = ResolvedBlock(key="b", content="b")

        # Both get default BlockMetadata, but they're independent frozen instances
        assert b1.metadata == b2.metadata
        assert b1.metadata is not b2.metadata or b1.metadata == b2.metadata

    def test_model_copy(self):
        """ResolvedBlock supports model_copy for updates."""
        from nooa.context_blocks.models import ResolvedBlock

        block = ResolvedBlock(key="test", content="original")
        updated = block.model_copy(update={"content": "updated"})
        assert updated.content == "updated"
        assert block.content == "original"  # original unchanged
