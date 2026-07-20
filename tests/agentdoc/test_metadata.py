# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agentdoc._metadata — metadata storage functions."""

from __future__ import annotations

from nooa.agentdoc._metadata import (
    get_docs_metadata,
    get_field_metadata,
    is_expand_false,
    set_docs_metadata,
    set_field_metadata,
)


class SampleClass:
    """A plain class for attaching metadata."""

    x: int = 0
    y: str = "hello"


def sample_func():
    """A plain function."""


class TestGetDocsMetadata:
    def test_returns_empty_dict_when_unset(self):
        class Fresh:
            pass

        assert get_docs_metadata(Fresh) == {}

    def test_returns_empty_dict_for_plain_instance(self):
        assert get_docs_metadata(42) == {}

    def test_returns_stored_dict(self):
        class Tagged:
            _agentdoc_docs = {"hidden": True}

        assert get_docs_metadata(Tagged) == {"hidden": True}


class TestSetDocsMetadata:
    def test_sets_single_key(self):
        class C:
            pass

        set_docs_metadata(C, hidden=True)
        assert get_docs_metadata(C)["hidden"] is True

    def test_merges_with_existing(self):
        class C:
            pass

        set_docs_metadata(C, hidden=True)
        set_docs_metadata(C, description="hi")
        meta = get_docs_metadata(C)
        assert meta["hidden"] is True
        assert meta["description"] == "hi"

    def test_overwrites_existing_key(self):
        class C:
            pass

        set_docs_metadata(C, expand=True)
        set_docs_metadata(C, expand=False)
        assert get_docs_metadata(C)["expand"] is False

    def test_silently_ignores_non_settable(self):
        # Built-in types are read-only; should not raise
        set_docs_metadata(int, hidden=True)  # no crash

    def test_works_on_functions(self):
        def f():
            pass

        set_docs_metadata(f, description="a function")
        assert get_docs_metadata(f)["description"] == "a function"


class TestGetFieldMetadata:
    def test_returns_empty_dict_when_unset(self):
        class C:
            pass

        assert get_field_metadata(C, "x") == {}

    def test_returns_stored_field_dict(self):
        class C:
            _agentdoc_fields_docs = {"x": {"hidden": True}}

        assert get_field_metadata(C, "x") == {"hidden": True}

    def test_missing_field_returns_empty(self):
        class C:
            _agentdoc_fields_docs = {"x": {"hidden": True}}

        assert get_field_metadata(C, "z") == {}


class TestSetFieldMetadata:
    def test_sets_field_metadata(self):
        class C:
            x: int

        set_field_metadata(C, "x", hidden=True)
        assert get_field_metadata(C, "x")["hidden"] is True

    def test_creates_fields_attr_when_absent(self):
        class C:
            pass

        assert not hasattr(C, "_agentdoc_fields_docs")
        set_field_metadata(C, "name", description="The name")
        assert hasattr(C, "_agentdoc_fields_docs")

    def test_merges_multiple_fields(self):
        class C:
            x: int
            y: str

        set_field_metadata(C, "x", hidden=True)
        set_field_metadata(C, "y", description="Y field")
        assert get_field_metadata(C, "x") == {"hidden": True}
        assert get_field_metadata(C, "y") == {"description": "Y field"}

    def test_merges_multiple_calls_same_field(self):
        class C:
            x: int

        set_field_metadata(C, "x", hidden=True)
        set_field_metadata(C, "x", description="also X")
        meta = get_field_metadata(C, "x")
        assert meta["hidden"] is True
        assert meta["description"] == "also X"

    def test_silently_ignores_frozen_objects(self):
        # Should not raise even on objects that don't allow attribute setting
        set_field_metadata(42, "x", hidden=True)  # no crash


class TestIsExpandFalse:
    def test_returns_false_when_no_metadata(self):
        class C:
            pass

        assert is_expand_false(C) is False

    def test_returns_false_when_expand_true(self):
        class C:
            pass

        set_docs_metadata(C, expand=True)
        assert is_expand_false(C) is False

    def test_returns_true_when_expand_false(self):
        class C:
            pass

        set_docs_metadata(C, expand=False)
        assert is_expand_false(C) is True

    def test_returns_false_for_non_type(self):
        assert is_expand_false("hello") is False
        assert is_expand_false(42) is False
        assert is_expand_false(None) is False
