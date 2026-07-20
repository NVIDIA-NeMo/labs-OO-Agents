# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for storage/markers.py edge cases.

Covers:
- is_nosnapshot_field() line 54 (return False when fallback returns None)
- _resolve_single() lines 72-73 (eval failure path)
"""

from nooa.storage.markers import _resolve_single, is_nosnapshot_field


class TestResolveSingle:
    """Tests for _resolve_single() annotation resolution."""

    def test_eval_failure_returns_none(self):
        """_resolve_single returns None when the annotation string cannot be eval'd."""

        class ClassWithBadAnnotation:
            __module__ = "__main__"
            # String annotation that references a name not in __main__'s namespace
            __annotations__ = {"field": "NotARealTypeAtAll"}

        result = _resolve_single(ClassWithBadAnnotation, "field")
        assert result is None

    def test_non_string_annotation_returned_as_is(self):
        """_resolve_single returns a non-string annotation directly."""

        class ClassWithIntAnnotation:
            __annotations__ = {"field": int}

        result = _resolve_single(ClassWithIntAnnotation, "field")
        assert result is int

    def test_missing_field_returns_none(self):
        """_resolve_single returns None when the field is not annotated."""

        class ClassWithNoAnnotation:
            pass

        result = _resolve_single(ClassWithNoAnnotation, "missing")
        assert result is None


class TestIsNosnapshotField:
    """Tests for is_nosnapshot_field() edge cases."""

    def test_returns_false_when_type_hints_fail_and_resolve_single_returns_none(self):
        """Covers the return False on line 54: hint is None after fallback."""

        # A class with a forward-reference annotation that:
        # 1. Exists in __annotations__ (so the name check passes)
        # 2. Causes get_type_hints() to raise (unresolvable forward ref)
        # 3. _resolve_single() also returns None (name not in module namespace)
        class ClassWithUnresolvableAnnotation:
            x: "TypeThatDoesNotExistAnywhere"  # noqa: F821

        result = is_nosnapshot_field(ClassWithUnresolvableAnnotation, "x")
        assert result is False
