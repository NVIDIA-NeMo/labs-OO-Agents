# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for doc(instance) including dynamically-attached instance fields.

Issue #199: doc(instance) should match doc(type(instance)) and additionally
surface fields attached dynamically to that specific instance.

Plain classes and dataclasses store dynamic attributes in __dict__ (already
rendered). Pydantic models declared with ``extra="allow"`` store dynamically
assigned fields in ``__pydantic_extra__`` instead — these were previously
dropped. All three are covered here.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from nooa.agentdoc import doc, pformat, spec


class _Baz(BaseModel):
    """A baz model that allows extra fields."""

    model_config = {"extra": "allow"}

    label: str = "x"

    def greet(self, name: str) -> str:
        """Say hi."""
        return f"hi {name}"


@dataclass
class _Foo:
    """A foo dataclass."""

    label: str = "x"

    def greet(self, name: str) -> str:
        """Say hi."""
        return f"hi {name}"


class _Bar:
    """A plain bar class."""

    label: str = "x"

    def greet(self, name: str) -> str:
        """Say hi."""
        return f"hi {name}"


# ---------------------------------------------------------------------------
# Pydantic extra="allow" — the fix
# ---------------------------------------------------------------------------


class TestPydanticExtraFields:
    def test_instance_shows_dynamic_pydantic_extra(self):
        z = _Baz()
        z.dynamic_field = 123  # lands in __pydantic_extra__, not __dict__
        z.tags = ["a", "b"]
        out = doc(z)
        assert "dynamic_field: int = 123" in out
        assert "tags: list = ['a', 'b']" in out

    def test_type_does_not_show_dynamic_fields(self):
        # doc(type) is a class-only view: dynamic instance state must not leak in.
        out = doc(_Baz)
        assert "dynamic_field" not in out
        assert "tags" not in out

    def test_pformat_shows_dynamic_pydantic_extra(self):
        z = _Baz()
        z.dynamic_field = 123
        out = pformat(z)  # repr-style path
        assert "dynamic_field=123" in out
        assert out.startswith("_Baz(")

    def test_instance_hidden_extra_is_excluded(self):
        # Per-instance spec(self, name, hidden=True) must hide a dynamic extra.
        z = _Baz()
        z.shown = 1
        z.secret = 2
        spec(z, "secret", hidden=True)
        out = doc(z)
        assert "shown" in out
        assert "secret" not in out


# ---------------------------------------------------------------------------
# Plain class / dataclass — regression guards (already worked, keep working)
# ---------------------------------------------------------------------------


class TestDictBackedDynamicFields:
    def test_plain_class_shows_dynamic_dict_field(self):
        b = _Bar()
        b.dynamic_field = 123
        out = doc(b)
        assert "dynamic_field: int = 123" in out

    def test_dataclass_shows_dynamic_dict_field(self):
        f = _Foo()
        f.dynamic_field = 123
        out = doc(f)
        assert "dynamic_field: int = 123" in out


# ---------------------------------------------------------------------------
# Type-view parity — doc(instance) is a superset of doc(type)'s structure
# ---------------------------------------------------------------------------


class TestTypeViewParity:
    def test_instance_doc_matches_type_structure(self):
        z = _Baz()
        z.dynamic_field = 123
        type_out = doc(_Baz)
        inst_out = doc(z)

        # Class header, docstring, declared field, and method are all present in
        # the instance view exactly as in the type view.
        assert "class _Baz(BaseModel):" in type_out
        assert "class _Baz(BaseModel):" in inst_out
        assert '"""A baz model that allows extra fields."""' in inst_out
        assert "label: str = 'x'" in inst_out
        assert "def greet(self, name: str) -> str:" in inst_out
        # ...and the instance view additionally carries the dynamic field.
        assert "dynamic_field: int = 123" in inst_out
