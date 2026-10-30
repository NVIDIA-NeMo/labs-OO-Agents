# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic accessors are documented, not evaluated or turned into stored fields."""

from functools import cached_property
from typing import ClassVar

import pytest
from pydantic import BaseModel

from nooa.agentdoc import doc, hidden, spec


class Parent(BaseModel):
    stored: int = 1

    @property
    def view(self) -> str:
        """A computed view."""
        raise AssertionError("documentation must not evaluate a property")

    @cached_property
    def cached(self) -> int:
        """An unevaluated cached view."""
        raise AssertionError("documentation must not evaluate a cached property")

    @property
    @hidden
    def secret(self) -> str:
        raise AssertionError("hidden accessor was evaluated")

    @cached_property
    @hidden
    def secret_cached(self) -> str:
        raise AssertionError("hidden cached accessor was evaluated")


class Child(Parent):
    pass


@pytest.mark.parametrize("target", [Parent, Parent(), Child, Child()])
def test_pydantic_properties_are_visible_without_getter_evaluation(target):
    text = doc(target)
    assert "view: str" in text and "A computed view." in text
    assert "cached: int" in text
    assert "secret" not in text
    assert "model_fields_set" not in text and "model_extra" not in text
    assert "model_dump" not in text
    assert set(Child.model_fields) == {"stored"}
    assert Child().model_dump() == {"stored": 1}


def test_derived_accessor_type_wins_once():
    class Override(Parent):
        @property
        def view(self) -> int:
            raise AssertionError("not evaluated")

    text = doc(Override)
    assert text.count("view:") == 1
    assert "view: int" in text and "view: str" not in text


def test_hidden_override_does_not_reveal_parent_property():
    class Override(Parent):
        @property
        @hidden
        def view(self) -> int:
            raise AssertionError("not evaluated")

    assert "view:" not in doc(Override)


def test_non_property_override_does_not_reveal_parent_property():
    class Override(Parent):
        view: ClassVar[str] = "not an accessor"

    assert "view:" not in doc(Override)


def test_explicit_unhide_is_honored():
    class Unhidden(Parent):
        pass

    spec(Unhidden, "secret", hidden=False)
    assert "secret: str" in doc(Unhidden)


def test_inherited_explicit_unhide_is_honored():
    class UnhiddenBase(Parent):
        pass

    class Inherits(UnhiddenBase):
        pass

    class Rehidden(UnhiddenBase):
        pass

    spec(UnhiddenBase, "secret", hidden=False)
    spec(Rehidden, "secret", hidden=True)  # the most-derived declaration wins
    assert "secret: str" in doc(UnhiddenBase)
    assert "secret: str" in doc(Inherits) and "secret: str" in doc(Inherits())
    assert "secret" not in doc(Rehidden)


def test_inherited_explicit_unhide_on_plain_and_dataclass_accessors():
    import dataclasses

    class PlainBase:
        @property
        @hidden
        def secret(self) -> str:
            raise AssertionError("not evaluated")

    @dataclasses.dataclass
    class DataBase:
        stored: int = 1

        @property
        @hidden
        def secret(self) -> str:
            raise AssertionError("not evaluated")

    for base in (PlainBase, DataBase):
        spec(base, "secret", hidden=False)
        child = type(f"{base.__name__}Child", (base,), {})
        if dataclasses.is_dataclass(base):
            child = dataclasses.dataclass(child)
        assert "secret: str" in doc(base)
        assert "secret: str" in doc(child), base.__name__


def test_mixin_accessor_after_base_model_is_documented():
    class Mixin:
        @property
        def mixed(self) -> bool:
            raise AssertionError("not evaluated")

    class Model(BaseModel, Mixin):
        stored: int = 1

    assert "mixed: bool" in doc(Model)
