# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for inherited field visibility in doc().

Verifies that doc() walks the full MRO so parent-class fields appear
in the output alongside child-class fields.
"""

from __future__ import annotations

from typing import Annotated

from nooa.agentdoc import doc, spec
from nooa.agentdoc._structured import extract_type_info
from nooa.agentdoc._visibility import hidden

# ---------------------------------------------------------------------------
# Single-level inheritance
# ---------------------------------------------------------------------------


class Animal:
    """A living creature."""

    name: str = "unnamed"
    legs: int = 4


class Dog(Animal):
    """A domestic dog."""

    breed: str = "mixed"


class TestSingleInheritance:
    def test_doc_shows_parent_fields(self):
        result = doc(Dog)
        assert "name" in result
        assert "legs" in result

    def test_doc_shows_child_fields(self):
        result = doc(Dog)
        assert "breed" in result

    def test_parent_fields_appear_before_child_fields(self):
        result = doc(Dog)
        assert result.index("name") < result.index("breed")
        assert result.index("legs") < result.index("breed")

    def test_extract_type_info_includes_parent_fields(self):
        info = extract_type_info(Dog)
        field_names = [f.name for f in info.fields]
        assert "name" in field_names
        assert "legs" in field_names
        assert "breed" in field_names

    def test_parent_fields_before_child_in_field_list(self):
        info = extract_type_info(Dog)
        field_names = [f.name for f in info.fields]
        assert field_names.index("name") < field_names.index("breed")

    def test_class_still_shown_correctly(self):
        result = doc(Dog)
        assert "class Dog" in result
        assert "A domestic dog." in result


# ---------------------------------------------------------------------------
# Deep inheritance chain
# ---------------------------------------------------------------------------


class Vehicle:
    """A means of transport."""

    speed: float = 0.0
    fuel: str = "gasoline"


class Car(Vehicle):
    """A four-wheeled vehicle."""

    doors: int = 4


class ElectricCar(Car):
    """A battery-powered car."""

    battery_kwh: float = 75.0


class TestDeepInheritance:
    def test_grandparent_fields_shown(self):
        result = doc(ElectricCar)
        assert "speed" in result
        assert "fuel" in result

    def test_parent_fields_shown(self):
        result = doc(ElectricCar)
        assert "doors" in result

    def test_own_fields_shown(self):
        result = doc(ElectricCar)
        assert "battery_kwh" in result

    def test_field_ordering_grandparent_parent_child(self):
        info = extract_type_info(ElectricCar)
        field_names = [f.name for f in info.fields]
        assert field_names.index("speed") < field_names.index("doors")
        assert field_names.index("doors") < field_names.index("battery_kwh")


# ---------------------------------------------------------------------------
# Multiple inheritance (diamond)
# ---------------------------------------------------------------------------


class Flyable:
    """Something that can fly."""

    max_altitude: float = 10_000.0


class Swimmable:
    """Something that can swim."""

    max_depth: float = 50.0


class Duck(Flyable, Swimmable):
    """A duck — flies and swims."""

    quacks: bool = True


class TestMultipleInheritance:
    def test_all_parent_fields_shown(self):
        result = doc(Duck)
        assert "max_altitude" in result
        assert "max_depth" in result
        assert "quacks" in result

    def test_field_ordering_follows_mro(self):
        """MRO of Duck: [Duck, Flyable, Swimmable, object].
        Reversed for collection: [Swimmable, Flyable, Duck].
        So order: max_depth (Swimmable), max_altitude (Flyable), quacks (Duck).
        """
        info = extract_type_info(Duck)
        field_names = [f.name for f in info.fields]
        assert field_names.index("max_altitude") < field_names.index("quacks")
        assert field_names.index("max_depth") < field_names.index("quacks")


class TestDiamondInheritance:
    def test_diamond_field_not_duplicated(self):
        """If A defines a field and B, C both inherit A, D(B, C) should show it once."""

        class A:
            shared: int = 0

        class B(A):
            b_field: str = "b"

        class C(A):
            c_field: str = "c"

        class D(B, C):
            d_field: float = 1.0

        info = extract_type_info(D)
        field_names = [f.name for f in info.fields]
        assert field_names.count("shared") == 1
        assert "b_field" in field_names
        assert "c_field" in field_names
        assert "d_field" in field_names


# ---------------------------------------------------------------------------
# Child overrides parent field
# ---------------------------------------------------------------------------


class TestChildOverridesParent:
    def test_child_redeclares_field_with_new_default(self):
        """Child re-declaring a parent field should use child's default."""

        class Parent:
            x: int = 10

        class Child(Parent):
            x: int = 99  # override

        info = extract_type_info(Child)
        field_names = [f.name for f in info.fields]
        assert field_names.count("x") == 1
        # Should use child's default
        x_field = next(f for f in info.fields if f.name == "x")
        assert x_field.default == 99

    def test_child_unhides_parent_hidden_field(self):
        """Child re-declaring without hidden makes field visible in doc()."""

        class Parent:
            secret: Annotated[str, hidden] = "shh"

        class Child(Parent):
            secret: str = "visible"  # re-declared without hidden

        result = doc(Child)
        assert "secret" in result


# ---------------------------------------------------------------------------
# Hidden parent fields remain hidden in child (if not re-declared)
# ---------------------------------------------------------------------------


class TestHiddenInheritance:
    def test_hidden_parent_field_not_shown_in_child(self):
        class Parent:
            api_key: Annotated[str, hidden] = ""
            label: str = "parent"

        class Child(Parent):
            child_field: int = 0

        result = doc(Child)
        assert "label" in result
        assert "child_field" in result
        assert "api_key" not in result


# ---------------------------------------------------------------------------
# Inherited un-annotated class attributes
#
# Un-annotated class-level attributes (e.g. `shell = ShellTools()`,
# `todos = TodoManager()`) declared on a base class were dropped from the
# child's doc() because step 2 of field extraction only scanned the leaf
# class's own __dict__. Annotated fields and methods survived (they walk the
# MRO); bare attributes did not.
# ---------------------------------------------------------------------------


class _Tool:
    """A tool-like object used as an un-annotated class attribute."""


class _ToolA:
    pass


class _ToolB:
    pass


class TestInheritedUnannotatedAttrs:
    def test_inherited_bare_attr_present(self):
        """A bare (un-annotated) class attribute on the parent appears on the child."""

        class Parent:
            annotated_attr: _Tool = _Tool()
            bare_attr = _Tool()

        class Child(Parent):
            pass

        info = extract_type_info(Child)
        field_names = [f.name for f in info.fields]
        assert "annotated_attr" in field_names
        assert "bare_attr" in field_names

    def test_inherited_bare_attr_rendered_as_instance_marker(self):
        """The inherited bare attr renders with a ClassName() instance marker."""

        class Parent:
            bare_attr = _Tool()

        class Child(Parent):
            pass

        result = doc(Child)
        assert "bare_attr" in result
        assert "_Tool()" in result

    def test_deep_inheritance_grandparent_bare_attr(self):
        """Un-annotated attr declared on a grandparent shows on the grandchild."""

        class Grandparent:
            gp_tool = _Tool()

        class Parent(Grandparent):
            pass

        class Child(Parent):
            pass

        field_names = [f.name for f in extract_type_info(Child).fields]
        assert "gp_tool" in field_names

    def test_leaf_override_wins(self):
        """When a child re-declares a parent bare attr, the child's value wins."""

        class Parent:
            tool = _ToolA()

        class Child(Parent):
            tool = _ToolB()

        info = extract_type_info(Child)
        field_names = [f.name for f in info.fields]
        assert field_names.count("tool") == 1
        tool_field = next(f for f in info.fields if f.name == "tool")
        # Default marker formats as ClassName()
        assert repr(tool_field.default) == "_ToolB()"

    def test_parent_spec_hidden_attr_suppressed(self):
        """spec(Parent, attr, hidden=True) on a bare parent attr stays out of child doc()."""

        class Parent:
            visible_tool = _Tool()
            secret_tool = _Tool()

        spec(Parent, "secret_tool", hidden=True)

        class Child(Parent):
            pass

        result = doc(Child)
        assert "visible_tool" in result
        assert "secret_tool" not in result

    def test_leaf_spec_unhide_wins_over_parent_hidden(self):
        """Leaf-level spec(hidden=False) overrides a parent spec(hidden=True)."""

        class Parent:
            tool = _Tool()

        spec(Parent, "tool", hidden=True)

        class Child(Parent):
            pass

        spec(Child, "tool", hidden=False)

        result = doc(Child)
        assert "tool" in result

    def test_inherited_attr_ordering(self):
        """Inherited bare attr renders before the child's own bare attr."""

        class Parent:
            parent_tool = _Tool()

        class Child(Parent):
            child_tool = _Tool()

        field_names = [f.name for f in extract_type_info(Child).fields]
        assert field_names.index("parent_tool") < field_names.index("child_tool")

    def test_metaclass_attribute_not_leaked(self):
        """An attribute defined only on a metaclass must not appear in child doc()."""

        class Meta(type):
            meta_only_attr = _Tool()

        class Parent(metaclass=Meta):
            bare_attr = _Tool()

        class Child(Parent):
            pass

        field_names = [f.name for f in extract_type_info(Child).fields]
        assert "bare_attr" in field_names
        assert "meta_only_attr" not in field_names


class TestInheritedUnannotatedAgentAttrs:
    """The real-world case: a base Agent with un-annotated tool attrs."""

    def test_agent_subclass_inherits_bare_tool_attrs(self):
        from nooa import Agent
        from nooa.unifiedllm import FakeLLMClient

        fake_llm = FakeLLMClient([])

        class _Shell:
            """Stand-in for ShellTools."""

        class _Todos:
            """Stand-in for TodoManager."""

        class BaseAgent(Agent, llm=fake_llm):
            shell = _Shell()
            todos = _Todos()

        class ChildAgent(BaseAgent):
            pass

        field_names = [f.name for f in extract_type_info(ChildAgent).fields]
        assert "shell" in field_names
        assert "todos" in field_names

        result = doc(ChildAgent)
        assert "shell" in result
        assert "todos" in result
        # Framework internals must not leak into the child's doc().
        assert "_enable_tracing" not in field_names
        assert "runtime" not in result
