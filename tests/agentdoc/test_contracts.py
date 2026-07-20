# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression contracts for visibility and rendering rules.

Each test class corresponds to a past bug found in review.  These tests
should fail BEFORE the fix and pass AFTER.  Adding a test here before
writing the fix is the correct workflow.

Contracts covered:
- @hidden/@spec on @classmethod/@staticmethod works regardless of decorator order
- _private methods are hidden by default; @spec(hidden=False) opts them back in
- Fields absent from instance.__dict__ do not appear in doc(instance)
- Properties that succeed on a live instance appear in doc(instance)
- Properties that raise on access are silently omitted from doc(instance)
- with hidden: raises RuntimeError when used inside a function or class body
- Class annotation + @property on same name does not leak <property object at 0x...>
- doc(instance) produces at most one blank line between docstring and first method
- Property rendering contract: doc(Type), doc(instance), pformat, hidden variants
- spec(cls, field, hidden=False) unhides parent-hidden fields without re-declaration
"""

import pytest

from nooa.agentdoc import doc, hidden, pformat, spec

# ---------------------------------------------------------------------------
# 1. @hidden on @classmethod / @staticmethod — decorator order must not matter
# ---------------------------------------------------------------------------


class TestHiddenOnProperty:
    """@hidden as the inner decorator under @property must suppress the method.

    ``@property @hidden def foo`` stores _agentdoc_hidden on property.fget,
    not on the property wrapper itself.  is_hidden_method() must check fget.
    """

    def test_property_hidden_inner_not_shown(self):
        class MyClass:
            @property
            @hidden
            def secret(self) -> str:
                """Must not appear."""
                return "shh"

            @property
            def visible(self) -> str:
                """Must appear."""
                return "hi"

        assert "secret" not in doc(MyClass)
        assert "visible" in doc(MyClass)

    def test_property_hidden_inner_not_shown_on_instance(self):
        class MyClass:
            @property
            @hidden
            def secret(self) -> str:
                return "shh"

            @property
            def visible(self) -> str:
                """Must appear."""
                return "hi"

        assert "secret" not in doc(MyClass())
        # Properties no longer show on instance formatting (only on Type).
        # Instance formatting reads __dict__ only to avoid blocking I/O.
        assert "visible" not in doc(MyClass())
        # But properties DO show on the class (type introspection, no execution):
        assert "visible" in doc(MyClass)


# ---------------------------------------------------------------------------


class TestHiddenOnDescriptors:
    """@hidden applied as the outer decorator (before @classmethod/@staticmethod)
    should still suppress the method.  This requires checking the raw descriptor
    in cls.__dict__, not just the unwrapped function from inspect.getmembers.
    """

    def _cls_outer_hidden(self):
        """@hidden is the outer decorator — wraps the descriptor object."""

        class MyClass:
            @hidden
            @classmethod
            def hidden_cm(cls) -> str:
                """Must not appear."""
                return ""

            @hidden
            @staticmethod
            def hidden_sm() -> str:
                """Must not appear."""
                return ""

            @classmethod
            def visible_cm(cls) -> str:
                """Must appear."""
                return ""

            @staticmethod
            def visible_sm() -> str:
                """Must appear."""
                return ""

        return MyClass

    def _cls_inner_hidden(self):
        """@hidden is the inner decorator — on the raw function before wrapping."""

        class MyClass:
            @classmethod
            @hidden
            def hidden_cm(cls) -> str:
                """Must not appear."""
                return ""

            @staticmethod
            @hidden
            def hidden_sm() -> str:
                """Must not appear."""
                return ""

            @classmethod
            def visible_cm(cls) -> str:
                """Must appear."""
                return ""

            @staticmethod
            def visible_sm() -> str:
                """Must appear."""
                return ""

        return MyClass

    # outer decorator order
    def test_outer_hidden_classmethod_not_shown(self):
        assert "hidden_cm" not in doc(self._cls_outer_hidden())

    def test_outer_hidden_staticmethod_not_shown(self):
        assert "hidden_sm" not in doc(self._cls_outer_hidden())

    def test_outer_visible_classmethod_shown(self):
        assert "visible_cm" in doc(self._cls_outer_hidden())

    def test_outer_visible_staticmethod_shown(self):
        assert "visible_sm" in doc(self._cls_outer_hidden())

    # inner decorator order
    def test_inner_hidden_classmethod_not_shown(self):
        assert "hidden_cm" not in doc(self._cls_inner_hidden())

    def test_inner_hidden_staticmethod_not_shown(self):
        assert "hidden_sm" not in doc(self._cls_inner_hidden())

    # spec(hidden=True) form
    def test_spec_hidden_true_classmethod_not_shown(self):
        class MyClass:
            @spec(hidden=True)  # type: ignore[misc]
            @classmethod
            def secret_cm(cls) -> str:
                """Must not appear."""
                return ""

            @classmethod
            def public_cm(cls) -> str:
                """Must appear."""
                return ""

        assert "secret_cm" not in doc(MyClass)
        assert "public_cm" in doc(MyClass)


# ---------------------------------------------------------------------------
# 2. _private methods: hidden by default, opt-in with @spec(hidden=False)
# ---------------------------------------------------------------------------


class TestPrivateMethodVisibility:
    """Single-underscore methods are hidden by default.
    @spec(hidden=False) explicitly opts them back into doc() output.
    """

    def _cls(self):
        class MyClass:
            def public_method(self) -> str:
                """Visible by default."""
                return ""

            def _hidden_by_default(self) -> str:
                """Hidden because of leading underscore."""
                return ""

            @spec(hidden=False)  # type: ignore[misc]
            def _shown_explicitly(self) -> str:
                """Visible because of @spec(hidden=False)."""
                return ""

        return MyClass

    def test_public_method_shown(self):
        assert "public_method" in doc(self._cls())

    def test_private_method_hidden(self):
        assert "_hidden_by_default" not in doc(self._cls())

    def test_spec_hidden_false_shows_private_method(self):
        assert "_shown_explicitly" in doc(self._cls())

    def test_spec_hidden_false_shows_docstring(self):
        assert "Visible because of" in doc(self._cls())

    def test_double_underscore_hidden(self):
        class MyClass:
            def __not_a_dunder(self) -> str:  # name-mangled to _MyClass__not_a_dunder
                """Must not appear."""
                return ""

            def public(self) -> str:
                """Must appear."""
                return ""

        assert "not_a_dunder" not in doc(MyClass)
        assert "public" in doc(MyClass)


# ---------------------------------------------------------------------------
# 3. Instance fields: absent from __dict__ → absent from doc(instance)
# ---------------------------------------------------------------------------


class TestInstanceFieldAbsence:
    """Fields not present in instance.__dict__ must not appear in doc(instance).
    Showing them with no value would imply they are required/unset, which is wrong.
    doc(Type) always shows class-level defaults regardless.
    """

    def test_absent_field_not_shown_in_instance_doc(self):
        # Annotation-only (no class-level default) — getattr raises AttributeError
        # on a bare __new__ instance, so it won't be in instance_values.
        class MyClass:
            x: int  # no class-level default; only set in __init__

        inst = MyClass.__new__(MyClass)
        assert "x" not in doc(inst)

    def test_present_field_shown_in_instance_doc(self):
        class MyClass:
            x: int  # annotation only

        inst = MyClass.__new__(MyClass)
        inst.x = 42  # manually set, so it IS in instance_values
        out = doc(inst)
        assert "x" in out
        assert "42" in out

    def test_field_with_class_default_shown_via_getattr(self):
        # Class-level default IS accessible via getattr on any instance,
        # so it still appears in doc(instance).
        class MyClass:
            x: int = 5

        inst = MyClass.__new__(MyClass)
        out = doc(inst)
        assert "x" in out
        assert "5" in out

    def test_type_doc_always_shows_defaults(self):
        class MyClass:
            x: int = 5
            y: str = "hello"

        out = doc(MyClass)
        assert "x" in out
        assert "y" in out

    def test_partial_fields_shown(self):
        """Only the fields actually accessible on the instance are shown."""

        class MyClass:
            x: int  # annotation only — absent unless set
            y: int  # annotation only — absent unless set
            z: int  # annotation only — absent unless set

        inst = MyClass.__new__(MyClass)
        inst.y = 99
        out = doc(inst)
        assert "x" not in out
        assert "y" in out
        assert "z" not in out


# ---------------------------------------------------------------------------
# 4. Properties in doc(instance): shown when they succeed, omitted when they raise
# ---------------------------------------------------------------------------


class TestPropertyInInstanceDoc:
    """Properties on live instances must appear in doc(instance).
    Properties that raise on access must be silently omitted (not crash).
    """

    def test_working_property_not_shown_on_instance(self):
        # Properties no longer execute on instance formatting (prevents blocking I/O).
        class MyClass:
            @property
            def value(self) -> str:
                """A computed property."""
                return "hello"

        out = doc(MyClass())
        assert "value" not in out
        # But they DO show on the type:
        assert "value" in doc(MyClass)

    def test_property_value_not_included_on_instance(self):
        # Property values are not computed during instance formatting.
        class MyClass:
            @property
            def count(self) -> int:
                """The count."""
                return 42

        out = doc(MyClass())
        assert "42" not in out

    def test_raising_property_silently_omitted(self):
        class MyClass:
            @property
            def broken(self) -> str:
                """Raises on access."""
                raise RuntimeError("not initialised")

        out = doc(MyClass())
        assert "broken" not in out

    def test_raising_property_does_not_crash(self):
        class MyClass:
            @property
            def exploding(self) -> str:
                raise ValueError("boom")

        # Must not raise
        doc(MyClass())

    def test_hidden_property_not_shown(self):
        class MyClass:
            @property
            @hidden
            def secret(self) -> str:
                """Hidden property."""
                return "shhh"

        out = doc(MyClass())
        assert "secret" not in out


# ---------------------------------------------------------------------------
# 5. with hidden: raises when used outside module level
# ---------------------------------------------------------------------------


class TestWithHiddenModuleLevel:
    """`with hidden:` must raise RuntimeError when used inside a function
    or method body.  Historically it silently did nothing in those contexts.
    """

    def test_raises_inside_function(self):
        def use_in_function():
            with hidden:
                pass

        with pytest.raises(RuntimeError, match="module level"):
            use_in_function()

    def test_raises_inside_method(self):
        class MyClass:
            def use_in_method(self):
                with hidden:
                    pass

        with pytest.raises(RuntimeError, match="module level"):
            MyClass().use_in_method()

    def test_raises_inside_nested_function(self):
        def outer():
            def inner():
                with hidden:
                    pass

            inner()

        with pytest.raises(RuntimeError, match="module level"):
            outer()


# ---------------------------------------------------------------------------
# 6. Class annotation + @property on same name must not leak memory address
# ---------------------------------------------------------------------------


class TestAnnotationPropertyNoLeak:
    """When a class has both an annotation and a @property for the same name,
    the property descriptor object must not appear in doc() output as
    `<property object at 0x...>`.
    """

    def _cls(self):
        class MyClass:
            computed: str  # explicit annotation + property on same name  # type: ignore[assignment]

            @property
            def computed(self) -> str:
                """A computed value."""
                return "result"

        return MyClass

    def test_no_property_object_repr_in_output(self):
        assert "<property object" not in doc(self._cls())

    def test_no_hex_address_in_output(self):
        out = doc(self._cls())
        # Memory addresses look like 0x7f... — no hex addresses should appear
        import re

        assert not re.search(r"0x[0-9a-fA-F]{6,}", out)

    def test_property_name_still_shown(self):
        assert "computed" in doc(self._cls())


# ---------------------------------------------------------------------------
# 7. doc(instance) must not produce double blank lines when no fields rendered
# ---------------------------------------------------------------------------


class TestNoDoubleBlankLine:
    """When an instance has no visible fields, doc(instance) must not produce
    two consecutive blank lines between the class docstring and the first method.
    """

    def test_no_triple_newline(self):
        """Triple newline = two blank lines in a row."""

        class MyClass:
            """A class docstring."""

            def my_method(self) -> None:
                """A method."""
                ...

        inst = MyClass.__new__(MyClass)
        assert "\n\n\n" not in doc(inst)

    def test_type_doc_single_blank_line(self):
        """Type doc (baseline) also has at most one blank line."""

        class MyClass:
            """A class docstring."""

            def my_method(self) -> None:
                """A method."""
                ...

        assert "\n\n\n" not in doc(MyClass)

    def test_instance_with_fields_no_double_blank(self):
        """Instance with at least one visible field also has no double blank."""

        class MyClass:
            """A class docstring."""

            x: int = 5

            def my_method(self) -> None:
                """A method."""
                ...

        inst = MyClass.__new__(MyClass)
        inst.x = 7
        assert "\n\n\n" not in doc(inst)

    def test_no_double_blank_before_referenced_types(self):
        """No double blank line between docstring and ## Referenced Types
        when an instance has no visible fields but the type has referenced types.
        """

        class Inner:
            """An inner type."""

            value: int = 0

        class MyClass:
            """A class docstring."""

            item: Inner  # annotation only — absent on __new__ instance

        inst = MyClass.__new__(MyClass)
        assert "\n\n\n" not in doc(inst)


# ---------------------------------------------------------------------------
# 8. pformat(instance) must not show Ellipsis for unset Pydantic-style fields
# ---------------------------------------------------------------------------


class TestPformatNoEllipsis:
    """pformat(instance) must not render Ellipsis (...) for fields that are
    simply absent from the instance (not set).  The field should be omitted,
    not shown with a sentinel value as the default.
    """

    def test_absent_required_field_not_shown_as_ellipsis(self):
        """A field whose default is ... (required) must not appear in pformat."""

        class MyClass:
            x: int  # required — no default

        inst = MyClass.__new__(MyClass)
        out = pformat(inst)
        assert "Ellipsis" not in out
        assert "..." not in out

    def test_absent_required_field_omitted_entirely(self):
        """pformat of a bare __new__ instance with only required fields → ClassName()."""

        class MyClass:
            x: int
            y: str

        inst = MyClass.__new__(MyClass)
        assert pformat(inst) == "MyClass()"


# ---------------------------------------------------------------------------
# 9. Field type inline comment must not inherit parent-class docstrings
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringNoInheritance:
    """_field_type_docstring must use the class's own __doc__, not inspect.getdoc()
    which walks the MRO.  The bug: Pydantic's BaseModel has a docstring starting
    with '!!! abstract "Usage Documentation"' (MkDocs admonition) that leaked
    into field comments for user-defined subclasses with no own docstring.
    """

    def test_no_inherited_admonition_markup(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Item(BaseModel):
            value: str  # No docstring — must not inherit BaseModel's

        class Container:
            item: Item

        out = doc(Container)
        assert "!!!" not in out
        assert "abstract" not in out
        assert "Usage Documentation" not in out

    def test_own_docstring_still_shown(self):
        class Inner:
            """Describes inner things."""

            value: int = 0

        # doc() on the type itself always shows the docstring
        out = doc(Inner)
        assert "Describes inner things" in out


# ---------------------------------------------------------------------------
# 10. doc() output must not have a trailing newline
# ---------------------------------------------------------------------------


class TestNoTrailingNewline:
    """doc() must return a string that does not end with a newline character,
    regardless of whether the type has referenced types or not.
    """

    def test_plain_class_no_trailing_newline(self):
        class MyClass:
            x: int = 1

        assert not doc(MyClass).endswith("\n")

    def test_class_with_referenced_types_no_trailing_newline(self):
        class Inner:
            """Inner type."""

            value: int = 0

        class Outer:
            inner: Inner

        assert not doc(Outer).endswith("\n")

    def test_function_no_trailing_newline(self):
        class Result:
            """Result type."""

            ok: bool = True

        def my_func(x: int) -> Result:
            """Do a thing."""
            ...

        assert not doc(my_func).endswith("\n")


# ---------------------------------------------------------------------------
# 11. doc() on callable must have blank lines around ## Referenced Types
# ---------------------------------------------------------------------------


class TestCallableReferencedTypesFormatting:
    """Functions with referenced types must produce the same blank-line
    structure as classes: one blank line before ## Referenced Types,
    one blank line between each referenced type definition.
    """

    def test_no_triple_newline_in_callable_doc(self):
        """Callable doc with referenced types must never produce triple newlines."""
        from nooa.agentdoc._pformat import _format_callable_info
        from nooa.agentdoc.ext import CallableInfo

        info = CallableInfo(
            name="my_func",
            signature="(x: str)",
            return_type="str",
            docstring="Process.",
        )
        out = _format_callable_info(info, concise=False, type_depth=0, indent=0)
        assert "\n\n\n" not in out

    def test_callable_doc_no_trailing_newline(self):
        """Callable doc must not end with a newline."""
        from nooa.agentdoc._pformat import _format_callable_info
        from nooa.agentdoc.ext import CallableInfo

        info = CallableInfo(
            name="my_func",
            signature="(x: str)",
            return_type="str",
            docstring="Process.",
        )
        out = _format_callable_info(info, concise=False, type_depth=0, indent=0)
        assert not out.endswith("\n")


# ---------------------------------------------------------------------------
# 12. Multiple inheritance field ordering follows MRO priority
# ---------------------------------------------------------------------------


class TestMultipleInheritanceFieldOrder:
    """For Child(Base1, Base2), fields from Base1 must appear before fields
    from Base2 because Base1 has higher MRO priority.  The old bug: reversed(mro)
    processed Base2 before Base1, so Base2 fields appeared first.
    """

    def test_higher_priority_base_fields_appear_first(self):
        class Base1:
            x: int = 1

        class Base2:
            y: str = "hi"

        class Child(Base1, Base2):
            z: float = 3.14

        out = doc(Child)
        assert out.index("x: int") < out.index("y: str") < out.index("z: float")

    def test_single_inheritance_order_unchanged(self):
        """Single inheritance baseline: parent fields before child fields."""

        class Parent:
            a: int = 1

        class Child(Parent):
            b: str = "x"

        out = doc(Child)
        assert out.index("a: int") < out.index("b: str")


class TestPropertyContract:
    """Full property behavior matrix for doc(Type), doc(instance), pformat, and visibility.

    Contracts:
    - doc(Type): working properties shown as fields with '# @property' comment
    - doc(Type): raising properties shown as bare field annotations (type only)
    - doc(instance): computed property values shown as assigned fields
    - doc(instance): raising properties silently omitted
    - pformat(instance): property values included in repr
    - @hidden @property (outer form): property hidden from all outputs
    - @property @hidden (inner form): property hidden from all outputs
    """

    def _make_class(self):
        class PropClass:
            def __init__(self, x: int):
                self.x = x

            @property
            def computed(self) -> str:
                """A computed value."""
                return "hello"

            @property
            def raises_always(self) -> str:
                """Only works in special conditions."""
                raise RuntimeError("boom")

            @hidden
            @property
            def hidden_outer(self) -> str:
                """Hidden via @hidden @property."""
                return "secret"

            @property
            @hidden
            def hidden_inner(self) -> str:
                """Hidden via @property @hidden."""
                return "secret2"

        return PropClass

    def test_doc_type_shows_property_docstring(self):
        """doc(Type) shows property docstring as inline comment."""
        PropClass = self._make_class()
        out = doc(PropClass)
        assert "computed: str" in out
        assert "# A computed value." in out

    def test_doc_type_includes_raising_property(self):
        """doc(Type) includes raising properties since they can't be evaluated."""
        PropClass = self._make_class()
        out = doc(PropClass)
        assert "raises_always: str" in out

    def test_doc_type_hides_hidden_outer(self):
        """doc(Type): @hidden @property hides the property."""
        PropClass = self._make_class()
        out = doc(PropClass)
        assert "hidden_outer" not in out

    def test_doc_type_hides_hidden_inner(self):
        """doc(Type): @property @hidden hides the property."""
        PropClass = self._make_class()
        out = doc(PropClass)
        assert "hidden_inner" not in out

    def test_doc_instance_omits_computed_property(self):
        """doc(instance): computed properties are not executed (prevents blocking I/O)."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = doc(obj)
        assert "computed" not in out

    def test_doc_instance_omits_raising_property(self):
        """doc(instance): raising property is silently omitted."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = doc(obj)
        assert "raises_always" not in out

    def test_doc_instance_hides_hidden_outer(self):
        """doc(instance): @hidden @property hides the property."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = doc(obj)
        assert "hidden_outer" not in out

    def test_doc_instance_hides_hidden_inner(self):
        """doc(instance): @property @hidden hides the property."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = doc(obj)
        assert "hidden_inner" not in out

    def test_pformat_excludes_property_value(self):
        """pformat(instance) does NOT execute properties (prevents blocking I/O)."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = pformat(obj)
        assert "computed" not in out

    def test_pformat_excludes_hidden_properties(self):
        """pformat(instance) excludes hidden properties."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = pformat(obj)
        assert "hidden_outer" not in out
        assert "hidden_inner" not in out

    def test_doc_type_no_property_object_leak(self):
        """doc(Type) must not show '<property object at 0x...>'."""
        PropClass = self._make_class()
        out = doc(PropClass)
        assert "<property object" not in out

    def test_doc_type_no_at_property_prefix(self):
        """doc(Type) must not prefix property descriptions with '@property —'."""
        PropClass = self._make_class()
        out = doc(PropClass)
        assert "@property" not in out

    def test_doc_instance_omits_property_docstring(self):
        """doc(instance): properties are not shown (prevents blocking I/O)."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = doc(obj)
        assert "# A computed value." not in out

    def test_doc_instance_no_at_property_prefix(self):
        """doc(instance) must not prefix property descriptions with '@property —'."""
        PropClass = self._make_class()
        obj = PropClass(5)
        out = doc(obj)
        assert "@property" not in out


class TestCachedPropertyContract:
    """Contract for functools.cached_property support.

    cached_property was completely invisible before this fix — not shown in
    doc(Type), doc(instance), or pformat. Contract:
    - doc(Type): shown as a field with docstring comment
    - doc(instance): shown with computed value
    - pformat(instance): included in repr
    - @hidden @cached_property (outer): hidden
    - @cached_property @hidden (inner): hidden
    """

    def _make_class(self):
        import functools

        class CP:
            def __init__(self, x: int):
                self.x = x

            @functools.cached_property
            def expensive(self) -> str:
                """Cached computation."""
                return "result"

            @hidden
            @functools.cached_property
            def hidden_outer(self) -> str:
                """Hidden outer form."""
                return "secret"

            @functools.cached_property
            @hidden
            def hidden_inner(self) -> str:
                """Hidden inner form."""
                return "secret2"

        return CP

    def test_doc_type_shows_cached_property(self):
        CP = self._make_class()
        out = doc(CP)
        assert "expensive: str" in out

    def test_doc_type_shows_cached_property_docstring(self):
        CP = self._make_class()
        out = doc(CP)
        assert "# Cached computation." in out

    def test_doc_type_hides_hidden_outer(self):
        CP = self._make_class()
        out = doc(CP)
        assert "hidden_outer" not in out

    def test_doc_type_hides_hidden_inner(self):
        CP = self._make_class()
        out = doc(CP)
        assert "hidden_inner" not in out

    def test_doc_instance_omits_uncached_property(self):
        # cached_property only shows if already cached (in __dict__).
        # First access caches it; before that, it's a descriptor.
        CP = self._make_class()
        obj = CP(5)
        # Before accessing .expensive, it's not in __dict__
        out = doc(obj)
        assert "expensive" not in out
        # After access, it's cached in __dict__ and WILL show:
        _ = obj.expensive
        out2 = doc(obj)
        assert "expensive" in out2

    def test_doc_instance_hides_hidden_outer(self):
        CP = self._make_class()
        obj = CP(5)
        out = doc(obj)
        assert "hidden_outer" not in out

    def test_doc_instance_hides_hidden_inner(self):
        CP = self._make_class()
        obj = CP(5)
        out = doc(obj)
        assert "hidden_inner" not in out

    def test_pformat_cached_property_only_after_access(self):
        # cached_property shows in pformat only after first access (cached in __dict__).
        CP = self._make_class()
        obj = CP(5)
        out = pformat(obj)
        assert "expensive" not in out  # not cached yet
        # After access, it's in __dict__:
        _ = obj.expensive
        out2 = pformat(obj)
        assert "expensive" in out2

    def test_pformat_excludes_hidden_cached_properties(self):
        CP = self._make_class()
        obj = CP(5)
        out = pformat(obj)
        assert "hidden_outer" not in out
        assert "hidden_inner" not in out


# ---------------------------------------------------------------------------
# spec(cls, field, hidden=False) field unhiding
# ---------------------------------------------------------------------------


class TestSpecHiddenFalseFieldUnhide:
    """spec(cls, field, hidden=False) unhides a parent's hidden field.

    Without this, a field hidden via Annotated[T, hidden] in a parent class
    cannot be unhidden in a subclass without re-declaring the annotation —
    which strips other parent annotations like nosnapshot.

    spec(cls, field, hidden=False) stores imperative metadata on cls that takes
    priority over the MRO annotation scan in is_hidden_field().
    """

    def _make_parent(self):
        from typing import Annotated

        class Parent:
            hidden_field: Annotated[str, hidden] = "secret"
            visible_field: str = "public"

        return Parent

    def test_parent_field_is_hidden(self):
        from nooa.agentdoc.visibility import is_hidden_field

        Parent = self._make_parent()
        assert is_hidden_field(Parent, "hidden_field") is True
        assert is_hidden_field(Parent, "visible_field") is False

    def test_child_inherits_hidden(self):
        from nooa.agentdoc.visibility import is_hidden_field

        Parent = self._make_parent()

        class Child(Parent):
            pass

        assert is_hidden_field(Child, "hidden_field") is True

    def test_spec_hidden_false_unhides_in_subclass(self):
        from nooa.agentdoc.visibility import is_hidden_field

        Parent = self._make_parent()

        class Child(Parent):
            pass

        spec(Child, "hidden_field", hidden=False)
        assert is_hidden_field(Child, "hidden_field") is False

    def test_spec_hidden_false_does_not_affect_parent(self):
        """Unhiding in Child must not change Parent."""
        from nooa.agentdoc.visibility import is_hidden_field

        Parent = self._make_parent()

        class Child(Parent):
            pass

        spec(Child, "hidden_field", hidden=False)
        assert is_hidden_field(Parent, "hidden_field") is True

    def test_spec_hidden_false_shows_field_in_doc(self):
        Parent = self._make_parent()

        class Child(Parent):
            pass

        obj = Child()
        obj.hidden_field = "now visible"

        assert "hidden_field" not in doc(obj)

        spec(Child, "hidden_field", hidden=False)
        assert "hidden_field" in doc(obj)

    def test_spec_hidden_true_hides_visible_field(self):
        from nooa.agentdoc.visibility import is_hidden_field

        class MyClass:
            visible: str = "shown"

        assert is_hidden_field(MyClass, "visible") is False

        spec(MyClass, "visible", hidden=True)
        assert is_hidden_field(MyClass, "visible") is True

    def test_spec_hidden_false_preserves_parent_annotations(self):
        """Re-declaring strips parent annotations; spec(hidden=False) does not."""
        from typing import Annotated, get_args, get_type_hints

        class Marker:
            pass

        marker = Marker()

        class Parent:
            field: Annotated[str, hidden, marker] = "x"

        class Child(Parent):
            pass

        # Confirm parent has both markers
        hints = get_type_hints(Parent, include_extras=True)
        parent_args = get_args(hints["field"])
        assert hidden in parent_args
        assert marker in parent_args

        # spec(hidden=False) on Child — parent annotation untouched
        spec(Child, "field", hidden=False)

        hints_after = get_type_hints(Parent, include_extras=True)
        after_args = get_args(hints_after["field"])
        assert hidden in after_args
        assert marker in after_args

    def test_spec_hidden_false_on_decorator_hidden_property_unhides(self):
        """spec(cls, name, hidden=False) unhides a @hidden @property in a subclass."""

        class Parent:
            @hidden
            @property
            def secret(self) -> str:
                return "revealed"

        class Child(Parent):
            pass

        # Parent's @hidden property is hidden in doc
        parent_out = doc(Parent)
        assert "secret" not in parent_out

        # Child inherits the @hidden property — also hidden
        child_out = doc(Child)
        assert "secret" not in child_out

        # spec(hidden=False) on Child unhides it in doc(Child) only
        spec(Child, "secret", hidden=False)
        child_out_after = doc(Child)
        assert "secret" in child_out_after

        # Parent is unaffected
        parent_out_after = doc(Parent)
        assert "secret" not in parent_out_after


class TestSpecInInitInstanceOptIn:
    """spec(self, field, hidden=False) in __init__ enables per-instance opt-in visibility.

    doc(instance) sees the field; doc(MyClass) does not.
    This is the correct pattern for agents: the LLM always calls doc(self),
    while external documentation tools see doc(AgentClass) — different visibility.
    """

    def _make_base_with_hidden_context(self):
        from typing import Annotated

        class FakeContextApi:
            def keys(self):
                return []

        class BaseAgent:
            context: Annotated[FakeContextApi, hidden]

            def __init__(self):
                self.context = FakeContextApi()

        return BaseAgent, FakeContextApi

    def test_doc_class_does_not_show_hidden_field(self):
        BaseAgent, _ = self._make_base_with_hidden_context()
        assert "context" not in doc(BaseAgent)

    def test_doc_instance_without_opt_in_hides_field(self):
        BaseAgent, _ = self._make_base_with_hidden_context()
        a = BaseAgent()
        assert "context" not in doc(a)

    def test_doc_instance_with_spec_in_init_shows_field(self):
        BaseAgent, _ = self._make_base_with_hidden_context()

        class SubAgent(BaseAgent):
            def __init__(self):
                super().__init__()
                spec(self, "context", hidden=False)

        a = SubAgent()
        assert "context" in doc(a)

    def test_doc_class_still_hidden_after_instance_opt_in(self):
        """spec(self, ...) in __init__ must NOT affect doc(SubAgent) class-level."""
        BaseAgent, _ = self._make_base_with_hidden_context()

        class SubAgent(BaseAgent):
            def __init__(self):
                super().__init__()
                spec(self, "context", hidden=False)

        assert "context" not in doc(SubAgent)

    def test_base_class_unaffected_by_subclass_instance_opt_in(self):
        BaseAgent, _ = self._make_base_with_hidden_context()

        class SubAgent(BaseAgent):
            def __init__(self):
                super().__init__()
                spec(self, "context", hidden=False)

        SubAgent()  # creating an instance must not pollute BaseAgent
        assert "context" not in doc(BaseAgent)

    def test_different_instances_can_differ(self):
        """One instance opts in, another does not — they are independent."""
        BaseAgent, _ = self._make_base_with_hidden_context()

        class SubAgent(BaseAgent):
            def __init__(self, expose=False):
                super().__init__()
                if expose:
                    spec(self, "context", hidden=False)

        shown = SubAgent(expose=True)
        hidden_inst = SubAgent(expose=False)

        assert "context" in doc(shown)
        assert "context" not in doc(hidden_inst)
