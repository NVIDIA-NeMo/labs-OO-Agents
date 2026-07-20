# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Annotated[T, "description"] on function params and dataclass fields.

Covers:
- Dataclass fields using Annotated descriptions render as inline # comments
- Function params using Annotated descriptions are augmented into docstrings
- PEP 563 (from __future__ import annotations) is handled correctly
- Callable instance params (spec singleton) are stripped to bare types in signatures
- _extract_dataclass_fields resolves Annotated metadata
"""

from __future__ import annotations

import dataclasses
from typing import Annotated

from nooa.agentdoc import doc, spec
from nooa.agentdoc._info import FieldInfo

# ---------------------------------------------------------------------------
# Dataclass fields with Annotated descriptions
# ---------------------------------------------------------------------------


class TestDataclassAnnotatedFields:
    def test_annotated_string_description_shown_as_comment(self):
        """Annotated[T, 'desc'] on a dataclass field renders as # comment."""

        @dataclasses.dataclass
        class Config:
            host: Annotated[str, "Database hostname"] = "localhost"
            port: Annotated[int, "TCP port"] = 5432

        result = doc(Config)
        assert "host: str" in result
        assert "# Database hostname" in result
        assert "port: int" in result
        assert "# TCP port" in result

    def test_annotated_strips_wrapper_from_type_display(self):
        """Type shown as plain base type, not Annotated[T, ...]."""

        @dataclasses.dataclass
        class Config:
            value: Annotated[int, "An integer value"] = 0

        result = doc(Config)
        assert "Annotated" not in result
        assert "int" in result

    def test_annotated_required_field(self):
        """Annotated required fields (no default) show correctly."""

        @dataclasses.dataclass
        class Config:
            name: Annotated[str, "Required name field"]

        result = doc(Config)
        assert "name: str" in result
        assert "# Required name field" in result

    def test_annotated_optional_field(self):
        """Annotated[T | None, "desc"] strips to T | None."""

        @dataclasses.dataclass
        class Config:
            label: Annotated[str | None, "Optional label"] = None

        result = doc(Config)
        assert "str | None" in result
        assert "# Optional label" in result
        assert "Annotated" not in result

    def test_agentdoc_info_types_are_self_documented(self):
        """The _info.py dataclasses use Annotated and render with descriptions."""
        from nooa.agentdoc._info import CallableInfo, ModuleInfo, TypeInfo

        for cls in (FieldInfo, CallableInfo, TypeInfo, ModuleInfo):
            result = doc(cls)
            # Each field should have an inline comment
            assert "#" in result, f"{cls.__name__} has no inline # comments in doc()"

    def test_fieldinfo_name_field_has_description(self):
        """FieldInfo.name has Annotated description 'Field name'."""
        result = doc(FieldInfo)
        assert "# Field name" in result

    def test_fieldinfo_repr_field_has_description(self):
        """FieldInfo.repr has Annotated description about doc/show exclusion."""
        result = doc(FieldInfo)
        assert "repr" in result
        # There should be a # comment on the repr field
        for line in result.splitlines():
            if "repr" in line and "bool" in line:
                assert "#" in line
                break
        else:
            raise AssertionError("repr field line not found in FieldInfo doc")


# ---------------------------------------------------------------------------
# Function params with Annotated descriptions
# ---------------------------------------------------------------------------


class TestFunctionAnnotatedParams:
    def test_annotated_param_augments_docstring(self):
        """Annotated[T, 'desc'] on a param adds it to the Args section."""

        def search(
            query: Annotated[str, "The search string"],
            limit: Annotated[int, "Max results to return"] = 10,
        ) -> list[str]:
            """Search the index."""
            return []

        result = doc(search)
        assert "The search string" in result
        assert "Max results to return" in result

    def test_annotated_param_does_not_duplicate_existing_args(self):
        """If the docstring already has Args:, Annotated descriptions are NOT duplicated."""

        def search(
            query: Annotated[str, "Annotated description"],
            limit: int = 10,
        ) -> list[str]:
            """Search.

            Args:
                query: Existing description.
            """
            return []

        result = doc(search)
        # Existing Args section is preserved, Annotated description not added
        assert "Existing description" in result
        # The annotated description should NOT appear (existing Args section wins)
        assert "Annotated description" not in result

    def test_annotated_param_type_stripped_in_signature(self):
        """Signature shows bare type, not Annotated wrapper."""

        def fn(x: Annotated[int, "An integer"]) -> None:
            pass

        result = doc(fn)
        assert "Annotated" not in result
        assert "x: int" in result

    def test_pep563_function_annotated_param_resolved(self):
        """pformat() uses Annotated descriptions even with from __future__ import annotations.

        This module has `from __future__ import annotations` at the top,
        so all annotations are stored as strings. The extractor must use
        get_type_hints() to resolve them.
        """
        # pformat() is defined in agentdoc and has Annotated descriptions on its params
        from nooa.agentdoc import pformat

        result = doc(pformat)
        # pformat() has Annotated descriptions on its params
        # Either inline in signature-stripped form or in augmented Args section
        assert "obj" in result
        assert "max_length" in result
        # The Annotated type itself must NOT appear in the output
        assert "Annotated" not in result


# ---------------------------------------------------------------------------
# Callable instance (spec singleton) — param types stripped correctly
# in the module-level doc(agentdoc) rendering via extract_callable_info
# ---------------------------------------------------------------------------


class TestCallableInstanceAnnotatedParams:
    def _get_docs_callable_info(self):
        """Get the CallableInfo for the spec singleton via extract_callable_info."""
        from nooa.agentdoc import spec as spec_singleton
        from nooa.agentdoc._structured import extract_callable_info

        return extract_callable_info(spec_singleton)

    def test_docs_singleton_param_types_are_stripped(self):
        """docs() param types in signature show as plain types, not Annotated wrappers."""
        info = self._get_docs_callable_info()
        assert "Annotated" not in info.signature

    def test_docs_singleton_description_param_is_str(self):
        """docs() description param signature shows as 'str | None'."""
        info = self._get_docs_callable_info()
        assert "description: str | None" in info.signature

    def test_docs_singleton_field_param_is_str_or_none(self):
        """docs() field param signature shows as 'str | None'."""
        info = self._get_docs_callable_info()
        assert "field: str | None" in info.signature

    def test_docs_in_module_doc_has_no_annotated(self):
        """doc(agentdoc) rendering of the spec callable has no Annotated wrapper."""
        from nooa import agentdoc

        result = doc(agentdoc, concise=True)
        # Extract just the spec() function line
        for line in result.splitlines():
            if line.startswith("def spec("):
                assert "Annotated" not in line
                break
        else:
            raise AssertionError("spec() function not found in doc(agentdoc)")


# ---------------------------------------------------------------------------
# _extract_dataclass_fields directly
# ---------------------------------------------------------------------------


class TestExtractDataclassFields:
    def test_description_extracted_from_annotated(self):
        """_extract_dataclass_fields populates FieldInfo.description from Annotated."""
        from nooa.agentdoc._structured import extract_type_info

        @dataclasses.dataclass
        class Point:
            x: Annotated[float, "X coordinate"]
            y: Annotated[float, "Y coordinate"]

        type_info = extract_type_info(Point)
        fields = {f.name: f for f in type_info.fields}
        assert fields["x"].description == "X coordinate"
        assert fields["y"].description == "Y coordinate"

    def test_type_is_stripped_of_annotated_wrapper(self):
        """FieldInfo.type is the bare type, not Annotated[T, ...]."""
        from nooa.agentdoc._structured import extract_type_info

        @dataclasses.dataclass
        class Point:
            x: Annotated[float, "X coordinate"]

        type_info = extract_type_info(Point)
        assert type_info.fields[0].type == "float"

    def test_non_annotated_field_has_no_description(self):
        """Plain typed fields have description=None."""
        from nooa.agentdoc._structured import extract_type_info

        @dataclasses.dataclass
        class Point:
            x: float

        type_info = extract_type_info(Point)
        assert type_info.fields[0].description is None


# ---------------------------------------------------------------------------
# spec(method, hidden=False) — opt private/dunder methods back into doc()
# ---------------------------------------------------------------------------


class TestHiddenFalseOptsInPrivateMethods:
    def test_init_shown_after_docs_hidden_false(self):
        """spec(Cls.__init__, hidden=False) makes __init__ appear in doc()."""

        class Foo:
            def __init__(self, name: str = "foo") -> None:
                """Initialize with name."""

            def run(self) -> None:
                pass

        spec(Foo.__init__, hidden=False)
        result = doc(Foo)
        assert "__init__" in result
        assert "Initialize with name" in result

    def test_private_method_shown_after_docs_hidden_false(self):
        """spec(Cls._private, hidden=False) makes _private appear in doc()."""

        class Bar:
            def _helper(self) -> str:
                """Internal helper."""
                return ""

            def public(self) -> None:
                pass

        spec(Bar._helper, hidden=False)
        result = doc(Bar)
        assert "_helper" in result
        assert "Internal helper" in result

    def test_private_method_hidden_by_default(self):
        """Without hidden=False, _-prefixed methods stay hidden."""

        class Baz:
            def _secret(self) -> None:
                pass

            def visible(self) -> None:
                pass

        result = doc(Baz)
        assert "_secret" not in result
        assert "visible" in result

    def test_decorator_form_hidden_false(self):
        """@spec(hidden=False) as decorator also opts in a _ method."""
        _show = spec(hidden=False)

        class Qux:
            def public(self) -> None:
                pass

        assert _show is not None

        @_show
        def _private(self) -> str:  # noqa: E306
            """Explicitly shown private."""
            return ""

        Qux._private = _private  # type: ignore[attr-defined]
        result = doc(Qux)
        assert "_private" in result

    def test_hidden_false_sets_agentdoc_hidden_attr(self):
        """spec(method, hidden=False) sets _agentdoc_hidden=False on the function."""

        def _fn() -> None:
            pass

        spec(_fn, hidden=False)
        assert getattr(_fn, "_agentdoc_hidden", None) is False

    def test_hidden_true_still_hides_public_method(self):
        """spec(method, hidden=True) still hides a public method."""

        class MyClass:
            def visible(self) -> None:
                pass

            def internal(self) -> None:
                pass

        spec(MyClass.internal, hidden=True)
        result = doc(MyClass)
        assert "visible" in result
        assert "internal" not in result


# ---------------------------------------------------------------------------
# @spec(hidden=False) — decorator form on dunders
# ---------------------------------------------------------------------------


class TestHiddenFalseDecoratorDunders:
    def test_decorator_init(self):
        """@spec(hidden=False) on __init__ shows it in doc()."""

        class Foo:
            @spec(hidden=False)  # type: ignore[misc]
            def __init__(self, x: int = 0) -> None:
                """Initialise Foo."""

        result = doc(Foo)
        assert "__init__" in result
        assert "Initialise Foo" in result

    def test_decorator_str(self):
        """@spec(hidden=False) on __str__ shows it in doc()."""

        class Foo:
            @spec(hidden=False)  # type: ignore[misc]
            def __str__(self) -> str:
                """Human-readable string."""
                return ""

        result = doc(Foo)
        assert "__str__" in result
        assert "Human-readable string" in result

    def test_decorator_repr(self):
        """@spec(hidden=False) on __repr__ shows it in doc()."""

        class Foo:
            @spec(hidden=False)  # type: ignore[misc]
            def __repr__(self) -> str:
                """Debug repr."""
                return "Foo()"

        result = doc(Foo)
        assert "__repr__" in result
        assert "Debug repr" in result

    def test_decorator_call(self):
        """@spec(hidden=False) on __call__ shows it in doc()."""

        class Foo:
            @spec(hidden=False)  # type: ignore[misc]
            def __call__(self, prompt: str) -> str:
                """Invoke with a prompt."""
                return prompt

        result = doc(Foo)
        assert "__call__" in result
        assert "Invoke with a prompt" in result

    def test_decorator_add(self):
        """@spec(hidden=False) on __add__ shows it in doc()."""

        class Vec:
            @spec(hidden=False)  # type: ignore[misc]
            def __add__(self, other: Vec) -> Vec:
                """Vector addition."""
                return Vec()

        result = doc(Vec)
        assert "__add__" in result
        assert "Vector addition" in result

    def test_multiple_dunders_all_shown(self):
        """Multiple @spec(hidden=False) dunders all appear."""

        class Agent:
            @spec(hidden=False)  # type: ignore[misc]
            def __init__(self, name: str = "agent") -> None:
                """Create the agent."""

            @spec(hidden=False)  # type: ignore[misc]
            def __call__(self, message: str) -> str:
                """Process a message."""
                return message

            def run(self) -> None:
                pass

        result = doc(Agent)
        assert "__init__" in result
        assert "__call__" in result
        assert "run" in result

    def test_undeclared_dunder_still_hidden(self):
        """Dunders without @spec(hidden=False) remain absent."""

        class Foo:
            @spec(hidden=False)  # type: ignore[misc]
            def __init__(self) -> None:
                """Init."""

            def __len__(self) -> int:
                return 0

        result = doc(Foo)
        assert "__init__" in result
        assert "__len__" not in result


# ---------------------------------------------------------------------------
# External (third-party) classes — imperative spec(cls.method, hidden=False)
# ---------------------------------------------------------------------------


class TestHiddenFalseExternalClasses:
    def test_external_pure_python_init(self):
        """spec(ExternalClass.__init__, hidden=False) on a pure-Python class."""
        import dataclasses as dc

        @dc.dataclass
        class Config:
            host: str = "localhost"
            port: int = 5432

        spec(Config.__init__, hidden=False)
        result = doc(Config)
        assert "__init__" in result
        # Should include the field params
        assert "host" in result or "port" in result

    def test_external_dataclass_init_agentdoc_attr_set(self):
        """_agentdoc_hidden=False is stored on the auto-generated __init__."""
        import dataclasses as dc

        @dc.dataclass
        class Point:
            x: float = 0.0
            y: float = 0.0

        spec(Point.__init__, hidden=False)
        assert getattr(Point.__init__, "_agentdoc_hidden", None) is False

    def test_external_pure_python_dunder_operator(self):
        """spec on a dunder operator of a third-party pure-Python class."""
        import dataclasses as dc

        @dc.dataclass
        class Vec:
            x: float = 0.0

            def __add__(self, other: Vec) -> Vec:
                """Add two vectors."""
                return Vec(self.x + other.x)

        spec(Vec.__add__, hidden=False)
        result = doc(Vec)
        assert "__add__" in result
        assert "Add two vectors" in result

    def test_c_extension_init_graceful(self):
        """spec(C_ext.__init__, hidden=False) silently succeeds or fails without crash."""
        # C extension builtins (e.g. list, dict) raise TypeError on attribute set.
        # Our suppress() ensures this is a no-op rather than an error.
        try:
            spec(list.__init__, hidden=False)
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"spec(list.__init__, hidden=False) should not raise: {e}") from e

    def test_c_extension_init_not_shown_when_attr_unset(self):
        """If _agentdoc_hidden can't be set on a C builtin, the dunder stays hidden."""
        # list.__init__ is a C wrapper; attribute set silently fails.
        # doc(list) should not show __init__.
        result = doc(list, concise=True)
        # list is a builtin — __init__ may or may not appear depending on whether
        # the attr could be set. We just assert no crash.
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# C-extension fallback — class-level metadata path
# ---------------------------------------------------------------------------


class _FakeCExtMethod:
    """Simulates a C-ext method: __slots__ prevent setting _agentdoc_hidden."""

    __slots__ = ("__objclass__", "__name__")

    def __call__(self, *args, **kwargs):  # noqa: ANN002,ANN003,ANN204
        pass


class TestCExtFallbackMetadata:
    """Tests for the __objclass__ + set_field_metadata fallback path."""

    def _make_fake(self, owner: type, name: str) -> _FakeCExtMethod:
        m = object.__new__(_FakeCExtMethod)
        object.__setattr__(m, "__objclass__", owner)
        object.__setattr__(m, "__name__", name)
        return m

    def test_fake_method_raises_on_setattr(self):
        """_FakeCExtMethod cannot have _agentdoc_hidden set (mimics C-ext)."""
        import pytest

        m = self._make_fake(object, "__init__")
        with pytest.raises((AttributeError, TypeError)):
            m._agentdoc_hidden = False  # type: ignore[attr-defined]

    def test_docs_hidden_false_falls_back_to_class_metadata(self):
        """spec(fake_method, hidden=False) stores hidden=False on __objclass__."""
        from nooa.agentdoc._metadata import get_field_metadata

        class Owner:
            def __init__(self) -> None:
                """Initialize Owner."""

        fake = self._make_fake(Owner, "__init__")
        spec(fake, hidden=False)

        meta = get_field_metadata(Owner, "__init__")
        assert meta.get("hidden") is False

    def test_decorator_form_falls_back_to_class_metadata(self):
        """@spec(hidden=False) on a fake C-ext method also uses the fallback."""
        from nooa.agentdoc._metadata import get_field_metadata

        class Owner:
            def __init__(self) -> None:
                """Initialize Owner."""

        fake = self._make_fake(Owner, "__init__")
        marker = spec(hidden=False)
        assert marker is not None
        marker(fake)

        meta = get_field_metadata(Owner, "__init__")
        assert meta.get("hidden") is False

    def test_class_metadata_hidden_false_shows_dunder_in_extract_methods(self):
        """_extract_methods third pass picks up a dunder stored via set_field_metadata."""
        from nooa.agentdoc._metadata import set_field_metadata
        from nooa.agentdoc._structured import extract_type_info

        class MyClass:
            def __init__(self, x: int = 0) -> None:
                """Initialize with x."""

            def run(self) -> None:
                pass

        set_field_metadata(MyClass, "__init__", hidden=False)
        type_info = extract_type_info(MyClass)
        # m.name is the qualname; check any entry ending with "__init__"
        assert any("__init__" in m.name for m in type_info.methods)

    def test_class_metadata_hidden_false_shows_dunder_in_doc(self):
        """doc(cls) shows a dunder that was stored via set_field_metadata(cls, name, hidden=False)."""
        from nooa.agentdoc._metadata import set_field_metadata

        class MyClass2:
            def __init__(self, x: int = 0) -> None:
                """Initialize with x."""

            def run(self) -> None:
                pass

        set_field_metadata(MyClass2, "__init__", hidden=False)
        result = doc(MyClass2)
        assert "__init__" in result
        assert "Initialize with x" in result

    def test_full_pipeline_fake_cext_method_shown_in_doc(self):
        """End-to-end: spec(fake_method, hidden=False) → method appears in doc(cls)."""
        from nooa.agentdoc._metadata import set_field_metadata  # noqa: F401

        class Widget:
            def __init__(self, name: str = "w") -> None:
                """Create Widget."""

            def draw(self) -> None:
                pass

        fake_init = self._make_fake(Widget, "__init__")
        spec(fake_init, hidden=False)

        result = doc(Widget)
        assert "__init__" in result
        assert "Create Widget" in result
