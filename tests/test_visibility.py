# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import types
from typing import Annotated


def test_hidden_decorator_marks_method():
    """@hidden sets _agentdoc_hidden on the function."""
    from nooa.agentdoc import hidden

    @hidden
    def my_method():
        pass

    assert my_method._agentdoc_hidden is True  # type: ignore[attr-defined]


def test_hidden_decorator_preserves_function():
    """@hidden returns the original function, not a wrapper."""
    from nooa.agentdoc import hidden

    def my_method():
        return 42

    result = hidden(my_method)
    assert result is my_method
    assert result() == 42


def test_is_hidden_method_true():
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import is_hidden_method

    @hidden
    def secret():
        pass

    assert is_hidden_method(secret) is True


def test_is_hidden_method_false():
    from nooa.agentdoc.visibility import is_hidden_method

    def public():
        pass

    assert is_hidden_method(public) is False


def test_is_hidden_field_with_annotated():
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import is_hidden_field

    class MyClass:
        secret: Annotated[str, hidden] = ""
        public: str = ""

    assert is_hidden_field(MyClass, "secret") is True
    assert is_hidden_field(MyClass, "public") is False


def test_is_hidden_field_missing_name():
    from nooa.agentdoc.visibility import is_hidden_field

    class MyClass:
        public: str = ""

    assert is_hidden_field(MyClass, "nonexistent") is False


def test_is_hidden_field_subclass_override():
    """Subclass re-declaring without hidden unhides the field."""
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import is_hidden_field

    class Base:
        secret: Annotated[str, hidden] = ""

    class Child(Base):
        secret: str = ""  # unhides

    assert is_hidden_field(Base, "secret") is True
    assert is_hidden_field(Child, "secret") is False


def test_hidden_importable_from_nooa():
    """hidden is importable from the top-level package."""
    from nooa import hidden

    assert callable(hidden)
    assert hasattr(hidden, "__enter__")  # with hidden: context manager


def test_filter_exec_globals_default_includes_all():
    """By default all module names are included (simplified: visible by default)."""
    from nooa.agentdoc.visibility import filter_module_globals

    mod = types.ModuleType("test_mod")
    mod.some_import = "json"  # type: ignore[attr-defined]
    mod.some_const = 42  # type: ignore[attr-defined]
    mod.MyClass = type("MyClass", (), {})  # type: ignore[attr-defined]

    filtered = filter_module_globals(mod)
    assert "some_import" in filtered
    assert "some_const" in filtered
    assert "MyClass" in filtered


def test_filter_exec_globals_excludes_hidden_names_set():
    """Names in _agentdoc_hidden_names are excluded from filtering."""
    from nooa.agentdoc.visibility import filter_module_globals

    mod = types.ModuleType("test_mod")
    mod.public_const = 42  # type: ignore[attr-defined]
    mod.secret_const = 99  # type: ignore[attr-defined]
    mod._agentdoc_hidden_names = {"secret_const"}  # type: ignore[attr-defined]

    filtered = filter_module_globals(mod)
    assert "public_const" in filtered
    assert "secret_const" not in filtered


def test_filter_exec_globals_excludes_hidden_decorated_function():
    """Module-level functions decorated with @hidden are excluded."""
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import filter_module_globals

    mod = types.ModuleType("test_mod")

    def public_fn():
        pass

    @hidden
    def secret_fn():
        pass

    mod.public_fn = public_fn  # type: ignore[attr-defined]
    mod.secret_fn = secret_fn  # type: ignore[attr-defined]

    filtered = filter_module_globals(mod)
    assert "public_fn" in filtered
    assert "secret_fn" not in filtered


def test_filter_exec_globals_excludes_hidden_decorated_class():
    """Module-level classes decorated with @hidden are excluded."""
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import filter_module_globals

    mod = types.ModuleType("test_mod")

    class PublicType:
        pass

    @hidden
    class InternalHelper:
        pass

    mod.PublicType = PublicType  # type: ignore[attr-defined]
    mod.InternalHelper = InternalHelper  # type: ignore[attr-defined]

    filtered = filter_module_globals(mod)
    assert "PublicType" in filtered
    assert "InternalHelper" not in filtered


def test_filter_exec_globals_excludes_annotated_hidden_variable():
    """Module-level variables with Annotated[T, hidden] are excluded."""
    from nooa.agentdoc.visibility import filter_module_globals, is_hidden_module_variable

    mod = types.ModuleType("test_mod")
    mod.__annotations__ = {"api_key": "Annotated[str, hidden]"}  # type: ignore[attr-defined]
    mod.api_key = "secret"  # type: ignore[attr-defined]
    mod.public_var = "ok"  # type: ignore[attr-defined]
    # Ensure hidden is in module namespace for eval of annotation
    import nooa.agentdoc as vis

    mod.hidden = vis.hidden  # type: ignore[attr-defined]
    mod.Annotated = Annotated  # type: ignore[attr-defined]

    assert is_hidden_module_variable(mod, "api_key") is True
    assert is_hidden_module_variable(mod, "public_var") is False

    filtered = filter_module_globals(mod)
    assert "public_var" in filtered
    assert "api_key" not in filtered


def test_with_hidden_context_manager_adds_names_to_hidden_set():
    """Names defined inside ``with hidden:`` are added to _agentdoc_hidden_names."""
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import filter_module_globals

    mod = types.ModuleType("fake_hidden_mod")
    sys.modules["fake_hidden_mod"] = mod
    mod.existing = "before"  # type: ignore[attr-defined]

    hidden._enter_for_module(mod)
    mod.secret_import = "sensitive"  # type: ignore[attr-defined]
    mod.INTERNAL = 999  # type: ignore[attr-defined]
    hidden.__exit__(None, None, None)

    assert getattr(mod, "_agentdoc_hidden_names", set()) >= {"secret_import", "INTERNAL"}
    assert "existing" not in getattr(mod, "_agentdoc_hidden_names", set())

    filtered = filter_module_globals(mod)
    assert "existing" in filtered
    assert "secret_import" not in filtered
    assert "INTERNAL" not in filtered

    del sys.modules["fake_hidden_mod"]


def test_is_hidden_field_with_future_annotations():
    """is_hidden_field works when from __future__ import annotations is active."""
    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import is_hidden_field

    # Simulate what from __future__ import annotations does:
    # annotations are stored as strings at the class level.
    # We test by using get_type_hints which resolves them.
    class MyClass:
        secret: Annotated[str, hidden] = ""
        public: str = ""

    # Simulate string annotations as __future__.annotations would produce
    MyClass.__annotations__ = {
        "secret": "Annotated[str, hidden]",
        "public": "str",
    }

    # is_hidden_field should still resolve correctly via get_type_hints
    # Note: get_type_hints needs the names in the class's module namespace
    # In real usage, hidden and Annotated are importable. Here we patch
    # the module globals to make resolution work.
    import nooa.agentdoc._visibility as vis_mod

    # Store originals and ensure Annotated + hidden are resolvable
    MyClass.__module__ = vis_mod.__name__
    assert is_hidden_field(MyClass, "secret") is True
    assert is_hidden_field(MyClass, "public") is False


def test_is_hidden_field_fallback_when_get_type_hints_fails():
    """When get_type_hints(klass) fails (e.g. parent has unresolvable annotations),
    is_hidden_field falls back to per-field resolution and still detects Annotated[T, hidden].
    """
    from typing import get_type_hints

    from nooa.agentdoc import hidden
    from nooa.agentdoc.visibility import is_hidden_field

    # Create a module that has Annotated and hidden, but also an annotation that
    # references a type that doesn't exist -> get_type_hints(klass) will fail.
    mod = types.ModuleType("test_hidden_fallback_mod")
    mod.__dict__.update({"Annotated": Annotated, "hidden": hidden})
    sys.modules[mod.__name__] = mod

    class Klass:
        secret: Annotated[str, hidden] = ""
        other: "NonexistentType" = None  # type: ignore[name-defined]  # noqa: F821

    Klass.__module__ = mod.__name__
    Klass.__annotations__ = {
        "secret": "Annotated[str, hidden]",
        "other": "NonexistentType",
    }

    # get_type_hints fails because NonexistentType is not in the module
    try:
        get_type_hints(Klass, include_extras=True)
    except NameError:
        pass  # expected
    else:
        raise AssertionError("get_type_hints should have failed")

    # is_hidden_field should still return True for "secret" via _resolve_single_annotation
    assert is_hidden_field(Klass, "secret") is True
    assert is_hidden_field(Klass, "other") is False

    # Cleanup
    del sys.modules[mod.__name__]
