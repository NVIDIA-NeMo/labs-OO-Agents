# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agentdoc._visibility — hidden markers and filter_module_globals."""

from __future__ import annotations

import sys
import types
from typing import Annotated

import pytest

from nooa.agentdoc._visibility import (
    _Hidden,
    filter_module_globals,
    hidden,
    is_hidden_field,
    is_hidden_method,
    is_hidden_module_variable,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic modules
# ---------------------------------------------------------------------------


def _make_module(name: str, **attrs) -> types.ModuleType:
    """Build a throw-away module with given attributes."""
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _cleanup_module(name: str) -> None:
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# hidden singleton
# ---------------------------------------------------------------------------


class TestHiddenSingleton:
    def test_hidden_is_Hidden_instance(self):
        assert isinstance(hidden, _Hidden)

    def test_repr(self):
        assert repr(hidden) == "hidden"


# ---------------------------------------------------------------------------
# @hidden decorator
# ---------------------------------------------------------------------------


class TestHiddenDecorator:
    def test_marks_function(self):
        @hidden
        def secret():
            pass

        assert getattr(secret, "_agentdoc_hidden", False) is True

    def test_marks_method(self):
        class C:
            @hidden
            def internal(self):
                pass

        assert getattr(C.internal, "_agentdoc_hidden", False) is True

    def test_returns_original_function(self):
        def f():
            return 42

        result = hidden(f)
        assert result is f
        assert result() == 42

    def test_is_hidden_method_true_for_decorated(self):
        @hidden
        def f():
            pass

        assert is_hidden_method(f) is True

    def test_is_hidden_method_false_without_decorator(self):
        def f():
            pass

        assert is_hidden_method(f) is False


# ---------------------------------------------------------------------------
# is_hidden_field — Annotated[T, hidden]
# ---------------------------------------------------------------------------


class TestIsHiddenField:
    def test_hidden_field_via_annotated(self):
        class C:
            secret: Annotated[str, hidden]

        assert is_hidden_field(C, "secret") is True

    def test_non_hidden_field(self):
        class C:
            visible: str

        assert is_hidden_field(C, "visible") is False

    def test_field_not_in_class(self):
        class C:
            x: int

        assert is_hidden_field(C, "nonexistent") is False

    def test_hidden_inherited_field(self):
        class Parent:
            secret: Annotated[str, hidden]

        class Child(Parent):
            pass

        assert is_hidden_field(Child, "secret") is True

    def test_unhide_in_subclass(self):
        """Subclass re-declaring without hidden should unhide the field."""

        class Parent:
            x: Annotated[str, hidden]

        class Child(Parent):
            x: str  # redeclare without hidden

        assert is_hidden_field(Child, "x") is False

    def test_multiple_metadata_in_annotated(self):
        """Hidden marker among other metadata."""

        class C:
            x: Annotated[str, "some_meta", hidden]

        assert is_hidden_field(C, "x") is True

    def test_annotated_without_hidden_marker(self):
        class C:
            x: Annotated[str, "not_hidden"]

        assert is_hidden_field(C, "x") is False


# ---------------------------------------------------------------------------
# is_hidden_module_variable
# ---------------------------------------------------------------------------


class TestIsHiddenModuleVariable:
    def test_hidden_module_var(self):
        mod = _make_module("_test_mod_hidden_var")
        try:
            mod.__dict__["__annotations__"] = {"SECRET": Annotated[str, hidden]}
            mod.__dict__["SECRET"] = "shh"
            assert is_hidden_module_variable(mod, "SECRET") is True
        finally:
            _cleanup_module("_test_mod_hidden_var")

    def test_non_hidden_module_var(self):
        mod = _make_module("_test_mod_visible_var")
        try:
            mod.__dict__["__annotations__"] = {"VISIBLE": str}
            mod.__dict__["VISIBLE"] = "ok"
            assert is_hidden_module_variable(mod, "VISIBLE") is False
        finally:
            _cleanup_module("_test_mod_visible_var")

    def test_unannotated_module_var(self):
        mod = _make_module("_test_mod_unannotated")
        try:
            mod.__dict__["PLAIN"] = 42
            assert is_hidden_module_variable(mod, "PLAIN") is False
        finally:
            _cleanup_module("_test_mod_unannotated")


# ---------------------------------------------------------------------------
# with hidden: context manager
# ---------------------------------------------------------------------------


class TestHiddenContextManager:
    def test_names_defined_inside_are_hidden(self):
        mod = _make_module("_test_mod_ctx")
        try:
            hidden._enter_for_module(mod)
            mod.__dict__["INSIDE"] = "secret"
            hidden.__exit__(None, None, None)
            assert "INSIDE" in mod.__dict__["_agentdoc_hidden_names"]
        finally:
            _cleanup_module("_test_mod_ctx")

    def test_names_before_block_not_hidden(self):
        mod = _make_module("_test_mod_ctx2")
        try:
            mod.__dict__["BEFORE"] = "visible"
            hidden._enter_for_module(mod)
            mod.__dict__["AFTER"] = "secret"
            hidden.__exit__(None, None, None)
            hidden_names = mod.__dict__["_agentdoc_hidden_names"]
            assert "BEFORE" not in hidden_names
            assert "AFTER" in hidden_names
        finally:
            _cleanup_module("_test_mod_ctx2")

    def test_exit_without_enter_raises(self):
        # Create a fresh _Hidden instance to avoid affecting the global `hidden`
        h = _Hidden()
        with pytest.raises(RuntimeError, match="without matching __enter__"):
            h.__exit__(None, None, None)

    def test_nested_blocks(self):
        mod = _make_module("_test_mod_nested")
        try:
            hidden._enter_for_module(mod)
            mod.__dict__["OUTER"] = "o"
            hidden._enter_for_module(mod)
            mod.__dict__["INNER"] = "i"
            hidden.__exit__(None, None, None)
            assert "INNER" in mod.__dict__["_agentdoc_hidden_names"]
            hidden.__exit__(None, None, None)
            assert "OUTER" in mod.__dict__["_agentdoc_hidden_names"]
        finally:
            _cleanup_module("_test_mod_nested")


# ---------------------------------------------------------------------------
# filter_module_globals
# ---------------------------------------------------------------------------


class TestFilterModuleGlobals:
    def test_returns_visible_names(self):
        mod = _make_module("_test_fmg_basic")
        try:
            mod.__dict__["VISIBLE"] = 42

            def visible_func():
                pass

            mod.__dict__["visible_func"] = visible_func
            result = filter_module_globals(mod)
            assert "VISIBLE" in result
            assert "visible_func" in result
        finally:
            _cleanup_module("_test_fmg_basic")

    def test_excludes_dunder_names(self):
        mod = _make_module("_test_fmg_dunder")
        try:
            result = filter_module_globals(mod)
            assert "__name__" not in result
            assert "__doc__" not in result
        finally:
            _cleanup_module("_test_fmg_dunder")

    def test_excludes_agentdoc_internal_names(self):
        mod = _make_module("_test_fmg_internal")
        try:
            mod.__dict__["_agentdoc_hidden_names"] = set()
            result = filter_module_globals(mod)
            assert "_agentdoc_hidden_names" not in result
        finally:
            _cleanup_module("_test_fmg_internal")

    def test_excludes_at_hidden_decorated_functions(self):
        mod = _make_module("_test_fmg_at_hidden")
        try:

            @hidden
            def secret():
                pass

            mod.__dict__["secret"] = secret
            result = filter_module_globals(mod)
            assert "secret" not in result
        finally:
            _cleanup_module("_test_fmg_at_hidden")

    def test_excludes_annotated_hidden_vars(self):
        mod = _make_module("_test_fmg_ann_hidden")
        try:
            mod.__dict__["__annotations__"] = {"API_KEY": Annotated[str, hidden]}
            mod.__dict__["API_KEY"] = "secret"
            result = filter_module_globals(mod)
            assert "API_KEY" not in result
        finally:
            _cleanup_module("_test_fmg_ann_hidden")

    def test_excludes_with_hidden_block_names(self):
        mod = _make_module("_test_fmg_ctx_hidden")
        try:
            hidden._enter_for_module(mod)
            mod.__dict__["ctx_secret"] = "shh"
            hidden.__exit__(None, None, None)
            result = filter_module_globals(mod)
            assert "ctx_secret" not in result
        finally:
            _cleanup_module("_test_fmg_ctx_hidden")

    def test_visible_by_default(self):
        """Everything not explicitly hidden should pass through."""
        mod = _make_module("_test_fmg_visible")
        try:
            mod.__dict__["a"] = 1
            mod.__dict__["b"] = "hello"
            mod.__dict__["c"] = [1, 2, 3]
            result = filter_module_globals(mod)
            assert "a" in result
            assert "b" in result
            assert "c" in result
        finally:
            _cleanup_module("_test_fmg_visible")


# ---------------------------------------------------------------------------
# is_hidden_module_variable — string annotation branch
# ---------------------------------------------------------------------------


class TestIsHiddenModuleVariableStringAnnotation:
    def test_string_annotation_hidden_variable(self):
        """is_hidden_module_variable handles string annotations (from __future__ import annotations)."""
        import types

        mod = types.ModuleType("test_string_ann_mod")
        # Simulate string annotation stored as a string (as if from __future__ import annotations)
        mod.__annotations__ = {"api_key": "Annotated[str, hidden]"}
        # Put hidden and Annotated in the module namespace so eval can resolve them
        from nooa.agentdoc._visibility import hidden as hidden_marker

        mod.hidden = hidden_marker  # type: ignore[attr-defined]
        from typing import Annotated

        mod.Annotated = Annotated  # type: ignore[attr-defined]

        from nooa.agentdoc._visibility import is_hidden_module_variable

        assert is_hidden_module_variable(mod, "api_key") is True

    def test_string_annotation_visible_variable(self):
        """is_hidden_module_variable returns False for non-hidden string annotation."""
        import types

        mod = types.ModuleType("test_string_ann_mod2")
        mod.__annotations__ = {"label": "str"}

        from nooa.agentdoc._visibility import is_hidden_module_variable

        assert is_hidden_module_variable(mod, "label") is False
