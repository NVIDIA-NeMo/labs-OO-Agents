# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agentdoc.visibility public submodule — verifies the public import path works."""

from typing import Annotated


class TestVisibilityPublicImports:
    def test_all_four_functions_importable(self):
        from nooa.agentdoc.visibility import (
            filter_module_globals,
            is_hidden_field,
            is_hidden_method,
            is_hidden_module_variable,
        )

        assert callable(filter_module_globals)
        assert callable(is_hidden_field)
        assert callable(is_hidden_method)
        assert callable(is_hidden_module_variable)

    def test_all_in_dunder_all(self):
        import nooa.agentdoc.visibility as vis

        assert "filter_module_globals" in vis.__all__
        assert "is_hidden_field" in vis.__all__
        assert "is_hidden_method" in vis.__all__
        assert "is_hidden_module_variable" in vis.__all__

    def test_is_hidden_field_via_public_import(self):
        from nooa.agentdoc import hidden
        from nooa.agentdoc.visibility import is_hidden_field

        class MyClass:
            secret: Annotated[str, hidden] = ""
            visible: str = ""

        assert is_hidden_field(MyClass, "secret") is True
        assert is_hidden_field(MyClass, "visible") is False

    def test_is_hidden_method_via_public_import(self):
        from nooa.agentdoc import hidden
        from nooa.agentdoc.visibility import is_hidden_method

        @hidden
        def my_func(): ...

        assert is_hidden_method(my_func) is True

    def test_is_hidden_module_variable_via_public_import(self):
        # Use the private module as a test subject since it has hidden annotations
        import nooa.agentdoc._visibility as mod
        from nooa.agentdoc.visibility import is_hidden_module_variable

        # Just verify it's callable and returns a bool
        result = is_hidden_module_variable(mod, "hidden")
        assert isinstance(result, bool)

    def test_filter_module_globals_via_public_import(self):
        from nooa import agentdoc
        from nooa.agentdoc.visibility import filter_module_globals

        result = filter_module_globals(agentdoc)
        assert isinstance(result, dict)
        assert "spec" in result
        assert "doc" in result
        assert "pformat" in result


class TestIsHiddenFieldMatchesDocOutput:
    def test_is_hidden_field_consistent_with_doc(self):
        """is_hidden_field() must agree with what doc() actually renders."""
        from typing import Annotated

        from nooa.agentdoc import doc, hidden
        from nooa.agentdoc._visibility import is_hidden_field

        class Config:
            visible: str = "ok"
            secret: Annotated[str, hidden] = "shh"

        assert not is_hidden_field(Config, "visible")
        assert is_hidden_field(Config, "secret")

        result = doc(Config)
        assert "visible" in result
        assert "secret" not in result

    def test_is_hidden_field_spec_imperative_consistent_with_doc(self):
        """is_hidden_field() reflects spec() imperative form and doc() agrees."""
        from nooa.agentdoc import doc, spec
        from nooa.agentdoc._visibility import is_hidden_field

        class Model:
            public: str = "a"
            internal: str = "b"

        spec(Model, "internal", hidden=True)

        assert not is_hidden_field(Model, "public")
        assert is_hidden_field(Model, "internal")

        result = doc(Model)
        assert "public" in result
        assert "internal" not in result
