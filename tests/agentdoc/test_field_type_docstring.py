# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agentdoc._pformat._field_type_docstring."""

from __future__ import annotations

from nooa.agentdoc._info import REQUIRED
from nooa.agentdoc._pformat import _field_type_docstring
from nooa.agentdoc._structured import _ClassRef, _InstanceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MyTool:
    """A tool that processes inputs."""


class NoDocTool:
    pass


class MultiLineTool:
    """First line.

    Second paragraph here.
    """


# ---------------------------------------------------------------------------
# Sentinel / ellipsis inputs return None
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringSentinels:
    def test_required_returns_none(self):
        assert _field_type_docstring(REQUIRED, None) is None

    def test_ellipsis_returns_none(self):
        assert _field_type_docstring(..., None) is None

    def test_none_value_returns_none(self):
        assert _field_type_docstring(None, None) is None


# ---------------------------------------------------------------------------
# _InstanceRef — resolved via context_obj
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringInstanceRef:
    def test_resolves_class_from_context(self):
        class Agent:
            tool = MyTool

        ref = _InstanceRef("MyTool")
        result = _field_type_docstring(ref, Agent)
        assert result == "A tool that processes inputs."

    def test_no_context_obj_returns_none(self):
        ref = _InstanceRef("MyTool")
        result = _field_type_docstring(ref, None)
        assert result is None

    def test_class_not_found_anywhere_returns_none(self):
        ref = _InstanceRef("_NonExistentClass_XYZ_12345")
        result = _field_type_docstring(ref, None)
        assert result is None

    def test_class_no_docstring_returns_none(self):
        class Agent:
            no_doc = NoDocTool

        ref = _InstanceRef("NoDocTool")
        result = _field_type_docstring(ref, Agent)
        assert result is None

    def test_only_first_line_returned(self):
        class Agent:
            multi = MultiLineTool

        ref = _InstanceRef("MultiLineTool")
        result = _field_type_docstring(ref, Agent)
        assert result == "First line."


# ---------------------------------------------------------------------------
# _ClassRef — resolved via context_obj
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringClassRef:
    def test_resolves_class_ref(self):
        class Agent:
            ToolClass = MyTool

        ref = _ClassRef("MyTool")
        result = _field_type_docstring(ref, Agent)
        assert result == "A tool that processes inputs."

    def test_class_ref_no_context_returns_none(self):
        ref = _ClassRef("MyTool")
        result = _field_type_docstring(ref, None)
        assert result is None


# ---------------------------------------------------------------------------
# Actual type objects as default
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringTypeDefault:
    def test_class_type_default_with_docstring(self):
        result = _field_type_docstring(MyTool, None)
        assert result == "A tool that processes inputs."

    def test_builtin_type_returns_none(self):
        # We don't want docstrings for str, int, etc.
        result = _field_type_docstring(str, None)
        assert result is None

    def test_class_no_docstring_returns_none(self):
        result = _field_type_docstring(NoDocTool, None)
        assert result is None


# ---------------------------------------------------------------------------
# Plain scalar defaults return None
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringScalars:
    def test_string_default_returns_none(self):
        assert _field_type_docstring("hello", None) is None

    def test_int_default_returns_none(self):
        assert _field_type_docstring(0, None) is None

    def test_bool_default_returns_none(self):
        assert _field_type_docstring(True, None) is None

    def test_float_default_returns_none(self):
        assert _field_type_docstring(3.14, None) is None


# ---------------------------------------------------------------------------
# Non-scalar non-type instances — use type's docstring
# ---------------------------------------------------------------------------


class TestFieldTypeDocstringInstances:
    def test_instance_of_documented_class(self):
        instance = MyTool()
        result = _field_type_docstring(instance, None)
        assert result == "A tool that processes inputs."

    def test_instance_of_undocumented_class_returns_none(self):
        instance = NoDocTool()
        result = _field_type_docstring(instance, None)
        assert result is None
