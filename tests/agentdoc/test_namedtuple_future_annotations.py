# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NamedTuple with from __future__ import annotations."""

from __future__ import annotations

from typing import NamedTuple

from nooa.agentdoc._structured import extract_type_info


class Point(NamedTuple):
    x: int
    y: str


class PointWithDefault(NamedTuple):
    x: int
    label: str = "origin"


class TestNamedTupleFutureAnnotations:
    def test_field_resolves_to_int_not_forwardref(self):
        info = extract_type_info(Point)
        field = next(f for f in info.fields if f.name == "x")
        assert field.type == "int"
        assert "ForwardRef" not in field.type

    def test_field_resolves_to_str_not_forwardref(self):
        info = extract_type_info(Point)
        field = next(f for f in info.fields if f.name == "y")
        assert field.type == "str"
        assert "ForwardRef" not in field.type

    def test_field_with_default_resolves_type(self):
        info = extract_type_info(PointWithDefault)
        field = next(f for f in info.fields if f.name == "label")
        assert field.type == "str"
        assert "ForwardRef" not in field.type
