# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TypedDict with from __future__ import annotations."""

from __future__ import annotations

from typing import Annotated

from typing_extensions import TypedDict

from nooa.agentdoc._structured import extract_type_info


class ServerConfig(TypedDict):
    host: str
    port: int
    label: Annotated[str, "Human-readable name"]


class PartialConfig(TypedDict, total=False):
    timeout: int
    retries: Annotated[int, "Max retry attempts"]


class TestTypedDictFutureAnnotations:
    def test_required_field_resolves_to_str_not_forwardref(self):
        info = extract_type_info(ServerConfig)
        field = next(f for f in info.fields if f.name == "host")
        assert field.type == "str"
        assert "ForwardRef" not in field.type

    def test_required_field_with_annotation_resolves_description(self):
        info = extract_type_info(ServerConfig)
        field = next(f for f in info.fields if f.name == "label")
        assert field.description == "Human-readable name"
        assert "ForwardRef" not in field.type

    def test_optional_field_resolves_to_int_not_forwardref(self):
        info = extract_type_info(PartialConfig)
        field = next(f for f in info.fields if f.name == "timeout")
        assert field.type == "int"
        assert "ForwardRef" not in field.type

    def test_optional_field_with_annotation_uses_description_not_optional(self):
        info = extract_type_info(PartialConfig)
        field = next(f for f in info.fields if f.name == "retries")
        assert field.description == "Max retry attempts"
