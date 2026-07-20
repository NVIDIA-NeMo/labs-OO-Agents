# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for discovering referenced types from Pydantic models with forward refs and NewType."""

from typing import NewType

from pydantic import BaseModel

from nooa.agentdoc._discover import discover_referenced_types

# ---------------------------------------------------------------------------
# Bug 1: Forward-ref Pydantic fields not discovered
# ---------------------------------------------------------------------------


class ChildModel(BaseModel):
    value: int = 0


class ParentWithForwardRef(BaseModel):
    """Parent with a forward-ref field that Pydantic resolves at model_rebuild time."""

    child: "ChildModel | None" = None


ParentWithForwardRef.model_rebuild()


class ParentWithDirectRef(BaseModel):
    """Parent with a direct (non-string) annotation — should already work."""

    child: ChildModel | None = None


def test_forward_ref_pydantic_field_discovered():
    """Forward-ref Pydantic fields should appear in referenced types."""
    refs = discover_referenced_types(ParentWithForwardRef)
    ref_names = [t.__name__ for t in refs]
    assert "ChildModel" in ref_names, f"Expected ChildModel in referenced types, got: {ref_names}"


def test_direct_ref_pydantic_field_discovered():
    """Direct-ref Pydantic fields should appear in referenced types (baseline)."""
    refs = discover_referenced_types(ParentWithDirectRef)
    ref_names = [t.__name__ for t in refs]
    assert "ChildModel" in ref_names, f"Expected ChildModel in referenced types, got: {ref_names}"


# ---------------------------------------------------------------------------
# Bug 2: NewType fields not discovered
# ---------------------------------------------------------------------------

UserId = NewType("UserId", int)


class ModelWithNewType(BaseModel):
    """Model using a NewType field."""

    user_id: UserId = UserId(0)
    name: str = ""


def test_newtype_field_format_shows_name():
    """NewType fields should render with their alias name, not the base type."""
    from nooa.agentdoc._structured import extract_type_info

    info = extract_type_info(ModelWithNewType)
    field = next(f for f in info.fields if f.name == "user_id")
    assert field.type == "UserId", f"Expected 'UserId', got: {field.type!r}"
