# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for strict-mode schema fixes from review findings."""

from pydantic import BaseModel, create_model

from nooa.unifiedllm.unifiedllm import _clean_schema, _strict_schema_valid


class TestStrictSchemaRequiredArray:
    """Review finding 1: setdefault preserves partial required on nested objects."""

    def test_nested_object_with_optional_fields_forces_all_required(self):
        """Strict mode must force ALL properties into required, even if the
        original schema has optional fields (partial required array)."""

        class Inner(BaseModel):
            name: str
            age: int | None = None  # Optional — NOT in Pydantic's required

        ReturnResult = create_model("ReturnResult", result=(Inner, ...))
        raw = ReturnResult.model_json_schema()
        cleaned = _clean_schema(raw, strict=True)

        # The inner object must have both 'name' and 'age' in required
        inner = cleaned["properties"]["result"]
        assert inner["type"] == "object"
        assert set(inner["required"]) == set(inner["properties"].keys()), (
            f"Strict mode must require ALL properties. "
            f"required={inner['required']}, properties={list(inner['properties'].keys())}"
        )


class TestStrictSchemaValid:
    """Review finding 2: _strict_schema_valid must check required == properties."""

    def test_rejects_partial_required(self):
        """Schema with object that has fewer required than properties is invalid."""
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],  # Missing "age" — invalid for strict
                    "additionalProperties": False,
                }
            },
            "required": ["result"],
            "additionalProperties": False,
        }
        assert not _strict_schema_valid(schema), (
            "_strict_schema_valid should reject schemas where required != properties.keys()"
        )

    def test_accepts_full_required(self):
        """Schema with object that has all properties required is valid."""
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                }
            },
            "required": ["result"],
            "additionalProperties": False,
        }
        assert _strict_schema_valid(schema)

    def test_rejects_missing_type(self):
        """Schema with property lacking type key is invalid (Any type)."""
        schema = {
            "type": "object",
            "properties": {"result": {}},
            "required": ["result"],
            "additionalProperties": False,
        }
        assert not _strict_schema_valid(schema)
