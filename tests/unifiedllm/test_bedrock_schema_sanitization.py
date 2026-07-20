# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Bedrock JSON schema sanitization.

Bedrock Claude rejects JSON schemas containing integer/number constraints
(minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf),
string constraints (minLength, maxLength, pattern), and array constraints
(maxItems, minItems > 1). This module tests that the sanitization function
strips those keywords for Bedrock models while leaving schemas untouched
for other providers.

See: gl-134
"""

import pytest
from pydantic import BaseModel, Field

from nooa.unifiedllm.unifiedllm import (
    _is_bedrock_model,
    _maybe_sanitize_response_format,
    _sanitize_schema_for_bedrock,
)


class Rating(BaseModel):
    """A model with integer constraints that Bedrock rejects."""

    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(..., min_length=1, max_length=500)


class NestedModel(BaseModel):
    """Nested model with constraints at multiple levels."""

    score: int = Field(..., gt=0, lt=100)

    class Inner(BaseModel):
        value: int = Field(..., ge=0, le=10)

    details: Inner


class TestIsBedrockModel:
    """Test Bedrock model string detection."""

    @pytest.mark.parametrize(
        "model",
        [
            "aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            "aws/anthropic/claude-haiku-4-5-v1",
            "openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        ],
    )
    def test_bedrock_models_detected(self, model):
        assert _is_bedrock_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4-20250514",
            "azure/gpt-4o",
            "gemini/gemini-2.0-flash",
        ],
    )
    def test_non_bedrock_models_not_detected(self, model):
        assert _is_bedrock_model(model) is False


class TestSanitizeSchemaForBedrock:
    """Test that forbidden keywords are stripped from JSON schemas."""

    def test_strips_minimum_maximum_from_integer(self):
        schema = Rating.model_json_schema()
        sanitized = _sanitize_schema_for_bedrock(schema)

        rating_props = sanitized["properties"]["rating"]
        assert "minimum" not in rating_props
        assert "maximum" not in rating_props
        # type should be preserved
        assert rating_props["type"] == "integer"

    def test_strips_exclusive_min_max(self):
        schema = NestedModel.model_json_schema()
        sanitized = _sanitize_schema_for_bedrock(schema)

        score_props = sanitized["properties"]["score"]
        assert "exclusiveMinimum" not in score_props
        assert "exclusiveMaximum" not in score_props

    def test_strips_string_length_constraints(self):
        """minLength/maxLength stripped defensively — blog says unsupported."""
        schema = Rating.model_json_schema()
        sanitized = _sanitize_schema_for_bedrock(schema)

        comment_props = sanitized["properties"]["comment"]
        assert "minLength" not in comment_props
        assert "maxLength" not in comment_props

    def test_strips_nested_constraints(self):
        """Constraints in nested $defs are also stripped."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
            },
            "required": ["score"],
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer", "minimum": 0, "maximum": 10},
                    },
                    "required": ["value"],
                },
            },
        }
        sanitized = _sanitize_schema_for_bedrock(schema)

        assert "exclusiveMinimum" not in sanitized["properties"]["score"]
        assert "exclusiveMaximum" not in sanitized["properties"]["score"]
        inner = sanitized["$defs"]["Inner"]
        assert "minimum" not in inner["properties"]["value"]
        assert "maximum" not in inner["properties"]["value"]

    def test_preserves_type_and_required(self):
        schema = Rating.model_json_schema()
        sanitized = _sanitize_schema_for_bedrock(schema)

        assert sanitized["type"] == "object"
        assert "rating" in sanitized["required"]
        assert "comment" in sanitized["required"]
        assert sanitized["properties"]["rating"]["type"] == "integer"
        assert sanitized["properties"]["comment"]["type"] == "string"

    def test_returns_deep_copy(self):
        """Sanitization should not mutate the original schema."""
        schema = Rating.model_json_schema()
        original_rating = schema["properties"]["rating"].copy()
        _sanitize_schema_for_bedrock(schema)
        assert schema["properties"]["rating"] == original_rating

    def test_strips_max_items_preserves_valid_min_items(self):
        """maxItems should be stripped; minItems=1 is valid and preserved."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["tags"],
        }
        sanitized = _sanitize_schema_for_bedrock(schema)
        tags_props = sanitized["properties"]["tags"]
        assert "maxItems" not in tags_props
        assert tags_props["minItems"] == 1  # valid, preserved
        assert tags_props["type"] == "array"
        assert tags_props["items"] == {"type": "string"}

    def test_strips_pattern(self):
        """pattern stripped defensively — blog says unsupported."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": "^[a-z]+@[a-z]+\\.com$",
                }
            },
            "required": ["email"],
        }
        sanitized = _sanitize_schema_for_bedrock(schema)
        assert "pattern" not in sanitized["properties"]["email"]

    def test_strips_multiple_of(self):
        """multipleOf should be stripped."""
        schema = {
            "type": "object",
            "properties": {
                "even_number": {
                    "type": "integer",
                    "multipleOf": 2,
                }
            },
            "required": ["even_number"],
        }
        sanitized = _sanitize_schema_for_bedrock(schema)
        assert "multipleOf" not in sanitized["properties"]["even_number"]

    def test_clamps_min_items_to_1(self):
        """minItems > 1 should be clamped to 1, not stripped."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                }
            },
            "required": ["tags"],
        }
        sanitized = _sanitize_schema_for_bedrock(schema)
        assert sanitized["properties"]["tags"]["minItems"] == 1

    def test_preserves_min_items_0_and_1(self):
        """minItems of 0 or 1 should be preserved."""
        for val in (0, 1):
            schema = {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": val,
                    }
                },
                "required": ["tags"],
            }
            sanitized = _sanitize_schema_for_bedrock(schema)
            assert sanitized["properties"]["tags"]["minItems"] == val

    def test_fixes_additional_properties_true(self):
        """additionalProperties: true should be set to false."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": True,
        }
        sanitized = _sanitize_schema_for_bedrock(schema)
        assert sanitized["additionalProperties"] is False

    def test_preserves_additional_properties_false(self):
        """additionalProperties: false should be left alone."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
        sanitized = _sanitize_schema_for_bedrock(schema)
        assert sanitized["additionalProperties"] is False


class TestMaybeSanitizeResponseFormat:
    """Test the integration point that decides whether to sanitize."""

    def test_bedrock_model_returns_dict(self):
        """Bedrock models should get a sanitized dict response_format."""

        class R(BaseModel):
            rating: int = Field(..., ge=1, le=5)

        result = _maybe_sanitize_response_format("aws/anthropic/bedrock-claude-sonnet-4-5-v1", R)
        assert isinstance(result, dict)
        assert result["type"] == "json_schema"
        schema = result["json_schema"]["schema"]
        assert "minimum" not in schema["properties"]["rating"]
        assert "maximum" not in schema["properties"]["rating"]

    def test_non_bedrock_model_returns_class(self):
        """Non-Bedrock models should get the original Pydantic class."""

        class R(BaseModel):
            rating: int = Field(..., ge=1, le=5)

        result = _maybe_sanitize_response_format("openai/gpt-4o", R)
        assert result is R

    def test_comprehensive_pydantic_model(self):
        """End-to-end: a real Pydantic model with all constraint types."""

        class FullModel(BaseModel):
            rating: int = Field(..., ge=1, le=5)
            score: int = Field(..., gt=0, lt=100)
            factor: int = Field(..., multiple_of=2)
            name: str = Field(..., min_length=1, max_length=50)
            tags: list[str] = Field(..., min_length=1, max_length=10)

        schema = FullModel.model_json_schema()
        sanitized = _sanitize_schema_for_bedrock(schema)

        # All keywords documented as unsupported by Bedrock
        forbidden = {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "maxItems",
            "minLength",
            "maxLength",
            "pattern",
        }

        def _collect_keys(node, found=None):
            if found is None:
                found = set()
            if isinstance(node, dict):
                found.update(node.keys())
                for v in node.values():
                    _collect_keys(v, found)
            elif isinstance(node, list):
                for item in node:
                    _collect_keys(item, found)
            return found

        all_keys = _collect_keys(sanitized)
        present_forbidden = all_keys & forbidden
        assert present_forbidden == set(), f"Forbidden keys still present: {present_forbidden}"
