# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test how Annotated return types are handled in tool specifications.

This test verifies the current behavior: plain Annotated metadata is NOT
preserved in tool specifications, but Pydantic Field descriptions ARE.
"""

from typing import Annotated

from pydantic import Field, create_model

from nooa.unifiedllm import Tool


class TestAnnotatedReturnTypes:
    """Tests for Annotated return type handling in tool specifications."""

    def test_annotated_metadata_not_preserved_in_tool_schema(self):
        """Annotated metadata (plain strings) is not preserved in tool parameter schema.

        This is expected Pydantic v2 behavior: Annotated[type, "string"] metadata
        is not automatically converted to JSON schema descriptions.

        To add descriptions, use Pydantic Field instead:
        Annotated[str, Field(description="...")]
        """
        return_type = Annotated[str, "A funny name for a bunny"]

        # Create model like _build_return_result_tool does
        ReturnResultModel = create_model(
            "TestReturnResult",
            result=(return_type, ...),
        )

        def return_result(result: str) -> str:
            return result

        tool = Tool(
            name="return_result",
            description="Return the final result",
            callable=return_result,
            parameters_model=ReturnResultModel,
        )

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        # Current behavior: description is NOT present
        assert "description" not in result_field
        assert result_field["type"] == "string"

    def test_pydantic_field_description_is_preserved(self):
        """Pydantic Field descriptions ARE preserved in tool parameter schema.

        This is the recommended way to add descriptions to return types.
        """
        return_type = Annotated[str, Field(description="A funny name for a bunny")]

        ReturnResultModel = create_model(
            "TestReturnResult",
            result=(return_type, ...),
        )

        def return_result(result: str) -> str:
            return result

        tool = Tool(
            name="return_result",
            description="Return the final result",
            callable=return_result,
            parameters_model=ReturnResultModel,
        )

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        # Pydantic Field description IS preserved
        assert "description" in result_field
        assert result_field["description"] == "A funny name for a bunny"
        assert result_field["type"] == "string"

    def test_field_with_create_model_tuple_syntax(self):
        """Field can also be used in create_model tuple syntax."""
        ReturnResultModel = create_model(
            "TestReturnResult",
            result=(str, Field(..., description="Expected output format")),
        )

        def return_result(result: str) -> str:
            return result

        tool = Tool(
            name="return_result",
            description="Return the final result",
            callable=return_result,
            parameters_model=ReturnResultModel,
        )

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        assert "description" in result_field
        assert result_field["description"] == "Expected output format"

    def test_complex_annotated_with_multiple_metadata(self):
        """Multiple Annotated metadata items - only Field is used by Pydantic."""
        return_type = Annotated[
            int,
            "This is a plain string metadata",
            Field(description="This is the actual Field description", ge=0, le=100),
            "Another plain string",
        ]

        ReturnResultModel = create_model(
            "TestReturnResult",
            result=(return_type, ...),
        )

        def return_result(result: int) -> int:
            return result

        tool = Tool(
            name="return_result",
            description="Return the final result",
            callable=return_result,
            parameters_model=ReturnResultModel,
        )

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        # Only the Field metadata is used
        assert result_field["description"] == "This is the actual Field description"
        assert result_field["type"] == "integer"
        # Field constraints should also be present
        assert result_field.get("minimum") == 0
        assert result_field.get("maximum") == 100
