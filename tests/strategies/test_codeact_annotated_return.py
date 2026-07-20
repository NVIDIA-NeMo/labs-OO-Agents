# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that Annotated return types with string descriptions work in return_result tool."""

from typing import Annotated

from pydantic import Field

from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import CompletionClient, Tool


class FakeLLM(CompletionClient):
    """Fake LLM that returns canned responses."""

    def __init__(self):
        super().__init__(model="fake")
        self.responses = []
        self.call_count = 0

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class TestAnnotatedReturnTypeDescriptions:
    """Test that plain string metadata in Annotated return types is preserved in tool schema."""

    def test_extract_annotated_description_plain_string(self):
        """Test extracting plain string description from Annotated."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = Annotated[str, "A funny name for a bunny"]
        base_type, desc = strategy._extract_annotated_description(return_type)

        assert base_type is str
        assert desc == "A funny name for a bunny"

    def test_extract_annotated_description_no_metadata(self):
        """Test plain type without Annotated."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = str
        base_type, desc = strategy._extract_annotated_description(return_type)

        assert base_type is str
        assert desc is None

    def test_extract_annotated_description_with_field(self):
        """Test that Field is not extracted (already structured)."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = Annotated[str, Field(description="Field description")]
        result_type, desc = strategy._extract_annotated_description(return_type)

        # Should return original type and no description (Field already there)
        assert desc is None

    def test_extract_annotated_description_multiple_metadata(self):
        """Test that first string is extracted from multiple metadata items."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = Annotated[int, "Must be positive", 42, {"key": "value"}]
        base_type, desc = strategy._extract_annotated_description(return_type)

        assert base_type is int
        assert desc == "Must be positive"

    def test_build_return_result_tool_with_annotated_description(self):
        """Test that _build_return_result_tool creates tool with description from Annotated."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = Annotated[str, "A funny name for a bunny"]
        tool = strategy._build_return_result_tool(return_type, "my_method")

        # Check tool was created
        assert isinstance(tool, Tool)
        assert tool.name == "return_result"

        # Check the parameter schema has the description
        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        assert "description" in result_field
        assert result_field["description"] == "A funny name for a bunny"
        assert result_field["type"] == "string"

    def test_build_return_result_tool_without_description(self):
        """Test that plain types work as before (no description)."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = str
        tool = strategy._build_return_result_tool(return_type, "my_method")

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        # Should not have description
        assert "description" not in result_field
        assert result_field["type"] == "string"

    def test_build_return_result_tool_with_field(self):
        """Test that existing Field descriptions are preserved."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = Annotated[str, Field(description="Field description")]
        tool = strategy._build_return_result_tool(return_type, "my_method")

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        # Field description should be preserved
        assert "description" in result_field
        assert result_field["description"] == "Field description"

    def test_build_return_result_tool_complex_type_with_description(self):
        """Test Annotated with complex types like list[str]."""
        strategy = CodeActStrategy(config=CodeActConfig())

        return_type = Annotated[list[str], "List of generated names"]
        tool = strategy._build_return_result_tool(return_type, "generate_names")

        schema = tool.get_parameter_schema()
        result_field = schema.get("properties", {}).get("result", {})

        assert "description" in result_field
        assert result_field["description"] == "List of generated names"
        assert result_field["type"] == "array"
        assert result_field["items"]["type"] == "string"
