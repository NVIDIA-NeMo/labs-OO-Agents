# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Annotated type handling in agentdoc.

This test suite ensures that:
1. Annotated[T, metadata] types are rendered as just T
2. Pydantic Field constraints are extracted and shown
3. String descriptions from Annotated are extracted
4. Union types are formatted as X | Y instead of Union[X, Y]
"""

from typing import Annotated

from pydantic import BaseModel, Field

from nooa.agentdoc import doc
from nooa.agentdoc._structured import format_type


class TestFormatType:
    """Test the format_type() function with Annotated types."""

    def test_annotated_simple_type(self):
        """Annotated[str, "description"] should format as just str."""
        annotation = Annotated[str, "A string parameter"]
        assert format_type(annotation) == "str"

    def test_annotated_generic_type(self):
        """Annotated[list[int], "description"] should format as list[int]."""
        annotation = Annotated[list[int], "A list of integers"]
        assert format_type(annotation) == "list[int]"

    def test_annotated_union_type(self):
        """Annotated[str | None, "description"] should format as str | None."""
        annotation = Annotated[str | None, "Optional string"]
        result = format_type(annotation)
        assert result == "str | None"

    def test_union_type_formatting(self):
        """Union types should format as X | Y instead of Union[X, Y]."""
        result = format_type(str | None)
        assert result == "str | None"

    def test_nested_generics(self):
        """Nested generics should format correctly."""
        result = format_type(dict[str, list[int]])
        assert result == "dict[str, list[int]]"


class TestPydanticConstraints:
    """Test that Pydantic Field constraints are extracted and shown."""

    def test_numeric_constraints(self):
        """Numeric constraints (ge, le, gt, lt) should be extracted."""

        class Model(BaseModel):
            value: Annotated[int, Field(ge=1, le=100, description="A constrained int")]

        output = doc(Model)
        # Should show: value: int  # A constrained int [≥1, ≤100]
        assert "value: int" in output
        assert "A constrained int [≥1, ≤100]" in output

    def test_gt_constraint(self):
        """Greater-than constraint should be extracted."""

        class Model(BaseModel):
            value: Annotated[float, Field(gt=0, description="Positive float")]

        output = doc(Model)
        assert "value: float" in output
        assert "Positive float [>0]" in output

    def test_string_constraints(self):
        """String length constraints should be extracted."""

        class Model(BaseModel):
            name: Annotated[str, Field(min_length=1, max_length=50, description="A name")]

        output = doc(Model)
        assert "name: str" in output
        assert "A name [min_len=1, max_len=50]" in output

    def test_mixed_constraints(self):
        """Multiple constraint types should be combined."""

        class Model(BaseModel):
            age: Annotated[int, Field(default=0, ge=0, le=150, description="Age in years")]

        output = doc(Model)
        assert "age: int = 0" in output
        assert "Age in years [≥0, ≤150]" in output

    def test_no_constraints(self):
        """Fields without constraints should show just the description."""

        class Model(BaseModel):
            name: Annotated[str, Field(description="Person's name")]

        output = doc(Model)
        assert "name: str" in output
        assert "# Person's name" in output
        # Should not show empty brackets
        assert "[]" not in output


class TestMethodSignatures:
    """Test that method signatures with Annotated parameters are clean."""

    def test_annotated_parameter(self):
        """Method parameters with Annotated should show clean types."""

        class Tool:
            def process(
                self,
                data: Annotated[str, "Input data to process"],
            ) -> Annotated[int, "Number of items processed"]:
                """Process some data."""
                ...

        output = doc(Tool.process)
        # Should show: def Tool.process(self, data: str) -> int:
        assert "Tool.process(self, data: str) -> int:" in output
        # Should NOT show Annotated in the signature
        assert "Annotated" not in output

    def test_multiple_annotated_parameters(self):
        """Multiple Annotated parameters should all be clean."""

        class Tool:
            def insert(
                self,
                table: Annotated[str, "Target table"],
                data: Annotated[dict, "Row data"],
            ) -> Annotated[int, "Row ID"]:
                """Insert a row."""
                ...

        output = doc(Tool.insert)
        assert "Tool.insert(self, table: str, data: dict) -> int:" in output
        assert "Annotated" not in output

    def test_annotated_with_default(self):
        """Annotated parameters with defaults should format correctly."""

        class Tool:
            def query(
                self,
                limit: Annotated[int, "Max results"] = 100,
            ) -> list:
                """Query data."""
                ...

        output = doc(Tool.query)
        assert "Tool.query(self, limit: int = 100) -> list:" in output


class TestPlainClassAnnotatedFields:
    """Test that plain classes with Annotated fields work correctly."""

    def test_class_level_annotated_fields(self):
        """Class-level Annotated fields should show descriptions."""

        class Config:
            """Configuration class."""

            host: Annotated[str, "Server hostname"] = "localhost"
            port: Annotated[int, "Server port"] = 8080

        output = doc(Config)
        assert "host: str = 'localhost'  # Server hostname" in output
        assert "port: int = 8080  # Server port" in output

    def test_annotated_without_default(self):
        """Annotated fields without defaults should still show descriptions."""

        class Config:
            """Configuration class."""

            required_field: Annotated[str, "A required field"]

        output = doc(Config)
        assert "required_field: str  # A required field" in output


class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_nested_annotated_in_pydantic(self):
        """Nested Annotated types in Pydantic models."""

        class Model(BaseModel):
            items: Annotated[list[Annotated[int, "Item value"]], Field(description="List of items")]

        output = doc(Model)
        # Outer Annotated should be stripped, showing list[int]
        assert "items: list[int]" in output

    def test_annotated_with_pydantic_field(self):
        """Annotated with Field should extract constraints from Field."""

        class Model(BaseModel):
            value: Annotated[int, Field(default=10, ge=0, description="A value")]

        output = doc(Model)
        assert "value: int = 10  # A value [≥0]" in output

    def test_union_with_multiple_types(self):
        """Union with multiple types should format correctly."""

        class Model(BaseModel):
            value: Annotated[str | int | None, Field(description="Multi-type field")]

        output = doc(Model)
        # Should show clean union
        assert "value: str | int | None" in output
        assert "# Multi-type field" in output


class TestParameterDescriptionAugmentation:
    """Test that Annotated parameter descriptions are added to docstrings."""

    def test_single_parameter_description(self):
        """Single parameter description should be added as Args section."""

        class Tool:
            def process(
                self,
                data: Annotated[str, "Input data to process"],
            ) -> None:
                """Process some data."""
                ...

        output = doc(Tool.process)
        assert "Args:" in output
        assert "data: Input data to process" in output

    def test_multiple_parameter_descriptions(self):
        """Multiple parameter descriptions should be listed."""

        class Tool:
            def insert(
                self,
                table: Annotated[str, "Target table name"],
                data: Annotated[dict, "Row data to insert"],
            ) -> Annotated[int, "ID of inserted row"]:
                """Insert a row."""
                ...

        output = doc(Tool.insert)
        assert "Args:" in output
        assert "table: Target table name" in output
        assert "data: Row data to insert" in output
        assert "Returns:" in output
        assert "ID of inserted row" in output

    def test_return_description_only(self):
        """Return description without param descriptions should work."""

        class Tool:
            def get_count(self) -> Annotated[int, "Total count of items"]:
                """Get the count."""
                ...

        output = doc(Tool.get_count)
        assert "Returns:" in output
        assert "Total count of items" in output

    def test_preserves_existing_args_section(self):
        """Existing Args section should not be duplicated."""

        class Tool:
            def query(
                self,
                limit: Annotated[int, "Max rows (this should be ignored)"],
            ) -> None:
                """Query data.

                Args:
                    limit: Maximum rows to return (manual description)
                """
                ...

        output = doc(Tool.query)
        # Should have Args section
        assert "Args:" in output
        # Should preserve manual description, not add Annotated one
        assert "manual description" in output
        assert "this should be ignored" not in output

    def test_preserves_existing_returns_section(self):
        """Existing Returns section should not be duplicated."""

        class Tool:
            def compute(self) -> Annotated[int, "Ignored return description"]:
                """Compute result.

                Returns:
                    The computed result (manual description)
                """
                ...

        output = doc(Tool.compute)
        assert "Returns:" in output
        assert "manual description" in output
        assert "Ignored return description" not in output

    def test_mixed_annotated_and_plain_parameters(self):
        """Mix of Annotated and plain parameters should only document Annotated ones."""

        class Tool:
            def process(
                self,
                name: Annotated[str, "Name to process"],
                count: int,
                data: Annotated[dict, "Data to use"],
            ) -> None:
                """Process data."""
                ...

        output = doc(Tool.process)
        assert "Args:" in output
        assert "name: Name to process" in output
        assert "data: Data to use" in output
        # count should be in signature but not in Args section
        assert "count: int" in output  # In signature
        # Extract just the Args section
        if "Args:" in output:
            args_start = output.index("Args:")
            args_section = (
                output[args_start : args_start + 200]
                if len(output) > args_start + 200
                else output[args_start:]
            )
            # count should not be documented in Args section
            assert "count:" not in args_section

    def test_no_augmentation_without_descriptions(self):
        """Methods without Annotated descriptions should not get Args/Returns added."""

        class Tool:
            def simple(self, x: int, y: str) -> bool:
                """A simple method."""
                ...

        output = doc(Tool.simple)
        assert "Args:" not in output
        assert "Returns:" not in output

    def test_empty_docstring_gets_augmented(self):
        """Methods with no docstring should get Args/Returns sections."""

        class Tool:
            def process(
                self,
                data: Annotated[str, "Data to process"],
            ) -> Annotated[int, "Result code"]: ...

        output = doc(Tool.process)
        assert "Args:" in output
        assert "data: Data to process" in output
        assert "Returns:" in output
        assert "Result code" in output

    def test_async_method_parameter_descriptions(self):
        """Async methods should also get parameter descriptions."""

        class Tool:
            async def fetch(
                self,
                url: Annotated[str, "URL to fetch"],
            ) -> Annotated[dict, "Response data"]:
                """Fetch data from URL."""
                ...

        output = doc(Tool.fetch)
        assert "async def" in output and ".fetch" in output
        assert "Args:" in output
        assert "url: URL to fetch" in output
        assert "Returns:" in output
        assert "Response data" in output
