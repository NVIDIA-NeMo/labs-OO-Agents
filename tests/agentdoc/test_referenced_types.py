# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for referenced types discovery and formatting."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from nooa.agentdoc import doc
from nooa.agentdoc._discover import _is_custom_type, discover_referenced_types


# Test fixtures
class QueryRequest(BaseModel):
    """A database query request."""

    sql: str = Field(description="SQL query to execute")
    limit: int = 100


class QueryResult(BaseModel):
    """Result of a database query."""

    rows: list[dict]
    row_count: int = 0


class DatabaseTool:
    """Database operations tool."""

    def query(self, request: QueryRequest) -> QueryResult:
        """Execute a SQL query."""
        return QueryResult(rows=[])

    def get_count(self) -> int:
        """Get total query count."""
        return 0


class SimpleClass:
    """A class with only builtin types."""

    def process(self, data: str, count: int = 10) -> list[str]:
        """Process data."""
        return [data] * count


def test_discover_referenced_types_basic():
    """Test basic type discovery from a class."""
    discovered = discover_referenced_types(DatabaseTool)

    # Should find QueryRequest and QueryResult
    names = [t.__name__ for t in discovered]
    assert "QueryRequest" in names
    assert "QueryResult" in names
    assert len(discovered) == 2


def test_discover_referenced_types_no_custom_types():
    """Test that builtin types are not discovered."""
    discovered = discover_referenced_types(SimpleClass)

    # Should find nothing (str, int, list are builtins)
    assert len(discovered) == 0


def test_discover_referenced_types_deduplication():
    """Test that duplicate types are deduplicated."""

    class ToolWithDuplicates:
        def method1(self, req: QueryRequest) -> QueryResult:
            return QueryResult(rows=[])

        def method2(self, req: QueryRequest) -> QueryResult:
            return QueryResult(rows=[])

        def method3(self) -> QueryRequest:
            return QueryRequest(sql="", limit=100)

    discovered = discover_referenced_types(ToolWithDuplicates)

    # Should only have 2 unique types despite being used 5 times
    assert len(discovered) == 2
    names = [t.__name__ for t in discovered]
    assert "QueryRequest" in names
    assert "QueryResult" in names


def test_discover_from_field_annotations():
    """Test discovery from class field type annotations."""

    class ToolWithFields:
        """Tool with typed fields."""

        request: QueryRequest
        result: QueryResult
        count: int = 0

    discovered = discover_referenced_types(ToolWithFields)

    names = [t.__name__ for t in discovered]
    assert "QueryRequest" in names
    assert "QueryResult" in names
    assert len(discovered) == 2  # int should be excluded


def test_discover_from_annotated():
    """Test discovery from Annotated type hints."""

    class ToolWithAnnotated:
        def process(
            self, request: Annotated[QueryRequest, "The request to process"]
        ) -> Annotated[QueryResult, "The result"]:
            return QueryResult(rows=[])

    discovered = discover_referenced_types(ToolWithAnnotated)

    names = [t.__name__ for t in discovered]
    assert "QueryRequest" in names
    assert "QueryResult" in names


def test_discover_from_generics():
    """Test discovery from generic type hints."""

    class ToolWithGenerics:
        def batch_process(self, requests: list[QueryRequest]) -> dict[str, QueryResult]:
            return {}

    discovered = discover_referenced_types(ToolWithGenerics)

    names = [t.__name__ for t in discovered]
    assert "QueryRequest" in names
    assert "QueryResult" in names
    # list, dict, str should be excluded


def test_is_custom_type_builtins():
    """Test that builtin types are correctly identified."""
    assert not _is_custom_type(str)
    assert not _is_custom_type(int)
    assert not _is_custom_type(list)
    assert not _is_custom_type(dict)


def test_is_custom_type_custom():
    """Test that custom types are correctly identified."""
    assert _is_custom_type(QueryRequest)
    assert _is_custom_type(QueryResult)
    assert _is_custom_type(DatabaseTool)


def test_doc_includes_referenced_types():
    """Test that doc() output includes Referenced Types section."""
    output = doc(DatabaseTool)

    # Should contain the Referenced Types section
    assert "## Referenced Types" in output

    # Should show the types in concise format (directly under header, no ### subheaders)
    assert "class QueryRequest(BaseModel):" in output
    assert "class QueryResult(BaseModel):" in output

    # Main class should come before referenced types
    main_class_pos = output.find("class DatabaseTool:")
    ref_section_pos = output.find("## Referenced Types")
    assert main_class_pos < ref_section_pos


def test_doc_no_referenced_types_if_none():
    """Test that Referenced Types section is omitted if no custom types."""
    output = doc(SimpleClass)

    # Should NOT contain the Referenced Types section
    assert "## Referenced Types" not in output


def test_referenced_types_no_recursion():
    """Test that referenced types don't show their own referenced types."""

    class OuterRequest(BaseModel):
        inner: QueryRequest

    class OuterTool:
        def process(self, req: OuterRequest) -> QueryResult:
            return QueryResult(rows=[])

    output = doc(OuterTool)

    # Should show OuterRequest in referenced types
    assert "class OuterRequest(BaseModel):" in output

    # But OuterRequest's doc should not recursively show QueryRequest
    # (it should be concise and not have its own "## Referenced Types")
    # Split at the OuterRequest class to check it doesn't have nested referenced types
    outer_request_section = output.split("class OuterRequest(BaseModel):")[1].split("class ")[0]
    assert "## Referenced Types" not in outer_request_section


def test_referenced_types_sorted_alphabetically():
    """Test that referenced types are sorted alphabetically."""

    class ZebraResult(BaseModel):
        data: str

    class AppleRequest(BaseModel):
        query: str

    class MixedTool:
        def method1(self) -> ZebraResult:
            return ZebraResult(data="")

        def method2(self, req: AppleRequest) -> None:
            return None

    output = doc(MixedTool)

    # Find positions in output
    apple_pos = output.find("class AppleRequest(BaseModel):")
    zebra_pos = output.find("class ZebraResult(BaseModel):")

    # AppleRequest should come before ZebraResult (alphabetical)
    assert apple_pos < zebra_pos


def test_doc_callable_includes_referenced_types():
    """doc() on a function/method should show referenced types."""
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        sql: str

    class QueryResult(BaseModel):
        rows: list[dict]

    def query_function(request: QueryRequest) -> QueryResult:
        """Execute a query."""
        return QueryResult(rows=[])

    output = doc(query_function)

    # Should show the function signature with qualified name
    assert "query_function(request: QueryRequest) -> QueryResult" in output
    assert "request: QueryRequest" in output
    assert "-> QueryResult" in output

    # Should show Referenced Types section
    assert "## Referenced Types" in output

    # Should show concise spec for each type (no ### subheaders)
    assert "class QueryRequest(BaseModel):" in output
    assert "sql: str" in output
    assert "class QueryResult(BaseModel):" in output
    assert "rows: list[dict]" in output


def test_doc_callable_concise_no_referenced_types():
    """doc(callable, concise=True) should NOT show referenced types."""
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        sql: str

    def query_function(request: QueryRequest) -> str:
        """Execute a query."""
        return ""

    output = doc(query_function, concise=True)

    # Should show the function signature with qualified name
    assert "query_function(request: QueryRequest) -> str" in output

    # Should NOT show Referenced Types section
    assert "## Referenced Types" not in output
    assert "### QueryRequest" not in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
