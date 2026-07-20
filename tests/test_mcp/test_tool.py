# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for mcp_nooa.tool.MCPTool interface.

Contract-focused: assert public interface (construction, factory method,
generated methods delegate) without depending on implementation details.
"""

import pytest

pytest.importorskip("mcp")

from contextlib import asynccontextmanager  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from nooa.mcp.tool import MCPTool, MCPToolSpec, _make_dynamic_class  # noqa: E402

# ============================================================================
# Helper Functions
# ============================================================================


def _create_mock_client_with_session(mock_session_result: Any = None):
    """Create a mock MCP client with a session that returns the given result."""
    mock_client = MagicMock()
    mock_session = AsyncMock()

    if mock_session_result is not None:
        # If result has content attribute, set it up
        if hasattr(mock_session_result, "content"):
            mock_session.call_tool.return_value = mock_session_result
        else:
            # Create a result object with content
            result = MagicMock()
            if isinstance(mock_session_result, str):
                content_item = MagicMock()
                content_item.text = mock_session_result
                result.content = [content_item]
            else:
                result.content = mock_session_result
            mock_session.call_tool.return_value = result
    else:
        # Default: return a simple result
        result = MagicMock()
        content_item = MagicMock()
        content_item.text = "result"
        result.content = [content_item]
        mock_session.call_tool.return_value = result

    @asynccontextmanager
    async def mock_connect():
        yield mock_session

    mock_client.connect_to_server = mock_connect
    return mock_client, mock_session


# ============================================================================
# MCPTool Constructor Tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_generates_dynamic_tool_class_with_methods():
    """Dynamic class generation creates methods for each MCP tool.

    Verifies that:
    - A dynamic class is created with the correct name (server name -> ClassNameTool)
    - Methods are generated for each tool with proper signatures
    - Method docstrings include parameter descriptions and constraints
    - Generated methods can be called and delegate to client session
    """
    tool_specs = [
        MCPToolSpec(
            "definition",
            "Get definition",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path to the file to get the definition of",
                    },
                    "line": {
                        "type": "integer",
                        "description": "The line number to get the definition of",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["filepath", "line"],
            },
            required={"filepath", "line"},
        ),
        MCPToolSpec(
            "find-references",
            "Find references",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path to the file to find references in",
                    },
                    "line": {
                        "type": "integer",
                        "description": "The line number to find references in",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["filepath", "line"],
            },
            required={"filepath", "line"},
        ),
    ]

    mock_client, mock_session = _create_mock_client_with_session("definition result")

    # Create dynamic class and instance
    dynamic_class = _make_dynamic_class("lang-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "lang-server")

    assert instance is not None
    assert isinstance(instance, MCPTool)
    assert instance._client is mock_client
    assert instance._server_name == "lang-server"
    assert "definition" in dir(instance)
    assert callable(instance.definition)  # type: ignore[attr-defined]
    docstring = instance.definition.__doc__  # type: ignore[attr-defined]
    assert docstring is not None
    assert "Get definition" in docstring
    # Docstring has 4-space indentation for Args section
    assert "filepath (str): The path to the file to get the definition of" in docstring
    assert "line (int): The line number to get the definition of" in docstring
    assert "min=0" in docstring
    assert "max=100" in docstring
    assert "find_references" in dir(instance)
    assert callable(instance.find_references)  # type: ignore[attr-defined]
    docstring = instance.find_references.__doc__  # type: ignore[attr-defined]
    assert docstring is not None
    assert "Find references" in docstring
    assert "filepath (str): The path to the file to find references in" in docstring
    assert "line (int): The line number to find references in" in docstring
    assert "min=0" in docstring
    assert "max=100" in docstring

    result = await instance.definition(filepath="src/main.py", line=10)  # type: ignore[attr-defined]
    assert result == "definition result"
    mock_session.call_tool.assert_awaited_once_with(
        "definition", {"filepath": "src/main.py", "line": 10}
    )

    # Dynamic class name: "lang-server" -> "LangServerTool"
    assert instance.__class__.__name__ == "LangServerTool", (
        f"Expected {instance.__class__.__name__} to be LangServerTool"
    )
    assert issubclass(instance.__class__, MCPTool), (
        f"Expected {instance.__class__.__name__} to be a subclass of MCPTool"
    )


@pytest.mark.asyncio
async def test_create_with_custom_child_class_uses_child_class():
    """Custom child class can be instantiated directly.

    When a child class is defined, it can be instantiated directly
    without dynamic class generation.
    """
    mock_client, _ = _create_mock_client_with_session()

    class LanguageServerTool(MCPTool):
        async def definition(self, filepath: str, line: int) -> Any:
            return await self._call_tool("definition", {"filepath": filepath, "line": line})

    instance = LanguageServerTool(mock_client, "language-server")

    assert isinstance(instance, LanguageServerTool)
    assert instance._client is mock_client
    assert instance._server_name == "language-server"
    assert not hasattr(instance, "ping")  # Only definition method exists


@pytest.mark.asyncio
async def test_generated_method_calls_through_to_mcp_session():
    """Generated tool methods correctly delegate through client to MCP session.

    Integration test to verify that generated methods properly flow through the call chain:
    generated method -> _call_tool() -> client session -> MCP server
    """
    mock_client, mock_session = _create_mock_client_with_session("definition result")

    tool_specs = [
        MCPToolSpec(
            "definition",
            "Get definition",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "",
                    },
                    "line": {
                        "type": "integer",
                        "description": "",
                    },
                },
                "required": ["filepath", "line"],
            },
            required={"filepath", "line"},
        ),
    ]

    dynamic_class = _make_dynamic_class("test_server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test_server")

    assert instance is not None
    assert "definition" in dir(instance), f"definition not in {dir(instance)}"
    assert callable(instance.definition), f"definition is not callable {instance.definition}"

    result = await instance.definition(filepath="x", line=1)  # type: ignore[attr-defined]
    assert result == "definition result"
    mock_session.call_tool.assert_awaited_once_with("definition", {"filepath": "x", "line": 1})


@pytest.mark.asyncio
async def test_parameter_with_default_none_gets_union_type():
    """Parameters with default=None in JSON schema should get str | None = None annotation."""
    tool_specs = [
        MCPToolSpec(
            "search",
            "Search with optional cursor",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor",
                        "default": None,
                    },
                },
                "required": ["query"],
            },
            required={"query"},
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("search result")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    assert instance is not None

    # Check that the method has the correct signature with union type
    import inspect

    sig = inspect.signature(instance.search)
    assert "query" in sig.parameters
    assert "cursor" in sig.parameters

    # cursor should have type str | None and default None
    cursor_param = sig.parameters["cursor"]
    assert cursor_param.default is None
    # Check annotation is union type (str | None)
    import types

    annotation = cursor_param.annotation
    assert isinstance(annotation, types.UnionType), f"Expected union type, got {annotation}"
    assert str in annotation.__args__ and type(None) in annotation.__args__

    # Test calling with cursor=None explicitly — value equals default, so it's omitted
    result = await instance.search(query="test", cursor=None)
    assert result == "search result"
    mock_session.call_tool.assert_awaited_once_with("search", {"query": "test"})

    # Test calling without cursor — also omitted (same behavior)
    mock_session.call_tool.reset_mock()
    result = await instance.search(query="test2")
    mock_session.call_tool.assert_awaited_once_with("search", {"query": "test2"})

    # Test calling with cursor set to a value — sent
    mock_session.call_tool.reset_mock()
    result = await instance.search(query="test3", cursor="abc123")
    mock_session.call_tool.assert_awaited_once_with(
        "search", {"query": "test3", "cursor": "abc123"}
    )


@pytest.mark.asyncio
async def test_parameter_without_default_is_required():
    """Parameters without default should not have a default value (required)."""
    tool_specs = [
        MCPToolSpec(
            "get_item",
            "Get item by ID",
            {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Item identifier",
                    },
                },
                "required": ["item_id"],
            },
            required={"item_id"},
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("item result")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    assert instance is not None

    import inspect

    sig = inspect.signature(instance.get_item)
    item_id_param = sig.parameters["item_id"]
    # Required parameter should not have a default
    assert item_id_param.default is inspect.Parameter.empty
    assert item_id_param.annotation is str

    # Test calling (should work without default)
    result = await instance.get_item(item_id="123")
    assert result == "item result"
    mock_session.call_tool.assert_awaited_once_with("get_item", {"item_id": "123"})


@pytest.mark.asyncio
async def test_parameter_with_non_none_default():
    """Parameters with non-None default should use that default value."""
    tool_specs = [
        MCPToolSpec(
            "list_items",
            "List items with pagination",
            {
                "type": "object",
                "properties": {
                    "page_size": {
                        "type": "integer",
                        "description": "Number of items per page",
                        "default": 10,
                    },
                },
            },
            required=set(),
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("items result")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    assert instance is not None

    import inspect

    sig = inspect.signature(instance.list_items)
    page_size_param = sig.parameters["page_size"]
    assert page_size_param.default == 10
    assert page_size_param.annotation is int

    # Test calling with no args — page_size equals default (10), so it's omitted
    result = await instance.list_items()
    assert result == "items result"
    mock_session.call_tool.assert_awaited_once_with("list_items", {})

    # Test calling with non-default value — sent
    mock_session.call_tool.reset_mock()
    result = await instance.list_items(page_size=25)
    mock_session.call_tool.assert_awaited_once_with("list_items", {"page_size": 25})

    # Test calling with same value as default — still omitted
    mock_session.call_tool.reset_mock()
    result = await instance.list_items(page_size=10)
    mock_session.call_tool.assert_awaited_once_with("list_items", {})


@pytest.mark.asyncio
async def test_optional_parameter_without_default_in_schema():
    """A param absent from ``required`` with no schema default becomes optional.

    Optionality is decided by the schema's ``required`` list, so a parameter
    that is neither required nor default-bearing must still be optional: it
    gets a synthesized ``None`` default and a ``T | None`` annotation, and it
    is omitted from the call when left unset (``_call_tool`` strips None).
    """
    tool_specs = [
        MCPToolSpec(
            "filter",
            "Filter items",
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Filter by name",
                    },
                },
            },
            required=set(),  # name is not required, but no default in schema
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("filter result")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    assert instance is not None

    import inspect
    import types

    sig = inspect.signature(instance.filter)
    name_param = sig.parameters["name"]
    # Not required and no schema default -> optional with synthesized None
    assert name_param.default is None
    annotation = name_param.annotation
    assert isinstance(annotation, types.UnionType), f"Expected union type, got {annotation}"
    assert str in annotation.__args__ and type(None) in annotation.__args__

    # Left unset -> omitted from the call.
    await instance.filter()
    mock_session.call_tool.assert_awaited_once_with("filter", {})

    # Set to a value -> sent.
    mock_session.call_tool.reset_mock()
    await instance.filter(name="widget")
    mock_session.call_tool.assert_awaited_once_with("filter", {"name": "widget"})


@pytest.mark.asyncio
async def test_required_params_always_sent():
    """Required parameters are always included in the call, even with common values."""
    tool_specs = [
        MCPToolSpec(
            "create_item",
            "Create an item",
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Item name",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Item count",
                    },
                },
                "required": ["name", "count"],
            },
            required={"name", "count"},
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("created")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    result = await instance.create_item(name="test", count=0)
    assert result == "created"
    mock_session.call_tool.assert_awaited_once_with("create_item", {"name": "test", "count": 0})


@pytest.mark.asyncio
async def test_mixed_required_and_optional_params():
    """Only optional params with default values are omitted; required params always sent."""
    tool_specs = [
        MCPToolSpec(
            "search",
            "Search items",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor",
                        "default": None,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Results per page",
                        "default": 25,
                    },
                    "max_snippet_size": {
                        "type": "integer",
                        "description": "Max snippet size",
                        "default": 500,
                    },
                },
                "required": ["query"],
            },
            required={"query"},
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("results")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    # Only query — all optional params omitted
    await instance.search(query="test")
    mock_session.call_tool.assert_awaited_once_with("search", {"query": "test"})

    # Override some optional params
    mock_session.call_tool.reset_mock()
    await instance.search(query="test", page_size=10, max_snippet_size=3000)
    mock_session.call_tool.assert_awaited_once_with(
        "search", {"query": "test", "page_size": 10, "max_snippet_size": 3000}
    )


@pytest.mark.asyncio
async def test_required_param_with_default_is_mandatory():
    """A param listed in ``required`` stays mandatory even if it declares a default.

    Optionality is driven by the ``required`` list, not by the presence of a
    ``default``: a required param carrying a schema default must have no default
    in the generated signature (the caller is forced to supply it).
    """
    tool_specs = [
        MCPToolSpec(
            "run",
            "Run with a required-but-defaulted param",
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Execution mode",
                        "default": "fast",  # has a default...
                    },
                },
                "required": ["mode"],  # ...but is required
            },
            required={"mode"},
        )
    ]

    mock_client, mock_session = _create_mock_client_with_session("ran")

    dynamic_class = _make_dynamic_class("test-server", tool_specs, MCPTool)
    instance = dynamic_class(mock_client, "test-server")

    import inspect

    sig = inspect.signature(instance.run)
    mode_param = sig.parameters["mode"]
    # Required -> no default in the signature despite the schema default.
    assert mode_param.default is inspect.Parameter.empty
    assert mode_param.annotation is str

    # Required params are always sent, even when equal to the schema default.
    await instance.run(mode="fast")
    mock_session.call_tool.assert_awaited_once_with("run", {"mode": "fast"})
