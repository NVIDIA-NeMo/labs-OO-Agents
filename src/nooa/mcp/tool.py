# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing MCP per-server tool base and factory.

Dynamic per-server tool classes inherit from MCPTool. Each instance manages
its own MCP connection (1-1 mapping between tool and client).
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import copy
import json
import types
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import httpx

from .client import create_mcp_client
from .oauth import handle_mcp_oauth


def _flatten_exceptions(exc: BaseException) -> list[BaseException]:
    """Return leaf exceptions so user-facing MCP connection errors are actionable."""
    if isinstance(exc, ExceptionGroup):
        leaves: list[BaseException] = []
        for child in exc.exceptions:
            leaves.extend(_flatten_exceptions(child))
        return leaves
    return [exc]


def _describe_exception(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        body = ""
        try:
            # If the body is already in memory, render it. Otherwise read it
            # first — a streaming response (streamable-http MCP transports)
            # raises ResponseNotRead on ``.text`` until read, which would
            # otherwise mask the real HTTP error (e.g. a 401) behind a confusing
            # "Attempted to access streaming response content" message.
            if not response.is_stream_consumed:
                response.read()
            body = response.text.strip()
        except Exception:
            body = ""
        if len(body) > 500:
            body = body[:500] + "..."
        suffix = f": {body}" if body else ""
        return f"HTTP {response.status_code} {response.reason_phrase}{suffix}"
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _describe_exceptions(exceptions: Sequence[BaseException]) -> str:
    descriptions = []
    for exc in exceptions:
        descriptions.extend(_describe_exception(leaf) for leaf in _flatten_exceptions(exc))
    # Preserve order but avoid duplicating identical nested task-group failures.
    deduped = list(dict.fromkeys(descriptions))
    return "; ".join(deduped)


@dataclass
class MCPToolSpec:
    """Represents an MCP tool specification from a server.

    Attributes:
        name: Tool name
        description: Tool description
        input_schema: Raw JSON schema dictionary
        required: Set of required parameter names
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    required: set[str] = field(default_factory=set)


def _json_schema_type_to_python_type(json_type: str) -> type:
    """Convert JSON schema type string to Python type.

    Args:
        json_type: JSON schema type (e.g., "string", "integer", "number")

    Returns:
        Python type corresponding to the JSON schema type
    """
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, str)


def _value_to_ast_node(value: Any) -> ast.expr:
    """Convert a JSON-schema default value into an AST expression node."""
    return ast.parse(repr(value), mode="eval").body


def _create_method_from_schema(
    tool_name: str,
    method_name: str,
    description: str,
    input_schema: dict[str, Any],
    required: set[str],
    mcp_tool_class: type,
) -> types.FunctionType:
    """Create a properly typed async method from JSON schema.

    Extracts type information and validation constraints from JSON Schema
    and includes them in the generated method's docstring. Supported constraints:
    - Numeric: minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf
    - String: minLength, maxLength, pattern, format
    - Array: minItems, maxItems, uniqueItems
    - Object: minProperties, maxProperties
    - Enum/Const: enum, const

    See JSON Schema validation spec:
    https://json-schema.org/understanding-json-schema/reference

    Args:
        tool_name: Original tool name (for calling)
        method_name: Python method name (normalized)
        description: Method docstring
        input_schema: Raw JSON schema dictionary (must have "type": "object" and "properties")
        required: Set of required parameter names from JSON Schema
        mcp_tool_class: The MCPTool class (for type annotation)

    Returns:
        An async function with proper type annotations and constraint documentation.
    """
    if not isinstance(input_schema, dict):
        properties = {}
    else:
        properties = input_schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

    # Build parameter annotations and defaults
    annotations: dict[str, Any] = {"return": Any}
    defaults: dict[str, Any] = {}
    param_names: list[str] = []

    # Process each property
    param_constraints: dict[str, list[str]] = {}  # Store constraints for docstring
    for param_name, param_schema in properties.items():
        # Skip if not a valid Python identifier or not a dict
        if not param_name.isidentifier() or not isinstance(param_schema, dict):
            continue

        param_names.append(param_name)

        # Get type
        json_type = param_schema.get("type", "string")
        param_type = _json_schema_type_to_python_type(json_type)

        # Extract description
        param_desc = param_schema.get("description", "")

        # Extract constraints (handle both camelCase and snake_case)
        # Use explicit None checks to handle 0 values correctly
        minimum = (
            param_schema.get("minimum")
            if param_schema.get("minimum") is not None
            else param_schema.get("min")
        )
        maximum = (
            param_schema.get("maximum")
            if param_schema.get("maximum") is not None
            else param_schema.get("max")
        )
        exclusive_minimum = param_schema.get("exclusiveMinimum")
        exclusive_maximum = param_schema.get("exclusiveMaximum")
        multiple_of = param_schema.get("multipleOf")
        pattern = param_schema.get("pattern", "")
        format_str = param_schema.get("format", "")
        enum = param_schema.get("enum")
        const = param_schema.get("const")
        param_default = param_schema.get("default")
        has_default = "default" in param_schema

        constraints = []
        # Numeric constraints
        if minimum is not None:
            constraints.append(f"min={minimum}")
        if maximum is not None:
            constraints.append(f"max={maximum}")
        if exclusive_minimum is not None:
            constraints.append(f"exclusive_min={exclusive_minimum}")
        if exclusive_maximum is not None:
            constraints.append(f"exclusive_max={exclusive_maximum}")
        if multiple_of is not None:
            constraints.append(f"multiple_of={multiple_of}")
        # String constraints
        if pattern:
            # Truncate long patterns
            if len(pattern) > 50:
                pattern = pattern[:47] + "..."
            constraints.append(f"pattern={pattern!r}")
        if format_str:
            constraints.append(f"format={format_str!r}")
        # Enum and const
        if enum:
            enum_values = enum if isinstance(enum, list) else []
            if len(enum_values) <= 5:
                constraints.append(f"enum={enum_values}")
            else:
                constraints.append(
                    f"enum=[{', '.join(map(str, enum_values[:3]))}, ... ({len(enum_values)} total)]"
                )
        if const is not None:
            constraints.append(f"const={const!r}")

        if constraints:
            param_constraints[param_name] = constraints

        # Optionality is driven by the schema's ``required`` list, NOT by the
        # mere presence of a ``default``: a param listed in ``required`` is
        # mandatory even if it carries a default, and a param absent from
        # ``required`` is optional even without one.
        if param_name in required:
            # Required -> mandatory positional argument (no default in the
            # generated signature), regardless of any schema default.
            annotations[param_name] = param_type
        elif has_default:
            # Optional with an explicit default value.
            defaults[param_name] = param_default
            # If default is None, adjust type annotation to include None
            if param_default is None:
                # Create union type: str | None, int | None, etc.
                annotations[param_name] = param_type | type(None)
            else:
                annotations[param_name] = param_type
        else:
            # Optional but no schema default -> synthesize None so the caller
            # may omit it; ``_call_tool`` strips None before sending.
            defaults[param_name] = None
            annotations[param_name] = param_type | type(None)

    # Build param descriptions for docstring
    param_docs = []
    for param_name in param_names:
        param_type = annotations[param_name]
        type_name = param_type.__name__ if hasattr(param_type, "__name__") else str(param_type)

        # Get description from schema
        param_schema = properties.get(param_name, {})
        param_desc = param_schema.get("description", "") if isinstance(param_schema, dict) else ""

        constraints_str = ", ".join(param_constraints.get(param_name, []))
        if constraints_str:
            param_docs.append(f"    {param_name} ({type_name}): {param_desc} ({constraints_str})")
        else:
            param_docs.append(f"    {param_name} ({type_name}): {param_desc}")

    docstring = f"""
{description}

Args:
    {"\n".join(param_docs)}
"""

    # Build function arguments using AST
    args = []
    # Add 'self' parameter
    args.append(ast.arg(arg="self", annotation=None))

    # Add required parameters
    for param_name in param_names:
        if param_name not in defaults:
            param_type = annotations[param_name]
            # Convert type to AST Name or Attribute node
            type_node = _type_to_ast_node(param_type)
            args.append(ast.arg(arg=param_name, annotation=type_node))

    # Add optional parameters with defaults
    for param_name in param_names:
        if param_name in defaults:
            param_type = annotations[param_name]
            type_node = _type_to_ast_node(param_type)
            args.append(ast.arg(arg=param_name, annotation=type_node))

    comprehension = ast.comprehension(
        target=ast.Tuple(
            elts=[ast.Name(id="k", ctx=ast.Store()), ast.Name(id="v", ctx=ast.Store())],
            ctx=ast.Store(),
        ),
        iter=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="locals", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                attr="items",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        ),
        ifs=[
            ast.Compare(
                left=ast.Name(id="k", ctx=ast.Load()),
                ops=[ast.NotEq()],
                comparators=[ast.Constant(value="self")],
            ),
            # Omit optional params whose value equals their default
            # (i.e. caller didn't pass them, so they still have the schema default)
            ast.UnaryOp(
                op=ast.Not(),
                operand=ast.BoolOp(
                    op=ast.And(),
                    values=[
                        # k in _defaults
                        ast.Compare(
                            left=ast.Name(id="k", ctx=ast.Load()),
                            ops=[ast.In()],
                            comparators=[ast.Name(id="_defaults", ctx=ast.Load())],
                        ),
                        # v == _defaults[k]
                        ast.Compare(
                            left=ast.Name(id="v", ctx=ast.Load()),
                            ops=[ast.Eq()],
                            comparators=[
                                ast.Subscript(
                                    value=ast.Name(id="_defaults", ctx=ast.Load()),
                                    slice=ast.Name(id="k", ctx=ast.Load()),
                                    ctx=ast.Load(),
                                )
                            ],
                        ),
                    ],
                ),
            ),
        ],
        is_async=0,
    )

    kwargs_dict_comp = ast.DictComp(
        key=ast.Name(id="k", ctx=ast.Load()),
        value=ast.Name(id="v", ctx=ast.Load()),
        generators=[comprehension],
    )

    kwargs_assign = ast.Assign(
        targets=[ast.Name(id="kwargs", ctx=ast.Store())],
        value=kwargs_dict_comp,
    )

    return_stmt = ast.Return(
        value=ast.Await(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()), attr="_call_tool", ctx=ast.Load()
                ),
                args=[ast.Constant(value=tool_name), ast.Name(id="kwargs", ctx=ast.Load())],
                keywords=[],
            )
        )
    )

    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value=docstring)),
        kwargs_assign,
        return_stmt,
    ]

    func_defaults: list[ast.expr] = []
    for param_name in param_names:
        if param_name in defaults:
            func_defaults.append(_value_to_ast_node(defaults[param_name]))

    decorator_list: list[ast.expr] = []

    # Create async function node
    func_node = ast.AsyncFunctionDef(
        name=method_name,
        args=ast.arguments(
            args=args,
            posonlyargs=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=func_defaults,
        ),
        body=body,
        decorator_list=decorator_list,
        returns=_type_to_ast_node(Any),
        type_params=[],
    )

    module = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(module)

    code = compile(module, filename="<dynamic>", mode="exec")
    namespace: dict[str, Any] = {
        mcp_tool_class.__name__: mcp_tool_class,
        "Any": Any,
        "_defaults": defaults,
        "__builtins__": __builtins__,
    }
    exec(code, namespace)
    func = namespace[method_name]

    # Set annotations
    func.__annotations__ = {
        "return": Any,
        **{name: annotations[name] for name in param_names},
    }

    return func


def _type_to_ast_node(python_type: object) -> ast.expr:
    """Convert a Python type to an AST node.

    Args:
        python_type: Python type to convert to AST (can be simple type or union like str | None)

    Returns:
        AST expression node representing the type (Name node for simple types, BinOp for unions)
    """
    if python_type is Any:
        return ast.Name(id="Any", ctx=ast.Load())
    # Handle union types (X | Y syntax)
    elif isinstance(python_type, types.UnionType):
        # For union types like str | None, create BinOp with BitOr
        args = python_type.__args__ if hasattr(python_type, "__args__") else []
        if len(args) >= 2:
            # Build left-associative: ((a | b) | c) for a | b | c
            left = _type_to_ast_node(args[0])
            for arg in args[1:]:
                right = _type_to_ast_node(arg)
                left = ast.BinOp(left=left, op=ast.BitOr(), right=right)
            return left
        # Fallback if can't parse union
        return ast.Name(id="Any", ctx=ast.Load())
    elif python_type is type(None):
        # NoneType -> None (in type annotations, None is a Constant node)
        return ast.Constant(value=None)
    elif isinstance(python_type, type):
        return ast.Name(id=python_type.__name__, ctx=ast.Load())
    else:
        # Fallback for complex types
        return ast.Name(id="Any", ctx=ast.Load())


def _make_dynamic_class(
    server_name: str,
    tool_specs: list[MCPToolSpec],
    mcp_tool_class: type,
) -> type:
    """Build a subclass of MCPTool with one hint-typed method per tool.

    Class docstring describes the server; each method's docstring is the
    tool description from the MCP server.

    Args:
        server_name: Name of the MCP server (used for class naming)
        tool_specs: List of tool specifications from the MCP server
        mcp_tool_class: Base MCPTool class to subclass

    Returns:
        Dynamically created class with methods for each MCP tool
    """
    methods: dict[str, Any] = {}
    tool_names: list[str] = []

    for spec in tool_specs:
        tool_name = spec.name
        # Method name is the tool name; normalize for Python if needed (e.g. find-references -> find_references)
        method_name = tool_name.replace("-", "_") if not tool_name.isidentifier() else tool_name
        if not method_name or not method_name.isidentifier():
            continue
        tool_names.append(tool_name)

        # Create method with proper signature from JSON schema
        method = _create_method_from_schema(
            tool_name=tool_name,
            method_name=method_name,
            description=spec.description or f"MCP tool: {spec.name}",
            input_schema=spec.input_schema,
            required=spec.required,
            mcp_tool_class=mcp_tool_class,
        )

        methods[method_name] = method

    # Class name from server name, e.g. "language-server" -> "LanguageServerTool"
    class_name = (
        "".join(
            word.capitalize() for word in server_name.replace("-", " ").replace("_", " ").split()
        )
        + "Tool"
    )
    if not class_name.isidentifier():
        class_name = "DynamicMCPServerTools"

    class_doc = f"MCP server '{server_name}'."
    reserved = {"__doc__", "_tool_method_names"}
    collisions = reserved.intersection(methods)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            f"MCP server {server_name!r} exposes tool(s) with reserved method name(s): {names}"
        )
    dynamic_class = type(
        class_name,
        (mcp_tool_class,),
        # ``_tool_method_names`` records exactly the methods this factory
        # generated from the server's tool schemas, so consumers (e.g. an
        # agent-facing registry rendering tools as free functions) can
        # enumerate them without name-heuristic guessing against base-class
        # members. This is the factory's own knowledge — it does not couple
        # MCPManager to any Skill/registry.
        {**methods, "__doc__": class_doc, "_tool_method_names": frozenset(methods)},
    )

    return dynamic_class


def _load_mcp_config(mcp_file: Path | None = None) -> dict[str, dict]:
    """Load MCP server configuration from .mcp.json file.

    Args:
        mcp_file: Path to .mcp.json file (default: .mcp.json in cwd)

    Returns:
        Dictionary mapping server names to their configuration
    """
    mcp_path = mcp_file or Path(".mcp.json")
    if mcp_path.exists():
        try:
            mcp_data = json.loads(mcp_path.read_text())
            return mcp_data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _prepare_server_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy an untrusted server config without interpreting its values.

    MCP configuration can come from a repository-local ``.mcp.json`` or
    ``config.toml``. In particular, ``${VAR}`` placeholders must remain literal
    so repository data cannot read values from the host environment.
    """
    return copy.deepcopy(config)


class MCPTool:
    """Abstract base class for per-server MCP tool instances.

    Each instance manages its own MCP connection (1-1 mapping between tool and client).
    Subclasses or dynamically generated classes add one method per MCP tool.

    Example Usage:
    >>> # Custom child class
    >>> class LanguageServerTool(MCPTool):
    >>>     async def definition(self, filepath: str, line: int) -> Any:
    >>>         return await self._call_tool("definition", {"filepath": filepath, "line": line})
    """

    def __init__(
        self,
        client: Any,
        server_name: str,
        refresh_ctx: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with client and server name.

        Args:
            client: MCP client for this tool instance
            server_name: Name of the server this tool instance targets
            refresh_ctx: Optional context for auto-refreshing OAuth on a 401
                during a tool call (server_url + oauth params + transport bits).
                When present, ``_call_tool`` silently refreshes the access token
                and retries once on a 401 instead of surfacing the raw error.
        """
        self._client = client
        self._server_name = server_name
        self._refresh_ctx = refresh_ctx or {}

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Internal: invoke a tool on the server. Used by generated methods only.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments (optional, defaults to empty dict)

        Returns:
            Tool execution result from the MCP server
        """
        # Strip None values — MCP servers use Pydantic and reject None for optional params
        clean_args = {k: v for k, v in (arguments or {}).items() if v is not None}

        try:
            return await self._invoke(tool_name, clean_args)
        except Exception as exc:
            # A cached access token can expire between connect and call. Surface
            # the *real* cause (a streaming MCP transport otherwise masks a 401
            # behind ResponseNotRead / an opaque TaskGroup ExceptionGroup), and
            # if it's a 401 and we have refresh context, transparently refresh
            # the token and retry ONCE before giving up.
            leaves = _flatten_exceptions(exc)
            is_401 = any(
                isinstance(leaf, httpx.HTTPStatusError) and leaf.response.status_code == 401
                for leaf in leaves
            )
            if is_401 and self._refresh_ctx:
                refreshed = await self._refresh_access_token()
                if refreshed:
                    return await self._invoke(tool_name, clean_args)
            details = _describe_exceptions(leaves)
            if is_401:
                raise RuntimeError(
                    f"MCP tool {tool_name!r} on {self._server_name!r} failed: 401 "
                    f"Unauthorized (token expired and auto-refresh did not succeed). "
                    f"Re-authenticate with `/mcp connect {self._server_name}`."
                    + (f" [{details}]" if details else "")
                ) from exc
            raise RuntimeError(
                f"MCP tool {tool_name!r} on {self._server_name!r} failed"
                + (f": {details}" if details else "")
            ) from exc

    async def _invoke(self, tool_name: str, clean_args: dict[str, Any]) -> Any:
        """Single tool-call attempt against the current client/session."""
        async with self._client.connect_to_server() as session:
            result = await session.call_tool(tool_name, clean_args)

        if hasattr(result, "content") and result.content:
            for content in result.content:
                if hasattr(content, "text"):
                    return content.text
            return result.content

        return result

    async def _refresh_access_token(self) -> bool:
        """Refresh the OAuth access token and rebuild the client. Returns True on success.

        Uses the cached refresh token via ``handle_mcp_oauth`` (use_cache=True),
        so no human interaction is needed when a valid refresh token exists.
        """
        ctx = self._refresh_ctx
        server_url = ctx.get("server_url")
        if not server_url:
            return False
        try:
            from .oauth import handle_mcp_oauth

            token = await handle_mcp_oauth(
                server_url=server_url,
                redirect_uri=ctx.get("redirect_uri", "http://localhost:0/callback"),
                client_id=ctx.get("client_id"),
                scope=ctx.get("scope"),
                open_browser=False,  # unattended refresh — never launch a browser
                manual=False,
                use_cache=True,
            )
            headers = dict(ctx.get("headers") or {})
            headers["Authorization"] = f"{token.token_type} {token.access_token}"
            self._client = create_mcp_client(
                transport=ctx.get("transport"),
                url=server_url,
                command=ctx.get("command"),
                args=ctx.get("args"),
                env=ctx.get("env"),
                headers=headers,
            )
            return True
        except Exception:
            return False


class MCPManager:
    """Manager for creating and connecting to MCP server tool instances.

    Example Usage:
    >>> # List available servers
    >>> servers = MCPManager.list_servers()
    >>> print(servers)  # ["maas-confluence-stg", "langfuse", ...]

    >>> # Auto-generated dynamic class
    >>> tool = MCPManager.create_from_server("language-server")
    >>> await tool.definition(filepath="src/main.py", line=10)
    >>> await tool.find_references(filepath="src/main.py", line=10)

    >>> # With explicit config
    >>> tool = MCPManager.create_from_server("my-server", url="https://...", transport="streamable-http")
    """

    @staticmethod
    def list_servers(
        mcp_file: Path | None = None,
        servers: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """List all available MCP servers from config.

        Args:
            mcp_file: Optional path to a VS Code / Claude-style .mcp.json file.
            servers: Optional inline server config from the TUI config.toml.

        Returns:
            List of configured server names.

        Example:
            >>> servers = MCPManager.list_servers()
            >>> print(servers)  # ["maas-confluence-stg", "langfuse"]
        """
        config = _load_mcp_config(mcp_file)
        config.update(servers or {})
        return list(config.keys())

    @staticmethod
    def create_from_server(
        server_name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        transport: Literal["stdio", "sse", "streamable-http"] | None = None,
        oauth_client_id: str | None = None,
        oauth_redirect_uri: str | None = None,
        oauth_scope: str | None = None,
        oauth_open_browser: bool | None = None,
        oauth_manual: bool | None = None,
        oauth_code_prompt: Callable[[str], Awaitable[str]] | None = None,
        oauth_browser_open: Callable[[str], Awaitable[bool]] | None = None,
        oauth_timeout: float | None = None,
        mcp_file: Path | None = None,
        servers: dict[str, dict[str, Any]] | None = None,
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ) -> MCPTool:
        """Create a per-server tool instance; connects to the MCP server.

        Dynamically generates a class with methods for each tool on the server.

        Args:
            server_name: Server name (used to lookup config in .mcp.json if provided)
            command: Command to run the server (for stdio transport)
            args: Command arguments (for stdio transport)
            env: Environment variables for the server process (for stdio transport)
            url: HTTP endpoint URL (for HTTP transports)
            headers: Optional headers for HTTP requests (for HTTP transports)
            transport: Transport type - "stdio", "sse", or "streamable-http"
            oauth_client_id: OAuth client ID (if OAuth is required)
            oauth_redirect_uri: OAuth redirect URI. Defaults to the server's
                config value, then http://127.0.0.1:0/callback.
            oauth_scope: OAuth scopes (optional)
            oauth_open_browser: Whether to automatically open browser for OAuth. Defaults to config, then True.
            oauth_manual: Use out-of-band OAuth (link + pasted code). Defaults to config, then False.
            oauth_browser_open: Async hook to open the auth URL in a reachable browser (host handoff).
            mcp_file: Path to .mcp.json file (default: .mcp.json in cwd)
            servers: Optional inline server config from the TUI config.toml.
            tool_call_timeout: How long one tool call may take before it fails.
                Raise it for servers whose tools wrap slow work such as an LLM call.

        Returns:
            An MCPTool instance (dynamically generated class with methods for each tool).
        """

        # Helper to run async code synchronously
        def _run_sync(coro):
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            except RuntimeError:
                return asyncio.run(coro)

        # Load config from .mcp.json
        configured_servers = _load_mcp_config(mcp_file)
        configured_servers.update(servers or {})
        config_server = _prepare_server_config(configured_servers.get(server_name, {}))

        # Merge provided args with config (provided args take precedence): start
        # from the config headers, then let caller-supplied headers override them
        # (including Authorization).
        merged_headers = dict(config_server.get("headers") or {})
        merged_headers.update(headers or {})
        headers = merged_headers
        url = url or config_server.get("url")
        raw_transport = transport or config_server.get("transport") or "stdio"
        if raw_transport not in ("stdio", "sse", "streamable-http"):
            raise ValueError(f"Unsupported MCP transport {raw_transport!r}")
        transport = raw_transport
        command = command or config_server.get("command")
        args = args or config_server.get("args")
        env = env or config_server.get("env")
        oauth_client_id = oauth_client_id or config_server.get("oauth_client_id")
        oauth_redirect_uri = oauth_redirect_uri or config_server.get(
            "oauth_redirect_uri", "http://127.0.0.1:0/callback"
        )
        oauth_scope = oauth_scope or config_server.get("oauth_scope")
        oauth_open_browser = bool(
            oauth_open_browser
            if oauth_open_browser is not None
            else config_server.get("oauth_open_browser", True)
        )
        oauth_manual = bool(
            oauth_manual if oauth_manual is not None else config_server.get("oauth_manual", False)
        )

        # Create client and connect
        client = create_mcp_client(
            transport=transport,
            url=url,
            command=command,
            args=args,
            env=env,
            headers=headers,
            tool_call_timeout=tool_call_timeout,
        )

        # Connect and list tools (with OAuth retry if needed)
        tools_result = None
        try:

            async def _connect_and_list():
                async with client.connect_to_server() as session:
                    return await session.list_tools()

            tools_result = _run_sync(_connect_and_list())
        except Exception as e:
            exceptions = e.exceptions if isinstance(e, ExceptionGroup) else [e]

            auth_exceptions = [
                ex
                for ex in exceptions
                if isinstance(ex, httpx.HTTPStatusError) and ex.response.status_code == 401
            ]
            non_auth_exceptions = [ex for ex in exceptions if ex not in auth_exceptions]

            if auth_exceptions:
                try:
                    token = _run_sync(
                        handle_mcp_oauth(
                            server_url=url or config_server.get("url") or "",
                            redirect_uri=oauth_redirect_uri,
                            client_id=oauth_client_id,
                            scope=oauth_scope,
                            open_browser=oauth_open_browser,
                            manual=oauth_manual,
                            code_prompt=oauth_code_prompt,
                            browser_open=oauth_browser_open,
                            timeout=oauth_timeout,
                        )
                    )
                    headers["Authorization"] = f"{token.token_type} {token.access_token}"

                    client = create_mcp_client(
                        transport=transport,
                        url=url,
                        command=command,
                        args=args,
                        env=env,
                        headers=headers,
                        tool_call_timeout=tool_call_timeout,
                    )

                    async def _connect_and_list_retry():
                        async with client.connect_to_server() as session:
                            return await session.list_tools()

                    tools_result = _run_sync(_connect_and_list_retry())
                except Exception as retry_error:
                    all_exceptions = auth_exceptions + [retry_error] + non_auth_exceptions
                    details = _describe_exceptions(all_exceptions)
                    raise RuntimeError(
                        "Connection failed after OAuth authentication"
                        + (f": {details}" if details else "")
                    ) from retry_error

            if non_auth_exceptions:
                if len(non_auth_exceptions) == 1:
                    raise non_auth_exceptions[0] from e
                raise ExceptionGroup(
                    "Non-authentication errors during connection", non_auth_exceptions
                ) from e

        # Parse tools
        assert tools_result is not None, "tools_result must be set by connect or OAuth retry"
        tool_specs = []
        for tool in tools_result.tools:
            input_schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
            required = set(input_schema.get("required", []))
            tool_specs.append(
                MCPToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=input_schema,
                    required=required,
                )
            )

        # Generate dynamic class and create instance. Stash a refresh context so
        # _call_tool can transparently refresh the OAuth token + retry once on a
        # 401 mid-session (cached access tokens expire between connect and call).
        refresh_ctx = {
            "server_url": url or config_server.get("url") or "",
            "redirect_uri": oauth_redirect_uri,
            "client_id": oauth_client_id,
            "scope": oauth_scope,
            "headers": headers,
            "transport": transport,
            "command": command,
            "args": args,
            "env": env,
        }
        dynamic_class = _make_dynamic_class(server_name, tool_specs, MCPTool)
        instance = object.__new__(dynamic_class)
        instance.__init__(client, server_name, refresh_ctx=refresh_ctx)
        return instance
