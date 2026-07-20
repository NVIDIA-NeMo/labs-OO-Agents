# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ellipsis detection utilities for agent method bodies.

This module provides functions to detect and extract code from functions
that use `...` as a marker for LLM code generation.

Functions:
    has_ellipsis_body: Check if function body ENDS with `...`
    get_pre_ellipsis_code: Extract setup code before `...`
"""

import ast
import inspect
import textwrap
import tokenize
from collections.abc import Callable
from typing import Any


def _get_function_ast(func: Callable[..., Any]) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Get the AST node for a function.

    Handles both regular source-based functions and dynamically-generated ones
    that have _generated_source attached.

    Returns:
        The FunctionDef or AsyncFunctionDef node, or None if not found.
    """
    source = None

    # Try to get source from inspect
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
    except (OSError, IndentationError, SyntaxError, tokenize.TokenError):
        # Fall back to _generated_source for dynamically generated functions
        if hasattr(func, "_generated_source"):
            source = getattr(func, "_generated_source")  # noqa: B009

    if source is None:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # Find the function definition in the AST
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func.__name__:
                return node

    return None


def _get_body_without_docstring(
    body: list[ast.stmt],
) -> list[ast.stmt]:
    """Return function body with docstring removed if present."""
    if (
        len(body) > 0
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_ellipsis_stmt(stmt: ast.stmt) -> bool:
    """Check if a statement is an ellipsis expression."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is ...
    )


def has_ellipsis_body(func: Callable[..., Any]) -> bool:
    """Check if a function body ends with `...` (ellipsis).

    This returns True if the function ends with ellipsis, even if there's
    other setup code before it (e.g., variable initialization).

    Args:
        func: Function to check

    Returns:
        True if body ends with ellipsis (may have code before it)
    """
    func_def = _get_function_ast(func)

    if func_def is not None:
        body = _get_body_without_docstring(func_def.body)

        if len(body) == 0:
            return False

        # Check if last statement is ellipsis
        return _is_ellipsis_stmt(body[-1])

    # For dynamically generated functions without source, check bytecode
    # A function with just `...` compiles to just RESUME, LOAD_CONST(None), RETURN_VALUE
    # This is a heuristic - if the function is very short, it might be ellipsis
    try:
        code = func.__code__
        # Very short bytecode (<=3 instructions after RESUME) often indicates ellipsis
        if code.co_code and len(code.co_code) <= 12:  # ~3-4 instructions
            return True
    except Exception:
        pass

    # If we can't determine, assume it's not ellipsis (safer default)
    return False


def get_pre_ellipsis_code(func: Callable[..., Any]) -> str | None:
    """Extract code before the `...` marker in a function body.

    Returns the setup code that appears between docstring and ellipsis,
    formatted as executable Python code.

    Args:
        func: Function to extract code from

    Returns:
        Python code string, or None if no pre-ellipsis code exists
    """
    func_def = _get_function_ast(func)
    if func_def is None:
        return None

    body = _get_body_without_docstring(func_def.body)

    if len(body) == 0:
        return None

    # Check if last statement is ellipsis
    if not _is_ellipsis_stmt(body[-1]):
        return None

    # Get statements before ellipsis
    pre_ellipsis = body[:-1]

    if len(pre_ellipsis) == 0:
        return None

    # Convert AST back to source code
    code_lines = [ast.unparse(stmt) for stmt in pre_ellipsis]
    return "\n".join(code_lines)
