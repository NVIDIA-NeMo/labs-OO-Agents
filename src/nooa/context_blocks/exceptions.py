# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exceptions for context-blocks library."""


class BlockError(Exception):
    """Base exception for block-related errors."""

    pass


class BlockSyntaxError(BlockError):
    """Raised when a DynamicContext expression contains invalid Python syntax.

    This typically happens when LLM-generated code passes plain text
    instead of a valid Python expression.

    Example:
        # WRONG - plain text, not a Python expression
        self.context.set_dynamic("event", "A comet streaks across the sky")

        # CORRECT - use a string directly for static content
        self.context["event"] = "A comet streaks across the sky"

        # CORRECT - variable reference for dynamic content
        self.context.set_dynamic("event", "self.current_event")
    """

    def __init__(self, key: str, expr: str, original_error: Exception):
        self.key = key
        self.expr = expr
        self.original_error = original_error

        expr_preview = expr[:100] + "..." if len(expr) > 100 else expr

        message = (
            f"DynamicContext expression has invalid Python syntax.\n"
            f"  key: {key}\n"
            f"  expr: {expr_preview}\n"
            f"  Error: {original_error}\n\n"
            f"DynamicContext() requires a valid Python expression.\n"
            f"For static content, assign a string directly:\n"
            f'  self.context["{key}"] = "your content here"'
        )
        super().__init__(message)


class ProtectedBlockError(BlockError):
    """Raised when trying to modify or remove a protected block.

    Protected blocks are set by the framework and cannot be modified
    by generated code at runtime.
    """

    def __init__(self, key: str, operation: str = "modify"):
        self.key = key
        self.operation = operation
        message = f'Cannot {operation} protected block: "{key}"'
        super().__init__(message)


class DynamicNotResolvedError(BlockError):
    """Raised when accessing a DynamicContext block before it has been resolved.

    DynamicContext blocks are evaluated during _prepare_context(), which runs
    at the start of each LLM generation call. Accessing a DynamicContext block
    before the first LLM turn means its expression hasn't been evaluated yet.

    Solutions:
    - Use a static block instead: self.context["key"] = "value"
    - Wait until after the first LLM call to access the value
    - Call the expression directly if you need the value immediately
    """

    def __init__(self, key: str, expr: str):
        self.key = key
        self.expr = expr
        message = (
            f'DynamicContext block "{key}" has not been resolved yet.\n'
            f"  Expression: {expr}\n"
            f"  DynamicContext blocks are evaluated at the start of each LLM turn.\n"
            f"  Before the first turn, the value is not available.\n\n"
            f"Solutions:\n"
            f'  - Use a static block: self.context["{key}"] = "value"\n'
            f"  - Access the value after the first LLM call\n"
            f"  - Evaluate the expression directly: {expr}"
        )
        super().__init__(message)
