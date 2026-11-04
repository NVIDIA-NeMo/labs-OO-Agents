# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exceptions for context-blocks library."""


class BlockError(Exception):
    """Base exception for block-related errors."""

    pass


class ContextAssemblyError(BlockError):
    """Raised when a request's context cannot be assembled or rendered.

    Assembly is deterministic local code, so strategies propagate this error
    instead of retrying it as a model or provider failure.
    """


class UnsupportedContextLayout(ContextAssemblyError):
    """Raised when a provider cannot preserve an assembled context layout."""


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
