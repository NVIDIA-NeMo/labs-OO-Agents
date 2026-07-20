# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for the evaluation pipeline."""

from __future__ import annotations

from typing import Any


def classify_error_type(error_msg: str | None) -> str | None:
    """Classify error type from error message string.

    Since we only have the error message (not the original exception),
    we classify based on common patterns.
    """
    if not error_msg:
        return None

    error_lower = error_msg.lower()

    if "memory soft limit" in error_lower:
        return "MemoryWarning"
    if "memoryerror" in error_lower or ("memory" in error_lower and "limit" in error_lower):
        return "MemoryError"
    if "timeout" in error_lower:
        return "TimeoutError"
    if "connection" in error_lower or "network" in error_lower:
        return "ConnectionError"
    if "permission" in error_lower or "access denied" in error_lower:
        return "PermissionError"
    if "not found" in error_lower or "does not exist" in error_lower:
        return "NotFoundError"
    if "validation" in error_lower or "invalid" in error_lower:
        return "ValidationError"
    if "syntax" in error_lower or "parse" in error_lower:
        return "SyntaxError"
    if "import" in error_lower or "module" in error_lower:
        return "ImportError"
    if "type" in error_lower and "error" in error_lower:
        return "TypeError"
    if "attribute" in error_lower:
        return "AttributeError"
    if "key" in error_lower and "error" in error_lower:
        return "KeyError"
    if "index" in error_lower and ("error" in error_lower or "out of" in error_lower):
        return "IndexError"

    return "ExecutionError"


def merge_eval_metadata(
    config_test_meta: dict[str, str | int | float | bool] | None,
    task_meta: dict[str, Any] | None,
) -> dict[str, str | int | float | bool]:
    """Merge eval metadata from config+test level with task-level metadata.

    Merge order: config+test < task (task-level overrides).
    Only scalar values (str, int, float, bool) from task metadata are included.
    """
    merged: dict[str, str | int | float | bool] = {}
    if config_test_meta:
        merged |= config_test_meta
    if task_meta:
        for k, v in task_meta.items():
            if isinstance(v, (str, int, float, bool)):
                merged[k] = v
    return merged


def sanitize_for_json(value: Any) -> Any:
    """Sanitize a value for JSON serialization.

    Converts non-serializable types (like coroutines) to string representations.
    This prevents serialization errors when the agent returns un-awaited coroutines.
    """
    import inspect

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if inspect.iscoroutine(value):
        value.close()
        return (
            f"<unawaited coroutine: {value.__name__ if hasattr(value, '__name__') else 'unknown'}>"
        )
    if inspect.isgenerator(value) or inspect.isasyncgen(value):
        return f"<generator: {type(value).__name__}>"

    try:
        import json

        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)
