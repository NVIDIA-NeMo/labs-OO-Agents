# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error formatting helpers for CodeAct strategy.

Pure functions that format validation errors into actionable LLM feedback.
Extracted from CodeActStrategy to keep the strategy class focused on control flow.
"""

import inspect
import json
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ErrorDetails

from nooa.agentdoc import pformat
from nooa.agentdoc.visibility import is_hidden_field
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG, TruncationConfig
from nooa.strategy_validation import InvariantError


def format_validation_error(
    error: Exception,
    return_type: Any,
    actual_value: Any = None,
    truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
) -> str:
    """Format validation error as actionable return_result() feedback for the LLM."""
    if isinstance(error, InvariantError):
        return (
            f"Method invariant failed: {error}\n"
            "Revise the answer so it satisfies this invariant, then call return_result() again."
        )

    if isinstance(error, json.JSONDecodeError):
        return (
            f"return_result() failed: Could not parse JSON - {error.msg}\n"
            "Make sure your arguments are valid JSON values."
        )

    if isinstance(error, PydanticValidationError):
        return _format_pydantic_error(error, return_type, actual_value, truncation_config)

    return f"return_result() failed: {error}"


def _format_pydantic_error(
    error: PydanticValidationError,
    return_type: Any,
    actual_value: Any = None,
    truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
) -> str:
    """Format Pydantic validation errors as actionable return_result() feedback."""
    errors = error.errors()
    type_hint = get_type_hint_str(return_type)

    # Check if errors are primarily about missing fields
    missing_field_errors = [e for e in errors if e.get("type") == "missing"]

    if missing_field_errors:
        # Format with expected schema for missing field errors
        return _format_missing_fields_error(
            missing_field_errors, errors, return_type, actual_value, truncation_config
        )

    if len(errors) == 1:
        # Single error - format concisely
        return _format_single_error(
            errors[0],
            return_type,
            type_hint,
            actual_value=actual_value,
            truncation_config=truncation_config,
        )

    # Multiple errors - list them
    lines = [f"return_result() failed with {len(errors)} errors:"]
    for err in errors:
        lines.append(
            f"  • {_format_single_error(err, return_type, type_hint, brief=True, actual_value=actual_value, truncation_config=truncation_config)}"
        )

    lines.append(
        f"\nExpected return type: {type_hint} - Got: {_format_value_for_error(actual_value, truncation_config)}"
    )
    return "\n".join(lines)


def _format_missing_fields_error(
    missing_errors: list[ErrorDetails],
    all_errors: list[ErrorDetails],
    return_type: Any,
    actual_value: Any,
    truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
) -> str:
    """Format error message for missing required fields with expected schema."""
    # Extract the missing field names
    # Note: We wrap the value in a model with "result" field, so error locations
    # start with "result" - we skip that to show the actual field names
    missing_fields = []
    for err in missing_errors:
        loc = err.get("loc", ())
        if loc:
            loc_list = list(loc)
            # Skip the "result" wrapper field if present
            if loc_list[0] == "result" and len(loc_list) > 1:
                # Use the second element as the actual field name
                field = (
                    loc_list[1]
                    if isinstance(loc_list[1], str)
                    else _format_error_path(tuple(loc_list[1:]))
                )
            else:
                field = loc_list[0] if isinstance(loc_list[0], str) else _format_error_path(loc)
            missing_fields.append(field)

    # Deduplicate fields (in case same field appears multiple times)
    missing_fields = list(dict.fromkeys(missing_fields))

    # Build the error message
    lines = ["return_result() failed - missing required fields:"]

    # Show expected schema
    expected_schema = _format_expected_schema(return_type)
    lines.append(f"Expected: {expected_schema}")

    # Show what was actually received if available
    if actual_value is not None:
        got_repr = _format_actual_value(actual_value, truncation_config)
        lines.append(f"Got: {got_repr}")

    # List the specific missing fields
    if missing_fields:
        lines.append(f"Missing: {', '.join(missing_fields)}")

    return "\n".join(lines)


def _format_expected_schema(return_type: Any) -> str:
    """Format a return type as a compact schema showing structure.

    For TypedDicts: RouterResult { agents_called: list[str], results: dict }
    For Pydantic: UserModel { name: str, age: int }
    For simple types: just the type name
    """
    name = getattr(return_type, "__name__", None)
    if name is None:
        name = get_type_hint_str(return_type)

    # Check if it's a TypedDict (has __annotations__ and is dict subclass)
    if hasattr(return_type, "__annotations__") and isinstance(return_type, type):
        annotations = inspect.get_annotations(return_type)
        if annotations:
            items = [(k, v) for k, v in annotations.items() if not is_hidden_field(return_type, k)]
            fields = ", ".join(f"{k}: {get_type_hint_str(v)}" for k, v in items)
            return f"{name} {{ {fields} }}"

    # Check if it's a Pydantic model
    if hasattr(return_type, "model_fields"):
        items = [
            (k, v)
            for k, v in return_type.model_fields.items()
            if not is_hidden_field(return_type, k)
        ]
        fields = ", ".join(f"{k}: {get_type_hint_str(v.annotation)}" for k, v in items)
        return f"{name} {{ {fields} }}"

    return name


def _pformat(value: Any, tc: TruncationConfig) -> str:
    """Format a value using the agent's value-render truncation settings."""
    return pformat(value, **tc.event_format.model_dump())


def _format_value_for_error(
    value: Any, truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG
) -> str:
    """Format value for (value: ...) in error messages."""
    return _pformat(value, truncation_config)


def _format_actual_value(
    value: Any, truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG
) -> str:
    """Format the actual value received for error display."""
    if isinstance(value, dict):
        # Show dict keys and values compactly
        if len(value) <= 5:
            items = ", ".join(f"'{k}': {_format_value_brief(v)}" for k, v in value.items())
            return f"{{ {items} }}"
        else:
            # Too many items, show keys only
            keys = ", ".join(f"'{k}'" for k in list(value.keys())[:5])
            return f"{{ {keys}, ... ({len(value)} fields) }}"
    elif isinstance(value, (list, tuple)):
        type_name = "list" if isinstance(value, list) else "tuple"
        if len(value) <= 3:
            items = ", ".join(_format_value_brief(v) for v in value)
            return f"[{items}]" if isinstance(value, list) else f"({items})"
        return f"{type_name} with {len(value)} items"
    else:
        return _pformat(value, truncation_config)


def _format_value_brief(value: Any) -> str:
    """Format a single value briefly for display."""
    if isinstance(value, str):
        if len(value) > 20:
            return f"'{value[:17]}...'"
        return repr(value)
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, dict):
        return "{...}"
    elif isinstance(value, (list, tuple)):
        return "[...]" if isinstance(value, list) else "(...)"
    else:
        # Brief display: just show the type — pformat for non-trivial objects would
        # always produce multiline output in a brief inline context.
        return f"{type(value).__name__}(...)"


def _format_got_line(
    actual_value: Any,
    brief: bool = False,
    truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
) -> str:
    """Format 'Got: <type> (value: ...)' for result-level type errors."""
    actual_type = type(actual_value).__name__
    truncated = _format_value_for_error(actual_value, truncation_config)
    return f"Got: {actual_type} (value: {truncated})" if not brief else f"Got: {actual_type}"


def _format_single_error(
    err: ErrorDetails,
    return_type: Any,
    type_hint: str,
    brief: bool = False,
    actual_value: Any = None,
    truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
) -> str:
    """Format a single Pydantic error into an actionable message."""
    loc = err["loc"]
    msg = err["msg"]
    error_type = err["type"]

    # Got value line: show the actual value if available
    is_result_level = not loc or (len(loc) == 1 and loc[0] == "result")
    got_line = (
        _format_got_line(actual_value, brief, truncation_config)
        if (actual_value is not None and is_result_level)
        else ""
    )

    # Get example of correct return_result() usage for a type
    example = get_type_example(return_type)

    # Root-level error - the entire value is the wrong type
    if not loc:
        if brief:
            return f"Expected {type_hint}, got wrong type"

        result = f"Expected: {type_hint}\n"
        if got_line:
            result += got_line + "\n"
        if example:
            result += f"Example: return_result({example})"
        return result

    # Format the path for display
    path = _format_error_path(loc)

    # Check if this is a root-level list/dict element error (loc starts with int)
    # vs a named field error (loc starts with string)
    first_part = loc[0]
    is_root_collection_error = isinstance(first_part, int)
    is_nested = len(loc) > 1  # More than one level deep

    if is_root_collection_error:
        # Error in an element of a root-level list/tuple
        # e.g., list[int] where one element is wrong
        if error_type == "missing":
            if brief:
                return f"{path}: missing required field"
            return f"return_result(result=[...]) - '{path}' is missing a required field."

        if "type" in error_type or error_type in (
            "int_parsing",
            "float_parsing",
            "bool_parsing",
        ):
            if brief:
                return f"{path}: wrong type - {msg.lower()}"
            line = f"return_result(result=[...]) - '{path}' has wrong type.\nExpected: {type_hint}"
            return line

        if brief:
            return f"{path}: {msg.lower()}"
        return f"return_result(result=[...]) - '{path}': {msg}"

    # Named field error
    field = first_part

    if error_type == "missing":
        if is_nested:
            # Nested missing field, e.g., items[1].price
            if brief:
                return f"'{path}': field required"
            return f"return_result({field}=...) - '{path}' is missing (field required)."
        else:
            # Top-level missing parameter
            if brief:
                return f"Missing required parameter '{field}'"
            return f"return_result() is missing required parameter '{field}'."

    if "type" in error_type or error_type in (
        "int_parsing",
        "float_parsing",
        "bool_parsing",
        "int_from_float",
    ):
        if brief:
            return f"'{path}' has wrong type - {msg.lower()}"
        line = f"return_result({field}=...) - '{path}' has wrong type.\nExpected: {type_hint}"
        if got_line:
            line += "\n" + got_line
        if example:
            line += f"\nExample: return_result({example})"
        return line

    # Generic field error
    if brief:
        return f"'{path}': {msg.lower()}"
    return f"return_result({field}=...) - '{path}': {msg}"


def _format_error_path(loc: tuple[Any, ...]) -> str:
    """Format error location as readable path like 'items[0].name' or '[2]'."""
    if not loc:
        return "value"

    result = ""
    for part in loc:
        if isinstance(part, int):
            # Array index - use bracket notation
            result += f"[{part}]"
        else:
            # Field name - use dot notation after previous parts
            if result:
                result += "."
            result += str(part)

    return result


def get_type_hint_str(return_type: Any) -> str:
    """Get a human-readable string for a type hint."""
    if return_type is None:
        return "value"

    # Handle typing generics like list[int], dict[str, Any]
    origin = getattr(return_type, "__origin__", None)
    if origin is not None:
        args: tuple[Any, ...] = getattr(return_type, "__args__", ())
        if origin is list:
            if args:
                return f"list[{get_type_hint_str(args[0])}]"
            return "list"
        elif origin is dict:
            if len(args) >= 2:
                return f"dict[{get_type_hint_str(args[0])}, {get_type_hint_str(args[1])}]"
            return "dict"
        elif origin is tuple:
            if args:
                inner = ", ".join(get_type_hint_str(a) for a in args)
                return f"tuple[{inner}]"
            return "tuple"

    # Handle basic types and classes
    if isinstance(return_type, type):
        return return_type.__name__

    # Fallback to string representation
    return str(return_type)


def get_type_example(return_type: Any) -> str | None:
    """Get an example of correct return_result() usage for a type."""
    origin = getattr(return_type, "__origin__", None)

    if origin is list:
        args = getattr(return_type, "__args__", ())
        if args:
            inner = get_type_hint_str(args[0])
            if inner == "str":
                return 'result=["item1", "item2", "item3"]'
            elif inner == "int":
                return "result=[1, 2, 3]"
            elif inner == "float":
                return "result=[1.0, 2.5, 3.14]"
        return "result=[...]"

    if origin is dict:
        return 'result={"key": "value"}'

    if return_type is str:
        return 'result="your answer"'

    if return_type is int:
        return "result=42"

    if return_type is float:
        return "result=3.14"

    if return_type is bool:
        return "result=True"

    return None
