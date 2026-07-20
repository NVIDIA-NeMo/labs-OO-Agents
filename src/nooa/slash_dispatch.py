# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed dispatch for @slash_command methods."""

import inspect
import shlex
import types
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


@dataclass
class SlashCommandResult:
    """Result from a slash command, passed to the agent via queue.

    Attributes:
        command: The command name (e.g. "config").
        args: Raw argument string as typed by the user.
        value: The return value from the slash command method (any type).
        text: String representation for the agent prompt (auto-derived if not set).
    """

    command: str
    args: str
    value: Any = None
    text: str | None = None
    output_to_agent: bool = True

    def __str__(self) -> str:
        if self.text is not None:
            return self.text
        if self.value is None:
            return f"/{self.command} {self.args}".strip()
        if isinstance(self.value, str):
            return self.value
        return repr(self.value)


class CoercionError(Exception):
    """Raised when argument parsing/coercion fails."""

    def __init__(self, message: str, hint: str = ""):
        self.message = message
        self.hint = hint
        super().__init__(message)


def parse_typed_args(method: Any, raw_args: str) -> dict[str, Any]:
    """Parse raw argument string into typed keyword arguments based on method signature.

    If the method signature is just ``(self, args: str)`` or ``(self, args: str = ...)``,
    returns ``{"args": raw_args}`` (backward-compatible passthrough).

    Otherwise, shlex-splits the raw string and binds positionally to the method's
    parameters, coercing types based on annotations.

    Raises:
        CoercionError: If parsing fails (wrong number of args, bad type, etc.).
    """
    sig = inspect.signature(method)
    params = list(sig.parameters.values())

    # Skip 'self' if present (bound methods won't have it, but just in case)
    if params and params[0].name == "self":
        params = params[1:]

    if not params:
        return {}

    # Backward compat: single `args: str` parameter
    if len(params) == 1 and params[0].name == "args":
        ann = params[0].annotation
        if ann is inspect.Parameter.empty or ann is str:
            return {"args": raw_args}

    # Parse with shlex
    try:
        tokens = shlex.split(raw_args) if raw_args.strip() else []
    except ValueError as e:
        raise CoercionError(f"Failed to parse arguments: {e}") from None

    # Bind positionally
    hints = get_type_hints(method)
    result: dict[str, Any] = {}
    required_count = sum(1 for p in params if p.default is inspect.Parameter.empty)

    if len(tokens) < required_count:
        param_names = [p.name for p in params[:required_count]]
        raise CoercionError(
            f"Expected {required_count} argument(s): {', '.join(param_names)}, got {len(tokens)}",
            hint=_build_usage_hint(method, params),
        )

    for i, param in enumerate(params):
        if i < len(tokens):
            ann = hints.get(param.name, str)
            try:
                result[param.name] = _coerce(tokens[i], ann)
            except (ValueError, TypeError):
                allowed = _describe_type(ann)
                raise CoercionError(
                    f"Argument '{param.name}': cannot convert '{tokens[i]}' to {allowed}",
                    hint=_build_usage_hint(method, params),
                ) from None
        elif param.default is not inspect.Parameter.empty:
            result[param.name] = param.default
        # else: already checked above with required_count

    # If there are extra tokens beyond the declared params, join them into the last param
    if len(tokens) > len(params) and params:
        last = params[-1]
        ann = hints.get(last.name, str)
        if ann is str or ann is inspect.Parameter.empty:
            # Re-join everything from that position onward
            result[last.name] = " ".join(tokens[len(params) - 1 :])

    return result


def derive_completions(method: Any) -> tuple[str, ...]:
    """Derive tab-completion values from Literal type hints on the first parameter."""
    sig = inspect.signature(method)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    if not params:
        return ()

    hints = get_type_hints(method)
    first_param = params[0]
    ann = hints.get(first_param.name)
    if ann is None:
        return ()

    # Handle Literal["a", "b", "c"]
    if get_origin(ann) is Literal:
        return tuple(str(a) for a in get_args(ann))

    # Handle Optional[Literal[...]] = Union[Literal[...], None]
    if get_origin(ann) in (Union, types.UnionType):
        for arg in get_args(ann):
            if get_origin(arg) is Literal:
                return tuple(str(a) for a in get_args(arg))

    return ()


def _coerce(value: str, annotation: Any) -> Any:
    """Coerce a string value to the annotated type."""
    if annotation is inspect.Parameter.empty or annotation is str:
        return value

    # Literal — validate against allowed values
    if get_origin(annotation) is Literal:
        allowed = get_args(annotation)
        str_allowed = [str(a) for a in allowed]
        if value in str_allowed:
            idx = str_allowed.index(value)
            return allowed[idx]
        raise ValueError(f"must be one of: {', '.join(str_allowed)}")

    # Optional[X] = Union[X, None], or PEP 604 `X | Y`
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            last_exc: Exception | None = None
            for arg in args:
                try:
                    return _coerce(value, arg)
                except (ValueError, TypeError) as e:
                    last_exc = e
            if last_exc is not None:
                raise last_exc

    # Basic types
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is bool:
        return value.lower() in ("true", "1", "yes", "on")
    if annotation is str:
        return value

    # Fallback: try calling the type
    return annotation(value)


def _describe_type(annotation: Any) -> str:
    """Human-readable description of a type annotation."""
    if annotation is inspect.Parameter.empty or annotation is str:
        return "str"
    if get_origin(annotation) is Literal:
        allowed = get_args(annotation)
        return f"one of ({', '.join(repr(a) for a in allowed)})"
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return " | ".join(_describe_type(a) for a in args)
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _build_usage_hint(method: Any, params: list) -> str:
    """Build a usage hint string from method parameters."""
    parts = []
    hints = get_type_hints(method)
    for p in params:
        ann = hints.get(p.name)
        if ann and get_origin(ann) is Literal:
            allowed = "|".join(str(a) for a in get_args(ann))
            part = f"<{allowed}>"
        elif p.default is not inspect.Parameter.empty:
            part = f"[{p.name}]"
        else:
            part = f"<{p.name}>"
        parts.append(part)
    return " ".join(parts)
