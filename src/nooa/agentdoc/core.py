# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core introspection functions for agentdoc."""

import inspect
from typing import Annotated, Any, Literal

from nooa.agentdoc._pformat import _pformat_to_str as _pformat
from nooa.agentdoc._pformat import render_document
from nooa.agentdoc._structured import format_type as _format_type_impl
from nooa.agentdoc.doc_config import DocConfig
from nooa.agentdoc.format import (
    format_value_summary,
    truncate_docstring,
)

DEFAULT_CONFIG = DocConfig()


def _is_documentable_collection(obj: list | tuple) -> bool:
    """Check if a list/tuple contains documentable objects.

    Returns True if all items are types, functions, methods, or modules.
    Returns False for data collections (lists of ints, strings, etc.).

    This determines whether doc([a, b, c]) should be flattened to doc(a, b, c)
    or treated as doc(data_list).
    """
    if not obj:
        return False

    for item in obj:
        # Documentable objects: types, functions, methods, modules
        if isinstance(item, type):
            continue
        if inspect.isfunction(item) or inspect.ismethod(item):
            continue
        if inspect.ismodule(item):
            continue
        # Not a documentable object - treat as data collection
        return False

    return True


def doc(
    *objs: Any,
    concise: Annotated[bool, "Show first-line docstrings only"] = False,
    inline_depth: Annotated[
        int,
        "Levels of referenced types to expand inline; 0 = none, 1 = direct (default), 2+ = transitive",
    ] = 1,
) -> str:
    """Render prompt-ready API documentation for one or more Python objects.

    Types show their declared API and defaults. Instances show that same contract
    enriched with current values and public runtime fields. Visibility rules are
    honored, properties stay unevaluated, and custom ``__repr__`` methods never
    hide the API.

    Referenced types are expanded inline and deduplicated across the result.

    Args:
        *objs: Objects to document: ``doc(MyClass)``, ``doc(Class, fn)``, or
            ``doc([A, B])``.
        concise: Show only the first line of each docstring.
        inline_depth: Referenced-type depth. ``0`` disables expansion, ``1``
            includes direct references (the default), and ``2+`` includes
            transitive references to that depth.

    Returns:
        The formatted API documentation.
    """
    # Flatten input: handle lists/tuples passed as single argument
    # Only flatten if the list/tuple contains documentable objects (types, callables, modules)
    flat_objs: list[Any] = []
    for obj in objs:
        if isinstance(obj, (list, tuple)) and _is_documentable_collection(obj):
            flat_objs.extend(obj)
        else:
            flat_objs.append(obj)

    if not flat_objs:
        raise ValueError("doc() requires at least one object")

    if not isinstance(inline_depth, int) or isinstance(inline_depth, bool):
        raise TypeError("inline_depth must be a non-negative integer")
    if inline_depth < 0:
        raise ValueError("inline_depth must be a non-negative integer")

    # One shared pipeline handles both single- and multi-object documents.
    return render_document(flat_objs, concise=concise, inline_depth=inline_depth)


def methods(
    obj: Annotated[Any, "Object to introspect"],
    detail: Annotated[Literal["summary", "full"], "Level of detail"] = "summary",
    config: Annotated[DocConfig | None, "Doc configuration; uses default if None"] = None,
) -> str:
    """List methods with signatures — lower-level alternative to ``doc()``.

    Use when you want method signatures only, without fields or docstrings.
    ``doc(obj)`` is usually the right choice; use ``methods()`` when you need
    the signature list as a building block.

    Args:
        obj: Object to introspect
        detail: ``"summary"`` (signature + first docstring line) or ``"full"``
        config: Optional configuration
    """
    cfg: DocConfig = DEFAULT_CONFIG if config is None else config

    lines = []

    from nooa.agentdoc._visibility import is_hidden_method

    for name in sorted(dir(obj)):
        if cfg.should_hide(name):
            continue

        try:
            attr = getattr(obj, name)
        except AttributeError:
            continue

        if not callable(attr):
            continue

        # Skip classes
        if inspect.isclass(attr):
            continue

        if is_hidden_method(attr):
            continue

        # For modules, skip imported functions (only show functions defined in this module)
        if inspect.ismodule(obj):
            attr_module = getattr(attr, "__module__", None)
            if attr_module and attr_module != obj.__name__:
                continue

        # Format method as proper Python definition
        try:
            sig = inspect.signature(attr)
            is_async = inspect.iscoroutinefunction(attr)

            # Build proper Python method signature with def/async def
            async_prefix = "async def " if is_async else "def "
            # Use clean signature formatter to strip module prefixes
            sig_str = f"{async_prefix}{name}{_format_signature_clean(sig)}:"

            if detail == "full":
                lines.append(sig_str)

                # Add docstring if available (in triple quotes)
                if cfg.include_docstrings:
                    doc_str = inspect.getdoc(attr)
                    if doc_str:
                        doc_lines = truncate_docstring(doc_str, max_lines=3).split("\n")
                        if len(doc_lines) == 1:
                            # Single line docstring
                            lines.append(f'    """{doc_lines[0]}"""')
                        else:
                            # Multi-line docstring
                            lines.append('    """')
                            for line in doc_lines:
                                lines.append(f"    {line}")
                            lines.append('    """')
            else:
                # Summary: signature + first line of docstring
                lines.append(sig_str)

                if cfg.include_docstrings:
                    doc_str = inspect.getdoc(attr)
                    if doc_str:
                        first_line = truncate_docstring(doc_str, max_lines=1)
                        lines.append(f'    """{first_line}"""')

        except (ValueError, TypeError):
            # Fallback for methods we can't introspect
            lines.append(f"def {name}(...):")

    return "\n".join(lines)


def variables(
    obj: Annotated[Any, "Object to introspect"],
    config: Annotated[DocConfig | None, "Doc configuration; uses default if None"] = None,
) -> str:
    """List non-callable attributes with current values — lower-level alternative to ``pformat()``.

    Use when you want a variable listing only, without type structure.
    ``pformat(instance)`` is usually the right choice; use ``variables()`` when
    you need the attribute list as a building block.

    Args:
        obj: Object to introspect
        config: Optional configuration
    """
    cfg: DocConfig = DEFAULT_CONFIG if config is None else config

    from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

    lines = []
    seen_names = set()
    obj_type = type(obj)

    # First, get instance attributes from __dict__
    if hasattr(obj, "__dict__"):
        for name, value in sorted(obj.__dict__.items()):
            if cfg.should_hide(name):
                continue
            if _is_hidden_field(obj_type, name):
                continue

            if callable(value):
                continue

            seen_names.add(name)

            # Format variable with type hint if available
            type_name = type(value).__name__
            value_str = format_value_summary(value, cfg)

            # Add type annotation if enabled
            line = (
                f"{name}: {type_name} = {value_str}"
                if cfg.include_types
                else f"{name} = {value_str}"
            )

            # Add drill-down hint if it's a complex object
            if cfg.include_hints and _should_show_hint(value):
                line += f"  # use doc(self.{name}) for details"

            lines.append(line)

    # Also check for class-level attributes (like tools and child agent classes)
    try:
        for name in dir(obj_type):
            if name in seen_names or cfg.should_hide(name):
                continue
            if _is_hidden_field(obj_type, name):
                continue

            try:
                attr = getattr(type(obj), name)
            except AttributeError:
                continue

            # Skip methods and properties (but include classes)
            if callable(attr) and not inspect.isclass(attr):
                continue
            if isinstance(attr, property):
                continue

            # Skip method descriptors
            if inspect.ismethoddescriptor(attr):
                continue

            # This is a class-level attribute (instance, class, or other)
            seen_names.add(name)

            # For classes, use doc(concise=True) to get a nice summary
            if inspect.isclass(attr):
                class_name = attr.__name__
                value_str = doc(attr, concise=True, inline_depth=0)
                line = f"{name}: type[{class_name}] = {value_str}"

                # Append hint within the existing comment if present
                if cfg.include_hints and _should_show_hint(attr):
                    hint_text = f"use doc(self.{name}) for details"
                    if "  #" in line:
                        # Append to existing comment with semicolon
                        line += f"; {hint_text}"
                    else:
                        # Start new comment
                        line += f"  # {hint_text}"
            else:
                # For instances/objects, use normal formatting
                type_name = type(attr).__name__
                value_str = format_value_summary(attr, cfg)

                line = (
                    f"{name}: {type_name} = {value_str}"
                    if cfg.include_types
                    else f"{name} = {value_str}"
                )

                # Add hint as separate comment for non-class objects
                if cfg.include_hints and _should_show_hint(attr):
                    line += f"  # use doc(self.{name}) for details"

            lines.append(line)
    except (TypeError, AttributeError):
        pass

    return "\n".join(lines)


def _format_signature_clean(sig: inspect.Signature) -> str:
    """Format a signature with clean type names (no module prefixes).

    Uses _format_type_for_doc() to format each parameter annotation,
    ensuring consistent clean formatting without module paths like
    'agents.router.ValidatorResult' -> 'ValidatorResult'.

    Args:
        sig: Signature object from inspect.signature()

    Returns:
        Formatted signature string like '(self, values: list[float]) -> ValidatorResult'
    """
    params = []

    for param_name, param in sig.parameters.items():
        # Build parameter string
        parts = [param_name]

        # Add annotation if present
        if param.annotation is not inspect.Parameter.empty:
            type_str = _format_type_impl(param.annotation)
            parts.append(f": {type_str}")

        # Add default if present (use _pformat for consistent Rich-style formatting)
        if param.default is not inspect.Parameter.empty:
            default_str = _pformat(param.default, max_string=50, max_length=5, max_depth=1)
            parts.append(f" = {default_str}")

        params.append("".join(parts))

    # Format return annotation
    return_str = ""
    if sig.return_annotation is not inspect.Signature.empty:
        return_type = _format_type_impl(sig.return_annotation)
        return_str = f" -> {return_type}"

    return f"({', '.join(params)}){return_str}"


def _should_show_hint(value: Any) -> bool:
    """Check if a value should have a drill-down hint.

    Args:
        value: Value to check

    Returns:
        True if value is complex enough to warrant a hint
    """
    # Show hints for objects with methods (not primitive types)
    if isinstance(value, (str, int, float, bool, type(None))):
        return False

    # Show hints for classes (like child agents, tool classes)
    if inspect.isclass(value):
        return True

    # Show hints for lists/dicts that might contain complex objects
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0

    # Show hints for other objects with methods
    try:
        for name in dir(value):
            if not name.startswith("_"):
                attr = getattr(value, name)
                if callable(attr):
                    return True
    except AttributeError:
        pass

    return False
