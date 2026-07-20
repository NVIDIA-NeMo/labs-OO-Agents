# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pprint formatting for nooa and agentdoc.

This module defines the canonical ``_pformat`` implementation used by:

- ``nooa/agentdoc/format.py`` for value summaries
- ``nooa/agentdoc/__init__.py`` for the public ``pformat``/``pprint`` API

The unified pformat() function handles:
- Regular values (list, dict, str, etc.) with truncation
- Types (classes) with Python class syntax
- Functions/methods with signature and docstring
- Modules with docstring and public functions
- Info objects (TypeInfo, CallableInfo, ModuleInfo) directly
"""

import functools
import importlib
import inspect
import io
import re
from typing import Any

from nooa.agentdoc._info import REQUIRED, CallableInfo, ModuleInfo, TypeInfo
from nooa.agentdoc._metadata import is_expand_false
from nooa.agentdoc._structured import _ClassRef, _InstanceRef
from nooa.agentdoc.protocols import SupportsInstanceValues


def _truncate_docstring_lines(docstring: str | None, max_lines: int = 1) -> str:
    """Truncate a docstring to a specified number of lines."""
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    if len(lines) <= max_lines:
        return docstring.strip()
    return "\n".join(lines[:max_lines]).strip()


def _resolve_field_type_class(field_default: Any, context_obj: type | None) -> type | None:
    """Resolve a field's default marker to the actual type class, or None."""
    if field_default is REQUIRED or field_default is ...:
        return None

    if isinstance(field_default, (_InstanceRef, _ClassRef)):
        class_name = getattr(field_default, "class_name", None) or getattr(
            field_default, "name", None
        )
        if class_name and context_obj is not None:
            _mod_name = getattr(context_obj, "__module__", None)
            try:
                _mod_ns = vars(importlib.import_module(_mod_name)) if _mod_name else None
            except ImportError:
                _mod_ns = None
            for ns in (vars(context_obj), _mod_ns):
                if ns and class_name in ns and isinstance(ns[class_name], type):
                    return ns[class_name]
        return None
    elif isinstance(field_default, type) and field_default.__module__ != "builtins":
        return field_default
    elif not isinstance(field_default, (str, int, float, bool, bytes, type(None))):
        cls = type(field_default)
        return None if cls.__module__ == "builtins" else cls
    return None


def _resolve_field_type_by_name(type_name: str, context_obj: type | None) -> type | None:
    """Resolve a type name string to a class via context_obj's module namespace."""
    if not type_name or context_obj is None:
        return None
    _mod_name = getattr(context_obj, "__module__", None)
    try:
        _mod_ns = vars(importlib.import_module(_mod_name)) if _mod_name else {}
    except (ImportError, Exception):
        _mod_ns = {}
    for ns in (vars(context_obj), _mod_ns):
        candidate = ns.get(type_name)
        if isinstance(candidate, type):
            return candidate
    return None


def _collect_referenced_types_transitive(
    seed_types: set[type],
    exclude: type | None = None,
) -> list[type]:
    """Collect all referenced types transitively via BFS, deduplicated and sorted.

    Args:
        seed_types: Initial set of types to start from (already filtered).
        exclude: Primary type to exclude from results (avoid self-reference).

    Returns:
        Sorted list of all reachable custom types (no duplicates).
    """
    from nooa.agentdoc._discover import discover_referenced_types

    all_types: set[type] = set()
    # Exclude the primary type from the seed too: a type that references itself in its
    # own method signatures (common, e.g. DataFrame methods returning DataFrame) must
    # not list itself under "Referenced Types".
    frontier = {t for t in seed_types if t is not exclude}
    while frontier:
        all_types.update(frontier)
        next_frontier: set[type] = set()
        for t in frontier:
            for new_t in discover_referenced_types(t):
                if new_t not in all_types and not is_expand_false(new_t) and new_t is not exclude:
                    next_frontier.add(new_t)
        frontier = next_frontier
    return sorted(all_types, key=lambda t: t.__name__)


def _field_type_docstring(
    field_default: Any, context_obj: type | None, type_name: str | None = None
) -> str | None:
    """Return the first line of the docstring of a field's type."""
    cls = _resolve_field_type_class(field_default, context_obj)
    if cls is None and type_name:
        cls = _resolve_field_type_by_name(type_name, context_obj)
    if cls is None:
        return None
    # Use cls.__doc__ directly — inspect.getdoc() walks MRO and can inherit
    # irrelevant parent docstrings (e.g. BaseModel's MkDocs admonition markup).
    raw_doc = cls.__doc__
    if not raw_doc:
        return None
    docstring = inspect.cleandoc(raw_doc)
    return docstring.split("\n")[0].strip()


def _pformat(
    _object: Any,
    _stream: Any,
    *,
    max_length: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    concise: bool = False,
    inline_depth: int | None = None,
    expand_all: bool = False,
    instance_mode: str = "repr",
    _depth: int = 0,
    _indent: int = 0,
) -> None:
    """Write pretty-formatted representation of *_object* to *_stream*.

    Args:
        _object: Object to format.
        _stream: Writer with a ``write(str)`` method. May be a
            ``TruncatingStringIO`` (hard char cap) or plain ``io.StringIO``.
        max_length: Max elements per container (None=unlimited).
        max_string: Max string chars (None=unlimited).
        max_depth: Max nesting depth (None=unlimited).
        concise: If True, show first-line docstrings only.
        inline_depth: How deep to expand referenced types inline.
        expand_all: Always expand containers.
        instance_mode: How to format instances — "repr" or "type".
        _depth: Current nesting depth (internal).
        _indent: Current indentation level (internal).
    """
    # Resolve inline_depth default
    if inline_depth is None:
        inline_depth = 0 if concise else 1

    # Use match statement for clean dispatch on Info types and Python objects
    match _object:
        case TypeInfo():
            _stream.write(
                _format_type_info(
                    _object,
                    concise=concise,
                    type_depth=inline_depth,
                    max_length=max_length,
                    indent=_indent,
                    context_obj=None,
                )
            )
        case CallableInfo():
            _stream.write(
                _format_callable_info(
                    _object,
                    concise=concise,
                    type_depth=inline_depth,
                    indent=_indent,
                )
            )
        case ModuleInfo():
            _stream.write(_format_module_info(_object, concise=concise, indent=_indent))
        case type():
            # Python type (class) - extract and format
            from nooa.agentdoc._structured import extract_type_info

            type_info = extract_type_info(_object)
            _stream.write(
                _format_type_info(
                    type_info,
                    concise=concise,
                    type_depth=inline_depth,
                    max_length=max_length,
                    indent=_indent,
                    context_obj=_object,  # Pass the original type for discovery
                )
            )
        case _ if inspect.ismodule(_object):
            from nooa.agentdoc.registry import get_module_info_extractor

            extractor = get_module_info_extractor(_object)
            if extractor is not None:
                module_info = extractor(_object)
            else:
                from nooa.agentdoc._structured import extract_module_info

                module_info = extract_module_info(_object)
            _stream.write(_format_module_info(module_info, concise=concise, indent=_indent))
        case _ if inspect.isfunction(_object) or inspect.ismethod(_object):
            from nooa.agentdoc._structured import extract_callable_info

            callable_info = extract_callable_info(_object)
            _stream.write(
                _format_callable_info(
                    callable_info,
                    concise=concise,
                    type_depth=inline_depth,
                    indent=_indent,
                    context_obj=_object,  # Pass the function/method for type discovery
                )
            )
        case _ if _is_structured_instance(_object):
            # Instance of a structured type
            if instance_mode == "type":
                # Show type structure with runtime values (for doc())
                from nooa.agentdoc._structured import extract_type_info
                from nooa.agentdoc._visibility import is_hidden_field
                from nooa.agentdoc.protocols import SupportsInstanceValues
                from nooa.agentdoc.registry import get_type_info_extractor

                obj_type = type(_object)
                # Check if instance has spec() overrides that could unhide class-hidden fields.
                # spec(self, "field", hidden=False) in __init__ → per-instance opt-in.
                _instance_fields_meta = vars(_object).get("_agentdoc_fields_docs") or {}
                _has_instance_overrides = any(
                    meta.get("hidden") is False for meta in _instance_fields_meta.values()
                )
                extractor = get_type_info_extractor(obj_type)
                if extractor:
                    result = extractor(_object)
                    if isinstance(result, tuple):
                        type_info, values = result
                    else:
                        type_info = result
                        values = _extract_instance_values(_object, result)
                    # Filter with instance (supports instance-level spec() overrides)
                    type_info = TypeInfo(
                        name=type_info.name,
                        base=type_info.base,
                        fields=[
                            f for f in type_info.fields if not is_hidden_field(_object, f.name)
                        ],
                        methods=type_info.methods,
                        docstring=type_info.docstring,
                    )
                elif isinstance(_object, SupportsInstanceValues):
                    if _has_instance_overrides:
                        # Get all fields (including class-hidden) then re-filter by instance.
                        # Use _skip_protocol to get raw fields; take methods from protocol path.
                        raw_info = extract_type_info(
                            obj_type, _skip_protocol=True, _include_hidden=True
                        )
                        protocol_info = extract_type_info(obj_type)
                        type_info = TypeInfo(
                            name=protocol_info.name,
                            base=protocol_info.base,
                            fields=[
                                f for f in raw_info.fields if not is_hidden_field(_object, f.name)
                            ],
                            methods=protocol_info.methods,
                            docstring=protocol_info.docstring,
                        )
                    else:
                        type_info = extract_type_info(obj_type)
                    values = _object.__instance_values__()
                else:
                    if _has_instance_overrides:
                        raw_info = extract_type_info(
                            obj_type, _skip_protocol=True, _include_hidden=True
                        )
                        protocol_info = extract_type_info(obj_type)
                        type_info = TypeInfo(
                            name=protocol_info.name,
                            base=protocol_info.base,
                            fields=[
                                f for f in raw_info.fields if not is_hidden_field(_object, f.name)
                            ],
                            methods=protocol_info.methods,
                            docstring=protocol_info.docstring,
                        )
                    else:
                        type_info = extract_type_info(obj_type)
                    values = _extract_instance_values(_object, type_info)

                _stream.write(
                    _format_type_info(
                        type_info,
                        concise=concise,
                        type_depth=inline_depth,
                        max_length=max_length,
                        indent=_indent,
                        context_obj=obj_type,
                        instance_values=values,
                    )
                )
            else:
                # Show repr-style (for pprint())
                _stream.write(
                    _format_instance_repr(
                        _object,
                        max_length=max_length,
                        max_string=max_string,
                        max_depth=max_depth,
                        indent=_indent,
                    )
                )
        case _:
            # Regular value - use truncation formatting
            _format_value(
                _object,
                _stream,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
                depth=_depth,
                indent=_indent,
            )


def _pformat_to_str(
    _object: Any,
    *,
    max_length: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    concise: bool = False,
    inline_depth: int | None = None,
    expand_all: bool = False,
    instance_mode: str = "repr",
    _depth: int = 0,
    _indent: int = 0,
) -> str:
    """Convenience wrapper: call ``_pformat`` into an ``io.StringIO`` and return the string."""
    stream = io.StringIO()
    _pformat(
        _object,
        stream,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        concise=concise,
        inline_depth=inline_depth,
        expand_all=expand_all,
        instance_mode=instance_mode,
        _depth=_depth,
        _indent=_indent,
    )
    return stream.getvalue()


def _is_structured_instance(obj: Any) -> bool:
    """Check if object is an instance that should be formatted with type info.

    Returns True for:
    - Pydantic models
    - dataclasses
    - NamedTuples
    - attrs classes
    - Any custom class instance with __dict__ (not built-in types)

    Returns False for:
    - Types (classes themselves)
    - Built-in types (str, int, list, dict, etc.)
    - None
    """
    if isinstance(obj, type):
        return False

    # Skip built-in types and None
    if obj is None:
        return False

    if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
        return False

    from nooa.agentdoc._structured import _ClassRef, _InstanceRef

    if isinstance(obj, (_ClassRef, _InstanceRef)):
        return False

    obj_type = type(obj)

    # Check for NamedTuple BEFORE checking for regular tuples
    # NamedTuples have a _fields attribute
    if (
        hasattr(obj_type, "_fields")
        and isinstance(getattr(obj_type, "_fields", None), tuple)
        and isinstance(obj, tuple)
    ):
        return True

    # Now safe to exclude regular tuples, lists, sets, dicts
    if isinstance(obj, (list, tuple, set, frozenset, dict)):
        return False

    # Pydantic (check before builtins guard — classes defined in exec()/REPL
    # get __module__='builtins' but are still structured types)
    if hasattr(obj_type, "model_fields"):
        return True

    # dataclass
    import dataclasses

    if dataclasses.is_dataclass(obj_type):
        return True

    # attrs
    if hasattr(obj_type, "__attrs_attrs__"):
        return True

    # Skip if it's a built-in type (range, slice, memoryview, etc.)
    if obj_type.__module__ == "builtins":
        return False

    # If the class defines its own __repr__ (not inherited from object),
    # trust the library author's representation over field extraction.
    for klass in obj_type.__mro__:
        if klass is object:
            break
        if "__repr__" in klass.__dict__:
            return False

    # __slots__-only classes have no __dict__ but do have public slots
    if not hasattr(obj, "__dict__"):
        return any(
            slot
            for klass in obj_type.__mro__
            if klass is not object
            for slot in getattr(klass, "__slots__", ())
            if not slot.startswith("_")
        )

    # Any other custom class instance with __dict__
    return True


def _format_type_info(
    info: TypeInfo,
    *,
    concise: bool,
    type_depth: int = 0,
    max_length: int | None,
    indent: int,
    context_obj: type | None = None,
    instance_values: dict[str, Any] | None = None,
) -> str:
    """Format TypeInfo as Python class syntax.

    Shows:
    - Class name and base (if any)
    - Field names, types, defaults, descriptions
    - Extra instance attributes (if instance_values provided)
    - Method signatures and docstrings
    - Referenced Types section (only if context_obj provided and type_depth > 0)

    Args:
        info: TypeInfo to format
        concise: If True, show first-line docstrings only
        type_depth: How deep to recurse into referenced types (0 = none)
        max_length: Max fields/methods to show before truncation
        indent: Indentation level
        context_obj: Original type object for type discovery (None if formatting TypeInfo directly)
        instance_values: Runtime instance values to show instead of static defaults.
            When provided, field values come from this dict (current state)
            rather than from field.default (source code defaults).

    Returns:
        Python-style class definition string
    """
    lines = []
    ind = "    " * indent

    # Class header
    if info.base == "@dataclass":
        lines.append(f"{ind}@dataclass")
        lines.append(f"{ind}class {info.name}:")
    elif info.base == "@attrs":
        lines.append(f"{ind}@attrs")
        lines.append(f"{ind}class {info.name}:")
    elif info.base:
        lines.append(f"{ind}class {info.name}({info.base}):")
    else:
        lines.append(f"{ind}class {info.name}:")

    # Docstring
    if info.docstring:
        doc_text = _truncate_docstring_lines(info.docstring, max_lines=1 if concise else 9999)
        if "\n" in doc_text:
            lines.append(f'{ind}    """')
            for line in doc_text.split("\n"):
                lines.append(f"{ind}    {line}")
            lines.append(f'{ind}    """')
        else:
            lines.append(f'{ind}    """{doc_text}"""')
        lines.append("")

    # Handle Enum specially - show members as assignments
    if info.base == "Enum":
        fields = info.fields
        truncated = 0
        if max_length and len(fields) > max_length:
            truncated = len(fields) - max_length
            fields = fields[:max_length]

        for field in fields:
            value_str = _pformat_to_str(
                field.default,
                max_length=5,
                max_string=50,
                max_depth=1,
                inline_depth=0,
            )
            lines.append(f"{ind}    {field.name} = {value_str}")

        if truncated:
            # Truncation 3.0: bare ``...`` (matches the rest of the family).
            lines.append(f"{ind}    ...")

        return "\n".join(lines)

    # Regular structured types - show fields as annotations
    fields = info.fields
    truncated_fields = 0
    if max_length and len(fields) > max_length:
        truncated_fields = len(fields) - max_length
        fields = fields[:max_length]

    # Show fields with instance values if available
    seen_field_names = {f.name for f in fields}
    fields_start_len = len(lines)
    for field in fields:
        # Skip fields marked repr=False (Pydantic field parameter — intentionally hidden)
        if not field.repr:
            continue
        if instance_values is not None and field.name not in instance_values:
            continue
        line = f"{ind}    {field.name}: {field.type}"
        if instance_values is not None and field.name in instance_values:
            default_str = _format_value_to_str(
                instance_values[field.name],
                max_length=3,
                max_string=50,
                max_depth=1,
                expand_all=False,
                depth=0,
                indent=0,
            )
            line += f" = {default_str}"
        elif instance_values is None and field.default is not ...:
            default_str = _format_value_to_str(
                field.default,
                max_length=3,
                max_string=50,
                max_depth=1,
                expand_all=False,
                depth=0,
                indent=0,
            )
            line += f" = {default_str}"
        if field.description:
            line += f"  # {field.description}"
        elif type_doc := _field_type_docstring(field.default, context_obj, type_name=field.type):
            line += f"  # {type_doc}"
        lines.append(line)

    if truncated_fields:
        # Truncation 3.0: bare ``...`` (matches the rest of the family).
        lines.append(f"{ind}    ...")

    # Add extra instance attributes not in type fields (like tools assigned at runtime)
    if instance_values:
        extra_attrs = []
        from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

        for name, value in sorted(instance_values.items()):
            if name in seen_field_names:
                continue
            if name.startswith("_"):
                continue
            if context_obj is not None and _is_hidden_field(context_obj, name):
                continue
            # Skip callables unless they're classes
            if callable(value) and not isinstance(value, type):
                continue
            extra_attrs.append((name, value))

        if extra_attrs:
            if fields:
                lines.append("")
            # Truncate extra attrs if needed
            truncated_extra = 0
            if max_length and len(extra_attrs) > max_length:
                truncated_extra = len(extra_attrs) - max_length
                extra_attrs = extra_attrs[:max_length]

            for name, value in extra_attrs:
                type_name = type(value).__name__
                value_str = _format_value_to_str(
                    value,
                    max_length=3,
                    max_string=100,
                    max_depth=1,
                    expand_all=False,
                    depth=0,
                    indent=0,
                )
                lines.append(f"{ind}    {name}: {type_name} = {value_str}")

            if truncated_extra:
                lines.append(f"{ind}    ... +{truncated_extra} more instance attributes")

    # Methods
    if info.methods:
        if len(lines) > fields_start_len:
            lines.append("")

        methods = info.methods
        truncated_methods = 0
        if max_length and len(methods) > max_length:
            truncated_methods = len(methods) - max_length
            methods = methods[:max_length]

        for method in methods:
            method_lines = _format_callable_info(
                method,
                concise=concise,
                type_depth=0,  # Don't show nested references for methods within a class
                indent=indent + 1,
                as_method=True,
            )
            lines.append(method_lines)

        if truncated_methods:
            lines.append(f"{ind}    # ... +{truncated_methods} more methods")

    # Add Referenced Types section if we have context and type_depth > 0
    if context_obj is not None and type_depth > 0:
        from nooa.agentdoc._discover import (
            _extract_types_from_hint,
            _is_custom_type,
            discover_referenced_types,
        )
        from nooa.agentdoc._structured import extract_type_info

        seed_set: set[type] = {
            t for t in discover_referenced_types(context_obj) if not is_expand_false(t)
        }

        # Also discover types from extra instance attributes
        if instance_values:
            extra_discovered: set[type] = set()

            for name, value in instance_values.items():
                if name.startswith("_"):
                    continue
                # Skip callables unless they're classes
                if callable(value) and not isinstance(value, type):
                    continue

                # Extract type from the value
                value_type = type(value)
                if isinstance(value, type):
                    # It's a class itself (like WorkerAgent = WorkerAgent)
                    value_type = value

                # Extract types from the value's type
                _extract_types_from_hint(value_type, extra_discovered)

            # Filter to only custom types and add to seed_set
            for extra_type in extra_discovered:
                if _is_custom_type(extra_type) and not is_expand_false(extra_type):
                    seed_set.add(extra_type)

        # Collect all referenced types transitively (BFS), render flat
        referenced_types = _collect_referenced_types_transitive(seed_set, exclude=context_obj)
        if referenced_types:
            lines.append(f"{ind}## Referenced Types")

            for ref_type in referenced_types:
                ref_info = extract_type_info(ref_type)
                ref_doc = _format_type_info(
                    ref_info,
                    concise=True,
                    type_depth=0,  # All types already collected — no nested ## sections
                    max_length=max_length,
                    indent=indent,
                    context_obj=ref_type,
                )
                lines.append(ref_doc)
                lines.append("")

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).rstrip("\n")


def _format_callable_info(
    info: CallableInfo,
    *,
    concise: bool,
    type_depth: int = 0,
    indent: int,
    as_method: bool = False,
    context_obj: Any = None,
) -> str:
    """Format CallableInfo as Python function/method syntax.

    Args:
        info: CallableInfo to format
        concise: If True, show first-line docstrings only
        type_depth: How deep to recurse into referenced types (0 = none)
        indent: Indentation level
        as_method: If True, format as method (no standalone header)
        context_obj: Original callable object for type discovery (None if formatting CallableInfo directly)

    Returns:
        Python-style function definition string
    """
    lines = []
    ind = "    " * indent

    # Build signature line
    # Determine the display name based on context
    method_name = info.name

    if as_method and "." in method_name:
        # When shown within class, strip class name prefix
        # "ClassName.method_name" -> "method_name"
        method_name = method_name.split(".")[-1]
    elif ".<locals>." in method_name:
        # For nested functions (closures), strip the .<locals>. part
        # "main.<locals>.example_function" -> "example_function"
        method_name = method_name.split(".<locals>.")[-1]
    # Otherwise, keep the full qualified name (e.g., "ClassName.method", "top_level_function")

    if info.is_classmethod:
        lines.append(f"{ind}@classmethod")
    async_prefix = "async def " if info.is_async else "def "
    return_str = f" -> {info.return_type}" if info.return_type else ""
    sig_line = f"{ind}{async_prefix}{method_name}{info.signature}{return_str}:"
    lines.append(sig_line)

    # Docstring
    if info.docstring:
        doc_text = _truncate_docstring_lines(info.docstring, max_lines=1 if concise else 9999)
        if "\n" in doc_text and not concise:
            lines.append(f'{ind}    """')
            for line in doc_text.split("\n"):
                lines.append(f"{ind}    {line}")
            lines.append(f'{ind}    """')
        else:
            lines.append(f'{ind}    """{doc_text}"""')
    else:
        lines.append(f"{ind}    ...")

    # Add Referenced Types section if we have context and type_depth > 0
    # (not when formatting as a method within a class)
    if context_obj is not None and type_depth > 0 and not as_method:
        from nooa.agentdoc._discover import discover_referenced_types
        from nooa.agentdoc._structured import extract_type_info

        seed_set = {t for t in discover_referenced_types(context_obj) if not is_expand_false(t)}
        referenced_types = _collect_referenced_types_transitive(seed_set, exclude=context_obj)

        if referenced_types:
            lines.append("")
            lines.append(f"{ind}## Referenced Types")

            for ref_type in referenced_types:
                ref_info = extract_type_info(ref_type)
                ref_doc = _format_type_info(
                    ref_info,
                    concise=True,
                    type_depth=0,  # All types already collected — no nested ## sections
                    max_length=None,
                    indent=indent,
                    context_obj=ref_type,
                )
                lines.append(ref_doc)
                lines.append("")

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).rstrip("\n")


def _format_module_info(info: ModuleInfo, *, concise: bool, indent: int) -> str:
    """Format ModuleInfo as Python module syntax.

    Args:
        info: ModuleInfo to format
        concise: If True, show first-line docstrings only
        indent: Indentation level

    Returns:
        Python-style module documentation string
    """
    lines = []
    ind = "    " * indent

    # Module header
    lines.append(f"{ind}# {info.name}")
    lines.append("")

    # Module docstring
    if info.docstring:
        doc_text = _truncate_docstring_lines(info.docstring, max_lines=1 if concise else 9999)
        lines.append(f'{ind}"""')
        for line in doc_text.split("\n"):
            lines.append(f"{ind}{line}")
        lines.append(f'{ind}"""')
        lines.append("")

    # Submodules section
    if info.submodules:
        lines.append(f"{ind}# Submodules:")
        for sub_name, sub_doc in info.submodules:
            if sub_doc:
                lines.append(f"{ind}#   {sub_name} — {sub_doc}")
            else:
                lines.append(f"{ind}#   {sub_name}")
        lines.append("")

    # Build lookup dicts so we can render in __all__ declaration order
    cls_map = dict(info.classes)
    func_map = {func.name: func for func in info.functions}
    val_map = dict(info.values)

    # Use ordered_names to interleave classes, callables, and values in
    # their __all__ declaration order rather than grouping by type.
    render_names = (
        info.ordered_names
        if info.ordered_names
        else (
            [n for n, _ in info.classes]
            + [f.name for f in info.functions]
            + [n for n, _ in info.values]
        )
    )

    # Track whether we're in a value-run (dense, no blank lines between values)
    in_value_run = False
    for sym_name in render_names:
        if sym_name in cls_map:
            if in_value_run:
                lines.append("")
                in_value_run = False
            cls_doc = cls_map[sym_name]
            if cls_doc:
                lines.append(f"{ind}class {sym_name}:  # {cls_doc}")
            else:
                lines.append(f"{ind}class {sym_name}")
            lines.append("")
        elif sym_name in func_map:
            if in_value_run:
                lines.append("")
                in_value_run = False
            func = func_map[sym_name]
            func_lines = _format_callable_info(func, concise=concise, indent=indent)
            lines.append(func_lines)
            lines.append("")
        elif sym_name in val_map:
            lines.append(f"{ind}{sym_name} = {val_map[sym_name]}")
            in_value_run = True
        else:
            if in_value_run:
                lines.append("")
                in_value_run = False
            lines.append(f"{ind}# {sym_name}: (not accessible)")
            lines.append("")

    return "\n".join(lines).rstrip()


_MISSING = object()


@functools.lru_cache(maxsize=64)
def _get_type_hints_cached(obj_type: type) -> dict[str, Any]:
    """Cached wrapper around typing.get_type_hints(include_extras=True)."""
    import typing

    try:
        return typing.get_type_hints(obj_type, include_extras=True)
    except Exception:
        return {}


def _field_spec_override(obj_type: type, field_name: str, key: str) -> Any:
    """Read a per-field ``spec()`` override from ``Annotated`` type hints.

    Returns the value of *key* (e.g. ``"max_string"``) from the first
    ``SpecAnnotation`` found in ``Annotated[T, spec(...)]``, or ``_MISSING``
    if absent.
    """
    import typing

    from nooa.agentdoc._docs import SpecAnnotation

    hints = _get_type_hints_cached(obj_type)
    hint = hints.get(field_name)
    if hint is None or typing.get_origin(hint) is not typing.Annotated:
        return _MISSING
    for metadata in typing.get_args(hint)[1:]:
        if isinstance(metadata, SpecAnnotation):
            val = metadata.kwargs.get(key, _MISSING)
            if val is not _MISSING:
                return val
    return _MISSING


def _format_instance_repr(
    obj: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    indent: int,
) -> str:
    """Format an instance in repr-style.

    Shows: ClassName(field1=value1, field2=value2, ...)

    Args:
        obj: Instance to format
        max_length: Max fields to show
        max_string: Max string chars
        max_depth: Max nesting depth
        indent: Indentation level

    Returns:
        Repr-style string
    """
    from enum import Enum

    from nooa.agentdoc._structured import extract_type_info
    from nooa.agentdoc.registry import get_type_info_extractor

    obj_type = type(obj)
    type_name = obj_type.__name__

    # Special handling for Enums
    if isinstance(obj, Enum):
        return f"{type_name}.{obj.name}"

    # Special handling for NamedTuples - show constructor style
    if (
        hasattr(obj_type, "_fields")
        and isinstance(getattr(obj_type, "_fields", None), tuple)
        and isinstance(obj, tuple)
    ):
        fields = obj_type._fields
        values = []
        for i, (field_name, value) in enumerate(zip(fields, obj, strict=True)):
            if max_length and i >= max_length:
                values.append("...")
                break
            value_str = _format_value_to_str(
                value,
                max_length=max_length,
                max_string=max_string,
                max_depth=(max_depth - 1) if max_depth else None,
                expand_all=False,
                depth=0,
                indent=0,
            )
            values.append(f"{field_name}={value_str}")
        return f"{type_name}({', '.join(values)})"

    # Get field values
    extractor = get_type_info_extractor(obj)
    if extractor:
        result = extractor(obj)
        if isinstance(result, tuple):
            type_info, values = result
        else:
            type_info = result
            values = _extract_instance_values(obj, type_info)
        # Registry extractors may return unfiltered fields; apply hidden rules.
        # Local import avoids a circular dependency: _pformat ← _visibility ← _metadata ← _pformat.
        from nooa.agentdoc._visibility import is_hidden_field  # noqa: PLC0415

        type_info = TypeInfo(
            name=type_info.name,
            base=type_info.base,
            fields=[f for f in type_info.fields if not is_hidden_field(obj_type, f.name)],
            methods=type_info.methods,
            docstring=type_info.docstring,
        )
    else:
        type_info = extract_type_info(obj_type)
        if isinstance(obj, SupportsInstanceValues):
            values = obj.__instance_values__()
        else:
            values = _extract_instance_values(obj, type_info)

    # Determine if __instance_values__ was used (gives full control over what's shown)
    uses_instance_values_protocol = isinstance(obj, SupportsInstanceValues)

    # Build repr-style output
    parts = []
    field_count = 0
    # Respect Pydantic exclude=True — excluded fields should never be formatted
    # (they often contain large internal state like captured_locals).
    _repr_excluded: set[str] = set()
    if hasattr(obj_type, "model_fields"):
        _repr_excluded = {
            name
            for name, field_info in obj_type.model_fields.items()
            if getattr(field_info, "exclude", False)
        }
    eligible_fields = [f for f in type_info.fields if f.repr and f.name not in _repr_excluded]
    for field in eligible_fields:
        if max_length and field_count >= max_length:
            # Truncation 3.0: bare ``...`` matches the inner-marker convention
            # used everywhere else (dict items, set items, expanded multi-line).
            # The legacy ``... +N`` form was the last holdout of the old shape.
            parts.append("...")
            break

        # If __instance_values__ is implemented, only show fields it returned
        # (the protocol allows hiding fields by omitting them)
        if uses_instance_values_protocol and field.name not in values:
            continue

        # Get value from values dict or __dict__. Never use getattr as fallback —
        # it triggers descriptors that can block indefinitely.
        if field.name in values:
            value = values[field.name]
        else:
            _obj_dict = getattr(obj, "__dict__", None)
            if _obj_dict is not None and field.name in _obj_dict:
                value = _obj_dict[field.name]
            else:
                # Class-level plain values (non-descriptors) are safe.
                class_val = inspect.getattr_static(obj_type, field.name, _MISSING)
                if (
                    class_val is not _MISSING
                    and inspect.getattr_static(type(class_val), "__get__", None) is None
                ):
                    value = class_val
                else:
                    continue
        # Skip callable instance attributes (e.g. assigned lambdas/functions);
        # only keep types (classes), since those are intentional class-level tools.
        if callable(value) and not isinstance(value, type):
            continue
        # Per-field spec(max_string=...) overrides the caller's default.
        field_max_string = _field_spec_override(obj_type, field.name, "max_string")
        if field_max_string is _MISSING:
            field_max_string = max_string
        value_str = _format_value_to_str(
            value,
            max_length=max_length,
            max_string=field_max_string,
            max_depth=(max_depth - 1) if max_depth else None,
            expand_all=False,
            depth=0,
            indent=0,
        )
        parts.append(f"{field.name}={value_str}")
        field_count += 1

    # Add non-field attributes from __dict__ (skip hidden so they are not re-added)
    if hasattr(obj, "__dict__"):
        from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

        for name, value in sorted(values.items()):
            if any(f.name == name for f in type_info.fields):
                continue
            if name.startswith("_"):
                continue
            if _is_hidden_field(obj_type, name):
                continue
            if max_length and field_count >= max_length:
                break

            value_str = _format_value_to_str(
                value,
                max_length=max_length,
                max_string=max_string,
                max_depth=(max_depth - 1) if max_depth else None,
                expand_all=False,
                depth=0,
                indent=0,
            )
            parts.append(f"{name}={value_str}")
            field_count += 1

    # If no fields were extracted but the class defines a custom __repr__,
    # fall back to repr — the library author likely provides a richer view
    # (e.g. pandas DataFrame, matplotlib objects).
    if not parts:
        for klass in obj_type.__mro__:
            if klass is object:
                break
            if "__repr__" in klass.__dict__:
                try:
                    r = repr(obj)
                except Exception:
                    break
                if max_string is not None and len(r) > max_string:
                    n = len(r)
                    n_head = (max_string + 1) // 2
                    n_tail = max_string - n_head
                    head_repr = repr(r[:n_head])
                    if n_tail > 0:
                        tail_repr = repr(r[-n_tail:])
                        return f"{type_name}(repr_len={n}, [:{n_head}]={head_repr}, [-{n_tail}:]={tail_repr})"
                    return f"{type_name}(repr_len={n}, [:{n_head}]={head_repr})"
                return r

    return f"{type_name}({', '.join(parts)})"


def _extract_instance_values(obj: Any, type_info: TypeInfo) -> dict[str, Any]:
    """Extract current field values from an instance.

    Args:
        obj: Instance to extract values from
        type_info: TypeInfo describing the type

    Returns:
        Dictionary mapping field names to current values
    """
    values = {}

    # First, get values for type fields
    obj_type = type(obj)

    # Respect Pydantic's exclude=True — those fields often contain large
    # internal state (e.g. captured_locals with arbitrary user objects that
    # can trigger expensive I/O when formatted, blocking the event loop).
    _excluded_fields: set[str] = set()
    if hasattr(obj_type, "model_fields"):
        _excluded_fields = {
            name
            for name, field_info in obj_type.model_fields.items()
            if getattr(field_info, "exclude", False)
        }

    obj_dict = getattr(obj, "__dict__", None) or {}

    # Collect slot names for __slots__-based classes (no __dict__)
    _slots: set[str] = set()
    for cls in obj_type.__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            _slots.add(slots)
        else:
            _slots.update(slots)

    for field in type_info.fields:
        if field.name in _excluded_fields:
            continue
        # Read from __dict__ first (instance data, always safe).
        if field.name in obj_dict:
            values[field.name] = obj_dict[field.name]
        elif field.name in _slots:
            # __slots__ values are safe to read — they're data, not descriptors.
            try:
                values[field.name] = object.__getattribute__(obj, field.name)
            except AttributeError:
                pass
        elif isinstance(obj, dict) and field.name in obj:
            # TypedDict instances are dicts
            values[field.name] = obj[field.name]
        else:
            # Fall back to class-level plain values (non-descriptors).
            # This handles annotated class defaults like `x: int = 5`.
            # Descriptors (property, classmethod, etc.) are skipped — they
            # can trigger arbitrary I/O.
            class_val = inspect.getattr_static(obj_type, field.name, _MISSING)
            if (
                class_val is not _MISSING
                and inspect.getattr_static(type(class_val), "__get__", None) is None
            ):
                values[field.name] = class_val

    # Also include all __dict__ attributes (for instances without annotations)
    if hasattr(obj, "__dict__"):
        from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

        for name, value in obj.__dict__.items():
            if (
                name not in values
                and not name.startswith("_")
                and not callable(value)
                and not _is_hidden_field(obj_type, name)
                and name not in _excluded_fields
            ):
                values[name] = value

    return values


def _format_nested_instance(
    obj: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    depth: int,
    indent: int,
) -> str:
    """Format a structured instance in compact one-line form for nested display.

    Used when a structured instance (dataclass, Pydantic, etc.) appears inside
    a container (list, dict). Shows: ClassName(field1=val1, field2=val2, ... +N)

    Args:
        obj: Structured instance to format
        max_length: Max fields to show (default 3 for compact display)
        max_string: Max string chars per field value
        max_depth: Max nesting depth
        depth: Current nesting depth
        indent: Indentation level (unused for compact format)

    Returns:
        Compact one-line representation
    """
    from nooa.agentdoc._structured import extract_type_info
    from nooa.agentdoc.registry import get_type_info_extractor

    obj_type = type(obj)
    type_name = obj_type.__name__

    # Propagate caller's bounds straight through. None = unlimited; explicit
    # value = bound to that count/length. No hidden tighter defaults.
    nested_max_length = max_length
    nested_max_string = max_string

    # Get type info and values
    extractor = get_type_info_extractor(obj)
    if extractor:
        result = extractor(obj)
        if isinstance(result, tuple):
            type_info, values = result
        else:
            type_info = result
            values = _extract_instance_values(obj, type_info)
    else:
        type_info = extract_type_info(obj_type)
        if isinstance(obj, SupportsInstanceValues):
            values = obj.__instance_values__()
        else:
            values = _extract_instance_values(obj, type_info)

    # Collect field names in order (type fields first, then extra __dict__ attrs)
    # Only include fields that exist in values — fields absent from values are skipped
    # in the render loop below, so excluding them keeps truncated_count accurate.
    field_names = [f.name for f in type_info.fields if f.name in values]
    for name in values:
        if name not in field_names and not name.startswith("_"):
            field_names.append(name)

    # Truncate fields if needed
    truncated_count = 0
    if nested_max_length is not None and len(field_names) > nested_max_length:
        truncated_count = len(field_names) - nested_max_length
        field_names = field_names[:nested_max_length]

    # Format each field=value pair
    parts = []
    for name in field_names:
        if name in values:
            value = values[name]
            # Per-field spec(max_string=...) overrides the caller's default.
            field_max_string = _field_spec_override(obj_type, name, "max_string")
            if field_max_string is _MISSING:
                field_max_string = nested_max_string
            value_str = _format_value_to_str(
                value,
                max_length=nested_max_length,
                max_string=field_max_string,
                max_depth=(max_depth - 1) if max_depth else None,
                expand_all=False,
                depth=depth + 1,
                indent=0,
            )
            parts.append(f"{name}={value_str}")

    result = f"{type_name}({', '.join(parts)}"
    if truncated_count:
        result += f", ... +{truncated_count}"
    result += ")"

    return result


# ============================================================================
# Value formatting (for regular Python values)
# ============================================================================


def _format_value(
    _object: Any,
    _stream: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    expand_all: bool,
    depth: int,
    indent: int,
    _seen: set[int] | None = None,
) -> None:
    """Write a regular Python value to *_stream* with truncation.

    Args:
        _object: Value to format.
        _stream: Writer with a ``write(str)`` method.
        max_length: Max elements per container.
        max_string: Max string chars.
        max_depth: Max nesting depth.
        expand_all: Always expand containers.
        depth: Current nesting depth.
        indent: Current indentation level.
        _seen: ``id()`` of containers currently being rendered (cycle guard).
            Top-level callers leave this ``None``; recursive calls thread the
            same set through so nested containers can detect re-entry.
    """
    if _seen is None:
        _seen = set()

    # Check depth limit
    if max_depth is not None and depth >= max_depth:
        _stream.write(_format_shallow(_object, max_string))
        return

    # Handle by type
    if isinstance(_object, str):
        _stream.write(_format_string(_object, max_string))
        return

    if isinstance(_object, dict):
        _format_dict(
            _object,
            _stream,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth,
            indent=indent,
            _seen=_seen,
        )
        return

    if isinstance(_object, (list, tuple, set, frozenset)):
        _format_sequence(
            _object,
            _stream,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth,
            indent=indent,
            _seen=_seen,
        )
        return

    # Handle marker classes from _structured.py (they have clean __repr__)
    from nooa.agentdoc._structured import _ClassRef, _InstanceRef

    if isinstance(_object, (_ClassRef, _InstanceRef)):
        _stream.write(repr(_object))
        return

    # Handle class objects (like child agent classes) - show just the name
    if isinstance(_object, type):
        _stream.write(_object.__name__)
        return

    # Structured instances (dataclass, Pydantic, etc.) - format recursively
    if _is_structured_instance(_object):
        _stream.write(
            _format_nested_instance(
                _object,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                depth=depth,
                indent=indent,
            )
        )
        return

    # Fallback to repr for other types (numpy arrays, pandas frames, custom classes, …).
    # When the repr is over budget, emit the truncation 3.0 marker family —
    # same shape as the rest of the renderer, prefixed with the actual type
    # name so the LLM can tell what was truncated:
    #     ndarray(repr_len=2773, [:H]='array([0, 1, 2, ...', [-T:]='...]')
    try:
        result = repr(_object)
    except Exception:
        _stream.write(f"<{type(_object).__name__}>")
        return
    if max_string is not None and len(result) > max_string:
        n = len(result)
        n_head = (max_string + 1) // 2
        n_tail = max_string - n_head
        type_name = type(_object).__name__
        head_repr = repr(result[:n_head])
        if n_tail > 0:
            tail_repr = repr(result[-n_tail:])
            _stream.write(
                f"{type_name}(repr_len={n}, [:{n_head}]={head_repr}, [-{n_tail}:]={tail_repr})"
            )
        else:
            _stream.write(f"{type_name}(repr_len={n}, [:{n_head}]={head_repr})")
        return
    _stream.write(result)


def _format_value_to_str(
    _object: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    expand_all: bool = False,
    depth: int = 0,
    indent: int = 0,
) -> str:
    """Convenience wrapper: call ``_format_value`` into an ``io.StringIO`` and return the string."""
    tmp = io.StringIO()
    _format_value(
        _object,
        tmp,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        expand_all=expand_all,
        depth=depth,
        indent=indent,
    )
    return tmp.getvalue()


def _format_string(s: str, max_string: int | None) -> str:
    """Format a string, potentially truncating.

    Untruncated strings render with the usual repr (or triple-quote multiline
    form when the content has newlines or mixed quote types).

    Truncated strings use the slice-keys marker — same shape as ordered
    containers — so the count is upfront and head/tail chunks are explicitly
    anchored to their character positions::

        str(len=789516, [:250]='Lorem ipsum...', [-250:]='...end of string')

    This is the same marker family the rest of the renderer uses.
    """
    needs_multiline = "\n" in s or ("'" in s and '"' in s)

    # Untruncated → plain repr or triple-quote multiline.
    if max_string is None or len(s) <= max_string:
        return _format_multiline_string(s) if needs_multiline else repr(s)

    # Truncated → slice-keys marker. Split max_string evenly: ceiling head,
    # floor tail. Falls back to head-only when max_string < 2.
    n = len(s)
    n_head = (max_string + 1) // 2
    n_tail = max_string - n_head
    head = s[:n_head]
    tail = s[-n_tail:] if n_tail > 0 else ""

    head_repr = repr(head)
    if n_tail > 0:
        tail_repr = repr(tail)
        return f"str(len={n}, [:{n_head}]={head_repr}, [-{n_tail}:]={tail_repr})"
    return f"str(len={n}, [:{n_head}]={head_repr})"


def _format_multiline_string(s: str) -> str:
    """Format a string using triple quotes for better readability.

    Chooses quote style to avoid escaping:
    - Use ''' if string doesn't contain '''
    - Use \"\"\" if string doesn't contain \"\"\"
    - Fall back to repr() if string contains both
    """
    has_triple_single = "'''" in s
    has_triple_double = '"""' in s

    if has_triple_single and has_triple_double:
        # Both triple quote styles present - fall back to repr
        return repr(s)

    # Choose quote style that doesn't need escaping
    quote = '"""' if has_triple_single else "'''"

    # Format with actual newlines preserved
    return f"{quote}{s}{quote}"


def _format_shallow(_object: Any, max_string: int | None) -> str:
    """Format object shallowly (at max depth)."""
    type_name = type(_object).__name__

    if isinstance(_object, dict):
        if not _object:  # Empty dict - just show {}
            return "{}"
        return f"{{{type_name}: {len(_object)} items}}"
    if isinstance(_object, (list, tuple, set, frozenset)):
        brackets = _get_brackets(type(_object))
        if not _object:  # Empty container - just show [], (), etc.
            return brackets[0] + brackets[1]
        return f"{brackets[0]}{type_name}: {len(_object)} items{brackets[1]}"
    if isinstance(_object, str):
        return _format_string(_object, max_string)

    try:
        return repr(_object)
    except Exception:
        return f"<{type(_object).__name__}>"


def _format_dict(
    d: dict,
    _stream: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    expand_all: bool,
    depth: int,
    indent: int,
    _seen: set[int] | None = None,
) -> None:
    """Write a dictionary to *_stream*.

    Untruncated dicts render as plain Python literals (``{1: 2, 3: 4}``).

    Truncated dicts use the truncation 3.0 ``items`` marker — the same outer
    wrapper as truncated sets, since dict keys aren't positional anchors::

        dict(len=100, items={0: 42, 1: 17, ..., 98: 45, 99: 28})

    Cycles: if ``_seen`` already contains ``id(d)``, emit ``<cycle>`` and
    return. Same mechanism as ``_format_sequence`` — Python's built-in
    ``Py_ReprEnter`` handles this for ``dict.__repr__``; our recursive
    renderer needs to replicate it explicitly.
    """
    if _seen is None:
        _seen = set()
    if id(d) in _seen:
        _stream.write("<cycle>")
        return

    type_name = _get_type_name(type(d)) if type(d) is not dict else "dict"

    if not d:
        _stream.write("{}")
        return

    _seen.add(id(d))
    try:
        _format_dict_body(
            d,
            _stream,
            type_name,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth,
            indent=indent,
            _seen=_seen,
        )
    finally:
        _seen.discard(id(d))


def _format_dict_body(
    d,
    _stream,
    type_name,
    *,
    max_length,
    max_string,
    max_depth,
    expand_all,
    depth,
    indent,
    _seen,
):
    """Inner rendering body — split out so the cycle-guard try/finally above stays compact."""
    all_items = list(d.items())
    n_total = len(all_items)
    truncated = max_length is not None and n_total > max_length

    if truncated:
        assert max_length is not None  # narrowed by `truncated`; pyright can't see it
        n_head = (max_length + 1) // 2
        n_tail = max_length - n_head
        head_items = all_items[:n_head]
        tail_items = all_items[-n_tail:] if n_tail > 0 else []
    else:
        head_items = all_items
        tail_items = []

    def _fmt_item_to_str(k: object, v: object, current_indent: int) -> tuple[str, str]:
        k_str = _format_string(str(k), 50) if isinstance(k, str) else repr(k)
        tmp = io.StringIO()
        _format_value(
            v,
            tmp,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth + 1,
            indent=current_indent,
            _seen=_seen,
        )
        return k_str, tmp.getvalue()

    # Untruncated → plain Python literal.
    if not truncated:
        if not expand_all:
            parts = []
            for k, v in head_items:
                k_str, v_str = _fmt_item_to_str(k, v, 0)
                parts.append(f"{k_str}: {v_str}")
            trial = "{" + ", ".join(parts) + "}"
            if len(trial) < 120:
                _stream.write(trial)
                return

        inner_indent = "    " * (indent + 1)
        _stream.write("{\n")
        for k, v in head_items:
            k_str = _format_string(str(k), 50) if isinstance(k, str) else repr(k)
            _stream.write(f"{inner_indent}{k_str}: ")
            _format_value(
                v,
                _stream,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
                depth=depth + 1,
                indent=indent + 1,
                _seen=_seen,
            )
            _stream.write(",\n")
        _stream.write("    " * indent + "}")
        return

    # Truncated → marker form.
    visible = head_items + tail_items
    if not expand_all:
        parts = []
        for k, v in visible:
            k_str, v_str = _fmt_item_to_str(k, v, 0)
            parts.append(f"{k_str}: {v_str}")
        # Insert a "..." separator between head and tail for readability.
        if tail_items:
            sep_idx = len(head_items)
            parts.insert(sep_idx, "...")
        trial = type_name + "(len=" + str(n_total) + ", items={" + ", ".join(parts) + "})"
        if len(trial) < 120:
            _stream.write(trial)
            return

    # Expanded multi-line marker form.
    inner_indent = "    " * (indent + 1)
    _stream.write(f"{type_name}(len={n_total}, items={{\n")
    for k, v in head_items:
        k_str = _format_string(str(k), 50) if isinstance(k, str) else repr(k)
        _stream.write(f"{inner_indent}{k_str}: ")
        _format_value(
            v,
            _stream,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth + 1,
            indent=indent + 1,
            _seen=_seen,
        )
        _stream.write(",\n")
    if tail_items:
        _stream.write(f"{inner_indent}...\n")
        for k, v in tail_items:
            k_str = _format_string(str(k), 50) if isinstance(k, str) else repr(k)
            _stream.write(f"{inner_indent}{k_str}: ")
            _format_value(
                v,
                _stream,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
                depth=depth + 1,
                indent=indent + 1,
                _seen=_seen,
            )
            _stream.write(",\n")
    _stream.write("    " * indent + "})")


def _format_sequence(
    seq,
    _stream: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    expand_all: bool,
    depth: int,
    indent: int,
    _seen: set[int] | None = None,
) -> None:
    """Write a sequence (list, tuple, set, frozenset) to *_stream*.

    Untruncated sequences render as plain Python literals (``[1, 2, 3]``).

    Truncated sequences use the truncation 3.0 marker family:

    * **Ordered** (``list``, ``tuple``) — slice-keys notation that explicitly
      anchors the visible items to their positions::

          list(len=100, [:5]=[42, 17, 89, 33, 8], [-5:]=[56, 71, 12, 45, 28])

    * **Unordered** (``set``, ``frozenset``) — items-style wrapper, since
      positional slices aren't meaningful::

          set(len=100, items={42, 17, 89, ...})

    The marker's *presence* signals truncation — a bare ``[1, 2, 3]`` is
    always a complete value.

    Cycles: if ``_seen`` already contains ``id(seq)``, emit ``<cycle>`` and
    return. This matches Python's built-in ``Py_ReprEnter`` behaviour for
    ``list.__repr__``/``dict.__repr__``, which our recursive renderer
    otherwise wouldn't replicate (would RecursionError on
    ``x = []; x.append(x)``).
    """
    if _seen is None:
        _seen = set()
    if id(seq) in _seen:
        _stream.write("<cycle>")
        return

    brackets = _get_brackets(type(seq))
    type_name = _get_type_name(type(seq))
    is_ordered = isinstance(seq, (list, tuple))

    if not seq:
        _stream.write(brackets[0] + brackets[1])
        return

    # Register this container as currently being rendered. All recursive calls
    # below pass _seen, so any cycle back to this container emits ``<cycle>``.
    _seen.add(id(seq))
    try:
        _format_sequence_body(
            seq,
            _stream,
            brackets,
            type_name,
            is_ordered,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth,
            indent=indent,
            _seen=_seen,
        )
    finally:
        _seen.discard(id(seq))


def _format_sequence_body(
    seq,
    _stream,
    brackets,
    type_name,
    is_ordered,
    *,
    max_length,
    max_string,
    max_depth,
    expand_all,
    depth,
    indent,
    _seen,
):
    """Inner rendering body — split out so the cycle-guard try/finally above stays compact."""
    all_items = list(seq)
    n_total = len(all_items)
    truncated = max_length is not None and n_total > max_length

    if truncated:
        # narrowed by `truncated`; pyright can't see it
        assert max_length is not None
        # Compute head/tail split. Ordered → balanced head + tail; unordered
        # → all visible items go in the head (no positional anchor).
        if is_ordered:
            n_head = (max_length + 1) // 2  # ceiling half
            n_tail = max_length - n_head  # floor half
            head_items = all_items[:n_head]
            tail_items = all_items[-n_tail:] if n_tail > 0 else []
        else:
            n_head = max_length
            n_tail = 0
            head_items = all_items[:max_length]
            tail_items = []
    else:
        head_items = all_items
        tail_items = []
        n_head = len(all_items)
        n_tail = 0

    def _fmt_to_str(x: object, current_indent: int) -> str:
        tmp = io.StringIO()
        _format_value(
            x,
            tmp,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            depth=depth + 1,
            indent=current_indent,
            _seen=_seen,
        )
        return tmp.getvalue()

    # Untruncated → plain Python literal.
    if not truncated:
        if not expand_all:
            trial_parts = [_fmt_to_str(x, 0) for x in head_items]
            trial = brackets[0] + ", ".join(trial_parts) + brackets[1]
            if len(trial) < 120:
                _stream.write(trial)
                return

        inner_indent = "    " * (indent + 1)
        _stream.write(brackets[0] + "\n")
        for item in head_items:
            _stream.write(inner_indent)
            _format_value(
                item,
                _stream,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
                depth=depth + 1,
                indent=indent + 1,
                _seen=_seen,
            )
            _stream.write(",\n")
        _stream.write("    " * indent + brackets[1])
        return

    # Truncated → marker form.
    if not expand_all:
        if is_ordered:
            head_str = brackets[0] + ", ".join(_fmt_to_str(x, 0) for x in head_items) + brackets[1]
            if tail_items:
                tail_str = (
                    brackets[0] + ", ".join(_fmt_to_str(x, 0) for x in tail_items) + brackets[1]
                )
                trial = (
                    f"{type_name}(len={n_total}, [:{n_head}]={head_str}, [-{n_tail}:]={tail_str})"
                )
            else:
                trial = f"{type_name}(len={n_total}, [:{n_head}]={head_str})"
        else:
            # Unordered: include an internal "..." after the visible items so
            # the partial-ness is obvious from inside the braces. The exact
            # count is already in len=N; a bare "..." is the visual cue.
            inner = ", ".join(_fmt_to_str(x, 0) for x in head_items) + ", ..."
            items_str = brackets[0] + inner + brackets[1]
            trial = f"{type_name}(len={n_total}, items={items_str})"
        if len(trial) < 120:
            _stream.write(trial)
            return

    # Expanded multi-line marker form.
    inner_indent = "    " * (indent + 1)
    if is_ordered:
        _stream.write(f"{type_name}(len={n_total},\n")
        _stream.write(f"{inner_indent}[:{n_head}]={brackets[0]}\n")
        for item in head_items:
            _stream.write("    " + inner_indent)
            _format_value(
                item,
                _stream,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
                depth=depth + 1,
                indent=indent + 2,
                _seen=_seen,
            )
            _stream.write(",\n")
        _stream.write(f"{inner_indent}{brackets[1]},\n")
        if tail_items:
            _stream.write(f"{inner_indent}[-{n_tail}:]={brackets[0]}\n")
            for item in tail_items:
                _stream.write("    " + inner_indent)
                _format_value(
                    item,
                    _stream,
                    max_length=max_length,
                    max_string=max_string,
                    max_depth=max_depth,
                    expand_all=expand_all,
                    depth=depth + 1,
                    indent=indent + 2,
                    _seen=_seen,
                )
                _stream.write(",\n")
            _stream.write(f"{inner_indent}{brackets[1]},\n")
        _stream.write("    " * indent + ")")
    else:
        _stream.write(f"{type_name}(len={n_total}, items={brackets[0]}\n")
        for item in head_items:
            _stream.write(inner_indent)
            _format_value(
                item,
                _stream,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
                depth=depth + 1,
                indent=indent + 1,
                _seen=_seen,
            )
            _stream.write(",\n")
        # Internal "..." marker — partial-ness obvious from inside the braces too.
        _stream.write(f"{inner_indent}...\n")
        _stream.write("    " * indent + brackets[1] + ")")


def _get_brackets(seq_type: type) -> tuple[str, str]:
    """Get opening and closing brackets for sequence type."""
    if seq_type is list:
        return "[", "]"
    if seq_type is tuple:
        return "(", ")"
    if seq_type is set:
        return "{", "}"
    if seq_type is frozenset:
        return "frozenset({", "})"
    return "[", "]"


def _get_type_name(seq_type: type) -> str:
    """Type name to use as the truncation marker prefix (``list(len=N, ...)``)."""
    return seq_type.__name__
