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

import importlib
import inspect
import io
import re
from typing import TYPE_CHECKING, Any

from nooa.agentdoc._format_observation import record_truncation
from nooa.agentdoc._info import REQUIRED, CallableInfo, ModuleInfo, TypeInfo
from nooa.agentdoc._metadata import is_expand_false
from nooa.agentdoc._structured import _ClassRef, _InstanceRef
from nooa.agentdoc._visibility import _type_hint_scope
from nooa.agentdoc.protocols import SupportsInstanceValues

if TYPE_CHECKING:
    from nooa.snapshot import Limits


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


def _collect_referenced_types(
    seed_types: set[type],
    *,
    exclude: type | None = None,
    max_depth: int,
) -> list[type]:
    """Collect referenced types with bounded breadth-first traversal.

    ``seed_types`` are the direct references (depth 1). Each subsequent level
    follows references from the preceding level until ``max_depth`` is reached.
    """
    from nooa.agentdoc._discover import discover_referenced_types

    all_types: set[type] = set()
    # Exclude the primary type from the seed too: a type that references itself in its
    # own method signatures (common, e.g. DataFrame methods returning DataFrame) must
    # not list itself under "Referenced Types".
    frontier = {t for t in seed_types if t is not exclude}
    for _ in range(max_depth):
        if not frontier:
            break
        all_types.update(frontier)
        next_frontier: set[type] = set()
        for ref_type in frontier:
            for new_type in discover_referenced_types(ref_type):
                if (
                    new_type not in all_types
                    and not is_expand_false(new_type)
                    and new_type is not exclude
                ):
                    next_frontier.add(new_type)
        frontier = next_frontier
    return sorted(all_types, key=lambda type_: type_.__name__)


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


@_type_hint_scope()
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
    _indent: int = 0,
    _max_chars: int | None = None,
) -> None:
    """Write pretty-formatted representation of *_object* to *_stream*.

    Args:
        _object: Object to format.
        _stream: Writer with a ``write(str)`` method: plain ``io.StringIO``,
            or a chars ``nooa.snapshot.TextStream`` for a hard char cap.
        max_length: Max elements per container (None=unlimited).
        max_string: Max string chars (None=unlimited).
        max_depth: Max nesting depth (None=unlimited).
        concise: If True, show first-line docstrings only.
        inline_depth: How deep to expand referenced types inline.
        expand_all: Always expand containers.
        instance_mode: How to format instances — "repr" or "type".
        _indent: Current indentation level (internal).
        _max_chars: A value's capture budget in chars (``truncating_pformat``'s
            ``max_chars``): the value is captured within it, so memory and time
            follow the output, not the input. None: unbounded (internal).
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
            _stream.write(
                _format_module_info(
                    module_info,
                    concise=concise,
                    concise_members=bool(getattr(_object, "__agentdoc_concise_members__", False)),
                    indent=_indent,
                )
            )
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
        case _ if instance_mode == "type" and _is_structured_instance(
            _object, respect_custom_repr=False
        ):
            # Instance of a structured type, for doc(): its type structure with
            # runtime values. doc() deliberately ignores a custom __repr__:
            # documentation must retain the type's API contract.
            from nooa.agentdoc._structured import extract_type_info
            from nooa.agentdoc._visibility import is_hidden_field
            from nooa.agentdoc.registry import get_type_info_extractor

            obj_type = type(_object)
            # Check if instance has spec() overrides that could unhide class-hidden fields.
            # spec(self, "field", hidden=False) in __init__ → per-instance opt-in.
            _instance_fields_meta = (_instance_dict(_object) or {}).get(
                "_agentdoc_fields_docs"
            ) or {}
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
                    fields=[f for f in type_info.fields if not is_hidden_field(_object, f.name)],
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
                        fields=[f for f in raw_info.fields if not is_hidden_field(_object, f.name)],
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
                        fields=[f for f in raw_info.fields if not is_hidden_field(_object, f.name)],
                        methods=protocol_info.methods,
                        docstring=protocol_info.docstring,
                    )
                else:
                    type_info = extract_type_info(obj_type)
                values = _extract_instance_values(_object, type_info)

            # Apply per-instance visibility in every extraction branch.
            # This handles both hidden=False opt-ins and hidden=True overrides
            # on fields already present in the type-level contract.
            type_info = TypeInfo(
                name=type_info.name,
                base=type_info.base,
                fields=[f for f in type_info.fields if not is_hidden_field(_object, f.name)],
                methods=type_info.methods,
                docstring=type_info.docstring,
            )

            _stream.write(
                _format_type_info(
                    type_info,
                    concise=concise,
                    type_depth=inline_depth,
                    max_length=max_length,
                    indent=_indent,
                    context_obj=obj_type,
                    instance_values=values,
                    visibility_obj=_object,
                )
            )
        case _:
            # A value: nooa.snapshot's capture + render_python.
            _stream.write(
                _value_text(
                    _object,
                    max_length=max_length,
                    max_string=max_string,
                    max_depth=max_depth,
                    expand_all=expand_all,
                    max_chars=_max_chars,
                )
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
        _indent=_indent,
    )
    return stream.getvalue()


def _instance_dict(obj: Any) -> dict[str, Any] | None:
    """Return an instance ``__dict__`` without invoking ``__getattr__``."""
    try:
        value = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return None
    return value if isinstance(value, dict) else None


def _is_structured_instance(obj: Any, *, respect_custom_repr: bool = True) -> bool:
    """Check if object is an instance that should be formatted with type info.

    Returns True for:
    - Pydantic models
    - dataclasses
    - NamedTuples
    - attrs classes
    - Any custom class instance with __dict__ (not built-in types)

    Args:
        obj: Candidate instance.
        respect_custom_repr: If true, plain classes with custom ``__repr__``
            are treated as values. ``doc()`` disables this to retain type docs.

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

    # pformat() respects a custom __repr__, while doc() always renders the
    # type-level API contract and augments it with runtime instance fields.
    if respect_custom_repr:
        for klass in obj_type.__mro__:
            if klass is object:
                break
            if "__repr__" in klass.__dict__:
                return False

    # In doc mode, every non-builtin instance is documentable even when an
    # empty/private-only __slots__ leaves it with no runtime values. pformat()
    # still requires a public slot before choosing structured value rendering.
    if _instance_dict(obj) is None:
        if not respect_custom_repr:
            return True
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
    visibility_obj: Any = None,
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
        visibility_obj: Instance whose field-level visibility overrides apply.

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
        line = f"{ind}    {field.name}: {field.type}"
        if instance_values is not None and field.name in instance_values:
            default_str = _format_value_to_str(
                instance_values[field.name],
                max_length=3,
                max_string=50,
                max_depth=1,
                expand_all=False,
            )
            line += f" = {default_str}"
        elif field.default is not ...:
            default_str = _format_value_to_str(
                field.default,
                max_length=3,
                max_string=50,
                max_depth=1,
                expand_all=False,
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
            visibility_target = visibility_obj if visibility_obj is not None else context_obj
            if visibility_target is not None and _is_hidden_field(visibility_target, name):
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
        from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

        visible_field_names = {field.name for field in info.fields}
        seed_set: set[type] = {
            t
            for t in discover_referenced_types(
                context_obj,
                field_names=visible_field_names if visibility_obj is not None else None,
            )
            if not is_expand_false(t)
        }

        # Also discover types from extra instance attributes
        if instance_values:
            extra_discovered: set[type] = set()

            for name, value in instance_values.items():
                if name.startswith("_"):
                    continue
                if visibility_obj is not None and _is_hidden_field(visibility_obj, name):
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

        # Collect referenced types to the requested depth, then render them flat.
        referenced_types = _collect_referenced_types(
            seed_set, exclude=context_obj, max_depth=type_depth
        )
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
        referenced_types = _collect_referenced_types(
            seed_set, exclude=context_obj, max_depth=type_depth
        )

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


def _format_module_info(
    info: ModuleInfo,
    *,
    concise: bool,
    concise_members: bool = False,
    indent: int,
) -> str:
    """Format ModuleInfo as Python module syntax.

    Args:
        info: ModuleInfo to format
        concise: If True, show only the first line of every docstring
        concise_members: If True, keep the module docstring but shorten member docs
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
            func_lines = _format_callable_info(
                func, concise=concise or concise_members, indent=indent
            )
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

    obj_dict = _instance_dict(obj) or {}

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
            # A subclass may replace an inherited slot with an arbitrary
            # descriptor. Only invoke the concrete member descriptor found by
            # static lookup; properties and custom descriptors are not safe.
            slot_descriptor = inspect.getattr_static(obj_type, field.name, _MISSING)
            if inspect.ismemberdescriptor(slot_descriptor):
                try:
                    values[field.name] = slot_descriptor.__get__(obj, obj_type)
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

    # Also include runtime-only attributes that are not declared type fields.
    # Most objects store them in __dict__; Pydantic models with extra="allow"
    # store them separately in __pydantic_extra__.
    from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

    def _include_dynamic(name: str, value: Any) -> bool:
        return (
            name not in values
            and not name.startswith("_")
            and not callable(value)
            and not _is_hidden_field(obj, name)
            and name not in _excluded_fields
        )

    for name, value in obj_dict.items():
        if _include_dynamic(name, value):
            values[name] = value

    pydantic_extra = None
    if hasattr(obj_type, "model_fields"):
        try:
            pydantic_extra = object.__getattribute__(obj, "__pydantic_extra__")
        except AttributeError:
            pass
    if isinstance(pydantic_extra, dict):
        for name, value in pydantic_extra.items():
            if _include_dynamic(name, value):
                values[name] = value

    return values


# ============================================================================
# Value rendering: nooa.snapshot's capture + render_python
# ============================================================================

_LAYOUT_WIDTH = 120
"""agentdoc's layout: a container whose one-line form is this long or longer goes on several lines."""
_UNLIMITED_DEPTH = 64
"""``max_depth=None``: nesting this deep is still shown (capture is recursive)."""
_MAX_NODES = 200_000
"""The inspection work of one value preview: a bound even when every limit is ``None``."""
_MAX_CHARS = 100_000_000
"""capture's JSON budget, out of the way: pformat bounds items, strings and depth, not JSON size.

``truncating_pformat(max_chars=N)`` passes its own budget instead (``_value_limits``)."""
_MIN_CHARS = 4
"""The smallest capture budget ``Limits`` takes."""

_CUT_REASONS = {
    "max_items": "max_length",
    "max_length": "max_length",
    "max_nodes": "max_length",
    "key": "max_length",
    "max_depth": "max_depth",
}


def _value_limits(
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    max_chars: int | None = None,
) -> "Limits":
    """pformat's limits as one ``nooa.snapshot.Limits``: both ends kept, a bounded repr fallback.

    ``max_chars``: the capture's JSON budget (``truncating_pformat``'s bound);
    None: ``_MAX_CHARS``, out of the way.
    """
    from nooa.snapshot import Limits

    return Limits(
        max_chars=_MAX_CHARS if max_chars is None else max(max_chars, _MIN_CHARS),
        max_nodes=_MAX_NODES,
        max_depth=_UNLIMITED_DEPTH if max_depth is None else max_depth,
        max_items=max_length,
        max_string=max_string,
        tail_share=0.5,
        fallback="repr",
    )


def _record_cuts(snapshot: Any) -> None:
    """Report what the snapshot left out to ``observe_truncation`` (the prefill warning)."""
    from nooa.snapshot import Mapping, Object, Sequence, Text, Unavailable

    stack = [snapshot]
    while stack:
        node = stack.pop()
        node_type = type(node)
        if node_type is Text:
            if not node.complete:
                record_truncation("max_string")
            continue
        if node_type is Unavailable:
            if node.reason in _CUT_REASONS:
                record_truncation(_CUT_REASONS[node.reason])
            continue
        if node_type is Sequence:
            stack.extend(node.head + node.tail)
        elif node_type is Mapping:
            for key, child in node.head + node.tail:
                if type(key) is str and len(key) > 50:
                    record_truncation("dictionary_key")
                stack.append(child)
        elif node_type is Object:
            stack.extend(child for _name, child in node.fields)
        else:
            continue
        if node.reason in _CUT_REASONS:
            record_truncation(_CUT_REASONS[node.reason])


def _value_text(
    _object: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    expand_all: bool = False,
    max_chars: int | None = None,
) -> str:
    """A value in pformat's notation: ``render_python(capture(value, limits))``.

    Values are read by ``nooa.snapshot.capture`` without running their code;
    what it cannot read falls back to a ``repr`` cut to ``max_string``
    (``Limits(fallback="repr")``). agentdoc's own default markers keep their
    ``repr``. The layout is agentdoc's (``render_python(width=120)``;
    ``expand_all`` puts every container on several lines). ``max_chars``: the
    capture's budget (a bounded preview, see ``_value_limits``).
    """
    from nooa.snapshot import capture, render_python

    if type(_object) is _ClassRef or type(_object) is _InstanceRef:
        return repr(_object)
    snapshot = capture(_object, _value_limits(max_length, max_string, max_depth, max_chars))
    _record_cuts(snapshot)
    return render_python(snapshot, width=0 if expand_all else _LAYOUT_WIDTH)


def _format_value_to_str(
    _object: Any,
    *,
    max_length: int | None,
    max_string: int | None,
    max_depth: int | None,
    expand_all: bool = False,
) -> str:
    """A field default or instance value in ``doc()``: a class by its name, else ``_value_text``."""
    if isinstance(_object, type):
        return _object.__name__
    return _value_text(
        _object,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        expand_all=expand_all,
    )
