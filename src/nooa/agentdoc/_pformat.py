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

from nooa.agentdoc._document import (
    DocumentSubject,
    collect_subject_references,
    normalize_subjects,
)
from nooa.agentdoc._format_observation import record_truncation
from nooa.agentdoc._info import REQUIRED, CallableInfo, ModuleInfo, TypeInfo
from nooa.agentdoc._introspection import _is_structured_instance
from nooa.agentdoc._structured import _ClassRef, _InstanceRef
from nooa.agentdoc._visibility import _type_hint_scope

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


def _format_document_subject(
    subject: DocumentSubject,
    *,
    concise: bool,
    max_length: int | None = None,
    indent: int = 0,
) -> str:
    """Format one already-normalized document subject without discovering references."""
    info = subject.info
    if isinstance(info, TypeInfo):
        return _format_type_info(
            info,
            concise=concise,
            type_depth=0,
            max_length=max_length,
            indent=indent,
            context_obj=subject.represented_type,
            instance_values=subject.values,
            visibility_obj=subject.obj if subject.values is not None else None,
        )
    if isinstance(info, CallableInfo):
        return _format_callable_info(info, concise=concise, type_depth=0, indent=indent)
    if isinstance(info, ModuleInfo):
        return _format_module_info(
            info,
            concise=concise,
            concise_members=bool(getattr(subject.obj, "__agentdoc_concise_members__", False)),
            indent=indent,
        )
    return _pformat_to_str(
        subject.obj,
        concise=concise,
        inline_depth=0,
        max_length=None,
        max_string=None,
        max_depth=3,
        instance_mode="type",
    )


def render_document(objs: list[Any], *, concise: bool, inline_depth: int) -> str:
    """Render primaries and one shared referenced-types section."""
    subjects = normalize_subjects(objs)
    sections = [_format_document_subject(subject, concise=concise) for subject in subjects]
    references = collect_subject_references(subjects, max_depth=inline_depth)
    if references:
        sections.extend(("", "## Referenced Types"))
        sections.extend(_format_document_subject(subject, concise=True) for subject in references)
    return "\n".join(sections)


def _render_inline_references(
    subject: DocumentSubject, *, depth: int, max_length: int | None, indent: int
) -> str:
    """Render the historical inline reference suffix used by direct ``pformat`` calls."""
    references = collect_subject_references([subject], max_depth=depth)
    if not references:
        return ""
    ind = "    " * indent
    rendered = [
        _format_document_subject(ref, concise=True, max_length=max_length, indent=indent)
        for ref in references
    ]
    return f"{ind}## Referenced Types\n" + "\n\n".join(rendered)


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
            subject = normalize_subjects([_object])[0]
            assert isinstance(subject.info, TypeInfo)
            type_info, values = subject.info, subject.values or {}
            _stream.write(
                _format_type_info(
                    type_info,
                    concise=concise,
                    type_depth=inline_depth,
                    max_length=max_length,
                    indent=_indent,
                    context_obj=type(_object),
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

    # Direct pformat() keeps its historical inline suffix, but delegates graph policy.
    if context_obj is not None and type_depth > 0:
        suffix = _render_inline_references(
            DocumentSubject(
                context_obj,
                info,
                instance_values,
                context_obj if isinstance(context_obj, type) else type(context_obj),
            ),
            depth=type_depth,
            max_length=max_length,
            indent=indent,
        )
        if suffix:
            lines.append(suffix)

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

    # Direct pformat() keeps its historical inline suffix, but delegates graph policy.
    if context_obj is not None and type_depth > 0 and not as_method:
        suffix = _render_inline_references(
            DocumentSubject(context_obj, info),
            depth=type_depth,
            max_length=None,
            indent=indent,
        )
        if suffix:
            lines.append("")
            lines.append(suffix)

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
    """Report what the snapshot left out to ``observe_truncation`` (the prefill warning).

    A cut inside a node-valued mapping key (a tuple key, say) is reported as
    ``dictionary_key``: keys have their own fixed cap, which the prefill
    limits do not change.
    """
    from nooa.snapshot import Mapping, Object, Sequence, Text, Unavailable

    stack: list[tuple[Any, bool]] = [(snapshot, False)]  # (node, inside a key)
    while stack:
        node, in_key = stack.pop()
        node_type = type(node)
        if node_type is Text:
            if not node.complete:
                record_truncation("dictionary_key" if in_key else "max_string")
            continue
        if node_type is Unavailable:
            if node.reason in _CUT_REASONS:
                record_truncation("dictionary_key" if in_key else _CUT_REASONS[node.reason])
            continue
        if node_type is Sequence:
            stack.extend((child, in_key) for child in node.head + node.tail)
        elif node_type is Mapping:
            for key, child in node.head + node.tail:
                if type(key) is str:
                    if len(key) > 50:
                        record_truncation("dictionary_key")
                else:  # a node key: its cuts are the key's
                    stack.append((key, True))
                stack.append((child, in_key))
        elif node_type is Object:
            stack.extend((child, in_key) for _name, child in node.fields)
        else:
            continue
        if node.reason in _CUT_REASONS:
            record_truncation("dictionary_key" if in_key else _CUT_REASONS[node.reason])


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
