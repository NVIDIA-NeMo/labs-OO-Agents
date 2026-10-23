# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified structured type extraction for agentdoc.

Extracts a common representation from Pydantic, dataclass, NamedTuple,
TypedDict, attrs, Enum, and plain classes.
"""

import contextlib
import dataclasses
import enum
import functools
import inspect
import types
import typing
from typing import Any

from nooa.agentdoc._info import REQUIRED, CallableInfo, FieldInfo, ModuleInfo, TypeInfo
from nooa.agentdoc.protocols import SupportsCallableInfo, SupportsTypeInfo

# Re-export REQUIRED for backward compatibility
__all__ = [
    "REQUIRED",
    "TypeInfo",
    "CallableInfo",
    "ModuleInfo",
    "FieldInfo",
    "extract_type_info",
    "extract_callable_info",
    "extract_module_info",
    "format_type",
]


class _ClassRef:
    """Marker for class attribute defaults - formats as just the class name."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _InstanceRef:
    """Marker for instance attribute defaults - formats as ClassName()."""

    def __init__(self, class_name: str):
        self.class_name = class_name

    def __repr__(self) -> str:
        return f"{self.class_name}()"


def _convert_to_marker(value: Any) -> Any:
    """Convert a value to a clean marker for display.

    - Primitives (str, int, float, bool, None) are returned as-is
    - Empty containers ([], {}, set()) are returned as-is
    - REQUIRED sentinel is returned as-is
    - Classes are converted to _ClassRef
    - Instances of custom classes are converted to _InstanceRef

    This ensures that defaults render cleanly in documentation.
    """
    # Return REQUIRED as-is
    if value is REQUIRED:
        return REQUIRED

    # Return primitives as-is
    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    # Return empty containers as-is
    if value == [] or value == {} or value == set():
        return value

    # Return small simple containers as-is (only if all elements are primitives)
    if isinstance(value, (list, tuple, dict, set, frozenset)) and len(value) <= 5:
        try:
            if isinstance(value, dict):
                all_primitive = all(
                    isinstance(k, (str, int, float, bool, type(None)))
                    and isinstance(v, (str, int, float, bool, type(None)))
                    for k, v in value.items()
                )
            else:
                all_primitive = all(
                    isinstance(x, (str, int, float, bool, type(None))) for x in value
                )
            if all_primitive:
                return value
        except TypeError:
            pass

    # Convert classes to _ClassRef
    if isinstance(value, type):
        return _ClassRef(value.__name__)

    # Convert instances of custom classes to _InstanceRef
    # (not built-in types like list, dict, str, etc.)
    value_type = type(value)
    if value_type.__module__ not in ("builtins", "__builtin__"):
        return _InstanceRef(value_type.__name__)

    # Fallback: return as-is (will use repr in formatting)
    return value


def extract_type_info(
    obj: type,
    *,
    _skip_protocol: bool = False,
    _include_hidden: bool = False,
) -> TypeInfo:
    """Extract TypeInfo from any class-like type.

    Resolution order:
    1. Registry extractor (for third-party types)
    2. __type_info__ protocol (for classes that implement it)
    3. Automatic extraction

    Fields marked hidden (Annotated[T, hidden]) are excluded
    unless _include_hidden=True.

    Args:
        obj: A class (Pydantic model, dataclass, NamedTuple, TypedDict, or plain class)
        _skip_protocol: Internal flag to skip protocol check (prevents recursion
            when called from within __type_info__ implementations)
        _include_hidden: If True, return all fields including hidden ones.
            Used by _pformat when building instance docs with instance-level spec() overrides.

    Returns:
        TypeInfo with unified field/method information
    """
    from nooa.agentdoc._visibility import is_hidden_field

    if not _skip_protocol:
        # 1. Check extractor registry first (for third-party types)
        from nooa.agentdoc.registry import get_type_info_extractor

        extractor = get_type_info_extractor(obj)
        if extractor:
            result = extractor(obj)
            # Extractor returns TypeInfo for types, or (TypeInfo, values) tuple for instances
            if isinstance(result, tuple):
                result = result[0]
            elif not isinstance(result, TypeInfo):
                result = None
            if result is not None:
                fields = (
                    result.fields
                    if _include_hidden
                    else [f for f in result.fields if not is_hidden_field(obj, f.name)]
                )
                result = TypeInfo(
                    name=result.name,
                    base=result.base,
                    fields=fields,
                    methods=result.methods,
                    docstring=result.docstring,
                )
                return result

        # 2. Check for custom __type_info__ protocol
        if isinstance(obj, SupportsTypeInfo):
            result = obj.__type_info__()
            fields = (
                result.fields
                if _include_hidden
                else [f for f in result.fields if not is_hidden_field(obj, f.name)]
            )
            result = TypeInfo(
                name=result.name,
                base=result.base,
                fields=fields,
                methods=result.methods,
                docstring=result.docstring,
            )
            return result

    # 3. Automatic extraction
    name = obj.__name__
    base = _detect_base(obj)
    fields = _extract_fields(obj, base)
    if not _include_hidden:
        fields = [f for f in fields if not is_hidden_field(obj, f.name)]
    methods = _extract_methods(obj)
    docstring = _extract_docstring(obj)

    return TypeInfo(
        name=name,
        base=base,
        fields=fields,
        methods=methods,
        docstring=docstring,
    )


def extract_callable_info(obj: Any) -> CallableInfo:
    """Extract CallableInfo from a function or method.

    Args:
        obj: A function or method

    Returns:
        CallableInfo with signature, docstring, and metadata
    """
    # Check for custom __callable_info__ protocol first
    if isinstance(obj, SupportsCallableInfo):
        return obj.__callable_info__()

    # Use __qualname__ to get the fully qualified name (includes class for methods)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "unknown")
    is_async = inspect.iscoroutinefunction(obj)
    docstring = inspect.getdoc(obj)
    # A classmethod bound method has obj.__self__ == the class (a type object)
    is_classmethod = inspect.ismethod(obj) and isinstance(getattr(obj, "__self__", None), type)

    try:
        # For bound methods (instance and class), use __func__ so that inspect.signature()
        # includes the first parameter (self / cls) in the rendered signature.
        sig_obj = obj
        if inspect.ismethod(obj) and hasattr(obj, "__func__"):
            sig_obj = obj.__func__

        sig = inspect.signature(sig_obj)

        # Resolve annotations once: handles `from __future__ import annotations` (PEP 563)
        # and preserves Annotated[T, metadata] for description extraction.
        # For callable instances (not plain functions/methods), use __call__ directly
        # because get_type_hints on the instance returns class-level field annotations.
        resolved = None
        if inspect.isfunction(sig_obj) or inspect.ismethod(sig_obj):
            hint_targets: tuple[Any, ...] = (sig_obj,)
        else:
            call_method = getattr(type(sig_obj), "__call__", None)  # noqa: B004
            hint_targets = (call_method,) if call_method is not None else (sig_obj,)
        for hint_target in hint_targets:
            try:
                resolved = typing.get_type_hints(hint_target, include_extras=True)
                break
            except (NameError, AttributeError, TypeError):
                continue

        signature_str = _format_signature_params(sig, resolved_hints=resolved)
        return_type = _format_return_type(sig, resolved_hints=resolved)

        # Augment docstring with parameter and return descriptions from Annotated
        docstring = _augment_docstring_with_annotated_descriptions(
            obj, sig, docstring, resolved_hints=resolved
        )
    except (ValueError, TypeError):
        # Fallback for built-ins or C functions
        signature_str = "(...)"
        return_type = "Any"

    return CallableInfo(
        name=name,
        signature=signature_str,
        return_type=return_type,
        docstring=docstring,
        is_async=is_async,
        is_classmethod=is_classmethod,
    )


def extract_module_info(obj: Any) -> ModuleInfo:
    """Extract ModuleInfo from a module.

    Uses __all__ if present to determine the exported public API; otherwise
    falls back to all non-underscore names. Symbols are categorised into
    classes, callables, and plain values.

    Args:
        obj: A Python module

    Returns:
        ModuleInfo with module name, docstring, and categorised exports
    """
    name = getattr(obj, "__name__", "module")
    docstring = inspect.getdoc(obj)

    all_names = getattr(obj, "__all__", None)
    # Preserve __all__ order so the public API appears in logical grouping
    candidate_names = (
        list(all_names)
        if all_names is not None
        else sorted(n for n in dir(obj) if not n.startswith("_"))
    )

    functions: list = []
    classes: list = []
    values: list = []
    submodules_from_attrs: list[tuple[str, str | None]] = []
    module_attr_names: set[str] = set()  # sym names that are modules (excluded from ordered_names)
    resolved_names: set[str] = set()  # names that were successfully resolved (not AttributeError)

    for sym_name in candidate_names:
        try:
            attr = getattr(obj, sym_name)
        except AttributeError:
            continue

        resolved_names.add(sym_name)

        if inspect.ismodule(attr):
            # Module-typed attributes render as submodule references, not raw repr
            module_attr_names.add(sym_name)
            qualified = attr.__name__
            raw_doc = inspect.getdoc(attr) or ""
            first_line: str | None = raw_doc.split("\n")[0].strip() or None
            # Strip "module.name — " prefix that many module docstrings include
            if first_line and first_line.startswith(f"{qualified} — "):
                first_line = first_line[len(f"{qualified} — ") :]
            submodules_from_attrs.append((qualified, first_line))
        elif inspect.isclass(attr):
            raw_doc = inspect.getdoc(attr) or ""
            first_line = raw_doc.split("\n")[0].strip() or None
            classes.append((sym_name, first_line))
        elif callable(attr):
            info = extract_callable_info(attr)
            # Use the exported symbol name when the object has no __name__ (e.g. singletons)
            if info.name in ("unknown", ""):
                from dataclasses import replace as _dc_replace

                info = _dc_replace(info, name=sym_name)
            functions.append(info)
        else:
            values.append((sym_name, repr(attr)))

    # Discover submodules: __submodules__ list takes priority, then attr-discovered ones
    submodules: list[tuple[str, str | None]] = []
    seen_submodule_names: set[str] = set()

    submodule_names = getattr(obj, "__submodules__", None)
    if (
        submodule_names
        and not isinstance(submodule_names, str)
        and hasattr(submodule_names, "__iter__")
    ):
        import importlib

        for sub_name in submodule_names:
            qualified = f"{name}.{sub_name}"
            seen_submodule_names.add(qualified)
            try:
                sub_mod = importlib.import_module(qualified)
                raw_sub_doc = inspect.getdoc(sub_mod) or ""
                first_line: str | None = raw_sub_doc.split("\n")[0].strip() or None
                # Strip "modulename — " prefix that many module docstrings include
                if first_line and first_line.startswith(f"{qualified} — "):
                    first_line = first_line[len(f"{qualified} — ") :]
                submodules.append((qualified, first_line))
            except ImportError:
                submodules.append((qualified, None))

    # Add attr-discovered submodules not already covered by __submodules__
    for qualified, first_line in submodules_from_attrs:
        if qualified not in seen_submodule_names:
            submodules.append((qualified, first_line))
            seen_submodule_names.add(qualified)

    return ModuleInfo(
        name=name,
        docstring=docstring,
        functions=functions,
        classes=classes,
        values=values,
        ordered_names=[
            n for n in candidate_names if n in resolved_names and n not in module_attr_names
        ],
        submodules=submodules,
    )


def _format_signature_params(
    sig: inspect.Signature,
    resolved_hints: dict[str, Any] | None = None,
) -> str:
    """Format signature parameters only (without return type).

    Args:
        sig: Signature object from inspect.signature()
        resolved_hints: Optional dict from get_type_hints(include_extras=True), used to
            resolve string annotations produced by ``from __future__ import annotations``.

    Returns:
        Formatted parameters string like "(self, x: int, y: str = 'default')"
    """
    params = []
    prev_kind = None
    had_var_positional = False

    for param_name, param in sig.parameters.items():
        kind = param.kind

        # Insert `/` separator after the last POSITIONAL_ONLY param
        if (
            prev_kind == inspect.Parameter.POSITIONAL_ONLY
            and kind != inspect.Parameter.POSITIONAL_ONLY
        ):
            params.append("/")

        # Insert `*` separator before first KEYWORD_ONLY when no *args present
        if (
            kind == inspect.Parameter.KEYWORD_ONLY
            and not had_var_positional
            and (prev_kind != inspect.Parameter.KEYWORD_ONLY)
        ):
            params.append("*")

        if kind == inspect.Parameter.VAR_POSITIONAL:
            had_var_positional = True
        prev_kind = kind

        # Build param string, adding * or ** prefix as needed
        prefix = ""
        if kind == inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
        elif kind == inspect.Parameter.VAR_KEYWORD:
            prefix = "**"

        parts = [prefix + param_name]

        # Skip annotations on self / cls
        if param_name in ("self", "cls"):
            params.append("".join(parts))
            continue

        # Prefer resolved hint (handles PEP 563 string annotations and Annotated stripping)
        hint = resolved_hints.get(param_name) if resolved_hints is not None else None
        if hint is None and param.annotation is not inspect.Parameter.empty:
            hint = param.annotation

        if hint is not None:
            type_str = format_type(hint)
            parts.append(f": {type_str}")

        # Add default if present (VAR_POSITIONAL / VAR_KEYWORD never have defaults)
        if param.default is not inspect.Parameter.empty:
            # Enum members: render as ClassName.MEMBER_NAME regardless of __repr__
            if isinstance(param.default, enum.Enum):
                default_repr = f"{type(param.default).__name__}.{param.default.name}"
            # Functions and lambdas: render by __name__ (e.g. 'my_validator' or '<lambda>')
            elif isinstance(param.default, types.FunctionType):
                default_repr = param.default.__name__
            else:
                default_repr = repr(param.default)
            # Sentinel / opaque objects (repr starts with "<") — show as "..."
            # Exception: '<lambda>' is intentionally kept as-is (handled above)
            if default_repr.startswith("<") and default_repr != "<lambda>":
                default_repr = "..."
            elif len(default_repr) > 30:
                default_repr = default_repr[:27] + "..."
            parts.append(f" = {default_repr}")

        params.append("".join(parts))

    return f"({', '.join(params)})"


def _format_return_type(
    sig: inspect.Signature,
    resolved_hints: dict[str, Any] | None = None,
) -> str:
    """Format the return type from a signature.

    Args:
        sig: Signature object from inspect.signature()
        resolved_hints: Optional dict from get_type_hints(include_extras=True).

    Returns:
        Return type as string, or "" if not annotated
    """
    hint = resolved_hints.get("return") if resolved_hints is not None else None
    if hint is None:
        if sig.return_annotation is inspect.Signature.empty:
            return ""
        hint = sig.return_annotation
    return format_type(hint)


def _detect_base(obj: type) -> str | None:
    """Detect the base type indicator."""
    # Pydantic
    if hasattr(obj, "model_fields"):
        return "BaseModel"

    # dataclass
    if dataclasses.is_dataclass(obj):
        return "@dataclass"

    # NamedTuple
    if (
        hasattr(obj, "_fields")
        and isinstance(getattr(obj, "_fields", None), tuple)
        and issubclass(obj, tuple)
    ):
        return "NamedTuple"

    # TypedDict
    if hasattr(obj, "__required_keys__") or hasattr(obj, "__optional_keys__"):
        return "TypedDict"

    # attrs (@attr.s, @define, @attrs)
    if hasattr(obj, "__attrs_attrs__"):
        return "@attrs"

    # Enum
    try:
        if isinstance(obj, type) and issubclass(obj, enum.Enum):
            return "Enum"
    except TypeError:
        pass

    return None


def _extract_fields(obj: type, base: str | None) -> list[FieldInfo]:
    """Extract fields based on the type kind."""
    if base == "BaseModel":
        return _extract_pydantic_fields(obj)
    if base == "@dataclass":
        return _extract_dataclass_fields(obj)
    if base == "NamedTuple":
        return _extract_namedtuple_fields(obj)
    if base == "TypedDict":
        return _extract_typeddict_fields(obj)
    if base == "@attrs":
        return _extract_attrs_fields(obj)
    if base == "Enum":
        if not issubclass(obj, enum.Enum):
            raise TypeError(f"Expected Enum subclass, got {obj}")
        return _extract_enum_members(obj)
    return _extract_plain_class_fields(obj)


def _extract_pydantic_fields(obj: type) -> list[FieldInfo]:
    """Extract fields from Pydantic model."""
    fields = []
    for name, field_info in obj.model_fields.items():
        type_str = format_type(field_info.annotation)

        # Get default
        try:
            from pydantic_core import PydanticUndefined

            has_default = field_info.default is not PydanticUndefined
        except ImportError:
            # pydantic_core is always bundled with Pydantic v2 in normal installations.
            # This branch is only reached in stripped/unusual environments.
            # Detect "no default" using PydanticUndefinedType from pydantic itself,
            # which correctly distinguishes Field(default=None) from no default.
            try:
                from pydantic.fields import PydanticUndefinedType  # type: ignore[attr-defined]

                has_default = not isinstance(field_info.default, PydanticUndefinedType)
            except ImportError:
                has_default = (
                    getattr(field_info, "default_factory", None) is not None
                    or field_info.default is not None
                )

        default = field_info.default if has_default else REQUIRED

        # Get repr parameter (default True if not specified)
        repr_value = field_info.repr if field_info.repr is not None else True

        # Get description - combine description with constraints
        description_parts = []

        # Start with the field description
        field_description = getattr(field_info, "description", None)
        if field_description:
            description_parts.append(field_description)

        # Extract validation constraints from metadata (Pydantic v2 style)
        constraints = []
        metadata = getattr(field_info, "metadata", [])

        for constraint in metadata:
            constraint_type = type(constraint).__name__

            # Numeric constraints
            if constraint_type == "Ge":
                constraints.append(f"≥{constraint.ge}")
            elif constraint_type == "Gt":
                constraints.append(f">{constraint.gt}")
            elif constraint_type == "Le":
                constraints.append(f"≤{constraint.le}")
            elif constraint_type == "Lt":
                constraints.append(f"<{constraint.lt}")

            # String constraints
            elif constraint_type == "MinLen":
                constraints.append(f"min_len={constraint.min_length}")
            elif constraint_type == "MaxLen":
                constraints.append(f"max_len={constraint.max_length}")
            elif hasattr(constraint, "pattern") and constraint.pattern is not None:
                constraints.append(f"pattern={constraint.pattern!r}")

        # Add constraints to description
        if constraints:
            constraint_str = ", ".join(constraints)
            description_parts.append(f"[{constraint_str}]")

        description = " ".join(description_parts) if description_parts else None

        fields.append(
            FieldInfo(
                name=name, type=type_str, default=default, description=description, repr=repr_value
            )
        )
    return fields


def _extract_dataclass_fields(obj: type) -> list[FieldInfo]:
    """Extract fields from dataclass."""
    # Resolve full hints (preserving Annotated metadata) for description extraction
    try:
        resolved = typing.get_type_hints(obj, include_extras=True)
    except (NameError, AttributeError, TypeError):
        resolved = {}

    fields = []
    for field in dataclasses.fields(obj):
        # Use resolved hint for type display (strips Annotated wrapper via format_type)
        hint = resolved.get(field.name, field.type)
        type_str = format_type(hint)

        # Extract description from Annotated metadata
        description = _extract_annotated_description(hint) if resolved else None

        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            # Call the factory to get the actual default value
            try:
                default = field.default_factory()
            except Exception:  # factory can raise anything
                # If factory fails (e.g., needs args), show the factory name
                default = (
                    f"{getattr(field.default_factory, '__name__', repr(field.default_factory))}()"
                )
        else:
            default = REQUIRED

        fields.append(
            FieldInfo(name=field.name, type=type_str, default=default, description=description)
        )

    # Also extract @property and @cached_property descriptors (same as _extract_plain_class_fields step 4)
    from nooa.agentdoc._metadata import get_field_metadata
    from nooa.agentdoc._visibility import is_hidden_field

    seen_names = {f.name for f in fields}
    for klass in reversed(obj.__mro__):
        if klass is object:
            continue
        for name, value in klass.__dict__.items():
            if not isinstance(value, (property, functools.cached_property)):
                continue
            if name.startswith("_"):
                continue
            if name in seen_names:
                continue
            explicitly_unhidden = get_field_metadata(obj, name).get("hidden") is False
            if not explicitly_unhidden:
                if is_hidden_field(obj, name):
                    continue
                if isinstance(value, property):
                    fget = value.fget
                    if fget is not None and getattr(fget, "_agentdoc_hidden", None) is True:
                        continue
                else:
                    if (
                        getattr(value, "_agentdoc_hidden", None) is True
                        or getattr(value.func, "_agentdoc_hidden", None) is True
                    ):
                        continue
            if isinstance(value, property):
                fields.append(_extract_property_field(obj, name, value))
            else:
                fields.append(_extract_cached_property_field(obj, name, value))
            seen_names.add(name)

    return fields


def _extract_namedtuple_fields(obj: type) -> list[FieldInfo]:
    """Extract fields from NamedTuple."""
    fields = []
    field_names = obj._fields
    try:
        field_types = typing.get_type_hints(obj, include_extras=True)
    except (NameError, AttributeError, TypeError):
        field_types = inspect.get_annotations(obj)
    defaults = getattr(obj, "_field_defaults", {})

    for name in field_names:
        hint = field_types.get(name, Any)
        type_str = format_type(hint)
        description = _extract_annotated_description(hint)
        default = defaults.get(name, REQUIRED)
        fields.append(FieldInfo(name=name, type=type_str, default=default, description=description))
    return fields


def _extract_typeddict_fields(obj: type) -> list[FieldInfo]:
    """Extract fields from TypedDict.

    For optional fields (total=False), marks them with description="optional"
    rather than showing a default value (which would be semantically incorrect).
    """
    try:
        annotations = typing.get_type_hints(obj, include_extras=True)
    except (NameError, AttributeError, TypeError):
        annotations = inspect.get_annotations(obj)
    required = getattr(obj, "__required_keys__", set())
    fields = []
    for name, type_hint in annotations.items():
        type_str = format_type(type_hint)
        # For TypedDict, optional means "key can be omitted", not "value can be None"
        # So we don't show a default value, but mark it as optional in the description.
        # If the type hint carries an Annotated description, prefer that over "optional".
        is_optional = name not in required
        annotated_desc = _extract_annotated_description(type_hint)
        if annotated_desc:
            description = annotated_desc
        elif is_optional:
            description = "optional"
        else:
            description = None
        fields.append(
            FieldInfo(name=name, type=type_str, default=REQUIRED, description=description)
        )
    return fields


def _extract_attrs_fields(obj: type) -> list[FieldInfo]:
    """Extract fields from attrs class."""
    fields = []
    try:
        import attr

        for attrib in attr.fields(obj):
            type_str = format_type(attrib.type) if attrib.type is not None else "Any"

            # Get default
            if attrib.default is attr.NOTHING:
                default = REQUIRED
            elif hasattr(attrib.default, "factory") and hasattr(attrib.default, "takes_self"):
                default = attrib.default  # Keep factory reference (attrs.Factory instance)
            else:
                default = attrib.default

            # Get description from metadata if available
            description = attrib.metadata.get("description") if attrib.metadata else None

            fields.append(
                FieldInfo(name=attrib.name, type=type_str, default=default, description=description)
            )
    except ImportError:
        # attrs not installed, fall back to annotations
        return _extract_plain_class_fields(obj)
    return fields


def _extract_enum_members(obj: type[enum.Enum]) -> list[FieldInfo]:
    """Extract members from Enum class."""
    fields = []
    for member in obj:
        # For enums: name is member name, type is the value type, default is the value
        value = member.value
        type_str = type(value).__name__
        fields.append(FieldInfo(name=member.name, type=type_str, default=value, description=None))
    return fields


def _augment_docstring_with_annotated_descriptions(
    func: Any,
    sig: inspect.Signature,
    original_docstring: str | None,
    resolved_hints: dict[str, Any] | None = None,
) -> str | None:
    """Augment a function's docstring with Args/Returns from Annotated metadata.

    Extracts parameter and return type descriptions from Annotated type hints
    and adds them as Args/Returns sections if they don't already exist.

    Args:
        func: The function/method object
        sig: The function's signature
        original_docstring: The original docstring (may be None)
        resolved_hints: Pre-resolved type hints (from get_type_hints); resolved
            here if not provided.

    Returns:
        Augmented docstring, or original if no descriptions found
    """
    if resolved_hints is None:
        try:
            resolved = typing.get_type_hints(func, include_extras=True)
        except (NameError, AttributeError, TypeError):
            resolved = None
    else:
        resolved = resolved_hints

    # Extract parameter descriptions from Annotated
    param_descriptions = {}
    for param_name, param in sig.parameters.items():
        # Prefer resolved hint; fall back to raw annotation from signature
        hint = resolved.get(param_name) if resolved is not None else None
        if hint is None and param.annotation is not inspect.Parameter.empty:
            hint = param.annotation
        if hint is not None:
            desc = _extract_annotated_description(hint)
            if desc:
                param_descriptions[param_name] = desc

    # Extract return type description from Annotated
    return_description = None
    ret_hint = resolved.get("return") if resolved is not None else None
    if ret_hint is None and sig.return_annotation is not inspect.Signature.empty:
        ret_hint = sig.return_annotation
    if ret_hint is not None:
        return_description = _extract_annotated_description(ret_hint)

    # If no descriptions found, return original
    if not param_descriptions and not return_description:
        return original_docstring

    # Check if docstring already has Args or Returns sections
    docstring = original_docstring or ""
    has_args_section = _has_docstring_section(docstring, "Args")
    has_returns_section = _has_docstring_section(docstring, "Returns")

    # Build augmented docstring
    parts = []
    if docstring:
        parts.append(docstring.rstrip())

    # Add Args section if we have descriptions and no existing section
    if param_descriptions and not has_args_section:
        if parts:
            parts.append("")  # Blank line before Args
        parts.append("Args:")
        for param_name, desc in param_descriptions.items():
            parts.append(f"    {param_name}: {desc}")

    # Add Returns section if we have description and no existing section
    if return_description and not has_returns_section:
        if parts:
            parts.append("")  # Blank line before Returns
        parts.append("Returns:")
        parts.append(f"    {return_description}")

    return "\n".join(parts) if parts else None


def _has_docstring_section(docstring: str, section_name: str) -> bool:
    """Check if a docstring already has a specific section (Args, Returns, etc.).

    Supports common docstring styles:
    - Google: "Args:" or "Returns:"
    - NumPy: "Parameters" or "Returns"
    - Sphinx: ":param" or ":return:"

    Args:
        docstring: The docstring to check
        section_name: Section name to look for ("Args" or "Returns")

    Returns:
        True if the section exists
    """
    if not docstring:
        return False

    docstring_lower = docstring.lower()

    if section_name == "Args":
        # Check for Args, Parameters (as a section header), or :param
        # "parameters" must appear as a header (followed by : or \n), not in prose
        return (
            "args:" in docstring_lower
            or "arguments:" in docstring_lower
            or "parameters:" in docstring_lower
            or "\nparameters\n" in docstring_lower  # NumPy style header
            or ":param" in docstring_lower
        )
    elif section_name == "Returns":
        # Check for Returns or :return:
        return (
            "returns:" in docstring_lower
            or ":return:" in docstring_lower
            or ":returns:" in docstring_lower
        )

    return False


def _extract_annotated_description(type_hint: Any) -> str | None:
    """Extract description from Annotated metadata.

    Handles (in priority order):
    - Annotated[T, "description"] -> returns "description"
    - Annotated[T, spec(description="...")] -> returns "..."
    - Annotated[T, Field(description="...")] -> returns "..."  (Pydantic FieldInfo)
    - Other types -> returns None

    Args:
        type_hint: Type annotation to check

    Returns:
        Description string if found, None otherwise
    """
    origin = typing.get_origin(type_hint)

    if not (hasattr(typing, "Annotated") and origin is typing.Annotated):
        return None

    args = typing.get_args(type_hint)
    if len(args) <= 1:
        return None

    # args[0] is the actual type, args[1:] are metadata
    for metadata in args[1:]:
        # Plain string annotation: Annotated[T, "description"]
        if isinstance(metadata, str):
            return metadata

        # SpecAnnotation from spec(description="..."): has .kwargs dict
        if hasattr(metadata, "kwargs") and isinstance(getattr(metadata, "kwargs", None), dict):
            desc = metadata.kwargs.get("description")
            if desc:
                return desc

        # Pydantic FieldInfo: Annotated[T, Field(description="...")]
        # Works for both plain classes and BaseModel subclasses.
        desc = getattr(metadata, "description", None)
        if isinstance(desc, str) and desc:
            return desc

    return None


def _extract_plain_class_fields(obj: type) -> list[FieldInfo]:
    """Extract fields from plain class annotations, class-level attributes, and __init__.

    This handles:
    1. Annotated class-level fields: `name: str = "default"` or `name: Annotated[str, "description"]`
    2. Non-annotated class attributes: `calculator = Calculator()`
    3. Instance attributes defined in __init__: `self.x: Annotated[int, "desc"] = 0`
    """
    fields = []
    seen_names = set()

    # 1. First extract annotated class-level fields
    # Use get_type_hints(include_extras=True) to resolve string annotations
    # (from `from __future__ import annotations`) and preserve Annotated metadata.
    # Fall back to raw __annotations__ if resolution fails (e.g. forward refs).
    try:
        resolved_hints = typing.get_type_hints(obj, include_extras=True)
    except (NameError, AttributeError, TypeError):
        resolved_hints = None

    # Collect annotation names across the full MRO (base-to-derived order) so that
    # inherited fields appear in doc() output.  We read each class's own
    # inspect.get_annotations(klass) (own annotations only, PEP 649 compatible)
    # and merge in MRO order: later (more-derived) entries override earlier ones,
    # so a child that re-declares a parent field keeps its own version.
    #
    # Sort order: shallower classes (len(mro) is smaller) first; among same-depth
    # siblings, respect forward-MRO priority (lower index = higher priority = first).
    # This ensures Base1 fields appear before Base2 fields for Child(Base1, Base2).
    all_raw_annotations: dict[str, Any] = {}
    mro_index = {c: i for i, c in enumerate(obj.__mro__)}
    for _klass in sorted(
        (c for c in obj.__mro__ if c is not object),
        key=lambda c: (len(c.__mro__), mro_index[c]),
    ):
        for name, raw_hint in inspect.get_annotations(_klass).items():
            all_raw_annotations[name] = raw_hint

    for name, _raw_hint in all_raw_annotations.items():
        type_hint = resolved_hints.get(name, _raw_hint) if resolved_hints is not None else _raw_hint
        if name.startswith("_"):
            continue

        # Format the type (this will extract actual type from Annotated)
        type_str = format_type(type_hint)

        # Extract description: Annotated metadata takes priority, then imperative spec()
        description = _extract_annotated_description(type_hint)
        if not description:
            from nooa.agentdoc._metadata import get_field_metadata

            description = get_field_metadata(obj, name).get("description")

        # Get default value without invoking descriptors, converting instances to clean markers.
        raw_default = inspect.getattr_static(obj, name, REQUIRED)
        if isinstance(raw_default, types.MemberDescriptorType):
            raw_default = REQUIRED
        # If a @property or @cached_property is defined, skip here — step 4 handles it
        if isinstance(raw_default, (property, functools.cached_property)):
            continue
        default = _convert_to_marker(raw_default)

        fields.append(FieldInfo(name=name, type=type_str, default=default, description=description))
        seen_names.add(name)

    # 2. Then extract non-annotated class-level attributes (tools, child agents, etc.).
    # Walk the full MRO (base → derived) so inherited un-annotated class attributes
    # (e.g. `shell = ShellTools()` declared on a base agent) appear in the child's
    # doc(), matching how annotated fields (step 1) and methods are collected.
    #
    # Candidate names are gathered only from the __dict__s of classes in obj.__mro__
    # (base-class attrs ordered before derived, mirroring step 1). Restricting the
    # name source to the class MRO ensures inspect.getattr_static() below resolves
    # each name within the class MRO before the metaclass, so metaclass attributes
    # are never pulled in. The effective (leaf-resolved) value is used, so the leaf
    # class wins on name collisions.
    # (mro_index was computed in step 1 above; reuse it for the same ordering.)
    candidate_names: list[str] = []
    seen_candidates: set[str] = set()  # O(1) dedup alongside the ordered list
    for _klass in sorted(
        (c for c in obj.__mro__ if c is not object),
        key=lambda c: (len(c.__mro__), mro_index[c]),
    ):
        for name in _klass.__dict__:
            if name not in seen_candidates:
                seen_candidates.add(name)
                candidate_names.append(name)

    for name in candidate_names:
        # Skip already-seen (annotated) fields
        if name in seen_names:
            continue
        # Skip private/dunder attributes
        if name.startswith("_"):
            continue
        # Resolve the effective value via the MRO (leaf wins) without invoking
        # descriptors. REQUIRED guards names that vanished between collection and lookup.
        value = inspect.getattr_static(obj, name, REQUIRED)
        if value is REQUIRED:
            continue
        # Skip methods, functions, classmethods, staticmethods, properties
        if callable(value) and not isinstance(value, type):
            continue
        if isinstance(value, (classmethod, staticmethod, property, functools.cached_property)):
            continue
        # Skip descriptors (like method wrappers)
        if hasattr(value, "__get__") and not isinstance(value, type):
            continue

        # Determine type and default representation
        if isinstance(value, type):
            # Class attribute (e.g., WorkerAgent = WorkerAgent)
            # Use "type[ClassName]" as the proper type annotation
            type_str = f"type[{value.__name__}]"
            # Store a _ClassRef marker so formatting knows to show just the name
            default = _ClassRef(value.__name__)
        else:
            # Instance attribute (e.g., calculator = Calculator())
            type_str = type(value).__name__
            # Store an _InstanceRef marker so formatting shows ClassName()
            default = _InstanceRef(type_str)

        fields.append(FieldInfo(name=name, type=type_str, default=default, description=None))
        seen_names.add(name)

    # 3. Extract instance attributes from __init__ (self.x: Type = value)
    init_fields = _extract_init_fields(obj)
    for field in init_fields:
        if field.name not in seen_names:
            fields.append(field)
            seen_names.add(field.name)

    # 4. Extract @property and @cached_property descriptors across the MRO
    from nooa.agentdoc._metadata import get_field_metadata
    from nooa.agentdoc._visibility import is_hidden_field

    for klass in reversed(obj.__mro__):
        if klass is object:
            continue
        for name, value in klass.__dict__.items():
            if not isinstance(value, (property, functools.cached_property)):
                continue
            if name.startswith("_"):
                continue
            if name in seen_names:
                continue
            # spec(hidden=False) explicitly unhides — overrides @hidden markers.
            # spec(hidden=True) and annotation-based hiding are checked via is_hidden_field.
            explicitly_unhidden = get_field_metadata(obj, name).get("hidden") is False
            if not explicitly_unhidden:
                if is_hidden_field(obj, name):
                    continue
                if isinstance(value, property):
                    fget = value.fget
                    if fget is not None and getattr(fget, "_agentdoc_hidden", None) is True:
                        continue
                else:
                    # functools.cached_property: @hidden may be on the wrapper or on .func
                    if (
                        getattr(value, "_agentdoc_hidden", None) is True
                        or getattr(value.func, "_agentdoc_hidden", None) is True
                    ):
                        continue
            if isinstance(value, property):
                fields.append(_extract_property_field(obj, name, value))
            else:
                fields.append(_extract_cached_property_field(obj, name, value))
            seen_names.add(name)

    return fields


def _extract_property_field(owner: type, name: str, prop: property) -> FieldInfo:
    """Build a FieldInfo for a single @property descriptor."""
    from nooa.agentdoc._metadata import get_field_metadata

    fget = prop.fget
    # Return type from getter annotation
    if fget is not None:
        hints: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            hints = typing.get_type_hints(fget)
        ret = hints.get("return", Any)
        type_str = format_type(ret)
    else:
        type_str = "Any"

    # Description: imperative spec() > getter docstring
    description = get_field_metadata(owner, name).get("description")
    if not description and fget is not None:
        raw_doc = inspect.getdoc(fget)
        if raw_doc:
            description = raw_doc.splitlines()[0]

    return FieldInfo(name=name, type=type_str, default=REQUIRED, description=description)


def _extract_cached_property_field(
    owner: type, name: str, cp: functools.cached_property
) -> FieldInfo:  # type: ignore[type-arg]
    """Build a FieldInfo for a single @cached_property descriptor."""
    from nooa.agentdoc._metadata import get_field_metadata

    func = cp.func
    hints: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        hints = typing.get_type_hints(func)
    ret = hints.get("return", Any)
    type_str = format_type(ret)

    description = get_field_metadata(owner, name).get("description")
    if not description:
        raw_doc = inspect.getdoc(func)
        if raw_doc:
            description = raw_doc.splitlines()[0]

    return FieldInfo(name=name, type=type_str, default=REQUIRED, description=description)


def _extract_init_fields(obj: type) -> list[FieldInfo]:
    """Extract instance attributes from __init__ methods across the MRO.

    Walks the class hierarchy (child first) to collect instance attributes
    from all __init__ methods, just like Python inheritance makes parent
    attributes available on child instances.

    Child class fields take precedence — if both child and parent define
    the same field name, the child's version wins.

    Parses the AST of each __init__ to find assignments like:
        self.x: int = 0                              # annotated
        self.name: Annotated[str, "description"] = "default"  # annotated with metadata
        self.result = 0                              # non-annotated (type inferred)

    Args:
        obj: The class to extract fields from

    Returns:
        List of FieldInfo for instance attributes (annotated and non-annotated)
    """
    all_fields: list[FieldInfo] = []
    seen_names: set[str] = set()

    # Walk MRO: child class first, then parents
    for klass in obj.__mro__:
        if klass is object:
            continue

        # Only look at __init__ defined directly on this class
        init_method = klass.__dict__.get("__init__")
        if init_method is None:
            continue

        # Extract fields from this class's __init__, resolving types
        # in the context of the class that defines them
        fields = _extract_init_fields_from_method(init_method, klass)
        for field in fields:
            if field.name not in seen_names:
                all_fields.append(field)
                seen_names.add(field.name)

    return all_fields


def _extract_init_fields_from_method(init_method: Any, context_class: type) -> list[FieldInfo]:
    """Extract instance attributes from a single __init__ method.

    Parses the AST of the method to find self.x assignments.

    Args:
        init_method: The __init__ function object
        context_class: The class that defines this __init__ (used for
            resolving type annotations in the correct module scope)

    Returns:
        List of FieldInfo for instance attributes found in this __init__
    """
    import ast
    import linecache
    import os.path
    import textwrap

    # Guard: skip methods without retrievable Python source (C extensions,
    # builtins, or methods with stale code objects from deleted modules).
    # inspect.getsource can segfault on such methods in CPython 3.12.
    if not callable(init_method):
        return []
    code = getattr(init_method, "__code__", None)
    if code is None:
        return []
    if not code.co_filename or (
        not os.path.isfile(code.co_filename) and not linecache.getlines(code.co_filename)
    ):
        return []

    # Get source code
    try:
        source = inspect.getsource(init_method)
    except (OSError, TypeError, SyntaxError):
        return []

    # Dedent to handle indented class definitions (preserves relative indentation)
    source = textwrap.dedent(source)

    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    fields = []
    seen_names: set[str] = set()

    # Extract __init__ parameter names to detect "from parameter" defaults
    init_params: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for arg in node.args.args:
                if arg.arg != "self":
                    init_params.add(arg.arg)
            break

    # Walk the AST looking for assignments to self.x
    for node in ast.walk(tree):
        # Handle annotated assignments: self.x: int = 0
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                field_name = target.attr

                # Skip private attributes
                if field_name.startswith("_"):
                    continue

                # Skip if already seen
                if field_name in seen_names:
                    continue
                seen_names.add(field_name)

                # Get the type annotation as source code
                type_source = ast.unparse(node.annotation)

                # Try to evaluate the type annotation in the class's context
                type_hint = _eval_type_annotation(type_source, context_class)

                if type_hint is not None:
                    type_str = format_type(type_hint)
                    description = _extract_annotated_description(type_hint)
                else:
                    # Fallback to using the source as-is
                    type_str = type_source
                    description = None

                # Get default value representation
                default = (
                    _ast_to_default(node.value, init_params) if node.value is not None else REQUIRED
                )

                fields.append(
                    FieldInfo(
                        name=field_name,
                        type=type_str,
                        default=default,
                        description=description,
                    )
                )

        # Handle non-annotated assignments: self.x = 0
        elif isinstance(node, ast.Assign):
            # Check each target (handles cases like self.a = self.b = 0)
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    field_name = target.attr

                    # Skip private attributes
                    if field_name.startswith("_"):
                        continue

                    # Skip if already seen (annotated version takes precedence)
                    if field_name in seen_names:
                        continue
                    seen_names.add(field_name)

                    # Infer type from the assigned value
                    type_str = _infer_type_from_ast(node.value)

                    # Get default value representation
                    default = _ast_to_default(node.value, init_params)

                    fields.append(
                        FieldInfo(
                            name=field_name,
                            type=type_str,
                            default=default,
                            description=None,
                        )
                    )

    return fields


def _infer_type_from_ast(node: Any) -> str:
    """Infer the type from an AST value node.

    Args:
        node: AST node representing a value

    Returns:
        Type name as string (e.g., "int", "str", "list", "Any" as fallback)
    """
    import ast

    if node is None:
        return "Any"

    # Handle constants (int, float, str, bool, None, bytes)
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "None"
        return type(node.value).__name__

    # Handle empty containers
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"

    # Handle calls like ClassName() or dict()
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return "Any"

    # Handle Name references (variable or class)
    if isinstance(node, ast.Name):
        return node.id

    # Fallback
    return "Any"


def _eval_type_annotation(type_source: str, cls: type) -> Any:
    """Safely evaluate a type annotation string in the class's context.

    Args:
        type_source: Type annotation as source code string
        cls: The class whose context to use

    Returns:
        Evaluated type hint, or None if evaluation fails
    """
    # Build evaluation context from the class's module
    eval_context: dict[str, Any] = {}

    # Add typing module
    import typing as typing_module

    eval_context.update(vars(typing_module))

    # Add the class's module globals
    module = inspect.getmodule(cls)
    if module:
        eval_context.update(vars(module))

    try:
        return eval(type_source, eval_context)  # noqa: S307
    except (NameError, AttributeError, TypeError, SyntaxError, ValueError):
        return None


def _ast_to_default(node: Any, init_params: set[str] | None = None) -> Any:
    """Convert an AST node to a default value representation.

    For simple literals, returns the actual value.
    For complex expressions, returns a string representation.

    Args:
        node: AST node representing the default value
        init_params: Set of __init__ parameter names (to mark as "from parameter")
    """
    import ast

    init_params = init_params or set()

    # Handle simple literals (ast.Constant covers all constants in Python 3.8+)
    if isinstance(node, ast.Constant):
        return node.value

    # Handle Name (like a variable reference or class name)
    if isinstance(node, ast.Name):
        # If this references an __init__ parameter, omit the default
        # (the value comes from the caller, not a fixed default)
        if node.id in init_params:
            return REQUIRED
        return _ClassRef(node.id)

    # Handle Call (like ClassName() or dict())
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return _InstanceRef(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Something like module.ClassName()
            return _InstanceRef(node.func.attr)

    # Handle empty containers
    if isinstance(node, ast.List) and len(node.elts) == 0:
        return []
    if isinstance(node, ast.Dict) and len(node.keys) == 0:
        return {}
    if isinstance(node, ast.Set) and len(node.elts) == 0:
        return set()

    # For complex expressions, return the source representation
    try:
        return ast.unparse(node)
    except ValueError:
        return REQUIRED


# Framework methods to hide from type extraction
_PYDANTIC_METHODS = {
    "construct",
    "copy",
    "dict",
    "from_orm",
    "json",
    "model_construct",
    "model_copy",
    "model_dump",
    "model_dump_json",
    "model_json_schema",
    "model_parametrized_name",
    "model_post_init",
    "model_rebuild",
    "model_validate",
    "model_validate_json",
    "model_validate_strings",
    "parse_file",
    "parse_obj",
    "parse_raw",
    "schema",
    "schema_json",
    "update_forward_refs",
    "validate",
}


def _extract_methods(obj: type) -> list[CallableInfo]:
    """Extract public methods as CallableInfo objects.

    Filters out framework methods (Pydantic BaseModel methods, etc.)
    and methods marked with @hidden.
    """
    from nooa.agentdoc._visibility import is_hidden_method

    methods = []
    seen_names = set()

    # Walk the MRO dictionaries directly instead of inspect.getmembers().
    # inspect.getmembers() calls getattr() for every name and can execute arbitrary
    # descriptors while formatting an object.
    for klass in obj.__mro__:
        for name, raw in vars(klass).items():
            value = raw
            if isinstance(raw, (classmethod, staticmethod)):
                value = raw.__func__
            if not inspect.isfunction(value) and not inspect.ismethod(value):
                continue

            explicitly_shown = (
                getattr(value, "_agentdoc_hidden", None) is False
                or getattr(raw, "_agentdoc_hidden", None) is False
            )
            if name.startswith("_") and not explicitly_shown:
                continue
            if name in seen_names:
                continue
            # Skip Pydantic BaseModel methods
            if name in _PYDANTIC_METHODS:
                continue
            if is_hidden_method(value) or is_hidden_method(raw):
                continue
            seen_names.add(name)
            info = extract_callable_info(value)
            if isinstance(raw, classmethod):
                info = dataclasses.replace(info, is_classmethod=True)
            methods.append(info)

    # Third pass: class-level metadata (covers C-extension methods that can't have
    # _agentdoc_hidden set directly — e.g. list.__init__ stored via __objclass__)
    from nooa.agentdoc._metadata import _FIELDS_ATTR  # noqa: PLC0415

    try:
        fields_meta: dict[str, Any] = vars(obj).get(_FIELDS_ATTR) or {}
    except TypeError:
        fields_meta = {}
    for name, meta in fields_meta.items():
        if meta.get("hidden") is not False:
            continue
        if name in seen_names:
            continue
        value = getattr(obj, name, None)
        if value is None or not callable(value):
            continue
        seen_names.add(name)
        methods.append(extract_callable_info(value))

    # Sort by source-definition order: walk MRO (derived → base) and record
    # first occurrence of each __qualname__.  Keying by __qualname__ matches
    # CallableInfo.name directly (which also uses __qualname__), avoiding the
    # need to strip the class prefix when looking up.
    source_order: dict[str, int] = {}
    idx = 0
    for klass in obj.__mro__:
        for name, value in klass.__dict__.items():
            try:
                qualname = getattr(value, "__qualname__", None) or f"{klass.__qualname__}.{name}"
            except Exception:
                qualname = f"{klass.__qualname__}.{name}"
            if qualname not in source_order:
                source_order[qualname] = idx
                idx += 1

    return sorted(methods, key=lambda m: source_order.get(m.name, idx))


def _extract_docstring(obj: type) -> str | None:
    """Extract docstring, filtering framework noise.

    Only returns docstrings defined directly on the class,
    not inherited from parent classes.
    """
    # Check if the class has its own __doc__ (not inherited from parent)
    own_doc = obj.__dict__.get("__doc__")

    if not own_doc:
        return None

    # Ensure it's a string (some built-ins have non-string __doc__ like member_descriptor)
    if not isinstance(own_doc, str):
        return None

    # Clean up the docstring
    docstring = inspect.cleandoc(own_doc)

    # Filter auto-generated dataclass docstring
    if dataclasses.is_dataclass(obj) and docstring.startswith(f"{obj.__name__}("):
        return None

    return docstring


def format_type(type_hint: Any) -> str:
    """Format a type hint as a readable string.

    This is the single source of truth for type formatting across:
    - prefill.py (return type display)
    - codeact.py (type annotations)
    - core.py (doc output)
    - _pformat.py (type display)

    Special handling:
    - Annotated[T, metadata...]: Extracts just the type T, ignoring metadata
    """
    if type_hint is None:
        return "None"
    if type_hint is type(None):
        return "None"
    if type_hint is ...:
        return "..."
    if isinstance(type_hint, str):
        return type_hint

    # Handle generics, unions, etc FIRST (before __name__ check)
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    if origin is not None:
        # Special handling for Annotated - extract just the actual type
        # Annotated[T, metadata...] has origin=Annotated and args=(T, metadata...)
        if hasattr(typing, "Annotated") and origin is typing.Annotated:
            if args:
                # First arg is the actual type, rest is metadata
                return format_type(args[0])
            # Fallback if no args (shouldn't happen)
            return "Any"

        # Special handling for UnionType (Python 3.10+ X | Y syntax)
        # This includes types.UnionType from runtime unions
        origin_str = str(origin)
        if "UnionType" in origin_str or origin_str == "typing.Union":
            if args:
                # Format as X | Y | Z instead of Union[X, Y, Z]
                formatted_args = [format_type(a) for a in args]
                return " | ".join(formatted_args)
            return "Any"

        # Get a clean name for the origin
        if hasattr(origin, "__name__"):
            origin_name = origin.__name__
        else:
            # For typing module types like Union
            origin_name = origin_str.split(".")[-1] if "." in origin_str else origin_str

        if args:
            # Special handling for Callable: args are ([param_types], return_type)
            if origin_name == "Callable" and len(args) == 2 and isinstance(args[0], list):
                param_types = args[0]
                return_type = args[1]
                params_str = ", ".join(format_type(p) for p in param_types)
                return_str = format_type(return_type)
                return f"Callable[[{params_str}], {return_str}]"

            args_str = ", ".join(format_type(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    # Simple types with __name__
    if hasattr(type_hint, "__name__"):
        return type_hint.__name__

    # Fallback: clean up typing module prefixes
    result = str(type_hint)
    return result.replace("typing.", "")
