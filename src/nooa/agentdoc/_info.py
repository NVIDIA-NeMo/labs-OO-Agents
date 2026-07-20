# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Info types for agentdoc introspection.

These dataclasses provide a composable, unified representation of Python
constructs (types, callables, modules) that can be formatted by pformat().
"""

from dataclasses import dataclass, field
from typing import Annotated, Any

# Sentinel for "no default" / required field
REQUIRED = ...


@dataclass
class FieldInfo:
    """Information about a class field/attribute."""

    name: Annotated[str, "Field name"]
    type: Annotated[str, "Type as string, e.g. 'str' or 'list[int]'"]
    default: Annotated[Any, "Default value; REQUIRED (...) means no default"] = REQUIRED
    description: Annotated[
        str | None, "Description from annotation, docstring, or Pydantic Field"
    ] = None
    repr: Annotated[bool, "If False, field is excluded from doc() and pformat() output"] = True


@dataclass
class CallableInfo:
    """Information about a function or method."""

    name: Annotated[str, "Function or method name"]
    signature: Annotated[str, "Formatted parameter list, e.g. '(x: int, y: str = \"hi\")'"]
    return_type: Annotated[str, "Return type as string"]
    docstring: Annotated[str | None, "Full docstring, or None if absent"]
    is_async: Annotated[bool, "True for async def functions"] = False
    is_classmethod: Annotated[bool, "True for @classmethod definitions"] = False


@dataclass
class TypeInfo:
    """Information about a class or type."""

    name: Annotated[str, "Class name"]
    base: Annotated[str | None, "Base indicator: 'BaseModel', '@dataclass', 'Enum', etc."]
    fields: Annotated[list[FieldInfo], "Class fields and attributes"]
    methods: Annotated[list[CallableInfo], "Class methods"]
    docstring: Annotated[str | None, "Class docstring"]


@dataclass
class ModuleInfo:
    """Information about a module."""

    name: Annotated[str, "Module name, e.g. 'mypackage.submodule'"]
    docstring: Annotated[str | None, "Module-level docstring"]
    functions: Annotated[list[CallableInfo], "Public callables exported by the module"]
    classes: Annotated[
        list[tuple[str, str | None]], "Public classes as (name, one-liner docstring) pairs"
    ] = field(default_factory=list)
    values: Annotated[list[tuple[str, str]], "Public non-callable values as (name, repr) pairs"] = (
        field(default_factory=list)
    )
    ordered_names: Annotated[list[str], "Symbol names in __all__ declaration order"] = field(
        default_factory=list
    )
    submodules: Annotated[
        list[tuple[str, str | None]], "Submodules as (qualified_name, one-liner docstring) pairs"
    ] = field(default_factory=list)
