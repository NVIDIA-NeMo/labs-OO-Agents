# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extraction protocols for custom documentation.

Objects can implement these protocols to provide custom extraction
that takes precedence over automatic introspection. The extracted
Info objects are then formatted consistently by pformat().

## Extraction Protocols

- `__type_info__()` - Override type extraction (classmethod)
- `__callable_info__()` - Override callable extraction (function attribute)
- `__instance_values__()` - Override instance value extraction (instance method)

These protocols return structured data (Info types), not formatted strings.
Formatting is always done by pformat() for consistency.
"""

from typing import Any, Protocol, runtime_checkable

from nooa.agentdoc._info import CallableInfo, TypeInfo


@runtime_checkable
class SupportsTypeInfo(Protocol):
    """Protocol for types that provide custom type info extraction.

    Implement this as a classmethod to control how your class is represented
    when introspected by agentdoc.

    Example:
        class MyClass:
            @classmethod
            def __type_info__(cls) -> TypeInfo:
                return TypeInfo(
                    name="MyClass",
                    base="CustomFramework",
                    fields=[FieldInfo("id", "int", ..., "Unique identifier")],
                    methods=[...],
                    docstring="A custom class."
                )
    """

    @classmethod
    def __type_info__(cls) -> TypeInfo:
        """Return TypeInfo for this class.

        Returns:
            TypeInfo with custom fields, methods, and docstring
        """
        ...


@runtime_checkable
class SupportsCallableInfo(Protocol):
    """Protocol for callables that provide custom callable info extraction.

    Implement this as a property or attribute on a function to control
    how it's represented when introspected by agentdoc.

    Example:
        def my_function():
            ...

        my_function.__callable_info__ = lambda: CallableInfo(
            name="my_function",
            signature="(x: int) -> str",
            return_type="str",
            docstring="Custom documentation.",
            is_async=False,
        )
    """

    def __callable_info__(self) -> CallableInfo:
        """Return CallableInfo for this callable.

        Returns:
            CallableInfo with custom signature, docstring, etc.
        """
        ...


@runtime_checkable
class SupportsInstanceValues(Protocol):
    """Protocol for instances that control runtime value extraction.

    Implement this to control which current values are shown. In ``doc()``,
    fields omitted from this mapping remain part of the type-level API contract;
    use ``hidden`` or ``spec(..., hidden=True)`` to hide documented fields.
    In ``pformat()``, omitted fields are excluded from the value representation.

    Example:
        class MyClass:
            def __instance_values__(self) -> dict[str, Any]:
                return {
                    "id": self.id,
                    "status": self.status,
                }
    """

    def __instance_values__(self) -> dict[str, Any]:
        """Return current instance values for documentation and formatting.

        Returns:
            Dictionary mapping field names to their current values. In
            ``doc()``, omitted fields retain their type-level defaults; in
            ``pformat()``, omitted fields are not rendered.
        """
        ...
