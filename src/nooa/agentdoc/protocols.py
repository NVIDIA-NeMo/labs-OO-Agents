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
    """Protocol for instances that control their value extraction.

    Implement this to control which field values are shown when an instance
    is documented (e.g., to hide internal state).

    Example:
        class MyClass:
            def __instance_values__(self) -> dict[str, Any]:
                return {
                    "id": self.id,
                    "status": self.status,
                    # Hide internal fields by not including them
                }
    """

    def __instance_values__(self) -> dict[str, Any]:
        """Return instance values for documentation.

        Returns:
            Dictionary mapping field names to their current values.
            Fields not in the dict will use their default values.
        """
        ...
