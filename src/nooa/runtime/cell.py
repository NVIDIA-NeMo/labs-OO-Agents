# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing inspection of one persistent Python-cell invocation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Annotated, Any

from nooa.agentdoc import spec

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


class CellCall:
    """Safe metadata for the method invocation being executed."""

    __slots__ = ("_id", "_name", "_signature", "_return_type")

    id: Annotated[str, spec(description="Unique invocation identifier")]
    name: Annotated[str, spec(description="Invoked method name")]
    signature: Annotated[str | None, spec(description="Declared method signature")]
    return_type: Annotated[type | None, spec(description="Expected method result type")]

    def __init__(self, call: CurrentCall) -> None:
        object.__setattr__(self, "_id", call.id)
        object.__setattr__(self, "_name", call.method_name)
        object.__setattr__(self, "_signature", call.signature)
        object.__setattr__(self, "_return_type", call.return_type)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("cell.call is read-only")

    @property
    def id(self) -> str:
        """Unique invocation identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Invoked method name."""
        return self._name

    @property
    def signature(self) -> str | None:
        """Declared method signature."""
        return self._signature

    @property
    def return_type(self) -> type | None:
        """Expected method result type."""
        return self._return_type

    def __repr__(self) -> str:
        signature = self.signature or "(self, ...) -> ..."
        # Use a Unicode arrow in model-facing text so XML escaping does not
        # turn Python's ``->`` annotation marker into the distracting ``-&gt;``.
        return f"{self.name}{_without_self(signature).replace(' -> ', ' → ')}"


class CellOutputs:
    """Read-only access to non-empty results from completed cells."""

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[int, Any] = {}

    def __getitem__(self, index: int) -> Any:
        """Return an output by execution number, or from the end for a negative index."""
        if index < 0:
            try:
                return list(self._values.values())[index]
            except IndexError:
                raise IndexError(
                    f"cell.outputs index {index} out of range (have {len(self._values)} outputs)"
                ) from None
        try:
            return self._values[index]
        except KeyError:
            raise KeyError(f"No output for execution {index}") from None

    def __len__(self) -> int:
        """Return the number of recorded non-empty outputs."""
        return len(self._values)

    @property
    def last(self) -> Any:
        """Return the most recent non-empty output."""
        if not self._values:
            raise IndexError("No outputs recorded yet")
        return next(reversed(self._values.values()))

    @property
    def indices(self) -> tuple[int, ...]:
        """Return the execution numbers that have recorded outputs."""
        return tuple(self._values)

    def _record(self, execution_count: int, value: Any) -> None:
        self._values[execution_count] = value

    def __repr__(self) -> str:
        indices = self.indices
        if not indices:
            return "CellOutputs (none)"
        return f"CellOutputs ({len(indices)} available: {list(indices)})"


class Cell:
    """Read-only view of the current Python-cell invocation."""

    __slots__ = (
        "_call_view",
        "_inputs",
        "_namespace",
        "_outputs",
        "_preloaded",
        "_summary_local_types",
        "_summary_modules",
        "_suppress_implicit_strings",
    )

    call: Annotated[CellCall, spec(description="Method invocation metadata")]
    inputs: Annotated[
        Mapping[str, Any], spec(description="Bound method inputs; values are also direct locals")
    ]
    locals: Annotated[
        Mapping[str, Any],
        spec(description="Non-module names committed by completed cells, including imports"),
    ]
    modules: Annotated[
        Mapping[str, ModuleType],
        spec(description="Available module aliases, both preloaded and imported"),
    ]
    outputs: Annotated[CellOutputs, spec(description="Non-empty results from completed cells")]

    def __init__(
        self,
        call: CurrentCall,
        *,
        preloaded: Mapping[str, Any],
        suppress_implicit_strings: bool = False,
    ) -> None:
        object.__setattr__(self, "_call_view", CellCall(call))
        object.__setattr__(self, "_inputs", dict(call.bound_parameters()))
        namespace = call.execution_locals
        if namespace is None:
            namespace = call.session_locals
        object.__setattr__(self, "_namespace", namespace if namespace is not None else {})
        object.__setattr__(self, "_preloaded", dict(preloaded))
        object.__setattr__(self, "_outputs", CellOutputs())
        object.__setattr__(self, "_summary_local_types", None)
        object.__setattr__(self, "_summary_modules", None)
        object.__setattr__(self, "_suppress_implicit_strings", suppress_implicit_strings)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("cell is read-only")

    @property
    def call(self) -> CellCall:
        """Method invocation metadata."""
        return self._call_view

    @property
    def outputs(self) -> CellOutputs:
        """Non-empty results from completed cells."""
        return self._outputs

    @property
    def inputs(self) -> Mapping[str, Any]:
        """Bound method inputs; values are also available as direct local names."""
        return MappingProxyType(self._inputs)

    def _committed(self) -> dict[str, Any]:
        return dict(self._namespace)

    @property
    def modules(self) -> Mapping[str, ModuleType]:
        """Available module aliases, including imports committed by completed cells."""
        namespace = dict(self._preloaded)
        namespace.update(self._committed())
        return MappingProxyType(
            {
                name: value
                for name, value in sorted(namespace.items())
                if isinstance(name, str)
                and not name.startswith("_")
                and isinstance(value, ModuleType)
            }
        )

    @property
    def locals(self) -> Mapping[str, Any]:
        """Non-module names committed by completed cells, including imported symbols."""
        excluded_names = {*self.inputs, *self._preloaded, "Out", "cell", "self"}
        values = {
            name: value
            for name, value in sorted(self._committed().items())
            if isinstance(name, str)
            and not name.startswith("_")
            and (
                name not in excluded_names
                or (name in self._preloaded and value is not self._preloaded[name])
            )
            and not isinstance(value, ModuleType)
        }
        return MappingProxyType(values)

    def _bind_namespace(self, namespace: dict[str, Any]) -> None:
        """Bind to the executor's committed namespace (framework use only)."""
        object.__setattr__(self, "_namespace", namespace)

    def _sync_summary(self, local_types: Mapping[str, str], modules: Mapping[str, str]) -> None:
        """Update parent-side display summaries after a sandboxed cell."""
        object.__setattr__(self, "_summary_local_types", dict(local_types))
        object.__setattr__(self, "_summary_modules", dict(modules))

    def _record_result(self, execution_count: int, result: Any) -> None:
        """Record one completed cell result (framework use only)."""
        if getattr(result, "error", None) is not None or not getattr(result, "has_return", False):
            return
        value = result.returned_value
        if value is None:
            return
        if (
            self._suppress_implicit_strings
            and not getattr(result, "explicit_return", False)
            and isinstance(value, str)
        ):
            return
        self._outputs._record(execution_count, value)

    def __repr__(self) -> str:
        lines = [f"cell.call: {self.call!r}"]
        lines.append(_typed_summary("cell.inputs", self.inputs))
        local_types = self._summary_local_types
        if local_types is None:
            lines.append(_typed_summary("cell.locals", self.locals))
        else:
            lines.append(_name_type_summary("cell.locals", local_types))
        module_names = self._summary_modules
        if module_names is None:
            lines.append(_module_summary(self.modules))
        else:
            lines.append(_module_name_summary(module_names))
        indices = self.outputs.indices
        if indices:
            last = self.outputs[-1]
            shown = ", ".join(str(index) for index in indices[:20])
            suffix = f", ... +{len(indices) - 20} more" if len(indices) > 20 else ""
            lines.append(
                f"cell.outputs: {len(indices)} available [{shown}{suffix}]; "
                f"latest: cell.outputs[{indices[-1]}] ({type(last).__name__})"
            )
        else:
            lines.append("cell.outputs: none")
        return "\n".join(lines)

    __str__ = __repr__


def _without_self(signature: str) -> str:
    """Remove a leading self parameter from a rendered method signature."""
    return signature.replace("(self, ", "(", 1).replace("(self)", "()", 1)


def _typed_summary(label: str, values: Mapping[str, Any], *, limit: int = 20) -> str:
    items = list(values.items())
    if not items:
        return f"{label}: none"
    shown = ", ".join(f"{name} ({type(value).__name__})" for name, value in items[:limit])
    suffix = f", ... +{len(items) - limit} more; inspect `{label}`" if len(items) > limit else ""
    return f"{label}: {shown}{suffix}"


def _module_summary(modules: Mapping[str, ModuleType], *, limit: int = 20) -> str:
    items = list(modules.items())
    if not items:
        return "cell.modules: none"
    shown = ", ".join(
        name if name == module.__name__ else f"{name} → {module.__name__}"
        for name, module in items[:limit]
    )
    suffix = (
        f", ... +{len(items) - limit} more; inspect `cell.modules`" if len(items) > limit else ""
    )
    return f"cell.modules: {shown}{suffix}"


def _name_type_summary(label: str, values: Mapping[str, str], *, limit: int = 20) -> str:
    items = list(values.items())
    if not items:
        return f"{label}: none"
    shown = ", ".join(f"{name} ({type_name})" for name, type_name in items[:limit])
    suffix = f", ... +{len(items) - limit} more; inspect `{label}`" if len(items) > limit else ""
    return f"{label}: {shown}{suffix}"


def _module_name_summary(modules: Mapping[str, str], *, limit: int = 20) -> str:
    items = list(modules.items())
    if not items:
        return "cell.modules: none"
    shown = ", ".join(
        name if name == module else f"{name} → {module}" for name, module in items[:limit]
    )
    suffix = (
        f", ... +{len(items) - limit} more; inspect `cell.modules`" if len(items) > limit else ""
    )
    return f"cell.modules: {shown}{suffix}"


__all__ = ["Cell", "CellCall", "CellOutputs"]
