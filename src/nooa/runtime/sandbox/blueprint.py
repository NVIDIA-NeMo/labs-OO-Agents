# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstructible worker state for start methods that cannot inherit it.

``fork`` hands the child a live agent, the ``return_result`` closure and the
``CurrentCall`` by inheritance. ``spawn`` (the only method Windows has) copies
nothing but picklable data, so the worker rebuilds its namespace from a
:class:`WorkerBlueprint`:

* module-level names -> import the agent class by ``module:qualname`` and run
  the same MRO globals filter the parent uses;
* ``return_result``  -> rebuilt from the return type; the cell only needs the
  signal behaviour and the optional ``kind`` discriminator rule;
* ``_call``          -> a stand-in carrying the fields generated prefill code
  reads (``return_type`` and friends);
* method parameters  -> ``call_kwargs``, which therefore must pickle.

Anything the rebuild cannot recover is surfaced as an explicit error instead of
silently diverging from the fork path.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WorkerBlueprint", "BlueprintError"]


class BlueprintError(RuntimeError):
    """The agent's state cannot be reconstructed in a spawned worker."""


@dataclass(frozen=True)
class _CallShim:
    """Stand-in for ``CurrentCall`` inside a spawned worker.

    Generated code touches ``_call.return_type`` (prefill and introspection
    cells). The rest is carried for parity with the fork path, which inherits
    the real object.
    """

    id: str = ""
    method_name: str = ""
    decorator: str = ""
    signature: str | None = None
    docstring: str | None = None
    is_async: bool = False
    return_type: Any = None


@dataclass(frozen=True)
class WorkerBlueprint:
    """Everything a spawned worker needs, as picklable data."""

    agent_module: str
    agent_qualname: str
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    return_type: Any = None
    call_meta: _CallShim = field(default_factory=_CallShim)

    # --- construction ------------------------------------------------------
    @classmethod
    def from_agent(cls, agent: Any, framework_builtins: dict[str, Any] | None) -> WorkerBlueprint:
        """Capture the agent's class reference and the call's picklable context.

        Raises :class:`BlueprintError` when the class cannot be imported by name
        (a class defined inside a function or created dynamically), because a
        spawned child has no other way to obtain it.
        """
        agent_class = type(agent)
        module = getattr(agent_class, "__module__", "")
        qualname = getattr(agent_class, "__qualname__", "")
        if not module or "<locals>" in qualname:
            raise BlueprintError(
                f"agent class {qualname or agent_class!r} is not importable by name; "
                "start_method='spawn' requires a module-level agent class. "
                "Use start_method='fork' on a Unix host for a locally defined class."
            )

        builtins = framework_builtins or {}
        call = builtins.get("_call")
        return_type = getattr(call, "return_type", None)
        kwargs = _picklable_kwargs(getattr(call, "kwargs", None) or {})
        if return_type is not None and not _is_picklable(return_type):
            raise BlueprintError(
                f"return type {return_type!r} cannot cross a process boundary; "
                "start_method='spawn' needs a module-level return type"
            )

        shim = _CallShim(
            id=getattr(call, "id", "") or "",
            method_name=getattr(call, "method_name", "") or "",
            decorator=getattr(call, "decorator", "") or "",
            signature=getattr(call, "signature", None),
            docstring=getattr(call, "docstring", None),
            is_async=bool(getattr(call, "is_async", False)),
            return_type=return_type,
        )
        return cls(
            agent_module=module,
            agent_qualname=qualname,
            call_kwargs=kwargs,
            return_type=return_type,
            call_meta=shim,
        )

    # --- rebuild (runs in the child) ---------------------------------------
    def import_agent_class(self) -> type:
        """Import the agent class by name, with a diagnostic on failure."""
        try:
            module = importlib.import_module(self.agent_module)
        except ImportError as exc:
            raise BlueprintError(
                f"spawned worker cannot import agent module {self.agent_module!r}: {exc}"
            ) from exc
        obj: Any = module
        for part in self.agent_qualname.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError as exc:
                raise BlueprintError(
                    f"spawned worker cannot resolve {self.agent_qualname!r} "
                    f"in {self.agent_module!r}: {exc}"
                ) from exc
        if not isinstance(obj, type):
            raise BlueprintError(f"{self.agent_module}.{self.agent_qualname} is not a class")
        return obj

    def rebuild_builtins(self) -> dict[str, Any]:
        """Rebuild the framework builtins the worker's namespace expects."""
        from nooa.strategies.codeact import _ReturnResultSignal

        return_type = self.return_type

        def return_result(*args: Any, **kwargs: Any) -> None:
            """Submit the final answer from within execute_python code.

            Rebuilt in the child, so the docstring mirrors the one the fork path
            inherits. See ``CodeActStrategy._build_builtins`` for the rules.
            """
            if args:
                if len(args) > 1:
                    raise ValueError("return_result() takes at most 1 positional argument")
                if kwargs:
                    model_fields = getattr(return_type, "model_fields", {})
                    if "kind" not in model_fields:
                        raise ValueError("Cannot mix positional and keyword arguments")
                    if "kind" in kwargs:
                        raise ValueError(
                            "return_result() got kind both positionally and by keyword"
                        )
                    result = dict(kwargs)
                    result["kind"] = list(args)[0]
                    raise _ReturnResultSignal(result=result)
                raise _ReturnResultSignal(result={"result": list(args)[0]})
            raise _ReturnResultSignal(result=kwargs)

        builtins: dict[str, Any] = {
            "return_result": return_result,
            "_call": self.call_meta,
        }
        # Parameters come last, matching _build_builtins (a parameter named
        # return_result wins there too).
        builtins.update(self.call_kwargs)
        return builtins


def _is_picklable(value: Any) -> bool:
    import pickle

    try:
        pickle.dumps(value)
    except Exception:  # noqa: BLE001 - any failure means "cannot cross"
        return False
    return True


def _picklable_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the call parameters that can cross a process boundary."""
    kept: dict[str, Any] = {}
    unpicklable: list[str] = []
    for name, value in kwargs.items():
        if _is_picklable(value):
            kept[name] = value
        else:
            unpicklable.append(name)
    if unpicklable:
        raise BlueprintError(
            "start_method='spawn' cannot carry non-picklable method parameters: "
            + ", ".join(sorted(unpicklable))
            + ". Pass plain data, or use start_method='fork' on a Unix host."
        )
    return kept
