# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation markers for snapshot serialization.

Usage::

    from nooa.storage.markers import nosnapshot

    class MyAgent(Agent, llm=llm):
        db_conn: Annotated[Connection, nosnapshot]  # skipped during snapshot
        cache: Annotated[dict, nosnapshot]           # skipped during snapshot
        memory: list[str]                            # snapshotted normally
"""

import inspect
import sys
import typing
from typing import Annotated, Any


class _NoSnapshot:
    """Annotation marker for fields that should be excluded from snapshots.

    Use as ``Annotated[T, nosnapshot]`` on class fields. The snapshot
    extraction logic checks for this marker and skips annotated fields.

    Unlike ``hidden`` (which controls LLM visibility), ``nosnapshot``
    controls persistence — a field can be visible to the LLM but excluded
    from snapshots, or hidden from the LLM but included in snapshots.
    """

    def __repr__(self) -> str:
        return "nosnapshot"


nosnapshot = _NoSnapshot()


def is_nosnapshot_field(cls: type, name: str) -> bool:
    """Check if a field is annotated with Annotated[T, nosnapshot].

    Walks the MRO so subclasses can override parent annotations.
    """
    for klass in cls.__mro__:
        if name not in inspect.get_annotations(klass):
            continue
        try:
            resolved = typing.get_type_hints(klass, include_extras=True)
        except Exception:
            resolved = None

        hint = resolved.get(name) if resolved is not None else _resolve_single(klass, name)

        if hint is None:
            return False
        origin = typing.get_origin(hint)
        if origin is Annotated:
            args = typing.get_args(hint)
            return any(isinstance(arg, _NoSnapshot) for arg in args[1:])
        return False
    return False


def _resolve_single(klass: type, name: str) -> Any:
    """Resolve a single string annotation when full get_type_hints fails."""
    raw = inspect.get_annotations(klass).get(name)
    if raw is None or not isinstance(raw, str):
        return raw
    mod = sys.modules.get(klass.__module__)
    ns = vars(mod) if mod else {}
    try:
        return eval(raw, ns)  # noqa: S307
    except Exception:
        return None


def is_nosnapshot_value(value: Any) -> bool:
    """Check if a value's class has __nosnapshot__ = True.

    This allows classes to opt-out of snapshotting by setting a class attribute::

        class MyTool:
            __nosnapshot__ = True  # instances won't be snapshotted

    This is useful for dynamically-added attributes that don't have type annotations.
    """
    return getattr(type(value), "__nosnapshot__", False) is True


# ---------------------------------------------------------------------------
# @snapshotable — opt-in marker for custom classes
# ---------------------------------------------------------------------------


class _Snapshotable:
    """Class decorator that marks a class as snapshot-serializable.

    Usage::

        @snapshotable
        class MyConfig:
            def __init__(self, host: str, port: int = 8080):
                self.host = host
                self.port = port

    Sets ``cls.__snapshot_dict__ = True`` so that ``serialize()`` knows it
    can call ``vars(instance)`` to produce a serializable dict.
    """

    def __call__(self, cls: type) -> type:
        """Decorate a class to mark it as snapshotable."""
        cls.__snapshot_dict__ = True  # type: ignore[attr-defined]
        return cls

    def __repr__(self) -> str:
        return "snapshotable"


snapshotable = _Snapshotable()


def is_snapshotable_class(cls: type) -> bool:
    """Return True if *cls* has been decorated with ``@snapshotable``."""
    return getattr(cls, "__snapshot_dict__", False) is True
