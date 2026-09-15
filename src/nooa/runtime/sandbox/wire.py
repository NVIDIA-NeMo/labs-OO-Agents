# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgpack codec for the worker->parent side of the sandbox pipe.

Bytes from the worker are attacker-controlled (LLM-generated cells), so the
parent never unpickles them. Values cross as msgpack: primitives natively, a few
value types and the agent's declared data types (pydantic models, dataclasses,
enums, NamedTuples) as ``ExtType`` records the codec rebuilds itself. Decoding
can therefore only ever call the constructors listed in :meth:`Codec._decode`,
and agent types are rebuilt through their own validation. The parent->worker
direction stays pickle: the worker trusts the parent.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import pathlib
import sys
import typing
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
from pydantic import BaseModel

from nooa.runtime.sandbox.errors import CellSerializationError

HINT = (
    "Only JSON-like values (numbers, str, bytes, bool, None, list, tuple, set, dict), "
    "datetime/date/timedelta, Decimal, Path, UUID, numpy arrays, and the agent's declared "
    "data types (pydantic models, dataclasses, enums) can cross the sandbox boundary."
)

(
    _TUPLE,
    _SET,
    _FROZENSET,
    _DATETIME,
    _DATE,
    _TIMEDELTA,
    _DECIMAL,
    _PATH,
    _UUID,
    _NDARRAY,
    _ENUM,
    _MODEL,
    _DATACLASS,
    _NAMEDTUPLE,
) = range(14)


def type_key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class Codec:
    """Encode any value the worker can express; decode only into ``types``."""

    def __init__(self, types: Mapping[str, type] | None = None) -> None:
        self.types = dict(types or {})

    def dumps(self, value: Any) -> bytes:
        return msgpack.packb(value, default=self._encode, strict_types=True)

    def loads(self, data: bytes) -> Any:
        try:
            return msgpack.unpackb(data, ext_hook=self._decode, strict_map_key=False)
        except CellSerializationError:
            raise
        except Exception as exc:  # noqa: BLE001 - corrupt or forged stream
            raise CellSerializationError(f"malformed sandbox message ({exc})") from exc

    def _ext(self, code: int, payload: Any) -> msgpack.ExtType:
        return msgpack.ExtType(code, self.dumps(payload))

    def _encode(self, obj: Any) -> Any:  # noqa: C901 - flat type dispatch
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return self._ext(_NAMEDTUPLE, [type_key(type(obj)), list(obj)])
        if isinstance(obj, tuple):
            return self._ext(_TUPLE, list(obj))
        if isinstance(obj, frozenset):
            return self._ext(_FROZENSET, list(obj))
        if isinstance(obj, set):
            return self._ext(_SET, list(obj))
        if isinstance(obj, dict | list):  # subclasses such as OrderedDict, Counter
            return type(obj).__mro__[-2](obj)
        if isinstance(obj, enum.Enum):
            return self._ext(_ENUM, [type_key(type(obj)), obj.value])
        if isinstance(obj, BaseModel):
            return self._ext(_MODEL, [type_key(type(obj)), obj.model_dump()])
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
            return self._ext(_DATACLASS, [type_key(type(obj)), fields])
        if isinstance(obj, datetime.datetime):
            return self._ext(_DATETIME, obj.isoformat())
        if isinstance(obj, datetime.date):
            return self._ext(_DATE, obj.isoformat())
        if isinstance(obj, datetime.timedelta):
            return self._ext(_TIMEDELTA, [obj.days, obj.seconds, obj.microseconds])
        if isinstance(obj, decimal.Decimal):
            return self._ext(_DECIMAL, str(obj))
        if isinstance(obj, pathlib.PurePath):
            return self._ext(_PATH, str(obj))
        if isinstance(obj, uuid.UUID):
            return self._ext(_UUID, obj.bytes)
        np = sys.modules.get("numpy")
        if np is not None and isinstance(obj, np.generic):
            return obj.item()
        if np is not None and isinstance(obj, np.ndarray):
            descr = np.lib.format.dtype_to_descr(obj.dtype)
            body = list(obj.ravel()) if obj.dtype.hasobject else obj.tobytes()
            return self._ext(_NDARRAY, [descr, list(obj.shape), body])
        raise TypeError(f"cannot serialize {type(obj).__name__!r}")

    def _decode(self, code: int, data: bytes) -> Any:
        p = self.loads(data)
        if code == _TUPLE:
            return tuple(p)
        if code == _SET:
            return set(p)
        if code == _FROZENSET:
            return frozenset(p)
        if code == _DATETIME:
            return datetime.datetime.fromisoformat(p)
        if code == _DATE:
            return datetime.date.fromisoformat(p)
        if code == _TIMEDELTA:
            return datetime.timedelta(*p)
        if code == _DECIMAL:
            return decimal.Decimal(p)
        if code == _PATH:
            return pathlib.Path(p)
        if code == _UUID:
            return uuid.UUID(bytes=p)
        if code == _NDARRAY:
            return _ndarray(*p)
        key, body = p
        cls = self.types.get(key)
        if cls is None:
            raise CellSerializationError(f"{key!r} is not a data type declared by the agent")
        if code == _ENUM:
            return cls(body)
        if code == _MODEL:
            return cls.model_validate(body)
        if code == _DATACLASS:
            fields = dataclasses.fields(cls)
            inst = cls(**{f.name: body[f.name] for f in fields if f.init and f.name in body})
            for f in fields:  # init=False fields carry post-construction state
                if not f.init and f.name in body:
                    object.__setattr__(inst, f.name, body[f.name])
            return inst
        if code == _NAMEDTUPLE:
            return cls(*body)
        raise CellSerializationError(f"unknown sandbox message tag {code}")


def _ndarray(descr: Any, shape: list[int], body: Any) -> Any:
    import numpy as np

    dtype = np.dtype(descr)
    if dtype.hasobject:
        return np.fromiter(body, dtype=object, count=len(body)).reshape(shape)
    return np.frombuffer(body, dtype=dtype).reshape(shape).copy()


def send(conn: Any, message: dict[str, Any]) -> None:
    """Worker-side send; the parent decodes with :meth:`Codec.loads`."""
    conn.send_bytes(Codec().dumps(message))


def agent_types(agent: Any) -> dict[str, type]:
    """Data types a cell may hand the parent: pydantic models, dataclasses, enums and
    NamedTuples the LLM can name (visible module globals, public method annotations),
    closed over their field annotations. Ordinary classes are never constructed."""
    from nooa.agentdoc.visibility import filter_mro_module_globals

    cls = type(agent)
    seeds: list[Any] = list(filter_mro_module_globals(cls).values())
    for klass in cls.__mro__:
        if klass.__module__.split(".")[0] != "nooa":
            seeds.extend(
                t
                for name, member in vars(klass).items()
                if not name.startswith("_") and callable(member)
                for t in _annotated_types(member)
            )
    found: dict[str, type] = {}
    while seeds:
        t = seeds.pop()
        if isinstance(t, type) and _is_data_type(t) and type_key(t) not in found:
            found[type_key(t)] = t
            seeds.extend(_annotated_types(t))
    return found


def _is_data_type(t: type) -> bool:
    try:
        return (
            issubclass(t, enum.Enum | BaseModel)
            or (issubclass(t, tuple) and hasattr(t, "_fields"))
            or dataclasses.is_dataclass(t)
        )
    except TypeError:
        return False


def _annotated_types(obj: Any) -> list[type]:
    fields = getattr(obj, "model_fields", None)
    try:
        annotations = (
            [f.annotation for f in fields.values()]
            if isinstance(fields, dict)
            else list(typing.get_type_hints(obj, include_extras=True).values())
        )
    except Exception:  # noqa: BLE001 - unresolvable hints are skipped
        return []
    out: list[type] = []

    def walk(annotation: Any) -> None:
        if isinstance(annotation, type):
            out.append(annotation)
        for arg in typing.get_args(annotation):
            walk(arg)

    for annotation in annotations:
        walk(annotation)
    return out
