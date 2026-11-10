# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable opaque JSON without encoding/copying large string leaves per call."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer, SkipValidation


@dataclass(frozen=True, repr=False, slots=True)
class _FrozenObject(Mapping):
    """Deeply immutable, not hashable; immutability need not hash provider blobs."""

    _values: Mapping

    def __init__(self, values):
        """Freeze a JSON object after validating its key types."""
        frozen: dict[str, Any] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Native extension object keys must be strings, got {type(key).__name__}"
                )
            frozen[key] = freeze(value)
        object.__setattr__(self, "_values", MappingProxyType(frozen))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __deepcopy__(self, memo):
        return self


def freeze(value):
    """Return an immutable JSON value while rejecting lossy inputs."""
    if isinstance(value, _FrozenObject):
        return value
    if isinstance(value, dict):
        return _FrozenObject(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"Native extensions require finite JSON numbers, got {value}")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"Native extensions must contain JSON values only, got {type(value).__name__}")


def json_containers(value):
    """Allocate wire/persistence containers, borrowing immutable scalar leaves."""
    if isinstance(value, Mapping):
        return {key: json_containers(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_containers(item) for item in value]
    return value


def _native_object(value):
    if not isinstance(value, (dict, _FrozenObject)):
        raise ValueError("A native extension must be a JSON object")
    return freeze(value)


NativeJSON = Annotated[
    SkipValidation[Mapping[str, Any]],
    BeforeValidator(_native_object),
    PlainSerializer(json_containers),
]
