# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the ``test_extend_*`` files: named fake classes and an ndarray-like type."""

from __future__ import annotations

from typing import Any

from nooa.snapshot import Object, PythonRenderer, ndarray_like

FAKE_MODULE = "fakelib_not_installed_7c1e"


def _named(name: str, module: str = FAKE_MODULE, qualname: str | None = None) -> type[Any]:
    """A class that claims to live in ``module`` (which is never importable)."""
    cls = type(name, (), {"__module__": module})
    cls.__qualname__ = qualname or name
    return cls


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeNdarray:
    """Stands in for ``numpy.ndarray`` (C order); counts the elements read."""

    def __init__(self, values: list[float], shape: tuple[int, ...]) -> None:
        self._values = values
        self.shape = shape
        self.dtype = _DType("float64")
        self.read: list[tuple[int, ...]] = []

    def item(self, index: tuple[int, ...]) -> float:
        self.read.append(index)
        flat = 0
        for axis, i in enumerate(index):
            flat = flat * self.shape[axis] + i
        return self._value(flat)

    def _value(self, flat: int) -> float:
        return self._values[flat]

    def __repr__(self) -> str:
        raise AssertionError("repr must not run")


_FakeNdarray.__module__ = "fakenumpy_7c1e"
_FakeNdarray.__qualname__ = _FakeNdarray.__name__ = "ndarray"


class _HugeNdarray(_FakeNdarray):
    """A huge ndarray-like value whose elements are computed, never materialized."""

    def __init__(self, size: int) -> None:
        super().__init__([], (size // 1000, 1000))

    def _value(self, flat: int) -> float:
        return float(flat)


_HugeNdarray.__module__ = "fakenumpy_7c1e"
_HugeNdarray.__qualname__ = _HugeNdarray.__name__ = "ndarray"

NDARRAY = ndarray_like(
    lambda array: array.shape,
    lambda array, index: array.item(index),
    fields={"dtype": lambda array: array.dtype.name},
    kind="numpy.ndarray",
)
"""How numpy would be registered: ``capture.register(numpy.ndarray, NDARRAY)``."""


def render_ndarray(node: Object, r: PythonRenderer) -> str:
    shape, dtype, data = (node.field(name) for name in ("shape", "dtype", "data"))
    assert shape is not None and dtype is not None and data is not None
    return f"array({r.render(data)}, shape={r.render(shape)}, dtype={r.render(dtype)})"
