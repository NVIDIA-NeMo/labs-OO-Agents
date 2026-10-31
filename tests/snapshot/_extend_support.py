# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the ``test_extend_*`` files: named fake classes and an ndarray-like type."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nooa.snapshot import Object, Out, PythonRenderer

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
    """Stands in for ``numpy.ndarray``: its class claims the module ``numpy``."""

    def __init__(self, values: list[float], shape: tuple[int, ...]) -> None:
        self._values = values
        self.shape = shape
        self.dtype = _DType("float64")
        self.pulled = 0

    @property
    def flat(self) -> Iterator[float]:
        for value in self._values:
            self.pulled += 1
            yield value

    def __repr__(self) -> str:
        raise AssertionError("repr must not run")


_FakeNdarray.__module__ = "fakenumpy_7c1e"
_FakeNdarray.__qualname__ = "ndarray"


def capture_ndarray(array: Any, out: Out):
    return out.object(
        {
            "shape": out.capture(array.shape),
            "dtype": out.capture(array.dtype.name),
            "sample": out.sequence(array.flat),
        },
        type_name="ndarray",
        kind="numpy.ndarray",
    )


def render_ndarray(node: Object, r: PythonRenderer) -> str:
    shape, dtype, sample = (node.field(name) for name in ("shape", "dtype", "sample"))
    assert shape is not None and dtype is not None and sample is not None
    return f"array({r.render(sample)}, shape={r.render(shape)}, dtype={r.render(dtype)})"


class _LazyNdarray(_FakeNdarray):
    """A huge ndarray-like value whose ``flat`` is generated, never materialized."""

    def __init__(self, size: int) -> None:
        super().__init__([], (size // 1000, 1000))
        self._size = size

    @property
    def flat(self) -> Iterator[float]:
        for value in range(self._size):
            self.pulled += 1
            yield float(value)


_LazyNdarray.__module__ = "fakenumpy_7c1e"
_LazyNdarray.__qualname__ = "ndarray"
