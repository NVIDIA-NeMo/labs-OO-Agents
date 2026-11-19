# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""doc() of callable wrapper objects (``__wrapped__``) and of registered function views."""

from __future__ import annotations

import functools

from nooa.agentdoc import doc, spec
from nooa.agentdoc._structured import is_wrapped_callable
from nooa.agentdoc.ext import CallableInfo
from nooa.agentdoc.registry import (
    get_callable_info_extractor,
    unregister_callable_info_extractor,
)


@functools.lru_cache
def cached_add(x: int, y: int = 1) -> int:
    """Add two numbers (cached)."""
    return x + y


class _Dispatcher:
    """A wrapper like numpy's _ArrayFunctionDispatcher: callable, not a function."""

    def __init__(self, func):
        self.__wrapped__ = func
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__
        self.__doc__ = func.__doc__
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.__wrapped__(*args, **kwargs)


def _scale(values: list[float], factor: float = 2.0) -> list[float]:
    """Scale every value."""
    return [v * factor for v in values]


scale = _Dispatcher(_scale)


class TestWrappedCallables:
    def test_detection(self):
        assert is_wrapped_callable(cached_add)
        assert is_wrapped_callable(scale)
        assert not is_wrapped_callable(_scale)
        assert not is_wrapped_callable(_Dispatcher)
        assert not is_wrapped_callable(len)

    def test_lru_cache_wrapper_renders_as_function(self):
        out = doc(cached_add)
        assert out.splitlines()[0] == "def cached_add(x: int, y: int = 1) -> int:"
        assert "Add two numbers (cached)." in out
        assert "_lru_cache_wrapper" not in out

    def test_dispatcher_object_renders_as_function_without_calling_it(self):
        out = doc(scale)
        assert (
            out.splitlines()[0]
            == "def _scale(values: list[float], factor: float = 2.0) -> list[float]:"
        )
        assert "Scale every value." in out
        assert "class _Dispatcher" not in out
        assert scale.calls == 0

    def test_in_a_list(self):
        out = doc([cached_add, scale])
        assert "def cached_add(" in out and "def _scale(" in out


class TestCallableRegistry:
    def test_define_doc_replaces_a_function_view(self):
        def area(w, h):
            """Long original docstring."""
            return w * h

        @spec.define_doc(area)
        def _(func) -> CallableInfo:
            return CallableInfo(
                name="area", signature="(w, h)", return_type="float", docstring="Curated."
            )

        try:
            assert doc(area) == 'def area(w, h) -> float:\n    """Curated."""'
        finally:
            unregister_callable_info_extractor(area)
        assert get_callable_info_extractor(area) is None
        assert "Long original docstring." in doc(area)

    def test_lookup_is_by_identity(self):
        def f():
            """F."""

        def g():
            """G."""

        spec.define_doc(f)(lambda func: CallableInfo("f", "()", "", "Curated f."))
        try:
            assert get_callable_info_extractor(g) is None
            assert "Curated f." in doc(f)
        finally:
            unregister_callable_info_extractor(f)
