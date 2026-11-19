# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The numpy doc() adapter: short curated views whose signatures match NumPy."""

from __future__ import annotations

import builtins
import importlib
import inspect

import pytest

np = pytest.importorskip("numpy")

from nooa.agentdoc import doc, spec  # noqa: E402
from nooa.agentdoc.adapters import register_all  # noqa: E402
from nooa.agentdoc.ext import CallableInfo  # noqa: E402
from nooa.agentdoc.registry import (  # noqa: E402
    get_callable_info_extractor,
    get_type_info_extractor,
)


@pytest.fixture(autouse=True)
def adapter():
    # Reload to re-run the registrations even if a sibling test cleared the registry.
    import nooa.agentdoc.adapters.numpy as _numpy_adapter

    return importlib.reload(_numpy_adapter)


def _param_names(signature: str) -> set[str]:
    """Parameter names in a displayed signature like ``(self, a, b=1, *, c=None, **kw)``."""
    inner = signature.strip()[1:-1]
    names = set()
    depth = 0
    part = ""
    for char in inner + ",":
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            token = part.strip().lstrip("*").split("=")[0].split(":")[0].strip()
            if token and token not in ("/", "self"):
                names.add(token)
            part = ""
        else:
            part += char
    return names


def _lines(obj) -> int:
    return len(doc(obj).splitlines())


class TestCuratedViewsAreShort:
    def test_ndarray(self):
        out = doc(np.ndarray)
        assert out.startswith("class ndarray(numpy.ndarray):")
        assert "Pitfalls:" in out
        assert len(out.splitlines()) < 120

    def test_ndarray_instance_uses_the_same_view(self):
        assert "Pitfalls:" in doc(np.zeros(3))
        assert _lines(np.zeros(3)) < 120

    def test_numpy_module(self):
        out = doc(np)
        assert out.startswith("# numpy")
        assert "np.linalg.norm" in out and "default_rng" in out
        assert len(out.splitlines()) < 120

    def test_zeros(self):
        out = doc(np.zeros)
        assert out.splitlines()[0] == (
            "def zeros(shape, dtype=None, order='C', *, device=None, like=None) -> ndarray:"
        )
        assert "Return a new array of given shape and type, filled with zeros." in out
        assert "Examples" not in out and ">>>" not in out
        assert len(out.splitlines()) <= 20

    def test_linalg_norm_renders_as_function(self):
        out = doc(np.linalg.norm)
        assert out.splitlines()[0] == "def norm(x, ord=None, axis=None, keepdims=False):"
        assert "_ArrayFunctionDispatcher" not in out
        assert "Matrix or vector norm." in out
        assert len(out.splitlines()) <= 20

    @pytest.mark.parametrize("module", ["linalg", "random"])
    def test_submodules(self, module):
        out = doc(getattr(np, module))
        assert out.startswith(f"# numpy.{module}")
        assert len(out.splitlines()) < 60

    def test_ufunc(self):
        out = doc(np.add)
        assert out.startswith("def add(x1, x2, /, out=None, *, where=True")
        assert "Add arguments element-wise." in out


class TestFunctionViews:
    def test_every_registered_function_renders_short(self):
        failures = []
        for module in (np, np.linalg, np.fft, np.random):
            for name, obj in list(builtins.vars(module).items()):
                if name.startswith("_") or get_callable_info_extractor(obj) is None:
                    continue
                out = doc(obj)
                if not out.startswith("def ") or len(out.splitlines()) > 60:
                    failures.append(f"{module.__name__}.{name}")
        assert not failures, failures

    def test_signature_is_numpys(self):
        for func in (np.concatenate, np.where, np.unique, np.linalg.svd, np.random.default_rng):
            out = doc(func)
            assert str(inspect.signature(func)) in out.splitlines()[0]

    def test_one_line_per_parameter(self):
        out = doc(np.unique)
        params = out.split("Parameters:")[1].split("Returns:")[0].strip().splitlines()
        names = [line.split(":")[0].strip() for line in params]
        assert names == list(inspect.signature(np.unique).parameters)


def _check_methods_match(owner, view_text: str):
    """Each listed method exists and its parameters are those NumPy documents."""
    checked = 0
    for line in view_text.splitlines():
        line = line.strip()
        if not line.startswith("def "):
            continue
        name = line[4 : line.index("(")]
        signature = line[line.index("(") : line.rindex(")") + 1].split(" -> ")[0]
        method = getattr(owner, name)
        real = inspect.signature(method).parameters
        real_names = {p for p in real if p != "self"}
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in real.values())
        doc_line = (method.__doc__ or "").strip().splitlines()[0]
        shown = _param_names(signature)
        if "[" in signature:  # NumPy's own optional-argument notation, e.g. view([dtype])
            assert signature[1:].replace("self, ", "") in doc_line
            continue
        for param in shown:
            assert param in real_names or (has_var_kw and param in doc_line), (name, param)
        for param, spec_ in real.items():
            if param == "self" or spec_.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            assert param in shown or spec_.kind is inspect.Parameter.VAR_POSITIONAL, (name, param)
        checked += 1
    return checked


class TestSignaturesMatchNumpy:
    def test_ndarray_methods(self):
        assert _check_methods_match(np.ndarray, doc(np.ndarray)) >= 28

    def test_ndarray_fields_exist(self):
        info = get_type_info_extractor(np.ndarray)(np.ndarray)
        for field in info.fields:
            assert hasattr(np.ndarray, field.name)
        assert {"shape", "dtype", "ndim", "size", "T"} <= {f.name for f in info.fields}

    def test_generator_methods(self):
        assert _check_methods_match(np.random.Generator, doc(np.random.Generator)) == 8


class TestIdioms:
    def test_ndarray_idioms_run(self, adapter):
        builtins.exec(adapter.NDARRAY_IDIOMS, {"np": np})  # noqa: S102 - our own idioms

    def test_idioms_shown_in_doc(self, adapter):
        out = doc(np.ndarray)
        for line in adapter.NDARRAY_IDIOMS.splitlines():
            assert line in out


class TestWiring:
    def test_register_all_includes_numpy(self):
        assert "numpy" in register_all()

    def test_lazy_adapter_table(self):
        from nooa.agentdoc.core import _LAZY_ADAPTERS

        assert _LAZY_ADAPTERS["numpy"] == "nooa.agentdoc.adapters.numpy"

    def test_define_doc_accepts_a_function(self):
        def target(x):
            return x

        @spec.define_doc(target)
        def _(func) -> CallableInfo:
            return CallableInfo(name="target", signature="(x)", return_type="int", docstring="Hi.")

        assert doc(target) == 'def target(x) -> int:\n    """Hi."""'
