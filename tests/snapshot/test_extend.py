# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extending for third-party types: ``register_lazy``, ``Object.kind`` and render hooks."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from nooa.snapshot import (
    JsonRenderer,
    Limits,
    Object,
    Opaque,
    Out,
    OutputRenderer,
    PythonRenderer,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
    capture,
    preview_attributes,
    render_json,
    render_output,
    render_python,
)

FAKE_MODULE = "fakelib_not_installed_7c1e"


@pytest.fixture
def lazy() -> Iterator[Any]:
    """``capture.register_lazy`` with cleanup."""
    names: list[tuple[str, str]] = []

    def register(module: str, qualname: str, function: Any) -> None:
        capture.register_lazy(module, qualname, function)
        names.append((module, qualname))

    yield register
    for module, qualname in names:
        capture.unregister_lazy(module, qualname)


@pytest.fixture
def eager() -> Iterator[Any]:
    registered: list[type[Any]] = []

    def register(cls: type[Any], function: Any) -> None:
        capture.register(cls, function)
        registered.append(cls)

    yield register
    for cls in registered:
        capture.unregister(cls)


@pytest.fixture
def hooks() -> Iterator[Any]:
    """Register render hooks on a renderer with cleanup: ``hooks(renderer, kind, fn)``."""
    added: list[tuple[Any, str]] = []

    def register(renderer: Any, kind: str, function: Any) -> None:
        renderer.register(kind, function)
        added.append((renderer, kind))

    yield register
    for renderer, kind in added:
        renderer.unregister(kind)


def _named(name: str, module: str = FAKE_MODULE, qualname: str | None = None) -> type[Any]:
    """A class that claims to live in ``module`` (which is never importable)."""
    cls = type(name, (), {"__module__": module})
    cls.__qualname__ = qualname or name
    return cls


# --- register_lazy -------------------------------------------------------------------


def test_register_lazy_never_imports_the_module(lazy) -> None:
    before = set(sys.modules)
    lazy(FAKE_MODULE, "Widget", lambda value, out: out.capture("widget"))
    assert FAKE_MODULE not in sys.modules

    Widget = _named("Widget")
    assert capture(Widget()) == Text("widget", "", 6, "chars", True, None)
    assert FAKE_MODULE not in sys.modules
    assert set(sys.modules) - before == set()


def test_register_lazy_decorator_form() -> None:
    @capture.register_lazy(FAKE_MODULE, "Gadget")
    def gadget(value: Any, out: Out) -> Snapshot:
        return out.capture(1)

    try:
        assert capture(_named("Gadget")()) == Scalar(1)
    finally:
        capture.unregister_lazy(FAKE_MODULE, "Gadget")
    assert capture(_named("Gadget")()) == Opaque(FAKE_MODULE, "Gadget")


def test_register_lazy_matches_module_and_qualname(lazy) -> None:
    lazy(FAKE_MODULE, "Outer.Inner", lambda value, out: out.capture("inner"))

    assert render_python(capture(_named("Inner", qualname="Outer.Inner")())) == "'inner'"
    # Same qualname, other module; same module, other qualname: not registered.
    assert capture(_named("Inner", module="other", qualname="Outer.Inner")()) == Opaque(
        "other", "Inner"
    )
    assert capture(_named("Inner")()) == Opaque(FAKE_MODULE, "Inner")


def test_register_lazy_with_a_class_defined_in_this_module(lazy) -> None:
    class Local:
        pass

    lazy(__name__, Local.__qualname__, lambda value, out: out.capture("local"))
    assert render_python(capture(Local())) == "'local'"


def test_register_lazy_needs_names() -> None:
    with pytest.raises(TypeError):
        capture.register_lazy("", "x")
    with pytest.raises(TypeError):
        capture.register_lazy(FAKE_MODULE, None)  # type: ignore[arg-type]


def test_lazy_subclass_matches_and_unregistered_stays_opaque(lazy) -> None:
    Base = _named("Base")
    Child = type("Child", (Base,), {"__module__": "elsewhere"})
    lazy(FAKE_MODULE, "Base", lambda value, out: out.capture("base"))

    assert render_python(capture(Child())) == "'base'"
    assert capture(_named("Other")()) == Opaque(FAKE_MODULE, "Other")


def test_nearest_mro_wins_between_eager_and_lazy(lazy, eager) -> None:
    LazyBase = _named("LazyBase")

    class EagerMiddle(LazyBase):  # type: ignore[misc,valid-type]
        pass

    LazyLeaf = type("LazyLeaf", (EagerMiddle,), {"__module__": FAKE_MODULE})

    lazy(FAKE_MODULE, "LazyBase", lambda value, out: out.capture("lazy-base"))
    eager(EagerMiddle, lambda value, out: out.capture("eager-middle"))

    assert render_python(capture(LazyBase())) == "'lazy-base'"
    assert render_python(capture(EagerMiddle())) == "'eager-middle'"
    assert render_python(capture(LazyLeaf())) == "'eager-middle'"

    lazy(FAKE_MODULE, "LazyLeaf", lambda value, out: out.capture("lazy-leaf"))
    assert render_python(capture(LazyLeaf())) == "'lazy-leaf'"  # cache invalidated


def test_eager_wins_over_lazy_at_the_same_class(lazy, eager) -> None:
    Both = _named("Both")
    lazy(FAKE_MODULE, "Both", lambda value, out: out.capture("lazy"))
    eager(Both, lambda value, out: out.capture("eager"))

    assert render_python(capture(Both())) == "'eager'"


def test_lazy_resolution_is_cached_by_identity_and_forgets_dead_classes(lazy) -> None:
    import gc

    from nooa.snapshot import _capture

    lazy(FAKE_MODULE, "Cached", lambda value, out: out.capture(1))
    Cached = _named("Cached")
    assert capture(Cached()) == Scalar(1)
    marker = id(Cached)
    resolved = _capture._REGISTRY._resolved
    assert marker in resolved and resolved[marker][0]() is Cached

    del Cached
    gc.collect()
    assert marker not in resolved or resolved[marker][0]() is None


class _Recording(type):
    """A metaclass whose hooks raise and record: none may run during capture."""

    calls: list[str] = []

    def __hash__(cls) -> int:
        _Recording.calls.append("__hash__")
        raise AssertionError("metaclass __hash__ ran")

    def __eq__(cls, other: object) -> bool:
        _Recording.calls.append("__eq__")
        raise AssertionError("metaclass __eq__ ran")

    def __ne__(cls, other: object) -> bool:
        _Recording.calls.append("__ne__")
        raise AssertionError("metaclass __ne__ ran")

    def __getattribute__(cls, name: str) -> Any:
        _Recording.calls.append(f"__getattribute__({name})")
        raise AssertionError(f"metaclass __getattribute__({name}) ran")

    def __getattr__(cls, name: str) -> Any:
        _Recording.calls.append(f"__getattr__({name})")
        raise AssertionError(f"metaclass __getattr__({name}) ran")


def _hostile(name: str, bases: tuple[type[Any], ...] = ()) -> type[Any]:
    return _Recording(name, bases, {"__module__": FAKE_MODULE, "__qualname__": name})


def test_metaclass_hooks_never_run_during_lazy_dispatch(lazy, eager) -> None:
    Hostile = _hostile("Hostile")
    Sub = _hostile("Sub", (Hostile,))
    Unregistered = _hostile("Unregistered")
    _Recording.calls.clear()

    lazy(FAKE_MODULE, "Hostile", lambda value, out: out.capture("matched"))
    eager(int, lambda value, out: out.capture(0))  # a non-empty eager list to compare against

    for _ in range(2):  # the second round is served from the cache
        assert render_python(capture(Hostile())) == "'matched'"
        assert render_python(capture(Sub())) == "'matched'"
        assert render_python(capture([Unregistered()])) == "[<opaque: Unregistered>]"
    assert _Recording.calls == []


# --- Object.kind ---------------------------------------------------------------------


def test_kind_round_trips_through_the_builder(lazy) -> None:
    lazy(
        FAKE_MODULE,
        "Kinded",
        lambda value, out: out.object({"a": 1, "b": out.capture("x")}, kind="fake.kinded"),
    )
    snapshot = capture(_named("Kinded")())

    assert isinstance(snapshot, Object)
    assert snapshot.kind == "fake.kinded"
    assert snapshot.module == FAKE_MODULE
    assert snapshot.field("a") == Scalar(1)
    assert snapshot.field("b") == Text("x", "", 1, "chars", True, None)
    assert snapshot.field("missing") is None
    assert render_python(snapshot) == "Kinded(a=1, b='x')"
    assert json.loads(render_json(snapshot, max_chars=1_000)) == {"a": 1, "b": "x"}


def test_kind_survives_cuts(lazy) -> None:
    lazy(
        FAKE_MODULE,
        "Cut",
        lambda value, out: out.object(((f"f{i}", "y" * 50) for i in range(100)), kind="k"),
    )
    snapshot = capture(_named("Cut")(), Limits(max_chars=200))
    assert isinstance(snapshot, Object) and not snapshot.complete and snapshot.kind == "k"

    deep = capture([[_named("Cut")()]], Limits(max_depth=1))
    assert isinstance(deep, Sequence)


def test_default_kind_is_none_and_must_be_a_string(eager) -> None:
    class Plain:
        pass

    eager(Plain, lambda value, out: out.object([("a", 1)]))
    snapshot = capture(Plain())
    assert isinstance(snapshot, Object) and snapshot.kind is None

    eager(Plain, lambda value, out: out.object([], kind=3))
    assert capture(Plain()) == Unavailable("error", "Plain")


def test_prebuilt_node_too_large_for_its_slot_is_cut(lazy) -> None:
    def build(value: Any, out: Out) -> Snapshot:
        big = out.capture("z" * 500)
        return out.object({"big": big, "more": "w" * 500})

    lazy(FAKE_MODULE, "Pre", build)
    snapshot = capture(_named("Pre")(), Limits(max_chars=300))
    text = render_json(snapshot, max_chars=300)
    assert len(text) <= 300
    json.loads(text)


# --- render hooks --------------------------------------------------------------------

KIND = "fake.thing"


def _thing(**fields: Any) -> Object:
    nodes = tuple((name, capture(value)) for name, value in fields.items())
    return Object("Thing", nodes, True, None, FAKE_MODULE, KIND)


def test_render_python_hook(hooks) -> None:
    seen: list[Any] = []

    def hook(node: Object, r: PythonRenderer) -> str:
        seen.append(r)
        return f"thing<{r.render(node.fields[0][1])}>"

    hooks(render_python, KIND, hook)

    assert render_python([_thing(a="x" * 10)][0]) == "thing<'xxxxxxxxxx'>"
    assert isinstance(seen[0], PythonRenderer)
    nested = Sequence("list", 1, (_thing(a=1),), True, None)
    assert render_python(nested) == "[thing<1>]"


def test_render_python_hook_children_keep_notation(hooks) -> None:
    hooks(render_python, KIND, lambda node, r: f"T({r.render(node.fields[0][1])})")
    cut = capture("abcdefghij", Limits(max_string=4, tail_share=0.5))
    node = Object("Thing", (("s", cut),), True, None, "", KIND)

    assert render_python(node) == "T(str(len=10, [:2]='ab', [-2:]='ij'))"


def test_render_python_raising_or_non_str_hook_is_generic(hooks) -> None:
    def broken(node: Object, r: PythonRenderer) -> str:
        raise RuntimeError("boom")

    hooks(render_python, KIND, broken)
    assert render_python(_thing(a=1)) == "Thing(a=1)"

    hooks(render_python, KIND, lambda node, r: None)
    assert render_python(_thing(a=1)) == "Thing(a=1)"

    hooks(render_python, KIND, lambda node, r: r.render(node))  # recursion: generic
    assert render_python(_thing(a=1)) == "Thing(a=1)"


def test_no_hook_for_kind_is_generic() -> None:
    assert render_python(_thing(a=1)) == "Thing(a=1)"
    assert render_output(_thing(a=1)) == "Thing(a=1)"
    assert render_json(_thing(a=1), max_chars=100) == '{"a": 1}'


def test_render_output_hook_uses_output_notation(hooks) -> None:
    seen: list[Any] = []

    def hook(node: Object, r: OutputRenderer) -> str:
        seen.append(r)
        return "log:\n" + r.render(node.fields[0][1])

    hooks(render_output, KIND, hook)
    cut = Text("head", "tail", 1_000, "bytes", False, "max_length", omitted=992)
    node = Object("Log", (("text", cut),), True, None, "", KIND)

    assert render_output(node) == "log:\nhead\n[... 992 bytes omitted; full output not kept]\ntail"
    assert isinstance(seen[0], OutputRenderer)
    # Not registered for render_python: the Python notation is unaffected.
    assert render_python(node) == "Log(text=bytes(len=1000, [:4]=b'head', [-4:]=b'tail'))"


def test_render_output_text_is_unchanged_and_raising_hook_is_generic(hooks) -> None:
    text = Text("a", "b", 10, "chars", False, "max_length", omitted=8)
    assert render_output(text) == "a\n[... 8 chars omitted; full output not kept]\nb"

    hooks(render_output, KIND, lambda node, r: 1 / 0)
    assert render_output(_thing(a="x")) == "Thing(a=x)"


def test_render_json_hook(hooks) -> None:
    def hook(node: Object, r: JsonRenderer) -> str:
        return '{"thing": ' + r.render(node.fields[0][1], key="thing") + "}"

    hooks(render_json, KIND, hook)
    out = render_json(Sequence("list", 1, (_thing(a=[1, 2]),), True, None), max_chars=100)
    assert json.loads(out) == [{"thing": [1, 2]}]


def test_render_json_hook_children_are_cut_and_reported(hooks) -> None:
    def hook(node: Object, r: JsonRenderer) -> str:
        return '{"s": ' + r.render(node.fields[0][1], max_chars=r.max_chars - 8, key="s") + "}"

    hooks(render_json, KIND, hook)
    node = _thing(s="x" * 1_000)
    text = render_json(node, max_chars=50)

    assert len(text) <= 50
    assert json.loads(text)["s"].startswith("xxx")
    attributes = preview_attributes("output", node, max_chars=50)
    assert attributes["nooa.output.preview.incomplete"] is True
    assert attributes["nooa.output.preview.paths"] == ("/s",)


@pytest.mark.parametrize(
    "bad",
    [
        lambda node, r: "{not json",
        lambda node, r: '{"a": NaN}',
        lambda node, r: '"' + "x" * 5_000 + '"',  # valid JSON, too long
        lambda node, r: 42,  # not a str
        lambda node, r: 1 / 0,  # raises
        lambda node, r: r.render(node, max_chars=0),  # misuse of r raises
    ],
)
def test_render_json_bad_hook_falls_back_to_generic(hooks, bad) -> None:
    hooks(render_json, KIND, bad)
    node = _thing(a="y" * 200)

    for max_chars in (5, 40, 1_000):
        text = render_json(node, max_chars=max_chars)
        assert len(text) <= max_chars
        value = json.loads(text)
        assert type(value) is dict
        assert text == _generic_json(node, max_chars)


def _generic_json(node: Object, max_chars: int) -> str:
    from nooa.snapshot import _render

    hooks = render_json._hooks.functions.copy()
    render_json._hooks.functions.clear()
    try:
        return _render._render(node, max_chars)[0]
    finally:
        render_json._hooks.functions.update(hooks)


def test_render_json_hook_recursing_on_its_node_is_generic(hooks) -> None:
    hooks(render_json, KIND, lambda node, r: r.render(node))
    node = _thing(a=1)
    assert render_json(node, max_chars=100) == '{"a": 1}'


# --- a realistic example: an ndarray-like class, no numpy ------------------------------


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


def capture_ndarray(array: Any, out: Out) -> Snapshot:
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


def test_ndarray_like_example(lazy, hooks) -> None:
    before = set(sys.modules)
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)
    hooks(render_python, "numpy.ndarray", render_ndarray)

    small = _FakeNdarray([1.0, 2.0, 3.0, 4.0], (2, 2))
    snapshot = capture({"x": small})
    assert render_python(snapshot) == (
        "{'x': array([1.0, 2.0, 3.0, 4.0], shape=(2, 2), dtype='float64')}"
    )
    # No json hook: the trace preview stays plain, generic JSON.
    assert json.loads(render_json(snapshot, max_chars=1_000)) == {
        "x": {"shape": [2, 2], "dtype": "float64", "sample": [1.0, 2.0, 3.0, 4.0]}
    }

    big = _FakeNdarray([float(i) for i in range(1_000_000)], (1000, 1000))
    cut = capture(big, Limits(max_chars=200, max_items=3))
    assert isinstance(cut, Object) and cut.kind == "numpy.ndarray"
    assert render_python(cut) == (
        "array(list(len=?, [:3]=[0.0, 1.0, 2.0]), shape=(1000, 1000), dtype='float64')"
    )
    assert big.pulled < 10  # the budget stopped the sample; the rest was never pulled
    assert len(render_json(cut, max_chars=200)) <= 200
    assert set(sys.modules) - before == set()


def test_ndarray_like_failing_hook_falls_back(lazy, hooks) -> None:
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)
    hooks(render_python, "numpy.ndarray", lambda node, r: node.fields["shape"])  # a TypeError

    snapshot = capture(_FakeNdarray([1.0], (1,)))
    assert render_python(snapshot) == "ndarray(shape=(1,), dtype='float64', sample=[1.0])"
