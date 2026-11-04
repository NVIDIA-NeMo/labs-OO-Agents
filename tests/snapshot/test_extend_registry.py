# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extension registry: ``capture.register(T, shape)`` dispatch by class identity, and its safety."""

from __future__ import annotations

from typing import Any

import pytest

from nooa.snapshot import (
    Object,
    Scalar,
    Unavailable,
    _shapes,
    array_like,
    capture,
    mapping_like,
    ndarray_like,
    object_like,
    render_python,
    text_like,
)
from tests.snapshot._extend_support import (
    FAKE_MODULE,
    _named,
)


def _text(value: Any) -> Any:
    """A shape that captures every instance as the text ``value`` (``str(value)``)."""
    text = str(value)
    return text_like(lambda v: len(text), lambda v, start, stop: text[start:stop])


# --- register(T, shape): identity dispatch ---------------------------------------------


def test_register_decorator_form_and_unregister() -> None:
    Gadget = _named("Gadget")

    capture.register(Gadget, _text("gadget"))
    try:
        assert render_python(capture(Gadget())) == "'gadget'"
    finally:
        capture.unregister(Gadget)
    assert capture(Gadget()) == Unavailable("opaque", "Gadget")
    capture.unregister(Gadget)  # absent: a no-op


def test_register_needs_a_class() -> None:
    for not_a_class in ("numpy.ndarray", None, 1, _named("Widget")()):
        with pytest.raises(TypeError):
            capture.register(not_a_class, _text("x"))  # type: ignore[arg-type]


def test_register_needs_a_shape() -> None:
    Widget = _named("Widget")
    not_shapes: list[Any] = [
        lambda value, out: out.capture(1),  # the removed function-handler form
        None,
        "array_like",
        {"length": len},
        _shapes._PYDANTIC,  # internal shapes are not public
    ]
    for not_a_shape in not_shapes:
        with pytest.raises(TypeError):
            capture.register(Widget, not_a_shape)
    with pytest.raises(TypeError):
        capture.register(Widget)  # type: ignore[call-arg]
    assert capture(Widget()) == Unavailable("opaque", "Widget")  # nothing was registered


@pytest.mark.parametrize(
    "make",
    [
        lambda: array_like(1, iter),  # type: ignore[arg-type]  # not callable
        lambda: array_like(len, iter, fields={"items": len}),  # taken by the shape
        lambda: array_like(len, iter, fields={1: len}),  # type: ignore[arg-type]  # not a str name
        lambda: array_like(len, iter, fields=[("a", len)]),  # type: ignore[arg-type]  # not a dict
        lambda: mapping_like(len, iter, kind=3),  # type: ignore[arg-type]  # not a str kind
        lambda: ndarray_like(len, len, fields={"data": len}),  # type: ignore[arg-type]
        lambda: ndarray_like(len, len, labels={-1: len}),  # type: ignore[arg-type]
        lambda: ndarray_like(len, len, labels={"columns": len}),  # type: ignore[arg-type]
        lambda: object_like([("a", len)]),  # type: ignore[arg-type]
        lambda: object_like({"a": "not callable"}),  # type: ignore[arg-type]
        lambda: text_like(len, len, unit="lines"),  # type: ignore[arg-type]
    ],
)
def test_shape_constructors_check_their_arguments(make) -> None:
    with pytest.raises(TypeError):
        make()


def test_register_matches_the_class_not_its_name(registered) -> None:
    Inner = _named("Inner", qualname="Outer.Inner")
    registered(Inner, _text("inner"))

    assert render_python(capture(Inner())) == "'inner'"
    # Another class with the same module and qualname is not registered.
    assert capture(_named("Inner", qualname="Outer.Inner")()) == Unavailable("opaque", "Inner")


def test_subclass_matches_and_unregistered_stays_opaque(registered) -> None:
    Base = _named("Base")
    Child = type("Child", (Base,), {"__module__": "elsewhere"})
    registered(Base, _text("base"))

    assert render_python(capture(Child())) == "'base'"
    assert capture(_named("Other")()) == Unavailable("opaque", "Other")


def test_nearest_registered_class_in_the_mro_wins(registered) -> None:
    Base = _named("Base")

    class Middle(Base):  # type: ignore[misc,valid-type]
        pass

    Leaf = type("Leaf", (Middle,), {"__module__": FAKE_MODULE})

    registered(Middle, _text("middle"))
    registered(Base, _text("base"))  # registered later, still farther

    assert render_python(capture(Base())) == "'base'"
    assert render_python(capture(Middle())) == "'middle'"
    assert render_python(capture(Leaf())) == "'middle'"

    registered(Leaf, _text("leaf"))
    assert render_python(capture(Leaf())) == "'leaf'"  # a new registration applies
    capture.unregister(Leaf)
    assert render_python(capture(Leaf())) == "'middle'"
    capture.unregister(Middle)
    assert render_python(capture(Leaf())) == "'base'"


def test_registering_again_replaces_it(registered) -> None:
    Both = _named("Both")
    registered(Both, _text("first"))
    registered(Both, _text("second"))

    assert render_python(capture(Both())) == "'second'"


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


def test_metaclass_hooks_never_run_during_dispatch(registered) -> None:
    Hostile = _hostile("Hostile")
    Sub = _hostile("Sub", (Hostile,))
    Unregistered = _hostile("Unregistered")
    _Recording.calls.clear()

    # Before any registration, the hostile classes are looked up and miss.
    assert render_python(capture([Hostile(), Unregistered()])) == (
        "[<opaque: Hostile>, <opaque: Unregistered>]"
    )
    registered(int, _text(0))  # a non-empty registry to compare against
    assert render_python(capture([Unregistered()])) == "[<opaque: Unregistered>]"
    registered(Hostile, _text("matched"))

    for _ in range(2):
        assert render_python(capture(Hostile())) == "'matched'"
        assert render_python(capture(Sub())) == "'matched'"
        assert render_python(capture([Unregistered()])) == "[<opaque: Unregistered>]"
    capture.unregister(Hostile)
    assert render_python(capture(Sub())) == "<opaque: Sub>"
    assert _Recording.calls == []


def test_dispatch_survives_a_class_rename(registered) -> None:
    class Target:
        pass

    registered(Target, object_like({"ok": lambda value: True}))
    Target.__module__ = "renamed_module"
    Target.__qualname__ = "Renamed"

    renamed = capture(Target())
    assert isinstance(renamed, Object) and renamed.fields == (("ok", Scalar(True)),)


def test_registration_retains_only_registered_classes(registered) -> None:
    import gc
    import weakref

    Matched = _named("Matched")
    registered(Matched, _text(1))
    Seen = type("Seen", (Matched,), {})
    assert render_python(capture(Seen())) == "'1'"
    reference = weakref.ref(Seen)

    del Seen
    gc.collect()
    assert reference() is None  # a matched subclass is not kept


def test_public_api_is_what_extension_authors_and_consumers_use() -> None:
    import nooa.snapshot as snapshot

    for private in ("Deferred", "Out"):
        assert private not in snapshot.__all__ and not hasattr(snapshot, private)
    for shape in ("array_like", "ndarray_like", "mapping_like", "object_like", "text_like"):
        assert shape in snapshot.__all__
    for name in snapshot.__all__:
        assert getattr(snapshot, name) is not None
