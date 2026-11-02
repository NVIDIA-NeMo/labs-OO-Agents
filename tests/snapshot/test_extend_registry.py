# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extension registry: ``capture.register(T)`` dispatch by class identity, and its safety."""

from __future__ import annotations

from typing import Any

import pytest

from nooa.snapshot import (
    Object,
    Opaque,
    Out,
    Scalar,
    capture,
    render_python,
)
from tests.snapshot._extend_support import (
    FAKE_MODULE,
    _named,
)

# --- register(T): identity dispatch ---------------------------------------------------


def test_register_decorator_form_and_unregister() -> None:
    Gadget = _named("Gadget")

    @capture.register(Gadget)
    def gadget(value: Any, out: Out):
        return out.capture(1)

    try:
        assert capture(Gadget()) == Scalar(1)
    finally:
        capture.unregister(Gadget)
    assert capture(Gadget()) == Opaque(FAKE_MODULE, "Gadget")
    capture.unregister(Gadget)  # absent: a no-op


def test_register_needs_a_class() -> None:
    for not_a_class in ("numpy.ndarray", None, 1, _named("Widget")()):
        with pytest.raises(TypeError):
            capture.register(not_a_class)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            capture.register(not_a_class, lambda value, out: out.capture(1))  # type: ignore[arg-type]


def test_register_matches_the_class_not_its_name(eager) -> None:
    Inner = _named("Inner", qualname="Outer.Inner")
    eager(Inner, lambda value, out: out.capture("inner"))

    assert render_python(capture(Inner())) == "'inner'"
    # Another class with the same module and qualname is not registered.
    assert capture(_named("Inner", qualname="Outer.Inner")()) == Opaque(FAKE_MODULE, "Inner")


def test_subclass_matches_and_unregistered_stays_opaque(eager) -> None:
    Base = _named("Base")
    Child = type("Child", (Base,), {"__module__": "elsewhere"})
    eager(Base, lambda value, out: out.capture("base"))

    assert render_python(capture(Child())) == "'base'"
    assert capture(_named("Other")()) == Opaque(FAKE_MODULE, "Other")


def test_nearest_registered_class_in_the_mro_wins(eager) -> None:
    Base = _named("Base")

    class Middle(Base):  # type: ignore[misc,valid-type]
        pass

    Leaf = type("Leaf", (Middle,), {"__module__": FAKE_MODULE})

    eager(Middle, lambda value, out: out.capture("middle"))
    eager(Base, lambda value, out: out.capture("base"))  # registered later, still farther

    assert render_python(capture(Base())) == "'base'"
    assert render_python(capture(Middle())) == "'middle'"
    assert render_python(capture(Leaf())) == "'middle'"

    eager(Leaf, lambda value, out: out.capture("leaf"))
    assert render_python(capture(Leaf())) == "'leaf'"  # a new registration applies
    capture.unregister(Leaf)
    assert render_python(capture(Leaf())) == "'middle'"
    capture.unregister(Middle)
    assert render_python(capture(Leaf())) == "'base'"


def test_registering_again_replaces_it(eager) -> None:
    Both = _named("Both")
    eager(Both, lambda value, out: out.capture("first"))
    eager(Both, lambda value, out: out.capture("second"))

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


def test_metaclass_hooks_never_run_during_dispatch(eager) -> None:
    Hostile = _hostile("Hostile")
    Sub = _hostile("Sub", (Hostile,))
    Unregistered = _hostile("Unregistered")
    _Recording.calls.clear()

    # Before any registration, the hostile classes are looked up and miss.
    assert render_python(capture([Hostile(), Unregistered()])) == (
        "[<opaque: Hostile>, <opaque: Unregistered>]"
    )
    eager(int, lambda value, out: out.capture(0))  # a non-empty registry to compare against
    assert render_python(capture([Unregistered()])) == "[<opaque: Unregistered>]"
    eager(Hostile, lambda value, out: out.capture("matched"))

    for _ in range(2):
        assert render_python(capture(Hostile())) == "'matched'"
        assert render_python(capture(Sub())) == "'matched'"
        assert render_python(capture([Unregistered()])) == "[<opaque: Unregistered>]"
    capture.unregister(Hostile)
    assert render_python(capture(Sub())) == "<opaque: Sub>"
    assert _Recording.calls == []


def test_dispatch_survives_a_class_rename(eager) -> None:
    class Target:
        pass

    eager(Target, lambda value, out: out.object({"ok": True}))
    Target.__module__ = "renamed_module"
    Target.__qualname__ = "Renamed"

    renamed = capture(Target())
    assert isinstance(renamed, Object) and renamed.fields == (("ok", Scalar(True)),)
    assert renamed.module == "renamed_module"


def test_registration_retains_only_registered_classes(eager) -> None:
    import gc
    import weakref

    Matched = _named("Matched")
    eager(Matched, lambda value, out: out.capture(1))
    Seen = type("Seen", (Matched,), {})
    assert capture(Seen()) == Scalar(1)
    reference = weakref.ref(Seen)

    del Seen
    gc.collect()
    assert reference() is None  # a matched subclass is not kept


def test_public_api_is_what_extension_authors_and_consumers_use() -> None:
    import nooa.snapshot as snapshot

    assert "Deferred" not in snapshot.__all__ and not hasattr(snapshot, "Deferred")
    for name in snapshot.__all__:
        assert getattr(snapshot, name) is not None
