# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extension registry: ``register`` / ``register_lazy`` dispatch, its cache and its safety."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from nooa.snapshot import (
    Object,
    Opaque,
    Out,
    Scalar,
    Text,
    capture,
    render_python,
)
from tests.snapshot._extend_support import (
    FAKE_MODULE,
    _named,
)

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
    def gadget(value: Any, out: Out):
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
    assert render_python(capture(LazyLeaf())) == "'lazy-leaf'"  # a new registration applies


def test_eager_wins_over_lazy_at_the_same_class(lazy, eager) -> None:
    Both = _named("Both")
    lazy(FAKE_MODULE, "Both", lambda value, out: out.capture("lazy"))
    eager(Both, lambda value, out: out.capture("eager"))

    assert render_python(capture(Both())) == "'eager'"


def test_lazy_registration_does_not_retain_matched_classes(lazy) -> None:
    import gc
    import weakref

    lazy(FAKE_MODULE, "Matched", lambda value, out: out.capture(1))
    Matched = _named("Matched")
    assert capture(Matched()) == Scalar(1)
    assert capture(Matched()) == Scalar(1)  # matched again, by its current name
    reference = weakref.ref(Matched)

    del Matched
    gc.collect()
    assert reference() is None


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


# --- review findings: lazy renames


def test_lazy_dispatch_follows_a_class_rename(lazy) -> None:
    class Target:
        pass

    Target.__module__ = "before_rename"
    Target.__qualname__ = "Target"
    lazy("after_rename", "Target", lambda value, out: out.object({"ok": True}))

    assert isinstance(capture(Target()), Opaque)  # caches the miss
    Target.__module__ = "after_rename"
    renamed = capture(Target())
    assert isinstance(renamed, Object) and renamed.fields == (("ok", Scalar(True)),)
    Target.__qualname__ = "Other"
    assert isinstance(capture(Target()), Opaque)  # a cached hit is not kept either


def test_public_api_is_what_extension_authors_and_consumers_use() -> None:
    import nooa.snapshot as snapshot

    assert "Deferred" not in snapshot.__all__ and not hasattr(snapshot, "Deferred")
    for name in snapshot.__all__:
        assert getattr(snapshot, name) is not None
