# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capture(): shared budget, no user code, visibility, cycles, dataclasses and the registry."""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import pytest
from pydantic import BaseModel

from nooa import hidden, spec
from nooa.snapshot import (
    Limits,
    Mapping,
    Object,
    Opaque,
    Out,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
    capture,
    preview_attributes,
    render_json,
    render_python,
)
from tests.snapshot import _annotation_aliases


class Recorder:
    """An object that records (and refuses) every hook a careless walker would call."""

    calls: ClassVar[list[str]] = []

    def __repr__(self) -> str:
        Recorder.calls.append("repr")
        raise AssertionError("repr called")

    def __str__(self) -> str:
        Recorder.calls.append("str")
        raise AssertionError("str called")

    def __getattr__(self, name: str) -> Any:
        Recorder.calls.append(f"getattr:{name}")
        raise AssertionError("getattr called")

    def __getattribute__(self, name: str) -> Any:
        Recorder.calls.append(f"getattribute:{name}")
        return object.__getattribute__(self, name)

    def __iter__(self):
        Recorder.calls.append("iter")
        raise AssertionError("iter called")

    def __len__(self) -> int:
        Recorder.calls.append("len")
        raise AssertionError("len called")

    def __bool__(self) -> bool:
        Recorder.calls.append("bool")
        raise AssertionError("bool called")

    def __eq__(self, other: object) -> bool:
        Recorder.calls.append("eq")
        raise AssertionError("eq called")

    def __hash__(self) -> int:
        Recorder.calls.append("hash")
        raise AssertionError("hash called")

    @property
    def value(self) -> int:
        Recorder.calls.append("property")
        raise AssertionError("property called")


@pytest.fixture(autouse=True)
def _clear_recorder() -> None:
    Recorder.calls.clear()


# --- shared budget --------------------------------------------------------------


def test_complete_values_are_plain_nodes() -> None:
    snapshot = capture({"a": [1, 2.5, None, True], "b": ("x",), "c": b"raw"})

    assert snapshot == Mapping(
        "dict",
        3,
        (
            (
                "a",
                Sequence(
                    "list", 4, (Scalar(1), Scalar(2.5), Scalar(None), Scalar(True)), True, None
                ),
            ),
            ("b", Sequence("tuple", 1, (Text("x", "", 1, "chars", True, None),), True, None)),
            ("c", Scalar(b"raw")),
        ),
        True,
        None,
    )


def test_max_nodes_is_shared_by_the_whole_tree() -> None:
    value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    snapshot = capture(value, Limits(max_nodes=6))

    # root + first list + 3 items + second list = 6 inspections; then nothing more.
    assert isinstance(snapshot, Sequence)
    first, second = snapshot.items
    assert first == Sequence("list", 3, (Scalar(1), Scalar(2), Scalar(3)), True, None)
    assert second == Sequence("list", 3, (), False, "max_nodes")
    assert snapshot.length == 3 and snapshot.reason == "max_nodes"
    assert render_python(snapshot) == "list(len=3, [:2]=[[1, 2, 3], list(len=3, [:0]=[])])"


def test_max_chars_is_shared_across_siblings() -> None:
    snapshot = capture(["a" * 30, "b" * 30, "c" * 30], Limits(max_chars=50))
    text = render_json(snapshot, max_chars=50)

    parsed = json.loads(text)
    assert parsed[0] == "a" * 30
    assert parsed[1] == "b" * 12  # 50 - 2 brackets - 32 first - 2 separator - 2 quotes
    assert len(text) <= 50
    assert isinstance(snapshot, Sequence) and snapshot.reason == "max_length"


def test_max_depth_cuts_nested_containers() -> None:
    snapshot = capture({"a": {"b": {"c": [1]}}}, Limits(max_depth=2))

    assert render_python(snapshot) == "{'a': {'b': {dict: 1 items}}}"
    attributes = preview_attributes("output", snapshot)
    assert attributes["nooa.output.preview.paths"] == ("/a/b",)
    assert attributes["nooa.output.preview.reasons"] == ("max_depth",)


def test_max_items_keeps_head_and_tail() -> None:
    limits = Limits(max_items=4, tail_share=0.5)

    snapshot = capture(list(range(10)), limits)

    assert isinstance(snapshot, Sequence)
    assert [item.value for item in snapshot.items] == [0, 1]  # type: ignore[union-attr]
    assert [item.value for item in snapshot.tail] == [8, 9]  # type: ignore[union-attr]
    assert snapshot.length == 10 and snapshot.reason == "max_items"
    dict_snapshot = capture({str(i): i for i in range(10)}, limits)
    assert isinstance(dict_snapshot, Mapping)
    assert [key for key, _ in dict_snapshot.items] == ["0", "1"]
    assert [key for key, _ in dict_snapshot.tail] == ["8", "9"]


def test_max_string_splits_head_and_tail() -> None:
    snapshot = capture("abcdefghij" * 10, Limits(max_string=7, tail_share=0.5))

    assert snapshot == Text("abcd", "hij", 100, "chars", False, "max_length", omitted=93)


def test_tail_items_are_charged_to_the_budget() -> None:
    limits = Limits(max_chars=40, max_items=4, tail_share=0.5)
    snapshot = capture(["x" * 10] * 10, limits)

    assert isinstance(snapshot, Sequence)
    assert len(render_json(snapshot, max_chars=40)) <= 40
    assert snapshot.tail == () or len(snapshot.items) == 2


def test_bytes_are_cut_on_code_points() -> None:
    value = "é".encode() * 100  # 2-byte code points

    snapshot = capture(value, Limits(max_string=5))

    assert isinstance(snapshot, Text)
    assert snapshot.unit == "bytes"
    assert snapshot.head == "éé"  # 4 bytes: the fifth would split a code point
    assert snapshot.length == 200 and snapshot.omitted == 196
    assert render_python(snapshot) == "bytes(len=200, [:4]=b'\\xc3\\xa9\\xc3\\xa9')"


def test_invalid_bytes_count_replacements() -> None:
    snapshot = capture(b"ok\xff\xfe" + b"x" * 100, Limits(max_string=6))

    assert isinstance(snapshot, Text)
    assert snapshot.head == "ok\ufffd\ufffdxx"
    assert snapshot.replaced == 2


def test_invalid_limits_are_rejected() -> None:
    for bad in (
        Limits(max_items=-1),
        Limits(max_string=-1),
        Limits(tail_share=1.5),
        Limits(max_depth=-1),
    ):
        with pytest.raises(ValueError):
            capture(1, bad)


# --- no user code -------------------------------------------------------------------


def test_unknown_types_run_no_user_code() -> None:
    value = Recorder()

    snapshot = capture({"x": value, "y": [value]})

    assert isinstance(snapshot, Mapping)
    assert snapshot.items[0][1] == Opaque(__name__, "Recorder")
    assert Recorder.calls == []
    assert render_python(snapshot) == "{'x': <opaque: Recorder>, 'y': [<opaque: Recorder>]}"
    assert json.loads(render_json(snapshot, max_chars=1000)) == {
        "x": "<opaque: Recorder>",
        "y": ["<opaque: Recorder>"],
    }


def test_container_and_scalar_subclasses_are_opaque() -> None:
    class MyStr(str):
        pass

    class MyDict(dict):
        def items(self):
            raise AssertionError("items called")

    class MyInt(int):
        pass

    for value in (MyStr("x"), MyDict(a=1), MyInt(3)):
        assert isinstance(capture(value), Opaque)


def test_dataclass_properties_and_hooks_are_not_called() -> None:
    @dataclass
    class Point:
        x: int
        y: int

        @property
        def norm(self) -> int:
            raise AssertionError("property called")

        def __getattribute__(self, name: str) -> Any:
            if name in ("x", "y", "norm"):
                raise AssertionError(f"attribute hook for {name}")
            return object.__getattribute__(self, name)

    snapshot = capture(Point(1, 2))

    assert snapshot == Object("Point", (("x", Scalar(1)), ("y", Scalar(2))), True, None, __name__)


def test_custom_dict_descriptor_fails_closed() -> None:
    @dataclass
    class Sneaky:
        secret: str

    class Hooked(Sneaky):
        @property
        def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
            raise AssertionError("dict descriptor called")

    value = Hooked("s3cret")

    snapshot = capture(value)
    assert isinstance(snapshot, Object)
    assert "s3cret" not in render_python(snapshot)


# --- hidden fields --------------------------------------------------------------------


def test_pydantic_hidden_fields_are_omitted() -> None:
    class Model(BaseModel):
        public: str = "shown"
        secret: Annotated[str, hidden] = "s3cret"
        imperative: str = "also-secret"

    value = Model()
    spec(value, "imperative", hidden=True)

    snapshot = capture(value)

    assert render_python(snapshot) == "Model(public='shown')"
    assert snapshot.complete  # intentional omission is not a cut


def test_dataclass_hidden_fields_are_omitted() -> None:
    # Real annotation objects (this module's own annotations are strings).
    Config = dataclasses.make_dataclass(
        "Config",
        [
            ("name", str),
            ("token", Annotated[str, hidden]),
            ("password", Annotated[str, spec(hidden=True)]),
            ("_private", str, field(default="p")),
            ("shown_later", str, field(default="later")),
            ("no_repr", str, field(default="nr", repr=False)),
            ("CONSTANT", ClassVar[int], 5),
        ],
    )

    value = Config("n", "t0ken", "pw")
    spec(value, "shown_later", hidden=True)

    assert render_python(capture(value)) == "Config(name='n')"


def test_dataclass_class_level_spec_and_instance_override() -> None:
    @dataclass
    class Box:
        value: str = "v"

    spec(Box, "value", hidden=True)
    assert render_python(capture(Box())) == "Box()"
    shown = Box()
    spec(shown, "value", hidden=False)
    assert render_python(capture(shown)) == "Box(value='v')"


def test_dataclass_string_annotation_with_hidden_is_resolved() -> None:
    @dataclass
    class Later:  # PEP 563: the annotation is the string "Annotated[str, hidden]"
        token: Annotated[str, hidden]
        name: str = "n"

    snapshot = capture(Later("t0ken"))

    assert isinstance(snapshot, Object)
    assert [name for name, _ in snapshot.fields] == ["name"] and snapshot.complete


@pytest.mark.parametrize(
    "name", ["Literal", "AliasAnnotated", "AliasHidden", "Dotted", "SpecLiteral"]
)
def test_dataclass_string_annotation_aliases_are_hidden(name: str) -> None:
    value = getattr(_annotation_aliases, name)()

    rendered = render_json(capture(value), max_chars=2000)

    assert "S-" not in rendered
    assert json.loads(rendered)["ok"] == 1


def test_dataclass_spec_hidden_false_literal_stays_visible() -> None:
    snapshot = capture(_annotation_aliases.SpecLiteral())

    assert isinstance(snapshot, Object) and snapshot.complete
    assert [name for name, _ in snapshot.fields] == ["shown", "ok"]


@pytest.mark.parametrize("name", ["SpecUnreadable", "UnknownMarker"])
def test_dataclass_unreadable_string_marker_fails_closed_per_field(name: str) -> None:
    snapshot = capture(getattr(_annotation_aliases, name)())

    assert isinstance(snapshot, Object)
    assert [field_name for field_name, _ in snapshot.fields] == ["ok"]
    assert not snapshot.complete and snapshot.reason == "opaque"


def test_dataclass_string_annotations_without_markers_capture() -> None:
    snapshot = capture(_annotation_aliases.Visible(nested=[4]))

    assert isinstance(snapshot, Object) and snapshot.complete
    assert json.loads(render_json(snapshot, max_chars=2000)) == {
        "xs": [1, 2],
        "mapping": {},
        "forward": None,
        "constrained": 3,
        "nested": [4],
    }


def test_dataclass_inherited_string_annotation_resolves_in_its_own_module() -> None:
    @dataclass
    class Child(_annotation_aliases.AliasAnnotated):  # A is not a name in this module
        extra: str = "e"

    rendered = render_json(capture(Child()), max_chars=2000)

    assert json.loads(rendered) == {"ok": 1, "extra": "e"}


def _load_aliases_copy(name: str) -> Any:
    """A fresh copy of ``_annotation_aliases`` under ``name`` (kept out of ``sys.modules``)."""
    import importlib.util

    loader_spec = importlib.util.spec_from_file_location(name, _annotation_aliases.__file__)
    assert loader_spec is not None and loader_spec.loader is not None
    module = importlib.util.module_from_spec(loader_spec)
    sys.modules[name] = module
    try:
        loader_spec.loader.exec_module(module)
    finally:
        del sys.modules[name]  # e.g. a plugin loader that drops the module afterwards
    return module


def _assert_alias_fields_fail_closed(value: Any) -> None:
    snapshot = capture(value)

    assert "S-" not in render_json(snapshot, max_chars=2000)
    assert isinstance(snapshot, Object)
    assert [name for name, _ in snapshot.fields] == ["ok"]
    assert not snapshot.complete and snapshot.reason == "opaque"


def test_dataclass_alias_marker_in_a_module_not_in_sys_modules_fails_closed() -> None:
    module = _load_aliases_copy("_snapshot_unloaded_aliases")
    assert "_snapshot_unloaded_aliases" not in sys.modules

    _assert_alias_fields_fail_closed(module.AliasBoth())


def test_dataclass_alias_marker_with_a_reassigned_module_fails_closed() -> None:
    module = _load_aliases_copy("_snapshot_renamed_aliases")
    module.AliasBoth.__module__ = "_snapshot_no_such_module_anywhere"

    _assert_alias_fields_fail_closed(module.AliasBoth())


def test_dataclass_plain_annotations_without_a_module_stay_visible() -> None:
    module = _load_aliases_copy("_snapshot_unloaded_visible")

    snapshot = capture(module.Visible(nested=[4]))

    # ``Annotated`` itself can't be resolved without the module, so a field
    # with call metadata (``Field(gt=0)``) is unknown and fails closed.
    assert isinstance(snapshot, Object)
    assert [name for name, _ in snapshot.fields] == ["xs", "mapping", "forward", "nested"]
    assert not snapshot.complete and snapshot.reason == "opaque"


# --- cycles and repeated references -----------------------------------------------------


def test_cycles_become_refs_to_the_ancestor() -> None:
    inner: dict[str, Any] = {}
    outer = {"inner": inner}
    inner["back"] = outer

    snapshot = capture(outer)

    assert isinstance(snapshot, Mapping)
    inner_node = snapshot.items[0][1]
    assert isinstance(inner_node, Mapping)
    assert inner_node.items[0][1] == Ref("", "mapping")
    assert render_python(snapshot) == "{'inner': {'back': <cycle>}}"
    assert preview_attributes("output", snapshot)["nooa.output.preview.paths"] == ("/inner/back",)


def test_model_cycle_is_a_ref() -> None:
    @dataclass
    class Node:
        child: Any = None

    node = Node()
    node.child = node

    snapshot = capture(node)
    assert isinstance(snapshot, Object)
    assert snapshot.fields == (("child", Ref("", "object")),)


def test_repeated_references_are_expanded_each_time() -> None:
    shared = [1, 2]

    snapshot = capture({"a": shared, "b": shared, "c": (shared,)})

    assert render_python(snapshot) == "{'a': [1, 2], 'b': [1, 2], 'c': ([1, 2],)}"
    assert preview_attributes("output", snapshot)["nooa.output.preview.incomplete"] is False


# --- dataclasses ------------------------------------------------------------------------


def test_dataclass_declared_fields_in_order() -> None:
    @dataclass
    class Base:
        a: int = 1

    @dataclass
    class Child(Base):
        b: list[int] = field(default_factory=lambda: [2, 3])

    value = Child()
    object.__setattr__(value, "undeclared", "not shown")

    assert render_python(capture(value)) == "Child(a=1, b=[2, 3])"


def test_slotted_and_frozen_dataclasses() -> None:
    @dataclass(frozen=True, slots=True)
    class Frozen:
        a: int
        b: str

    assert render_python(capture(Frozen(1, "x"))) == "Frozen(a=1, b='x')"


def test_dataclass_class_object_is_opaque() -> None:
    @dataclass
    class Thing:
        a: int = 1

    assert isinstance(capture(Thing), Opaque)  # the class, not an instance


def test_dataclass_fields_obey_the_budget() -> None:
    @dataclass
    class Wide:
        a: str = "a" * 100
        b: str = "b" * 100

    snapshot = capture(Wide(), Limits(max_chars=60))
    text = render_json(snapshot, max_chars=60)

    assert len(text) <= 60
    assert isinstance(snapshot, Object) and not snapshot.complete


# --- registry ---------------------------------------------------------------------------


class Animal:
    def __init__(self, name: str) -> None:
        self.name = name


class Dog(Animal):
    pass


class Puppy(Dog):
    pass


@pytest.fixture
def registry():
    registered: list[type[Any]] = []

    def register(cls: type[Any], function: Any) -> None:
        capture.register(cls, function)
        registered.append(cls)

    yield register
    for cls in registered:
        capture.unregister(cls)


def test_registered_function_builds_through_out(registry) -> None:
    def animal(value: Animal, out: Out) -> Snapshot:
        return out.object([("name", value.name)])

    registry(Animal, animal)

    assert render_python(capture([Animal("rex")])) == "[Animal(name='rex')]"


def test_nearest_registration_in_the_mro_wins(registry) -> None:
    registry(Animal, lambda value, out: out.capture("animal"))
    registry(Dog, lambda value, out: out.capture("dog"))

    assert render_python(capture(Animal("a"))) == "'animal'"
    assert render_python(capture(Dog("d"))) == "'dog'"
    assert render_python(capture(Puppy("p"))) == "'dog'"


def test_registering_again_replaces(registry) -> None:
    registry(Dog, lambda value, out: out.capture(1))
    registry(Dog, lambda value, out: out.capture(2))

    assert capture(Dog("d")) == Scalar(2)


def test_registration_overrides_pydantic_view_for_subclass(registry) -> None:
    class Model(BaseModel):
        secret: str = "x"

    registry(Model, lambda value, out: out.object([], type_name="Redacted"))

    assert render_python(capture(Model())) == "Redacted()"


def test_exception_in_registered_function_is_unavailable(registry) -> None:
    def broken(value: Any, out: Out) -> Snapshot:
        raise RuntimeError("boom")

    registry(Animal, broken)

    snapshot = capture({"a": Animal("x")})
    assert isinstance(snapshot, Mapping)
    assert snapshot.items[0][1] == Unavailable("error", "Animal")
    attributes = preview_attributes("output", snapshot)
    assert attributes["nooa.output.preview.reasons"] == ("error",)


def test_node_not_built_by_out_is_rejected(registry) -> None:
    registry(Animal, lambda value, out: Scalar("x" * 10_000))  # type: ignore[arg-type]

    assert capture(Animal("a")) == Unavailable("error", "Animal")


def test_budget_is_enforced_through_out(registry) -> None:
    pulled: list[int] = []

    def endless():
        index = 0
        while True:
            pulled.append(index)
            yield "x" * 20
            index += 1

    registry(Animal, lambda value, out: out.sequence(endless()))

    snapshot = capture(Animal("a"), Limits(max_chars=100, max_nodes=50))
    text = render_json(snapshot, max_chars=100)

    assert len(text) <= 100
    assert isinstance(snapshot, Sequence) and not snapshot.complete
    assert snapshot.length is None
    assert len(pulled) < 10  # stopped at the budget, the rest never pulled


def test_out_sequence_obeys_max_items_and_nodes(registry) -> None:
    registry(Animal, lambda value, out: out.sequence(range(1_000_000), length=1_000_000))

    snapshot = capture(Animal("a"), Limits(max_items=3))
    assert isinstance(snapshot, Sequence)
    assert render_python(snapshot) == "list(len=1000000, [:3]=[0, 1, 2])"

    starved = capture(Animal("a"), Limits(max_nodes=5))
    assert isinstance(starved, Sequence) and starved.reason == "max_nodes"


def test_out_mapping_and_nested_values(registry) -> None:
    registry(Animal, lambda value, out: out.mapping([("name", value.name), ("tags", ["a"])]))

    assert render_python(capture(Animal("rex"))) == "{'name': 'rex', 'tags': ['a']}"


def test_registered_self_reference_is_a_cycle(registry) -> None:
    registry(Animal, lambda value, out: out.object([("me", value)]))

    snapshot = capture(Animal("a"))
    assert isinstance(snapshot, Object)
    assert snapshot.fields == (("me", Ref("", "object")),)


class _Node:
    def __init__(self, left: Any = None) -> None:
        self.left = left


def _node_depth(node: Any) -> int:
    children = [child for _, child in getattr(node, "fields", ())]
    children += list(getattr(node, "items", ()))
    children = [child[1] if isinstance(child, tuple) else child for child in children]
    return max([0] + [1 + _node_depth(child) for child in children])


def test_registered_recursive_type_respects_max_depth(registry) -> None:
    registry(_Node, lambda n, out: out.object({"left": out.capture(n.left)}, type_name="Node"))
    chain = None
    for _ in range(30):
        chain = _Node(chain)

    snapshot = capture(chain, Limits(max_depth=3))

    assert _node_depth(snapshot) == 3
    deepest = snapshot
    for _ in range(3):
        assert isinstance(deepest, Object)
        deepest = deepest.fields[0][1]
    assert isinstance(deepest, Object)
    assert deepest.fields == () and deepest.reason == "max_depth"


def test_registered_parts_in_sequences_and_mappings_get_their_slot(registry) -> None:
    registry(
        _Node,
        lambda n, out: out.object(
            {
                "items": out.sequence([out.capture(n.left)]),
                "by_key": out.mapping([("k", out.capture(n.left))]),
            },
            type_name="Node",
        ),
    )
    chain = None
    for _ in range(10):
        chain = _Node(chain)

    snapshot = capture(chain, Limits(max_depth=4))

    assert _node_depth(snapshot) <= 4
    assert isinstance(snapshot, Object)
    items = snapshot.fields[0][1]
    assert isinstance(items, Sequence) and isinstance(items.items[0], Object)


def test_registered_cycle_ref_ids_use_field_paths(registry) -> None:
    registry(_Node, lambda n, out: out.object({"left": out.capture(n.left)}, type_name="Node"))
    a = _Node()
    b = _Node(a)
    a.left = b
    holder = _Node(a)

    snapshot = capture(holder)

    node = snapshot
    for _ in range(3):
        assert isinstance(node, Object)
        node = node.fields[0][1]
    assert node == Ref("/left", "object")


def test_register_needs_a_type() -> None:
    with pytest.raises(TypeError):
        capture.register("Animal")  # type: ignore[arg-type]


def test_out_opaque(registry) -> None:
    registry(Animal, lambda value, out: out.opaque())

    assert capture(Animal("a")) == Opaque(__name__, "Animal")


# --- review findings: untrusted equality ------------------------------------


def test_unknown_field_type_metaclass_eq_never_runs() -> None:
    calls: list[str] = []

    class Meta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append("metaclass __eq__")
            raise RuntimeError("untrusted metaclass equality executed")

        __hash__ = type.__hash__

    class Unknown(metaclass=Meta):
        pass

    @dataclass
    class Holder:
        value: object

    snapshot = capture(Holder(Unknown()))

    assert calls == []
    assert isinstance(snapshot, Object)
    assert isinstance(snapshot.field("value"), Opaque)


class _CollidingKey:
    """A namespace key whose hash collides with a name capture looks up."""

    calls: ClassVar[list[object]] = []
    armed: ClassVar[bool] = True

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not _CollidingKey.armed:
            return False  # setting the test up
        _CollidingKey.calls.append(other)
        raise RuntimeError("untrusted __eq__ executed")


def test_class_namespace_foreign_key_is_never_compared() -> None:
    _CollidingKey.calls.clear()
    with pytest.warns(RuntimeWarning):
        unknown = type("Unknown", (), {_CollidingKey("__dataclass_fields__"): 1})

    snapshot = capture(unknown())

    assert _CollidingKey.calls == []
    assert isinstance(snapshot, Opaque)  # an unreadable namespace fails closed


def test_safe_dict_survives_a_namespace_changing_during_the_key_check() -> None:
    from nooa.snapshot import _capture

    namespace = {f"k{index}": index for index in range(8)}
    code = _capture._safe_dict.__code__
    mutations: list[int] = []

    def local(frame: Any, event: str, arg: Any) -> Any:
        # Like an import in another thread: the dict grows while the key loop
        # runs (``key`` is bound once the loop has started).
        if event == "line" and "key" in frame.f_locals and len(mutations) < 4:
            namespace[f"added{len(mutations)}"] = 0
            mutations.append(1)
        return local

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        return local if event == "call" and frame.f_code is code else None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        checked = _capture._safe_dict(
            namespace, namespace, _capture._State(_capture.DEFAULT_LIMITS)
        )
    finally:
        sys.settrace(previous)

    assert len(mutations) == 4  # the dict did change while ``_safe_dict`` ran
    assert checked is not None
    assert {f"k{index}" for index in range(8)} <= set(checked)


def test_module_namespace_foreign_key_is_never_compared(monkeypatch) -> None:
    import types

    module = types.ModuleType("_snapshot_hostile_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Secret:
        __module__ = module.__name__
        ok: int = 1
        token: Annotated[str, hidden] = "s3cret"  # a string annotation (PEP 563)

    Secret = dataclass(Secret)  # noqa: N806
    _CollidingKey.armed = False
    try:
        # First, so that a lookup of "Annotated" meets it before the real entry.
        module.__dict__[_CollidingKey("Annotated")] = 1  # type: ignore[index]
        module.__dict__["Annotated"] = Annotated
    finally:
        _CollidingKey.armed = True
    module.__dict__["hidden"] = hidden
    _CollidingKey.calls.clear()

    snapshot = capture(Secret())

    assert _CollidingKey.calls == []
    assert isinstance(snapshot, Object)
    assert snapshot.field("ok") == Scalar(1)
    assert snapshot.field("token") is None  # can't be read safely: omitted
    assert "s3cret" not in render_json(snapshot, max_chars=1000)


# --- review findings: caps


def test_bytes_capture_omits_a_code_point_beyond_max_string() -> None:
    value = "\U0001f600x".encode()

    snapshot = capture(value, Limits(max_string=1))

    assert isinstance(snapshot, Text)
    assert len((snapshot.head + snapshot.tail).encode()) <= 1
    assert snapshot.omitted == len(value) and not snapshot.complete
    two_byte = capture("\u00e9\u00e9".encode(), Limits(max_string=3))
    assert isinstance(two_byte, Text) and two_byte.head == "\u00e9" and two_byte.omitted == 2


def test_object_fields_obey_max_items() -> None:
    @dataclass
    class Data:
        a: int = 1
        b: int = 2
        c: int = 3

    class Model(BaseModel):
        a: int = 1
        b: int = 2

    cut = capture(Data(), Limits(max_items=1))
    assert isinstance(cut, Object)
    assert cut.fields == (("a", Scalar(1)),)
    assert not cut.complete and cut.reason == "max_items"
    model = capture(Model(), Limits(max_items=1))
    assert isinstance(model, Object) and [name for name, _ in model.fields] == ["a"]
    assert model.reason == "max_items"
    exact = capture(Data(), Limits(max_items=3))
    assert isinstance(exact, Object) and exact.complete and len(exact.fields) == 3
