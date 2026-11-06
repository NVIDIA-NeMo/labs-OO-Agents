# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cycle detection and the fixed snapshot depth limit (follow-up to #534)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from nooa.errors.storage import DeserializationError, SerializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import MAX_SNAPSHOT_DEPTH, deserialize, serialize


class _Node(BaseModel):
    """Model whose fields can form a cycle or a deep chain."""

    name: str = "node"
    child: Any = None


class _Extra(BaseModel):
    """Model that keeps unknown fields in ``model_extra``."""

    model_config = ConfigDict(extra="allow")


@dataclass
class _Data:
    """Dataclass that can form a cycle or a deep chain."""

    child: Any = None


@snapshotable
class _Object:
    """Opted-in object that can form a cycle or a deep chain."""

    def __init__(self, child: Any = None):
        """Store the child."""
        self.child = child


# -- cycles -----------------------------------------------------------------


def _list_cycle():
    box = []
    value = {"box": [box]}
    box.append(value)
    return value


def _dict_cycle():
    value = {"a": {"b": {}}}
    value["a"]["b"]["back"] = value["a"]
    return value


def _nested_model_cycle():
    outer = _Node(name="outer", child=_Node(name="inner"))
    outer.child.child = outer
    return {"m": outer}


def _dataclass_cycle():
    value = _Data(child=[_Data()])
    value.child[0].child = value
    return value


def _snapshotable_cycle():
    value = _Object(child={"k": _Object()})
    value.child["k"].child = value
    return value


def _tuple_cycle():
    inner = []
    value = (1, inner)
    inner.append(value)
    return value


def _extra_cycle():
    value = _Extra()
    value.loop = [value]
    return value


@pytest.mark.parametrize(
    ("make", "where", "back_to"),
    [
        (_list_cycle, "<root>['box'][0][0]", "<root> (dict)"),
        (_dict_cycle, "<root>['a']['b']['back']", "<root>['a'] (dict)"),
        (_nested_model_cycle, "<root>['m'].child.child", "<root>['m'] (_Node)"),
        (_dataclass_cycle, "<root>.child[0].child", "<root> (_Data)"),
        (_snapshotable_cycle, "<root>.child['k'].child", "<root> (_Object)"),
        (_tuple_cycle, "<root>[1][0]", "<root> (tuple)"),
        (_extra_cycle, "<root>.loop[0]", "<root> (_Extra)"),
    ],
    ids=["list", "dict", "nested-model", "dataclass", "snapshotable", "tuple", "model-extra"],
)
def test_cycle_raises_with_path(make, where, back_to):
    """Each recursive branch reports the cycle at once, naming both ends."""
    with pytest.raises(SerializationError) as caught:
        serialize(make())
    message = str(caught.value)
    assert message == f"Snapshot value contains a cycle: {where} refers back to {back_to}"
    assert caught.value.__cause__ is None


def test_late_self_reference_fails_fast():
    """A cycle after many entries fails in linear time (was seconds on main)."""
    value = {f"k{i}": {"x": [i]} for i in range(2000)}
    value["self"] = value
    start = time.perf_counter()
    with pytest.raises(SerializationError, match=r"<root>\['self'\] refers back to <root>"):
        serialize(value)
    assert time.perf_counter() - start < 0.5


# -- shared references --------------------------------------------------------


def test_diamond_shared_references_roundtrip():
    """The same object reached twice without a cycle is serialized twice."""
    leaf = {"labels": ["same", "object"]}
    left = _Node(name="left", child=leaf)
    right = _Data(child=leaf)
    value = {"left": left, "right": right, "both": [leaf, leaf], "pair": (leaf, leaf)}
    blob, allowlist = serialize(value)
    restored = deserialize(blob, allowlist)
    assert restored == value


def test_shared_reference_at_two_depths_is_not_a_cycle():
    """An ancestor's sibling can be reached again further down."""
    shared = [1, 2]
    value = {"a": shared, "b": {"c": shared, "d": [shared]}}
    blob, allowlist = serialize(value)
    assert deserialize(blob, allowlist) == value


# -- depth limit -------------------------------------------------------------

_WRAP = {
    "list": lambda c: [c],
    "dict": lambda c: {"c": c},
    "tuple": lambda c: (c,),
    "pydantic": lambda c: _Node(child=c),
    "dataclass": lambda c: _Data(child=c),
    "snapshotable": lambda c: _Object(child=c),
}


def _nest(kind: str, levels: int) -> Any:
    """Build ``levels`` nested containers of one kind."""
    value: Any = None
    for _ in range(levels):
        value = _WRAP[kind](value)
    return value


def _at_stack_depth(frames: int, fn):
    """Call ``fn`` from ``frames`` extra Python frames deep."""
    if frames == 0:
        return fn()
    return _at_stack_depth(frames - 1, fn)


def _roundtrip(value):
    blob, allowlist = serialize(value)
    return deserialize(blob, allowlist)


def _same_shape(kind: str, restored: Any, levels: int) -> None:
    for _ in range(levels):
        assert type(restored) is type(_WRAP[kind](None))
        restored = (
            restored[0]
            if kind in ("list", "tuple")
            else restored["c"]
            if kind == "dict"
            else restored.child
        )
    assert restored is None


@pytest.mark.parametrize("caller_frames", [0, 200])
@pytest.mark.parametrize("kind", list(_WRAP))
def test_depth_limit_is_fixed(kind, caller_frames):
    """The limit roundtrips and one more level fails, however deep the caller is."""
    restored = _at_stack_depth(caller_frames, lambda: _roundtrip(_nest(kind, MAX_SNAPSHOT_DEPTH)))
    _same_shape(kind, restored, MAX_SNAPSHOT_DEPTH)

    too_deep = _nest(kind, MAX_SNAPSHOT_DEPTH + 1)
    with pytest.raises(SerializationError) as caught:
        _at_stack_depth(caller_frames, lambda: serialize(too_deep))
    assert f"deeper than the limit of {MAX_SNAPSHOT_DEPTH} levels" in str(caught.value)
    assert "cycle" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_depth_error_names_abbreviated_path():
    """The message shows where the limit was hit without printing 200 segments."""
    with pytest.raises(SerializationError) as caught:
        serialize({"top": _nest("list", MAX_SNAPSHOT_DEPTH)})
    message = str(caught.value)
    assert "at <root>['top'][0][0]" in message
    assert "more)..." in message
    assert len(message) < 200


def _deep_blob(levels: int) -> Any:
    blob: Any = None
    for _ in range(levels):
        blob = [blob]
    return blob


@pytest.mark.parametrize("caller_frames", [0, 200])
def test_deserialize_over_limit_raises(caller_frames):
    """A blob deeper than the limit (e.g. hand-edited) is rejected, not a RecursionError."""
    assert _at_stack_depth(
        caller_frames, lambda: deserialize(_deep_blob(MAX_SNAPSHOT_DEPTH), set())
    ) == _deep_blob(MAX_SNAPSHOT_DEPTH)
    with pytest.raises(
        DeserializationError, match=f"deeper than the limit of {MAX_SNAPSHOT_DEPTH}"
    ):
        _at_stack_depth(
            caller_frames, lambda: deserialize(_deep_blob(MAX_SNAPSHOT_DEPTH + 1), set())
        )


def test_deserialize_far_over_limit_raises():
    """Even far beyond Python's recursion limit the error is the documented one."""
    with pytest.raises(DeserializationError, match="deeper than the limit"):
        deserialize(_deep_blob(5000), set())


def test_envelope_counts_as_one_level_on_restore():
    """A typed envelope (two JSON dicts) is one level, so save and restore agree."""
    blob, allowlist = serialize(_nest("pydantic", MAX_SNAPSHOT_DEPTH))
    assert deserialize(blob, allowlist).child is not None


# -- other recursion ----------------------------------------------------------


class _RecursiveSnapshotData(BaseModel):
    """A buggy ``__snapshot_data__`` that recurses forever."""

    def __snapshot_data__(self) -> dict:
        return self.__snapshot_data__()


def test_unrelated_recursion_error_is_not_called_a_cycle():
    """A RecursionError from a bug keeps the fallback, with an honest message."""
    with pytest.raises(SerializationError) as caught:
        serialize({"bad": _RecursiveSnapshotData()})
    message = str(caught.value)
    assert "recursion limit" in message
    assert "not a detected cycle" in message
    assert isinstance(caught.value.__cause__, RecursionError)


def _recurse():
    return _recurse()


class _RestoreHook(BaseModel):
    """A durable model whose restore hook recurses forever."""

    def __snapshot_data__(self) -> dict:
        return {}

    @classmethod
    def __restore_snapshot__(cls, data: dict) -> Any:
        return _recurse()


def test_deserialize_recursion_error_becomes_deserialization_error():
    """A RecursionError raised while restoring (e.g. in a hook) is wrapped too."""
    blob, allowlist = serialize(_RestoreHook())
    with pytest.raises(DeserializationError, match="recursion limit") as caught:
        deserialize(blob, allowlist)
    assert isinstance(caught.value.__cause__, RecursionError)


# -- __snapshot_data__ output -------------------------------------------------


class _Durable(BaseModel):
    """Model that owns its snapshot format, like ``LLMResponse``."""

    payload: Any = None

    def __snapshot_data__(self) -> dict:
        return {"payload": self.payload}


def _json_chain(kind: str, levels: int) -> Any:
    """``levels`` JSON containers of one kind, as ``__snapshot_data__`` might return."""
    blob: Any = None
    for _ in range(levels):
        if kind == "dict":
            blob = {"c": blob}
        elif kind == "list":
            blob = [blob]
        else:  # a hand-built tuple envelope: one level on restore
            blob = {"__type__": "tuple", "data": [blob]}
    return blob


def test_snapshot_data_over_limit_fails_on_save():
    """Deep ``__snapshot_data__`` output fails on save with a path, not later on restore."""
    # The envelope is level 0, so the payload may hold MAX_SNAPSHOT_DEPTH - 1 containers.
    fits = _Durable(payload=_json_chain("dict", MAX_SNAPSHOT_DEPTH - 1))
    blob, allowlist = serialize(fits)
    assert deserialize(blob, allowlist).payload == fits.payload

    with pytest.raises(SerializationError) as caught:
        serialize(_Durable(payload=_json_chain("dict", MAX_SNAPSHOT_DEPTH)))
    message = str(caught.value)
    assert f"deeper than the limit of {MAX_SNAPSHOT_DEPTH} levels" in message
    assert "at <root>.payload['c']['c']" in message
    assert "cycle" not in message


@pytest.mark.parametrize("kind", ["dict", "list", "tuple-envelope"])
@pytest.mark.parametrize("outer", [0, 3])
def test_snapshot_data_saves_iff_it_restores(kind, outer):
    """Save applies restore's accounting to ``__snapshot_data__`` output exactly."""
    # A hand-built tuple envelope in ``__snapshot_data__`` is data, not a type:
    # it is escaped as a plain dict (holding a list), so each level is two
    # containers on save and on restore.
    per_level = 2 if kind == "tuple-envelope" else 1
    # Exactly the levels that fit: the envelope plus ``outer`` lists sit above.
    limit = (MAX_SNAPSHOT_DEPTH - outer - 1) // per_level
    results = {}
    for levels in range(limit - 2, limit + 3):
        value: Any = _Durable(payload=_json_chain(kind, levels))
        for _ in range(outer):
            value = [value]
        try:
            blob, allowlist = serialize(value)
        except SerializationError:
            saved = False
        else:
            saved = True
            restored = deserialize(blob, allowlist)  # must not raise
            for _ in range(outer):
                restored = restored[0]
            assert restored.payload == _json_chain(kind, levels)
        results[levels] = saved
    assert results == {n: n <= limit for n in results}


# -- review follow-ups (#540) -------------------------------------------------


@pytest.mark.parametrize(
    ("value", "reason", "fragment"),
    [
        ([[[object()]]], "type", "type 'object' at <root>[0][0][0]"),
        ([[{1: 2}]], "key", "Dict key 1 (type: int) at <root>[0][0] is not a string"),
        ({"a": _Data(child={"b": {(1, 2): 3}})}, "key", "at <root>['a'].child['b']"),
    ],
)
def test_type_and_key_errors_name_the_path(value, reason, fragment):
    with pytest.raises(SerializationError) as info:
        serialize(value)
    assert fragment in str(info.value)
    assert info.value.reason == reason


def test_cycle_and_depth_errors_carry_reason():
    with pytest.raises(SerializationError) as cycle:
        serialize(_list_cycle())
    assert cycle.value.reason == "cycle"
    with pytest.raises(SerializationError) as deep:
        serialize(_deep_blob(MAX_SNAPSHOT_DEPTH + 1))
    assert deep.value.reason == "depth"
    with pytest.raises(DeserializationError) as restore:
        deserialize(_deep_blob(MAX_SNAPSHOT_DEPTH + 1), set())
    assert restore.value.reason == "depth"


def test_restore_skips_too_deep_attribute_and_warns(caplog):
    import logging
    from types import SimpleNamespace

    from nooa.storage.snapshot import AgentSnapshot

    snapshot = AgentSnapshot(
        attributes={"memory": ["a", "b"], "deep": _deep_blob(MAX_SNAPSHOT_DEPTH + 1), "n": 3}
    )
    agent = SimpleNamespace()
    with caplog.at_level(logging.WARNING, logger="nooa.storage.snapshot"):
        snapshot.restore(agent)
    assert agent.memory == ["a", "b"]
    assert agent.n == 3
    assert not hasattr(agent, "deep")
    (message,) = [r.getMessage() for r in caplog.records if "'deep'" in r.getMessage()]
    assert "skipping non-restorable attribute" in message
    assert "nests deeper than the limit" in message


def test_primitive_subclass_is_stored_as_plain_value():
    """Primitive subclasses pass through; a nosnapshot marker on them is not honoured."""

    class Marked(str):
        __nosnapshot__ = True

    blob, _ = serialize({"s": Marked("x")})
    assert blob == {"s": "x"}


@pytest.mark.parametrize("error", [TypeError, AttributeError, RuntimeError])
def test_restore_skips_any_unrestorable_attribute(caplog, monkeypatch, error):
    """Any failure restoring one attribute skips it; the others still load."""
    import logging
    from types import SimpleNamespace

    from nooa.storage import snapshot as snapshot_module
    from nooa.storage.snapshot import AgentSnapshot

    real = snapshot_module.deserialize

    def failing(blob, *args, **kwargs):
        if blob == "BAD":
            raise error("schema drift")
        return real(blob, *args, **kwargs)

    monkeypatch.setattr(snapshot_module, "deserialize", failing)
    snapshot = AgentSnapshot(attributes={"memory": ["a"], "broken": "BAD", "n": 3})
    agent = SimpleNamespace()
    with caplog.at_level(logging.WARNING, logger="nooa.storage.snapshot"):
        snapshot.restore(agent)
    assert agent.memory == ["a"]
    assert agent.n == 3
    assert not hasattr(agent, "broken")
    (record,) = [r for r in caplog.records if "'broken'" in r.getMessage()]
    assert "skipping non-restorable attribute" in record.getMessage()
    assert record.exc_info is not None and record.exc_info[0] is error  # traceback kept
