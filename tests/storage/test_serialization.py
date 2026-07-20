# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for storage/serialization.py — serialize() / deserialize() dispatch.

Covers every dispatch path, roundtrips, lenient restoration, and allowlist
rejection as specified in the Phase 1 design.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from nooa.errors.storage import DeserializationError, SerializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import SKIP, deserialize, serialize

# ---------------------------------------------------------------------------
# Fixture types
# ---------------------------------------------------------------------------


class MyModel(BaseModel):
    name: str
    value: int = 0


class NestedModel(BaseModel):
    inner: MyModel
    tags: list[str] = []


@dataclasses.dataclass
class Point:
    x: float
    y: float


@dataclasses.dataclass
class Line:
    start: Point
    end: Point


@snapshotable
class Config:
    def __init__(self, host: str, port: int = 8080):
        self.host = host
        self.port = port


@snapshotable
class Nested:
    """A snapshotable class that contains another snapshotable."""

    def __init__(self, config: Config, label: str = "default"):
        self.config = config
        self.label = label


class _NoSnapshotThing:
    __nosnapshot__ = True


class UnsupportedThing:
    """A plain class with no serialization support."""

    pass


@dataclasses.dataclass
class PointWithDefault:
    a: int
    b: int = 99


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class TestPrimitives:
    """serialize() returns JSON primitives as-is; deserialize() roundtrips."""

    @pytest.mark.parametrize(
        "value",
        [None, True, False, 0, 42, -1, 3.14, "", "hello"],
        ids=lambda v: repr(v),
    )
    def test_primitive_roundtrip(self, value: Any):
        blob, allowlist = serialize(value)
        assert blob == value
        assert allowlist == set()
        assert deserialize(blob, allowlist) == value

    def test_int_zero_is_not_skipped(self):
        """Ensure 0 isn't confused with a falsy sentinel."""
        blob, _ = serialize(0)
        assert blob == 0


# ---------------------------------------------------------------------------
# Collections (list, dict, tuple)
# ---------------------------------------------------------------------------


class TestCollections:
    def test_list_of_primitives(self):
        blob, al = serialize([1, "two", None])
        assert blob == [1, "two", None]
        assert al == set()
        assert deserialize(blob, al) == [1, "two", None]

    def test_dict_of_primitives(self):
        blob, al = serialize({"a": 1, "b": True})
        assert blob == {"a": 1, "b": True}
        assert al == set()
        assert deserialize(blob, al) == {"a": 1, "b": True}

    def test_tuple_roundtrip_preserves_type(self):
        """Tuples survive roundtrip as tuples via a type envelope."""
        blob, al = serialize((1, 2, 3))
        assert blob["__type__"] == "tuple"
        assert blob["data"] == [1, 2, 3]
        restored = deserialize(blob, al)
        assert restored == (1, 2, 3)
        assert type(restored) is tuple

    def test_tuple_with_nested_objects(self):
        """Tuples containing typed objects preserve both the tuple and object types."""
        value = (Point(x=1.0, y=2.0), MyModel(name="t", value=3))
        blob, al = serialize(value)
        restored = deserialize(blob, al)
        assert type(restored) is tuple
        assert isinstance(restored[0], Point)
        assert isinstance(restored[1], MyModel)

    def test_nested_collection(self):
        value = {"items": [1, {"nested": True}], "count": 2}
        blob, al = serialize(value)
        assert deserialize(blob, al) == value


# ---------------------------------------------------------------------------
# nosnapshot
# ---------------------------------------------------------------------------


class TestNoSnapshot:
    def test_nosnapshot_value_returns_SKIP(self):
        thing = _NoSnapshotThing()
        blob, al = serialize(thing)
        assert blob is SKIP
        assert al == set()

    def test_SKIP_sentinel_identity(self):
        """SKIP is a unique sentinel — not None, not False."""
        assert SKIP is not None
        assert SKIP is not False
        assert repr(SKIP) == "SKIP"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestPydantic:
    def test_simple_model_roundtrip(self):
        m = MyModel(name="test", value=42)
        blob, al = serialize(m)
        assert blob["__type__"] == "pydantic"
        assert blob["__class__"] == f"{MyModel.__module__}.MyModel"
        assert blob["data"] == {"name": "test", "value": 42}
        restored = deserialize(blob, al)
        assert isinstance(restored, MyModel)
        assert restored.name == "test"
        assert restored.value == 42

    def test_nested_pydantic_roundtrip(self):
        m = NestedModel(inner=MyModel(name="inner", value=1), tags=["a", "b"])
        blob, al = serialize(m)
        restored = deserialize(blob, al)
        assert isinstance(restored, NestedModel)
        assert isinstance(restored.inner, MyModel)
        assert restored.inner.name == "inner"
        assert restored.tags == ["a", "b"]

    def test_pydantic_in_list(self):
        items = [MyModel(name="a", value=1), MyModel(name="b", value=2)]
        blob, al = serialize(items)
        restored = deserialize(blob, al)
        assert len(restored) == 2
        assert all(isinstance(r, MyModel) for r in restored)

    def test_pydantic_schema_drift_tolerant(self):
        """Pydantic model_validate handles extra/missing fields gracefully."""
        blob = {
            "__type__": "pydantic",
            "__class__": f"{MyModel.__module__}.MyModel",
            "data": {"name": "old", "value": 5, "extra_field": "ignored"},
        }
        fqn = f"{MyModel.__module__}.MyModel"
        restored = deserialize(blob, {fqn})
        assert isinstance(restored, MyModel)
        assert restored.name == "old"
        assert restored.value == 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_simple_dataclass_roundtrip(self):
        p = Point(x=1.0, y=2.0)
        blob, al = serialize(p)
        assert blob["__type__"] == "dataclass"
        assert blob["__class__"] == f"{Point.__module__}.Point"
        assert blob["data"] == {"x": 1.0, "y": 2.0}
        restored = deserialize(blob, al)
        assert isinstance(restored, Point)
        assert restored.x == 1.0
        assert restored.y == 2.0

    def test_nested_dataclass_roundtrip(self):
        line = Line(start=Point(0, 0), end=Point(1, 1))
        blob, al = serialize(line)
        restored = deserialize(blob, al)
        assert isinstance(restored, Line)
        assert isinstance(restored.start, Point)
        assert restored.start.x == 0
        assert restored.end.y == 1

    def test_dataclass_lenient_extra_keys(self):
        """Deserialization ignores extra keys not accepted by __init__."""
        blob = {
            "__type__": "dataclass",
            "__class__": f"{Point.__module__}.Point",
            "data": {"x": 1.0, "y": 2.0, "z": 3.0},
        }
        fqn = f"{Point.__module__}.Point"
        restored = deserialize(blob, {fqn})
        assert isinstance(restored, Point)
        assert restored.x == 1.0
        assert restored.y == 2.0

    def test_dataclass_lenient_missing_keys_use_defaults(self):
        """Deserialization lets defaults fill missing keys."""
        blob = {
            "__type__": "dataclass",
            "__class__": f"{PointWithDefault.__module__}.PointWithDefault",
            "data": {"a": 1},
        }
        fqn = f"{PointWithDefault.__module__}.PointWithDefault"
        restored = deserialize(blob, {fqn})
        assert restored.a == 1
        assert restored.b == 99


# ---------------------------------------------------------------------------
# @snapshotable classes
# ---------------------------------------------------------------------------


class TestSnapshotable:
    def test_simple_roundtrip(self):
        cfg = Config(host="localhost", port=9090)
        blob, al = serialize(cfg)
        assert blob["__type__"] == "dict_class"
        assert blob["__class__"] == f"{Config.__module__}.Config"
        assert blob["data"] == {"host": "localhost", "port": 9090}
        restored = deserialize(blob, al)
        assert isinstance(restored, Config)
        assert restored.host == "localhost"
        assert restored.port == 9090

    def test_nested_snapshotable_roundtrip(self):
        nested = Nested(config=Config("example.com", 443), label="prod")
        blob, al = serialize(nested)
        restored = deserialize(blob, al)
        assert isinstance(restored, Nested)
        assert isinstance(restored.config, Config)
        assert restored.config.host == "example.com"
        assert restored.config.port == 443
        assert restored.label == "prod"

    def test_snapshotable_lenient_extra_keys(self):
        """Deserialization preserves extra keys via setattr for @snapshotable."""
        blob = {
            "__type__": "dict_class",
            "__class__": f"{Config.__module__}.Config",
            "data": {"host": "localhost", "port": 80, "extra": "preserved"},
        }
        fqn = f"{Config.__module__}.Config"
        restored = deserialize(blob, {fqn})
        assert isinstance(restored, Config)
        assert restored.host == "localhost"
        assert restored.port == 80
        assert hasattr(restored, "extra")
        assert restored.extra == "preserved"

    def test_snapshotable_mixed_in_collection(self):
        """@snapshotable inside a list/dict."""
        data = {"configs": [Config("a", 1), Config("b", 2)]}
        blob, al = serialize(data)
        restored = deserialize(blob, al)
        assert len(restored["configs"]) == 2
        assert all(isinstance(c, Config) for c in restored["configs"])


# ---------------------------------------------------------------------------
# Allowlist security
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_deserialization_rejects_unknown_class(self):
        blob = {
            "__type__": "pydantic",
            "__class__": "some.unknown.Module.ClassName",
            "data": {},
        }
        with pytest.raises(DeserializationError, match="not in the allowlist"):
            deserialize(blob, set())

    def test_allowlist_populated_during_serialize(self):
        m = MyModel(name="x", value=1)
        _, al = serialize(m)
        assert f"{MyModel.__module__}.MyModel" in al

    def test_allowlist_contains_all_types(self):
        """serialize() collects FQNs from Pydantic, dataclass, and @snapshotable."""
        data = {
            "model": MyModel(name="a", value=1),
            "point": Point(x=0, y=0),
            "config": Config("h", 80),
        }
        _, al = serialize(data)
        assert f"{MyModel.__module__}.MyModel" in al
        assert f"{Point.__module__}.Point" in al
        assert f"{Config.__module__}.Config" in al

    def test_nested_types_in_allowlist(self):
        """Nested enveloped types are all captured in the allowlist."""
        line = Line(start=Point(0, 0), end=Point(1, 1))
        _, al = serialize(line)
        assert f"{Line.__module__}.Line" in al
        assert f"{Point.__module__}.Point" in al


# ---------------------------------------------------------------------------
# Error: unknown type
# ---------------------------------------------------------------------------


class TestUnsupportedType:
    def test_serialize_raises_for_unknown_type(self):
        with pytest.raises(SerializationError, match="@snapshotable"):
            serialize(UnsupportedThing())

    def test_error_message_mentions_options(self):
        """Error message tells the developer their options."""
        with pytest.raises(SerializationError) as exc_info:
            serialize(UnsupportedThing())
        msg = str(exc_info.value)
        assert "nosnapshot" in msg
        assert "Pydantic" in msg or "pydantic" in msg.lower()

    def test_unsupported_nested_in_list(self):
        with pytest.raises(SerializationError):
            serialize([1, UnsupportedThing()])

    def test_unsupported_nested_in_dict(self):
        with pytest.raises(SerializationError):
            serialize({"key": UnsupportedThing()})


# ---------------------------------------------------------------------------
# Mixed types in a single structure
# ---------------------------------------------------------------------------


class TestMixed:
    def test_dict_with_mixed_values(self):
        data = {
            "name": "test",
            "count": 42,
            "model": MyModel(name="m", value=1),
            "point": Point(x=1, y=2),
            "config": Config("h", 80),
            "items": [1, MyModel(name="i", value=2)],
        }
        blob, al = serialize(data)
        restored = deserialize(blob, al)

        assert restored["name"] == "test"
        assert restored["count"] == 42
        assert isinstance(restored["model"], MyModel)
        assert isinstance(restored["point"], Point)
        assert isinstance(restored["config"], Config)
        assert isinstance(restored["items"][1], MyModel)

    def test_attributes_dict_roundtrip(self):
        """Simulates what snapshot.py does — serialize an attributes dict."""
        attrs = {
            "score": 95,
            "history": ["step1", "step2"],
            "result": MyModel(name="final", value=100),
        }
        blob, al = serialize(attrs)
        restored = deserialize(blob, al)
        assert restored["score"] == 95
        assert restored["history"] == ["step1", "step2"]
        assert isinstance(restored["result"], MyModel)
        assert restored["result"].value == 100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_list(self):
        blob, al = serialize([])
        assert blob == []
        assert deserialize(blob, al) == []

    def test_empty_dict(self):
        blob, al = serialize({})
        assert blob == {}
        assert deserialize(blob, al) == {}

    def test_deeply_nested(self):
        value = {"a": [{"b": [1, 2, {"c": True}]}]}
        blob, al = serialize(value)
        assert deserialize(blob, al) == value

    def test_dict_with_type_key_not_envelope(self):
        """A dict with '__type__' key but wrong format is NOT treated as envelope."""
        value = {"__type__": "not_a_real_type", "data": 42}
        blob, al = serialize(value)
        restored = deserialize(blob, al)
        assert restored == value

    def test_serialize_deserialize_idempotent(self):
        """serialize(deserialize(blob)) produces the same blob."""
        m = MyModel(name="test", value=1)
        blob1, al1 = serialize(m)
        restored = deserialize(blob1, al1)
        blob2, al2 = serialize(restored)
        assert blob1 == blob2


# ---------------------------------------------------------------------------
# Fix 1: Pydantic model with nested non-Pydantic objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Coord:
    lat: float
    lon: float


class LocationModel(BaseModel):
    name: str
    coord: Any  # holds a dataclass at runtime


class SnapshotableHolder(BaseModel):
    """Pydantic model that holds a @snapshotable object."""

    cfg: Any


class TestPydanticNestedNonPydantic:
    """Pydantic models containing dataclass/snapshotable fields must
    preserve the nested type through serialization roundtrip."""

    def test_pydantic_with_nested_dataclass_roundtrip(self):
        loc = LocationModel(name="HQ", coord=Coord(lat=37.7, lon=-122.4))
        blob, al = serialize(loc)
        restored = deserialize(blob, al)
        assert isinstance(restored, LocationModel)
        assert isinstance(restored.coord, Coord)
        assert restored.coord.lat == 37.7
        assert restored.coord.lon == -122.4

    def test_pydantic_with_nested_snapshotable_roundtrip(self):
        """Pydantic model holding a @snapshotable object roundtrips correctly."""
        cfg = Config(host="db.local", port=5432)
        holder = SnapshotableHolder(cfg=cfg)
        blob, al = serialize(holder)
        restored = deserialize(blob, al)
        assert isinstance(restored, SnapshotableHolder)
        assert isinstance(restored.cfg, Config)
        assert restored.cfg.host == "db.local"
        assert restored.cfg.port == 5432


# ---------------------------------------------------------------------------
# Fix 2: Nested/inner class import
# ---------------------------------------------------------------------------


class Outer:
    """Module-level class with an inner class, for testing nested import."""

    @snapshotable
    class Inner:
        def __init__(self, val: int = 0):
            self.val = val


class TestNestedClassImport:
    def test_nested_inner_class_roundtrip(self):
        """A @snapshotable inner class survives serialize/deserialize."""
        obj = Outer.Inner(val=42)
        blob, al = serialize(obj)
        restored = deserialize(blob, al)
        assert isinstance(restored, Outer.Inner)
        assert restored.val == 42

    def test_import_class_invalid_fqn(self):
        """_import_class raises DeserializationError for a single-part name."""
        with pytest.raises(DeserializationError, match="Invalid fully qualified name"):
            deserialize(
                {"__type__": "dict_class", "__class__": "NoDots", "data": {}},
                {"NoDots"},
            )

    def test_import_class_nonexistent_raises(self):
        """_import_class raises DeserializationError for unfindable class."""
        fqn = "nonexistent.module.Klass"
        with pytest.raises(DeserializationError, match="Cannot import class"):
            deserialize(
                {"__type__": "dict_class", "__class__": fqn, "data": {}},
                {fqn},
            )

    def test_legacy_nemo_oo_agents_class_path_imports_renamed_nooa_class(self):
        """Snapshots saved before the nooa rename restore through the new module path."""
        fqn = "nemo_oo_agents.storage.snapshot_vars.SnapshotVars"
        restored = deserialize(
            {"__type__": "dict_class", "__class__": fqn, "data": {"_data": {"answer": 42}}},
            {fqn},
        )
        from nooa.storage.snapshot_vars import SnapshotVars

        assert isinstance(restored, SnapshotVars)
        assert restored["answer"] == 42


# ---------------------------------------------------------------------------
# Bug 1: SKIP must not leak into nested collections
# ---------------------------------------------------------------------------


class _NoSnapshotItem:
    __nosnapshot__ = True


class ModelWithNoSnapshot(BaseModel):
    name: str
    transient: Any = None


@dataclasses.dataclass
class DataclassWithNoSnapshot:
    label: str
    transient: Any = None


@snapshotable
class SnapshotableWithNoSnapshot:
    def __init__(self, label: str, transient: Any = None):
        self.label = label
        self.transient = transient


class TestNoSnapshotNested:
    def test_nosnapshot_nested_in_dict(self):
        """Dict containing a nosnapshot value drops that key."""
        value = {"keep": "yes", "drop": _NoSnapshotItem()}
        blob, al = serialize(value)
        assert "keep" in blob
        assert "drop" not in blob
        assert blob == {"keep": "yes"}

    def test_nosnapshot_nested_in_list(self):
        """List containing a nosnapshot value removes that item."""
        value = [1, _NoSnapshotItem(), 3]
        blob, al = serialize(value)
        assert blob == [1, 3]

    def test_nosnapshot_nested_in_pydantic_model(self):
        """Pydantic model with a nosnapshot field value drops the field from data."""
        m = ModelWithNoSnapshot(name="test", transient=_NoSnapshotItem())
        blob, al = serialize(m)
        assert "transient" not in blob["data"]
        assert blob["data"]["name"] == "test"
        # Roundtrip: the field gets its default (None)
        restored = deserialize(blob, al)
        assert isinstance(restored, ModelWithNoSnapshot)
        assert restored.name == "test"
        assert restored.transient is None

    def test_nosnapshot_nested_in_dataclass(self):
        """Dataclass with a nosnapshot field value drops the field from data."""
        dc = DataclassWithNoSnapshot(label="test", transient=_NoSnapshotItem())
        blob, al = serialize(dc)
        assert "transient" not in blob["data"]
        restored = deserialize(blob, al)
        assert isinstance(restored, DataclassWithNoSnapshot)
        assert restored.label == "test"
        assert restored.transient is None

    def test_nosnapshot_nested_in_snapshotable(self):
        """Snapshotable with a nosnapshot attr value drops the attr from data."""
        obj = SnapshotableWithNoSnapshot(label="test", transient=_NoSnapshotItem())
        blob, al = serialize(obj)
        assert "transient" not in blob["data"]
        restored = deserialize(blob, al)
        assert isinstance(restored, SnapshotableWithNoSnapshot)
        assert restored.label == "test"
        assert restored.transient is None

    def test_nosnapshot_nested_in_tuple(self):
        """Tuple containing a nosnapshot value removes that item."""
        value = (1, _NoSnapshotItem(), 3)
        blob, al = serialize(value)
        assert blob["data"] == [1, 3]
        restored = deserialize(blob, al)
        assert restored == (1, 3)


# ---------------------------------------------------------------------------
# Bug 2: Enum serialization
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(enum.IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ModelWithEnum(BaseModel):
    color: Color
    priority: Priority


@dataclasses.dataclass
class DataclassWithEnum:
    color: Color
    priority: Priority


class TestEnum:
    def test_enum_with_string_value(self):
        """StrEnum-like enum serializes to its string value."""
        blob, al = serialize(Color.RED)
        assert blob == "red"
        assert al == set()

    def test_enum_with_int_value(self):
        """IntEnum serializes to its int value."""
        blob, al = serialize(Priority.HIGH)
        assert blob == 3
        assert al == set()

    def test_enum_in_pydantic_model_roundtrip(self):
        """Pydantic model with enum fields roundtrips correctly."""
        m = ModelWithEnum(color=Color.GREEN, priority=Priority.MEDIUM)
        blob, al = serialize(m)
        # Enum values are serialized as their primitives
        assert blob["data"]["color"] == "green"
        assert blob["data"]["priority"] == 2
        # Pydantic coerces primitives back to enum on model_validate
        restored = deserialize(blob, al)
        assert isinstance(restored, ModelWithEnum)
        assert restored.color is Color.GREEN
        assert restored.priority is Priority.MEDIUM

    def test_enum_in_dataclass_roundtrip(self):
        """Dataclass with enum fields: enums serialize to primitives,
        deserialize as primitives (dataclass __init__ receives raw values)."""
        dc = DataclassWithEnum(color=Color.BLUE, priority=Priority.LOW)
        blob, al = serialize(dc)
        assert blob["data"]["color"] == "blue"
        assert blob["data"]["priority"] == 1
        # Dataclass __init__ receives raw primitives — no coercion
        restored = deserialize(blob, al)
        assert isinstance(restored, DataclassWithEnum)
        assert restored.color == "blue"
        assert restored.priority == 1

    def test_enum_in_list(self):
        """Enum values in a list serialize to their primitives."""
        value = [Color.RED, Priority.HIGH]
        blob, al = serialize(value)
        assert blob == ["red", 3]

    def test_enum_in_dict_value(self):
        """Enum values in a dict serialize to their primitives."""
        value = {"color": Color.GREEN, "priority": Priority.LOW}
        blob, al = serialize(value)
        assert blob == {"color": "green", "priority": 1}

    def test_strenum_serializes_to_plain_string(self):
        """StrEnum members serialize to plain str, not enum objects."""
        import enum

        class StrColor(enum.StrEnum):
            RED = "red"
            BLUE = "blue"

        blob, _ = serialize(StrColor.RED)
        assert blob == "red"
        assert type(blob) is str  # NOT StrColor

    def test_intenum_serializes_to_plain_int(self):
        """IntEnum members serialize to plain int, not enum objects."""
        import enum

        class IntPriority(enum.IntEnum):
            LOW = 1
            HIGH = 3

        blob, _ = serialize(IntPriority.HIGH)
        assert blob == 3
        assert type(blob) is int  # NOT IntPriority


# ---------------------------------------------------------------------------
# Bug 3: extra="forbid" Pydantic models
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: int = 0


class TestPydanticExtraForbid:
    def test_pydantic_extra_forbid_lenient_restore(self):
        """Model with extra='forbid' deserializes when extra fields exist in blob."""
        blob = {
            "__type__": "pydantic",
            "__class__": f"{StrictModel.__module__}.StrictModel",
            "data": {"name": "test", "value": 42, "removed_field": "stale"},
        }
        fqn = f"{StrictModel.__module__}.StrictModel"
        restored = deserialize(blob, {fqn})
        assert isinstance(restored, StrictModel)
        assert restored.name == "test"
        assert restored.value == 42

    def test_pydantic_extra_forbid_normal_roundtrip(self):
        """Model with extra='forbid' roundtrips normally when no drift."""
        m = StrictModel(name="ok", value=7)
        blob, al = serialize(m)
        restored = deserialize(blob, al)
        assert isinstance(restored, StrictModel)
        assert restored.name == "ok"
        assert restored.value == 7


# ---------------------------------------------------------------------------
# Bug 4: Non-string dict keys
# ---------------------------------------------------------------------------


class TestNonStringDictKeys:
    def test_non_string_dict_keys_raise(self):
        """{1: 'value'} raises SerializationError with a clear message."""
        with pytest.raises(SerializationError, match="Dict key 1.*not a string"):
            serialize({1: "value"})

    def test_non_string_dict_keys_tuple_key(self):
        """Tuple key also raises SerializationError."""
        with pytest.raises(SerializationError, match="not a string"):
            serialize({(1, 2): "value"})


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class OptionalHolder(BaseModel):
    item: MyModel | None = None


class DeepModel(BaseModel):
    label: str
    payload: Any  # holds a dataclass at runtime


@snapshotable
class DeepConfig:
    def __init__(self, host: str, port: int = 80):
        self.host = host
        self.port = port


@dataclasses.dataclass
class DeepContainer:
    config: Any  # holds a snapshotable at runtime
    score: int = 0


class TestAdditionalEdgeCases:
    def test_optional_pydantic_model_none(self):
        """Optional[MyModel] field with value None roundtrips correctly."""
        holder = OptionalHolder(item=None)
        blob, al = serialize(holder)
        restored = deserialize(blob, al)
        assert isinstance(restored, OptionalHolder)
        assert restored.item is None

    def test_optional_pydantic_model_with_value(self):
        """Optional[MyModel] field with a value roundtrips correctly."""
        holder = OptionalHolder(item=MyModel(name="test", value=1))
        blob, al = serialize(holder)
        restored = deserialize(blob, al)
        assert isinstance(restored, OptionalHolder)
        assert isinstance(restored.item, MyModel)
        assert restored.item.name == "test"

    def test_empty_tuple_roundtrip(self):
        """() roundtrips as empty tuple."""
        blob, al = serialize(())
        assert blob["__type__"] == "tuple"
        assert blob["data"] == []
        restored = deserialize(blob, al)
        assert restored == ()
        assert type(restored) is tuple

    def test_deeply_nested_mixed_types(self):
        """Pydantic model containing a dataclass containing a snapshotable — full roundtrip."""
        value = DeepModel(
            label="deep",
            payload=DeepContainer(
                config=DeepConfig("db.local", 5432),
                score=99,
            ),
        )
        blob, al = serialize(value)
        restored = deserialize(blob, al)

        assert isinstance(restored, DeepModel)
        assert restored.label == "deep"
        assert isinstance(restored.payload, DeepContainer)
        assert restored.payload.score == 99
        assert isinstance(restored.payload.config, DeepConfig)
        assert restored.payload.config.host == "db.local"
        assert restored.payload.config.port == 5432
