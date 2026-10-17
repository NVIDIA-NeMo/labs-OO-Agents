# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SnapshotVars — filter-on-write dict-like store for snapshot-backed vars."""

import logging

from nooa.storage.json_snapshot import (  # noqa: F401
    snapshot_from_dict,
    snapshot_to_dict,
)
from nooa.storage.serialization import serialize
from nooa.storage.snapshot_vars import SnapshotVars


class _Unserializable:
    """A plain object with no serialization support."""


class TestSnapshotVarsDictBehavior:
    def test_acts_like_a_dict(self):
        v = SnapshotVars()
        v["a"] = 1
        v["b"] = [1, 2, 3]
        assert v["a"] == 1
        assert v["b"] == [1, 2, 3]
        assert "a" in v
        assert set(v) == {"a", "b"}
        assert len(v) == 2
        assert v.get("missing", "default") == "default"
        del v["a"]
        assert "a" not in v

    def test_init_from_dict_filters(self):
        v = SnapshotVars({"good": 1, "bad": _Unserializable()})
        assert v["good"] == 1
        assert "bad" not in v


class TestSnapshotVarsFiltersOnWrite:
    def test_unserializable_value_is_not_stored(self):
        v = SnapshotVars()
        v["live"] = _Unserializable()
        assert "live" not in v
        assert len(v) == 0

    def test_unserializable_write_logs_warning(self, caplog):
        v = SnapshotVars()
        with caplog.at_level(logging.WARNING):
            v["sock"] = _Unserializable()
        assert any("sock" in r.message for r in caplog.records)
        assert any("not be persisted" in r.message.lower() for r in caplog.records)

    def test_serializable_value_is_stored(self):
        from pydantic import BaseModel

        class M(BaseModel):
            x: int

        v = SnapshotVars()
        v["m"] = M(x=5)
        assert v["m"].x == 5


class TestSnapshotVarsRoundTrips:
    def test_snapshot_serialize_succeeds_even_after_bad_write(self):
        v = SnapshotVars()
        v["keep"] = {"token": "abc"}
        v["drop"] = _Unserializable()  # skipped on write
        blob, _allow = serialize(v)
        # The snapshotable envelope wraps the clean internal store.
        assert blob["__type__"] == "dict_class"
        assert blob["data"]["_data"] == {"keep": {"token": "abc"}}

    def test_snapshot_serialize_succeeds_even_after_non_string_key_write(self):
        v = SnapshotVars()
        v["keep"] = {"token": "abc"}
        v[123] = "bad key"  # skipped on write
        blob, _allow = serialize(v)
        assert blob["data"]["_data"] == {"keep": {"token": "abc"}}

    def test_round_trip_preserves_kept_values(self):
        from nooa.storage.serialization import deserialize

        v = SnapshotVars()
        v["keep"] = {"token": "abc"}
        blob, allow = serialize(v)
        restored = deserialize(blob, allow)
        assert isinstance(restored, SnapshotVars)
        assert restored["keep"] == {"token": "abc"}


class TestTodoVarsIntegration:
    """SnapshotVars must work as the type of Todo.vars (the wiring the MR claims)."""

    def test_todo_is_constructible_and_vars_is_snapshotvars(self):
        from nooa.tools.todo import Todo

        t = Todo(title="x")
        assert isinstance(t.vars, SnapshotVars)

    def test_todo_coerces_dict_vars_to_snapshotvars(self):
        from nooa.tools.todo import Todo

        t = Todo(title="x", vars={"a": 1})
        assert isinstance(t.vars, SnapshotVars)
        assert t.vars["a"] == 1

    def test_todo_manager_add_then_set_var(self):
        from nooa.tools.todo import TodoManager

        tm = TodoManager()
        t = tm.add("hi")
        t.v.commits = ["abc"]
        assert t.vars["commits"] == ["abc"]

    def test_todo_dict_vars_filters_unserializable(self):
        from nooa.tools.todo import Todo

        t = Todo(title="x", vars={"good": 1, "bad": _Unserializable()})
        assert t.vars["good"] == 1
        assert "bad" not in t.vars

    def test_todo_model_validate_round_trip_keeps_snapshotvars(self):
        from nooa.tools.todo import Todo

        t = Todo(title="x", vars={"a": 1})
        raw = t.model_dump()
        restored = Todo.model_validate(raw)
        assert isinstance(restored.vars, SnapshotVars)
        assert restored.vars["a"] == 1


class TestSnapshotVarsMappingHelpers:
    def test_update_filters_mixed(self):
        v = SnapshotVars()
        v.update({"good": 1, "bad": _Unserializable()})
        assert v["good"] == 1
        assert "bad" not in v

    def test_setdefault_filters_bad_default(self):
        v = SnapshotVars()
        v.setdefault("bad", _Unserializable())
        assert "bad" not in v
        v.setdefault("good", 7)
        assert v["good"] == 7


class TestSnapshotVarsPydantic:
    """A pydantic model embedding SnapshotVars dumps and validates in both modes."""

    @staticmethod
    def _model():
        from typing import Annotated

        from pydantic import BaseModel, Field

        from nooa import hidden

        class Holder(BaseModel):
            vars: Annotated[SnapshotVars, hidden] = Field(default_factory=SnapshotVars)

        return Holder

    def test_json_mode_writes_plain_data_for_json_values(self):
        holder = self._model()(vars={"a": 1, "b": ["x", {"c": None}]})
        assert holder.model_dump(mode="json") == {"vars": {"a": 1, "b": ["x", {"c": None}]}}
        back = type(holder).model_validate_json(holder.model_dump_json())
        assert isinstance(back.vars, SnapshotVars)
        assert dict(back.vars) == {"a": 1, "b": ["x", {"c": None}]}

    def test_python_mode_and_deep_copy_keep_snapshot_vars(self):
        holder = self._model()(vars={"a": 1})
        assert isinstance(holder.model_dump()["vars"], SnapshotVars)
        copied = holder.model_copy(deep=True)
        assert isinstance(copied.vars, SnapshotVars) and copied.vars is not holder.vars
        assert dict(copied.vars) == {"a": 1}

    def test_non_json_values_round_trip_as_a_tagged_snapshot_blob(self):
        from nooa.tools.todo import TodoComment

        comment = TodoComment(body="hi")
        holder = self._model()(vars={"pair": (1, 2), "comment": comment, "n": 3})
        dumped = holder.model_dump(mode="json")["vars"]
        assert set(dumped) == {"__snapshot_vars__"}
        assert dumped["__snapshot_vars__"]["allowlist"] == ["nooa.tools.todo.TodoComment"]
        back = type(holder).model_validate_json(holder.model_dump_json())
        assert dict(back.vars) == {"pair": (1, 2), "comment": comment, "n": 3}

    def test_a_tagged_blob_naming_a_class_outside_its_allowlist_is_refused(self):
        import pytest

        from nooa.errors.storage import DeserializationError

        blob = {
            "__snapshot_vars__": {
                "blob": {"x": {"__type__": "pydantic", "__class__": "os.Evil", "data": {}}},
                "allowlist": [],
            }
        }
        with pytest.raises(DeserializationError):
            self._model().model_validate({"vars": blob})

    def test_validation_wraps_a_plain_dict_and_keeps_the_write_filter(self):
        holder = self._model().model_validate({"vars": {"ok": 1, "bad": _Unserializable()}})
        assert isinstance(holder.vars, SnapshotVars)
        assert dict(holder.vars) == {"ok": 1}
        same = SnapshotVars({"k": "v"})
        assert self._model()(vars=same).vars is same
