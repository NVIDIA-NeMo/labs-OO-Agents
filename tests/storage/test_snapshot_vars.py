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

    def test_non_json_values_are_refused_in_json_mode_naming_the_key(self):
        import pytest
        from pydantic_core import PydanticSerializationError

        from nooa.tools.todo import TodoComment

        holder = self._model()(vars={"n": 3, "comment": TodoComment(body="hi")})
        with pytest.raises(PydanticSerializationError, match="var 'comment' holds a TodoComment"):
            holder.model_dump(mode="json")
        with pytest.raises(PydanticSerializationError, match="var 'comment'"):
            holder.model_dump_json()
        pair = self._model()(vars={"pair": (1, 2)})
        with pytest.raises(PydanticSerializationError, match="var 'pair' holds a tuple"):
            pair.model_dump_json()

    @staticmethod
    def _popen_payload(marker):
        return {
            "title": "innocent",
            "vars": {
                "__snapshot_vars__": {
                    "blob": {
                        "x": {
                            "__type__": "dict_class",
                            "__class__": "subprocess.Popen",
                            "data": {"args": ["touch", str(marker)]},
                        }
                    },
                    "allowlist": ["subprocess.Popen"],
                }
            },
        }

    def test_a_payload_carrying_its_own_allowlist_constructs_nothing(self, tmp_path):
        """Validating a Todo never imports or constructs a class named by the data."""
        import json
        import time

        import pytest
        from pydantic import ValidationError

        from nooa.tools.todo import Todo

        marker = tmp_path / "pwned"
        payload = self._popen_payload(marker)
        with pytest.raises(ValidationError, match="tagged SnapshotVars payloads are not accepted"):
            Todo.model_validate(payload)
        with pytest.raises(ValidationError, match="tagged SnapshotVars payloads are not accepted"):
            Todo.model_validate_json(json.dumps(payload))
        time.sleep(0.3)  # a spawned Popen would have touched the file by now
        assert not marker.exists()

    def test_plain_json_vars_round_trip_through_a_todo(self):
        from nooa.tools.todo import Todo

        todo = Todo(title="t", vars={"a": 1, "b": ["x", {"c": None}], "s": "y"})
        back = Todo.model_validate_json(todo.model_dump_json())
        assert dict(back.vars) == dict(todo.vars)
        assert Todo.model_validate(todo.model_dump(mode="json")) == back

    def test_non_json_todo_vars_still_survive_the_session_snapshot(self):
        from nooa.storage.serialization import deserialize, serialize
        from nooa.tools.todo import TodoComment, TodoManager

        tm = TodoManager()
        todo = tm.add("t")
        todo.vars["pair"] = (1, 2)
        comment = TodoComment(body="hi")
        todo.vars["comment"] = comment
        restored = deserialize(*serialize(tm))
        again = restored.get(todo.id)
        assert again.vars["pair"] == (1, 2)
        assert again.vars["comment"] == comment

    def test_validation_wraps_a_plain_dict_and_keeps_the_write_filter(self):
        holder = self._model().model_validate({"vars": {"ok": 1, "bad": _Unserializable()}})
        assert isinstance(holder.vars, SnapshotVars)
        assert dict(holder.vars) == {"ok": 1}
        same = SnapshotVars({"k": "v"})
        assert self._model()(vars=same).vars is same


class TestEnvelopeShapedTodoVars:
    """PR #451 review 3, finding 1: a var shaped like an envelope stays a plain dict."""

    INJECTED = (
        '{"title":"b","vars":{"x":{"__type__":"pydantic",'
        '"__class__":"nooa.tools.todo.Todo","data":{"title":"INJECTED"}}}}'
    )

    @staticmethod
    def _manager_with(todo):
        from nooa.tools.todo import TodoManager

        manager = TodoManager()
        manager._todos[todo.id] = todo
        manager._order.append(todo.id)
        return manager

    def test_an_injected_todo_envelope_comes_back_as_a_plain_dict(self):
        from nooa.storage.serialization import deserialize
        from nooa.tools.todo import Todo

        todo = Todo.model_validate_json(self.INJECTED)
        restored = deserialize(*serialize(self._manager_with(todo)))
        x = restored.get(todo.id).vars["x"]
        assert type(x) is dict
        assert x == {
            "__type__": "pydantic",
            "__class__": "nooa.tools.todo.Todo",
            "data": {"title": "INJECTED"},
        }

    def test_an_os_system_envelope_does_not_wipe_the_todo_manager(self):
        from nooa.storage.serialization import deserialize
        from nooa.tools.todo import Todo

        bad = Todo.model_validate_json(
            '{"title":"b","vars":{"x":{"__type__":"pydantic","__class__":"os.system","data":{}}}}'
        )
        manager = self._manager_with(bad)
        other = manager.add("keep me")
        restored = deserialize(*serialize(manager))
        assert restored.get(other.id).title == "keep me"
        assert restored.get(bad.id).vars["x"]["__class__"] == "os.system"


class TestTaggedKeyAndNonFiniteVars:
    """PR #451 review 3, findings 4 and 5."""

    TAGGED = {"blob": {"__type__": "tuple", "data": []}, "allowlist": ["subprocess.Popen"]}

    def test_a_var_named_snapshot_vars_holding_plain_data_crosses_sessions(self):
        from nooa.tools.todo import Todo

        for value in (1, "x", {"blob": 1}, {"blob": 1, "allowlist": [], "more": 2}):
            todo = Todo(title="z", vars={"__snapshot_vars__": value})
            assert (
                Todo.model_validate_json(todo.model_dump_json()).vars["__snapshot_vars__"] == value
            )
            later = Todo(title="z")
            later.vars["__snapshot_vars__"] = value
            back = Todo.model_validate(later.model_dump(mode="json"))
            assert back.vars["__snapshot_vars__"] == value

    def test_only_the_old_tagged_shape_is_refused_everywhere(self, caplog):
        import pytest
        from pydantic import ValidationError

        from nooa.storage.serialization import deserialize
        from nooa.tools.todo import Todo

        with pytest.raises(ValidationError, match="tagged SnapshotVars payloads are not accepted"):
            Todo(title="z", vars={"__snapshot_vars__": self.TAGGED})
        with pytest.raises(ValidationError, match="tagged SnapshotVars payloads are not accepted"):
            Todo(title="z", vars={"__snapshot_vars__": self.TAGGED, "other": 1})
        # Writing it is refused too (logged, not stored), so a Todo never holds what it can't send.
        todo = Todo(title="z")
        with caplog.at_level(logging.WARNING):
            todo.vars["__snapshot_vars__"] = self.TAGGED
        assert "__snapshot_vars__" not in todo.vars
        assert "tagged SnapshotVars payloads" in caplog.text
        assert Todo.model_validate_json(todo.model_dump_json()).vars == {}
        # A store restored with it (bypassing the write check) refuses the JSON dump.
        restored = deserialize(*serialize(SnapshotVars()))
        restored._data["__snapshot_vars__"] = self.TAGGED
        with pytest.raises(ValueError, match="tagged SnapshotVars payloads are not accepted"):
            restored.to_json_data()

    def test_non_finite_floats_are_refused_naming_the_key(self):
        import pytest
        from pydantic_core import PydanticSerializationError

        from nooa.tools.todo import Todo

        for value in (
            float("nan"),
            float("inf"),
            float("-inf"),
            [1.0, float("nan")],
            {"a": float("inf")},
        ):
            todo = Todo(title="f")
            todo.vars["f"] = value
            with pytest.raises(
                PydanticSerializationError, match="var 'f' holds a non-finite float"
            ):
                todo.model_dump_json()
            with pytest.raises(
                PydanticSerializationError, match="var 'f' holds a non-finite float"
            ):
                todo.model_dump(mode="json")

    def test_finite_floats_still_cross_sessions(self):
        from nooa.tools.todo import Todo

        todo = Todo(title="f", vars={"f": 1.5, "big": 1e308, "neg": -0.0})
        assert dict(Todo.model_validate_json(todo.model_dump_json()).vars) == {
            "f": 1.5,
            "big": 1e308,
            "neg": -0.0,
        }
