# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TodoVars proxy: a persistent dict, also usable as attributes."""

import pytest

from nooa.tools.todo import Todo, TodoManager


class TestTodoVars:
    """Tests for the TodoVars proxy."""

    def test_set_and_get(self):
        t = Todo(title="test")
        t.v.x = 42
        assert t.v.x == 42
        assert t.vars["x"] == 42

    def test_get_missing_raises_attribute_error(self):
        t = Todo(title="test")
        with pytest.raises(AttributeError, match="No var 'nope'"):
            _ = t.v.nope

    def test_delete(self):
        t = Todo(title="test")
        t.v.x = 1
        del t.v.x
        assert "x" not in t.vars

    def test_delete_missing_raises_attribute_error(self):
        t = Todo(title="test")
        with pytest.raises(AttributeError, match="No var 'x'"):
            del t.v.x

    def test_contains(self):
        t = Todo(title="test")
        assert "x" not in t.v
        t.v.x = 1
        assert "x" in t.v

    def test_repr(self):
        t = Todo(title="test")
        t.v.a = 1
        assert repr(t.v) == repr({"a": 1})

    def test_shared_with_vars_dict(self):
        """Proxy reads/writes go through the same dict as t.vars."""
        t = Todo(title="test")
        t.vars["via_dict"] = "hello"
        assert t.v.via_dict == "hello"
        t.v.via_proxy = "world"
        assert t.vars["via_proxy"] == "world"

    def test_shared_with_set_var(self):
        """Proxy is compatible with TodoManager.set_var / get_var."""
        mgr = TodoManager()
        t = mgr.add("test")
        mgr.set_var(t.id, "key", [1, 2, 3])
        assert t.v.key == [1, 2, 3]
        t.v.other = "val"
        assert mgr.get_var(t.id, "other") == "val"

    def test_survives_serialization(self):
        """Vars set via proxy survive snapshot round-trip."""
        mgr = TodoManager()
        t = mgr.add("test")
        t.v.data = {"nested": True}

        state = mgr.to_dict()
        mgr2 = TodoManager(state=state)
        t2 = mgr2.get(t.id)
        assert t2.v.data == {"nested": True}

    def test_multiple_proxy_instances_share_state(self):
        """Each .v access creates a new proxy but they share the dict."""
        t = Todo(title="test")
        v1 = t.v
        v2 = t.v
        v1.x = 99
        assert v2.x == 99

    def test_item_access_is_the_attribute_store(self):
        t = Todo(title="test")
        t.v["k"] = 1
        assert t.v.k == 1 and t.v["k"] == 1 and t.vars["k"] == 1
        t.v.other = 2
        assert t.v["other"] == 2
        del t.v["k"]
        assert "k" not in t.v and "k" not in t.vars

    def test_item_access_missing_raises_key_error(self):
        t = Todo(title="test")
        with pytest.raises(KeyError):
            _ = t.v["nope"]
        with pytest.raises(KeyError):
            del t.v["nope"]

    def test_item_access_validates_names_like_attributes(self):
        t = Todo(title="test")
        with pytest.raises(AttributeError, match=r"'keys' is reserved.*use set\('keys', value\)"):
            t.v["keys"] = 1
        with pytest.raises(AttributeError, match=r"'_owner' is reserved.*use get\('_owner'\)"):
            _ = t.v["_owner"]
        with pytest.raises(TypeError, match="var names are str, got int"):
            t.v[1] = "x"
        t.v.set("keys", 1)
        assert t.v.get("keys") == 1

    def test_item_writes_survive_serialization(self):
        mgr = TodoManager()
        t = mgr.add("test")
        t.v["data"] = {"nested": True}
        t2 = TodoManager(state=mgr.to_dict()).get(t.id)
        assert t2.v["data"] == {"nested": True} and t2.v.data == {"nested": True}
