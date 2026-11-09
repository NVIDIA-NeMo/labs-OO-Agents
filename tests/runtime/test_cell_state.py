# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saving and reviving a session namespace: plain data and cell-defined helpers only."""

import base64
import json

import msgpack

from nooa.runtime import cell_state
from nooa.runtime.execution_namespace import ExecutionNamespace

RESERVED = frozenset({"notification", "self", "return_result", "print"})


def namespace(**initial):
    return ExecutionNamespace(reserved=RESERVED, initial=initial)


async def fill(ns, source):
    with ns.enter({}):
        return await ns.cell(source)


async def test_plain_data_round_trips_with_its_types():
    ns = namespace()
    await fill(
        ns,
        "n = 41\nf = 2.5\ns = 'text'\nb = b'bytes'\nnothing = None\nflag = True\n"
        "items = [1, (2, 3), {4, 5}, frozenset({6})]\ntable = {'a': {1: 'one'}, 'b': [None]}",
    )
    saved = cell_state.capture(ns)
    assert saved["version"] == cell_state.FORMAT_VERSION
    assert sorted(saved["values"]) == ["b", "f", "flag", "items", "n", "nothing", "s", "table"]
    assert saved["functions"] == {} and saved["skipped"] == []
    json.dumps(saved)  # what the agent snapshot stores

    fresh = namespace()
    report = cell_state.apply(fresh, saved)
    assert report.failed == []
    g = fresh.globals
    assert (g["n"], g["f"], g["s"], g["b"], g["nothing"], g["flag"]) == (
        41,
        2.5,
        "text",
        b"bytes",
        None,
        True,
    )
    assert g["items"] == [1, (2, 3), {4, 5}, frozenset({6})]
    assert type(g["items"][1]) is tuple and type(g["items"][2]) is set
    assert g["table"] == {"a": {1: "one"}, "b": [None]}
    assert fresh.user_names >= set(saved["values"])


async def test_cell_defined_helper_is_revived_by_source_and_keeps_working():
    ns = namespace()
    assert (
        await fill(
            ns,
            "counter = 40\ndef bump(n=1, *, label='x'):\n global counter\n counter += n\n return counter\nreturn bump()",
        )
        == 41
    )
    saved = cell_state.capture(ns)
    assert "bump" in saved["functions"] and saved["functions"]["bump"].startswith("def bump(")

    fresh = namespace()
    report = cell_state.apply(fresh, saved)
    assert report.restored == ["bump", "counter"] and report.failed == []
    assert fresh.globals["bump"].__globals__ is fresh.globals
    assert await fill(fresh, "return bump()") == 42
    # Revived helpers keep their source, so the next restart keeps them too.
    again = cell_state.capture(fresh)
    assert again["functions"]["bump"] == saved["functions"]["bump"]
    assert again["values"]["counter"] != saved["values"]["counter"]


async def test_async_helper_with_annotations_is_revived_without_evaluating_them():
    ns = namespace()
    await fill(ns, "async def twice(x: 'Undefined') -> 'Undefined':\n return x * 2")
    saved = cell_state.capture(ns)
    fresh = namespace()
    assert cell_state.apply(fresh, saved).restored == ["twice"]
    assert await fill(fresh, "return await twice(21)") == 42


async def test_everything_that_could_run_code_on_load_is_skipped_with_a_reason():
    ns = namespace()
    await fill(
        ns,
        """
import json
from functools import lru_cache
class K:
    pass
anon = lambda: 1
@lru_cache(None)
def cached():
    return 1
def passthrough(f):
    return f
@passthrough
def decorated():
    return 1
def outer():
    x = 1
    def inner():
        return x
    return inner
closed = outer()
seed = 3
def defaulted(x=seed):
    return x
opaque = object()
_private = 5
""",
    )
    saved = cell_state.capture(ns)
    reasons = {entry["name"]: entry["reason"] for entry in saved["skipped"]}
    assert set(saved["values"]) == {"seed"}
    assert set(saved["functions"]) == {"outer", "passthrough", "defaulted"}  # x=seed is a name
    assert "_private" not in saved["values"] and "_private" not in reasons
    assert reasons["json"].startswith("module")
    assert set(saved["classes"]) == {"K"}  # a plain class with an empty body is restorable
    assert "lambda" in reasons["anon"]
    assert "decorator" in reasons["decorated"]
    assert reasons["cached"].startswith("not kept")  # an lru_cache wrapper object
    assert "closes over" in reasons["closed"]
    assert reasons["opaque"].startswith("not kept")
    assert reasons["lru_cache"].startswith("function not kept")


async def test_size_caps_skip_the_name_and_say_so():
    ns = namespace()
    await fill(ns, "big = b'x' * 5000\nsmall = 1\nmedium = b'y' * 1500")
    saved = cell_state.capture(ns, max_variable_bytes=2000, max_total_bytes=1600)
    reasons = {entry["name"]: entry["reason"] for entry in saved["skipped"]}
    assert "big" in reasons and "per-name cap" in reasons["big"]
    # newest first: medium (1500 B) fits the total, small (1 B) still fits after it
    assert set(saved["values"]) == {"medium", "small"}
    assert saved["bytes"] < 1600


def test_a_tampered_payload_costs_only_the_names_it_touches():
    good = base64.b64encode(cell_state.pack_value(1)).decode()
    forged = base64.b64encode(msgpack.packb(msgpack.ExtType(99, b"x"))).decode()
    snapshot = {
        "version": cell_state.FORMAT_VERSION,
        "values": {
            "ok": good,
            "forged": forged,
            "notb64": "***",
            "_hidden": good,
            "bad name": good,
        },
        "functions": {
            "fine": "def fine():\n    return 1",
            "deco": "@print\ndef deco():\n    return 1",
            "expr_default": "def expr_default(x=__import__('os')):\n    return x",
            "renamed": "def other():\n    return 1",
            "two": "def two():\n    return 1\nimport os",
            "cls": "class cls:\n    pass",
        },
    }
    fresh = namespace()
    report = cell_state.apply(fresh, snapshot)
    assert report.restored == ["fine", "ok"]
    failed = {entry["name"]: entry["reason"] for entry in report.failed}
    assert set(failed) == {
        "forged",
        "notb64",
        "_hidden",
        "bad name",
        "deco",
        "expr_default",
        "renamed",
        "two",
        "cls",
    }
    assert "decorators" in failed["deco"] and "literal" in failed["expr_default"]
    assert "os" not in fresh.globals and "deco" not in fresh.globals


def test_unknown_format_restores_nothing():
    fresh = namespace()
    report = cell_state.apply(fresh, {"version": 99, "values": {"x": "AQ=="}})
    assert report.restored == [] and report.failed == [
        {"name": "*", "reason": "unknown saved-state format"}
    ]
    assert "x" not in fresh.globals


def test_restore_notice_names_what_came_back_and_what_did_not():
    report = cell_state.RestoreReport(
        restored=["bump", "counter"],
        failed=[{"name": "deco", "reason": "function not restored: decorators run code"}],
        not_saved=[{"name": "json", "type": "module", "reason": "module: import it again"}],
    )
    text = cell_state.restore_notice(report)
    assert text.startswith("[python-state-restored]")
    assert "available again: bump, counter." in text
    assert "deco (function not restored" in text and "json (module: import it again)" in text
    assert "self.v and saved history remain" in text
    empty = cell_state.restore_notice(cell_state.RestoreReport())
    assert "could not be revived" in empty


async def test_cell_defined_classes_and_their_instances_round_trip():
    ns = namespace()
    await fill(
        ns,
        """
from pydantic import BaseModel
from dataclasses import dataclass
import enum
class Color(enum.Enum):
    RED = 'red'
    BLUE = 'blue'
class Item(BaseModel):
    \"\"\"One line item.\"\"\"
    name: str
    qty: int = 1
    color: Color = Color.RED
    def total(self, price: float = 2.0) -> float:
        return self.qty * price
@dataclass(frozen=True)
class Point:
    x: int
    y: int = 0
class Bag:
    limit = 3
    def describe(self):
        return f'bag of {len(self.items)}'
items = [Item(name='a'), Item(name='b', qty=2, color=Color.BLUE)]
origin = Point(1, 2)
bag = Bag()
bag.items = items
favourite = Color.BLUE
""",
    )
    saved = cell_state.capture(ns)
    assert set(saved["classes"]) == {"Color", "Item", "Point", "Bag"}
    assert set(saved["instances"]) == {"items", "origin", "bag", "favourite"}
    assert saved["models"] == {}
    assert {e["name"] for e in saved["skipped"]} <= {"dataclass", "enum", "BaseModel"}

    fresh = namespace()
    report = cell_state.apply(fresh, saved)
    assert report.failed == []
    g = fresh.globals
    assert g["items"][1].qty == 2 and g["items"][1].color is g["Color"].BLUE
    assert g["items"][0].total(price=3.0) == 3.0
    assert g["origin"] == g["Point"](1, 2) and g["origin"].y == 2
    assert g["bag"].describe() == "bag of 2" and g["bag"].limit == 3
    assert g["favourite"] is g["Color"].BLUE and type(g["favourite"]) is g["Color"]
    assert g["Item"].__doc__ == "One line item."
    # Library names the definitions needed are not left behind as user globals.
    assert "BaseModel" not in g and "dataclass" not in g and "enum" not in g
    # Round trip again: the revived classes carry their source.
    assert cell_state.capture(fresh)["classes"] == saved["classes"]


async def test_classes_whose_definition_would_run_code_are_skipped_with_a_reason():
    ns = namespace()
    await fill(
        ns,
        """
from pydantic import BaseModel, ConfigDict, field_validator
class Init:
    def __init__(self):
        self.x = 1
class Post(BaseModel):
    n: int = 0
    def model_post_init(self, ctx):
        pass
class Conf(BaseModel):
    model_config = ConfigDict(extra='allow')
class Valid(BaseModel):
    n: int = 0
    @field_validator('n')
    @classmethod
    def check(cls, v):
        return v
class Meta(metaclass=type):
    pass
class Computed:
    size = len('abc')
class Nested:
    class Inner:
        pass
init = Init()
""",
    )
    saved = cell_state.capture(ns)
    reasons = {entry["name"]: entry["reason"] for entry in saved["skipped"]}
    assert saved["classes"] == {}
    assert "__init__" in reasons["Init"] and "model_post_init" in reasons["Post"]
    assert "plain names" in reasons["Conf"]
    assert "decorator" in reasons["Valid"]
    assert "metaclass" in reasons["Meta"]
    assert "literal" in reasons["Computed"]
    assert "ClassDef" in reasons["Nested"]
    assert "cell class that is not restorable" in reasons["init"]


async def test_a_shadowed_library_name_fails_the_class_instead_of_running_it():
    saved = {
        "version": cell_state.FORMAT_VERSION,
        "classes": {"Item": "class Item(BaseModel):\n    n: int = 1"},
        "instances": {},
    }
    ns = namespace()
    await fill(ns, "BaseModel = object")  # a cell rebound the name
    report = cell_state.apply(ns, saved)
    assert report.restored == []
    assert "bound to something else" in report.failed[0]["reason"]
    assert "Item" not in ns.globals

    clean = namespace()
    assert cell_state.apply(clean, saved).restored == ["Item"]
    assert clean.globals["Item"](n=5).n == 5


def test_an_instance_of_a_class_that_did_not_restore_fails_only_itself():
    encoded = base64.b64encode(
        cell_state.pack_value({cell_state.INSTANCE_KEY: "Ghost", "data": {"n": 1}})
    ).decode()
    ok = base64.b64encode(cell_state.pack_value(2)).decode()
    saved = {
        "version": cell_state.FORMAT_VERSION,
        "instances": {"ghost": encoded},
        "values": {"ok": ok},
    }
    fresh = namespace()
    report = cell_state.apply(fresh, saved)
    assert report.restored == ["ok"]
    assert "Ghost" in report.failed[0]["reason"] and "ghost" not in fresh.globals
