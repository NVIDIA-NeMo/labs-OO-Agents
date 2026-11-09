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
    assert set(saved["values"]) == {"seed"} and set(saved["functions"]) == {"outer", "passthrough"}
    assert "_private" not in saved["values"] and "_private" not in reasons
    assert reasons["json"].startswith("module")
    assert "class" in reasons["K"]
    assert "lambda" in reasons["anon"]
    assert "decorator" in reasons["decorated"]
    assert reasons["cached"].startswith("not plain data")  # an lru_cache wrapper object
    assert "closes over" in reasons["closed"]
    assert "literal" in reasons["defaulted"]
    assert reasons["opaque"].startswith("not plain data")
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
