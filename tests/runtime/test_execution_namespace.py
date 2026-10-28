# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler acceptance tests: no models and no legacy copy/refresh engine."""

import asyncio
import traceback

import pytest

from nooa.runtime.execution_namespace import ExecutionNamespace

RESERVED = frozenset({"notification", "self", "return_result", "print"})


def namespace(**initial):
    return ExecutionNamespace(reserved=RESERVED, initial=initial)


async def test_counter_and_one_authoritative_dictionary():
    ns = namespace()
    with ns.enter({}):
        one = await ns.cell(
            "counter = 0\ndef bump():\n global counter\n counter += 1\n return counter\nreturn bump()"
        )
        two = await ns.cell("return bump()")
        three = await ns.cell("return counter")
    assert [one, two, three] == [1, 2, 2]
    with ns.enter({}):
        assert await ns.cell("return counter") == 2
    assert ns.globals["bump"].__globals__ is ns.globals
    assert not any(key.startswith("_nooa_ns_cell_") for key in ns.globals)


async def test_reader_cache_misses_are_fresh_but_hits_stay_cached():
    ns = namespace()
    with ns.enter({"notification": {"value": "A"}}):
        assert (
            await ns.cell("""
from functools import lru_cache
class Reader:
    def read(self):
        return notification['value']
reader = Reader()
@lru_cache(None)
def cached(key):
    return reader.read()
return cached('hit')
""")
            == "A"
        )
    with ns.enter({"notification": {"value": "B"}}):
        assert await ns.cell("return cached('miss'), cached('hit'), reader.read()") == (
            "B",
            "A",
            "B",
        )
    assert ns.globals["Reader"].read.__globals__ is ns.globals
    assert ns.globals["cached"].__wrapped__.__globals__ is ns.globals


async def test_complete_binding_forms_and_deletion():
    ns = namespace()
    with ns.enter({}):
        await ns.cell("""
import json as j
from math import sqrt as root
(a, *tail) = (1, 2, 3)
for item in [4]:
    loop = item
match {'a': [5, 6], 'extra': 7}:
    case {'a': [head, *rest], **remaining}:
        matched = True
try:
    raise ValueError('bad')
except ValueError as exc:
    error_text = str(exc)
values = [last := i for i in range(3)]
generator = (last_gen := k for k in range(2))
consumed = list(generator)
annotation_only: MissingType
annotated: MissingType = 12
class Thing:
    pass
obj = Thing()
obj.x: MissingType = 9
""")
        assert await ns.cell(
            "return j.loads('3'), root(9), head, rest, remaining, last, last_gen, annotated, obj.x"
        ) == (3, 3, 5, [6], {"extra": 7}, 2, 1, 12, 9)
        for absent in ("exc", "i", "k", "annotation_only", "MissingType"):
            assert absent not in ns.globals
        await ns.cell("del annotated, a\nremaining = 99")
        assert "annotated" not in ns.globals and "a" not in ns.globals
        assert ns.globals["remaining"] == 99


async def test_scopes_class_lookup_closures_comprehensions_and_defaults():
    ns = namespace()
    owner = object()
    with ns.enter({"self": owner, "notification": "A"}):
        answer = await ns.cell("""
class Reader:
    self = 'class self'
    first = self
    notification = notification + ' class'
    after = notification
    def read(self):
        return self, notification
    def explicit(self):
        global notification
        return notification
    nested = [notification for notification in ['comp']]
def closure(notification):
    def inner():
        nonlocal notification
        notification += '!'
        return notification
    return inner
inner = closure('closed')
def local(notification):
    return notification
alias = notification
def captured(value=notification):
    return value
reader = Reader()
return Reader.first, Reader.after, Reader.nested, reader.read()[0] is reader, inner(), local('local')
""")
        assert answer == ("class self", "A class", ["comp"], True, "closed!", "local")
    with ns.enter({"self": owner, "notification": "B"}):
        assert await ns.cell(
            "return reader.read()[1], reader.explicit(), inner(), alias, captured()"
        ) == ("B", "B", "closed!!", "A", "A")


@pytest.mark.parametrize(
    "code",
    [
        "notification = 1",
        "del notification",
        "import json as notification",
        "match 1:\n case notification: pass",
        "try: pass\nexcept Exception as self: pass",
        "def f():\n global notification\n notification = 1",
        "class C:\n global self\n del self",
        "__builtins__ = {}",
        "_nooa_ns_source = None",
    ],
)
async def test_reserved_global_write_delete_rejected(code):
    ns = namespace()
    with ns.enter({}):
        with pytest.raises(SyntaxError, match="reserved"):
            await ns.cell(code)


async def test_comprehension_shadow_does_not_write_reserved_global():
    ns = namespace()
    with ns.enter({"notification": "outer"}):
        assert await ns.cell("return [notification for notification in [1, 2]], notification") == (
            [1, 2],
            "outer",
        )


async def test_definitions_defaults_and_decorators_execute_once_in_order():
    ns = namespace()
    with ns.enter({}):
        assert (
            await ns.cell("""
seen = []
def default():
    seen.append('default')
    return 3
def decorator(label):
    seen.append('expr ' + label)
    def apply(fn):
        seen.append('apply ' + label)
        assert 'def helper' in fn._generated_source
        return fn
    return apply
@decorator('one')
@decorator('two')
def helper(value=default()):
    return value
return helper()
""")
            == 3
        )
        assert ns.globals["seen"] == ["expr one", "expr two", "default", "apply two", "apply one"]
        await ns.cell("helper()")
        assert len(ns.globals["seen"]) == 5


@pytest.mark.parametrize("catch", ["except:", "except BaseException:"])
async def test_real_return_is_not_a_catchable_exception(catch):
    ns = namespace()
    with ns.enter({}):
        assert await ns.cell(f"try:\n return 7\n{catch}\n return 9") == 7
        assert await ns.cell("try:\n return 7\nfinally:\n return 8") == 8


async def test_real_return_context_managers_and_finally():
    ns = namespace()
    with ns.enter({}):
        assert (
            await ns.cell("""
seen = []
class Manager:
    def __enter__(self): return self
    def __exit__(self, *args):
        seen.append(args)
        return True
    async def __aenter__(self): return self
    async def __aexit__(self, *args):
        seen.append(args)
        return True
try:
    with Manager():
        async with Manager():
            return 5
finally:
    seen.append('finally')
""")
            == 5
        )
        assert ns.globals["seen"] == [(None, None, None), (None, None, None), "finally"]


async def test_partial_error_and_cancel_write_through():
    ns = namespace(asyncio=asyncio)
    with ns.enter({}):
        with pytest.raises(ZeroDivisionError):
            await ns.cell("partial = 1\n1 / 0")
        assert ns.globals["partial"] == 1
        gate = asyncio.Event()
        ns.globals["gate"] = gate
        task = asyncio.create_task(
            ns.cell("before = 1\ntry:\n await gate.wait()\nfinally:\n after = 2")
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ns.globals["before"] == 1 and ns.globals["after"] == 2
    with ns.enter({}):
        assert await ns.cell("return after") == 2


async def test_originating_inputs_background_and_foreground_contexts():
    ns = namespace(asyncio=asyncio)
    gate = asyncio.Event()
    ns.globals["gate"] = gate
    with ns.enter({"notification": "A"}):
        await ns.cell(
            "async def worker():\n await gate.wait()\n return notification\ndef direct():\n return notification"
        )
        worker = asyncio.create_task(ns.globals["worker"]())
    with ns.enter({"notification": "B"}):
        assert await ns.cell("return direct()") == "B"
        gate.set()
        assert await worker == "A"


async def test_owner_isolation_reentrancy_release():
    one, two = namespace(), namespace()
    with one.enter({"notification": "one"}):
        with pytest.raises(RuntimeError, match="Concurrent/reentrant"):
            with one.enter({}):
                pass
        with two.enter({"notification": "two"}):
            assert await two.cell("return notification") == "two"
        assert await one.cell("return notification") == "one"
    with one.enter({"notification": "new"}):
        assert await one.cell("return notification") == "new"


async def test_original_source_traceback_line():
    ns = namespace()
    with ns.enter({}):
        with pytest.raises(ZeroDivisionError) as caught:
            await ns.cell("x = 1\n1 / 0\n", filename="Cell In[313]")
    frame = traceback.extract_tb(caught.value.__traceback__)[-1]
    assert frame.filename == "Cell In[313]" and frame.lineno == 2
    assert frame.line == "1 / 0"


async def test_completion_requires_innermost_call_identity_even_in_same_executor_task():
    one, two = namespace(), namespace()
    with one.enter({}):
        one.bind_call({"return_result": lambda value: value})
        with one.executing_cell():
            alias = await one.cell("return return_result")
            assert alias(1) == 1
            with two.enter({}):
                with two.executing_cell():
                    with pytest.raises(RuntimeError, match="foreground"):
                        alias(2)
            assert alias(3) == 3
    with pytest.raises(RuntimeError, match="foreground"):
        alias(4)


async def test_nonlocal_to_lifted_cell_has_clear_diagnostic_but_true_nested_closure_works():
    ns = namespace()
    with ns.enter({}):
        with pytest.raises(SyntaxError, match="nonlocal"):
            await ns.cell("x=1\ndef f():\n nonlocal x\n return x")
        assert await ns.cell(
            "def outer():\n x=1\n def inner():\n  nonlocal x\n  x+=1\n  return x\n return inner\nf=outer()\nreturn f(), f()"
        ) == (2, 3)


async def test_cleanup_stale_epoch_dependencies_and_opaque_values():
    ns = namespace()
    with ns.enter({}):
        await ns.cell("scratch=1\nkeep=2\ndef helper():\n return keep")
    snapshot = ns.cleanup_snapshot()
    assert ns.cleanup_candidates(snapshot) == ["scratch"]
    assert not ns.apply_cleanup(snapshot, ["keep"], active_work=False)
    assert not ns.apply_cleanup(snapshot, ["scratch"], active_work=True)
    with ns.enter({}):
        await ns.cell("scratch=1")  # even same-object rebinding invalidates the plan
    assert not ns.apply_cleanup(snapshot, ["scratch"], active_work=False)
    with ns.enter({}):
        await ns.cell("del helper")
    # Deleted helper aliases may still exist externally: its footprint remains.
    assert "keep" not in ns.cleanup_candidates(ns.cleanup_snapshot())
    assert ns.apply_cleanup(ns.cleanup_snapshot(), ["scratch"], active_work=False)
    with ns.enter({}):
        await ns.cell("scratch=1\nopaque=[]")
    assert ns.cleanup_candidates(ns.cleanup_snapshot()) == []


async def test_dynamic_provider_is_fallback_and_static_seeds_never_refresh():
    ns = namespace()
    first, second = object(), object()
    ns.update_framework({"Fresh": first})
    ns.initialize({"Fresh": first, "STATIC": 1})
    with ns.enter({}):
        assert await ns.cell("def read():\n return Fresh\nreturn read()") is first
        await ns.cell("alias=Fresh")
    assert "Fresh" not in ns.globals
    original_globals = ns.globals
    ns.update_framework({"Fresh": second, "STATIC": 99})
    ns.initialize({"STATIC": 999, "UNRELATED_MODULE_STATE": 123})
    with ns.enter({}):
        assert await ns.cell("return read(), alias, STATIC") == (second, first, 1)
    assert ns.globals is original_globals and ns.globals["read"].__globals__ is original_globals
    assert "UNRELATED_MODULE_STATE" not in ns.globals
    assert ns.advertised_framework({"STATIC": 999, "UNRELATED_MODULE_STATE": 123}) == {}
    assert ns.advertised_framework({"STATIC": 1}) == {"STATIC": 1}
    ns.update_framework({})
    with ns.enter({}):
        with pytest.raises(NameError):
            await ns.cell("read()")
        assert await ns.cell("return alias, STATIC") == (first, 1)


async def test_global_delete_provenance_is_execution_and_scope_correct():
    ns = namespace()
    value = object()
    ns.update_framework({"Fresh": value})
    with ns.enter({}):
        assert await ns.cell("""
if False:
    del Fresh
def local(Fresh):
    del Fresh
local(3)
class C:
    Fresh = 4
    del Fresh
class Reader:
    Fresh = 'class'
    own = Fresh
    def read(self):
        return Fresh
def outer(Fresh):
    def nonlocal_delete():
        nonlocal Fresh
        del Fresh
    nonlocal_delete()
outer(5)
return Fresh, Reader.own, Reader().read()
""") == (value, "class", value)
        assert "Fresh" not in ns.deleted_names
        await ns.cell("def remove():\n global Fresh\n del Fresh\nremove()")
        assert "del Fresh" in ns.globals["remove"]._generated_source
        assert "_nooa_ns_delete" not in ns.globals["remove"]._generated_source
        with pytest.raises(NameError):
            await ns.cell("Fresh")
        await ns.cell("Fresh='user'\ndel Fresh")
    ns.update_framework({"Fresh": object()})
    with ns.enter({}):
        with pytest.raises(NameError):
            await ns.cell("Fresh")
        await ns.cell("Fresh=42")
        assert await ns.cell("return Fresh") == 42


async def test_global_delete_partial_effects_and_failed_delete_are_not_faked():
    ns = namespace()
    first, last = object(), object()
    ns.update_framework({"first": first, "last": last})
    with ns.enter({}):
        with pytest.raises(NameError):
            await ns.cell("del (first, missing, last)")
        assert ns.deleted_names == {"first"}
        assert await ns.cell("return last") is last
    ns.update_framework({"first": object(), "missing": 7, "last": last})
    with ns.enter({}):
        assert await ns.cell("return missing") == 7
        with pytest.raises(NameError):
            await ns.cell("first")
        await ns.cell("class C:\n global last\n del last")
        assert ns.deleted_names == {"first", "last"}


async def test_dynamic_class_named_like_builtin_and_implicit_exception_cleanup():
    ns = namespace()
    custom = object()
    ns.update_framework({"list": custom, "ExceptionAlias": custom})
    with ns.enter({}):
        assert await ns.cell("return list") is custom
        await ns.cell("del list")
        assert await ns.cell("return list") is list  # ordinary Python builtin survives
        await ns.cell("try:\n raise ValueError()\nexcept ValueError as ExceptionAlias:\n pass")
        assert "ExceptionAlias" not in ns.globals
        assert await ns.cell("return ExceptionAlias") is custom
        await ns.cell("del ExceptionAlias")
        with pytest.raises(NameError):
            await ns.cell("ExceptionAlias")


@pytest.mark.parametrize("initialized", [False, True])
async def test_advertising_and_availability_share_reserved_internal_exclusions(initialized):
    names = {
        "notification",
        "self",
        "return_result",
        "print",
        "pprint",
        "show",
        "python_cell_state",
        "Out",
    }
    ns = ExecutionNamespace(reserved=frozenset(names))
    blocked = names | {"__builtins__", "_nooa_ns_source", "_nooa_ns_probe"}
    values = {name: object() for name in blocked}
    good = object()
    values["OrdinaryDynamic"] = good
    ns.update_framework(values)
    assert ns.framework == {"OrdinaryDynamic": good}
    if initialized:
        ns.initialize(values)
    assert ns.advertised_framework(values, dynamic_names=values) == {"OrdinaryDynamic": good}
    # Even matching a real internal value does not make it provider-advertisable.
    assert (
        ns.advertised_framework(
            {"__builtins__": ns.globals["__builtins__"]}, dynamic_names={"__builtins__"}
        )
        == {}
    )
    with ns.enter({"notification": "input", "self": "owner"}):
        assert await ns.cell("return notification, self, OrdinaryDynamic") == (
            "input",
            "owner",
            good,
        )
