# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell source lives as long as code compiled from the cell, at most as long as its namespace."""

import gc
import inspect
import linecache
import traceback
import weakref
from uuid import uuid4

import pytest

from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace


def filename():
    return f"<namespace-lifetime-{uuid4().hex}>"


async def test_discarded_namespaces_release_all_their_sources():
    async def run():
        owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
        names = [filename(), filename()]
        with owner.enter({}):
            for index, name in enumerate(names):
                await owner.cell(f"def helper_{index}(): pass", name)
                assert linecache.getline(name, 1) == f"def helper_{index}(): pass"
        return weakref.ref(owner), names

    discarded = [await run() for _ in range(8)]
    gc.collect()
    for reference, names in discarded:
        assert reference() is None
        for name in names:
            linecache.checkcache(name)
            assert name not in linecache.cache


async def test_retained_namespace_keeps_no_source_of_a_cell_that_left_no_code():
    name = filename()
    owner = ExecutionNamespace(
        reserved=RUNTIME_NAMES, initial={"name": name, "linecache": linecache}
    )
    with owner.enter({}):
        # While the cell runs, its lines are there for a warning or a traceback.
        seen = await owner.cell("value = 1\nreturn linecache.getline(name, 2)", name)
    assert seen == "return linecache.getline(name, 2)"
    assert name not in linecache.cache and not owner._source_entries


async def test_source_does_not_accumulate_with_the_number_of_cells():
    owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
    names = [filename() for _ in range(250)]
    cells = [
        "value = {n}\ntotal = sum(range(value))",
        "def scratch(rows):\n    return [row * {n} for row in rows]\nresult = scratch(range(5))",
        "class Row:\n    def scale(self):\n        return {n}\nrow = Row()",
        "square = lambda x: x * {n}",
        "raise ValueError({n})",
    ]
    with owner.enter({}):
        for number, name in enumerate(names):
            try:
                await owner.cell(cells[number % len(cells)].format(n=number), name)
            except ValueError:
                pass
    gc.collect()  # a replaced class is freed by the cyclic collector
    kept = {name for name in names if name in linecache.cache}
    # Only what can still run: the current scratch(), Row.scale() and square().
    assert len(kept) == 3 and set(owner._source_entries) == kept
    assert sorted(linecache.getline(name, 1).strip() for name in kept) == [
        "class Row:",
        "def scratch(rows):",
        "square = lambda x: x * 248",
    ]


async def test_traceback_through_a_helper_defined_many_cells_earlier():
    owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
    first = filename()
    source = "def helper(x):\n    # the line that fails\n    return 1 / x\n"
    with owner.enter({}):
        await owner.cell(source, first)
        for number in range(250):
            await owner.cell(f"value = {number}", filename())
        with pytest.raises(ZeroDivisionError) as raised:
            await owner.cell("helper(0)", filename())
    frame = traceback.extract_tb(raised.value.__traceback__)[-1]
    assert (frame.filename, frame.lineno, frame.line) == (first, 3, "return 1 / x")
    assert '", line 3, in helper\n    return 1 / x\n' in "".join(
        traceback.format_exception(raised.value)
    )
    assert inspect.getsource(owner.globals["helper"]) == source
    del raised, frame
    gc.collect()
    assert list(owner._source_entries) == [first]
    # Rebinding the helper releases the source of the cell that defined it.
    with owner.enter({}):
        await owner.cell("helper = None", filename())
    assert first not in linecache.cache and not owner._source_entries


@pytest.mark.parametrize(
    ("source", "bound"),
    [
        (
            "def outer():\n    def inner():\n        return 1 / 0\n    return inner\nreturn outer()",
            "outer",
        ),
        ("class Thing:\n    def run(self):\n        return 1 / 0\nreturn Thing().run", "Thing"),
        ("return lambda: (\n    1 / 0\n)", ""),
        ("async def later():\n    return 1 / 0\nreturn later", "later"),
        ("def numbers():\n    yield 1 / 0\nreturn numbers().__next__", "numbers"),
    ],
)
async def test_source_lives_as_long_as_code_of_the_cell_can_still_run(source, bound):
    owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
    name = filename()
    with owner.enter({}):
        held = await owner.cell(source, name)
        if bound:
            await owner.cell(f"del {bound}", filename())
    gc.collect()
    try:
        if inspect.iscoroutinefunction(held):
            await held()
        else:
            held()
    except ZeroDivisionError as error:
        frame = traceback.extract_tb(error.__traceback__)[-1]
        assert frame.filename == name and "1 / 0" in frame.line
    else:
        pytest.fail("the cell's code did not raise")
    del held, frame
    gc.collect()
    assert name not in linecache.cache and not owner._source_entries


async def test_source_release_tolerates_an_entry_checked_or_removed_by_others():
    owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
    kept, cleared = filename(), filename()
    with owner.enter({}):
        await owner.cell("def kept(): pass", kept)
        await owner.cell("def cleared(): pass", cleared)
        linecache.checkcache()
        linecache.checkcache(kept)
        assert linecache.getlines(kept) == ["def kept(): pass"]
        del linecache.cache[cleared]  # as linecache.clearcache() by someone else would
        await owner.cell("del cleared", filename())
        assert cleared not in owner._source_entries
        await owner.cell("del kept", filename())
    assert kept not in linecache.cache and not owner._source_entries


async def test_retained_helper_keeps_owner_and_traceback_source():
    async def run():
        owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
        name = filename()
        with owner.enter({}):
            helper = await owner.cell("def helper():\n    return 1 / 0\nreturn helper", name)
        return weakref.ref(owner), name, helper

    reference, name, helper = await run()
    gc.collect()
    assert reference() is not None
    linecache.checkcache(name)
    assert linecache.getline(name, 2) == "    return 1 / 0\n"

    def check_traceback(function):
        try:
            function()
        except ZeroDivisionError as error:
            frame = traceback.extract_tb(error.__traceback__)[-1]
            assert (frame.filename, frame.lineno, frame.line) == (name, 2, "return 1 / 0")
        else:
            pytest.fail("helper did not raise")

    check_traceback(helper)
    del helper
    gc.collect()
    assert reference() is None and name not in linecache.cache


async def test_collection_does_not_remove_another_owners_reused_filename():
    name = filename()

    async def run():
        first = ExecutionNamespace(reserved=RUNTIME_NAMES)
        second = ExecutionNamespace(reserved=RUNTIME_NAMES)
        with first.enter({}):
            await first.cell("def value(): pass", name)
        old_entry = linecache.cache[name]
        with second.enter({}):
            await second.cell("def value(): pass", name)
        assert linecache.cache[name] == old_entry
        assert linecache.cache[name] is not old_entry
        return weakref.ref(first), second

    first, second = await run()
    gc.collect()
    assert first() is None
    assert linecache.getline(name, 1) == "def value(): pass"
    del second
    gc.collect()
    assert name not in linecache.cache


async def test_collection_leaves_an_equal_but_replaced_cache_entry():
    name = filename()

    async def run():
        owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
        with owner.enter({}):
            await owner.cell("def first(): pass", name)
            await owner.cell("def second(): pass", name)
        old = linecache.cache[name]
        replacement = (*old,)
        assert replacement == old and replacement is not old
        linecache.cache[name] = replacement
        return weakref.ref(owner), replacement

    reference, replacement = await run()
    try:
        gc.collect()
        assert reference() is None
        assert linecache.cache[name] is replacement
    finally:
        if linecache.cache.get(name) is replacement:
            del linecache.cache[name]


@pytest.mark.parametrize("source", ["if", "_call = 1", "yield 1\nreturn 2"])
@pytest.mark.parametrize("existing", [False, True])
async def test_compile_failure_does_not_publish_source(source, existing):
    owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
    name = filename()
    if existing:
        with owner.enter({}):
            await owner.cell("def value(): pass", name)
    previous = linecache.cache.get(name)
    assert (previous is not None) == existing
    with owner.enter({}), pytest.raises(SyntaxError):
        await owner.cell(source, name)
    assert linecache.cache.get(name) is previous


async def test_runtime_failure_source_lives_with_retained_traceback():
    async def run():
        owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
        name = filename()
        with owner.enter({}):
            try:
                await owner.cell("1 / 0", name)
            except ZeroDivisionError as error:
                return weakref.ref(owner), name, error
        pytest.fail("cell did not raise")

    reference, name, error = await run()
    gc.collect()
    assert reference() is not None
    assert traceback.extract_tb(error.__traceback__)[-1].line == "1 / 0"
    del error
    gc.collect()
    assert reference() is None and name not in linecache.cache


async def test_reserved_call_is_never_seeded_or_offered_by_a_provider():
    first, second, ordinary = object(), object(), object()
    owner = ExecutionNamespace(
        reserved=RUNTIME_NAMES, initial={"_call": first, "ordinary": ordinary}
    )
    owner.update_framework({"_call": first})
    owner.initialize({"_call": first})
    assert "_call" not in owner.globals and "_call" not in owner.framework
    assert owner.advertised_framework({"_call": first}, dynamic_names={"_call"}) == {}
    for value in (first, second):
        with owner.enter({}):
            owner.bind_call({"_call": value})
            assert await owner.cell("return _call, ordinary") == (value, ordinary)
            owner.bind_call({})
            with pytest.raises(RuntimeError, match="originating call context"):
                await owner.cell("return _call")
    assert "_call" not in owner.globals
