# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-cache lifetime follows the namespace, including externally held helpers."""

import gc
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
            for name in names:
                await owner.cell("value = 1", name)
                assert linecache.getline(name, 1) == "value = 1"
        return weakref.ref(owner), names

    discarded = [await run() for _ in range(8)]
    gc.collect()
    for reference, names in discarded:
        assert reference() is None
        for name in names:
            linecache.checkcache(name)
            assert name not in linecache.cache


async def test_retained_namespace_keeps_source_without_a_helper():
    owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
    name = filename()
    with owner.enter({}):
        await owner.cell("value = 1", name)
    gc.collect()
    linecache.checkcache(name)
    assert linecache.getline(name, 1) == "value = 1"
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None and name not in linecache.cache


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
            await first.cell("value = 1", name)
        old_entry = linecache.cache[name]
        with second.enter({}):
            await second.cell("value = 1", name)
        assert linecache.cache[name] == old_entry
        assert linecache.cache[name] is not old_entry
        return weakref.ref(first), second

    first, second = await run()
    gc.collect()
    assert first() is None
    assert linecache.getline(name, 1) == "value = 1"
    del second
    gc.collect()
    assert name not in linecache.cache


async def test_collection_leaves_an_equal_but_replaced_cache_entry():
    name = filename()

    async def run():
        owner = ExecutionNamespace(reserved=RUNTIME_NAMES)
        with owner.enter({}):
            await owner.cell("value = 1", name)
            await owner.cell("value = 2", name)
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
            await owner.cell("value = 1", name)
    previous = linecache.cache.get(name)
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
