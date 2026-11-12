# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Python differentials for lexical private-name global deletion."""

import textwrap

import pytest

from nooa import Agent
from nooa.runtime.execution_namespace import ExecutionNamespace
from nooa.unifiedllm import FakeLLMClient


async def native_namespace(source, initial):
    """Use CPython itself as the mangling oracle, with a real async wrapper."""
    values = dict(initial)
    wrapper = "async def _native_cell():\n" + textwrap.indent(source, "    ")
    exec(compile(wrapper, "<native-private-delete>", "exec"), values)
    function = values.pop("_native_cell")
    try:
        await function()
    except Exception as error:
        return values, error
    return values, None


def private_source(class_name, spelling, site, *, delete=True):
    body = f"global {spelling}\n{spelling} = 'private'\n"
    body += f"del {spelling}" if delete else "pass"
    if site == "class":
        return f"class {class_name}:\n" + textwrap.indent(body, "    ")
    if site == "nested_function":
        return (
            f"class {class_name}:\n    def drop():\n        def inner():\n"
            + textwrap.indent(body, "            ")
            + f"\n        inner()\n{class_name}.drop()"
        )
    prefix, call = ("async ", "await ") if site == "async_method" else ("", "")
    return (
        f"class {class_name}:\n    {prefix}def drop():\n"
        + textwrap.indent(body, "        ")
        + f"\n{call}{class_name}.drop()"
    )


@pytest.mark.parametrize("class_name", ["C", "_C", "__C", "___"])
@pytest.mark.parametrize("spelling", ["__x", "__x__"])
@pytest.mark.parametrize("site", ["class", "method", "nested_function", "async_method"])
@pytest.mark.parametrize("raw_present", [False, True])
async def test_private_global_delete_matches_native_binding_and_tombstone(
    class_name,
    spelling,
    site,
    raw_present,
):
    initial = {spelling: "unrelated raw binding"} if raw_present else {}
    stored, error = await native_namespace(
        private_source(class_name, spelling, site, delete=False), initial
    )
    assert error is None
    [actual_name] = [
        name for name, value in stored.items() if type(value) is str and value == "private"
    ]
    source = private_source(class_name, spelling, site)
    expected, error = await native_namespace(source, initial)
    assert error is None
    owner = ExecutionNamespace(reserved=frozenset(), initial=initial)
    with owner.enter({}):
        await owner.cell(source)
    keys = {actual_name, spelling}
    assert {key: owner.globals[key] for key in keys if key in owner.globals} == {
        key: expected[key] for key in keys if key in expected
    }
    assert actual_name not in owner.globals
    assert owner.deleted_names == {actual_name}
    if actual_name != spelling and raw_present:
        assert owner.globals[spelling] == "unrelated raw binding"


@pytest.mark.parametrize("method", [False, True])
async def test_nested_private_class_context_changes_and_restores(method):
    body = """global __x
__x = 'outer private'
class __Inner:
    global __x
    __x = 'inner private'
    del __x
del __x"""
    if method:
        source = (
            "class Outer:\n    def drop():\n" + textwrap.indent(body, "        ") + "\nOuter.drop()"
        )
    else:
        source = "class Outer:\n" + textwrap.indent(body, "    ")
    source += "\nglobal __after\n__after='after class'\ndel __after"
    initial = {"__x": "raw x", "__after": "raw after"}
    expected, error = await native_namespace(source, initial)
    assert error is None
    owner = ExecutionNamespace(reserved=frozenset(), initial=initial)
    with owner.enter({}):
        await owner.cell(source)
    keys = {"__x", "_Outer__x", "_Inner__x", "__after", "_Outer__after"}
    assert {key: owner.globals[key] for key in keys if key in owner.globals} == {
        key: expected[key] for key in keys if key in expected
    }
    assert owner.globals["__x"] == "raw x"
    assert owner.deleted_names == {"_Outer__x", "_Inner__x", "__after"}


@pytest.mark.parametrize("declaration,deletion", [("__x", "_C__x"), ("_C__x", "__x")])
async def test_global_declaration_and_delete_can_use_equivalent_private_spellings(
    declaration, deletion
):
    source = f"class C:\n global {declaration}\n __x='private'\n del {deletion}"
    initial = {"__x": "raw"}
    expected, error = await native_namespace(source, initial)
    assert error is None and expected["__x"] == "raw" and "_C__x" not in expected
    owner = ExecutionNamespace(reserved=frozenset(), initial=initial)
    with owner.enter({}):
        await owner.cell(source)
    assert owner.globals["__x"] == "raw" and "_C__x" not in owner.globals
    assert owner.deleted_names == {"_C__x"}


async def test_private_local_class_and_nonlocal_deletes_do_not_touch_globals():
    source = """
class C:
    __x = 'class local'
    del __x
    def local(__x):
        del __x
    def outer():
        __x = 'closure local'
        def inner():
            nonlocal __x
            del __x
        inner()
C.local('parameter')
C.outer()
"""
    initial = {"__x": "raw", "_C__x": "independent global"}
    expected, error = await native_namespace(source, initial)
    assert error is None
    owner = ExecutionNamespace(reserved=frozenset(), initial=initial)
    with owner.enter({}):
        await owner.cell(source)
    assert {key: owner.globals[key] for key in initial} == {key: expected[key] for key in initial}
    assert owner.deleted_names == set()


async def test_private_partial_delete_failed_target_and_provider_reactivation():
    source = """
class C:
    global __first, __missing, __last
    __first = 'first'
    __last = 'last'
    if False:
        del __last
    del (__first, __missing, __last)
"""
    initial = {"__first": "raw first", "__missing": "raw missing", "__last": "raw last"}
    expected, error = await native_namespace(source, initial)
    assert isinstance(error, NameError)
    owner = ExecutionNamespace(reserved=frozenset(), initial=initial)
    with owner.enter({}):
        with pytest.raises(NameError) as caught:
            await owner.cell(source)
    assert str(caught.value) == str(error)
    assert caught.value.name == error.name == "_C__missing"
    keys = {*initial, "_C__first", "_C__missing", "_C__last"}
    assert {key: owner.globals[key] for key in keys if key in owner.globals} == {
        key: expected[key] for key in keys if key in expected
    }
    assert owner.deleted_names == {"_C__first"}
    owner.update_framework({"_C__first": object(), "_C__missing": "new provider"})
    with owner.enter({}):
        await owner.cell("class C:\n def read():\n  return __first")
        with pytest.raises(NameError):
            await owner.cell("C.read()")
        assert await owner.cell("return _C__missing") == "new provider"
        assert await owner.cell("return __first") == "raw first"


class PrivateDeleteAgent(Agent, llm=FakeLLMClient()):
    pass


@pytest.mark.parametrize("raw_present", [False, True])
async def test_private_delete_through_actor_preserves_raw_binding_and_original_metadata(
    raw_present,
):
    agent = PrivateDeleteAgent()
    initial = {"__x": "unrelated raw"} if raw_present else {}
    owner = ExecutionNamespace(reserved=frozenset({"self"}), initial=initial)
    try:
        with owner.enter({}):
            result = await agent.runtime.execute_code(
                private_source("C", "__x", "nested_function"),
                execution_namespace=owner,
                wrap_in_function=True,
            )
        assert result.error is None, result.error
        assert "_C__x" not in owner.globals
        assert ("__x" in owner.globals) is raw_present
        if raw_present:
            assert owner.globals["__x"] == "unrelated raw"
        assert owner.deleted_names == {"_C__x"}
        source = owner.globals["C"].drop._generated_source
        assert "del __x" in source and "_nooa_ns_delete" not in source and "_C__x" not in source
    finally:
        await agent.aclose()
