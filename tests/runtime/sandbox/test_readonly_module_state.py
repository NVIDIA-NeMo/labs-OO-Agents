# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandbox cell mutating non-``self`` module-level state must fail LOUD.

Reads/calls/iteration of module-level state still work; only mutation raises
``SandboxStateError``. Immutables, functions, classes and modules are untouched.
"""

from __future__ import annotations

from typing import Any

import pytest

from nooa import Agent
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.errors import SandboxExecutionError
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.readonly import (
    ReadOnlyView,
    SandboxStateError,
    freeze_module_state,
)
from nooa.unifiedllm import FakeLLMClient

# --------------------------------------------------------------------------- #
# unit: ReadOnlyView / freeze_module_state (no worker process)
# --------------------------------------------------------------------------- #


def test_readonly_view_reads_forward_mutations_raise():
    d = ReadOnlyView({"a": 1}, "D")
    assert d["a"] == 1 and len(d) == 1 and "a" in d and dict(d.items()) == {"a": 1}
    with pytest.raises(SandboxStateError):
        d["b"] = 2
    with pytest.raises(SandboxStateError):
        d.update({"b": 2})

    lst = ReadOnlyView([1, 2], "L")
    assert lst[0] == 1 and list(lst) == [1, 2] and len(lst) == 2
    with pytest.raises(SandboxStateError):
        lst.append(3)
    with pytest.raises(SandboxStateError):
        lst[0] = 9

    s = ReadOnlyView({1, 2}, "S")
    assert 1 in s and len(s) == 2
    with pytest.raises(SandboxStateError):
        s.add(3)


def test_readonly_view_blocks_instance_attr_mutation():
    class Box:
        def __init__(self):
            self.x = 1

        def read(self):
            return self.x

    v = ReadOnlyView(Box(), "OBJ")
    assert v.x == 1 and v.read() == 1  # reads + method calls forward
    with pytest.raises(SandboxStateError):
        v.x = 2
    with pytest.raises(SandboxStateError):
        del v.x


def test_freeze_wraps_only_mutable_data():
    def fn():
        return 1

    class C:
        pass

    ns = {
        "MUT_LIST": [1],
        "MUT_DICT": {"a": 1},
        "MUT_SET": {1},
        "CONST_INT": 5,
        "CONST_STR": "x",
        "CONST_TUPLE": (1, 2),
        "A_FUNC": fn,
        "A_CLASS": C,
        "__dunder__": [1],  # left alone
    }
    freeze_module_state(ns)
    assert isinstance(ns["MUT_LIST"], ReadOnlyView)
    assert isinstance(ns["MUT_DICT"], ReadOnlyView)
    assert isinstance(ns["MUT_SET"], ReadOnlyView)
    # untouched: immutables, callables, classes, dunders
    assert ns["CONST_INT"] == 5 and ns["CONST_STR"] == "x" and ns["CONST_TUPLE"] == (1, 2)
    assert ns["A_FUNC"] is fn and ns["A_CLASS"] is C
    assert ns["__dunder__"] == [1] and not isinstance(ns["__dunder__"], ReadOnlyView)


# --------------------------------------------------------------------------- #
# integration: through the real sandbox worker
# --------------------------------------------------------------------------- #

# module-level mutable state (NOT on self) — the divergence vector
SHARED_DICT: dict[str, int] = {"n": 0}
SHARED_LIST: list[int] = [0]
CONST_LIMIT = 20  # immutable — must stay usable


class _Agent(Agent, llm=FakeLLMClient()):
    pass


def _rr_builtins() -> dict[str, Any]:
    from nooa.strategies.codeact import _ReturnResultSignal

    def return_result(*a: Any, **k: Any):
        raise _ReturnResultSignal(result={"result": a[0] if a else k})

    return {"return_result": return_result}


def _executor() -> SandboxedExecutor:
    return SandboxedExecutor(
        _Agent(),
        SandboxConfig(require=False),
        cell_timeout=10.0,
        framework_builtins=_rr_builtins(),
    )


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_cell_mutating_module_dict_fails_loud():
    ex = _executor()
    try:
        r = await ex.run_cell("SHARED_DICT['n'] = 99", execution_count=1)
        assert isinstance(r.error, SandboxExecutionError)
        assert r.error.original_type == "SandboxStateError"
        assert isinstance(r.error.original_error, SandboxStateError)
        assert "module-level state" in str(r.error)
    finally:
        await ex.aclose()


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_cell_mutating_module_list_fails_loud():
    ex = _executor()
    try:
        r = await ex.run_cell("SHARED_LIST.append(7)", execution_count=1)
        assert r.error is not None and "module-level state" in str(r.error)
    finally:
        await ex.aclose()


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_cell_reading_module_state_still_works():
    ex = _executor()
    try:
        r = await ex.run_cell(
            "print(SHARED_DICT['n'], SHARED_LIST[0], CONST_LIMIT, len(SHARED_LIST))",
            execution_count=1,
        )
        assert r.error is None, r.error
        assert "0 0 20 1" in r.stdout
    finally:
        await ex.aclose()
