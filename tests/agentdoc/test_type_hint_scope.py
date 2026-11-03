# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formatting reuses annotations without retaining visibility decisions."""

from __future__ import annotations

import asyncio
import gc
import typing
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass

import pytest

from nooa.agentdoc import hidden, pformat, pprint, spec, truncating_pformat
from nooa.agentdoc._visibility import _resolve_type_hints, _type_hint_scope, is_hidden_field


@dataclass
class Record:
    value: int
    secret: typing.Annotated[str, hidden] = "PRIVATE"


@pytest.mark.parametrize("render", [pformat, truncating_pformat, pprint])
def test_repeated_records_reuse_hints_but_keep_instance_overrides(render, monkeypatch, capsys):
    calls = Counter()
    original = typing.get_type_hints

    def counted(obj, *args, **kwargs):
        calls[obj] += 1
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(typing, "get_type_hints", counted)
    shown = Record(1, "VISIBLE")
    spec(shown, "secret", hidden=False)
    records = [Record(0), shown] * 20
    result = render(records)
    if result is None:
        result = capsys.readouterr().out
    assert "PRIVATE" not in result
    assert result.count("VISIBLE") == 20
    assert calls[Record] == 1


def test_later_render_observes_annotation_changes():
    @dataclass
    class Mutable:
        value: int = 1
        other: int = 2

    item = Mutable()
    assert "value=1" in pformat(item)
    Mutable.__annotations__["value"] = typing.Annotated[int, hidden]
    assert "value=1" not in pformat(item)
    Mutable.__annotations__["value"] = int
    assert "value=1" in pformat(item)


def test_unresolved_annotations_can_resolve_later_in_scope(monkeypatch):
    class Forward:
        value: FutureHint  # noqa: F821 -- installed after the first resolution attempt

    with _type_hint_scope():
        assert not is_hidden_field(Forward, "value")
        monkeypatch.setitem(globals(), "FutureHint", typing.Annotated[int, hidden])
        assert is_hidden_field(Forward, "value")


def test_scope_restores_outer_cache_and_cleans_up_after_exception(monkeypatch):
    calls = 0
    original = typing.get_type_hints

    def counted(obj, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(typing, "get_type_hints", counted)
    with pytest.raises(RuntimeError, match="render failed"), _type_hint_scope():
        assert is_hidden_field(Record, "secret")
        with _type_hint_scope():
            assert is_hidden_field(Record, "secret")
        assert is_hidden_field(Record, "secret")
        assert calls == 1
        raise RuntimeError("render failed")
    assert is_hidden_field(Record, "secret")
    assert calls == 2


@pytest.mark.parametrize("fail", [False, True])
def test_copied_context_expires_with_render(fail):
    @dataclass
    class Mutable:
        value: int = 1
        other: int = 2

    saved = []

    class Capture:
        def __repr__(self):
            saved.append(copy_context())
            return "Capture()"

    if fail:
        with pytest.raises(RuntimeError, match="render failed"), _type_hint_scope():
            pformat([Mutable(), Capture()])
            raise RuntimeError("render failed")
    else:
        pformat([Mutable(), Capture()])

    Mutable.__annotations__["value"] = typing.Annotated[int, hidden]
    assert "value=1" not in pformat(Mutable())
    # Direct inspection must not revive an expired cache either.
    assert saved[0].run(is_hidden_field, Mutable, "value")
    assert "value=1" not in saved[0].run(pformat, Mutable())
    Mutable.__annotations__["value"] = int
    assert "value=1" in saved[0].run(pformat, Mutable())


def test_copied_context_uses_separate_cache_in_another_thread():
    @dataclass
    class Mutable:
        value: int = 1
        other: int = 2

    with _type_hint_scope():
        assert not is_hidden_field(Mutable, "value")
        copied = copy_context()
        Mutable.__annotations__["value"] = typing.Annotated[int, hidden]
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(copied.run, is_hidden_field, Mutable, "value").result()
            assert "value=1" not in pool.submit(copied.run, pformat, Mutable()).result()
        # Work in the other thread did not replace or expire our render cache.
        assert not is_hidden_field(Mutable, "value")


def test_copied_context_does_not_retain_resolved_objects():
    def target(value: int) -> int:
        return value

    reference = weakref.ref(target)
    with _type_hint_scope():
        _resolve_type_hints(target)
        copied = copy_context()
    del target
    gc.collect()
    assert reference() is None
    # Keep the copied context alive until after checking the object's lifetime.
    assert copied.run(lambda: True)


async def test_task_created_during_render_starts_with_fresh_annotations():
    @dataclass
    class Mutable:
        value: int = 1
        other: int = 2

    tasks = []

    async def render_later():
        return pformat(Mutable())

    class Capture:
        def __repr__(self):
            tasks.append(asyncio.create_task(render_later()))
            return "Capture()"

    pformat([Mutable(), Capture()])
    Mutable.__annotations__["value"] = typing.Annotated[int, hidden]
    assert "value=1" not in await tasks[0]
