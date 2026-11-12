# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formatting reuses annotations without retaining visibility decisions."""

from __future__ import annotations

import gc
import inspect
import typing
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass

import pytest

from nooa.agentdoc import hidden, pformat, spec, truncating_pformat
from nooa.agentdoc._visibility import _type_hint_scope, is_hidden_field


@dataclass
class Record:
    value: int
    secret: typing.Annotated[str, hidden] = "PRIVATE"


def test_repeated_records_reuse_hints_but_keep_instance_overrides(monkeypatch):
    calls = Counter()
    original = typing.get_type_hints

    def counted(obj, *args, **kwargs):
        calls[obj] += 1
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(typing, "get_type_hints", counted)
    shown = Record(1, "VISIBLE")
    spec(shown, "secret", hidden=False)
    records = [Record(0), shown] * 20
    result = truncating_pformat(records)
    assert "PRIVATE" not in result
    assert result.count("VISIBLE") == 20
    assert calls[Record] == 1


def test_unresolved_annotations_can_resolve_later_in_scope(monkeypatch):
    class Forward:
        value: FutureHint  # noqa: F821 -- installed after the first resolution attempt

    with _type_hint_scope():
        assert not is_hidden_field(Forward, "value")
        monkeypatch.setitem(globals(), "FutureHint", typing.Annotated[int, hidden])
        assert is_hidden_field(Forward, "value")


@pytest.mark.parametrize("fail", [False, True])
def test_copied_context_expires_and_releases_cached_types(fail):
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
        with pytest.raises(RuntimeError, match="formatting failed"), _type_hint_scope():
            pformat([Mutable(), Capture()])
            raise RuntimeError("formatting failed")
    else:
        pformat([Mutable(), Capture()])

    Mutable.__annotations__["value"] = typing.Annotated[int, hidden]
    assert "value=1" not in pformat(Mutable())
    # Direct inspection must not revive an expired cache either.
    assert saved[0].run(is_hidden_field, Mutable, "value")
    assert "value=1" not in saved[0].run(pformat, Mutable())
    Mutable.__annotations__["value"] = int
    assert "value=1" in saved[0].run(pformat, Mutable())

    reference = weakref.ref(Mutable)
    del Mutable
    # CPython 3.12's getattr_static() keeps a bounded cache of class MROs.
    # Release that unrelated reference before testing our own cache's lifetime.
    shadowed_dict = getattr(inspect, "_shadowed_dict_from_mro_tuple", None)
    if hasattr(shadowed_dict, "cache_clear"):
        shadowed_dict.cache_clear()
    gc.collect()
    assert reference() is None


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
        # Work in the other thread did not replace or expire our formatting cache.
        assert not is_hidden_field(Mutable, "value")
