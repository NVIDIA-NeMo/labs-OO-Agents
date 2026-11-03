# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formatting reuses annotations without retaining visibility decisions."""

from __future__ import annotations

import typing
from collections import Counter
from dataclasses import dataclass

import pytest

from nooa.agentdoc import hidden, pformat, pprint, spec, truncating_pformat
from nooa.agentdoc._visibility import _is_hidden_field, _TypeHints


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

    hints = _TypeHints()
    assert not _is_hidden_field(Forward, "value", _hints=hints)
    monkeypatch.setitem(globals(), "FutureHint", typing.Annotated[int, hidden])
    assert _is_hidden_field(Forward, "value", _hints=hints)


@pytest.mark.parametrize("fail", [False, True])
def test_copied_context_does_not_retain_render_annotations(fail):
    from contextvars import copy_context

    from nooa.agentdoc._pformat import _pformat

    @dataclass
    class Mutable:
        value: int = 1
        other: int = 2

    saved = []

    class Capture:
        def __repr__(self):
            saved.append(copy_context())
            return "Capture()"

    class Sink:
        def write(self, value):
            if fail and "Capture()" in value:
                raise RuntimeError("write failed")

    if fail:
        with pytest.raises(RuntimeError, match="write failed"):
            _pformat([Mutable(), Capture()], Sink())
    else:
        pformat([Mutable(), Capture()])
    Mutable.__annotations__["value"] = typing.Annotated[int, hidden]
    assert saved
    assert "value=1" not in saved[0].run(pformat, Mutable())
    assert "other=2" in saved[0].run(pformat, Mutable())


def test_reentrant_public_render_has_its_own_cache(monkeypatch):
    calls = Counter()
    original = typing.get_type_hints

    def counted(obj, *args, **kwargs):
        calls[id(obj)] += 1
        return original(obj, *args, **kwargs)

    class Nested:
        def __repr__(self):
            return pformat(Record(2))

    monkeypatch.setattr(typing, "get_type_hints", counted)
    result = pformat([Record(1), Nested(), Record(3)])
    assert "PRIVATE" not in result
    assert calls[id(Record)] == 2


def test_value_summary_shares_annotations_across_repeated_objects(monkeypatch):
    from nooa.agentdoc.doc_config import DocConfig
    from nooa.agentdoc.format import format_value_summary

    calls = Counter()
    original = typing.get_type_hints

    def counted(obj, *args, **kwargs):
        calls[id(obj)] += 1
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(typing, "get_type_hints", counted)
    result = format_value_summary([Record(1), Record(2)], DocConfig())
    assert "PRIVATE" not in result
    assert calls[id(Record)] == 1
