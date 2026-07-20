# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cycle detection in ``_format_sequence`` and ``_format_dict``.

Python's built-in ``Py_ReprEnter`` short-circuits ``list.__repr__`` and
``dict.__repr__`` when it sees a container being re-entered — that's how
``x = []; x.append(x); repr(x)`` returns ``[[...]]`` instead of recursing
forever. The framework's recursive renderer is structurally similar but
doesn't get that machinery for free, so without an explicit guard
``pformat(x)`` would raise ``RecursionError``.

These tests pin the guard: any container whose ``id()`` is already on the
recursion stack emits ``<cycle>`` and returns without recursing.
"""

from __future__ import annotations

from nooa.agentdoc import pformat


class TestSelfReferentialContainers:
    def test_self_referential_list(self):
        """``x = []; x.append(x)`` — used to RecursionError, now emits ``<cycle>``."""
        x: list = []
        x.append(x)
        result = pformat(x)
        assert result == "[<cycle>]"

    def test_self_referential_dict(self):
        """A dict that holds itself as a value renders the inner ref as ``<cycle>``."""
        d: dict = {}
        d["self"] = d
        result = pformat(d)
        assert result == "{'self': <cycle>}"

    def test_self_referential_tuple_via_inner_list(self):
        """Tuple containing a list that points back to the tuple."""
        t: tuple = ([],)
        t[0].append(t)
        result = pformat(t)
        assert result == "([<cycle>])"

    def test_set_with_unhashable_cycle_skipped(self):
        """Sets can't contain themselves (unhashable), but a frozenset of frozensets
        could in theory; this test just confirms the simple case doesn't crash."""
        # Sets can hold other immutables but not themselves; check empty/full sets render fine.
        assert pformat({1, 2, 3}) == "{1, 2, 3}"


class TestMutualCycles:
    def test_two_lists_pointing_at_each_other(self):
        """``a -> b -> a`` two-step cycle."""
        a: list = []
        b: list = []
        a.append(b)
        b.append(a)
        result = pformat(a)
        # a holds b, b holds a — when we recurse into b we re-enter a, hence <cycle>.
        assert result == "[[<cycle>]]"

    def test_dict_with_list_holding_back_pointer(self):
        """Realistic shape: parent dict, child list, child references parent."""
        d: dict = {"name": "outer", "kids": []}
        d["kids"].append(d)
        result = pformat(d)
        assert "'name': 'outer'" in result
        assert "<cycle>" in result


class TestSharedReferencesAreNotCycles:
    """Same object referenced multiple times in different positions is NOT a cycle.
    Each occurrence renders fully — the cycle guard tracks the *currently-active*
    recursion stack, not every object ever seen."""

    def test_shared_inner_list(self):
        shared = [1, 2, 3]
        result = pformat([shared, shared, shared])
        # All three occurrences should render in full.
        assert result == "[[1, 2, 3], [1, 2, 3], [1, 2, 3]]"

    def test_shared_inner_dict(self):
        shared = {"a": 1}
        result = pformat({"x": shared, "y": shared})
        assert result == "{'x': {'a': 1}, 'y': {'a': 1}}"

    def test_shared_in_dict_value_then_list(self):
        shared = [1, 2, 3]
        result = pformat({"items": shared, "again": shared})
        assert "[1, 2, 3]" in result
        assert "<cycle>" not in result


class TestCyclesUnderTruncation:
    """Cycles still detected when the outer container is also truncated."""

    def test_cycle_inside_truncated_list(self):
        """Cycle in the head section of a list that triggers max_length truncation."""
        x: list = []
        x.append(x)
        # 100 copies of the same self-referential list — outer list also fits in head/tail
        result = pformat([x] * 100, max_length=10)
        assert "list(len=100" in result
        assert "<cycle>" in result

    def test_cycle_inside_truncated_dict(self):
        """Cycle reachable via a truncated dict's items."""
        x: list = []
        x.append(x)
        d = {f"k{i}": x for i in range(50)}
        result = pformat(d, max_length=4)
        assert "dict(len=50, items=" in result
        assert "<cycle>" in result


class TestCycleGuardCleanup:
    """The cycle guard uses ``id()`` and a try/finally to discard on exit.
    Re-rendering the same container twice must not spuriously emit ``<cycle>``
    on the second call (would happen if the ``_seen`` set leaked across calls).
    """

    def test_repeated_rendering_is_idempotent(self):
        x: list = [1, 2, 3]
        out1 = pformat(x)
        out2 = pformat(x)
        assert out1 == out2 == "[1, 2, 3]"
        assert "<cycle>" not in out1

    def test_cyclic_then_non_cyclic_no_leakage(self):
        """Render a cyclic structure first, then a fresh non-cyclic one;
        the second render must NOT inherit any leftover ``id()`` from the first."""
        cyc: list = []
        cyc.append(cyc)
        _ = pformat(cyc)
        # Now render a fresh structure that happens to reuse ids around (CPython
        # reuses freed ids quickly). It should render cleanly regardless.
        fresh = [[1], [2], [3]]
        out = pformat(fresh)
        assert out == "[[1], [2], [3]]"
        assert "<cycle>" not in out
