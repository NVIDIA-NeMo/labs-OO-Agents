# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision state encoding preserves selected content or rejects lossy layouts."""

import pytest

from nooa.context_blocks import Role
from nooa.context_blocks.exceptions import UnsupportedContextLayout
from nooa.context_view import Block, CacheBoundary
from nooa.decisions.state import decision_context_from_items
from nooa.events import DebugTrace, Message


def test_decision_items_preserve_values_order_and_selected_events():
    value = {"balance": 120, "eligible": True, "tags": ["new", None]}
    context, events = decision_context_from_items(
        [
            Block(key="account", content=value),
            CacheBoundary(),
            Message(content="first"),
            Message(content="second"),
        ]
    )
    assert context == {"account": value}
    assert events is not None
    assert [event["data"]["content"] for event in events] == ["first", "second"]


def test_decision_encoding_does_not_filter_events_selected_by_view():
    _, events = decision_context_from_items([DebugTrace(content="explicitly selected")])
    assert events is not None
    assert events[0]["data"]["content"] == "explicitly selected"


def test_empty_decision_items_do_not_add_history():
    assert decision_context_from_items([]) == ({}, None)


@pytest.mark.parametrize(
    "items,match",
    [
        ([Block(key="x", content=1), Block(key="x", content=2)], "Duplicate"),
        ([Block(key="x", content=1, role=Role.USER)], "requires role SYSTEM"),
        ([Message(content="earlier"), Block(key="x", content=1)], "follows an event"),
    ],
)
def test_decision_encoding_rejects_lossy_layouts(items, match):
    with pytest.raises(UnsupportedContextLayout, match=match):
        decision_context_from_items(items)
