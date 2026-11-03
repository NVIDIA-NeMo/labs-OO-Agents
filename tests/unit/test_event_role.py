# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public ``EventBase.event_role`` property."""

import warnings
from typing import ClassVar

import pytest

from nooa.context_blocks.events import AssistantEvent, EventBase, UserEvent
from nooa.context_blocks.models import Role
from nooa.events import BeforeTurn, Notification


@pytest.mark.parametrize(
    ("event", "role"),
    [
        (UserEvent(content="hi"), Role.USER),
        (AssistantEvent(content="hi"), Role.ASSISTANT),
        (BeforeTurn.model_construct(), Role.RUNTIME_EVENT),
        (Notification(source="s", description="d"), Role.USER),
    ],
)
def test_event_role_returns_the_class_role(event, role):
    assert event.event_role is role
    assert event.event_role is type(event)._role


def test_event_role_follows_a_subclass_override_and_is_read_only():
    class _Custom(EventBase):
        _role: ClassVar[Role] = Role.METADATA

    event = _Custom()
    assert event.event_role is Role.METADATA
    with pytest.raises((AttributeError, ValueError)):
        event.event_role = Role.USER  # type: ignore[misc]


def test_a_subclass_field_named_role_is_not_shadowed():
    # The accessor is not called ``role``: a base-class property of that name
    # would silently win over a subclass field ``role``.
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class _WithRoleField(EventBase):
            role: str = "reviewer"

    assert _WithRoleField(role="author").role == "author"
    assert _WithRoleField().event_role is Role.USER
