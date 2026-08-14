# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable metadata and transient lifecycle events for coding-agent sessions."""

from typing import Annotated, ClassVar

from pydantic import Field

from nooa.context_blocks import EventBase, Metadata
from nooa.context_blocks.roles import Role


class SessionStarted(Metadata):
    """Identity and environment recorded once when a session is created."""

    _role: ClassVar[Role] = Role.METADATA

    origin: str = ""
    model: str = ""
    agent: str = ""
    working_directory: str = ""


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class SessionUserMessage(Metadata):
    """Raw user text accepted by the agent runtime as a conversation turn."""

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""


class SessionResumed(EventBase):  # type: ignore[misc]
    """An interactive agent has been restored or initialized for a session."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    handler_aliases: ClassVar[tuple[str, ...]] = ("TuiSessionResumed",)

    session_id: Annotated[str, Field(description="The resumed or started session ID")]
    restored: Annotated[
        bool,
        Field(description="Whether a snapshot was restored into the agent"),
    ]


class SessionCleared(EventBase):  # type: ignore[misc]
    """An interactive agent's working state has been reset."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    handler_aliases: ClassVar[tuple[str, ...]] = ("TuiSessionCleared",)

    session_id: Annotated[
        str | None,
        Field(default=None, description="The new post-clear session ID, when known"),
    ]


# Transient lifecycle events are deliberately absent here: this tuple is the
# set persisted with the session, and those two are runtime-only.
SESSION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
)
