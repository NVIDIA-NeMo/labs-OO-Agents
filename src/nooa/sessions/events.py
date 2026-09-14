# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral durable metadata and transient lifecycle events for sessions."""

from typing import ClassVar

from pydantic import AliasChoices, Field

from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role
from nooa.events import SessionCleared as SessionCleared
from nooa.events import SessionResumed as SessionResumed


class SessionStarted(Metadata):
    """Identity and environment recorded once when a session is created."""

    _role: ClassVar[Role] = Role.METADATA

    host: str = Field(default="", validation_alias=AliasChoices("host", "origin"))
    model: str = ""
    agent: str = ""
    working_directory: str = ""


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class SessionUserMessage(Metadata):
    """Raw user text accepted by a host as a conversation turn."""

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""


SESSION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
)
