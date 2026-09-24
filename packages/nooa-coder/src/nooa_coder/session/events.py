# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable metadata for interactive coding-agent sessions."""

from typing import ClassVar

from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role


class SessionStarted(Metadata):
    """Identity, environment and tree position recorded once when a session is created.

    Records written before the session tree carry ``origin`` and
    ``working_directory`` instead of ``host`` and ``workspace``; the store
    reads either (``Metadata`` keeps unknown fields).
    """

    _role: ClassVar[Role] = Role.METADATA

    host: str = ""
    model: str = ""
    agent: str = ""
    workspace: str = ""
    parent_id: str | None = None
    depth: int = 0
    name: str | None = None
    retained: bool = False


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class SessionUserMessage(Metadata):
    """Raw user text accepted by the agent runtime as a conversation turn."""

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""


SESSION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
)
