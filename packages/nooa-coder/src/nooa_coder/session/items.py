# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data that crosses the Session boundary.

Everything here is a pydantic model with data fields only: no callables,
no live objects. Sessions, hosts and parent agents exchange these values;
live agents never cross.
"""

from typing import Literal

from pydantic import BaseModel, Field

SessionStatus = Literal["running", "idle", "retained", "closed", "on_disk"]


class Usage(BaseModel):
    """Token and cost totals: the session's own and those attributed from its children."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attributed_input_tokens: int = 0
    attributed_output_tokens: int = 0
    attributed_cost_usd: float = 0.0


class SessionInfo(BaseModel):
    """Metadata for one session, live or on disk.

    ``status`` is ``on_disk`` for a session read from the store; the
    registry reports ``running``, ``idle`` or ``retained`` for live ones.
    """

    id: str
    parent_id: str | None = None
    depth: int = 0
    name: str | None = None
    title: str | None = None
    title_is_user_set: bool = False
    workspace: str = ""
    host: str = ""
    agent: str = ""
    model: str = ""
    mode: str = "auto"
    status: SessionStatus = "on_disk"
    retained: bool = False
    created_at: float = 0.0
    last_active: float = 0.0
    turn_count: int = 0
    usage: Usage = Field(default_factory=Usage)
