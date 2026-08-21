# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named local presentation services used by the TUI renderer.

These callbacks are owned by the local composition root. They intentionally do
not expose an arbitrary executor or concrete agent/runtime objects.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .subapp import InAppSubview


@dataclass(frozen=True, slots=True)
class TUIHostServices:
    """Optional local-only presentation integrations for ``TUIApplication``."""

    open_todo_view: Callable[[], Awaitable[InAppSubview] | InAppSubview] | None = None
    open_memory_view: Callable[[], Awaitable[InAppSubview] | InAppSubview] | None = None
    record_stray_output: Callable[[str, str], None] | None = None
    replay_identity: Callable[[], tuple[set[str], list[tuple[int, int]]]] | None = None
    auxiliary_status: Callable[[], str] | None = None
    before_output_drain: Callable[[], Awaitable[None]] | None = None
