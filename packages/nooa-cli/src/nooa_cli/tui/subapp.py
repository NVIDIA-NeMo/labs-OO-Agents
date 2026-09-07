# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable in-app subview primitives for the terminal TUI.

A subview is not a nested prompt_toolkit Application. It is a lightweight
state/render/key object hosted by the single long-lived TUIApplication so
terminal ownership, resize handling, mouse input, and focus remain centralized.

Interactive input prompts no longer live here: they use the container-based
overlays in ``prompt_overlay``, which host a real prompt_toolkit ``Buffer`` so
the cursor and editing behave exactly like the /resume and /todos browsers.
This module keeps the read-only ANSI subview protocol used by overlay views
such as /activity.

Subview conventions:

- ``q`` closes the subview.
- ``Esc`` is contextual inside the subview (clear/cancel/back), not the generic
  close key.
- The host owns resize handling; subviews render to the supplied width/height and
  must not launch their own prompt_toolkit Application or terminal-size poller.
"""

from __future__ import annotations

from typing import Literal, Protocol

SubviewKeyResult = Literal["handled", "close", "ignored"]


class InAppSubview(Protocol):
    """A modal/browseable view hosted inside the main TUIApplication."""

    title: str

    def render(self, width: int, height: int) -> str:
        """Render this view as an ANSI/plain terminal frame."""

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        """Handle a semantic key action from the host application."""

    def on_open(self) -> None:
        """Called after the view becomes active."""

    def on_close(self) -> None:
        """Called just before the view is removed."""


def normalize_key_result(result: SubviewKeyResult) -> SubviewKeyResult:
    """Normalize a subview key result to its canonical literal.

    Every host-routed view returns an explicit :data:`SubviewKeyResult`
    literal; the legacy boolean forms are no longer accepted.
    """
    if result == "close":
        return "close"
    if result == "ignored":
        return "ignored"
    return "handled"
