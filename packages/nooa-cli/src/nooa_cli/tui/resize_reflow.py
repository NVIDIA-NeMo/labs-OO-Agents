# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State machine for source-backed transcript replay after terminal resize."""

from __future__ import annotations

from dataclasses import dataclass

TRANSCRIPT_REFLOW_DEBOUNCE_SECONDS = 0.075


@dataclass(frozen=True)
class ResizeObservation:
    """Result of sampling the terminal geometry."""

    changed: bool
    should_debounce: bool


@dataclass(frozen=True)
class ResizeReplayRequest:
    """A replay that is valid only for one settled resize generation."""

    width: int
    generation: int
    required: bool = False


class TranscriptResizeState:
    """Track observed geometry separately from the width actually replayed.

    Transcript wrapping depends on columns, not rows. Row-only changes are still
    observed so they extend the quiet period for an already-pending width replay,
    but they never initiate a destructive scrollback rebuild on their own.
    """

    def __init__(self) -> None:
        self.observed_size: tuple[int, int] | None = None
        self.replayed_width: int | None = None
        self.pending_width: int | None = None
        self.replay_required = False
        self.generation = 0

    def observe(self, size: tuple[int, int]) -> ResizeObservation:
        """Record a geometry sample and report whether to restart the debounce."""
        width, rows = (int(size[0]), int(size[1]))
        current = (width, rows)
        previous = self.observed_size
        if previous is None:
            self.observed_size = current
            self.replayed_width = width
            return ResizeObservation(changed=True, should_debounce=False)
        if current == previous:
            return ResizeObservation(changed=False, should_debounce=False)

        self.observed_size = current
        self.generation += 1
        if width != previous[0]:
            self.pending_width = width

        return ResizeObservation(
            changed=True,
            should_debounce=self.pending_width is not None,
        )

    def prepare_replay(self) -> ResizeReplayRequest | None:
        """Build a request once quiet without consuming work before success."""
        width = self.pending_width
        if width is None:
            return None
        if width == self.replayed_width and not self.replay_required:
            self.pending_width = None
            return None
        return ResizeReplayRequest(
            width=width,
            generation=self.generation,
            required=self.replay_required,
        )

    def request_replay(self) -> bool:
        """Require a rebuild at the observed width even when it did not change.

        This is reserved for recovery after prompt_toolkit had to compress its
        non-full-screen live region below the previously rendered height.  It
        is intentionally distinct from ordinary row observations, which stay
        cheap and never initiate transcript replay.
        """
        if self.observed_size is None:
            return False
        width = self.observed_size[0]
        if self.pending_width == width and self.replay_required:
            return False
        self.generation += 1
        self.pending_width = width
        self.replay_required = True
        return True

    def is_current(self, request: ResizeReplayRequest) -> bool:
        """Return whether a queued replay still matches the latest geometry."""
        return (
            request.generation == self.generation
            and self.observed_size is not None
            and request.width == self.observed_size[0]
            and request.width == self.pending_width
            and request.required == self.replay_required
        )

    def mark_replayed(self, request: ResizeReplayRequest) -> None:
        """Remember the width whose transcript was actually rebuilt."""
        self.replayed_width = request.width
        if (
            self.observed_size is not None
            and self.observed_size[0] == request.width
            and self.pending_width == request.width
        ):
            self.pending_width = None
            self.replay_required = False

    @property
    def has_pending_replay(self) -> bool:
        return self.pending_width is not None
