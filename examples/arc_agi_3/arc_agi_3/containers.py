# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI-3 environment data containers.

The step-info / step-record / reward-config dataclasses used by the environment
wrapper. (Trimmed to what the environment needs — the reference solver's state
containers are not vendored.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agi_3.grid import DiffGrid, Grid


# ---------------------------------------------------------------------------
# Step info (returned by env.step())
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    """Info dict returned by ARCAGI3Env.step() and ReplayEnvironment.step().

    Supports attribute access plus dict-style ``get`` and ``[]`` access.
    """

    state: str = ""
    step_id: int | None = None
    global_step: int = 0
    episode_step: int = 0
    levels_completed: int = 0
    win_levels: int = 0
    full_reset: bool = False
    actions_this_level: int = 0
    current_level: int = 0
    action_space: list = field(default_factory=list)
    available_actions: list[int] = field(default_factory=list)
    sdk_total_actions: int = 0
    sdk_resets: int = 0
    sdk_level_actions: list[int] = field(default_factory=list)
    diff_grid: DiffGrid | None = None
    diff_pixels: int = 0
    event: list[Grid] = field(default_factory=list)  # Unique intermediate frames.
    event_frame_count: int = 0  # Number of frames in event sequence.
    event_source_frame_count: int = 0  # Unique intermediate frames before decimation.
    event_frame_indices: list[int] = field(default_factory=list)  # 1-based source indices.
    event_decimation: str = ""  # Diagnostic sampling algorithm label.
    event_reduced_frame_position: int | None = None  # 0-based position in sampled event.
    event_reduced_frame_source_index: int | None = None  # 1-based source index.
    unique_colors_delta: dict = field(default_factory=dict)
    reward_components: dict = field(default_factory=dict)
    final_state: Grid | None = None
    final_event: list[Grid] | None = None
    # Replay-specific fields
    grid_unique: list[int] = field(default_factory=list)
    replay_action: Any = None
    replay_data: Any = None
    replay_exhausted: bool = False

    # Dict-style access to the same fields.
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key):
        return hasattr(self, key)


# ---------------------------------------------------------------------------
# Environment step record
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One entry in the environment step history.

    The environment builds these records from episode history and gameplay logs.
    """

    step: int
    grid: Grid
    action: str | None = None
    levels_completed: int = 0
    reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to plain dict for REPL injection (no dataclass dependency)."""
        return {
            "step": self.step,
            "grid": self.grid.data,
            "action": self.action,
            "previous_action": None,
            "action_chosen": self.action,
            "level": self.levels_completed,
            "level_completed": False,
        }


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Reward shaping configuration."""

    level_factor: float = 100.0
    gameover_penalty: float = 1.0
