# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RHAE scoring helpers for ARC-AGI-3.

The ARC-AGI-3 design paper allows an individual level to exceed 100%
efficiency, capped at 115%, while the environment score is capped by the
weighted fraction of completed levels.
"""

from __future__ import annotations

from functools import lru_cache

LEVEL_SCORE_CAP = 1.15
PERFECT_LEVEL_SCORE = 1.0


def rhae_level_score(baseline: int | float, ai_actions: int | float) -> float:
    """Return the per-level RHAE score.

    Level score is squared human-relative action efficiency capped at 115%.
    Invalid or unsolved levels score 0.
    """
    if baseline <= 0 or ai_actions <= 0:
        return 0.0
    return min(LEVEL_SCORE_CAP, (float(baseline) / float(ai_actions)) ** 2)


def _weighted_fraction(n_levels: int, completed_levels: int) -> float:
    if n_levels <= 0:
        return 0.0
    completed = max(0, min(int(completed_levels), n_levels))
    return (completed * (completed + 1) / 2) / (n_levels * (n_levels + 1) / 2)


def _bounded_level_score(score: float) -> float:
    return min(LEVEL_SCORE_CAP, max(0.0, float(score)))


def rhae_score_from_level_scores(
    level_scores: list[float],
    n_levels: int,
    completed_levels: int,
) -> float:
    """Return the capped environment RHAE from per-level scores."""
    if n_levels <= 0:
        return 0.0

    completed = max(0, min(int(completed_levels), n_levels))
    total_weight = n_levels * (n_levels + 1) / 2
    weighted_score = 0.0
    for i in range(completed):
        score = level_scores[i] if i < len(level_scores) else 0.0
        weighted_score += (i + 1) * _bounded_level_score(score)

    env_score = weighted_score / total_weight
    env_cap = _weighted_fraction(n_levels, completed)
    return min(env_cap, env_score)


def rhae_environment_score(
    baseline_actions: list[int],
    ai_actions_per_level: list[int],
    levels_completed: int,
) -> tuple[float, list[float]]:
    """Return ``(environment_score, per_level_scores)`` for a run state."""
    n_levels = len(baseline_actions)
    completed = max(0, min(int(levels_completed), n_levels))
    level_scores: list[float] = []
    for i in range(n_levels):
        if i < completed and i < len(ai_actions_per_level):
            level_scores.append(rhae_level_score(baseline_actions[i], ai_actions_per_level[i]))
        else:
            level_scores.append(0.0)

    return (
        rhae_score_from_level_scores(level_scores, n_levels, completed),
        level_scores,
    )


def rhae_bounds(
    baseline_actions: list[int],
    completed_level_scores: list[float],
    current_level: int,
    actions_this_level: int,
    *,
    done: bool = False,
) -> tuple[float, float]:
    """Return live ``(RHAE_L, RHAE_U)`` bounds for the current run.

    ``RHAE_L`` is the score if the game ends now. ``RHAE_U`` is the score if
    the current level is solved by the next action and all later levels score
    exactly 100%, with the environment cap applied.
    """
    n_levels = len(baseline_actions)
    if n_levels <= 0:
        return 0.0, 0.0

    completed = max(0, min(int(current_level), n_levels))
    lower_scores = [0.0] * n_levels
    for i in range(min(completed, len(completed_level_scores))):
        lower_scores[i] = _bounded_level_score(completed_level_scores[i])

    lower = rhae_score_from_level_scores(lower_scores, n_levels, completed)
    if done or completed >= n_levels:
        return lower, lower

    optimistic_scores = list(lower_scores)
    optimistic_scores[completed] = rhae_level_score(
        baseline_actions[completed],
        max(0, int(actions_this_level)) + 1,
    )
    for i in range(completed + 1, n_levels):
        optimistic_scores[i] = PERFECT_LEVEL_SCORE

    upper = rhae_score_from_level_scores(optimistic_scores, n_levels, n_levels)
    return lower, max(lower, upper)


@lru_cache(maxsize=256)
def get_baseline_actions_for_game(
    game_id: str,
    operation_mode: str = "offline",
) -> list[int]:
    """Return SDK baseline actions for ``game_id`` or ``[]`` if unavailable."""
    if not game_id:
        return []
    try:
        from arc_agi_3.environment import get_env_info

        info = get_env_info(game_id, operation_mode)
        return list(getattr(info, "baseline_actions", []) or [])
    except Exception:
        return []
