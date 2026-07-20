# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-3 Solver: Environment wrapper around the official ARC-AGI SDK.

Provides a unified RL-style interface (step/reset) with automatic RHAE-aligned
reward extraction and step counting. Includes ReplayEnvironment for offline
testing with stored trajectory data.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("arc_agi_3.environment")
import copy
import json
import logging as _logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from arc_agi import Arcade, OperationMode  # Official ARC-AGI SDK

_ACTION_CHOSEN_KEY = "action_chosen"
_PREVIOUS_ACTION_KEY = "previous_action"
_EVENT_AFTER_CHOSEN_ACTION_KEY = "event_after_chosen_action"
_EVENT_METADATA_SUFFIXES = (
    "frame_count",
    "source_frame_count",
    "frame_indices",
    "decimation",
    "reduced_frame_position",
    "reduced_frame_source_index",
)


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


# ---------------------------------------------------------------------------
# ARC-SDK diagnostic shim: log HTTP response bodies on RESET/STEP failures.
# ---------------------------------------------------------------------------
#
# The SDK's `RemoteEnv.reset` / `RemoteEnv.step` catch
# `requests.exceptions.RequestException` and log only `str(e)` — which for a
# 400/403/500 resolves to "400 Client Error: Bad Request for url: ...".
# The server's JSON response body (with the actual reason — scorecard at
# capacity, game not in set, invalid key, etc.) is inside `e.response.text`
# but otherwise does not surface.
#
# This shim patches `RemoteEnv.reset` ONCE at module import to wrap the
# body-logging around the SDK's own logic. It does NOT change any semantics
# (returns whatever the SDK returns, re-raises nothing the SDK didn't
# already re-raise) — purely adds an extra ERROR-level log line with the
# response body BEFORE the SDK's own log line fires.
def _install_sdk_error_body_logger() -> None:
    try:
        from arc_agi import remote_wrapper  # noqa: F401  (needed for shim flag)
    except ImportError:
        return
    if getattr(remote_wrapper, "_arc3_error_body_shim_installed", False):
        return
    remote_wrapper._arc3_error_body_shim_installed = True

    # Use the beam `logger` (imported at module top) so messages land in
    # the beam-managed file handlers attached to subprocess debug.log via
    # `add_file_handlers`. A vanilla `logging.getLogger(__name__)` has no
    # handler in Beam's stack and would not reach subprocess debug logs.

    # Patch `requests.sessions.Session.send` at the class level. The SDK's
    # `RemoteEnvironmentWrapper.__init__` creates its own `_session` and
    # immediately calls `self.reset()`, so a per-instance response-hook
    # attached *after* `__init__` (which is the only point we control)
    # cannot see the init-RESET response. A class-level wrap sees every
    # request from every Session, including the init-RESET — filter by
    # URL so we only spam logs for arcprize.org traffic.
    import requests.sessions as _req_sessions
    _orig_send = _req_sessions.Session.send

    def _patched_send(self, request, **kwargs):
        resp = _orig_send(self, request, **kwargs)
        try:
            url = str(getattr(request, "url", ""))
            if (
                "arcprize.org" in url
                and resp is not None
                and resp.status_code >= 400
                # 403 on /api/scorecard/{scid} is the documented response
                # for competition-mode cards (api.py:93-99 — "cannot get
                # scorecard that is in competition mode"). The launcher's
                # keep-alive ping hits this every 10 minutes; the 403
                # actually confirms our cookies are pinning to the right
                # replica (a 404 would mean the wrong one). Skip logging
                # so this expected protocol behaviour doesn't masquerade
                # as an error.
                and not (
                    resp.status_code == 403
                    and "/api/scorecard/" in url
                )
            ):
                body = resp.text[:1500] if resp.text else "<empty>"
                logger.error(
                    f"[ARC-SDK-diag] {resp.status_code} {url}\n"
                    f"  response_body: {body}"
                )
        except Exception:
            pass
        return resp

    _req_sessions.Session.send = _patched_send


_install_sdk_error_body_logger()


class StepWallClockExceeded(Exception):
    """Exception type for solver-owned step wall-clock cancellation.

    Step wall-clock cancellation is owned by the solver loop, which sends
    a forced action through the normal action path.
    """


class SDKStepError(RuntimeError):
    """Raised when the SDK cannot accept/confirm an env.step action.

    This is an infrastructure failure after the solver has already chosen
    an action. It should terminate the run with ``termination_reason`` set
    to ``sdk-error``; it should not route through solver crash recovery or
    choose another fallback action.
    """


class SessionLost(SDKStepError):
    """SDK error indicating unrecoverable session loss."""


# Number of consecutive ``None`` responses from ``env.step`` that count
# as "the server isn't coming back". Backoff between attempts grows
# until we declare the session dead.
#
# The ARC server can transiently return 400 "game <id> not found" for a valid
# game and recover minutes later. Twelve attempts with delays capped at 60
# seconds provide roughly six minutes for the session to recover.
_STEP_NONE_THRESHOLD = 12
_STEP_RETRY_DELAYS_S = (0, 1, 2, 4, 8, 16, 32, 60, 60, 60, 60, 60)
from arcengine import GameAction, GameState

from arc_agi_3.containers import RewardConfig, StepInfo, StepRecord
from arc_agi_3.grid import ARC3_COLORS, DiffGrid, Grid

# GameAction is Enum (NOT IntEnum) — build a value->enum mapping
GA_MAP = {m.value: m for m in GameAction}


# ---------------------------------------------------------------------------
# RHAE scoring helpers (pure functions, no env dependency)
# ---------------------------------------------------------------------------

from arc_agi_3.rhae import (
    rhae_environment_score as _rhae_game_score,
    rhae_level_score as _rhae_level_score,
)

_UNSET = object()

EVENT_DECIMATION_EVEN_TEMPORAL = "even_temporal_unique_non_anchor"
EVENT_DECIMATION_DURATION_WEIGHTED = "duration_weighted_unique_non_anchor"
EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED = (
    "stable_dominance_duration_weighted"
)

_EVENT_DECIMATION_LEGACY_ALIASES = {
    "even",
    "even_temporal",
    EVENT_DECIMATION_EVEN_TEMPORAL,
}
_EVENT_DECIMATION_DURATION_ALIASES = {
    "duration_weighted",
    EVENT_DECIMATION_DURATION_WEIGHTED,
}
_EVENT_DECIMATION_STABLE_DOMINANCE_ALIASES = {
    "stable_dominance",
    "stable_dominance_duration",
    EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED,
}


def _iso_from_unix(unix_time_s: float) -> str:
    return datetime.fromtimestamp(float(unix_time_s), timezone.utc).isoformat()


def _wall_clock_stamp() -> tuple[str, float]:
    unix_time_s = time.time()
    return _iso_from_unix(unix_time_s), unix_time_s


def _elapsed_seconds(start_monotonic: float, end_monotonic: float) -> float:
    return round(max(0.0, end_monotonic - start_monotonic), 6)


# ---------------------------------------------------------------------------
# SDK metadata helpers (cached, lazy)
# ---------------------------------------------------------------------------

_sdk_env_infos: list | None = None
_available_actions_cache: dict[str, list[int]] = {}


def _get_sdk_env_infos() -> list:
    """Fetch all EnvironmentInfo objects from SDK offline mode. Cached."""
    global _sdk_env_infos
    if _sdk_env_infos is None:
        arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        _sdk_env_infos = list(arcade.get_environments())
    return _sdk_env_infos


def get_env_info(game_id: str, operation_mode: str = "offline"):
    """Get the EnvironmentInfo for a specific game_id from SDK."""
    resolved = ARCAGI3Env.resolve_game_id(game_id, operation_mode)
    for info in _get_sdk_env_infos():
        if info.game_id == resolved:
            return info
    raise ValueError(f"Unknown game_id: {game_id}")


def _inject_reset(actions: list[int]) -> list[int]:
    """Ensure RESET (0) is always in available actions.

    The SDK never reports RESET in frame.available_actions, but it is
    always accepted by the engine.
    """
    if 0 not in actions:
        return actions + [0]
    return actions


def get_available_actions(game_id: str, operation_mode: str = "offline") -> list[int]:
    """Get available actions for a game by querying SDK. Cached.

    Always probes via NORMAL mode (not COMPETITION) to avoid creating
    throwaway scorecards.  Falls back to OFFLINE for the 3 preview games.
    """
    game_id = ARCAGI3Env.resolve_game_id(game_id, operation_mode)
    if game_id not in _available_actions_cache:
        key = os.environ.get("ARC_API_KEY", "")
        # Try NORMAL first (works for all online games without scorecard),
        # fall back to OFFLINE for preview-only games.
        for mode in (OperationMode.NORMAL, OperationMode.OFFLINE):
            try:
                arcade = Arcade(operation_mode=mode, arc_api_key=key)
                env = arcade.make(game_id, seed=0)
                if env is not None:
                    frame = env.reset()
                    _available_actions_cache[game_id] = _inject_reset(
                        list(frame.available_actions)
                    )
                    break
            except Exception:
                continue
        else:
            raise RuntimeError(
                f"SDK returned None for {game_id} in all modes"
            )
    return _available_actions_cache[game_id]


def get_title_to_game_id() -> dict[str, str]:
    """Build title -> game_id mapping from SDK."""
    return {info.title: info.game_id for info in _get_sdk_env_infos()}


def _env_metadata_tags(env) -> list[str]:
    for obj in (
        env,
        getattr(env, "info", None),
        getattr(getattr(env, "env", None), "info", None),
    ):
        tags = getattr(obj, "tags", None)
        if isinstance(tags, (list, tuple)):
            return [str(tag) for tag in tags]

    game_id = (
        str(getattr(env, "game_id", "") or "")
        or str(getattr(getattr(env, "info", None), "game_id", "") or "")
        or str(
            getattr(
                getattr(getattr(env, "env", None), "info", None),
                "game_id",
                "",
            ) or ""
        )
    )
    if "-" not in game_id:
        return []
    prefix, suffix = game_id.split("-", 1)
    path = Path("environment_files") / prefix / suffix / "metadata.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    tags = data.get("tags", [])
    return [str(tag) for tag in tags] if isinstance(tags, list) else []


def _env_has_keyboard_controls(env) -> bool:
    tags = [tag.lower().replace("-", "_") for tag in _env_metadata_tags(env)]
    if any("keyboard" in tag for tag in tags):
        return True
    if any("click" in tag for tag in tags):
        return False

    actions = {
        str(action).upper()
        for action in (getattr(env, "action_names", []) or [])
    }
    return bool(actions & {"UP", "DOWN", "LEFT", "RIGHT", "1", "2", "3", "4"})


def _resolve_add_grid_orientation(value, env=None) -> bool:
    if isinstance(value, bool):
        return value
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"always", "true", "1", "yes", "on"}:
        return True
    if mode in {"never", "false", "0", "no", "off", ""}:
        return False
    if mode == "when_keyboard_active":
        return _env_has_keyboard_controls(env)
    raise ValueError(
        "add_grid_orientation must be never, always, or "
        f"when_keyboard_active; got {value!r}"
    )


def _unique_colors_delta(prev_grid: Grid | None, curr_grid: Grid) -> dict:
    """Compute which named colors appeared/disappeared between two grids."""
    prev_colors = set(prev_grid.unique_colors()) if prev_grid else set()
    curr_colors = set(curr_grid.unique_colors())
    def _names(indices):
        return [ARC3_COLORS[c] if c < len(ARC3_COLORS) else str(c)
                for c in sorted(indices)]
    return {
        "appeared": _names(curr_colors - prev_colors),
        "disappeared": _names(prev_colors - curr_colors),
    }


def _grid_to_agent_data(grid) -> Any:
    """Return a detached list-like grid payload for agent REPL variables."""
    if grid is None:
        return None
    data = getattr(grid, "data", grid)
    if hasattr(data, "tolist"):
        data = data.tolist()
    return copy.deepcopy(data)


def _event_to_agent_data(event) -> list:
    if not event:
        return []
    return [_grid_to_agent_data(frame) for frame in event]


_OUTER_GREEN_COLOR = 14


def _outer_green_mask(frame: np.ndarray, green: int = _OUTER_GREEN_COLOR) -> np.ndarray:
    arr = np.asarray(frame, dtype=int)
    if arr.ndim != 2:
        return np.zeros(arr.shape, dtype=bool)
    green_mask = arr == green
    mask = np.zeros(arr.shape, dtype=bool)
    if not green_mask.any():
        return mask

    h, w = arr.shape
    if h == 0 or w == 0:
        return mask
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if green_mask[r, c] and not mask[r, c]:
                mask[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if green_mask[r, c] and not mask[r, c]:
                mask[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                0 <= nr < h
                and 0 <= nc < w
                and green_mask[nr, nc]
                and not mask[nr, nc]
            ):
                mask[nr, nc] = True
                stack.append((nr, nc))
    return mask


def _anti_outer_green_hold(frames: list[np.ndarray]) -> list[np.ndarray]:
    filtered: list[np.ndarray] = []
    held: np.ndarray | None = None
    for frame in frames:
        arr = np.asarray(frame, dtype=int)
        mask = _outer_green_mask(arr)
        out = arr.copy()
        if held is not None and mask.any():
            out[mask] = held[mask]
        if held is None:
            held = out.copy()
        else:
            held[~mask] = arr[~mask]
        filtered.append(out)
    return filtered


def _collapse_adjacent_duplicate_arrays(frames: list[np.ndarray]) -> list[np.ndarray]:
    collapsed: list[np.ndarray] = []
    for frame in frames:
        if collapsed and np.array_equal(frame, collapsed[-1]):
            continue
        collapsed.append(frame)
    return collapsed


def _level_completion_display_frames(raw_frames) -> list[np.ndarray]:
    """Return completed-level frames for reflection, excluding next-level grid."""
    arrs = [np.asarray(raw, dtype=int) for raw in (raw_frames or [])]
    if len(arrs) < 2:
        return []
    filtered = _anti_outer_green_hold(arrs[:-1])
    return _collapse_adjacent_duplicate_arrays(filtered)


def _event_frame_indices_for_count(count: int) -> list[int]:
    return list(range(1, max(int(count or 0), 0) + 1))


def _normalize_event_frame_indices(indices, frame_count: int) -> list[int]:
    frame_count = max(int(frame_count or 0), 0)
    if frame_count <= 0:
        return []
    if not isinstance(indices, (list, tuple)):
        return _event_frame_indices_for_count(frame_count)
    normalized: list[int] = []
    for value in indices:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            return _event_frame_indices_for_count(frame_count)
    if len(normalized) != frame_count:
        return _event_frame_indices_for_count(frame_count)
    return normalized


def _normalize_event_reduced_frame_position(
    position,
    *,
    frame_count: int,
) -> int | None:
    frame_count = max(int(frame_count or 0), 0)
    if frame_count <= 0:
        return None
    try:
        value = int(position)
    except (TypeError, ValueError):
        return None
    if 0 <= value < frame_count:
        return value
    return None


def _event_metadata_from_row(row: dict) -> dict[str, Any]:
    event = row.get("event") or []
    frame_count = len(event)
    try:
        source_count = int(row.get("event_source_frame_count", 0) or 0)
    except (TypeError, ValueError):
        source_count = 0
    if frame_count and source_count <= 0:
        source_count = int(row.get("event_frame_count", frame_count) or frame_count)
    if not frame_count:
        source_count = 0
    indices = _normalize_event_frame_indices(
        row.get("event_frame_indices"),
        frame_count,
    )
    reduced_position = _normalize_event_reduced_frame_position(
        row.get("event_reduced_frame_position"),
        frame_count=frame_count,
    )
    if reduced_position is None and row.get("event_reduced_frame_source_index") is not None:
        try:
            source_index = int(row.get("event_reduced_frame_source_index"))
            if source_index in indices:
                reduced_position = indices.index(source_index)
        except (TypeError, ValueError):
            reduced_position = None
    reduced_source_index = None
    if reduced_position is not None:
        reduced_source_index = indices[reduced_position]
    return {
        "event_frame_count": frame_count,
        "event_source_frame_count": source_count,
        "event_frame_indices": indices,
        "event_decimation": row.get("event_decimation") or "",
        "event_reduced_frame_position": reduced_position,
        "event_reduced_frame_source_index": reduced_source_index,
    }


def _prefixed_event_fields(row: dict, prefix: str) -> dict:
    event_frames = copy.deepcopy(row.get("event") or [])
    metadata = _event_metadata_from_row(row)
    fields = {prefix: event_frames}
    for suffix in _EVENT_METADATA_SUFFIXES:
        fields[f"{prefix}_{suffix}"] = metadata[f"event_{suffix}"]
    return fields


def _agent_transition_entry(env, row: dict) -> dict:
    step = int(row["step"])
    if step < 0 or step >= len(env.state_history):
        raise RuntimeError(
            "Environment trajectory projection cannot find pre-action "
            f"state for step {step}: len(state_history)="
            f"{len(env.state_history)}"
        )
    event_meta = _event_metadata_from_row(row)
    return {
        "step": step,
        "grid": _grid_to_agent_data(env.state_history[step]),
        "action_chosen": row.get("action") or "",
        "level": row.get("level", row.get("levels_completed", 0)),
        "level_completed": bool(row.get("level_completed", False)),
        "diff_pixels": row.get("diff_pixels", 0),
        "actions_this_level": row.get("actions_this_level", 0),
        "unique_colors_delta": copy.deepcopy(
            row.get("unique_colors_delta") or {}
        ),
        "done": bool(row.get("done", False)),
        "state": row.get("state", ""),
        "event": copy.deepcopy(row.get("event") or []),
        **event_meta,
        "final_state": copy.deepcopy(row.get("final_state")),
        "final_event": (
            None if row.get("final_event") is None
            else copy.deepcopy(row.get("final_event") or [])
        ),
    }


def _agent_current_entry(env, visible_step: int) -> dict:
    if visible_step < 0 or visible_step >= len(env.state_history):
        raise RuntimeError(
            "Environment trajectory projection cannot find current state "
            f"for step {visible_step}: len(state_history)="
            f"{len(env.state_history)}"
        )
    last_transition = None
    if visible_step > 0 and visible_step - 1 < len(env.transition_history):
        last_transition = env.transition_history[visible_step - 1]
    diff_grid = getattr(env, "diff_grid", None)
    if visible_step == getattr(env, "global_step", visible_step):
        diff_pixels = (
            diff_grid.num_changed if diff_grid is not None else 0
        )
    else:
        diff_pixels = (
            last_transition.get("diff_pixels", 0)
            if last_transition is not None else 0
        )
    return {
        "step": visible_step,
        "grid": _grid_to_agent_data(env.state_history[visible_step]),
        "action_chosen": "",
        "level": getattr(env, "current_level", 0),
        "level_completed": False,
        "diff_pixels": diff_pixels,
        "actions_this_level": getattr(env, "actions_this_level", 0),
        "unique_colors_delta": copy.deepcopy(
            (last_transition or {}).get("unique_colors_delta") or {}
        ),
        "event": [],
        "event_frame_count": 0,
        "event_source_frame_count": 0,
        "event_frame_indices": [],
        "event_decimation": "",
        "event_reduced_frame_position": None,
        "event_reduced_frame_source_index": None,
        "final_state": None,
        "final_event": None,
        "done": False,
        "state": "",
        "episode_step": getattr(env, "episode_step", None),
    }


def _level_from_observation_row(env, row: dict | None, grid: Grid | None):
    value = getattr(grid, "level", None)
    if value is not None:
        return value
    if row is not None:
        for key in ("level", "levels_completed"):
            value = row.get(key)
            if value is not None:
                return value
    return getattr(env, "current_level", 0)


def _actions_this_level_from_observation_row(
    row: dict | None,
    level: int,
) -> int:
    if row is not None and row.get("level_completed", False):
        row_level = row.get("level", row.get("levels_completed"))
        if row_level is not None and row_level != level:
            return 0
    if row is not None and row.get("actions_this_level") is not None:
        return row.get("actions_this_level", 0)
    return 0


def _level_start_step_from_observation_rows(
    env,
    level: int,
    visible_step: int | None = None,
) -> int:
    visible_step = _validate_projection_visible_step(env, visible_step)
    start = None
    for step in range(visible_step, -1, -1):
        row_level = getattr(env.state_history[step], "level", None)
        if row_level is None:
            raise RuntimeError(
                "Environment observation level-start lookup found state "
                f"without level metadata at step {step}"
            )
        if row_level == level:
            start = step
        elif start is not None:
            break
    return 0 if start is None else start


def _agent_reset_observation_entry(env) -> dict:
    grid = env.state_history[0]
    return {
        "step": 0,
        "grid": _grid_to_agent_data(grid),
        "producer_action": "",
        "producer_transition_step": None,
        "level": _level_from_observation_row(env, None, grid),
        "level_completed": False,
        "diff_pixels": 0,
        "actions_this_level": 0,
        "unique_colors_delta": {},
        "done": False,
        "state": "",
        "event": [],
        "event_frame_count": 0,
        "event_source_frame_count": 0,
        "event_frame_indices": [],
        "event_decimation": "",
        "event_reduced_frame_position": None,
        "event_reduced_frame_source_index": None,
        "final_state": None,
        "final_event": None,
        "episode_step": getattr(grid, "step", 0),
    }


def _agent_observation_entry(env, step: int) -> dict:
    step = int(step)
    if step < 0 or step >= len(env.state_history):
        raise RuntimeError(
            "Environment observation projection cannot find state "
            f"for step {step}: len(state_history)={len(env.state_history)}"
        )
    if step == 0:
        return _agent_reset_observation_entry(env)

    row = env.transition_history[step - 1]
    action = row.get("action") or ""
    grid = env.state_history[step]
    entry = _agent_transition_entry(env, row)
    level = _level_from_observation_row(env, row, grid)
    entry.pop(_ACTION_CHOSEN_KEY, None)
    entry.update({
        "step": step,
        "grid": _grid_to_agent_data(grid),
        "producer_action": action,
        "producer_transition_step": step - 1,
        "level": level,
        "actions_this_level": _actions_this_level_from_observation_row(
            row, level
        ),
        "episode_step": row.get("episode_step", getattr(grid, "step", step)),
    })
    entry.pop(_PREVIOUS_ACTION_KEY, None)
    return entry


def _agent_action_transition_entry(env, row: dict) -> dict:
    step = int(row["step"])
    if step < 0 or step >= len(env.transition_history):
        raise RuntimeError(
            "Environment transition projection cannot find transition "
            f"for step {step}: len(transition_history)="
            f"{len(env.transition_history)}"
        )
    entry = _agent_observation_entry(env, step)
    action = row.get("action") or ""
    entry.update({
        "action_chosen": action,
        "action": action,
        "transition_step": step,
        "from_step": step,
        "to_step": step + 1,
        "reward": row.get("reward", 0.0),
        "done": bool(row.get("done", False)),
        "state": row.get("state", ""),
        "level": row.get(
            "level",
            row.get("levels_completed", entry.get("level", 0)),
        ),
        "level_completed": bool(row.get("level_completed", False)),
        "diff_pixels": row.get("diff_pixels", 0),
        "actions_this_level": row.get(
            "actions_this_level",
            entry.get("actions_this_level", 0),
        ),
        "unique_colors_delta": copy.deepcopy(
            row.get("unique_colors_delta") or {}
        ),
        "final_state": copy.deepcopy(row.get("final_state")),
        "final_event": (
            None if row.get("final_event") is None
            else copy.deepcopy(row.get("final_event") or [])
        ),
        "episode_step": row.get("episode_step", entry.get("episode_step")),
    })
    entry.update(_prefixed_event_fields(row, _EVENT_AFTER_CHOSEN_ACTION_KEY))
    return entry


def _validate_projection_visible_step(env, visible_step: int | None) -> int:
    validate = getattr(env, "validate_step_ledger", None)
    if callable(validate):
        validate()
    global_step = int(getattr(env, "global_step", 0))
    if visible_step is None:
        visible_step = global_step
    visible_step = int(visible_step)
    if visible_step < 0 or visible_step > global_step:
        raise RuntimeError(
            "Environment projection step is outside the env ledger: "
            f"visible_step={visible_step}, global_step={global_step}"
        )
    return visible_step


def _get_observation_from_ledgers(env, step: int) -> dict:
    visible_step = _validate_projection_visible_step(env, step)
    return _agent_observation_entry(env, visible_step)


def _get_observations_from_ledgers(
    env,
    visible_step: int | None = None,
) -> list[dict]:
    visible_step = _validate_projection_visible_step(env, visible_step)
    return [
        _agent_observation_entry(env, step)
        for step in range(visible_step + 1)
    ]


def _get_transitions_from_ledgers(
    env,
    visible_step: int | None = None,
) -> list[dict]:
    visible_step = _validate_projection_visible_step(env, visible_step)
    return [
        _agent_action_transition_entry(env, row)
        for row in env.transition_history[:visible_step]
    ]


def _get_agent_trajectory_from_ledgers(
    env,
    *,
    scope: str = "current",
    visible_step: int | None = None,
    level: int | None = None,
) -> list[dict]:
    """Project env ledgers into the agent-facing trajectory contract.

    The current trajectory is played transition-state rows plus one current
    observation tail. Completed-level reflection is transition-state rows only,
    including the final action that completed the level.
    """
    validate = getattr(env, "validate_step_ledger", None)
    if callable(validate):
        validate()

    global_step = int(getattr(env, "global_step", 0))
    if visible_step is None:
        visible_step = global_step if scope == "current" else global_step - 1
    visible_step = int(visible_step)

    if scope == "current":
        entries = [
            _agent_action_transition_entry(env, row)
            for row in env.transition_history[:visible_step]
        ]
        current_entry = _agent_observation_entry(env, visible_step)
        current_entry[_ACTION_CHOSEN_KEY] = ""
        entries.append(current_entry)
        steps = [entry["step"] for entry in entries]
        expected = list(range(visible_step + 1))
        if steps != expected:
            raise RuntimeError(
                "Current trajectory projection is not dense: "
                f"steps={steps[:10]}... expected={expected[:10]}..."
            )
        return entries

    if scope == "completed_level":
        if level is None:
            raise ValueError("completed_level trajectory requires level")
        max_transition_step = min(visible_step, global_step - 1)
        return [
            _agent_action_transition_entry(env, row)
            for row in env.transition_history
            if int(row.get("step", -1)) <= max_transition_step
            and row.get("level", row.get("levels_completed", 0)) == level
        ]

    raise ValueError(f"Unknown agent trajectory scope: {scope!r}")


def _decimate_frame_indices(frames: list[np.ndarray], k: int) -> list[int]:
    """Select k evenly spaced frame indices, preserving temporal order."""
    if k <= 0 or not frames:
        return []
    if len(frames) <= k:
        return list(range(len(frames)))
    if k == 1:
        return [0]
    return sorted({
        round(i * (len(frames) - 1) / (k - 1))
        for i in range(k)
    })


def _decimate_frames(frames: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Select k evenly spaced frames, preserving temporal order."""
    return [frames[n] for n in _decimate_frame_indices(frames, k)]


@dataclass
class _EventSourceFrame:
    source_index: int
    frame: np.ndarray
    first_duration: int
    total_duration: int
    max_duration: int
    raw_start: int
    raw_end: int
    duplicate_occurrences: int = 1


def _event_source_frames(
    middle: list[np.ndarray],
    *,
    first_anchor: np.ndarray | None,
    final_anchor: np.ndarray,
) -> list[_EventSourceFrame]:
    """Collapse event frames while preserving per-frame hold durations."""
    if not middle:
        return []

    adjacent: list[dict[str, Any]] = []
    for raw_index, frame in enumerate(middle):
        if adjacent and np.array_equal(frame, adjacent[-1]["frame"]):
            adjacent[-1]["duration"] += 1
            adjacent[-1]["raw_end"] = raw_index
        else:
            adjacent.append({
                "frame": frame,
                "duration": 1,
                "raw_start": raw_index,
                "raw_end": raw_index,
            })

    seen: set[bytes] = {final_anchor.tobytes()}
    if first_anchor is not None:
        seen.add(first_anchor.tobytes())
    unique_by_key: dict[bytes, _EventSourceFrame] = {}
    unique: list[_EventSourceFrame] = []

    for run in adjacent:
        frame = run["frame"]
        key = frame.tobytes()
        duration = int(run["duration"])
        if key in seen:
            existing = unique_by_key.get(key)
            if existing is not None:
                existing.total_duration += duration
                existing.max_duration = max(existing.max_duration, duration)
                existing.duplicate_occurrences += 1
            continue
        seen.add(key)
        source = _EventSourceFrame(
            source_index=len(unique),
            frame=frame,
            first_duration=duration,
            total_duration=duration,
            max_duration=duration,
            raw_start=int(run["raw_start"]),
            raw_end=int(run["raw_end"]),
        )
        unique_by_key[key] = source
        unique.append(source)

    return unique


def _duration_weighted_source_indices(
    sources: list[_EventSourceFrame],
    k: int,
) -> list[int]:
    """Select source indices by duration-weighted quantiles."""
    if k <= 0 or not sources:
        return []
    if len(sources) <= k:
        return [source.source_index for source in sources]

    weights = [max(int(source.total_duration), 1) for source in sources]
    total = sum(weights)
    selected: set[int] = set()
    cumulative = 0
    pos = 0
    for target_i in range(k):
        target = (target_i + 0.5) * total / k
        while pos < len(weights) - 1 and cumulative + weights[pos] < target:
            cumulative += weights[pos]
            pos += 1
        selected.add(sources[pos].source_index)

    if len(selected) < min(k, len(sources)):
        for source in sorted(
            sources,
            key=lambda item: (
                -item.total_duration,
                item.raw_start,
                item.source_index,
            ),
        ):
            selected.add(source.source_index)
            if len(selected) >= min(k, len(sources)):
                break

    return sorted(selected)


def _stable_dominance_decimation_candidates(
    sources: list[_EventSourceFrame],
    k: int,
    *,
    min_consecutive_frames: int,
    stable_dominance_threshold: float,
    min_stable_frames: int,
) -> tuple[bool, list[_EventSourceFrame], str, float, int]:
    effective_min_stable = (
        int(min_stable_frames)
        if int(min_stable_frames or 0) > 0
        else max(3, int(k) // 2)
    )
    stable = [
        source
        for source in sources
        if source.max_duration >= min_consecutive_frames
    ]
    total_duration = sum(source.total_duration for source in sources)
    stable_duration = sum(source.total_duration for source in stable)
    stable_share = stable_duration / total_duration if total_duration else 0.0

    if len(sources) <= k:
        return False, sources, "not_over_budget", stable_share, effective_min_stable
    if stable_share < stable_dominance_threshold:
        return (
            False,
            sources,
            "stable_duration_share_below_threshold",
            stable_share,
            effective_min_stable,
        )
    if len(stable) < effective_min_stable:
        return (
            False,
            sources,
            "stable_count_below_min",
            stable_share,
            effective_min_stable,
        )
    return True, stable, "filter_on", stable_share, effective_min_stable


def _select_event_source_indices(
    sources: list[_EventSourceFrame],
    k: int,
    *,
    decimation_algorithm: str,
    min_consecutive_frames: int,
    stable_dominance_threshold: float,
    min_stable_frames: int,
) -> tuple[list[int], str]:
    """Select source indices from unique event frames."""
    if k <= 0:
        return [], "display_disabled"
    if not sources:
        return [], "none"
    if len(sources) <= k:
        return [source.source_index for source in sources], "none"

    algorithm = (decimation_algorithm or "").strip()
    if algorithm in _EVENT_DECIMATION_LEGACY_ALIASES:
        frames = [source.frame for source in sources]
        return _decimate_frame_indices(frames, k), EVENT_DECIMATION_EVEN_TEMPORAL

    if algorithm in _EVENT_DECIMATION_DURATION_ALIASES:
        return (
            _duration_weighted_source_indices(sources, k),
            EVENT_DECIMATION_DURATION_WEIGHTED,
        )

    if algorithm in _EVENT_DECIMATION_STABLE_DOMINANCE_ALIASES:
        filter_on, candidates, reason, stable_share, effective_min_stable = (
            _stable_dominance_decimation_candidates(
                sources,
                k,
                min_consecutive_frames=min_consecutive_frames,
                stable_dominance_threshold=stable_dominance_threshold,
                min_stable_frames=min_stable_frames,
            )
        )
        selected = _duration_weighted_source_indices(candidates, k)
        if filter_on:
            label = (
                f"{EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED}"
                f":filter_on:min{min_consecutive_frames}"
                f":share{stable_share:.2f}"
            )
        else:
            label = (
                f"{EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED}"
                f":filter_off:{reason}"
                f":min_stable{effective_min_stable}"
                f":share{stable_share:.2f}"
            )
        return selected, label

    raise ValueError(
        "Unknown event_decimation_algorithm "
        f"{decimation_algorithm!r}. Expected one of: "
        f"{EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED!r}, "
        f"{EVENT_DECIMATION_DURATION_WEIGHTED!r}, "
        f"{EVENT_DECIMATION_EVEN_TEMPORAL!r}."
    )


def _best_reduced_event_position(
    frames: list[np.ndarray],
    *,
    previous_anchor: np.ndarray | None,
    next_anchor: np.ndarray | None,
) -> int | None:
    """Choose one sampled event frame for compact role display.

    Scores against the two stable action anchors available during extraction:
    pre-action grid and next settled grid (the SDK frame sequence tail).
    """
    if not frames:
        return None
    anchors = [anchor for anchor in (previous_anchor, next_anchor) if anchor is not None]
    if not anchors:
        return 0
    scores = [
        sum(int(np.count_nonzero(frame != anchor)) for anchor in anchors)
        for frame in frames
    ]
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def _empty_event_sample(
    decimation: str = "none",
    *,
    final_state: Grid | None = None,
) -> dict[str, Any]:
    return {
        "frames": [],
        "source_frame_count": 0,
        "frame_indices": [],
        "decimation": decimation,
        "reduced_frame_position": None,
        "reduced_frame_source_index": None,
        "final_state": final_state,
    }


def _extract_event_sample(
    frame: Any,
    *,
    level_up: bool,
    step: int = 0,
    level: int = 0,
    prev_grid_data,
    max_frames: int = 5,
    decimation_algorithm: str = EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED,
    min_consecutive_frames: int = 2,
    stable_dominance_threshold: float = 0.5,
    min_stable_frames: int = 0,
    report_original_event_size: bool = False,
) -> dict[str, Any]:
    """Extract sampled event frames and reported frame-count metadata.

    Pipeline:

      1. **Trim leading** — drop any frames at the head of the sequence
         that are identical to the pre-action grid (``prev_grid_data``).
      2. **Trim trailing** — drop any frames at the tail that are
         identical to the settled grid (``raw_frames[-1]``).
      3. **Collapse adjacent duplicates with duration accounting** — keep one
         frame per visual state, while tracking repeated-frame hold duration.
      4. **Remove duplicate/anchor frames** — drop frames identical to the
         pre-action grid, the settled grid, or an earlier kept event frame,
         while accumulating durations for repeated non-adjacent states.
      5. **Decimate** — if more than ``max_frames`` remain, select frames with
         the configured event-decimation algorithm. Temporal order is preserved.

    Returns a dict with Grid frames plus event-size metadata. When
    ``report_original_event_size`` is false, metadata describes the sampled
    event sequence itself. When true, metadata preserves 1-based indices in
    the filtered unique intermediate source sequence. On level-up, the source
    sequence is first converted into completed-level frames and the result is
    consumed as ``final_event`` rather than normal ``event``.
    """
    raw_frames = getattr(frame, 'frame', None) or []
    final_state = None
    if level_up:
        raw_frames = _level_completion_display_frames(raw_frames)
        if raw_frames:
            final_state = Grid(
                data=[
                    [int(v) for v in row]
                    for row in raw_frames[-1].tolist()
                ],
                step=step,
                level=level,
            )
    if len(raw_frames) < 2:
        return _empty_event_sample(final_state=final_state)

    final = np.asarray(raw_frames[-1], dtype=int)
    first = (
        np.asarray(prev_grid_data, dtype=int)
        if prev_grid_data is not None else None
    )

    arrs = [np.asarray(raw, dtype=int) for raw in raw_frames]

    # Step 1: trim leading frames equal to the pre-action anchor.
    lo = 0
    if first is not None:
        while lo < len(arrs) and np.array_equal(arrs[lo], first):
            lo += 1

    # Step 2: trim trailing frames equal to the settled anchor.
    hi = len(arrs)
    while hi > lo and np.array_equal(arrs[hi - 1], final):
        hi -= 1

    middle = arrs[lo:hi]
    if not middle:
        return _empty_event_sample(final_state=final_state)

    # Steps 3-4: collapse adjacent duplicates, remove anchors/duplicates, and
    # retain per-frame duration so decimation can prefer long-visible states.
    sources = _event_source_frames(
        middle,
        first_anchor=first,
        final_anchor=final,
    )

    if not sources:
        return _empty_event_sample(final_state=final_state)

    source_count = len(sources)
    try:
        max_frames = int(max_frames)
    except (TypeError, ValueError):
        max_frames = 5
    try:
        min_consecutive_frames = max(1, int(min_consecutive_frames))
    except (TypeError, ValueError):
        min_consecutive_frames = 2
    try:
        stable_dominance_threshold = float(stable_dominance_threshold)
    except (TypeError, ValueError):
        stable_dominance_threshold = 0.5
    try:
        min_stable_frames = max(0, int(min_stable_frames or 0))
    except (TypeError, ValueError):
        min_stable_frames = 0

    selected_indices, decimation = _select_event_source_indices(
        sources,
        max_frames,
        decimation_algorithm=decimation_algorithm,
        min_consecutive_frames=min_consecutive_frames,
        stable_dominance_threshold=stable_dominance_threshold,
        min_stable_frames=min_stable_frames,
    )

    by_index = {source.source_index: source.frame for source in sources}
    selected_frames = [
        by_index[idx]
        for idx in selected_indices
        if idx in by_index
    ]
    reduced_position = _best_reduced_event_position(
        selected_frames,
        previous_anchor=first,
        next_anchor=final,
    )
    reduced_source_index = (
        selected_indices[reduced_position] + 1
        if reduced_position is not None else None
    )
    reported_indices = [idx + 1 for idx in selected_indices]
    reported_source_count = source_count
    reported_reduced_source_index = reduced_source_index
    if not report_original_event_size:
        reported_source_count = len(selected_frames)
        reported_indices = _event_frame_indices_for_count(reported_source_count)
        reported_reduced_source_index = (
            reduced_position + 1
            if reduced_position is not None else None
        )

    return {
        "frames": [
            Grid(data=selected.tolist(), step=step, level=level)
            for selected in selected_frames
        ],
        "source_frame_count": reported_source_count,
        "frame_indices": reported_indices,
        "decimation": decimation,
        "reduced_frame_position": reduced_position,
        "reduced_frame_source_index": reported_reduced_source_index,
        "final_state": final_state,
    }


def _extract_event_frames(
    frame: Any,
    *,
    level_up: bool,
    step: int = 0,
    level: int = 0,
    prev_grid_data,
    max_frames: int = 5,
    decimation_algorithm: str = EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED,
    min_consecutive_frames: int = 2,
    stable_dominance_threshold: float = 0.5,
    min_stable_frames: int = 0,
) -> list[Grid]:
    """Return only the sampled event frames."""
    if level_up:
        return []
    return _extract_event_sample(
        frame,
        level_up=level_up,
        step=step,
        level=level,
        prev_grid_data=prev_grid_data,
        max_frames=max_frames,
        decimation_algorithm=decimation_algorithm,
        min_consecutive_frames=min_consecutive_frames,
        stable_dominance_threshold=stable_dominance_threshold,
        min_stable_frames=min_stable_frames,
    )["frames"]


class ARCAGI3Env:
    """Unified environment wrapper around the official arc_agi SDK.

    Wraps Arcade + EnvironmentWrapper to provide a standard RL-style
    interface (step/reset) with automatic reward extraction and step counting.
    """

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        operation_mode: str = "offline",
        reward_config: RewardConfig | None = None,
        api_key: str | None = None,
        scorecard_id: str | None = None,
        render_mode: str | None = None,
        results_dir: str | Path | None = None,
        max_event_frames: int = 5,
        event_decimation_algorithm: str = (
            EVENT_DECIMATION_STABLE_DOMINANCE_DURATION_WEIGHTED
        ),
        event_decimation_min_consecutive_frames: int = 2,
        event_decimation_stable_dominance_threshold: float = 0.5,
        event_decimation_min_stable_frames: int = 0,
        report_original_event_size: bool = False,
        max_step_wall_seconds: float | None = None,
        arcade: Arcade | None = None,
    ):
        self._raw_game_id = game_id
        self.seed = seed
        self.reward_config = reward_config or RewardConfig()
        self._max_event_frames = max_event_frames
        self._event_decimation_algorithm = event_decimation_algorithm
        self._event_decimation_min_consecutive_frames = (
            event_decimation_min_consecutive_frames
        )
        self._event_decimation_stable_dominance_threshold = (
            event_decimation_stable_dominance_threshold
        )
        self._event_decimation_min_stable_frames = (
            event_decimation_min_stable_frames
        )
        self._report_original_event_size = bool(report_original_event_size)

        # Step wall-clock budget. None disables the solver-side guard.
        # The env treats this as diagnostic telemetry only; it never raises
        # before attempting the SDK action.
        self._max_step_wall_seconds = max_step_wall_seconds
        # Attempt time is diagnostic; confirmed time is what the solver guard
        # uses to decide when to cancel reasoning and force an action.
        self._last_step_attempt_at: float = time.monotonic()
        self._last_step_confirmed_at: float = self._last_step_attempt_at

        # Step counters (sequencing only — authoritative action counts
        # come from the SDK scorecard via _sync_scorecard_stats)
        self.global_step = 0
        self.episode_step = 0
        self.episode = 0
        self._session_recoveries = 0
        # Session-loss diagnostics. Unrecoverable SDK failures raise
        # SDKStepError and terminate the run as sdk-error.
        self._session_lost = False
        self._session_lost_reason: str | None = None

        # Per-level tracking — populated from SDK scorecard
        self.actions_this_level = 0
        # Counter read only by the grinding-reset mechanism.
        # Zeroed on RESET action and on level_up; incremented otherwise.
        self.actions_since_reset = 0
        self.current_level = 0
        self.sdk_total_actions = 0
        self.sdk_resets = 0
        self.sdk_level_actions: list[int] = []
        self._local_level_actions: list[int] = []

        # Episode history — stores every grid seen as Grid objects
        self.episode_history: list[Grid] = []
        # Global action ledger. ``state_history[N]`` is the current state
        # before action N / after N executed actions; ``transition_history[N]``
        # is the action issued at global step N.
        self.state_history: list[Grid] = []
        self.transition_history: list[dict] = []

        # Previous frame for reward computation
        self._prev_frame = None
        self._prev_grid: Grid | None = None

        # Diff between consecutive grids (computed by env, not solver)
        self.diff_grid: DiffGrid | None = None

        # Full gameplay log — every step with all metadata
        self.gameplay_log: list[dict] = []
        self._results_dir: Path | None = Path(results_dir) if results_dir else None

        # SDK setup
        self._operation_mode = operation_mode
        key = api_key or os.environ.get("ARC_API_KEY", "")
        mode_map = {
            "online": OperationMode.ONLINE,
            "offline": OperationMode.OFFLINE,
            "normal": OperationMode.NORMAL,
            "competition": OperationMode.COMPETITION,
        }
        op_mode = mode_map.get(operation_mode, OperationMode.OFFLINE)

        self.arcade = arcade or Arcade(operation_mode=op_mode, arc_api_key=key)

        # Bridge SDK logger into beam logger so we see all SDK errors/warnings
        _sdk_logger = _logging.getLogger("arc_agi.base")
        _sdk_logger.handlers.clear()
        _sdk_logger.setLevel(_logging.DEBUG)

        class _SDKLogBridge(_logging.Handler):
            def emit(self, record):
                msg = f"[ARC-SDK] {record.getMessage()}"
                if record.levelno >= _logging.ERROR:
                    logger.error(msg)
                elif record.levelno >= _logging.WARNING:
                    logger.warning(msg)
                elif record.levelno >= _logging.INFO:
                    logger.info(msg)

        _sdk_logger.addHandler(_SDKLogBridge())

        # ARC Prize's backend sits behind an AWS Application Load Balancer
        # that pins clients to a target replica via the AWSALBAPP-* cookies
        # it sets on the response. The scorecard a replica creates is only
        # visible to that replica until replication catches up (observed
        # delay: tens of seconds to indefinite). If a subprocess's Arcade
        # session lands on a different replica than the launcher's — and
        # by default it does, because each subprocess starts with an empty
        # cookie jar — the subprocess's RESET hits a replica that has
        # never seen the scorecard, and the server returns 400 "game X
        # not found" (api.py:391-395's shared fallthrough message for
        # `scorecard is None`). Fix: propagate the launcher's session
        # cookies via ARC_SDK_SESSION_COOKIES (set by the launcher right
        # after create_scorecard). Injecting them here, before arcade.make
        # fires the init-RESET, routes every SDK request for this
        # subprocess to the same replica that holds the card.
        _cookies_blob = os.environ.get("ARC_SDK_SESSION_COOKIES", "").strip()
        if _cookies_blob and hasattr(self.arcade, "_session"):
            import json as _json
            _cookies = _json.loads(_cookies_blob)
            self.arcade._session.cookies.update(_cookies)
            logger.info(
                f"[session-cookies] loaded {len(_cookies)} cookies from "
                f"ARC_SDK_SESSION_COOKIES ({sorted(_cookies.keys())})"
            )

        self._scorecard_id = scorecard_id
        self._render_mode = render_mode

        make_kwargs: dict = {"seed": seed}
        if scorecard_id:
            make_kwargs["scorecard_id"] = scorecard_id
        if render_mode:
            make_kwargs["render_mode"] = render_mode
        self.env = self.arcade.make(game_id, **make_kwargs)

        if self.env is None:
            raise RuntimeError(f"SDK returned None environment for {game_id}")

        # Use SDK-resolved game_id (e.g., "vc33-9851e02b") for scorecard matching
        self.game_id = getattr(self.env.info, 'game_id', game_id)

        # Increase SDK HTTP timeout from hardcoded 10s to 30s
        if hasattr(self.env, '_session'):
            _original_send = self.env._session.send
            def _patched_send(request, **kwargs):
                kwargs['timeout'] = 30
                return _original_send(request, **kwargs)
            self.env._session.send = _patched_send

        # Metadata from SDK
        self.baseline_actions = list(self.env.info.baseline_actions)
        self._available_actions_ints: list[int] = []

        # ------------------------------------------------------------------
        # Snapshot the init-reset frame for one-shot consumption.
        # ------------------------------------------------------------------
        # The SDK wrapper constructor unconditionally calls `self.reset()`.
        # On success, the returned
        # `FrameDataRaw` is cached at `self.env._last_response`. That one
        # RESET POST is the ONLY RESET we need for game start — all our
        # bookkeeping in `ARCAGI3Env.reset()` is pure Python.
        #
        # Firing a second `env.reset()` in competition mode can trigger the
        # server's existing-environment rejection. The constructor's cached
        # frame is therefore the only RESET response consumed at game start.
        #
        # Capture the init-reset frame here; `_reset_with_retry` consumes
        # it on its FIRST call (game start) and falls through to a real
        # `env.reset()` for subsequent calls (session recovery, episode
        # restart). Session recovery's reset() sends a guid, which identifies
        # the existing session and allows level resets in competition mode.
        self._pending_initial_frame = getattr(self.env, "_last_response", None)

    def _step_with_retry(self, action, data=None, reasoning=None):
        """Call SDK env.step() with retry on None (server transient failure).

        On persistent None (server session wipe), attempts env.reset() to
        recover the session. If recovery succeeds, returns the reset frame
        with state='SESSION_RECOVERED' so the caller can route through the
        harness reset flow.

        Step wall-clock telemetry is warning-only here. The solver loop owns
        cancellation and forced-action selection; the env still attempts the
        SDK action once called.
        """
        now = time.monotonic()
        elapsed_since_last_confirmed = now - self._last_step_confirmed_at
        self._last_step_attempt_at = now
        if (self._max_step_wall_seconds is not None
                and elapsed_since_last_confirmed
                > self._max_step_wall_seconds):
            logger.warning(
                "env.step called %.1fs after the previous SDK-confirmed "
                "step (max_step_wall_seconds=%s). Attempting SDK action "
                "anyway; solver-side cancellation should normally prevent "
                "this late call.",
                elapsed_since_last_confirmed,
                self._max_step_wall_seconds,
            )

        # Try the step up to _STEP_NONE_THRESHOLD times with backoff. A
        # transient network hiccup or replica blip should clear within
        # the ~15 s budget; persistent failure means the server-side
        # session is dead and recovery must take over.
        consecutive_nones = 0
        for delay in _STEP_RETRY_DELAYS_S:
            if delay > 0:
                time.sleep(delay)
            try:
                frame = self.env.step(action, data=data, reasoning=reasoning)
            except Exception as exc:
                raise SDKStepError(
                    f"SDK env.step raised {type(exc).__name__}: {exc} "
                    f"(action={action}, data={data})"
                ) from exc
            if frame is not None:
                self._last_step_confirmed_at = time.monotonic()
                if consecutive_nones > 0:
                    logger.info(
                        f"env.step() recovered after {consecutive_nones} "
                        f"None response(s)"
                    )
                return frame
            consecutive_nones += 1
            next_idx = min(consecutive_nones, len(_STEP_RETRY_DELAYS_S) - 1)
            if consecutive_nones < _STEP_NONE_THRESHOLD:
                logger.warning(
                    f"env.step() returned None "
                    f"({consecutive_nones}/{_STEP_NONE_THRESHOLD}, "
                    f"action={action}, data={data}); retrying in "
                    f"{_STEP_RETRY_DELAYS_S[next_idx]}s..."
                )

        # _STEP_NONE_THRESHOLD consecutive Nones from env.step. Try one
        # session recovery via env.reset() (uses the wrapper's stored
        # guid, which competition mode treats as a level-reset rather
        # than a fresh registration).
        logger.error(
            f"env.step() returned None on {consecutive_nones} consecutive "
            f"attempts (action={action}, data={data}). Attempting session "
            f"recovery via env.reset()..."
        )
        self._session_recoveries += 1
        try:
            frame = self.env.reset()
        except Exception as e:
            logger.error(f"Session recovery env.reset() failed: {e}")
            frame = None
        if frame is not None:
            self._last_step_confirmed_at = time.monotonic()
            logger.warning(
                f"Session recovered via env.reset() "
                f"(recovery #{self._session_recoveries}). "
                f"Resuming from level {frame.levels_completed}."
            )
            # Tag the frame so the caller knows this was a forced reset
            frame._session_recovered = True
            return frame

        # Recovery also failed. The server-side session is gone for good.
        raise SDKStepError(
            f"env.step() returned None on {consecutive_nones} consecutive "
            f"attempts and env.reset() recovery also failed "
            f"(action={action}, data={data})"
        )

    def seconds_since_last_sdk_step(self) -> float:
        """Seconds since the last SDK-confirmed reset/step result."""
        return time.monotonic() - self._last_step_confirmed_at

    def _reset_with_retry(self):
        """Return the initial/post-reset frame.

        The SDK's `arcade.make()` already fired a RESET during its
        `__init__` and cached the result at `self.env._last_response`,
        snapshotted into `self._pending_initial_frame` at construction
        time. The FIRST call to this method consumes that snapshot —
        no extra HTTP. All subsequent calls (session recovery, episode
        restart) fall through to a real `env.reset()`; those carry the
        guid from the init-reset so the server treats them as level
        resets (allowed in competition mode) rather than env
        registrations (blocked by api.py:417).

        Retry on None once, since the request may be a transient
        server flake on the real-reset path.
        """
        if self._pending_initial_frame is not None:
            frame = self._pending_initial_frame
            self._pending_initial_frame = None
            return frame

        frame = self.env.reset()
        if frame is not None:
            return frame
        logger.warning("env.reset() returned None, retrying in 1s...")
        time.sleep(1)
        frame = self.env.reset()
        if frame is not None:
            logger.info("env.reset() retry succeeded")
            return frame
        raise RuntimeError("env.reset() returned None after retry")

    def _make_session_lost_step_return(
        self,
        step_id: int | None = None,
    ) -> tuple[Grid, float, bool, StepInfo]:
        """Construct a structured result for an SDK session-loss state."""
        grid = self._prev_grid or self._extract_grid(
            self._prev_frame, step=self.episode_step, level=self.current_level
        ) if self._prev_frame is not None else None
        # If we somehow have no prev_frame, fall back to a zero-filled
        # 64x64 grid so the contract is satisfied (the level-0 default).
        if grid is None:
            grid = Grid(
                data=[[0] * 64 for _ in range(64)],
                step=self.episode_step,
                level=self.current_level,
            )
        info = StepInfo(
            state="SESSION_LOST",
            step_id=step_id,
            global_step=self.global_step,
            episode_step=self.episode_step,
            levels_completed=getattr(self._prev_frame, "levels_completed", self.current_level),
            win_levels=getattr(self._prev_frame, "win_levels", 0),
            full_reset=False,
            actions_this_level=self.actions_this_level,
            current_level=self.current_level,
            action_space=[],
            available_actions=[],
            sdk_total_actions=self.sdk_total_actions,
            sdk_resets=self.sdk_resets,
            sdk_level_actions=list(self.sdk_level_actions),
            diff_grid=DiffGrid.no_change(grid),
        )
        return grid, 0.0, True, info

    def _record_current_state(self, grid: Grid) -> None:
        """Record the current grid at ``state_history[global_step]``."""
        if len(self.state_history) < self.global_step:
            raise RuntimeError(
                "Environment state ledger is behind global_step: "
                f"len(state_history)={len(self.state_history)}, "
                f"global_step={self.global_step}"
            )
        if len(self.state_history) == self.global_step:
            self.state_history.append(grid)
        else:
            self.state_history[self.global_step] = grid
        self.validate_step_ledger()

    def _record_transition(
        self,
        *,
        step_id: int,
        action_name: str,
        action_data: dict | None,
        grid: Grid,
        reward: float,
        done: bool,
        info: StepInfo,
        level: int | None = None,
        level_completed: bool = False,
        diff_pixels: Any = _UNSET,
        actions_this_level: int | None = None,
        unique_colors_delta: Any = _UNSET,
        event=(),
        event_frame_count: Any = _UNSET,
        event_source_frame_count: Any = _UNSET,
        event_frame_indices: Any = _UNSET,
        event_decimation: str | None = None,
        event_reduced_frame_position: Any = _UNSET,
        event_reduced_frame_source_index: Any = _UNSET,
        final_state: Any = _UNSET,
        final_event: Any = _UNSET,
    ) -> None:
        """Record the action transition at ``transition_history[step_id]``."""
        if len(self.transition_history) != step_id:
            raise RuntimeError(
                "Environment transition ledger is not dense before step "
                f"{step_id}: len(transition_history)="
                f"{len(self.transition_history)}"
            )
        if len(self.state_history) != step_id + 1:
            raise RuntimeError(
                "Environment state ledger is not positioned at the pre-action "
                f"state for step {step_id}: len(state_history)="
                f"{len(self.state_history)}"
            )
        event_data = _event_to_agent_data(event)
        event_count = len(event_data)
        if event_source_frame_count is _UNSET:
            source_count = event_count
        else:
            try:
                source_count = int(event_source_frame_count or 0)
            except (TypeError, ValueError):
                source_count = event_count
            if event_count and source_count <= 0:
                source_count = event_count
        indices = _normalize_event_frame_indices(event_frame_indices, event_count)
        reduced_position = _normalize_event_reduced_frame_position(
            (
                getattr(info, "event_reduced_frame_position", None)
                if event_reduced_frame_position is _UNSET
                else event_reduced_frame_position
            ),
            frame_count=event_count,
        )
        if event_reduced_frame_source_index is _UNSET:
            reduced_source_index = (
                indices[reduced_position]
                if reduced_position is not None and reduced_position < len(indices)
                else getattr(info, "event_reduced_frame_source_index", None)
            )
        else:
            reduced_source_index = event_reduced_frame_source_index
        final_state_value = (
            getattr(info, "final_state", None)
            if final_state is _UNSET else final_state
        )
        final_event_value = (
            getattr(info, "final_event", None)
            if final_event is _UNSET else final_event
        )
        self.transition_history.append({
            "step": step_id,
            "action": action_name,
            "action_data": action_data,
            "reward": reward,
            "done": done,
            "state": info.state,
            "levels_completed": info.levels_completed,
            "level": (
                level if level is not None
                else getattr(info, "current_level", info.levels_completed)
            ),
            "level_completed": bool(level_completed),
            "episode": self.episode,
            "episode_step": self.episode_step,
            "diff_pixels": (
                info.diff_pixels if diff_pixels is _UNSET else diff_pixels
            ),
            "actions_this_level": (
                info.actions_this_level
                if actions_this_level is None else actions_this_level
            ),
            "unique_colors_delta": copy.deepcopy(
                info.unique_colors_delta
                if unique_colors_delta is _UNSET else unique_colors_delta
            ) or {},
            "event": event_data,
            "event_frame_count": event_count,
            "event_source_frame_count": source_count,
            "event_frame_indices": indices,
            "event_decimation": event_decimation or "",
            "event_reduced_frame_position": reduced_position,
            "event_reduced_frame_source_index": reduced_source_index,
            "final_state": _grid_to_agent_data(final_state_value),
            "final_event": (
                None if final_event_value is None
                else _event_to_agent_data(final_event_value)
            ),
        })
        self.state_history.append(grid)
        self.validate_step_ledger()

    def validate_step_ledger(self) -> None:
        """Raise if the environment's global step ledger is not dense."""
        if not self.state_history:
            if self.global_step != 0 or self.transition_history:
                raise RuntimeError(
                    "Environment state ledger is empty after steps were issued"
                )
            return
        if len(self.state_history) != self.global_step + 1:
            raise RuntimeError(
                "Environment state ledger length does not match global_step: "
                f"len(state_history)={len(self.state_history)}, "
                f"global_step={self.global_step}"
            )
        if len(self.transition_history) != self.global_step:
            raise RuntimeError(
                "Environment transition ledger length does not match "
                f"global_step: len(transition_history)="
                f"{len(self.transition_history)}, global_step={self.global_step}"
            )
        steps = [row.get("step") for row in self.transition_history]
        expected = list(range(self.global_step))
        if steps != expected:
            raise RuntimeError(
                "Environment transition ledger is not dense: "
                f"steps={steps[:10]}... expected={expected[:10]}..."
            )

    def get_agent_trajectory(
        self,
        *,
        scope: str = "current",
        visible_step: int | None = None,
        level: int | None = None,
    ) -> list[dict]:
        """Return the agent-facing trajectory rebuilt from env ledgers."""
        return _get_agent_trajectory_from_ledgers(
            self,
            scope=scope,
            visible_step=visible_step,
            level=level,
        )

    def get_observation(self, step: int) -> dict:
        return _get_observation_from_ledgers(self, step)

    def get_observations(
        self,
        visible_step: int | None = None,
    ) -> list[dict]:
        return _get_observations_from_ledgers(self, visible_step)

    def get_transitions(
        self,
        visible_step: int | None = None,
    ) -> list[dict]:
        return _get_transitions_from_ledgers(self, visible_step)

    def level_start_step(
        self,
        level: int,
        visible_step: int | None = None,
    ) -> int:
        return _level_start_step_from_observation_rows(self, level, visible_step)

    def reset(self) -> Grid:
        """Reset the current episode. Returns the initial grid (64x64)."""
        reset_started_at, reset_started_unix_time_s = _wall_clock_stamp()
        reset_started_monotonic = time.monotonic()
        self.episode_step = 0
        self.episode += 1
        self.current_level = 0
        self.actions_this_level = 0
        self.actions_since_reset = 0
        self._local_level_actions = []
        self.episode_history = []

        frame = self._reset_with_retry()
        reset_finished_at, reset_finished_unix_time_s = _wall_clock_stamp()
        reset_finished_monotonic = time.monotonic()
        self._last_step_confirmed_at = time.monotonic()

        self.current_level = frame.levels_completed
        grid = self._extract_grid(frame, step=0, level=self.current_level)
        self._prev_frame = frame
        self._prev_grid = grid

        # g_0 vs g_0 = no change
        self.diff_grid = DiffGrid.no_change(grid)

        # Populate available_actions from frame (stable within a game)
        if not self._available_actions_ints:
            self._available_actions_ints = _inject_reset(list(frame.available_actions))

        # Sync counters from SDK scorecard
        self._sync_scorecard_stats()

        self.episode_history.append(grid)
        self._record_current_state(grid)

        # Log reset
        self.gameplay_log.append({
            "type": "reset",
            "timestamp": reset_finished_at,
            "unix_time_s": reset_finished_unix_time_s,
            "reset_started_at": reset_started_at,
            "reset_started_unix_time_s": reset_started_unix_time_s,
            "reset_finished_at": reset_finished_at,
            "reset_finished_unix_time_s": reset_finished_unix_time_s,
            "reset_duration_seconds": _elapsed_seconds(
                reset_started_monotonic,
                reset_finished_monotonic,
            ),
            "global_step": self.global_step,
            "episode": self.episode,
            "episode_step": 0,
            "levels_completed": self.current_level,
            "available_actions": self.action_space_ints,
            "sdk_total_actions": self.sdk_total_actions,
            "sdk_resets": self.sdk_resets,
            "diff_pixels": 0,
        })
        self._save_gameplay()

        return grid

    def step(
        self,
        action: Any,
        data: dict | None = None,
        reasoning: Any | None = None,
    ) -> tuple[Grid, float, bool, StepInfo]:
        """Execute an action in the environment.

        Args:
            action: Action as string name ("UP", "CLICK 32 15"), int, or GameAction enum.
            data: Optional action data for CLICK (e.g. {"x": 32, "y": 32}).
                  Can also be parsed from string form "CLICK 32 15".
            reasoning: Optional JSON-serializable payload to pass through to
                the ARC SDK/API as action reasoning metadata.

        Returns:
            (grid, reward, done, info) tuple.
            info includes diff_grid, diff_pixels, and available_actions.
        """
        step_id = self.global_step
        pre_action_level = self.current_level
        actions_before = self.actions_this_level

        # Convert string/int to GameAction enum
        if isinstance(action, str):
            action_int, parsed_data = self.action_to_int(action)
            if parsed_data and data is None:
                data = parsed_data
            action = GA_MAP[action_int]
        elif isinstance(action, int):
            action_int = action
            action = GA_MAP[action]
        else:
            action_int = action.value if hasattr(action, "value") else action

        action_repr = action_int
        if isinstance(action_int, tuple):
            action_repr = list(action_int)
        elif data:
            action_repr = [action_int, data.get("x", 0), data.get("y", 0)]

        action_started_at, action_started_unix_time_s = _wall_clock_stamp()
        action_started_monotonic = time.monotonic()
        frame = self._step_with_retry(action, data=data, reasoning=reasoning)
        action_finished_at, action_finished_unix_time_s = _wall_clock_stamp()
        action_finished_monotonic = time.monotonic()

        self.global_step = step_id + 1
        self.episode_step += 1
        self.actions_this_level += 1
        transition_actions_this_level = self.actions_this_level

        # Detect level completion
        level_up = (
            self._prev_frame is not None
            and frame.levels_completed > self._prev_frame.levels_completed
        )

        completed_level = None
        if level_up:
            completed_level = self._prev_frame.levels_completed
            self._local_level_actions.append(self.actions_this_level)
            self._sync_scorecard_stats()
            reward, done, components = self._compute_reward(frame, completed_level)
            self.current_level = frame.levels_completed
            transition_actions_this_level = actions_before + 1
            self.actions_this_level = 0
        else:
            reward, done, components = self._compute_reward(frame, None)

        if action_int == 0 or level_up:
            self.actions_since_reset = 0
        else:
            self.actions_since_reset += 1

        grid = self._extract_grid(frame, step=self.episode_step, level=self.current_level)
        event_sample = _extract_event_sample(
            frame,
            level_up=level_up,
            step=self.episode_step,
            level=(
                completed_level
                if level_up and completed_level is not None
                else self.current_level
            ),
            prev_grid_data=self._prev_grid.data if self._prev_grid else None,
            max_frames=self._max_event_frames,
            decimation_algorithm=self._event_decimation_algorithm,
            min_consecutive_frames=(
                self._event_decimation_min_consecutive_frames
            ),
            stable_dominance_threshold=(
                self._event_decimation_stable_dominance_threshold
            ),
            min_stable_frames=self._event_decimation_min_stable_frames,
            report_original_event_size=self._report_original_event_size,
        )
        if level_up:
            event = []
            final_event = event_sample["frames"]
            final_state = event_sample["final_state"]
            event_metadata = _empty_event_sample("level_up_suppressed")
        else:
            event = event_sample["frames"]
            final_event = None
            final_state = None
            event_metadata = event_sample
        event_frame_count = len(event or [])

        # Compute diff between previous and current grid
        if self._prev_grid is not None:
            self.diff_grid = DiffGrid.from_grids(self._prev_grid, grid)
        else:
            self.diff_grid = DiffGrid.no_change(grid)

        colors_delta = _unique_colors_delta(self._prev_grid, grid)

        # If session was recovered via reset, override state so caller
        # can detect it and route through the harness reset flow.
        if getattr(frame, '_session_recovered', False):
            _state = "SESSION_RECOVERED"
        else:
            _state = str(frame.state) if hasattr(frame.state, "value") else frame.state

        info = StepInfo(
            state=_state,
            step_id=step_id,
            global_step=self.global_step,
            episode_step=self.episode_step,
            levels_completed=frame.levels_completed,
            win_levels=frame.win_levels,
            full_reset=frame.full_reset,
            actions_this_level=self.actions_this_level,
            current_level=self.current_level,
            action_space=frame.available_actions,
            available_actions=self.action_space_ints,
            sdk_total_actions=self.sdk_total_actions,
            sdk_resets=self.sdk_resets,
            sdk_level_actions=self.sdk_level_actions,
            diff_grid=self.diff_grid,
            diff_pixels=self.diff_grid.num_changed,
            event=event or [],
            event_frame_count=event_frame_count,
            event_source_frame_count=int(event_metadata["source_frame_count"]),
            event_frame_indices=list(event_metadata["frame_indices"]),
            event_decimation=str(event_metadata["decimation"]),
            event_reduced_frame_position=event_metadata["reduced_frame_position"],
            event_reduced_frame_source_index=event_metadata["reduced_frame_source_index"],
            unique_colors_delta=colors_delta,
            reward_components=components,
            final_state=final_state,
            final_event=final_event,
        )

        self._prev_frame = frame
        self._prev_grid = grid

        # Record in episode history
        self.episode_history.append(grid)

        # Log step to gameplay log
        self.gameplay_log.append({
            "type": "step",
            "timestamp": action_finished_at,
            "unix_time_s": action_finished_unix_time_s,
            "action_started_at": action_started_at,
            "action_started_unix_time_s": action_started_unix_time_s,
            "action_finished_at": action_finished_at,
            "action_finished_unix_time_s": action_finished_unix_time_s,
            "action_duration_seconds": _elapsed_seconds(
                action_started_monotonic,
                action_finished_monotonic,
            ),
            "global_step": self.global_step,
            "episode": self.episode,
            "episode_step": self.episode_step,
            "action": action_repr,
            "action_data": data,
            "reward": reward,
            "done": done,
            "diff_pixels": self.diff_grid.num_changed,
            "levels_completed": frame.levels_completed,
            "state": str(frame.state) if hasattr(frame.state, "value") else frame.state,
            "available_actions": self.action_space_ints,
            "sdk_total_actions": self.sdk_total_actions,
            "sdk_resets": self.sdk_resets,
        })
        self._record_transition(
            step_id=step_id,
            action_name=self.action_to_name(action_int, data),
            action_data=data,
            grid=grid,
            reward=reward,
            done=done,
            info=info,
            level=pre_action_level,
            level_completed=level_up,
            diff_pixels=None if level_up else info.diff_pixels,
            actions_this_level=transition_actions_this_level,
            unique_colors_delta=None if level_up else colors_delta,
            event=event,
            event_frame_count=info.event_frame_count,
            event_source_frame_count=info.event_source_frame_count,
            event_frame_indices=info.event_frame_indices,
            event_decimation=info.event_decimation,
            event_reduced_frame_position=info.event_reduced_frame_position,
            event_reduced_frame_source_index=info.event_reduced_frame_source_index,
            final_state=info.final_state,
            final_event=info.final_event,
        )
        self._save_gameplay()

        return grid, reward, done, info

    def set_results_dir(self, path: str | Path) -> None:
        """Set the directory where gameplay logs are saved."""
        self._results_dir = Path(path)

    def _save_gameplay(self) -> None:
        """Save gameplay log to results dir (called after every step/reset)."""
        if self._results_dir is None:
            return
        try:
            gameplay_path = self._results_dir / "gameplay.json"
            self._results_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            with open(gameplay_path, "w") as f:
                _json.dump({
                    "game_id": self.game_id,
                    "seed": self.seed,
                    "global_step": self.global_step,
                    "episode": self.episode,
                    "levels_completed": self.current_level,
                    "total_steps": len([e for e in self.gameplay_log if e["type"] == "step"]),
                    "log": self.gameplay_log,
                }, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save gameplay log: {e}")

    def _sync_scorecard_stats(self) -> None:
        """Read authoritative action/reset counters from the SDK scorecard.

        Called on reset and on level completion only.
        Updates sdk_total_actions, sdk_resets, sdk_level_actions.

        Skipped in competition mode — the API blocks get_scorecard() during
        active competition. Use get_scorecard() after all games are done.
        """
        if self._operation_mode == "competition":
            return
        try:
            sc = self.arcade.get_scorecard(self._scorecard_id)
        except Exception as e:
            logger.warning(f"get_scorecard() failed (non-fatal): {e}")
            return
        if not sc:
            return
        d = sc.model_dump() if hasattr(sc, "model_dump") else {}
        for env_data in d.get("environments", []):
            if env_data.get("id") == self.game_id:
                run = env_data["runs"][-1]
                self.sdk_total_actions = run.get("actions", 0)
                self.sdk_resets = run.get("resets", 0)
                self.sdk_level_actions = run.get("level_actions", [])
                return

    def try_sync_scorecard(self) -> bool:
        """Attempt scorecard sync regardless of operation mode.

        Useful after competition ends when get_scorecard() may become available.
        Returns True if sync succeeded and sdk_level_actions was updated.
        """
        try:
            sc = self.arcade.get_scorecard(self._scorecard_id)
        except Exception as e:
            if self._operation_mode == "competition" and _http_status(e) == 403:
                logger.info(
                    "try_sync_scorecard() unavailable while competition "
                    "scorecard reads are forbidden (HTTP 403)"
                )
                return False
            logger.warning(f"try_sync_scorecard() failed: {e}")
            return False
        if not sc:
            return False
        d = sc.model_dump() if hasattr(sc, "model_dump") else {}
        for env_data in d.get("environments", []):
            if env_data.get("id") == self.game_id:
                run = env_data["runs"][-1]
                self.sdk_total_actions = run.get("actions", 0)
                self.sdk_resets = run.get("resets", 0)
                self.sdk_level_actions = run.get("level_actions", [])
                return True
        return False

    def render(self) -> None:
        """Render the environment to terminal (if render_mode was set)."""
        if self._render_mode and hasattr(self.env, "render"):
            self.env.render()

    # Canonical action name mapping — single source of truth
    ACTION_NAMES = {
        0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT",
        4: "RIGHT", 5: "USE", 6: "CLICK", 7: "UNDO",
    }
    ACTION_FROM_NAME = {v: k for k, v in ACTION_NAMES.items()}

    @property
    def action_space(self) -> list:
        """Available actions from the underlying environment."""
        return self.env.action_space

    @property
    def action_space_ints(self) -> list[int]:
        """Available actions as integer values."""
        return list(self._available_actions_ints)

    @property
    def action_names(self) -> list[str]:
        """Available actions as text names (e.g., ['UP', 'DOWN', 'LEFT', 'RIGHT'])."""
        return [self.ACTION_NAMES[i] for i in self._available_actions_ints]

    def _env_has_keyboard_controls(self) -> bool:
        return _env_has_keyboard_controls(self)

    def _resolve_add_grid_orientation(self, value) -> bool:
        return _resolve_add_grid_orientation(value, self)

    def get_history(self) -> list[StepRecord]:
        """Build structured step history from gameplay_log + episode_history.

        Returns a list of StepRecord objects. The first entry (step 0) always
        has action=None (the initial observation). Subsequent entries correspond
        to actions taken and their resulting grids.

        episode_history and the step-type entries in gameplay_log are 1:1
        aligned (index 0 = reset grid, index k = grid after k-th step).
        """
        records: list[StepRecord] = []
        if not self.episode_history:
            return records

        records.append(StepRecord(
            step=0,
            grid=self.episode_history[0],
            action=None,
            levels_completed=0,
            reward=0.0,
        ))

        step_idx = 1
        for log_entry in self.gameplay_log:
            if log_entry.get("type") != "step":
                continue

            grid = (
                self.episode_history[step_idx]
                if step_idx < len(self.episode_history)
                else self.episode_history[-1]
            )

            action_repr = log_entry.get("action", None)
            if isinstance(action_repr, list):
                action_int = action_repr[0]
            elif isinstance(action_repr, int):
                action_int = action_repr
            else:
                action_int = action_repr
            action_name = self.ACTION_NAMES.get(action_int, str(action_repr)) if isinstance(action_int, int) else str(action_repr)

            if isinstance(action_repr, list) and len(action_repr) == 3 and action_int == 6:
                action_name = f"CLICK {action_repr[1]} {action_repr[2]}"

            records.append(StepRecord(
                step=step_idx,
                grid=grid,
                action=action_name,
                levels_completed=log_entry.get("levels_completed", 0),
                reward=log_entry.get("reward", 0.0),
            ))
            step_idx += 1

        return records

    @classmethod
    def action_to_int(cls, name: str, x: int = -1, y: int = -1) -> tuple[int, dict | None]:
        """Convert action name to (action_int, action_data).

        E.g., 'UP' -> (1, None), 'CLICK 32 15' -> (6, {'x': 32, 'y': 15})
        """
        parts = name.strip().upper().split()
        action_name = parts[0]
        if action_name not in cls.ACTION_FROM_NAME:
            raise ValueError(f"Unknown action: '{action_name}'")
        action_int = cls.ACTION_FROM_NAME[action_name]
        action_data = None
        if action_int == 6:
            if len(parts) >= 3:
                action_data = {"x": int(parts[1]), "y": int(parts[2])}
            elif x >= 0 and y >= 0:
                action_data = {"x": x, "y": y}
            else:
                raise ValueError("CLICK requires x y coordinates")
        return action_int, action_data

    @classmethod
    def action_to_name(cls, action_int: int, action_data: dict | None = None) -> str:
        """Convert action int to text name.

        E.g., 1 -> 'UP', (6, {'x': 32, 'y': 15}) -> 'CLICK 32 15'
        """
        name = cls.ACTION_NAMES.get(action_int, str(action_int))
        if action_int == 6 and action_data:
            return f"CLICK {action_data.get('x', 0)} {action_data.get('y', 0)}"
        return name

    @classmethod
    def list_games(cls, operation_mode: str = "offline") -> list[str]:
        """List all available game IDs."""
        mode_map = {
            "online": OperationMode.ONLINE,
            "offline": OperationMode.OFFLINE,
            "normal": OperationMode.NORMAL,
            "competition": OperationMode.COMPETITION,
        }
        arcade = Arcade(
            operation_mode=mode_map.get(operation_mode, OperationMode.OFFLINE),
            arc_api_key=os.environ.get("ARC_API_KEY", ""),
        )
        return [e.game_id for e in arcade.get_environments()]

    @classmethod
    def _match_game_ids(cls, name: str, game_ids: list[str]) -> list[str]:
        """Match name against game_ids by exact full, exact title, or prefix."""
        lower = name.lower()

        def _uniq(seq: list[str]) -> list[str]:
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        exact_full = [gid for gid in game_ids if gid.lower() == lower]
        if exact_full:
            return _uniq(exact_full)

        exact_title = [gid for gid in game_ids if gid.split("-")[0].lower() == lower]
        if exact_title:
            return _uniq(exact_title)

        prefix_matches = [
            gid for gid in game_ids
            if gid.lower().startswith(lower) or gid.split("-")[0].lower().startswith(lower)
        ]
        return _uniq(prefix_matches)

    @classmethod
    def resolve_game_ids(cls, name: str, operation_mode: str = "offline") -> list[str]:
        """Resolve a full ID, exact short title, or unique/non-unique prefix to game IDs."""
        if not name:
            return []
        matches = cls._match_game_ids(name, cls.list_games(operation_mode))
        if matches or operation_mode != "offline":
            return matches

        logger.info(f"Game '{name}' not found offline, attempting download...")
        normal_ids = cls.list_games("normal")
        normal_matches = cls._match_game_ids(name, normal_ids)
        if not normal_matches:
            return []

        api_key = os.environ.get("ARC_API_KEY", "")
        arcade = Arcade(operation_mode=OperationMode.NORMAL, arc_api_key=api_key)
        for game_id in normal_matches:
            try:
                arcade.make(game_id)
                logger.info(f"Downloaded game '{game_id}' for offline use")
            except Exception as e:
                logger.warning(f"Failed to download game '{game_id}': {e}")

        return cls._match_game_ids(name, cls.list_games(operation_mode))

    @classmethod
    def resolve_game_id(cls, name: str, operation_mode: str = "offline") -> str:
        """Resolve a single game id or raise on unknown/ambiguous prefixes."""
        matches = cls.resolve_game_ids(name, operation_mode)
        if not matches:
            raise ValueError(f"Unknown game id/prefix '{name}'")
        if len(matches) > 1:
            sample = ", ".join(matches[:8])
            raise ValueError(f"Ambiguous game prefix '{name}': matches {sample}")
        return matches[0]

    def get_scorecard(self) -> dict | None:
        """Get current scorecard."""
        if self.arcade is not None:
            try:
                sc = self.arcade.get_scorecard(self._scorecard_id)
                if sc:
                    return sc.model_dump() if hasattr(sc, "model_dump") else vars(sc)
            except Exception as e:
                if self._operation_mode == "competition" and _http_status(e) == 403:
                    logger.info(
                        "get_scorecard() unavailable while competition "
                        "scorecard reads are forbidden (HTTP 403)"
                    )
                    return None
                logger.warning(f"Failed to get scorecard: {e}")
        return None

    @staticmethod
    def _frame_to_int_grid(raw_array) -> list[list[int]]:
        """Convert a raw numpy frame array to list[list[int]].

        Single point of conversion — ensures all grid data is integer
        regardless of the dtype returned by the ARC-AGI SDK.
        """
        import numpy as np
        return np.asarray(raw_array, dtype=int).tolist()

    def _extract_grid(self, frame: Any, step: int = 0, level: int = 0) -> Grid:
        """Extract grid from FrameDataRaw.frame and wrap as Grid.

        Uses the last frame in the list. For normal steps there's only one frame.
        On level transitions the SDK returns 2+ frames: old level final → new level
        initial. Taking [-1] gives the most recent (new level) state.
        """
        data = self._frame_to_int_grid(frame.frame[-1])
        h, w = len(data), len(data[0]) if data else 0
        # Grid.diff_count, Grid.render_text, and all DSL functions assume 64x64
        if (h, w) != (64, 64):
            raise ValueError(f"SDK returned {h}x{w} grid, expected 64x64")
        return Grid(data=data, step=step, level=level)

    def _compute_reward(
        self, curr_frame: Any, completed_level: int | None = None,
    ) -> tuple[float, bool, dict]:
        """Compute reward from frame data.

        Args:
            curr_frame: Current SDK frame after the action.
            completed_level: Index of the level that was just completed,
                or None if no level was completed on this step.

        Reward = level_factor / actions_this_level  (on level completion only).
        """
        done = curr_frame.state in (GameState.WIN, GameState.GAME_OVER)

        r_level = 0.0
        if completed_level is not None:
            if completed_level < len(self.sdk_level_actions):
                actions_used = self.sdk_level_actions[completed_level]
            else:
                actions_used = self.actions_this_level
            r_level = self.reward_config.level_factor / max(actions_used, 1)

        r_gameover = 0.0
        if curr_frame.state == GameState.GAME_OVER:
            r_gameover = -self.reward_config.gameover_penalty

        reward = r_level + r_gameover
        components = {
            "level": r_level,
            "gameover": r_gameover,
        }

        return reward, done, components

    def compute_rhae_scores(self) -> tuple[float, list[float]]:
        """Compute RHAE game score and per-level scores from current state."""
        level_actions = self.sdk_level_actions if self.sdk_level_actions else getattr(self, '_local_level_actions', [])
        return _rhae_game_score(
            self.baseline_actions, level_actions, self.current_level,
        )

    def render_episode(self) -> str:
        """Render a text summary of the episode history."""
        if not self.episode_history:
            return "No episode history recorded."
        lines = [
            f"Episode {self.episode} — {len(self.episode_history)} frames, "
            f"game={self.game_id}"
        ]
        for i, grid in enumerate(self.episode_history):
            colors = grid.unique_colors()
            lines.append(
                f"  [{i:3d}] step={grid.step} level={grid.level} "
                f"colors={colors}"
            )
            if i > 0:
                diff = grid.diff_count(self.episode_history[i - 1])
                lines.append(f"        diff_from_prev={diff} pixels")
        return "\n".join(lines)


class ReplayEnvironment:
    """Replay environment that plays back stored trajectory data.

    Provides the same reset()/step() interface as ARCAGI3Env but reads from
    stored exploration data (deep_exploration_results.json) instead of the
    live SDK. Useful for local testing without API access.
    """

    def __init__(
        self,
        game_title: str,
        data_path: str | Path | None = None,
        scenario: str = "trajectory",
        level_factor: float = 100.0,
    ):
        """Initialize a replay environment.

        Args:
            game_title: Game title (e.g. "LS20", "FT09", "VC33").
            data_path: Path to deep_exploration_results.json.
                       Defaults to arc_agi_3/demo/deep_exploration_results.json.
            scenario: Which data to replay — "trajectory" for the main trajectory,
                      or a scenario name from level_completion (e.g. "repeat_action1").
            level_factor: Reward = level_factor / actions_this_level on level completion.
        """
        self.game_title = game_title
        self.scenario = scenario
        self.level_factor = level_factor

        if data_path is None:
            data_path = (
                Path(__file__).parent / "demo" / "deep_exploration_results.json"
            )
        self._data_path = Path(data_path)

        # Load exploration data
        with open(self._data_path) as f:
            all_data = json.load(f)

        game_data = all_data.get("games", {}).get(game_title)
        if game_data is None:
            raise ValueError(
                f"Game '{game_title}' not found. "
                f"Available: {list(all_data.get('games', {}).keys())}"
            )

        self.game_id = game_data["game_id"]
        self.action_effects = game_data.get("action_effects", {})

        # Resolve trajectory steps
        if scenario == "trajectory":
            traj = game_data.get("trajectory", {})
            # trajectory can be {"steps": [...]} or just [...]
            self._steps = traj.get("steps", []) if isinstance(traj, dict) else traj
        else:
            # Look in level_completion scenarios
            lc = all_data.get("level_completion", {}).get(game_title, {})
            sc_data = lc.get(scenario, {})
            self._steps = sc_data.get("action_sequence", [])
            if not self._steps:
                available = list(lc.keys()) if lc else []
                raise ValueError(
                    f"Scenario '{scenario}' not found for {game_title}. "
                    f"Available: {available}"
                )

        # Load grid snapshots (.npy files)
        grid_dir = self._data_path.parent / "grids"
        self._grid_snapshots: dict[str, np.ndarray] = {}
        for tag in ("reset", "step5", "step20"):
            npy_path = grid_dir / f"{game_title}_{tag}.npy"
            if npy_path.exists():
                self._grid_snapshots[tag] = np.load(npy_path)

        # Game metadata from SDK (get_available_actions already injects RESET)
        env_info = get_env_info(self.game_id)
        self.baseline_actions = list(env_info.baseline_actions)
        self._available_actions_ints = get_available_actions(self.game_id, "offline")
        self.sdk_total_actions = 0
        self.sdk_resets = 0
        self.sdk_level_actions: list[int] = []

        # State
        self._cursor = 0
        self.episode_step = 0
        self.global_step = 0
        self.episode = 0
        self.current_level = 0
        self.actions_this_level = 0
        self.episode_history: list[Grid] = []
        self.state_history: list[Grid] = []
        self.transition_history: list[dict] = []
        self._prev_grid: Grid | None = None
        self.diff_grid: DiffGrid | None = None
        self.gameplay_log: list[dict] = []

    def _record_current_state(self, grid: Grid) -> None:
        if len(self.state_history) < self.global_step:
            raise RuntimeError(
                "Replay state ledger is behind global_step: "
                f"len(state_history)={len(self.state_history)}, "
                f"global_step={self.global_step}"
            )
        if len(self.state_history) == self.global_step:
            self.state_history.append(grid)
        else:
            self.state_history[self.global_step] = grid
        self.validate_step_ledger()

    def _record_transition(
        self,
        *,
        step_id: int,
        action_name: str,
        action_data: Any,
        grid: Grid,
        reward: float,
        done: bool,
        info: StepInfo,
        level: int | None = None,
        level_completed: bool = False,
        diff_pixels: Any = _UNSET,
        actions_this_level: int | None = None,
        unique_colors_delta: Any = _UNSET,
        event=(),
        event_frame_count: Any = _UNSET,
        event_source_frame_count: Any = _UNSET,
        event_frame_indices: Any = _UNSET,
        event_decimation: str | None = None,
        event_reduced_frame_position: Any = _UNSET,
        event_reduced_frame_source_index: Any = _UNSET,
        final_state: Any = _UNSET,
        final_event: Any = _UNSET,
    ) -> None:
        if len(self.transition_history) != step_id:
            raise RuntimeError(
                "Replay transition ledger is not dense before step "
                f"{step_id}: len(transition_history)="
                f"{len(self.transition_history)}"
            )
        if len(self.state_history) != step_id + 1:
            raise RuntimeError(
                "Replay state ledger is not positioned at the pre-action "
                f"state for step {step_id}: len(state_history)="
                f"{len(self.state_history)}"
            )
        event_data = _event_to_agent_data(event)
        event_count = len(event_data)
        if event_source_frame_count is _UNSET:
            source_count = event_count
        else:
            try:
                source_count = int(event_source_frame_count or 0)
            except (TypeError, ValueError):
                source_count = event_count
            if event_count and source_count <= 0:
                source_count = event_count
        indices = _normalize_event_frame_indices(event_frame_indices, event_count)
        reduced_position = _normalize_event_reduced_frame_position(
            (
                getattr(info, "event_reduced_frame_position", None)
                if event_reduced_frame_position is _UNSET
                else event_reduced_frame_position
            ),
            frame_count=event_count,
        )
        if event_reduced_frame_source_index is _UNSET:
            reduced_source_index = (
                indices[reduced_position]
                if reduced_position is not None and reduced_position < len(indices)
                else getattr(info, "event_reduced_frame_source_index", None)
            )
        else:
            reduced_source_index = event_reduced_frame_source_index
        final_state_value = (
            getattr(info, "final_state", None)
            if final_state is _UNSET else final_state
        )
        final_event_value = (
            getattr(info, "final_event", None)
            if final_event is _UNSET else final_event
        )
        self.transition_history.append({
            "step": step_id,
            "action": action_name,
            "action_data": action_data,
            "reward": reward,
            "done": done,
            "state": info.state,
            "levels_completed": info.levels_completed,
            "level": (
                level if level is not None
                else getattr(info, "current_level", info.levels_completed)
            ),
            "level_completed": bool(level_completed),
            "episode": self.episode,
            "episode_step": self.episode_step,
            "diff_pixels": (
                info.diff_pixels if diff_pixels is _UNSET else diff_pixels
            ),
            "actions_this_level": (
                info.actions_this_level
                if actions_this_level is None else actions_this_level
            ),
            "unique_colors_delta": copy.deepcopy(
                info.unique_colors_delta
                if unique_colors_delta is _UNSET else unique_colors_delta
            ) or {},
            "event": event_data,
            "event_frame_count": event_count,
            "event_source_frame_count": source_count,
            "event_frame_indices": indices,
            "event_decimation": event_decimation or "",
            "event_reduced_frame_position": reduced_position,
            "event_reduced_frame_source_index": reduced_source_index,
            "final_state": _grid_to_agent_data(final_state_value),
            "final_event": (
                None if final_event_value is None
                else _event_to_agent_data(final_event_value)
            ),
        })
        self.state_history.append(grid)
        self.validate_step_ledger()

    def validate_step_ledger(self) -> None:
        if not self.state_history:
            if self.global_step != 0 or self.transition_history:
                raise RuntimeError("Replay state ledger is empty after steps")
            return
        if len(self.state_history) != self.global_step + 1:
            raise RuntimeError(
                "Replay state ledger length does not match global_step: "
                f"len(state_history)={len(self.state_history)}, "
                f"global_step={self.global_step}"
            )
        if len(self.transition_history) != self.global_step:
            raise RuntimeError(
                "Replay transition ledger length does not match global_step: "
                f"len(transition_history)={len(self.transition_history)}, "
                f"global_step={self.global_step}"
            )
        steps = [row.get("step") for row in self.transition_history]
        expected = list(range(self.global_step))
        if steps != expected:
            raise RuntimeError(
                "Replay transition ledger is not dense: "
                f"steps={steps[:10]}... expected={expected[:10]}..."
            )

    def get_agent_trajectory(
        self,
        *,
        scope: str = "current",
        visible_step: int | None = None,
        level: int | None = None,
    ) -> list[dict]:
        return _get_agent_trajectory_from_ledgers(
            self,
            scope=scope,
            visible_step=visible_step,
            level=level,
        )

    def get_observation(self, step: int) -> dict:
        return _get_observation_from_ledgers(self, step)

    def get_observations(
        self,
        visible_step: int | None = None,
    ) -> list[dict]:
        return _get_observations_from_ledgers(self, visible_step)

    def get_transitions(
        self,
        visible_step: int | None = None,
    ) -> list[dict]:
        return _get_transitions_from_ledgers(self, visible_step)

    def level_start_step(
        self,
        level: int,
        visible_step: int | None = None,
    ) -> int:
        return _level_start_step_from_observation_rows(self, level, visible_step)

    def reset(self) -> Grid:
        """Reset and return the initial grid."""
        self._cursor = 0
        self.episode_step = 0
        self.episode += 1
        self.current_level = 0
        self.actions_this_level = 0
        self.episode_history = []

        # Use stored reset grid if available
        if "reset" in self._grid_snapshots:
            data = ARCAGI3Env._frame_to_int_grid(self._grid_snapshots["reset"])
        else:
            data = [[0] * 64 for _ in range(64)]

        grid = Grid(data=data, step=0, level=0)
        self._prev_grid = grid
        self.diff_grid = DiffGrid.no_change(grid)
        self.episode_history.append(grid)
        self._record_current_state(grid)
        return grid

    def step(
        self,
        action: Any = None,
        data: dict | None = None,
        reasoning: Any | None = None,
    ) -> tuple[Grid, float, bool, StepInfo]:
        """Advance one step in the stored trajectory.

        The action argument is accepted for interface compatibility but the
        replay always follows the stored sequence. ``reasoning`` is accepted
        for parity with ARCAGI3Env.step and ignored.
        """
        if self._cursor >= len(self._steps):
            grid = self._prev_grid or Grid.empty()
            return grid, 0.0, True, StepInfo(replay_exhausted=True)

        step_id = self.global_step
        step_data = self._steps[self._cursor]
        self._cursor += 1
        pre_action_level = self.current_level
        actions_before = self.actions_this_level
        self.episode_step += 1
        self.global_step += 1
        self.actions_this_level += 1
        transition_actions_this_level = self.actions_this_level

        # Extract step metadata
        diff_pixels = step_data.get("diff_pixels", 0)
        levels_completed = step_data.get("levels_completed", 0)
        state = step_data.get("state", "NOT_FINISHED")
        grid_unique = step_data.get("grid_unique", [])
        step_action = step_data.get("action", 0)
        step_data_coords = step_data.get("data", None)
        event_frames = step_data.get("event") or []
        event_source_frame_count = int(
            step_data.get("event_source_frame_count") or len(event_frames)
        )
        event_frame_indices = _normalize_event_frame_indices(
            step_data.get("event_frame_indices"),
            len(event_frames),
        )
        event_decimation = step_data.get("event_decimation") or ""
        event_reduced_frame_position = _normalize_event_reduced_frame_position(
            step_data.get("event_reduced_frame_position"),
            frame_count=len(event_frames),
        )
        event_reduced_frame_source_index = (
            event_frame_indices[event_reduced_frame_position]
            if event_reduced_frame_position is not None
            else None
        )
        final_state_data = step_data.get("final_state")
        final_event_frames = step_data.get("final_event")

        reward = 0.0
        r_level = 0.0
        level_up = levels_completed > pre_action_level
        if level_up:
            r_level = self.level_factor / max(self.actions_this_level, 1)
            reward = r_level
            self.current_level = levels_completed
            transition_actions_this_level = actions_before + 1
            self.actions_this_level = 0
        else:
            self.current_level = levels_completed
        done = state in ("WIN", "GAME_OVER")

        # Synthesize grid: apply stored snapshot if at matching step, else
        # create a grid matching the stored unique colors
        grid = self._synthesize_grid(grid_unique)

        # Compute diff between consecutive grids
        if self._prev_grid is not None:
            self.diff_grid = DiffGrid.from_grids(self._prev_grid, grid)
        else:
            self.diff_grid = DiffGrid.no_change(grid)

        colors_delta = _unique_colors_delta(self._prev_grid, grid)

        self._prev_grid = grid
        self.episode_history.append(grid)

        info = StepInfo(
            state=state,
            step_id=step_id,
            global_step=self.global_step,
            episode_step=self.episode_step,
            levels_completed=levels_completed,
            actions_this_level=self.actions_this_level,
            current_level=self.current_level,
            action_space=self._available_actions_ints,
            available_actions=self._available_actions_ints,
            diff_grid=self.diff_grid,
            diff_pixels=self.diff_grid.num_changed if self.diff_grid.num_changed > 0 else diff_pixels,
            event=event_frames,
            event_frame_count=len(event_frames),
            event_source_frame_count=event_source_frame_count,
            event_frame_indices=event_frame_indices,
            event_decimation=event_decimation,
            event_reduced_frame_position=event_reduced_frame_position,
            event_reduced_frame_source_index=event_reduced_frame_source_index,
            unique_colors_delta=colors_delta,
            final_state=(
                Grid(data=final_state_data, step=self.episode_step, level=pre_action_level)
                if final_state_data is not None else None
            ),
            final_event=(
                [Grid(data=frame_data, step=self.episode_step, level=pre_action_level)
                 for frame_data in final_event_frames]
                if final_event_frames is not None else ([] if level_up else None)
            ),
            grid_unique=grid_unique,
            replay_action=step_action,
            replay_data=step_data_coords,
            reward_components={
                "level": r_level,
                "gameover": -1.0 if state == "GAME_OVER" else 0.0,
            },
        )
        self._record_transition(
            step_id=step_id,
            action_name=ARCAGI3Env.action_to_name(step_action, step_data_coords),
            action_data=step_data_coords,
            grid=grid,
            reward=reward,
            done=done,
            info=info,
            level=pre_action_level,
            level_completed=level_up,
            diff_pixels=None if level_up else info.diff_pixels,
            actions_this_level=transition_actions_this_level,
            unique_colors_delta=None if level_up else colors_delta,
            event=info.event,
            event_frame_count=info.event_frame_count,
            event_source_frame_count=info.event_source_frame_count,
            event_frame_indices=info.event_frame_indices,
            event_decimation=info.event_decimation,
            event_reduced_frame_position=info.event_reduced_frame_position,
            event_reduced_frame_source_index=info.event_reduced_frame_source_index,
            final_state=info.final_state,
            final_event=info.final_event,
        )

        return grid, reward, done, info

    @property
    def action_space_ints(self) -> list[int]:
        """Available actions as integer values."""
        return list(self._available_actions_ints)

    @property
    def action_names(self) -> list[str]:
        """Available actions as text names."""
        return [ARCAGI3Env.ACTION_NAMES[i] for i in self._available_actions_ints]

    def _env_has_keyboard_controls(self) -> bool:
        return _env_has_keyboard_controls(self)

    def _resolve_add_grid_orientation(self, value) -> bool:
        return _resolve_add_grid_orientation(value, self)

    def compute_rhae_scores(self) -> tuple[float, list[float]]:
        """Compute RHAE game score and per-level scores from current state."""
        level_actions = self.sdk_level_actions if self.sdk_level_actions else getattr(self, '_local_level_actions', [])
        return _rhae_game_score(
            self.baseline_actions, level_actions, self.current_level,
        )

    def render_episode(self) -> str:
        """Render a text summary of the episode history."""
        if not self.episode_history:
            return "No episode history recorded."
        lines = [
            f"Replay Episode {self.episode} — {len(self.episode_history)} frames, "
            f"game={self.game_title}"
        ]
        for i, grid in enumerate(self.episode_history):
            colors = grid.unique_colors()
            lines.append(
                f"  [{i:3d}] step={grid.step} level={grid.level} "
                f"colors={colors}"
            )
            if i > 0:
                diff = grid.diff_count(self.episode_history[i - 1])
                lines.append(f"        diff_from_prev={diff} pixels")
        return "\n".join(lines)

    def get_trajectory_steps(self) -> list[dict]:
        """Return the full stored trajectory for inspection."""
        return list(self._steps)

    def _synthesize_grid(self, grid_unique: list[int]) -> Grid:
        """Synthesize a Grid from stored data.

        Uses actual .npy snapshots when the cursor matches a stored snapshot,
        otherwise builds a representative grid from the unique color list.
        """
        # Check for exact snapshot match
        snapshot_map = {"reset": 0, "step5": 5, "step20": 20}
        for tag, step_num in snapshot_map.items():
            if self.episode_step == step_num and tag in self._grid_snapshots:
                return Grid(
                    data=ARCAGI3Env._frame_to_int_grid(self._grid_snapshots[tag]),
                    step=self.episode_step,
                    level=self.current_level,
                )

        # Build a synthetic grid using unique colors
        if grid_unique:
            bg = grid_unique[0]
        else:
            bg = 0

        data = [[bg] * 64 for _ in range(64)]

        # Place non-background colors in distinct regions
        non_bg = [c for c in grid_unique if c != bg]
        if non_bg:
            region_size = max(1, 60 // len(non_bg))
            for i, color in enumerate(non_bg):
                y_start = 2 + i * region_size
                y_end = min(y_start + region_size - 1, 63)
                for y in range(y_start, y_end + 1):
                    for x in range(2, min(2 + region_size, 63)):
                        data[y][x] = color

        return Grid(data=data, step=self.episode_step, level=self.current_level)
