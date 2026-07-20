# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewer-compatible run output writer.

Writes the files the arc_league viewer / playground server needs to discover and
display a run (see PLAN.md "Viewer compatibility"):

- ``team_<name>/`` marker dir (doubles as the agent workspace)
- ``agent_logs/<team>/team_leader/events.jsonl`` — ``solver_start`` + ``env_step``
  events carrying ``grid_data`` (the viewer's primary grid/timeline source)
- ``steps/step_%04d.json`` — per-step observations (grid fallback; from
  ``env.get_observation(step)`` when available)
- ``result.json`` — final summary (``gameplay.json`` is written by ARCAGI3Env
  itself via ``results_dir``)

Stdlib-only: imported by harness.py.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path


def _now() -> tuple[str, float]:
    return datetime.now(UTC).isoformat(), time.time()


def build_event_summary(action_results: list[dict], mode: str = "count") -> dict | None:
    """Per-turn animation summary for the agent's state, from the turn's
    ``action_results`` (each carries ``event_frame_count``). COUNT ONLY: returns
    ``{frames_by_action, total_frames, note}`` — or ``None`` when the mode is
    ``off`` or no action in the turn animated (``total_frames == 0``). Pure/stdlib
    so both the harness and the tests use the exact same logic."""
    if mode == "off":
        return None
    frames_by_action = [int(r.get("event_frame_count", 0) or 0) for r in action_results]
    total = sum(frames_by_action)
    if total <= 0:
        return None
    return {
        "frames_by_action": frames_by_action,
        "total_frames": total,
        "note": "call trajectory(include_events=True) for the frames",
    }


class RunRecorder:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        game_id: str,
        alias: str = "",
        team: str = "nemo",
        llm_uri: str = "",
        variant: str = "",
    ):
        self.run_dir = Path(run_dir)
        # game_id (real) goes only into result.json (not agent-visible). The
        # agent-readable solver_start event carries `alias` instead.
        self.game_id = game_id
        self.alias = alias or game_id
        self.team = team
        self.llm_uri = llm_uri
        self.variant = variant
        self.steps_dir = self.run_dir / "steps"
        self.events_path = self.run_dir / "agent_logs" / team / "team_leader" / "events.jsonl"
        # team_<name>/ is the discovery marker (_detect_run_type) and workspace.
        (self.run_dir / f"team_{team}" / "shared").mkdir(parents=True, exist_ok=True)
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    # ------------------------------------------------------------- events

    def _event(self, event: str, **kwargs) -> None:
        ts, unix = _now()
        line = {
            "timestamp": ts,
            "unix_time_s": unix,
            "event": event,
            "agent_id": "team_leader",
            **kwargs,
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def solver_start(
        self,
        *,
        max_steps: int,
        available_actions: list[str],
        baseline_actions: list[int] | None = None,
        config: dict | None = None,
    ) -> None:
        self._event(
            "solver_start",
            game_id=self.alias,  # opaque handle — real id stays in result.json
            llm_uri=self.llm_uri,
            max_steps=max_steps,
            available_actions=available_actions,
            # SDK "par" actions per level — the viewer needs it (+ per-level
            # ai_actions) to compute RHAE_L/U. Viewer-only: not on the agent surface.
            baseline_actions=list(baseline_actions or []),
            depth=1,
            config={"solver": "nemo_single_agent", "variant": self.variant, **(config or {})},
        )

    def level_score(
        self, *, level: int, ai_actions: int, baseline_actions: int, rhae_score: float
    ) -> None:
        """Per-level RHAE for the viewer (dispatched by viewer.update_state as
        ``level_score``): populates rhae_level_scores + baseline_actions so the
        dashboard/status show RHAE_L/U. Viewer-only — never reaches the agent."""
        self._event(
            "level_score",
            level=level,
            ai_actions=ai_actions,
            baseline_actions=baseline_actions,
            rhae_score=rhae_score,
            rhae_pct=round(rhae_score * 100, 2),
        )

    def env_step(
        self,
        *,
        step: int,
        grid: list[list[int]],
        action_name: str,
        levels_completed: int,
        diff_pixels: int,
        event_frames: list | None = None,
    ) -> None:
        frames = list(event_frames or [])
        extra: dict = {}
        # Store the decimated animation frames (64x64 int grids) only when the
        # action animated — keeps static steps cheap. Surfaced to the agent via
        # trajectory(include_events=True); never on the agent's live state. Keyed
        # `event_frames` (not `event`, which is the event-type field).
        if frames:
            extra["event_frames"] = frames
        self._event(
            "env_step",
            step=step,
            levels_completed=levels_completed,
            diff_pixels=diff_pixels,
            event_frame_count=len(frames),
            action_name=action_name,
            grid_data=grid,
            **extra,
        )

    def agent_turn(self, *, turn: int, step: int, actions: list[str], rationale: str) -> None:
        """Non-viewer extra: record the agent's submitted plan for analysis."""
        self._event("agent_turn", step=step, turn=turn, actions=actions, rationale=rationale)

    # ---------------------------------------------------- analyze messages

    def step_messages(
        self,
        *,
        env_step: int,
        turn: int,
        action: str,
        batch: list[str],
        batch_index: int,
        rationale: str,
        result: dict | None,
        state_header: str,
    ) -> None:
        """Per-env-step user/assistant markdown for the viewer's analyze tab.

        The analyze view resolves reasoning by exact step number
        (messages/step_{NNN}_round_00_{user,assistant}.md), so every env step of
        a multi-action turn gets a dump carrying the turn's rationale."""
        msgs = self.events_path.parent / "messages"
        msgs.mkdir(exist_ok=True)
        user = (
            f"# Turn {turn} — env step {env_step}\n\n"
            f"{state_header}\n\n"
            f"Executing action {batch_index + 1}/{len(batch)} of this turn's batch.\n"
        )
        outcome = ""
        if result is not None:
            outcome = (
                f"\n## Outcome of `{action}`\n\n"
                f"- diff_pixels: {result.get('diff_pixels')}\n"
                f"- state: {result.get('state')}\n"
                f"- level_completed: {result.get('level_completed')}\n"
                f"- reward: {result.get('reward')}\n"
            )
        assistant = (
            f"# Agent turn {turn}\n\n"
            f"## Rationale / prediction\n\n{rationale.strip() or '(none)'}\n\n"
            f"## Action batch\n\n"
            + "\n".join(
                f"{'->' if i == batch_index else '  '} {i + 1}. `{a}`" for i, a in enumerate(batch)
            )
            + "\n"
            + outcome
        )
        (msgs / f"step_{env_step:03d}_round_00_user.md").write_text(user)
        (msgs / f"step_{env_step:03d}_round_00_assistant.md").write_text(assistant)

    # -------------------------------------------------------------- steps/

    def step_json(self, step: int, obs: dict) -> None:
        (self.steps_dir / f"step_{step:04d}.json").write_text(json.dumps(obs))

    def fallback_observation(
        self,
        *,
        step: int,
        grid: list[list[int]],
        level: int,
        producer_action: str,
        state: str,
        done: bool,
        diff_pixels: int,
        level_completed: bool,
        episode_step: int,
    ) -> dict:
        """Minimal record satisfying the viewer's parse_step_record contract."""
        return {
            "step": step,
            "grid": grid,
            "producer_action": producer_action if step > 0 else "",
            "producer_transition_step": step - 1 if step > 0 else None,
            "level": level,
            "level_completed": level_completed,
            "diff_pixels": diff_pixels,
            "actions_this_level": 0,
            "unique_colors_delta": {},
            "done": done,
            "state": state,
            "event": [],
            "event_frame_count": 0,
            "event_source_frame_count": 0,
            "event_frame_indices": [],
            "event_decimation": None,
            "event_reduced_frame_position": None,
            "event_reduced_frame_source_index": None,
            "final_state": None,
            "final_event": None,
            "episode_step": episode_step,
        }

    # ------------------------------------------------------------- result

    def result(
        self,
        *,
        levels_completed: int,
        total_steps: int,
        termination_reason: str,
        extras: dict | None = None,
    ) -> None:
        payload = {
            "game_id": self.game_id,
            "operation_mode": "offline",
            "seed": 0,
            "llm_uri": self.llm_uri,
            "solver": "nemo_single_agent",
            "variant": self.variant,
            "levels_completed": levels_completed,
            "total_steps": total_steps,
            "wall_time_seconds": round(time.time() - self._t0, 3),
            "termination_reason": termination_reason,
            **(extras or {}),
        }
        (self.run_dir / "result.json").write_text(json.dumps(payload, indent=2))
