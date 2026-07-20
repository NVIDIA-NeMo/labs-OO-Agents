# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI-3 environment harness — the game side of the file-IPC loop.

Runs in this venv (install the arcade SDK via the `arc` extra):

    python examples/arc_agi_3/harness.py \
        --run-dir <run_dir> --game ls20

Protocol (files under ``<run_dir>/ipc/``):
- waits for ``agent_ready`` (the TUI agent's tail producer is attached),
- resets the env and appends state lines to ``states.jsonl``,
- polls ``actions.jsonl`` for the agent's next action batch, executes up to
  ``--max-actions-per-turn`` of them (stopping early on level completion /
  WIN / GAME_OVER), records everything viewer-compatibly (recorder.py),
  appends the new state, and repeats until WIN or a cap is hit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

from recorder import RunRecorder, build_event_summary  # noqa: E402

GRID_HEX = "0123456789abcdef"


def _hx(v: int) -> str:
    """Hex digit for a grid value, or '?' if out of the 0-15 palette range."""
    return GRID_HEX[v] if 0 <= v <= 15 else "?"


def grid_rows(grid_data: list[list[int]]) -> list[str]:
    return ["".join(GRID_HEX[v] if 0 <= v <= 15 else "?" for v in row) for row in grid_data]


def diff_summary(prev: list[list[int]] | None, curr: list[list[int]]) -> str:
    if prev is None:
        return "first state"
    changed = [
        (r, c, prev[r][c], curr[r][c])
        for r in range(len(curr))
        for c in range(len(curr[0]))
        if prev[r][c] != curr[r][c]
    ]
    if not changed:
        return "no change"
    rows = [r for r, *_ in changed]
    cols = [c for _, c, *_ in changed]
    head = ", ".join(f"({r},{c}):{_hx(o)}->{_hx(n)}" for r, c, o, n in changed[:12])
    more = f" (+{len(changed) - 12} more)" if len(changed) > 12 else ""
    return (
        f"{len(changed)} cells changed in rows {min(rows)}-{max(rows)}, "
        f"cols {min(cols)}-{max(cols)}: {head}{more}"
    )


def state_name(info_state) -> str:
    s = str(info_state)
    return s.split(".")[-1] if "." in s else s


def _opt_int(x: str) -> int | None:
    """int, or None for 'none'/'unlimited'/negative (an uncapped limit)."""
    if str(x).strip().lower() in ("none", "", "unlimited") or str(x).strip() == "-1":
        return None
    v = int(x)
    return None if v < 0 else v


def _opt_float(x: str) -> float | None:
    """float, or None for 'none'/'unlimited'/negative (disables the checkpoint)."""
    if str(x).strip().lower() in ("none", "", "unlimited") or str(x).strip() == "-1":
        return None
    v = float(x)
    return None if v < 0 else v


def _default_action(available: list[str], last_action: str | None) -> str:
    """Fallback action when the agent stays silent past the ladder, so the harness
    can force-advance the game instead of terminating it. Priority:
      1. UNDO                    (if available — least-harmful, reverts the last move)
      2. the last executed action (agent already vetted it)
      3. a random non-RESET action (only reachable before the agent's first action,
         when UNDO is absent). CLICK gets center coords since it needs x/y.
    """
    if "UNDO" in available:
        return "UNDO"
    if last_action:
        return last_action
    choices = [a for a in available if a not in ("RESET", "UNDO")]
    if choices:
        a = random.choice(choices)
        return "CLICK 32 32" if a == "CLICK" else a
    return available[0] if available else "RESET"


def apply_action_cap(actions: list[str], cap: int) -> list[str]:
    """Truncate a submitted batch to the per-turn cap; ``cap <= 0`` = unlimited."""
    return actions if cap <= 0 else actions[:cap]


def truncation_note(*, executed: int, requested: int) -> str:
    """State note telling the agent its batch tail was dropped ('' if it wasn't).

    The agent plans against action_results; a silently missing tail would read
    as 'my actions did nothing' — say explicitly what never ran.
    """
    if requested <= executed:
        return ""
    return (
        f"only the first {executed} of your {requested} submitted actions were executed; "
        f"actions {executed + 1}..{requested} were NOT executed (per-turn action cap)."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--game", required=True)
    p.add_argument(
        "--alias",
        default="the game",
        help="opaque handle written into agent-visible state notes "
        "(the real --game id is never surfaced to the agent)",
    )
    p.add_argument("--variant", default="")
    p.add_argument("--llm-uri", default="")
    p.add_argument("--operation-mode", default="offline")
    p.add_argument(
        "--max-turns",
        type=_opt_int,
        default=None,
        help="max agent turns (action batches); None/unlimited by default",
    )
    p.add_argument(
        "--max-env-steps",
        type=int,
        default=5000,
        help="max total environment actions (the main run bound)",
    )
    p.add_argument(
        "--max-actions-per-turn",
        type=int,
        default=20,
        help="cap on actions executed per agent turn; <=0 = unlimited",
    )
    p.add_argument(
        "--allowed-game-overs",
        type=int,
        default=-1,
        help="auto-RESET budget after GAME_OVER (-1 = unlimited, "
        "0 = never auto-reset; exhausted budget ends the run)",
    )
    p.add_argument("--agent-ready-timeout", type=float, default=300.0)
    p.add_argument(
        "--agent-turn-timeout",
        type=float,
        default=1200.0,
        help="seconds of agent silence before giving up (agent_timeout) "
        "— the outer/kill rung of the timeout ladder",
    )
    p.add_argument(
        "--nudge-after",
        type=_opt_float,
        default=900.0,
        help="seconds of agent silence before the FIRST (gentle) reminder "
        "state. At agent-turn-timeout a second URGENT nudge follows; "
        "the game is never terminated on silence — see --fallback-window",
    )
    p.add_argument(
        "--fallback-window",
        type=float,
        default=90.0,
        help="after the URGENT nudge at agent-turn-timeout, seconds to wait "
        "before the harness FORCE-ADVANCES the game with a default "
        "action (UNDO > last action > random non-RESET) instead of "
        "terminating. Kept short (~one quick response cycle): a "
        "responsive agent submits fast, a stuck/dead one never will, "
        "so a long wait only prolongs the stall. Bounded by "
        "max-env-steps, never a deliberate kill",
    )
    p.add_argument(
        "--scorecard-id",
        default="",
        help="competition shared scorecard id (from scorecard_broker.py) "
        "— all games in a fleet share one card; empty lets the SDK "
        "auto-mint a per-game card (offline ignores this)",
    )
    p.add_argument(
        "--event-prompt",
        choices=["off", "count"],
        default="count",
        help="animation-event summary in the agent's state: 'count' adds "
        "per-action frame counts + a pointer to trajectory() on turns "
        "that animated; 'off' omits it. The full frames are always in "
        "trajectory(include_events=True) regardless",
    )
    p.add_argument(
        "--visual",
        choices=["off", "only", "additive"],
        default="off",
        help="grid-as-image mode (the agent shows the PNG; 'only' also "
        "DROPS the hex grid_rows from the state so the image REPLACES "
        "the grid, 'additive' keeps both). Grid stays in events.jsonl "
        "either way (trajectory recovers exact hex)",
    )
    return p.parse_args()


def wait_for_file(path: Path, timeout: float, what: str) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out after {timeout}s waiting for {what} ({path})")
        time.sleep(0.5)


class ActionReader:
    """Tail actions.jsonl: yields one parsed action batch per line."""

    def __init__(self, path: Path):
        self.path = path
        self._pos = 0

    def next_batch(self, timeout: float) -> dict | None:
        t0 = time.time()
        while True:
            with self.path.open() as f:
                f.seek(self._pos)
                line = f.readline()
                if line and line.endswith("\n"):
                    self._pos = f.tell()
                    line = line.strip()
                    if line:
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[harness] skipping malformed action line: {line!r}")
                            continue
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.5)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    ipc = run_dir / "ipc"
    ipc.mkdir(parents=True, exist_ok=True)
    states_path = ipc / "states.jsonl"
    actions_path = ipc / "actions.jsonl"
    states_path.touch()
    actions_path.touch()

    from arc_agi_3.environment import ARCAGI3Env

    recorder = RunRecorder(
        run_dir, game_id=args.game, alias=args.alias, llm_uri=args.llm_uri, variant=args.variant
    )

    # Online/competition/normal use full backend ids (e.g. "ls20-9607627b"); the
    # SDK's make() does NOT resolve short codes. Offline keys envs by the short
    # code directly, so leave it untouched (regression-safe).
    env_game_id = args.game
    if args.operation_mode != "offline":
        try:
            env_game_id = ARCAGI3Env.resolve_game_id(args.game, args.operation_mode)
            print(f"[harness] resolved game {args.game} -> {env_game_id} ({args.operation_mode})")
        except Exception as e:
            print(f"[harness] game-id resolution failed for {args.game!r}: {e}")

    scorecard_id = args.scorecard_id or None
    print(
        f"[harness] creating env game={args.game} mode={args.operation_mode} "
        f"scorecard={scorecard_id or '(auto)'}"
    )
    env = ARCAGI3Env(
        game_id=env_game_id,
        operation_mode=args.operation_mode,
        results_dir=str(run_dir),
        scorecard_id=scorecard_id,
    )

    print(f"[harness] waiting for agent_ready (timeout {args.agent_ready_timeout}s)")
    wait_for_file(ipc / "agent_ready", args.agent_ready_timeout, "TUI agent")

    grid = env.reset()
    env_steps = 0
    turn = 0
    prev_grid: list[list[int]] | None = None

    # Per-level RHAE for the viewer. baseline_actions = the SDK "par" per level for
    # the REAL game; the viewer needs it (+ per-level ai_actions) to show RHAE_L/U.
    # This lives ONLY in events.jsonl / result.json (harness side) — it never
    # reaches the agent (the open()-jail excludes agent_logs and trajectory() only
    # surfaces env_step fields), so it is not an identity/difficulty leak.
    try:
        from arc_agi_3.rhae import get_baseline_actions_for_game, rhae_level_score

        baseline_actions = get_baseline_actions_for_game(args.game, args.operation_mode)
        if not baseline_actions:
            # get_baseline_actions_for_game swallows lookup errors and returns [] —
            # typically the SDK found no environment_files/ under the harness cwd.
            # Without baselines every RHAE figure (live viewer AND result.json) is
            # 0; surface the cause instead of failing silently.
            print(
                f"[harness] WARNING: no RHAE baselines for {args.game} "
                f"({args.operation_mode}) — check environment_files/ under the "
                "harness cwd (run_solver: ARC_DATA_DIR). RHAE will report 0."
            )
    except Exception as e:
        print(f"[harness] baseline_actions unavailable ({e}); RHAE will be limited")
        baseline_actions = []

        def rhae_level_score(_b, _a):
            return 0.0

    level_action_count = 0  # actions spent on the current (in-progress) level

    recorder.solver_start(
        max_steps=args.max_env_steps,
        available_actions=list(env.action_names),
        baseline_actions=baseline_actions,
        config={"max_turns": args.max_turns, "max_actions_per_turn": args.max_actions_per_turn},
    )
    recorder.env_step(
        step=0, grid=grid.data, action_name="RESET", levels_completed=0, diff_pixels=0
    )
    _record_step_obs(
        recorder,
        env,
        step=0,
        grid=grid.data,
        level=grid.level or 0,
        action="",
        state="NOT_FINISHED",
        done=False,
        diff_pixels=0,
        level_completed=False,
        episode_step=0,
    )

    def write_state(
        *, state: str, level: int, levels_completed: int, action_results: list[dict], note: str = ""
    ) -> None:
        entry = {
            "turn": turn,
            "step": env_steps,
            "level": level,
            "state": state,
            "levels_completed": levels_completed,
            "available_actions": list(env.action_names),
            "action_results": action_results,
            "diff_summary": diff_summary(prev_grid, grid.data),
        }
        if args.visual == "only":
            # image REPLACES the grid: drop the hex rows, point at trajectory for hex.
            entry["grid_note"] = (
                "grid shown as an image (visual=only); call "
                "self.trajectory(last_n=1, include_grids=True) for "
                "exact hex if you need programmatic values."
            )
        else:
            entry["grid_rows"] = grid_rows(grid.data)
        if note:
            entry["note"] = note
        # Animation summary (count only) — present only on turns that animated.
        summary = build_event_summary(action_results, args.event_prompt)
        if summary:
            entry["event_summary"] = summary
        with states_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    write_state(
        state="NOT_FINISHED",
        level=grid.level or 0,
        levels_completed=0,
        action_results=[],
        note=f"game start: {args.alias}. Discover the rules, then solve all levels.",
    )

    reader = ActionReader(actions_path)
    levels_completed = 0
    termination = "unknown"
    final_state = "NOT_FINISHED"
    game_overs = 0
    last_action: str | None = None  # last agent-executed action, for force-advance
    forced_advances = 0

    def auto_reset(after_action: str) -> str:
        """GAME_OVER recovery: harness executes RESET as a
        recorded step. Returns a note fragment describing what happened."""
        nonlocal grid, env_steps, levels_completed, final_state, game_overs
        budget = args.allowed_game_overs
        if budget == 0 or (budget > 0 and game_overs >= budget):
            return (
                f"GAME_OVER after {after_action!r}; auto-reset budget exhausted "
                f"({game_overs}/{budget})."
            )
        game_overs += 1
        death_step = env_steps
        try:
            grid, r_reward, r_done, r_info = env.step("RESET")
        except Exception as e:
            return f"GAME_OVER after {after_action!r}; harness RESET raised {type(e).__name__}: {e}"
        env_steps += 1
        levels_completed = r_info.levels_completed
        final_state = state_name(r_info.state)
        diff_px = r_info.diff_pixels if isinstance(r_info.diff_pixels, int) else 0
        recorder.env_step(
            step=env_steps,
            grid=grid.data,
            action_name="RESET",
            levels_completed=levels_completed,
            diff_pixels=diff_px,
        )
        _record_step_obs(
            recorder,
            env,
            step=env_steps,
            grid=grid.data,
            level=grid.level or levels_completed,
            action="RESET",
            state=final_state,
            done=r_done,
            diff_pixels=diff_px,
            level_completed=False,
            episode_step=env_steps,
        )
        budget_str = "∞" if budget < 0 else str(budget)
        print(
            f"[harness] auto-RESET after GAME_OVER at step {death_step} "
            f"({game_overs}/{budget_str}) -> state={final_state} levels={levels_completed}"
        )
        if final_state == "GAME_OVER":
            return (
                f"GAME_OVER after {after_action!r}; harness RESET did NOT recover "
                f"(still GAME_OVER)."
            )
        return (
            f"GAME_OVER after {after_action!r} at step {death_step} (death state is in "
            f"trajectory) — harness auto-RESET applied "
            f"(game_overs {game_overs}/{budget_str}); the grid you see is post-reset."
        )

    while True:
        if args.max_turns is not None and turn >= args.max_turns:
            termination = "max_turns"
            break
        if env_steps >= args.max_env_steps:
            termination = "max_env_steps"
            break

        # Silence ladder (time since the agent's last action batch). We NEVER
        # terminate the game on silence — we escalate, then force-advance:
        #   0..nudge_after (900)      : quiet (the agent downshifts effort at 600s)
        #   nudge_after (900)         : gentle reminder to submit
        #   agent-turn-timeout (1200) : URGENT nudge — respond now with ONLY the action
        #   + fallback_window (90)    : still silent -> force-advance with a default
        #                               action (UNDO > last > random non-RESET)
        kill_after = args.agent_turn_timeout
        nudge_after = args.nudge_after
        fw = args.fallback_window
        forced = False

        first_wait = nudge_after if (nudge_after and nudge_after < kill_after) else kill_after
        batch = reader.next_batch(first_wait)
        if batch is None and first_wait < kill_after:
            print(f"[harness] no actions for {first_wait:.0f}s — nudging agent")
            turn += 1
            write_state(
                state=final_state,
                level=grid.level or levels_completed,
                levels_completed=levels_completed,
                action_results=[],
                note=f"reminder: no actions received for {first_wait:.0f}s — "
                "the game is waiting. Submit actions for THIS turn.",
            )
            batch = reader.next_batch(kill_after - first_wait)
        if batch is None:
            # agent-turn-timeout: URGENT, minimal-response nudge before force-advancing.
            print(
                f"[harness] no actions for {kill_after:.0f}s — URGENT nudge "
                f"(force-advance in {fw:.0f}s)"
            )
            turn += 1
            write_state(
                state=final_state,
                level=grid.level or levels_completed,
                levels_completed=levels_completed,
                action_results=[],
                note=f"URGENT: {kill_after:.0f}s with no action. Respond THIS "
                "turn with ONLY self.submit_actions([...]) then "
                "return_result(RespondReason.WAIT) — no analysis, no other "
                "code. If you stay silent the harness will apply a default "
                "action to keep the game going.",
            )
            batch = reader.next_batch(fw)
        if batch is None:
            # Never terminate: force-advance with a default action so the game keeps
            # progressing (bounded by max-env-steps, not a deliberate kill).
            fa = _default_action(list(env.action_names), last_action)
            forced_advances += 1
            print(
                f"[harness] agent silent {kill_after + fw:.0f}s — force-advancing "
                f"with default action {fa!r} (#{forced_advances})"
            )
            batch = {"actions": [fa], "rationale": f"harness force-advance (agent silent): {fa}"}
            forced = True
        actions = [a for a in batch.get("actions", []) if isinstance(a, str)]
        requested_actions = len(actions)
        actions = apply_action_cap(actions, args.max_actions_per_turn)
        # A dropped tail must be LOUD on the next state: the agent may have
        # truncated at submit time (truncated_from in the batch entry) or the
        # backstop cap above may have cut a directly-written oversized batch.
        try:
            requested_actions = max(requested_actions, int(batch.get("truncated_from") or 0))
        except (TypeError, ValueError):
            pass
        recorder.agent_turn(
            turn=turn,
            step=env_steps,
            actions=actions,
            rationale=str(batch.get("rationale", ""))[:2000],
        )

        results: list[dict] = []
        note = truncation_note(executed=len(actions), requested=requested_actions)
        for i, action in enumerate(actions):
            prev_grid = [row[:] for row in grid.data]
            prev_levels = levels_completed
            try:
                grid, reward, done, info = env.step(action)
            except Exception as e:
                note = f"action {action!r} raised {type(e).__name__}: {e}"
                print(f"[harness] {note}")
                break
            env_steps += 1
            level_action_count += 1
            last_action = action  # remember for a future force-advance fallback
            levels_completed = info.levels_completed
            final_state = state_name(info.state)
            level_completed = levels_completed > prev_levels
            diff_px = info.diff_pixels if isinstance(info.diff_pixels, int) else 0
            # Animation frames this action produced (decimated 64x64 int grids).
            # Reused for events.jsonl (-> trajectory) + the per-action count in the
            # state's action_results. Never opened by the agent directly.
            event_frames: list = []
            try:
                _obs = env.get_observation(env_steps)
                event_frames = list(_obs.get("event") or [])
            except Exception:
                event_frames = []
            results.append(
                {
                    "action": action,
                    "reward": reward,
                    "diff_pixels": diff_px,
                    "state": final_state,
                    "level_completed": level_completed,
                    "event_frame_count": len(event_frames),
                }
            )
            recorder.env_step(
                step=env_steps,
                grid=grid.data,
                action_name=action,
                levels_completed=levels_completed,
                diff_pixels=diff_px,
                event_frames=event_frames,
            )
            if level_completed:
                # Emit per-level RHAE for the viewer (the level index just cleared
                # is prev_levels, 0-based). ai_actions = actions spent on it.
                lvl = prev_levels
                bl = baseline_actions[lvl] if lvl < len(baseline_actions) else 0
                recorder.level_score(
                    level=lvl,
                    ai_actions=level_action_count,
                    baseline_actions=bl,
                    rhae_score=rhae_level_score(bl, level_action_count),
                )
                level_action_count = 0
            _record_step_obs(
                recorder,
                env,
                step=env_steps,
                grid=grid.data,
                level=grid.level or levels_completed,
                action=action,
                state=final_state,
                done=done,
                diff_pixels=diff_px,
                level_completed=level_completed,
                episode_step=env_steps,
                reward=reward,
            )
            # Real prompts/responses are written live by ViewerMessageExporter
            # (launcher-side, from traces); the harness no longer writes synthetic
            # analyze-tab messages.
            print(
                f"[harness] turn={turn} step={env_steps} action={action} "
                f"state={final_state} levels={levels_completed} diff={diff_px}"
            )
            remaining = len(actions) - i - 1
            if final_state == "WIN":
                note = "WIN — game solved!"
                break
            if final_state == "GAME_OVER":
                note = auto_reset(action)
                if remaining:
                    note += f" {remaining} queued actions skipped."
                break
            if level_completed and remaining:
                note = (
                    f"level completed after {action!r} — {remaining} queued "
                    "actions skipped so you can re-plan on the new level."
                )
                break

        if forced:
            fnote = (
                f"harness FORCE-ADVANCED with {batch['actions'][0]!r} — you were "
                f"silent past {kill_after + fw:.0f}s. Submit an action THIS turn "
                "to take back control."
            )
            note = f"{fnote} {note}".strip()
        turn += 1
        write_state(
            state=final_state,
            level=grid.level or levels_completed,
            levels_completed=levels_completed,
            action_results=results,
            note=note,
        )

        if final_state == "WIN":
            termination = "win"
            break
        if final_state == "GAME_OVER":
            # only reachable when auto-reset was exhausted/failed — end the run.
            termination = "game_over"
            break

    extras: dict = {"turns": turn, "game_overs": game_overs}
    try:
        game_score, level_scores = env.compute_rhae_scores()
        extras.update({"rhae_game_score": game_score, "rhae_level_scores": level_scores})
    except Exception:
        pass
    recorder.result(
        levels_completed=levels_completed,
        total_steps=env_steps,
        termination_reason=termination,
        extras=extras,
    )
    if termination != "win":
        write_state(
            state=final_state,
            level=grid.level or levels_completed,
            levels_completed=levels_completed,
            action_results=[],
            note=f"harness stopped: {termination}. No further actions will be executed.",
        )
    try:
        env.close()
    except Exception:
        pass
    print(f"[harness] done: {termination} levels={levels_completed} steps={env_steps}")
    return 0 if termination in ("win",) else 1


def _record_step_obs(
    recorder: RunRecorder,
    env,
    *,
    step: int,
    grid,
    level: int,
    action: str,
    state: str,
    done: bool,
    diff_pixels: int,
    level_completed: bool,
    episode_step: int,
    reward: float = 0.0,
) -> None:
    """Prefer the env's own observation record; fall back to a minimal one."""
    try:
        obs = env.get_observation(step)
        if not isinstance(obs, dict) or obs.get("grid") is None:
            raise ValueError("unusable observation")
    except Exception:
        obs = recorder.fallback_observation(
            step=step,
            grid=grid,
            level=level,
            producer_action=action,
            state=state,
            done=done,
            diff_pixels=diff_pixels,
            level_completed=level_completed,
            episode_step=episode_step,
        )
    # The viewer's single-step detail reads `action_chosen`/`action` (not
    # producer_action) and `reward` — inject them so Step Details renders.
    obs.setdefault("action_chosen", action)
    obs.setdefault("reward", reward)
    recorder.step_json(step, obs)


if __name__ == "__main__":
    sys.exit(main())
