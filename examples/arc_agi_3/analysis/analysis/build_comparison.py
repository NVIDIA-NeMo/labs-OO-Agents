#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two ARC-AGI-3 competition fleets OVER TIME:

  A (baseline)     20260710_154254_competition_memory_visual        (grid-game-solver, FINISHED)
  B (world-model)  20260711_193827_competition_memory_visual_wm     (interactive-game-solver, RUNNING)

Two families of curves, both on a shared run-relative wall-clock axis:
  * RHAE_L / RHAE_U   — fleet-mean live RHAE bounds (rhae.rhae_bounds), the same
                        lower/upper the dashboard shows (winning levels collapse
                        upper->lower automatically).
  * Budget            — fleet cumulative $ (and input/output/cached split) from every
                        game's llm_call token counts at the gpt-5.5 price.

Because A finished and B is still running, A is TRUNCATED to B's elapsed wall-clock
window so the comparison is apples-to-apples (same number of minutes of fleet time).

Run (needs matplotlib -> the progressive-learning venv):
    progressive-learning/.venv/bin/python \
        results/arc_agi_3/nemo_solver/20260711_193827_competition_memory_visual_wm/analysis/build_comparison.py

Outputs (into this analysis/ dir): rhae_over_time.png, budget_over_time.png, comparison_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # examples/arc_agi_3/analysis/analysis -> repo
# arc_agi_3.rhae lives in the progressive-learning checkout; allow an override.
_PL = os.environ.get("ARC_PL_DIR") or str(REPO / "progressive-learning")
sys.path.insert(0, _PL)

_ap = argparse.ArgumentParser(
    description="Over-time RHAE + budget comparison of two competition runs."
)
_ap.add_argument("run_b", help="the run to analyze (e.g. a world-model run)")
_ap.add_argument(
    "--baseline",
    default=None,
    help="baseline run to compare against; omit to plot run_b alone",
)
_ap.add_argument("--out", default=None, help="output dir (default: <run_b>/analysis)")
_ap.add_argument("--label-a", default="baseline")
_ap.add_argument("--label-b", default="this run")
_args = _ap.parse_args()

import matplotlib  # noqa: E402
from arc_agi_3.rhae import rhae_bounds, rhae_level_score  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_B = Path(_args.run_b).resolve()
RUN_A = Path(_args.baseline).resolve() if _args.baseline else None
ANALYSIS_DIR = Path(_args.out).resolve() if _args.out else (RUN_B / "analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
LABEL_A, LABEL_B = _args.label_a, _args.label_b

# gpt-5.5 canonical pricing, $/Mtok (progressive-learning/arc_agi_3/llm_configs.py)
PRICE = {"input": 5.0, "output": 30.0, "cached": 0.50}
GRID_S = 20.0  # time-grid resolution (seconds)


def _events(path: str, cutoff_abs: float | None):
    """Chronological events for one game; stop early once past cutoff_abs (unix s)."""
    out = []
    for line in open(path, errors="ignore"):
        if '"unix_time_s"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("unix_time_s")
        if t is None:
            continue
        if cutoff_abs is not None and t > cutoff_abs:
            break  # events.jsonl is append-order chronological
        out.append(d)
    return out


def _game_event_files(run: Path):
    return sorted(glob.glob(f"{run}/*/memory/*/agent_logs/nemo/team_leader/events.jsonl"))


def run_start(run: Path) -> float:
    """Earliest event unix time across the whole fleet = the run clock's t0."""
    t0 = None
    for fp in _game_event_files(run):
        for line in open(fp, errors="ignore"):
            if '"unix_time_s"' in line:
                try:
                    t = json.loads(line).get("unix_time_s")
                except json.JSONDecodeError:
                    continue
                if t is not None:
                    t0 = t if t0 is None else min(t0, t)
                break
    return t0 or 0.0


def per_game_series(evs, t0: float):
    """[(t, rhae_L, rhae_U, cum_in, cum_out, cum_cached)] on the shared run clock.

    A sample is emitted after every event so both the RHAE step-curve (updated on
    env_step) and the cumulative-budget curve (updated on llm_call) are dense.
    """
    baseline: list[int] = []
    level_scores: list[float] = []
    last_level_end_step = 0
    cur_levels = 0
    cin = cout = ccached = 0
    L = U = 0.0
    series = []
    for e in evs:
        et = e.get("event")
        t = e["unix_time_s"] - t0
        if et == "solver_start":
            baseline = [int(x) for x in e.get("baseline_actions", []) or []]
        elif et == "llm_call":
            cin += e.get("input_tokens", 0) or 0
            cout += e.get("output_tokens", 0) or 0
            ccached += e.get("cache_read_tokens", 0) or 0
        elif et == "env_step":
            step = e.get("step", 0) or 0
            lc = e.get("levels_completed", 0) or 0
            while cur_levels < lc and cur_levels < len(baseline):
                acts = max(1, step - last_level_end_step)
                level_scores.append(rhae_level_score(baseline[cur_levels], acts))
                last_level_end_step = step
                cur_levels += 1
            if baseline:
                L, U = rhae_bounds(
                    baseline,
                    level_scores,
                    cur_levels,
                    max(0, step - last_level_end_step),
                    done=False,
                )
        series.append((t, L, U, cin, cout, ccached))
    return series


def fleet_grid(run: Path, t0: float, t_max: float):
    """Aggregate all games onto a common grid: RHAE = mean over started games,
    budget = sum over games (last sample <= grid time). Returns dict of arrays."""
    cutoff_abs = t0 + t_max + 1.0
    games = []
    for fp in _game_event_files(run):
        s = per_game_series(_events(fp, cutoff_abs), t0)
        if s:
            games.append(s)
    n = int(t_max // GRID_S) + 1
    grid = [i * GRID_S for i in range(n + 1)]
    out = {
        "t_min": [g / 60.0 for g in grid],
        "rhae_l": [],
        "rhae_u": [],
        "usd_in": [],
        "usd_out": [],
        "usd_cached": [],
        "usd_total": [],
    }
    # per-game cursor into its series (grid is ascending -> advance monotonically)
    cur = [0] * len(games)
    last = [None] * len(games)  # last sample seen at/-before the grid time
    for gt in grid:
        for i, s in enumerate(games):
            while cur[i] < len(s) and s[cur[i]][0] <= gt:
                last[i] = s[cur[i]]
                cur[i] += 1
        started = [last[i] for i in range(len(games)) if last[i] is not None]
        if started:
            out["rhae_l"].append(sum(x[1] for x in started) / len(started) * 100)
            out["rhae_u"].append(sum(x[2] for x in started) / len(started) * 100)
        else:
            out["rhae_l"].append(0.0)
            out["rhae_u"].append(0.0)
        cin = sum(x[3] for x in started)
        cout = sum(x[4] for x in started)
        ccached = sum(x[5] for x in started)
        uncached = max(cin - ccached, 0)
        u_in = uncached / 1e6 * PRICE["input"]
        u_ca = ccached / 1e6 * PRICE["cached"]
        u_out = cout / 1e6 * PRICE["output"]
        out["usd_in"].append(u_in)
        out["usd_cached"].append(u_ca)
        out["usd_out"].append(u_out)
        out["usd_total"].append(u_in + u_ca + u_out)
    return out


def _final(grid: dict) -> dict:
    return {
        "rhae_l_final": round(grid["rhae_l"][-1], 2),
        "rhae_u_final": round(grid["rhae_u"][-1], 2),
        "usd_total": round(grid["usd_total"][-1], 2),
        "usd_in": round(grid["usd_in"][-1], 2),
        "usd_out": round(grid["usd_out"][-1], 2),
        "usd_cached": round(grid["usd_cached"][-1], 2),
    }


def main() -> int:
    t0_b = run_start(RUN_B)
    # B's elapsed window = last event on the B clock.
    last_b = 0.0
    for fp in _game_event_files(RUN_B):
        for line in reversed(open(fp, errors="ignore").readlines()):
            if '"unix_time_s"' in line:
                try:
                    last_b = max(last_b, json.loads(line)["unix_time_s"] - t0_b)
                except (json.JSONDecodeError, KeyError):
                    pass
                break
    t_max = last_b
    b = fleet_grid(RUN_B, t0_b, t_max)
    a = fleet_grid(RUN_A, run_start(RUN_A), t_max) if RUN_A else None  # baseline (optional)
    mins = t_max / 60.0
    trunc = " (baseline truncated to this run's window)" if a else ""

    # ---- Figure 1: RHAE_L / RHAE_U over time ----------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    if a:
        ax.plot(a["t_min"], a["rhae_u"], color="#1f77b4", ls="--", label=f"{LABEL_A} — RHAE_U")
        ax.plot(a["t_min"], a["rhae_l"], color="#1f77b4", ls="-", label=f"{LABEL_A} — RHAE_L")
    ax.plot(b["t_min"], b["rhae_u"], color="#ff7f0e", ls="--", label=f"{LABEL_B} — RHAE_U")
    ax.plot(b["t_min"], b["rhae_l"], color="#ff7f0e", ls="-", label=f"{LABEL_B} — RHAE_L")
    ax.set_xlabel("fleet wall-clock elapsed (min)")
    ax.set_ylabel("RHAE (%)  — fleet mean")
    ax.set_title(f"RHAE_L / RHAE_U over time  ({mins:.0f} min){trunc}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "rhae_over_time.png", dpi=120)
    plt.close(fig)

    # ---- Figure 2: budget over time ($ total + in/out/cached split) -----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    if a:
        ax1.plot(a["t_min"], a["usd_total"], color="#1f77b4", label=f"{LABEL_A} — total $")
    ax1.plot(b["t_min"], b["usd_total"], color="#ff7f0e", label=f"{LABEL_B} — total $")
    ax1.set_ylabel("cumulative spend ($)")
    ax1.set_title(f"Budget over time  ({mins:.0f} min, gpt-5.5 pricing){trunc}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    runs = [(b, "#ff7f0e", LABEL_B)] + ([(a, "#1f77b4", LABEL_A)] if a else [])
    for run, color, lbl in runs:
        ax2.plot(run["t_min"], run["usd_in"], color=color, ls="-", label=f"{lbl} — input $")
        ax2.plot(run["t_min"], run["usd_out"], color=color, ls="--", label=f"{lbl} — output $")
        ax2.plot(run["t_min"], run["usd_cached"], color=color, ls=":", label=f"{lbl} — cached $")
    ax2.set_xlabel("fleet wall-clock elapsed (min)")
    ax2.set_ylabel("cumulative spend by component ($)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "budget_over_time.png", dpi=120)
    plt.close(fig)

    summary = {"window_minutes": round(mins, 1), "this_run": _final(b)}
    if a:
        summary["baseline"] = _final(a)
    (ANALYSIS_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    tr = summary["this_run"]
    base = f"  base_RHAE_L={summary['baseline']['rhae_l_final']}" if a else ""
    print(
        f"window={mins:.1f}min  RHAE_L={tr['rhae_l_final']} RHAE_U={tr['rhae_u_final']} "
        f"$={tr['usd_total']}{base}"
    )
    print(
        f"wrote rhae_over_time.png, budget_over_time.png, comparison_summary.json -> {ANALYSIS_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
