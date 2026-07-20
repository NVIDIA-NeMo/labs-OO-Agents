#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-game world-model usage analysis for the interactive-game-solver (WM) run.

For every game it reports, from the PERSISTED model helpers and the agent's EXECUTED
cells, HOW the world model is actually used:

  built    — what the model helper(s) define: encode / predict / render / planner
             (plan|bfs|search|is_goal|solve|path) — capability, not just a stub.
  called   — how often the agent invokes each capability via self.h.<module>.<fn>
             in its execute_python cells (encode = perception, predict = forward
             simulation, planner = search/goal-seeking over the model).
  retrodict — cells that call predict AND compare it to reality (mism/retrodict),
             i.e. the model is being validated, not just written.
  tier     — built-only < perception < planning < predictive < full(predict+search).

Writes: world_model_usage.md (comparison table) + world_model_usage.json.
Run with any python3 (stdlib + grep):  python3 analyze_world_models.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

_ap = argparse.ArgumentParser(description="Per-game world-model usage audit for a run.")
_ap.add_argument("run_dir", help="multi-run container (or single run) to analyze")
_ap.add_argument("--out", default=None, help="output dir (default: <run_dir>/world_model)")
_args = _ap.parse_args()
RUN = Path(_args.run_dir).resolve()
OUT = Path(_args.out).resolve() if _args.out else (RUN / "world_model")
OUT.mkdir(parents=True, exist_ok=True)
SKIP = {"red_team", "analysis", "world_model", "memory"}

ENC = re.compile(r"encode", re.I)
PRED = re.compile(r"predict|simulate|rollout|step_state|safe_step|forward", re.I)
PLAN = re.compile(r"\b(plan|bfs|dfs|search|is_goal|solve|path|expand|frontier|astar|greedy)", re.I)


def _cap(fn: str) -> str | None:
    if ENC.search(fn):
        return "encode"
    if PRED.search(fn):
        return "predict"
    if PLAN.search(fn):
        return "planner"
    return None


def _grep_count(pat: str, root: Path) -> list[str]:
    if not root.exists():
        return []
    r = subprocess.run(["grep", "-rhoiE", pat, str(root)], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x]


def main() -> int:
    now = time.time()
    rows = []
    for d in sorted(RUN.iterdir()):
        if not d.is_dir() or d.name in SKIP or not (d / "memory").is_dir():
            continue
        g = d.name
        run = next(iter((d / "memory").glob("2*")), None)
        if not run:
            continue

        # ---- built: model helper defs ----
        helpers = list((run / "team_nemo" / "shared" / "helpers").glob("*.py"))
        model_lines, defs = 0, []
        for p in helpers:
            try:
                src = p.read_text(errors="ignore")
            except OSError:
                continue
            model_lines += src.count("\n")
            defs += re.findall(r"^\s*def\s+([a-zA-Z0-9_]+)", src, re.M)
        built = {
            "encode": any(_cap(x) == "encode" for x in defs),
            "predict": any(_cap(x) == "predict" for x in defs),
            "render": any("render" in x.lower() for x in defs),
            "planner": any(_cap(x) == "planner" for x in defs),
        }

        # ---- called: self.h.<mod>.<fn> in executed cells ----
        msgs = run / "agent_logs" / "nemo" / "team_leader" / "messages"
        calls = {"encode": 0, "predict": 0, "planner": 0, "other": 0}
        for m in _grep_count(r"self\.h\.[a-z0-9_]+\.[a-z0-9_]+", msgs):
            fn = m.rsplit(".", 1)[-1]
            calls[_cap(fn) or "other"] += 1
        # retrodiction: cells that predict AND check against reality
        retro = (
            len(_grep_count(r"(mism|retrodict|predicted.*actual|actual.*predict)", msgs))
            if calls["predict"]
            else 0
        )

        # ---- tier ----
        if calls["encode"] + calls["predict"] + calls["planner"] == 0:
            tier = "built-only"
        elif calls["predict"] and calls["planner"]:
            tier = "full(pred+search)"
        elif calls["predict"]:
            tier = "predictive"
        elif calls["planner"]:
            tier = "planning"
        else:
            tier = "perception"

        # ---- status ----
        acts = run / "ipc" / "actions.jsonl"
        sts = run / "ipc" / "states.jsonl"
        n_act = sum(1 for _ in open(acts)) if acts.exists() else 0
        n_st = sum(1 for _ in open(sts)) if sts.exists() else 0
        done = (run / "result.json").exists()
        alive = (
            subprocess.run(
                ["pgrep", "-f", f"launcher.py.*{g}.*{RUN.name.split('_')[0]}"], capture_output=True
            ).returncode
            == 0
        )
        act_age = (now - acts.stat().st_mtime) / 60 if acts.exists() else 0
        status = (
            "done"
            if done
            else ("STUCK" if alive and n_act and act_age > 25 else "running" if alive else "ended")
        )

        rows.append(
            {
                "game": g,
                "status": status,
                "actions": n_act,
                "states": n_st,
                "model_files": len(helpers),
                "model_lines": model_lines,
                "n_funcs": len(defs),
                "built": built,
                "calls": calls,
                "retrodict": retro,
                "tier": tier,
                "last_action_min_ago": round(act_age, 1),
            }
        )

    # sort: tier depth desc, then calls
    order = {
        "full(pred+search)": 4,
        "predictive": 3,
        "planning": 2,
        "perception": 1,
        "built-only": 0,
    }
    rows.sort(
        key=lambda r: (order.get(r["tier"], 0), r["calls"]["predict"] + r["calls"]["planner"]),
        reverse=True,
    )

    (OUT / "world_model_usage.json").write_text(json.dumps(rows, indent=2))

    def yn(b):
        return "✓" if b else "·"

    lines = [
        "# World-model usage per game — `" + RUN.name + "`",
        "",
        "How each game's agent builds AND uses its world model. **built** = capability defined in the",
        "persisted `team_nemo/shared/helpers/*.py`; **called** = times invoked via `self.h.<mod>.<fn>`",
        "in executed cells (encode=perception, predict=forward-sim, plan=search/goal). **tier**: ",
        "built-only < perception < planning < predictive < full(predict+search).",
        "",
        "| game | status | tier | model (lines/funcs) | built E/P/R/Plan | called Enc/Pred/Plan | retrodict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = r["built"]
        lines.append(
            f"| {r['game']} | {r['status']} | {r['tier']} | {r['model_lines']}/{r['n_funcs']} | "
            f"{yn(b['encode'])}/{yn(b['predict'])}/{yn(b['render'])}/{yn(b['planner'])} | "
            f"{r['calls']['encode']}/{r['calls']['predict']}/{r['calls']['planner']} | "
            f"{r['retrodict'] or '·'} |"
        )

    # aggregate
    n = len(rows)
    agg = {t: sum(1 for r in rows if r["tier"] == t) for t in order}
    lines += [
        "",
        "## Fleet summary",
        f"- **{n} games** with a persisted world model.",
        f"- **built:** encode {sum(r['built']['encode'] for r in rows)}/{n}, "
        f"predict {sum(r['built']['predict'] for r in rows)}/{n}, "
        f"planner {sum(r['built']['planner'] for r in rows)}/{n}, "
        f"render {sum(r['built']['render'] for r in rows)}/{n}.",
        f"- **tiers:** full(pred+search) {agg['full(pred+search)']}, predictive {agg['predictive']}, "
        f"planning {agg['planning']}, perception {agg['perception']}, built-only {agg['built-only']}.",
        f"- **encode calls total** {sum(r['calls']['encode'] for r in rows):,}, "
        f"predict {sum(r['calls']['predict'] for r in rows):,}, "
        f"planner {sum(r['calls']['planner'] for r in rows):,}.",
        f"- **retrodiction** (validating predict vs reality): "
        f"{sum(1 for r in rows if r['retrodict'])} games.",
        f"- **STUCK** (hung in a search cell, actions frozen >25 min): "
        f"{[r['game'] for r in rows if r['status'] == 'STUCK']}.",
        "",
        "> Note: predict-rollout **search** is both the deepest world-model use AND the failure mode —",
        "> the STUCK games hung in non-terminating search cells (see red_team analysis).",
    ]
    (OUT / "world_model_usage.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote world_model_usage.md + .json -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
