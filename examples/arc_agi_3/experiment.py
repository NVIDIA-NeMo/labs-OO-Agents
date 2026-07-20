# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the memory-vs-mdfiles comparison matrix for one game.

    python examples/arc_agi_3/experiment.py --game ls20 [--max-turns 60] [--skip-seeded]

Phase 1: fresh runs of both variants (in parallel).
Phase 2: "second episode" — each variant re-runs the game seeded with its own
phase-1 knowledge (memory.sqlite / knowledge/*.md + helpers). This isolates the
knowledge-store contribution: same game, same model, prior knowledge available.
Finally prints the comparison table (compare.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
VARIANTS = ["memory", "mdfiles"]


def launch(
    game: str,
    variant: str,
    *,
    group: str,
    tag: str,
    max_turns: int,
    max_env_steps: int,
    turn_timeout: float,
    nudge_after: float = 900.0,
    effort_ladder: str = "600:medium",
    model: str | None = None,
    reasoning_effort: str | None = None,
    seed_from: Path | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(EXAMPLE_DIR / "run_solver.py"),
        "--game",
        game,
        "--variant",
        variant,
        "--group",
        group,
        "--tag",
        tag,
        "--max-turns",
        str(max_turns),
        "--max-env-steps",
        str(max_env_steps),
        "--agent-turn-timeout",
        str(turn_timeout),
        "--nudge-after",
        str(nudge_after),
        "--effort-ladder",
        effort_ladder,
        "--kill-tmux",
    ]
    if model:
        cmd += ["--model", model]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]
    if seed_from is not None:
        cmd += ["--seed-knowledge", str(seed_from)]
    print(f"[experiment] launching {variant} {tag}: {' '.join(cmd[1:])}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_all(procs: dict[str, subprocess.Popen]) -> None:
    for name, p in procs.items():
        out, _ = p.communicate()
        print(f"--- {name} (exit {p.returncode}) ---")
        print("\n".join(out.strip().splitlines()[-6:]))


def find_run(group_dir: Path, variant: str, tag: str) -> Path | None:
    runs = sorted(group_dir.glob(f"*_{variant}_{tag}"), reverse=True)
    return runs[0] if runs else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--group", default="nemo_solver")
    p.add_argument("--max-turns", default=None, help="max agent turns; None/unlimited by default")
    p.add_argument("--max-env-steps", type=int, default=5000)
    p.add_argument(
        "--turn-timeout",
        type=float,
        default=1200.0,
        help="agent silence before agent_timeout (kill rung)",
    )
    p.add_argument(
        "--nudge-after",
        type=float,
        default=900.0,
        help="agent silence before the harness reminder state",
    )
    p.add_argument(
        "--effort-ladder",
        default="600:medium",
        help="reasoning-effort downshift rungs, e.g. '600:medium'",
    )
    p.add_argument(
        "--seeded-max-turns",
        default=None,
        help="max turns for the seeded phase; None/unlimited by default",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--skip-seeded", action="store_true")
    p.add_argument("--serial", action="store_true", help="run variants one at a time")
    args = p.parse_args()

    group_dir = REPO_ROOT / "results" / "arc_agi_3" / args.group
    stamp = time.strftime("%H%M%S")

    # Phase 1 — fresh runs
    tag1 = f"fresh{stamp}"
    procs: dict[str, subprocess.Popen] = {}
    for v in VARIANTS:
        procs[v] = launch(
            args.game,
            v,
            group=args.group,
            tag=tag1,
            max_turns=args.max_turns,
            max_env_steps=args.max_env_steps,
            turn_timeout=args.turn_timeout,
            nudge_after=args.nudge_after,
            effort_ladder=args.effort_ladder,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        if args.serial:
            wait_all({v: procs.pop(v)})
        else:
            time.sleep(5)  # stagger startup (env download locks, tmux)
    wait_all(procs)

    if not args.skip_seeded:
        # Phase 2 — second episode seeded with own phase-1 knowledge
        tag2 = f"seeded{stamp}"
        procs = {}
        for v in VARIANTS:
            prior = find_run(group_dir, v, tag1)
            if prior is None:
                print(f"[experiment] no phase-1 run found for {v}; skipping seeded run")
                continue
            procs[v] = launch(
                args.game,
                v,
                group=args.group,
                tag=tag2,
                max_turns=args.seeded_max_turns,
                max_env_steps=args.max_env_steps,
                turn_timeout=args.turn_timeout,
                nudge_after=args.nudge_after,
                effort_ladder=args.effort_ladder,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                seed_from=prior / "team_nemo" / "shared",
            )
            if args.serial:
                wait_all({v: procs.pop(v)})
            else:
                time.sleep(5)
        wait_all(procs)

    subprocess.run([sys.executable, str(EXAMPLE_DIR / "compare.py"), "--group", args.group])


if __name__ == "__main__":
    main()
