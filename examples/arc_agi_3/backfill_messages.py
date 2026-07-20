# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backfill analyze-tab message dumps for runs recorded before step_messages existed.

    python examples/arc_agi_3/backfill_messages.py <run_dir> [...]

Reconstructs ``agent_logs/nemo/team_leader/messages/step_NNN_round_00_*.md`` from
the ``agent_turn`` + ``env_step`` events in events.jsonl. Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def backfill(run_dir: Path) -> int:
    events_path = run_dir / "agent_logs" / "nemo" / "team_leader" / "events.jsonl"
    if not events_path.exists():
        return 0
    msgs = events_path.parent / "messages"
    msgs.mkdir(exist_ok=True)
    turns: list[dict] = []
    steps: list[dict] = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "agent_turn":
            turns.append(e)
        elif e.get("event") == "env_step" and e.get("step", 0) > 0:
            steps.append(e)

    def turn_for(step_no: int) -> dict | None:
        cur = None
        for t in turns:
            if t["step"] < step_no:
                cur = t
            else:
                break
        return cur

    written = 0
    for s in steps:
        n = s["step"]
        t = turn_for(n)
        if t is None:
            continue
        batch = t.get("actions", [])
        idx = n - t["step"] - 1
        marker = lambda i, _idx=idx: "->" if i == _idx else "  "  # noqa: E731
        user = (
            f"# Turn {t['turn']} — env step {n}\n\n"
            f"levels_completed={s.get('levels_completed')} "
            f"diff_pixels={s.get('diff_pixels')}\n\n"
            f"Executing action {idx + 1}/{len(batch)} of this turn's batch.\n"
        )
        assistant = (
            f"# Agent turn {t['turn']}\n\n"
            f"## Rationale / prediction\n\n{t.get('rationale', '').strip() or '(none)'}\n\n"
            f"## Action batch\n\n"
            + "\n".join(f"{marker(i)} {i + 1}. `{a}`" for i, a in enumerate(batch))
            + f"\n\n## Outcome of `{s.get('action_name')}`\n\n"
            f"- diff_pixels: {s.get('diff_pixels')}\n"
            f"- levels_completed: {s.get('levels_completed')}\n"
        )
        (msgs / f"step_{n:03d}_round_00_user.md").write_text(user)
        (msgs / f"step_{n:03d}_round_00_assistant.md").write_text(assistant)
        written += 1
    return written


def backfill_step_actions(run_dir: Path) -> int:
    """Inject `action_chosen`/`reward` into steps/*.json (viewer Step Details
    reads those keys, not producer_action). Reward joined from gameplay.json."""
    rewards: dict[int, float] = {}
    gameplay = run_dir / "gameplay.json"
    if gameplay.exists():
        try:
            for entry in json.loads(gameplay.read_text()).get("log", []):
                if entry.get("type") == "step":
                    rewards[entry.get("global_step")] = entry.get("reward", 0.0)
        except json.JSONDecodeError:
            pass
    fixed = 0
    for f in sorted(run_dir.glob("steps/step_*.json")):
        try:
            obs = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "action_chosen" in obs:
            continue
        obs["action_chosen"] = obs.get("producer_action", "")
        obs.setdefault("reward", rewards.get(obs.get("step"), 0.0))
        f.write_text(json.dumps(obs))
        fixed += 1
    return fixed


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]]
    if not targets:
        root = Path(__file__).resolve().parents[2] / "results" / "arc_agi_3"
        targets = sorted(
            {p.parents[3] for p in root.glob("*/*/agent_logs/*/team_leader/events.jsonl")}
        )
    for t in targets:
        n = backfill(t)
        a = backfill_step_actions(t)
        print(f"{t.name}: {n} step messages, {a} step actions injected")


if __name__ == "__main__":
    main()
