# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate and compare ARC-AGI-3 solver runs (memory vs mdfiles variants).

    python examples/arc_agi_3/compare.py [--results-root results/arc_agi_3] [--group ...]

Reads every ``result.json`` under the results root, groups by (game, variant),
prints a comparison table, and writes ``comparison.json`` + ``comparison.md``
next to the runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

METRICS = [
    ("levels_completed", "levels"),
    ("total_steps", "env steps"),
    ("turns", "turns"),
    ("wall_time_seconds", "wall s"),
    ("rhae_game_score", "rhae"),
]


def load_runs(results_root: Path, groups: list[str] | None) -> list[dict]:
    runs = []
    for result_path in sorted(results_root.glob("*/*/result.json")):
        group = result_path.parent.parent.name
        if groups and group not in groups:
            continue
        try:
            r = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("solver") != "nemo_single_agent":
            continue
        r["_run"] = result_path.parent.name
        r["_group"] = group
        r["_seeded"] = "seeded" in result_path.parent.name
        runs.append(r)
    return runs


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default=str(REPO_ROOT / "results" / "arc_agi_3"))
    p.add_argument(
        "--group",
        action="append",
        default=None,
        help="restrict to these grouping dirs (default: all)",
    )
    args = p.parse_args()
    root = Path(args.results_root)

    runs = load_runs(root, args.group)
    if not runs:
        print(f"no nemo_single_agent runs under {root}")
        return

    by_key = defaultdict(list)
    for r in runs:
        game = (r.get("game_id") or "?").split("-")[0]
        seeded = "+seed" if r["_seeded"] else ""
        by_key[(game, r.get("variant", "?") + seeded)].append(r)

    header = ["game", "variant", "n"] + [label for _, label in METRICS] + ["terminations"]
    rows = []
    for (game, variant), rs in sorted(by_key.items()):
        row = [game, variant, str(len(rs))]
        for key, _ in METRICS:
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            row.append(fmt(statistics.mean(vals)) if vals else "-")
        terms = defaultdict(int)
        for r in rs:
            terms[r.get("termination_reason", "?")] += 1
        row.append(",".join(f"{k}:{v}" for k, v in sorted(terms.items())))
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    lines = [
        " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=False)),
        "-|-".join("-" * w for w in widths),
    ]
    lines += [" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=False)) for row in rows]
    table = "\n".join(lines)
    print(table)

    out = {
        "runs": [dict(r.items()) for r in runs],
        "summary": {f"{g}/{v}": len(rs) for (g, v), rs in by_key.items()},
    }
    (root / "comparison.json").write_text(json.dumps(out, indent=2))
    (root / "comparison.md").write_text(
        "# ARC-AGI-3 nemo solver — variant comparison\n\n" + table + "\n\n"
        "Per-run details in comparison.json.\n"
    )
    print(f"\nwrote {root}/comparison.json and comparison.md ({len(runs)} runs)")


if __name__ == "__main__":
    main()
