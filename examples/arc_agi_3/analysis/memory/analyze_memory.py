#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-game memory-store analysis for a run.

For every game with a persisted ``team_nemo/shared/memory.sqlite`` (the memory
variant), reports what the agent actually curated: total memories, breakdown by
type (episode/reflection/info/todo/...) and importance, reinforcement/recall
activity, and the top tags. A fleet summary and a per-game table are written.

The stores are opened READ-ONLY (immutable) so a live run can be analysed
without disturbing it.

Usage (any python3, stdlib only):
    python3 analyze_memory.py <run_dir> [--out <output_dir>]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

SKIP = {"red_team", "analysis", "world_model", "memory"}


def _connect_ro(path: Path) -> sqlite3.Connection | None:
    try:
        return sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    except sqlite3.Error:
        return None


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def analyze_store(path: Path) -> dict | None:
    con = _connect_ro(path)
    if con is None:
        return None
    try:
        cols = _columns(con, "memories")
        if not cols:
            return None
        where = " WHERE archived = 0" if "archived" in cols else ""
        rows = con.execute(f"SELECT * FROM memories{where}").fetchall()  # noqa: S608
        names = [d[0] for d in con.execute(f"SELECT * FROM memories{where} LIMIT 1").description]  # noqa: S608
    except sqlite3.Error:
        con.close()
        return None
    con.close()
    idx = {n: i for i, n in enumerate(names)}
    by_type: collections.Counter = collections.Counter()
    tags: collections.Counter = collections.Counter()
    reinforced = recalled = 0
    imps = []
    for r in rows:
        if "type" in idx:
            by_type[r[idx["type"]]] += 1
        if "importance" in idx and r[idx["importance"]] is not None:
            imps.append(float(r[idx["importance"]]))
        if "reinforcement_count" in idx:
            reinforced += int(r[idx["reinforcement_count"]] or 0)
        if "recalled_count" in idx:
            recalled += int(r[idx["recalled_count"]] or 0)
        if "tags" in idx and r[idx["tags"]]:
            try:
                for t in json.loads(r[idx["tags"]]):
                    tags[t] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return {
        "total": len(rows),
        "by_type": dict(by_type),
        "avg_importance": round(sum(imps) / len(imps), 2) if imps else 0.0,
        "reinforced": reinforced,
        "recalled": recalled,
        "top_tags": [t for t, _ in tags.most_common(8)],
    }


def _find_store(game_dir: Path) -> Path | None:
    direct = game_dir / "team_nemo" / "shared" / "memory.sqlite"
    if direct.exists():
        return direct
    for run in sorted((game_dir / "memory").glob("2*")) if (game_dir / "memory").is_dir() else []:
        cand = run / "team_nemo" / "shared" / "memory.sqlite"
        if cand.exists():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-game memory-store analysis for a run.")
    ap.add_argument("run_dir", help="multi-run container (or single run) to analyze")
    ap.add_argument("--out", default=None, help="output dir (default: <run_dir>/memory)")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else (run / "memory")
    out.mkdir(parents=True, exist_ok=True)

    # A run_dir may be a multi-run container (per-game subdirs) or a single run.
    candidates = [d for d in sorted(run.iterdir()) if d.is_dir() and d.name not in SKIP]
    games: list[tuple[str, Path]] = []
    for d in candidates:
        store = _find_store(d)
        if store:
            games.append((d.name, store))
    if not games:  # single-run dir
        store = run / "team_nemo" / "shared" / "memory.sqlite"
        if store.exists():
            games.append((run.name, store))

    rows = []
    for name, store in games:
        stats = analyze_store(store)
        if stats:
            rows.append({"game": name, **stats})
    rows.sort(key=lambda r: r["total"], reverse=True)

    (out / "memory_usage.json").write_text(json.dumps(rows, indent=2))

    lines = [
        f"# Memory usage per game — `{run.name}`",
        "",
        "Curated long-term memory per game (memory variant). **total** = live "
        "(non-archived) memories; **types** = episode/reflection/info/todo; "
        "**reinf/recall** = reinforcement + recall activity.",
        "",
        "| game | total | types | avg imp | reinf | recall | top tags |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        types = ", ".join(f"{k}:{v}" for k, v in sorted(r["by_type"].items()))
        lines.append(
            f"| {r['game']} | {r['total']} | {types or '·'} | {r['avg_importance']} | "
            f"{r['reinforced']} | {r['recalled']} | {', '.join(r['top_tags'][:5]) or '·'} |"
        )
    n = len(rows)
    total_mem = sum(r["total"] for r in rows)
    lines += [
        "",
        "## Fleet summary",
        f"- **{n} games** with a persisted memory store.",
        f"- **{total_mem} memories** total; mean {round(total_mem / n, 1) if n else 0} per game.",
    ]
    (out / "memory_usage.md").write_text("\n".join(lines) + "\n")
    print(f"analyzed {n} store(s), {total_mem} memories -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
