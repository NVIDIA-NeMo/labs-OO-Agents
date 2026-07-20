# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualitative knowledge-store analysis for ARC-AGI-3 solver runs.

    uv run python examples/arc_agi_3/analyze_knowledge.py <run_dir> [<run_dir> ...]

For memory-variant runs: memory counts by type/importance, edges, reflection
maintenance history, and usage KPIs (recall/injection rates) from the store's
observability module. For mdfiles runs: file sizes, entry counts, hypothesis
table stats. Prints one report per run.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def analyze_memory(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "select type, importance, status, archived, access_count, content from memories"
    ).fetchall()
    print(f"  memories: {len(rows)} total")
    by_type = Counter(r[0] for r in rows)
    print(f"    by type: {dict(by_type)}")
    arch = sum(1 for r in rows if r[3])
    fetched = sum(1 for r in rows if r[4])
    print(f"    archived: {arch} | fetched at least once: {fetched}/{len(rows)}")
    edges = con.execute("select count(*) from memory_edges").fetchone()[0]
    print(f"    edges: {edges}")
    try:
        maint = con.execute(
            "select details from maintenance_log order by rowid desc limit 5"
        ).fetchall()
        print(f"    maintenance entries (reflections etc.): {len(maint)} (last 5 shown)")
        for (d,) in maint:
            print(f"      {d[:140]}")
    except sqlite3.OperationalError:
        pass
    todos = [(r[2] or "open") for r in rows if r[0] == "todo"]
    if todos:
        print(f"    todos: {dict(Counter(todos))}")
    con.close()


def analyze_md(knowledge_dir: Path) -> None:
    files = sorted(knowledge_dir.glob("*.md"))
    if not files:
        print("  no knowledge .md files")
        return
    for f in files:
        text = f.read_text()
        entries = len(re.findall(r"^#{1,3} ", text, re.M))
        table_rows = len(re.findall(r"^\|[^-|]", text, re.M))
        print(f"  {f.name}: {len(text)} chars, {entries} headed entries, {table_rows} table rows")


def analyze_run(run_dir: Path) -> None:
    print(f"\n=== {run_dir.name} ===")
    result = run_dir / "result.json"
    if result.exists():
        r = json.loads(result.read_text())
        print(
            f"  result: levels={r.get('levels_completed')} steps={r.get('total_steps')} "
            f"turns={r.get('turns')} wall={r.get('wall_time_seconds')}s "
            f"termination={r.get('termination_reason')}"
        )
    ws = run_dir / "team_nemo" / "shared"
    db = ws / "memory.sqlite"
    if db.exists():
        analyze_memory(db)
    kd = ws / "knowledge"
    if kd.is_dir():
        analyze_md(kd)
    helpers = list((ws / "helpers").glob("*.py")) if (ws / "helpers").is_dir() else []
    print(f"  helper modules: {[h.name for h in helpers]}")
    actions = run_dir / "ipc" / "actions.jsonl"
    if actions.exists():
        batches = [json.loads(ln) for ln in actions.read_text().splitlines() if ln.strip()]
        sizes = [len(b.get("actions", [])) for b in batches]
        if sizes:
            print(
                f"  action batches: {len(sizes)} (avg {sum(sizes) / len(sizes):.1f} "
                f"actions/batch, max {max(sizes)})"
            )


def dump_memories(db_path: Path) -> None:
    """Print every memory in full (newest last)."""
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "select id, type, importance, status, archived, access_count, content "
        "from memories order by created_at"
    ).fetchall()
    for mid, typ, imp, status, archived, fetches, content in rows:
        flags = " ".join(
            f
            for f in (
                f"status={status}" if status else "",
                "ARCHIVED" if archived else "",
                f"fetched×{fetches}" if fetches else "",
            )
            if f
        )
        print(f"\n[{typ}] {mid[:8]} imp={imp} {flags}")
        print("  " + content.replace("\n", "\n  "))
    print(f"\n{len(rows)} memories in {db_path}")
    con.close()


def main() -> None:
    args = list(sys.argv[1:])
    dump = "--dump" in args
    targets = [Path(a) for a in args if a != "--dump"]
    if not targets:
        root = Path(__file__).resolve().parents[2] / "results" / "arc_agi_3"
        targets = sorted(p.parent for p in root.glob("*/*/result.json"))
    for t in targets:
        if dump:
            db = t / "team_nemo" / "shared" / "memory.sqlite"
            if db.exists():
                dump_memories(db)
            else:
                print(f"{t}: no memory.sqlite")
        else:
            analyze_run(t)


if __name__ == "__main__":
    main()
