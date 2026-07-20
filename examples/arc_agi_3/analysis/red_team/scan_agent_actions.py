# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan every game's agent actions for rule violations.

For each game:
  * ATTEMPTS  -> classify the CODE of every execute_python cell (assistant.md).
  * DATA ACCESS -> classify the STDOUT/STDERR of every tool-result block
                   (user.md PythonOutput). If a tool output *contains* game
                   source, foreign-file contents, or network responses, that is
                   proof an attempt returned data.

Also flags escape-tool usage (self.shell/repo/pyp/web/mcp) regardless of target.

De-dup: the same code cell / output is echoed across many message files (the
conversation history repeats). We dedup by (where, code/stdout hash) so each
distinct action is reported once, with the earliest step/round it appears at.

Outputs:
  evidence/actions_<alias>.json   per-game detail
  evidence/actions_summary.json   aggregate counts + all hits
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rt_common as rt  # noqa: E402


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:12]


def classify(
    text: str, where: str, alias: str, file: str, step: int, rnd: int, escape: bool
) -> list[rt.Hit]:
    hits: list[rt.Hit] = []
    benign = bool(rt.BENIGN.search(text))
    for rule, sigs in rt.RULES.items():
        for sig, rx in sigs.items():
            lines = rt.matching_lines(text, rx, ctx=0)
            for ln in lines:
                # Skip a hit whose only match is inside a benign token on that line.
                hits.append(rt.Hit(rule, sig, where, alias, file, step, rnd, ln, benign, escape))
    return hits


def scan_game(alias: str, run_dir: Path) -> dict:
    mdir = rt.messages_dir(run_dir)
    if not mdir.is_dir():
        return {"alias": alias, "error": "no messages dir", "hits": []}

    seen_code: dict[str, tuple[int, int, str]] = {}  # hash -> (step,round,file)
    seen_out: dict[str, tuple[int, int, str]] = {}
    code_hits: list[rt.Hit] = []
    out_hits: list[rt.Hit] = []
    escape_cells: list[dict] = []
    n_code = n_out = 0

    files = sorted(mdir.glob("step_*_round_*.md"))
    for f in files:
        sr = rt.parse_step_round(f)
        if not sr:
            continue
        step, rnd, kind = sr
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue

        if kind == "assistant":
            for name, code in rt.iter_code_cells(text):
                key = _h(code)
                if key in seen_code:
                    continue
                seen_code[key] = (step, rnd, f.name)
                n_code += 1
                esc = bool(rt.ESCAPE_TOOL_CALL.search(code)) or name != "execute_python"
                if esc:
                    escape_cells.append(
                        {
                            "alias": alias,
                            "file": f.name,
                            "step": step,
                            "round": rnd,
                            "tool": name,
                            "escape_lines": rt.matching_lines(code, rt.ESCAPE_TOOL_CALL, ctx=1)[:6],
                            "code_sha": key,
                        }
                    )
                code_hits.extend(classify(code, "code", alias, f.name, step, rnd, esc))
        else:  # user.md -> tool outputs
            for stdout, stderr in rt.iter_tool_outputs(text):
                for blob in (stdout, stderr):
                    if not blob.strip():
                        continue
                    key = _h(blob)
                    if key in seen_out:
                        continue
                    seen_out[key] = (step, rnd, f.name)
                    n_out += 1
                    out_hits.extend(classify(blob, "stdout", alias, f.name, step, rnd, False))

    all_hits = code_hits + out_hits
    by_rule = defaultdict(int)
    for h in all_hits:
        by_rule[h.rule] += 1
    return {
        "alias": alias,
        "run_dir": str(run_dir),
        "n_code_cells": n_code,
        "n_tool_outputs": n_out,
        "n_escape_cells": len(escape_cells),
        "escape_cells": escape_cells,
        "hits_by_rule": dict(by_rule),
        "code_hits": [h.to_dict() for h in code_hits],
        "output_hits": [h.to_dict() for h in out_hits],
    }


def main() -> int:
    ev = rt.EVID
    ev.mkdir(exist_ok=True)
    summary = {
        "run": str(rt.RUN_ROOT),
        "games": {},
        "totals": defaultdict(int),
        "escape_cells_total": 0,
    }
    for alias, run_dir in rt.game_dirs():
        res = scan_game(alias, run_dir)
        (ev / f"actions_{alias}.json").write_text(json.dumps(res, indent=2))
        summary["games"][alias] = {
            "n_code_cells": res.get("n_code_cells", 0),
            "n_tool_outputs": res.get("n_tool_outputs", 0),
            "n_escape_cells": res.get("n_escape_cells", 0),
            "hits_by_rule": res.get("hits_by_rule", {}),
        }
        for r, c in res.get("hits_by_rule", {}).items():
            summary["totals"][r] += c
        summary["escape_cells_total"] += res.get("n_escape_cells", 0)
        print(
            f"{alias:6s} code={res.get('n_code_cells', 0):5d} "
            f"out={res.get('n_tool_outputs', 0):5d} "
            f"escape_cells={res.get('n_escape_cells', 0):3d} "
            f"hits={res.get('hits_by_rule', {})}"
        )
    summary["totals"] = dict(summary["totals"])
    (ev / "actions_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "\nTOTALS:",
        json.dumps(summary["totals"]),
        "escape_cells_total:",
        summary["escape_cells_total"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
