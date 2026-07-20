# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token/cost report for ARC-AGI-3 solver runs, from their OTLP trace files.

    python3 examples/arc_agi_3/cost_report.py [run_dir ...] \
        [--price-in 1.25] [--price-cached 0.125] [--price-out 10.0]   # $/Mtok

Sums llm.token_count.* attributes over every LLM span in <run>/traces/*.jsonl.
Prices default to 0 (report tokens only) — pass your gateway's rates for $.
Runs recorded before trace wiring (the FT09 batch) have no traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LLM_SPANS = {"aresponses", "responses", "acompletion", "completion"}


def run_tokens(run_dir: Path) -> dict | None:
    files = sorted(run_dir.glob("traces/*.jsonl"))
    if not files:
        return None
    tot = {"calls": 0, "prompt": 0, "cache_read": 0, "completion": 0, "reasoning": 0}
    for path in files:
        for line in path.open():
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rs in doc.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for sp in ss.get("spans", []):
                        if sp.get("name") not in LLM_SPANS:
                            continue
                        attrs = {
                            a["key"]: list(a["value"].values())[0] for a in sp.get("attributes", [])
                        }
                        p = int(attrs.get("llm.token_count.prompt", 0) or 0)
                        if not p and not attrs.get("llm.token_count.completion"):
                            continue
                        tot["calls"] += 1
                        tot["prompt"] += p
                        tot["cache_read"] += int(
                            attrs.get("llm.token_count.prompt_details.cache_read", 0) or 0
                        )
                        tot["completion"] += int(attrs.get("llm.token_count.completion", 0) or 0)
                        tot["reasoning"] += int(
                            attrs.get("llm.token_count.completion_details.reasoning", 0) or 0
                        )
    return tot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="*")
    p.add_argument("--price-in", type=float, default=0.0, help="$ per Mtok uncached input")
    p.add_argument("--price-cached", type=float, default=0.0, help="$ per Mtok cached input")
    p.add_argument("--price-out", type=float, default=0.0, help="$ per Mtok output")
    args = p.parse_args()

    targets = [Path(r) for r in args.runs]
    if not targets:
        root = Path(__file__).resolve().parents[2] / "results" / "arc_agi_3"
        targets = sorted(d for d in root.glob("*/*/") if (d / "traces").is_dir())

    header = (
        f"{'run':<48} {'calls':>6} {'prompt':>12} {'cached':>12} {'output':>10} {'reasoning':>10}"
    )
    if args.price_in or args.price_out:
        header += f" {'$est':>8}"
    print(header)
    print("-" * len(header))
    grand = 0.0
    for t in targets:
        tok = run_tokens(t)
        if tok is None:
            print(f"{t.name:<48} (no traces)")
            continue
        line = (
            f"{t.name:<48} {tok['calls']:>6} {tok['prompt']:>12,} "
            f"{tok['cache_read']:>12,} {tok['completion']:>10,} {tok['reasoning']:>10,}"
        )
        if args.price_in or args.price_out:
            cost = (
                (tok["prompt"] - tok["cache_read"]) * args.price_in
                + tok["cache_read"] * args.price_cached
                + tok["completion"] * args.price_out
            ) / 1e6
            grand += cost
            line += f" {cost:>8.2f}"
        print(line)
    if grand:
        print(f"{'TOTAL':<48} {'':>6} {'':>12} {'':>12} {'':>10} {'':>10} {grand:>8.2f}")


if __name__ == "__main__":
    main()
