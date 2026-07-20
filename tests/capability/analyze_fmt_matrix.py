# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze the fmt_matrix run.

Reads all .noo-eval.jsonl files under a results dir, filters to
fmt_<format>_<type>_<strategy> tests, computes pass rate per cell, and
emits a markdown report grouped by format.

Usage:
    uv run python tests/capability/analyze_fmt_matrix.py results/fmt_matrix \
        > tests/capability/REPORT_fmt_matrix.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Order in which to render formats / types / strategies.
FORMAT_ORDER = ["today_verbose", "xml", "lower", "slice_keys"]
TYPE_ORDER = ["list", "tuple", "dict", "set", "pydantic", "dataclass", "json", "records"]
STRATEGY_ORDER = ["predict", "codeact"]
# Display order for models — small-class first, flagship last.
MODEL_ORDER = [
    "nemotron3-nano-30b",
    "nemotron-super-49b",
    "claude-haiku-azure",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gpt-5-mini",
    "qwen3-80b",
    "qwen3.5-35b",
    "gpt-5.2",
    "claude-sonnet",
    "nemotron-3-super-v3",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]


def parse_test_name(name: str) -> tuple[str, str, str] | None:
    """[real]fmt_<format>_<type>_<strategy> -> (format, type, strategy).

    Returns None if not a fmt_/realfmt_ test.
    """
    if name.startswith("realfmt_"):
        parts = name[len("realfmt_") :].split("_")
    elif name.startswith("fmt_"):
        parts = name[len("fmt_") :].split("_")
    else:
        return None
    # parts[-1] = strategy ("predict" or "codeact")
    # parts[-2] = type
    # parts[:-2] = format (may be 1 or 2 tokens, e.g. "today_verbose" or "slice_keys")
    if len(parts) < 3:
        return None
    strategy = parts[-1]
    typ = parts[-2]
    fmt = "_".join(parts[:-2])
    return fmt, typ, strategy


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in results_dir.rglob("*.noo-eval.jsonl"):
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                if d.get("_type") != "result":
                    continue
                rows.append(d)
    return rows


def aggregate(rows: list[dict]):
    """Returns (cells, errors_by_cell) keyed by (format, type, strategy, model)."""
    cells: dict[tuple, list[bool]] = defaultdict(list)
    errors: dict[tuple, int] = defaultdict(int)
    for r in rows:
        parsed = parse_test_name(r.get("test_name") or "")
        if not parsed:
            continue
        fmt, typ, strategy = parsed
        model = r["model"]
        key = (fmt, typ, strategy, model)
        cells[key].append(bool(r.get("passed")))
        if r.get("error"):
            errors[key] += 1
    return cells, errors


def fmt_pct(passed: int, total: int) -> str:
    if total == 0:
        return "—"
    pct = 100 * passed / total
    return f"{passed}/{total} ({pct:.0f}%)"


def render_format_section(
    fmt: str,
    cells: dict,
    errors: dict,
    types_seen: list[str],
    models_seen: list[str],
) -> list[str]:
    out = []
    out.append(f"## Format: `{fmt}`")
    out.append("")

    # One table per strategy, with model as row, type as column.
    for strategy in STRATEGY_ORDER:
        out.append(f"### Strategy: {strategy}")
        out.append("")
        header = ["model"] + types_seen + ["overall"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join("---" for _ in header) + "|")
        for model in models_seen:
            row = [model]
            row_passed = 0
            row_total = 0
            for typ in types_seen:
                key = (fmt, typ, strategy, model)
                results = cells.get(key, [])
                if not results:
                    row.append("—")
                    continue
                passed = sum(1 for x in results if x)
                total = len(results)
                row_passed += passed
                row_total += total
                cell = fmt_pct(passed, total)
                err = errors.get(key, 0)
                if err:
                    cell += f" *err={err}*"
                row.append(cell)
            row.append(fmt_pct(row_passed, row_total))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return out


def render_format_summary(
    cells: dict,
    fmts_seen: list[str],
    types_seen: list[str],
    strategies_seen: list[str],
    models_seen: list[str],
) -> list[str]:
    """Top-level summary: format × strategy, averaged over types and models."""
    out = ["## Summary: format × strategy (averaged over types and models)", ""]
    header = ["format"] + [s for s in STRATEGY_ORDER if s in strategies_seen] + ["both"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for fmt in fmts_seen:
        row = [fmt]
        both_p = 0
        both_t = 0
        for strategy in [s for s in STRATEGY_ORDER if s in strategies_seen]:
            p = 0
            t = 0
            for typ in types_seen:
                for model in models_seen:
                    results = cells.get((fmt, typ, strategy, model), [])
                    p += sum(1 for x in results if x)
                    t += len(results)
            both_p += p
            both_t += t
            row.append(fmt_pct(p, t))
        row.append(fmt_pct(both_p, both_t))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def render_type_summary(
    cells: dict,
    fmts_seen: list[str],
    types_seen: list[str],
    strategies_seen: list[str],
    models_seen: list[str],
) -> list[str]:
    """type × format pivot, averaged over models, one table per strategy."""
    out = ["## Summary: type × format (averaged over models)", ""]
    for strategy in [s for s in STRATEGY_ORDER if s in strategies_seen]:
        out.append(f"### {strategy}")
        out.append("")
        header = ["type"] + fmts_seen
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join("---" for _ in header) + "|")
        for typ in types_seen:
            row = [typ]
            for fmt in fmts_seen:
                p = 0
                t = 0
                for model in models_seen:
                    results = cells.get((fmt, typ, strategy, model), [])
                    p += sum(1 for x in results if x)
                    t += len(results)
                if t == 0:
                    row.append("—")
                else:
                    row.append(fmt_pct(p, t))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No result rows found under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    cells, errors = aggregate(rows)

    # Deduce dimensions actually present.
    fmts_seen = [f for f in FORMAT_ORDER if any(k[0] == f for k in cells)]
    types_seen = [t for t in TYPE_ORDER if any(k[1] == t for k in cells)]
    strategies_seen = [s for s in STRATEGY_ORDER if any(k[2] == s for k in cells)]
    models_seen = [m for m in MODEL_ORDER if any(k[3] == m for k in cells)]
    # Catch anything we forgot in the orderings.
    extra_models = sorted({k[3] for k in cells} - set(models_seen))
    models_seen.extend(extra_models)

    n_cells = len(cells)
    n_samples = sum(len(v) for v in cells.values())

    out = []
    out.append("# Format × Type × Strategy matrix")
    out.append("")
    out.append(
        f"Aggregated {n_samples} samples across {n_cells} cells "
        f"(formats={len(fmts_seen)}, types={len(types_seen)}, "
        f"strategies={len(strategies_seen)}, models={len(models_seen)}).",
    )
    out.append("")

    out += render_format_summary(cells, fmts_seen, types_seen, strategies_seen, models_seen)
    out += render_type_summary(cells, fmts_seen, types_seen, strategies_seen, models_seen)

    for fmt in fmts_seen:
        # Restrict types_seen to those that actually have data for this format
        types_for_fmt = [
            t
            for t in types_seen
            if any((fmt, t, s, m) in cells for s in strategies_seen for m in models_seen)
        ]
        out += render_format_section(fmt, cells, errors, types_for_fmt, models_seen)

    print("\n".join(out))


if __name__ == "__main__":
    main()
