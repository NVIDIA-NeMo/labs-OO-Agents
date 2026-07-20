# ARC-AGI-3 run analyses

Reusable analyses for a solver run (a multi-run container from `run_multi.py`, or a
single run from `run_solver.py`). Every tool takes the **run dir to analyze** as an
argument and writes results to an **external output dir** — nothing is hardcoded to a
specific run, so these work on any run and store results anywhere.

| dir | tool | reads | writes | interpreter |
|-----|------|-------|--------|-------------|
| `red_team/` | `run_scan.sh <run> [out]` | agent cells + tool output | `evidence/*.json` + `escape_digest.md` | this venv (stdlib) |
| `world_model/` | `analyze_world_models.py <run> [--out]` | helpers + executed cells | `world_model_usage.{md,json}` | this venv (stdlib) |
| `memory/` | `analyze_memory.py <run> [--out]` | `team_nemo/shared/memory.sqlite` | `memory_usage.{md,json}` | this venv (stdlib) |
| `analysis/` | `build_comparison.py <run_b> [--baseline A] [--out]` | `events.jsonl` | `rhae_over_time.png`, `budget_over_time.png`, `comparison_summary.json` | **progressive-learning venv** (matplotlib + `arc_agi_3.rhae`) |

## Examples

```bash
RUN=results/arc_agi_3/nemo_solver/<container>
OUT=/tmp/analysis_out

# Red-team leakage/guardrail audit (internet / game-source / foreign-data / name-leak):
examples/arc_agi_3/analysis/red_team/run_scan.sh "$RUN" "$OUT/red_team"

# World-model usage per game:
.venv/bin/python examples/arc_agi_3/analysis/world_model/analyze_world_models.py "$RUN" --out "$OUT/world_model"

# Memory-store curation per game (memory variant):
.venv/bin/python examples/arc_agi_3/analysis/memory/analyze_memory.py "$RUN" --out "$OUT/memory"

# RHAE + budget over time (single run, or compare two with --baseline).
# Needs matplotlib + arc_agi_3.rhae -> run from the progressive-learning venv:
cd progressive-learning && .venv/bin/python \
  ../examples/arc_agi_3/analysis/analysis/build_comparison.py "$RUN" \
  --out "$OUT/analysis" --label-b "this run"
```

## Notes

- **red_team**: `run_scan.sh` sets `RT_RUN_ROOT` (run to audit) and `RT_OUT` (evidence
  dir) for the scanners. Individual scanners can be run directly with those env vars.
  `test_guard_evasion.py` / `test_harness_log_exposure.py` / `test_memstore_residual.py`
  are root-only isolation unit tests (fork+setuid) — they exercise the cell guard and
  uid-drop and skip cleanly without root.
- **analysis**: two-run comparison. Pass the run to analyze as `run_b`; add
  `--baseline <other_run>` to overlay a second fleet (the baseline is truncated to
  `run_b`'s elapsed window). Override the progressive-learning location with
  `ARC_PL_DIR` if it isn't at `<repo>/progressive-learning`.
- **memory**: opens stores read-only (`immutable=1`), so a live run can be analysed
  without disturbing it.
