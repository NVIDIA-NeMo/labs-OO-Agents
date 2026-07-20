#!/bin/bash
# Run all red-team scanners against an external run dir, writing evidence to an
# external out dir. Usage:
#   ./run_scan.sh <run_dir> [out_dir]
# <run_dir>  = a multi-run container (or single run) to audit.
# [out_dir]  = evidence output dir (default: <run_dir>/red_team/evidence).
set -u
RUN="${1:?usage: run_scan.sh <run_dir> [out_dir]}"
# Resolve run/out to ABSOLUTE paths BEFORE cd-ing into the script dir, so
# relative arguments (from any cwd) still work.
RT_RUN_ROOT="$(cd "$RUN" && pwd)"; export RT_RUN_ROOT
OUT="${2:-$RT_RUN_ROOT/red_team/evidence}"
mkdir -p "$OUT"; RT_OUT="$(cd "$OUT" && pwd)"; export RT_OUT
cd "$(dirname "$0")"
echo "[red_team] run=$RT_RUN_ROOT"
echo "[red_team] out=$RT_OUT"
for s in scan_internet scan_returned_data scan_escape scan_agent_actions \
         first_violation scan_memstore_xref scan_harness_exposure \
         scan_name_leak digest_escape; do
  echo "==== $s ===="; python3 "$s.py" 2>&1 | tail -4
done
echo "[red_team] evidence written to $RT_OUT"
