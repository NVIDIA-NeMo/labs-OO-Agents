#!/usr/bin/env bash
#
# Post-validate the PoCs an agent submitted: replay each one against the fixed
# build and fill in fix_exit_code in the server's poc.db, then print a summary.
#
# Usage:
#   scripts/validate.sh [RUN_DIR]
#
# RUN_DIR defaults to the most recent runs/validation_10task_* directory. Pass a
# path to validate a different run (e.g. runs/logs for a single-task run).
#
# Requires the CyberGym server (scripts/start_server.sh) to be running.
#
set -euo pipefail
source "$(dirname "$0")/config.sh"
activate_venv

if [ -z "${CYBERGYM_API_KEY:-}" ]; then
  echo "CYBERGYM_API_KEY is not set. Run scripts/setup.sh first (it generates one in .env)." >&2
  echo "It must match the key the running server was started with." >&2
  exit 1
fi
# verify_agent_result.py reads CYBERGYM_API_KEY from the environment.
export CYBERGYM_API_KEY

RUN_DIR="${1:-$(ls -td "$AGENT_REPO"/runs/validation_10task_* 2>/dev/null | head -1 || true)}"
if [ -z "${RUN_DIR:-}" ] || [ ! -d "$RUN_DIR" ]; then
  echo "No run directory found. Pass one explicitly: scripts/validate.sh <run-dir>" >&2
  exit 1
fi

POC_DB="$AGENT_REPO/runs/server/poc.db"
echo "==> Validating runs under $RUN_DIR"

mapfile -t args_files < <(find "$RUN_DIR" -name args.json | sort)
if [ "${#args_files[@]}" -eq 0 ]; then
  echo "No args.json files found under $RUN_DIR" >&2
  exit 1
fi

for args in "${args_files[@]}"; do
  agent_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["agent_id"])' "$args")
  echo "==> verifying agent_id=$agent_id"
  # verify_agent_result.py POSTs to /verify-agent-pocs (using $CYBERGYM_API_KEY)
  # to run the fixed-build check, then prints each PoC record from the DB.
  python3 "$CYBERGYM_REPO/scripts/verify_agent_result.py" \
    --server "$CYBERGYM_SERVER" \
    --pocdb_path "$POC_DB" \
    --agent_id "$agent_id"
done

echo
echo "==> Scoring the single frozen final PoC and writing signed evidence"
if [ ! -d "$XEUS_CYBERGYM_REPO/src/xeus_cybergym" ]; then
  echo "Xeus CyberGym authority code not found at $XEUS_CYBERGYM_REPO" >&2
  exit 1
fi
PYTHONPATH="$CYBERGYM_REPO/src:$XEUS_CYBERGYM_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$AGENT_REPO/scripts/score_final.py" \
    --run-dir "$RUN_DIR" \
    --poc-db "$POC_DB" \
    --output-dir "$RUN_DIR/official_evidence"
