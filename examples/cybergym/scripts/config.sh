# Shared configuration for the CyberGym example scripts.
#
# Every other script in this directory sources this file. Override any value by
# exporting it before you run a script, e.g.:
#
#   MODEL=glm-5.2 CYBERGYM_SERVER=http://127.0.0.1:9000 scripts/run_subset.sh
#
# shellcheck shell=bash

# Root of this example (the directory that contains this scripts/ folder).
AGENT_REPO="${AGENT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AGENT_REPO

# CyberGym benchmark checkout + data (created by scripts/setup.sh).
export CYBERGYM_REPO="${CYBERGYM_REPO:-$AGENT_REPO/cybergym_repo}"
export CYBERGYM_DATA_DIR="${CYBERGYM_DATA_DIR:-$CYBERGYM_REPO/cybergym_data/data}"
export CYBERGYM_MASK_MAP="${CYBERGYM_MASK_MAP:-$CYBERGYM_REPO/mask_map.json}"

# CyberGym submission server.
export CYBERGYM_SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"

# Shared token between the CyberGym server and the validation step. It is not a
# real secret (only local /submit-fix, /query-poc, /verify-agent-pocs use it, and
# the agent's own /submit-vul is public), but we still keep it out of the repo:
# scripts/setup.sh generates one into the gitignored .env. Load it here if set.
if [ -z "${CYBERGYM_API_KEY:-}" ] && [ -f "$AGENT_REPO/.env" ]; then
  _cg_line=$(grep -E '^[[:space:]]*(export[[:space:]]+)?CYBERGYM_API_KEY=' "$AGENT_REPO/.env" | tail -n1 || true)
  if [ -n "$_cg_line" ]; then
    _cg_line=${_cg_line#*CYBERGYM_API_KEY=}
    _cg_line=${_cg_line%\"} ; _cg_line=${_cg_line#\"}
    _cg_line=${_cg_line%\'} ; _cg_line=${_cg_line#\'}
    export CYBERGYM_API_KEY="$_cg_line"
  fi
  unset _cg_line
fi

# Model + agent image.
export MODEL="${MODEL:-glm-5.2}"
export REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
export RUNNER_IMAGE="${RUNNER_IMAGE:-nooa/nooa-cybergym:latest}"

# Hard per-task wall-clock limit (seconds). The container is killed at this cap.
# Keep it above the agent's soft timeout (NOOA_CYBERGYM_SOFT_TIMEOUT_SEC, default
# 13920s / 3h52m) so the agent can return its best PoC gracefully before the kill.
export TIMEOUT="${TIMEOUT:-14400}"  # 4h

# Python virtualenv managed from this example's frozen uv lockfile.
export VENV="${VENV:-$AGENT_REPO/.venv}"

# The official CyberGym 10-task subset used by this example.
SUBSET_TASKS=(
  arvo:47101
  arvo:3938
  arvo:24993
  arvo:1065
  arvo:10400
  arvo:368
  oss-fuzz:42535201
  oss-fuzz:42535468
  oss-fuzz:370689421
  oss-fuzz:385167047
)

# Activate the virtualenv created by scripts/setup.sh.
activate_venv() {
  if [ ! -f "$VENV/bin/activate" ]; then
    echo "virtualenv not found at $VENV. Run scripts/setup.sh first." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}
