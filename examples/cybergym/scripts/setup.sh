#!/usr/bin/env bash
#
# One-time setup for the CyberGym example. Safe to re-run; each step is skipped
# or reused if it has already completed.
#
#   1. create a Python virtualenv
#   2. generate a local CyberGym API key in .env (if missing)
#   3. clone + install the CyberGym benchmark
#   4. fetch task data for the 10-task subset (Git LFS)
#   5. download the matching CyberGym Docker images
#   6. install this runner and build the agent image
#
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo "==> [1/6] Creating virtualenv at $VENV"
if [ ! -f "$VENV/bin/activate" ]; then
  python3 -m venv "$VENV"
fi
activate_venv
python3 -m pip install --upgrade pip

echo "==> [2/6] Ensuring a local CyberGym API key exists in .env"
ENV_FILE="$AGENT_REPO/.env"
touch "$ENV_FILE"
if grep -qE '^[[:space:]]*(export[[:space:]]+)?CYBERGYM_API_KEY=' "$ENV_FILE"; then
  echo "    CYBERGYM_API_KEY already present in $ENV_FILE"
else
  key="cybergym-$(python3 -c 'import uuid; print(uuid.uuid4())')"
  printf '\n# Local shared token between the CyberGym server and scripts/validate.sh.\nCYBERGYM_API_KEY=%s\n' "$key" >> "$ENV_FILE"
  export CYBERGYM_API_KEY="$key"
  echo "    Generated a new CYBERGYM_API_KEY in $ENV_FILE (gitignored)"
fi

echo "==> [3/6] Installing CyberGym into $CYBERGYM_REPO"
if [ ! -d "$CYBERGYM_REPO/.git" ]; then
  git clone https://github.com/sunblaze-ucb/cybergym.git "$CYBERGYM_REPO"
fi
(cd "$CYBERGYM_REPO" && python3 -m pip install -e '.[dev,server]')

echo "==> [4/6] Fetching task data for the subset (Git LFS)"
git lfs install
if [ ! -d "$CYBERGYM_REPO/cybergym_data/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone \
    https://huggingface.co/datasets/sunblaze-ucb/cybergym \
    "$CYBERGYM_REPO/cybergym_data"
fi
lfs_include=""
for task in "${SUBSET_TASKS[@]}"; do
  prefix="${task%%:*}"
  id="${task##*:}"
  lfs_include+="${lfs_include:+,}data/$prefix/$id/**"
done
git -C "$CYBERGYM_REPO/cybergym_data" lfs pull --include="$lfs_include"

echo "==> [5/6] Downloading CyberGym server Docker images for the subset"
(cd "$CYBERGYM_REPO" && python3 scripts/server_data/download_subset.py)

echo "==> [6/6] Installing this runner and building the agent image"
(cd "$AGENT_REPO" && python3 -m pip install -e . && docker build -t "$RUNNER_IMAGE" .)

echo
echo "==> Setup complete."
echo "    Next: start the server with scripts/start_server.sh (keep it running),"
echo "    then run tasks with scripts/run_subset.sh."
