#!/usr/bin/env bash
# TB1 Reliable Evaluation Monitor
# Cron: */5 * * * * /localhome/local-rcabral/nooa/util/eval_pipeline/tb1_monitor.sh
set -euo pipefail

# ── Source environment ────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="/localhome/local-rcabral/nooa"
# Source API keys from .env
set -a
source "$REPO_ROOT/.env"
set +a
export NEMO_OO_AGENTS_GIT_REF="${NEMO_OO_AGENTS_GIT_REF:-main}"
export OPENAI_API_KEY="$NVIDIA_INTERNAL_API_KEY"

# ── Config ────────────────────────────────────────────────────────────────────
JOBS_DIR="/localhome/local-rcabral/harbor_jobs/terminal_bench_baseline"
TASKS_CACHE="/localhome/local-rcabral/harbor_tasks_cache/terminal_bench"
SIF_CACHE="$HOME/3p/sif_cache"
BOOTSTRAP_OVERLAY="$HOME/3p/harbor_bootstrap_overlay"
LOG_FILE="$HOME/.tb1_eval_monitor.log"
LOCK_FILE="$HOME/.tb1_eval.lock"
EXPECTED_TASKS=241
N_CONCURRENT=2
MEMORY_MB=4096
STALE_MINUTES=45

log() { echo "[$(date -Iseconds)] $*" >> "$LOG_FILE"; }

# ── Prevent multiple instances ────────────────────────────────────────────────
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        log "Another monitor instance running (PID=$LOCK_PID) — exiting."
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi

# ── Step 1: Ensure tasks are cached ──────────────────────────────────────────
if [ ! -d "$TASKS_CACHE" ] || [ "$(ls "$TASKS_CACHE" 2>/dev/null | wc -l)" -lt "$EXPECTED_TASKS" ]; then
    log "Generating TB1 tasks..."
    cd "$REPO_ROOT"
    python util/harbor/generate_terminal_bench_tasks.py >> "$LOG_FILE" 2>&1
    mkdir -p "$TASKS_CACHE"
    rsync -a --delete util/harbor/tasks/terminal_bench/ "$TASKS_CACHE/"
    log "Cached $(ls "$TASKS_CACHE" | wc -l) tasks"
else
    log "Task cache OK: $(ls "$TASKS_CACHE" | wc -l) tasks"
fi

# ── Step 2: Ensure agent pre-cached ──────────────────────────────────────────
AGENT_CACHE="$BOOTSTRAP_OVERLAY/installed-agent/nooa"
if [ ! -d "$AGENT_CACHE/.git" ]; then
    log "Pre-caching agent..."
    mkdir -p "$(dirname "$AGENT_CACHE")"
    git clone --depth 1 --branch "$NEMO_OO_AGENTS_GIT_REF" \
        "$NEMO_OO_AGENTS_GIT_URL" "$AGENT_CACHE" >> "$LOG_FILE" 2>&1
    log "Agent cached"
fi

# ── Step 3: Check latest run status ──────────────────────────────────────────
mkdir -p "$JOBS_DIR"
LATEST_RUN=$(find "$JOBS_DIR" -maxdepth 1 -type d -name '2026*' | sort | tail -1)

if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/result.json" ]; then
    STATUS=$(python3 -c "
import json, os, time
d = json.load(open('$LATEST_RUN/result.json'))
finished = d.get('finished_at')
total = d.get('n_total_trials', 0)
s = d.get('stats', {})
done = s.get('n_trials', 0)
errors = s.get('n_errors', 0)
mean = 0
for v in s.get('evals', {}).values():
    m = v.get('metrics', [{}])
    if m: mean = m[0].get('mean', 0)

if finished:
    if done >= total and done >= $EXPECTED_TASKS and errors < total * 0.5:
        print(f'COMPLETE done={done} total={total} mean={mean:.3f} errors={errors}')
    else:
        print(f'NEEDS_RETRY done={done} total={total} mean={mean:.3f} errors={errors}')
else:
    # Check if stale (no modification in STALE_MINUTES)
    mtime = os.path.getmtime('$LATEST_RUN/result.json')
    age_min = (time.time() - mtime) / 60
    if age_min > $STALE_MINUTES:
        # Mark as finished (stale)
        d['finished_at'] = '2026-01-01T00:00:00'
        json.dump(d, open('$LATEST_RUN/result.json', 'w'), indent=2)
        print(f'STALE done={done} total={total} age={age_min:.0f}min')
    else:
        print(f'RUNNING done={done} total={total}')
")
    log "Status: $STATUS"

    if echo "$STATUS" | grep -q "^RUNNING"; then
        exit 0
    fi

    if echo "$STATUS" | grep -q "^COMPLETE"; then
        log "✅ TB1 baseline COMPLETE!"
        cd "$REPO_ROOT"
        uv run wtf comment gl-78 "TB1 baseline complete: $STATUS. Results: $LATEST_RUN/result.json" 2>/dev/null || true
        exit 0
    fi
fi

# ── Step 4: Check if harbor is already running ────────────────────────────────
if pgrep -f "harbor run.*terminal_bench" > /dev/null 2>&1; then
    log "Harbor process found — waiting."
    exit 0
fi

# ── Step 5: Launch evaluation ─────────────────────────────────────────────────
log "Launching TB1 baseline ($EXPECTED_TASKS tasks, $N_CONCURRENT concurrent, ${MEMORY_MB}MB)..."
echo $$ > "$LOCK_FILE"

cd "$REPO_ROOT"
nohup harbor run \
    --path "$TASKS_CACHE" \
    --jobs-dir "$JOBS_DIR" \
    -a nemo-oo-agents \
    -m aws/anthropic/bedrock-claude-sonnet-4-5-v1 \
    -e apptainer \
    --override-memory-mb $MEMORY_MB \
    --ek apptainer_image_cache_dir="$SIF_CACHE" \
    --ek apptainer_fakeroot=true \
    --ek apptainer_bootstrap_overlay="$BOOTSTRAP_OVERLAY" \
    --ak agent_type=baseline \
    --ae OPENAI_API_KEY="$NVIDIA_INTERNAL_API_KEY" \
    --ae NVIDIA_INTERNAL_API_KEY="$NVIDIA_INTERNAL_API_KEY" \
    --ae NEMO_OO_AGENTS_GIT_URL="$NEMO_OO_AGENTS_GIT_URL" \
    --ae NEMO_OO_AGENTS_GIT_REF="$NEMO_OO_AGENTS_GIT_REF" \
    -n $N_CONCURRENT \
    --yes \
    >> "$LOG_FILE" 2>&1 &

HARBOR_PID=$!
echo "$HARBOR_PID" > "$LOCK_FILE"
log "Harbor started (PID=$HARBOR_PID)"
rm -f "$LOCK_FILE"

