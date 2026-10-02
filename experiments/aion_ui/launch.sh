#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Resolve the interpreter without changing the workspace AionUi selected.
spike_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(CDPATH= cd -- "$spike_dir/../.." && pwd)"
if [[ ! -x "$repo_dir/.venv/bin/nooa-acp" ]]; then
    echo "Missing ACP environment. Run: uv sync --frozen --extra acp (in $repo_dir)" >&2
    exit 1
fi

if [[ "${1:-}" == "--demo" ]]; then
    shift
    export LITELLM_LOCAL_MODEL_COST_MAP=True
    exec "$repo_dir/.venv/bin/python" "$spike_dir/demo_agent.py" "$@"
fi

# The existing CLI owns model selection, secrets loading, and diagnostics.
exec "$repo_dir/.venv/bin/nooa-acp" "$@"
