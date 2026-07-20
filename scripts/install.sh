#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# nooa TUI installer — designed for `curl ... | sh` one-liner UX.
#
# What it does:
#   1. Installs `uv` if not already present.
#   2. Installs a supported Python interpreter (3.12 or 3.13).
#   3. Installs `nemo-labs-oo-agents-cli` as a `uv tool`, alongside
#      `nemo-oo-agents` (core) and `nemo-oo-agents-nvidia` (NVIDIA-gateway
#      model aliases registered via the
#      `nooa.bundled_configs` entry-point group).
#   4. Optionally prompts for `NVIDIA_INFERENCE_API_KEY` (via /dev/tty) and
#      writes it to `~/.config/nooa/secrets.yaml` (chmod 600). The CLI
#      loads this file into the process env automatically on every run —
#      no shell-rc edit needed.
#
# Skip the API-key prompt by setting `NEMO_OO_INSTALL_NONINTERACTIVE=1`
# or by piping into a non-TTY environment (e.g. CI).
#
# Force-reinstall everything with `NEMO_OO_INSTALL_REINSTALL=1`.

set -eu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_URL="${NEMO_OO_INSTALL_REPO:-https://github.com/NVIDIA-NeMo/labs-OO-Agents.git}"
REPO_REF="${NEMO_OO_INSTALL_REF:-main}"
SECRETS_FILE="${NEMO_OO_INSTALL_SECRETS_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/nooa/secrets.yaml}"

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

bold() { printf '\033[1m%s\033[0m' "$1"; }
green() { printf '\033[32m%s\033[0m' "$1"; }
yellow() { printf '\033[33m%s\033[0m' "$1"; }
red() { printf '\033[31m%s\033[0m' "$1"; }

info() { printf '%s %s\n' "$(green '==>')" "$1"; }
warn() { printf '%s %s\n' "$(yellow '!!!')" "$1" >&2; }
die() { printf '%s %s\n' "$(red 'ERROR:')" "$1" >&2; exit 1; }

# Fail early with a clear message if a required command is missing (pattern
# borrowed from Astral's official `uv` installer).
check_cmd() { command -v "$1" >/dev/null 2>&1; }
need_cmd() { check_cmd "$1" || die "required command not found: $1"; }

require_prereqs() {
    # `uv` is installed by this script; everything else must already exist.
    for cmd in curl sed chmod mkdir mv dirname date; do
        need_cmd "$cmd"
    done
}

# ---------------------------------------------------------------------------
# Step 1: uv
# ---------------------------------------------------------------------------

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        info "uv already installed ($(uv --version 2>/dev/null | head -1))"
        return 0
    fi
    info "Installing uv (Astral) — running the official installer..."
    # The official uv installer puts the binary in ~/.local/bin and adds
    # PATH lines to common shell rc files.
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Make uv visible for the rest of this script even before the user
    # opens a new shell.
    if [ -x "$HOME/.local/bin/uv" ]; then
        PATH="$HOME/.local/bin:$PATH"
        export PATH
    fi
    if ! command -v uv >/dev/null 2>&1; then
        die "uv install reported success but \`uv\` is still not on PATH. \
Open a new shell and re-run this installer, or add \`\$HOME/.local/bin\` to PATH."
    fi
}

# ---------------------------------------------------------------------------
# Step 2: managed Python
# ---------------------------------------------------------------------------

# Set when we detect a supported system Python (3.12 or 3.13) we can reuse, so that
# `uv tool install` is pinned to it and uv does not try to download one
# (which requires GitHub access).
SYSTEM_PYTHON=""

find_system_python() {
    # Probe common interpreter names for a supported Python version.
    for py in python3.13 python3.12 python3 python; do
        if ! command -v "$py" >/dev/null 2>&1; then
            continue
        fi
        ver=$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
        case "$ver" in
            3.12|3.13)
                SYSTEM_PYTHON="$(command -v "$py")"
                return 0
                ;;
        esac
    done
    return 1
}

ensure_python() {
    if find_system_python; then
        ver=$("$SYSTEM_PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)
        info "Reusing existing Python $ver at $SYSTEM_PYTHON (skipping uv python install)"
        return 0
    fi
    info "Ensuring uv-managed Python 3.12 is available..."
    uv python install 3.12
}

# ---------------------------------------------------------------------------
# Step 3: nemo-labs-oo-agents-cli (+ core + nvidia)
# ---------------------------------------------------------------------------

install_tool() {
    if [ "${NEMO_OO_INSTALL_REINSTALL:-0}" = "1" ]; then
        info "Force-reinstalling nemo-labs-oo-agents-cli..."
        UV_INSTALL_FLAGS="--reinstall"
    else
        UV_INSTALL_FLAGS=""
    fi

    # When we found a usable system Python, pin uv to it and disable
    # uv's Python downloads so the install works in sandboxes without
    # GitHub access.
    if [ -n "$SYSTEM_PYTHON" ]; then
        UV_PYTHON_FLAGS="--python $SYSTEM_PYTHON"
        UV_PYTHON_DOWNLOADS=never
        export UV_PYTHON_DOWNLOADS
    else
        UV_PYTHON_FLAGS=""
    fi

    info "Installing nemo-labs-oo-agents-cli (with core + NVIDIA aliases) as a uv tool..."
    uv tool install $UV_INSTALL_FLAGS $UV_PYTHON_FLAGS \
        "nemo-labs-oo-agents-cli @ git+${REPO_URL}@${REPO_REF}#subdirectory=packages/nemo-labs-oo-agents-cli" \
        --with "nemo-oo-agents @ git+${REPO_URL}@${REPO_REF}" \
        --with "nemo-oo-agents-nvidia @ git+${REPO_URL}@${REPO_REF}#subdirectory=packages/nemo-oo-agents-nvidia"
}

# ---------------------------------------------------------------------------
# Step 4: API key prompt
# ---------------------------------------------------------------------------

prompt_api_key() {
    if [ "${NEMO_OO_INSTALL_NONINTERACTIVE:-0}" = "1" ]; then
        info "Skipping API-key prompt (NEMO_OO_INSTALL_NONINTERACTIVE=1)."
        return 0
    fi

    # When piped through `| sh`, stdin is the script — read from /dev/tty
    # instead so the prompt actually reaches the user's terminal.
    if [ ! -r /dev/tty ]; then
        info "No /dev/tty available (CI / non-interactive shell). Skipping API-key prompt."
        return 0
    fi

    printf '\n'
    info "$(bold 'NVIDIA Inference HUB API key')"
    printf '    Most bundled aliases (claude-*, nemotron-*, gpt-5.x, …) route\n'
    printf '    through inference.nvidia.com and need NVIDIA_INFERENCE_API_KEY.\n'
    printf '    Get a key at https://inference.nvidia.com — or press Enter to skip.\n'
    printf '\n'
    printf '    NVIDIA_INFERENCE_API_KEY: '
    # `< /dev/tty` is the cross-shell way to read from the user even when
    # stdin is the install script.
    api_key=""
    read -r api_key < /dev/tty || true

    if [ -z "$api_key" ]; then
        info "No API key entered — you can set NVIDIA_INFERENCE_API_KEY later."
        return 0
    fi

    mkdir -p "$(dirname "$SECRETS_FILE")"
    # Emit the key as a double-quoted YAML scalar so values containing ':',
    # '#', leading '{'/'[', etc. can't malform the file. Inside double quotes
    # only '\' and '"' need escaping.
    escaped_key=$(printf '%s' "$api_key" | sed 's/\\/\\\\/g; s/"/\\"/g')
    # Write atomically so a concurrent installer can't read a half-written file.
    # The CLI reads this YAML and pushes `env:` into the process env on every
    # run (non-clobbering) — no shell-rc edit needed.
    tmp_file="${SECRETS_FILE}.tmp.$$"
    {
        printf '# Written by nooa install.sh on %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '# Loaded automatically by the nooa CLI (non-clobbering — a\n'
        printf '# value already exported in your shell wins). Add more keys under env:.\n'
        printf 'env:\n'
        printf '  NVIDIA_INFERENCE_API_KEY: "%s"\n' "$escaped_key"
    } > "$tmp_file"
    chmod 600 "$tmp_file"
    mv "$tmp_file" "$SECRETS_FILE"
    info "Wrote NVIDIA_INFERENCE_API_KEY to $SECRETS_FILE (mode 600)."
}

# ---------------------------------------------------------------------------
# Step 5: final instructions
# ---------------------------------------------------------------------------

print_next_steps() {
    printf '\n'
    info "$(green 'Install complete.')"
    printf '\n'

    if [ -f "$SECRETS_FILE" ]; then
        printf '%s\n' "$(bold 'Next steps:')"
        printf '  1. Open a new shell so the `~/.local/bin` PATH change takes effect.\n'
        printf '  2. Run the TUI — your API key is loaded automatically from\n'
        printf '     %s:\n' "$SECRETS_FILE"
        printf '\n'
        printf '       %s\n' "$(bold 'nooa tui')"
        printf '\n'
    else
        printf '%s\n' "$(bold 'Next steps:')"
        printf '  1. Open a new shell so the PATH change picks up.\n'
        printf '  2. Add your API key so it loads automatically on every run.\n'
        printf '     Create %s with:\n' "$SECRETS_FILE"
        printf '\n'
        printf '       env:\n'
        printf '         NVIDIA_INFERENCE_API_KEY: sk-…\n'
        printf '\n'
        printf '     (or just `export NVIDIA_INFERENCE_API_KEY=sk-…` in your shell)\n'
        printf '\n'
        printf '  3. Run the TUI:\n'
        printf '\n'
        printf '       %s\n' "$(bold 'nooa tui')"
        printf '\n'
    fi

    printf '%s\n' "$(bold 'Inspect / customize the bundled LLM aliases:')"
    printf '       nooa config show     # which YAMLs are loading\n'
    printf '       nooa config eject    # local copy at ~/.config/nooa/llm_config.yaml\n'
    printf '\n'
    printf '%s\n' "$(bold 'Upgrade later:')"
    printf '       uv tool upgrade nemo-labs-oo-agents-cli\n'
    printf '\n'
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
    info "nooa TUI installer — ref=${REPO_REF}"
    require_prereqs
    ensure_uv
    ensure_python
    install_tool
    prompt_api_key
    print_next_steps
}

main "$@"
