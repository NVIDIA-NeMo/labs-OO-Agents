#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run inside WSL 2 by .github/workflows/wsl.yml, as root, from the Windows checkout.
# Usage: bash scripts/wsl_ci.sh quick|full
set -euo pipefail
mode="${1:?usage: wsl_ci.sh quick|full}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates git rsync ripgrep >/dev/null
# Pinned uv release, verified against its published SHA256 before use (no curl | sh).
UV_VERSION=0.9.26
UV_SHA256=30ccbf0a66dc8727a02b0e245c583ee970bdafecf3a443c1686e1b30ec4939e8
uv_tgz="$(mktemp -d)/uv.tar.gz"
curl -LsSf -o "$uv_tgz" "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
echo "${UV_SHA256}  ${uv_tgz}" | sha256sum -c -
tar -xzf "$uv_tgz" -C "$(dirname "$uv_tgz")"
mkdir -p "$HOME/.local/bin"
install -m 0755 "$(dirname "$uv_tgz")/uv-x86_64-unknown-linux-gnu/uv" "$(dirname "$uv_tgz")/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/"
export PATH="$HOME/.local/bin:$PATH"
uv --version

# Work on the Linux filesystem: /mnt/c is slow and ignores chmod.
rsync -a --delete --exclude .venv ./ "$HOME/repo/"
cd "$HOME/repo"
git config --global --add safe.directory "$HOME/repo"
uname -a

uv sync --frozen --all-extras --no-extra sandbox
uv run --frozen python scripts/windows_smoke.py
case "$mode" in
  quick)
    uv run --frozen pytest -q tests/test_windows_platform.py tests/tools/test_bash_session.py \
      tests/storage tests/runtime/test_producers.py
    ;;
  full)
    uv run --frozen pytest -q -m "not integration and not stress and not sandbox"
    uv run --frozen pytest -q -m sandbox
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
