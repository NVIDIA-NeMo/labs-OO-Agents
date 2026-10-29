#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run inside WSL 2 by .github/workflows/wsl.yml, from the Windows checkout.
# As root it installs system packages, copies the checkout to a normal user's
# home and re-runs itself as that user; the tests never run as root (root
# bypasses file-permission checks).
# Usage: bash scripts/wsl_ci.sh quick|full
set -euo pipefail
mode="${1:?usage: wsl_ci.sh quick|full}"
case "$mode" in quick | full) ;; *)
  echo "unknown mode: $mode" >&2
  exit 2
  ;;
esac

if [ "$(id -u)" = 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates git rsync ripgrep >/dev/null
  id ci >/dev/null 2>&1 || useradd --create-home --shell /bin/bash ci
  # Work on the Linux filesystem: /mnt/c is slow and ignores chmod.
  rsync -a --delete --exclude .venv ./ /home/ci/repo/
  chown -R ci:ci /home/ci/repo
  exec runuser -u ci -- env HOME=/home/ci USER=ci bash /home/ci/repo/scripts/wsl_ci.sh "$mode"
fi

# Pinned uv release, verified against its published SHA256 before use (no curl | sh).
UV_VERSION=0.9.26
UV_SHA256=30ccbf0a66dc8727a02b0e245c583ee970bdafecf3a443c1686e1b30ec4939e8
uv_dir="$(mktemp -d)"
curl -LsSf -o "$uv_dir/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
echo "${UV_SHA256}  $uv_dir/uv.tar.gz" | sha256sum -c -
tar -xzf "$uv_dir/uv.tar.gz" -C "$uv_dir"
mkdir -p "$HOME/.local/bin"
install -m 0755 "$uv_dir/uv-x86_64-unknown-linux-gnu/uv" "$uv_dir/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/"
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/repo"
uname -a
id
uv --version

uv sync --frozen --all-extras --no-extra sandbox
uv run --frozen python scripts/windows_smoke.py
if [ "$mode" = quick ]; then
  uv run --frozen pytest -q tests/test_windows_platform.py tests/tools/test_bash_session.py \
    tests/storage tests/runtime/test_producers.py
else
  uv run --frozen pytest -q -m "not integration and not stress and not sandbox"
  uv run --frozen pytest -q -m sandbox
fi
