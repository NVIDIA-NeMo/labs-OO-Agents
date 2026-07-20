# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for nooa CLI commands.

Import from here instead of duplicating helpers across command modules:

    from nooa_cli._common import find_project_root, format_size
"""

from pathlib import Path


def find_project_root() -> Path:
    """Find the user's project root by walking up from the current directory.

    Starts at :func:`Path.cwd` — the directory the CLI was invoked from — and
    walks upward, returning the first ancestor (including the cwd itself) that
    contains a ``pyproject.toml``. Falls back to the current working directory
    if none is found.

    Walking up from the cwd (rather than from this module's ``__file__``) ensures
    the *user's* project is resolved. Resolving from ``__file__`` would instead
    find the installed CLI package's own ``pyproject.toml`` in an editable or
    monorepo install, which is never what a user of ``nooa`` wants.

    Uses a local implementation to keep CLI startup fast (avoids importing
    the heavy nooa core package at module level).
    """
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cwd


def format_size(size_bytes: int) -> str:
    """Format a byte count as human-readable (e.g. '4.2 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
