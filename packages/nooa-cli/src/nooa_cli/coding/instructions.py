# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository instruction discovery for interactive coding agents."""

from __future__ import annotations

from pathlib import Path


def discover_agent_instruction_files(working_directory: str | Path) -> tuple[Path, ...]:
    """Return applicable ``AGENTS.md`` files from repository root to cwd.

    A file in a deeper directory is appended after its parent instruction file,
    so the resulting context naturally gives the most local instructions the
    final word. Discovery stops at the nearest Git worktree root. Outside a Git
    worktree, only the working directory itself is considered.
    """
    cwd = Path(working_directory).resolve()
    root = _git_root(cwd)
    directories = [cwd]
    if root is not None:
        distance = len(cwd.relative_to(root).parts)
        directories = list(reversed((cwd, *cwd.parents[:distance])))

    return tuple(path for directory in directories if (path := directory / "AGENTS.md").is_file())


def render_agent_instructions(working_directory: str | Path) -> str:
    """Render applicable repository instructions as one context block."""
    sections: list[str] = []
    for path in discover_agent_instruction_files(working_directory):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if content:
            sections.append(f"Instructions from {path}:\n\n{content}")
    return "\n\n---\n\n".join(sections)


def _git_root(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return None
