# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository instruction discovery for interactive coding agents."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


#: Per-file and combined caps on repository instructions. The content is
#: workspace-controlled and lands in a prefix context block on every turn, so an
#: unbounded read is a memory and context-window risk at session setup.
_MAX_INSTRUCTION_FILE_CHARS = 100_000
_MAX_INSTRUCTION_TOTAL_CHARS = 200_000


def _truncate(content: str, limit: int, path: Path) -> str:
    if len(content) <= limit:
        return content
    logger.warning("Truncating repository instructions from %s at %d chars", path, limit)
    return content[:limit] + f"\n\n[... truncated at {limit} characters ...]"


def render_agent_instructions(working_directory: str | Path) -> str:
    """Render applicable repository instructions as one bounded context block."""
    sections: list[str] = []
    remaining = _MAX_INSTRUCTION_TOTAL_CHARS
    for path in discover_agent_instruction_files(working_directory):
        if remaining <= 0:
            logger.warning("Skipping repository instructions from %s: total limit reached", path)
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if not content:
            continue
        content = _truncate(content, min(_MAX_INSTRUCTION_FILE_CHARS, remaining), path)
        remaining -= len(content)
        sections.append(f"Instructions from {path}:\n\n{content}")
    return "\n\n---\n\n".join(sections)


def _git_root(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return None
