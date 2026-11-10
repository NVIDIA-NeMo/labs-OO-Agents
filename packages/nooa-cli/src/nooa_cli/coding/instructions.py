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
_SECTION_SEPARATOR = "\n\n---\n\n"


def render_agent_instructions(working_directory: str | Path) -> str:
    """Render applicable repository instructions as one bounded context block.

    Reads are bounded rather than truncated after the fact: the content is
    workspace-controlled and lands in a prefix context block on every turn, so
    reading a huge file in full before discarding most of it would still cost
    the memory. The budget covers the rendered text — headers, separators and
    truncation markers included — not just the retained file content.
    """
    sections: list[str] = []
    remaining = _MAX_INSTRUCTION_TOTAL_CHARS
    for path in discover_agent_instruction_files(working_directory):
        if remaining <= 0:
            logger.warning("Skipping repository instructions from %s: total limit reached", path)
            continue
        budget = min(_MAX_INSTRUCTION_FILE_CHARS, remaining)
        try:
            with path.open("r", encoding="utf-8") as stream:
                # One char past the budget is enough to know it was cut.
                content = stream.read(budget + 1)
        except (OSError, UnicodeError):
            continue
        truncated = len(content) > budget
        if truncated:
            content = content[:budget]
            logger.warning("Truncating repository instructions from %s at %d chars", path, budget)
        content = content.strip()
        if not content:
            continue
        if truncated:
            content += "\n\n[... truncated ...]"
        section = f"Instructions from {path}:\n\n{content}"
        remaining -= len(section) + len(_SECTION_SEPARATOR)
        sections.append(section)
    return _SECTION_SEPARATOR.join(sections)


def _git_root(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return None
