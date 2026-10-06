# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coding-agent components used by terminal and protocol hosts."""

from importlib import import_module

_EXPORT_MODULES = {
    "ActivityShellTools": "activity",
    "FileEdit": "activity",
    "TerminalCommandFinished": "activity",
    "TerminalCommandOutput": "activity",
    "TerminalCommandStarted": "activity",
    "CodingAgent": "agent",
    "discover_agent_instruction_files": "instructions",
    "render_agent_instructions": "instructions",
    "load_coding_skills_dirs": "nooa_coder.workspace.settings",
    "CodingSlashCommand": "slash_commands",
    "CodingSlashCommandRegistry": "slash_commands",
}


def __getattr__(name: str):
    """Load host components only when requested, not for leaf utility imports."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _EXPORT_MODULES[name]
    module = module if "." in module else f"{__name__}.{module}"
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "ActivityShellTools",
    "CodingAgent",
    "CodingSlashCommand",
    "CodingSlashCommandRegistry",
    "FileEdit",
    "TerminalCommandFinished",
    "TerminalCommandOutput",
    "TerminalCommandStarted",
    "discover_agent_instruction_files",
    "load_coding_skills_dirs",
    "render_agent_instructions",
]
