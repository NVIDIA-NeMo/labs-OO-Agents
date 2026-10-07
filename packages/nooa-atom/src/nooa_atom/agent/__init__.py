# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Atom agent and the components terminal and protocol hosts share with it."""

from importlib import import_module

_EXPORT_MODULES = {
    "ActivityShellTools": "activity",
    "FileEdit": "activity",
    "TerminalCommandFinished": "activity",
    "TerminalCommandOutput": "activity",
    "TerminalCommandStarted": "activity",
    "AtomAgent": "agent",
    "ExperimentalAtomAgent": "experimental_agent",
    "discover_agent_instruction_files": "instructions",
    "render_agent_instructions": "instructions",
    "load_skills_dirs": "nooa_atom.workspace.settings",
    "SlashCommand": "slash_commands",
    "SlashCommandRegistry": "slash_commands",
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
    "AtomAgent",
    "ExperimentalAtomAgent",
    "SlashCommand",
    "SlashCommandRegistry",
    "FileEdit",
    "TerminalCommandFinished",
    "TerminalCommandOutput",
    "TerminalCommandStarted",
    "discover_agent_instruction_files",
    "load_skills_dirs",
    "render_agent_instructions",
]
