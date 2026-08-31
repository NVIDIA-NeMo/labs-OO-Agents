# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coding-agent components used by terminal and protocol hosts."""

from nooa_cli.coding.activity import (
    ActivityShellTools,
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from nooa_cli.coding.agent import CodingAgent
from nooa_cli.coding.instructions import (
    discover_agent_instruction_files,
    render_agent_instructions,
)
from nooa_cli.coding.settings import load_coding_skills_dirs
from nooa_cli.coding.slash_commands import CodingSlashCommand, CodingSlashCommandRegistry

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
