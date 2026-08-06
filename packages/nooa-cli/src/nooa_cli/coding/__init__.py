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

__all__ = [
    "ActivityShellTools",
    "CodingAgent",
    "FileEdit",
    "TerminalCommandFinished",
    "TerminalCommandOutput",
    "TerminalCommandStarted",
    "discover_agent_instruction_files",
    "render_agent_instructions",
]
