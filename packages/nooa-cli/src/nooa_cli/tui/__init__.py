# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native terminal interface for NVIDIA Labs Object Oriented Agents (NOOA)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import BaseTUIAgent, TUIAgent
    from .config import AgentConfig, Config, SummarizationConfig, TUIConfig
    from .console import TUIConsole
    from .input_handler import TUIInputHandler
    from .theme import CATPPUCCIN_THEME, COLORS

_EXPORTS = {
    "BaseTUIAgent": ".agent",
    "TUIAgent": ".agent",
    "AgentConfig": ".config",
    "Config": ".config",
    "SummarizationConfig": ".config",
    "TUIConfig": ".config",
    "TUIConsole": ".console",
    "TUIInputHandler": ".input_handler",
    "CATPPUCCIN_THEME": ".theme",
    "COLORS": ".theme",
}


def __getattr__(name: str):
    """Lazy package exports keep ``import nooa_cli.tui.config`` lightweight."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "AgentConfig",
    "BaseTUIAgent",
    "Config",
    "SummarizationConfig",
    "TUIAgent",
    "TUIConfig",
    "TUIConsole",
    "TUIInputHandler",
    "CATPPUCCIN_THEME",
    "COLORS",
]
