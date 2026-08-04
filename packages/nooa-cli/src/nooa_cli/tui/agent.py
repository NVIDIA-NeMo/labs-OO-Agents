# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI compatibility names for the shared coding agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nooa_cli.coding.agent import CodingAgent


class TUIAgent(CodingAgent):
    """The default coding agent hosted by the native terminal UI."""

    def __init__(self, *, config: Any | None = None, **kwargs: Any) -> None:
        cwd = Path(getattr(config, "working_dir", "."))
        summarization = getattr(config, "summarization", None)
        super().__init__(cwd=cwd, summarization=summarization, **kwargs)


BaseTUIAgent = CodingAgent

__all__ = ["BaseTUIAgent", "TUIAgent"]
