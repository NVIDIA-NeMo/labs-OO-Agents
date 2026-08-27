# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI compatibility names for the shared coding agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nooa_cli.coding.agent import CodingAgent


def _to_core_summarization_config(value: Any) -> Any:
    if value is None:
        return None
    from nooa.interactive import SummarizationConfig

    if isinstance(value, SummarizationConfig):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return SummarizationConfig.model_validate(dump())
    return SummarizationConfig.model_validate(value)


class TUIAgent(CodingAgent):
    """The default coding agent hosted by the native terminal UI."""

    def __init__(self, *, config: Any | None = None, **kwargs: Any) -> None:
        cwd = Path(getattr(config, "working_dir", "."))
        summarization = _to_core_summarization_config(getattr(config, "summarization", None))
        super().__init__(cwd=cwd, summarization=summarization, **kwargs)


BaseTUIAgent = CodingAgent

__all__ = ["BaseTUIAgent", "TUIAgent"]
