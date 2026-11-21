# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-cli's own configurable limits: :class:`CodingLimits`.

Core limits are ``nooa.config.TruncationConfig`` fields; these are the coding
agent's (its own model: nooa-cli does not depend on nooa-atom).
Read the instance in force with ``nooa.config.effective(CodingLimits)`` when the
limit is applied. Set it like any config model, naming only the fields to
change::

    class MyAgent(CodingAgent, configs=[CodingLimits(instructions_total_chars=50_000)]): ...
    agent = MyAgent(configs=[CodingLimits(instructions_file_chars=20_000)])
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["CodingLimits"]


class CodingLimits(BaseModel):
    """Char budgets of text the coding agent puts in front of the model."""

    model_config = ConfigDict(frozen=True)

    instructions_file_chars: Annotated[
        int,
        Field(
            description=(
                "Most characters read from one repository instruction file "
                "(AGENTS.md) into the repository_instructions context block."
            )
        ),
    ] = 100_000
    instructions_total_chars: Annotated[
        int,
        Field(
            description=(
                "Most characters of the whole repository_instructions context block, "
                "headers and separators included."
            )
        ),
    ] = 200_000

    @model_validator(mode="after")
    def _check(self) -> "CodingLimits":
        errors = [
            f"{name} must be > 0, got {getattr(self, name)}"
            for name in ("instructions_file_chars", "instructions_total_chars")
            if getattr(self, name) <= 0
        ]
        if errors:
            raise ValueError("Invalid CodingLimits:\n" + "\n".join(f"  - {e}" for e in errors))
        return self
