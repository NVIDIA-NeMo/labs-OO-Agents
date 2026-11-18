# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-atom's own configurable limits: :class:`AtomLimits`.

Core limits are ``nooa.config.TruncationConfig`` fields; these are nooa-atom's.
Read the instance in force with ``nooa.config.effective(AtomLimits)`` when the
limit is applied. Set it like any config model, naming only the fields to
change::

    class MyAgent(AtomAgent, configs=[AtomLimits(max_job_tail=5_000)]): ...
    agent = MyAgent(configs=[AtomLimits(instructions_file_chars=20_000)])
    @strategy(CodeActStrategy(), configs=[AtomLimits(max_job_tail=500)])
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["AtomLimits"]


class AtomLimits(BaseModel):
    """Char budgets of text nooa-atom puts in front of the model."""

    model_config = ConfigDict(frozen=True)

    max_job_tail: Annotated[
        int,
        Field(
            description=(
                "Max chars kept of the end of each of stdout and stderr on a "
                "background shell job's done update (JobsApi). Not affected by "
                "TruncationConfig.capture.tail."
            )
        ),
    ] = 20_000
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
    def _check(self) -> "AtomLimits":
        errors = [
            f"{name} must be > 0, got {getattr(self, name)}"
            for name in ("max_job_tail", "instructions_file_chars", "instructions_total_chars")
            if getattr(self, name) <= 0
        ]
        if errors:
            raise ValueError("Invalid AtomLimits:\n" + "\n".join(f"  - {e}" for e in errors))
        return self
