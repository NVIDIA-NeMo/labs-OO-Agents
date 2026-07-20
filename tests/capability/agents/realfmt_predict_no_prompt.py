# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PredictStrategy agent for the inner-marker ablation.

Identical to ``realfmt_predict.RealFmtAgent`` except for one thing: the
class overrides ``_system_prompt`` to omit the truncation-guidance
section the framework normally adds. The ablation question is "does the
model understand the inner marker on its own?" — we don't want
performance to be confounded with the system-prompt's explicit
explanation of what `type(len=N, ...)` means.

Class name kept as `RealFmtAgent` to match the rest of the truncation
test suite.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from nooa import Agent
from nooa.decorators import strategy
from nooa.strategies import PredictStrategy
from tests.capability.agents.truncation_formats import _patch_eval_pipeline_loader

_patch_eval_pipeline_loader()


class Answer(BaseModel):
    answer: Annotated[
        int | None, Field(description="Integer answer, or None if cannot be determined")
    ]
    reason: Annotated[str, Field(description="Why you picked that answer (one or two sentences)")]


class RealFmtAgent(Agent):
    """Answer questions about the data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Intentionally omits the truncation-conventions section that the
        # base Agent docstring includes, so the ablation measures the marker's
        # intrinsic legibility.
        from nooa import Context

        self.context_manager["system_prompt"] = Context(
            expr=(
                'f"You are {type(self).__name__}, a Python agent working in an "\n'
                '"interactive session.\\n\\n## Context blocks\\n"\n'
                'f"{self.render_config.block_formatter.format_description()}\\n"'
            ),
            prefix=True,
        )

    @strategy(PredictStrategy())
    async def answer(
        self,
        data: Annotated[Any, "The data"],
        type_tag: Annotated[str, "Container shape"],
        fmt: Annotated[str, "Marker format"],
        question: Annotated[str, "The question"],
    ) -> Answer:
        """Answer the question based on the data."""
        ...
