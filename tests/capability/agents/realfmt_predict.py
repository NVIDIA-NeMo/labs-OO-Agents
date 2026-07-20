# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PredictStrategy variant of the realfmt format-comparison agent.

Class name `RealFmtAgent` and the method docstring are intentionally
identical to the ones in ``realfmt_codeact.py``. Tests pick which one to
run via the ``module:`` / ``class:`` fields in ``config_truncation.yaml``.

The agent receives ``data`` as a ``Wrapped`` instance (see
``truncation_formats.Wrapped``) — the fixture supplies raw Python data
plus ``type_tag`` and ``fmt`` kwargs, and the loader monkey-patch
auto-wraps them. The framework then uses ``Wrapped.__repr__`` (i.e. our
chosen marker format) when rendering the parameter dump in the prefill.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

from nooa import Agent
from nooa.decorators import strategy
from nooa.strategies import PredictStrategy
from tests.capability.agents.truncation_formats import _patch_eval_pipeline_loader

_patch_eval_pipeline_loader()


class Answer(BaseModel):
    is_answerable: Annotated[
        bool,
        Field(
            description="True if the answer can be determined from the data provided, False if it cannot be determined from the data"
        ),
    ]
    answer: Annotated[
        int | bool | None,
        Field(
            description="Answer value: use an integer for numeric questions, a boolean for yes/no membership questions, or None if it cannot be determined from the data"
        ),
    ]
    reason: Annotated[
        str,
        Field(
            description="Why you picked that answer, or why it cannot be determined (one or two sentences)"
        ),
    ]

    @field_validator("answer")
    @classmethod
    def answer_consistent_with_answerable(cls, v, info):
        """Enforce: if is_answerable=False, answer must be None."""
        if info.data.get("is_answerable") is False and v is not None:
            raise ValueError("answer must be None when is_answerable is False")
        return v


class RealFmtAgent(Agent):
    """Answer questions about the data."""

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
