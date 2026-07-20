# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON Question Answering Agent - tests semantic understanding of structured data."""

import json  # noqa: F401 — for LLM exec_globals
from typing import Annotated, Literal

from nooa import Agent


class JsonQAAgent(Agent):
    """You are an expert for extracting information from JSON."""

    async def answer_question(
        self,
        json_data: Annotated[str, "The JSON data to answer the question about"],
        question: Annotated[str, "The question to answer"],
    ) -> Literal["yes", "no"]:
        """Answer a natural language question about the provided JSON data."""
        ...
