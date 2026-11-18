# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The system prompt is the cleaned class docstring on every Python version.

Python 3.13+ strips docstring indentation at compile time; 3.12 keeps it.
``_resolve_system_prompt`` cleans the docstring so both give the same prompt.
"""

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class _Indented(Agent):
    """First line.

    Second paragraph, indented in the source.
        Nested line keeps its relative indent.
    """


def test_system_prompt_is_the_cleaned_class_docstring():
    agent = _Indented(llm=FakeLLMClient([]))
    assert agent._resolve_system_prompt() == (
        "First line.\n\nSecond paragraph, indented in the source.\n"
        "    Nested line keeps its relative indent."
    )


def test_raw_indented_docstring_is_cleaned():
    """A docstring set at runtime keeps its indentation even on 3.13."""

    class Raw(Agent):
        pass

    Raw.__doc__ = "Top.\n\n    Body {1 + 1}.\n    "
    assert Raw(llm=FakeLLMClient([]))._resolve_system_prompt() == "Top.\n\nBody 2."
