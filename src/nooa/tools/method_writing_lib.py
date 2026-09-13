# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MethodWriting — guidance for defining helpers and LLM-powered sub-calls."""

from nooa.skill import Skill


class MethodWriting(Skill):
    """Define local helpers and LLM-powered sub-calls at the top of a REPL cell.

    Choose the smallest mechanism that fits the work:

    - Use a normal ``def`` for deterministic computation such as filtering,
      formatting, math, or parsing a known syntax.
    - Use ``PredictStrategy`` for an independent, one-shot semantic task. Fan out
      independent calls concurrently with ``asyncio.gather``.
    - Use ``CodeActStrategy`` only when a subtask needs iterative execution or
      tools. Prefer the current call for simple work; sub-calls add cost and context.

    Examples:

    Deterministic helper:

        def celsius_to_fahrenheit(value: float) -> float:
            return value * 9 / 5 + 32

        converted = [celsius_to_fahrenheit(value) for value in temperatures]

    Example — concurrent one-shot semantic tasks:

        @strategy(PredictStrategy())
        async def detect_language(message: str) -> str:
            '''Return the message's ISO 639-1 language code.'''
            ...

        codes = await asyncio.gather(*(detect_language(message) for message in messages))

    Example — an iterative sub-call:

        @strategy(CodeActStrategy())
        async def investigate_failure(log_excerpt: str) -> str:
            '''Investigate the failure and return the most likely root cause.'''
            ...

        diagnosis = await investigate_failure(log_text)

    Define generated methods as standalone top-level ``async def`` functions with
    an ellipsis body. Their docstring is the prompt. Arguments are passed and
    rendered automatically, so do not interpolate parameter values into docstrings.

    Do not replace semantic judgment with keyword matching, regex, or hand-written
    scoring rules. Deterministic parsing of a known syntax is appropriate; semantic
    classification, extraction, and interpretation belong in an LLM-powered call.
    """

    pass
