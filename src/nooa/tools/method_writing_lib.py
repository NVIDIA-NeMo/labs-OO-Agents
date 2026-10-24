# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MethodWriting — guidance for defining helpers and LLM-powered sub-calls."""

from nooa.skill import Skill


class MethodWriting(Skill):
    """Define local helpers and LLM-powered sub-calls at the top of a REPL cell.

    Choose the smallest mechanism that fits:

    - A normal ``def`` for deterministic work: filtering, formatting, math,
      parsing a known syntax.
    - ``@strategy(PredictStrategy())`` for an independent one-shot semantic
      task; fan out independent calls with ``asyncio.gather``.
    - ``@strategy(CodeActStrategy())`` only when a subtask needs iterative
      execution or tools. Sub-calls add cost and context; prefer the current
      call for simple work.

    Generated methods are standalone top-level ``async def`` functions with an
    ellipsis body; the docstring is the prompt. Arguments are passed and
    rendered automatically, so do not interpolate them into the docstring.

    Example:

        @strategy(PredictStrategy())
        async def detect_language(message: str) -> str:
            '''Return the message's ISO 639-1 language code.'''
            ...

        codes = await asyncio.gather(*(detect_language(m) for m in messages))

    Do not replace semantic judgment with keyword matching, regex or
    hand-written scoring rules: semantic classification, extraction and
    interpretation belong in an LLM-powered call.
    """

    pass
