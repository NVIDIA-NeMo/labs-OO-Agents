# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MethodWriting — opt-in capability for agents to define helper functions."""

from nooa.skill import Skill


class MethodWriting(Skill):
    """Define helpers and LLM-powered sub-calls at the top of a REPL cell.

    ## Plain helper functions
    For deterministic logic (math, filtering, formatting, data transformation).
    Define at the top of the cell and call by name:

        def celsius_to_fahrenheit(c):
            return c * 9/5 + 32

        converted = [celsius_to_fahrenheit(t) for t in temperatures]
        return_result(converted)

    ## @strategy(PredictStrategy()) — per-item generation with fan-out
    For sub-tasks requiring language understanding (classification, extraction,
    interpretation). These tasks need LLM reasoning. Decorate a standalone
    async function with an ellipsis body. ``asyncio.gather`` runs the calls
    in parallel.

    Examples:

        @strategy(PredictStrategy())
        async def detect_language(message: str) -> str:
            \"\"\"Return the ISO 639-1 code for {{message}} (e.g. 'en', 'fr', 'ja').\"\"\"
            ...

        codes = await asyncio.gather(*(detect_language(m) for m in messages))
        return_result(codes)

    ## @strategy(CodeActStrategy()) — multi-step sub-calls
    For sub-tasks needing both reasoning and multi-step computation:

        @strategy(CodeActStrategy())
        async def plan_itinerary(request: str) -> Itinerary:
            \"\"\"Build a day-by-day travel itinerary from {{request}}.\"\"\"
            ...

        result = await plan_itinerary(payload)

    ## Rules
    - Use ``...`` (ellipsis) as the body — the framework implements the call
      via LLM. The docstring IS the prompt: use ``{{param}}`` placeholders to
      interpolate argument values.

    ## No heuristics for language understanding
    Never use keyword matching, regex, or hand-written rules for tasks
    requiring language understanding (classification, extraction,
    interpretation). These tasks need LLM reasoning — delegate to a
    ``@strategy(PredictStrategy())`` standalone function.

    Load this skill:
        doc(self.writing)
    """

    pass
