# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-shot strategy for typed decision backends."""

from __future__ import annotations

from typing import Any, cast

from nooa.decisions.schema import compile_decision_schema
from nooa.strategies.base import DecisionRuntimeServices, GenerationStrategy, RuntimeServices
from nooa.strategies.current_call import CurrentCall


class DecideStrategy(GenerationStrategy):
    """Compile the declared result, make one decision request, and reconstruct it."""

    def __init__(self, *, include_raw_response: bool = False) -> None:
        """Configure the strategy.

        Args:
            include_raw_response: Attach the decision API's raw response body to
                each detailed decision result (``BooleanDecision``,
                ``ChoiceDecision``, ``ScoreDecision``) as ``raw_response``, and
                store it in the call's ``DecisionRecord``.
        """
        self.include_raw_response = include_raw_response

    @property
    def uses_decision_model(self) -> bool:
        """Route this strategy's methods to a decision model instead of a chat LLM."""
        return True

    @property
    def requires_lock(self) -> bool:
        """Allow independent decision requests to run concurrently."""
        return False

    async def execute(self, runtime: RuntimeServices, call: CurrentCall) -> Any:
        """Compile the result type, call the decision model, and rebuild the result.

        The runtime resolves the decision model before execution and raises
        ``DecisionModelRequiredError`` when none is configured.
        """
        schema = compile_decision_schema(call.return_type, call.docstring)
        decision_runtime = cast(DecisionRuntimeServices, runtime)
        model = decision_runtime.decision_model
        if not getattr(model, "provides_probabilities", True):
            required = [output.name for output in schema.outputs if output.requires_probabilities]
            if required:
                from nooa.decisions.types import DecisionModelRequiredError

                names = ", ".join(repr(name) for name in required)
                raise DecisionModelRequiredError(
                    f"'{call.method_name}' needs probabilities for output(s) {names}: "
                    "detailed decision results and Threshold require a decision model "
                    f"that provides them, but {type(model).__name__} "
                    f"({getattr(model, 'model', '')!r}) does not."
                )
        context, events = await decision_runtime.decision_state_inputs()
        request = schema.request(
            call.bound_parameters(),
            context=context,
            events=events,
        )
        response = await decision_runtime.decide(
            request, include_raw_response=self.include_raw_response
        )
        return schema.reconstruct(response, include_raw_response=self.include_raw_response)


__all__ = ["DecideStrategy"]
