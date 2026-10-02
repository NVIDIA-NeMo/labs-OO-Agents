# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-shot strategy for typed decision backends."""

from __future__ import annotations

import dataclasses
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
                store it in the call's ``DecisionRecord``. Applies only to
                native decision-model calls.
        """
        self.include_raw_response = include_raw_response

    @property
    def uses_decision_model(self) -> bool:
        """Prefer the agent's decision model over its chat LLM."""
        return True

    @property
    def requires_lock(self) -> bool:
        """Allow independent decision requests to run concurrently."""
        return False

    async def execute(self, runtime: RuntimeServices, call: CurrentCall) -> Any:
        """Use a decision model, or fall back to Predict for primitive results."""
        schema = compile_decision_schema(call.return_type, call.docstring)
        decision_runtime = cast(DecisionRuntimeServices, runtime)
        if not decision_runtime.has_decision_model:
            schema.require_llm_fallback_support()
            request = schema.request(call.bound_parameters())
            from nooa.strategies.predict import PredictStrategy

            answers: dict[str, Any] | None = None
            exception_type: str | None = None
            try:
                # Give the chat model the same compiled questions a decision
                # model would receive, and bound score answers to their levels.
                docstring = "\n\n".join(
                    part for part in (call.docstring, schema.fallback_guidance()) if part
                )
                fallback_call = dataclasses.replace(
                    call,
                    docstring=docstring,
                    return_type=schema.fallback_result_type(),
                )
                result = schema.restore_fallback_result(
                    await runtime.execute_nested(PredictStrategy(), fallback_call)
                )
                answers = schema.fallback_answers(result)
                return result
            except BaseException as exc:
                exception_type = type(exc).__name__
                raise
            finally:
                decision_runtime.record_decision_fallback(
                    request,
                    answers=answers,
                    success=exception_type is None,
                    exception_type=exception_type,
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
