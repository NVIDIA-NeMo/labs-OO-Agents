# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chat-model adapter behind ``DecisionModel.from_llm``."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError, create_model

from nooa.decisions.client import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionRequest,
    DecisionResponse,
    InvalidDecisionResponseError,
    ScoreAnswer,
)

if TYPE_CHECKING:
    from nooa.unifiedllm.unifiedllm import UnifiedLLM

_SYSTEM_PROMPT = """You answer typed decision questions about a state.

You receive a JSON object with a "state" and "questions". Answer every question
using only the state and the question's instructions and criteria:
- "noul": answer true or false.
- "choice": answer with exactly one of the option IDs in "criteria".
- "score": answer with the index of the best-matching level in "criteria",
  counting from 0.

Return only the JSON object requested by the response schema."""


class LLMDecisionModel(DecisionModel):
    """Answer decision requests with structured output from a chat model.

    Each answer is a single selection, so it carries no probability evidence.
    It is returned as a one-hot distribution, with confidence 1.0, so that
    primitive results can be reconstructed. Records identify these calls with
    ``decision_source="llm"``.
    """

    provides_probabilities: ClassVar[bool] = False
    decision_source: ClassVar[Literal["native", "llm"]] = "llm"

    def __init__(self, llm: UnifiedLLM, *, max_attempts: int = 2) -> None:
        """Wrap a chat model.

        Args:
            llm: Chat client used for every decision request.
            max_attempts: Calls allowed per request when the reply does not
                match the response schema.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm = llm
        self.model = getattr(llm, "model", "") or ""
        self.max_attempts = max_attempts

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        """Ask the chat model to answer every question in one structured reply.

        Raises:
            InvalidDecisionResponseError: If no reply matches the response schema.
        """
        output_model = _answer_model(request)
        messages: list[Any] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "state": request.state,
                        "questions": {
                            name: asdict(question) for name, question in request.questions.items()
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        error: Exception | None = None
        for _ in range(self.max_attempts):
            response = await self.llm.acall(messages, output_model=output_model)
            try:
                parsed = _parse_reply(response, output_model)
            except (ValidationError, ValueError) as exc:
                error = exc
                continue
            usage = response.usage.model_dump() if response.usage is not None else {}
            return DecisionResponse(
                answers={
                    name: _answer(question, getattr(parsed, name))
                    for name, question in request.questions.items()
                },
                model=getattr(response, "model", None) or self.model,
                usage=usage,
            )
        raise InvalidDecisionResponseError(
            f"Chat model {self.model!r} did not return valid decision answers "
            f"after {self.max_attempts} attempts: {error}"
        )


def _answer_model(request: DecisionRequest) -> type[BaseModel]:
    """Build the structured-output schema for one request's questions."""
    fields: dict[str, Any] = {}
    for name, question in request.questions.items():
        if isinstance(question, BooleanQuestion):
            annotation: Any = bool
        elif isinstance(question, ChoiceQuestion):
            annotation = Literal[tuple(question.criteria)]  # type: ignore[valid-type]
        else:
            annotation = Annotated[int, Field(ge=0, le=len(question.criteria) - 1)]
        fields[name] = (annotation, ...)
    return create_model("DecisionAnswers", **fields)


def _parse_reply(response: Any, output_model: type[BaseModel]) -> BaseModel:
    """Validate a chat reply against the answer schema."""
    if isinstance(response.parsed, output_model):
        return response.parsed
    if isinstance(response.parsed, BaseModel):
        return output_model.model_validate(response.parsed.model_dump())
    if not response.content:
        raise ValueError("the chat model returned no content")
    return output_model.model_validate_json(response.content)


def _answer(question: Any, value: Any) -> DecisionAnswer:
    """Convert one selection into a one-hot normalized answer."""
    if isinstance(question, BooleanQuestion):
        return BooleanAnswer(probability_true=1.0 if value else 0.0)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            selected=value,
            probabilities={key: 1.0 if key == value else 0.0 for key in question.criteria},
            confidence=1.0,
        )
    levels = range(len(question.criteria))
    return ScoreAnswer(
        score=float(value),
        probabilities={level: 1.0 if level == value else 0.0 for level in levels},
        legend=dict(enumerate(question.criteria)),
        confidence=1.0,
    )


__all__ = ["LLMDecisionModel"]
