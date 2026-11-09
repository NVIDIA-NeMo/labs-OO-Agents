# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert decision requests and responses to and from OpenAI's Decisions API.

Used by ``DecisionClient`` with ``api_style="openai"``. OpenAI calls a boolean
question a *predicate*, and answers arrive as a list in question order rather
than as a mapping by name.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast

from nooa._immutable_json import freeze
from nooa.decisions.client import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionRefusedError,
    DecisionRequest,
    DecisionResponse,
    ScoreAnswer,
)
from nooa.decisions.types import Criterion

_ANSWER_TYPES = {BooleanQuestion: "predicate", ChoiceQuestion: "choice"}


def openai_payload(model: str, request: DecisionRequest) -> dict[str, Any]:
    """Build a ``POST /v1/decisions`` body from a normalized request.

    OpenAI accepts only text instructions and descriptions, so structured
    criteria are sent as JSON text. Predicates have no criteria field, so
    boolean criteria are appended to the instructions. Without images,
    ``input`` is the state as text; with images, it is one user message with
    the state text followed by each image, preceded by its alias.
    """
    questions: list[dict[str, Any]] = []
    for name, question in request.questions.items():
        instructions = _text(question.instructions)
        if isinstance(question, BooleanQuestion):
            if question.criteria is not None:
                instructions = (
                    f"{instructions}\n\n"
                    f"True: {_text(question.criteria['true'])}\n"
                    f"False: {_text(question.criteria['false'])}"
                )
            questions.append({"type": "predicate", "name": name, "instructions": instructions})
        elif isinstance(question, ChoiceQuestion):
            questions.append(
                {
                    "type": "choice",
                    "name": name,
                    "instructions": instructions,
                    "choices": [
                        _described({"value": value}, criterion)
                        for value, criterion in question.criteria.items()
                    ],
                }
            )
        else:
            questions.append(
                {
                    "type": "score",
                    "name": name,
                    "instructions": instructions,
                    "levels": [
                        _described({"label": str(level)}, criterion)
                        for level, criterion in enumerate(question.criteria)
                    ],
                }
            )
    state = request.state
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    input_value: Any = text
    if request.attachments:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for attachment in request.attachments:
            # The text immediately before each image associates its alias with
            # the named references in the state, including repeated inputs.
            content.append({"type": "input_text", "text": attachment.alias})
            image: dict[str, Any] = {"type": "input_image", "image_url": attachment.image_url}
            if attachment.detail is not None:
                image["detail"] = attachment.detail
            content.append(image)
        input_value = [{"role": "user", "content": content}]
    return {"model": model, "input": input_value, "questions": questions}


def parse_openai_response(
    data: Any, request: DecisionRequest, *, request_id: str | None = None
) -> DecisionResponse:
    """Convert an OpenAI ``Decision`` object into a ``DecisionResponse``.

    Answers arrive in question order and are matched to the request's
    questions by position. Score legends are rebuilt from the request because
    OpenAI returns level labels rather than criteria. The ``x-request-id``
    header value (``request_id``) is used as the response ID when the body
    has none.

    Raises:
        DecisionRefusedError: If the model refused any question.
        TypeError, ValueError, KeyError: If the body does not match the request.
    """
    if not isinstance(data, dict) or not isinstance(data.get("answers"), list):
        raise TypeError("expected an object containing an answers array")
    raw_answers = data["answers"]
    if len(raw_answers) != len(request.questions):
        raise ValueError(
            f"expected {len(request.questions)} answers in question order, got {len(raw_answers)}"
        )
    raw = cast("Mapping[str, Any]", freeze(data))

    answers: dict[str, DecisionAnswer] = {}
    refused: list[str] = []
    for (name, question), answer in zip(request.questions.items(), raw_answers, strict=True):
        if not isinstance(answer, dict):
            raise TypeError(f"answer {name!r} must be an object")
        if answer.get("name") not in (None, name):
            raise ValueError(f"answer {answer.get('name')!r} is out of order; expected {name!r}")
        answer_type = answer.get("type")
        if answer_type == "refusal":
            refused.append(name)
            continue
        expected = _ANSWER_TYPES.get(type(question), "score")
        if answer_type != expected:
            raise ValueError(f"answer {name!r} has type {answer_type!r}; expected {expected!r}")
        if isinstance(question, BooleanQuestion):
            answers[name] = BooleanAnswer(probability_true=_number(answer["probability"]))
        elif isinstance(question, ChoiceQuestion):
            answers[name] = ChoiceAnswer(
                selected=str(answer["choice"]),
                probabilities={
                    str(item["value"]): _number(item["probability"])
                    for item in answer["probabilities"]
                },
                confidence=_number(answer["confidence"]),
            )
        else:
            answers[name] = ScoreAnswer(
                score=_number(answer["score"]),
                probabilities={
                    int(item["value"]): _number(item["probability"])
                    for item in answer["probabilities"]
                },
                legend=dict(enumerate(question.criteria)),
                confidence=_number(answer["confidence"]),
            )
    if refused:
        raise DecisionRefusedError(tuple(refused), raw=raw)

    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        raise TypeError("usage must be an object")
    return DecisionResponse(
        answers=answers,
        model=str(data.get("model") or ""),
        id=str(data["id"]) if data.get("id") is not None else request_id,
        usage=usage,
        raw=raw,
    )


def _text(value: Criterion) -> str:
    """Render an instruction or criterion as the text OpenAI accepts."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _described(item: dict[str, Any], criterion: Criterion) -> dict[str, Any]:
    """Add a ``description`` to ``item`` when ``criterion`` is set, and return ``item``."""
    if criterion is not None:
        item["description"] = _text(criterion)
    return item


def _number(value: Any) -> float:
    """Read a finite JSON number, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"expected a finite number, got {value!r}")
    return float(value)
