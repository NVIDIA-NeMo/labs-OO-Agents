# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small fresh native decision fixtures, not a model evaluation suite."""

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from typing import Annotated, Literal

from opentelemetry.trace import INVALID_SPAN, use_span
from pydantic import BaseModel

from nooa import Agent, DecideStrategy, no_trace, strategy
from nooa.decisions import Criteria, Instructions, ScoreDecision
from nooa.decisions.schema import compile_decision_schema
from nooa.media import Image
from nooa.runtime.hooks import hooks_scope

from ._baseline import _NoHooks
from ._media import png_fixture


class _RoutingResult(BaseModel):
    supported: Annotated[bool, Instructions("Does the statement contain a support request?")]


class _FixtureResult(BaseModel):
    positive: Annotated[bool, Instructions("Is the number positive?")]
    sign: Annotated[
        Literal["negative", "zero", "positive"], Instructions("Classify the number's sign.")
    ]
    score: Annotated[
        ScoreDecision,
        Instructions("Rate the number's sign."),
        Criteria("negative", "zero", "positive"),
    ]


class _ImageResult(BaseModel):
    accessible: Annotated[bool, Instructions("Can you access and interpret the attached image?")]
    color: Annotated[
        Literal["red", "green", "blue"], Instructions("What is the dominant color in the image?")
    ]


class _DecisionFixture(Agent):
    @no_trace
    @strategy(DecideStrategy())
    async def routing(self, statement: str) -> _RoutingResult:
        """Check the statement."""
        ...

    @no_trace
    @strategy(DecideStrategy())
    async def composite(self, number: int) -> _FixtureResult:
        """Evaluate the supplied number using the declared criteria."""
        ...

    @no_trace
    @strategy(DecideStrategy())
    async def positive(self, number: int) -> bool:
        """Is the number positive?"""
        ...

    @no_trace
    @strategy(DecideStrategy())
    async def image(self, image: Image) -> _ImageResult:
        """Inspect the attached image."""
        ...


def case(name):
    if name == "routing":
        return (
            "routing",
            _RoutingResult,
            "Check the statement.",
            {"statement": "The support request requires a decision."},
        )
    if name == "decision:typed":
        return (
            "composite",
            _FixtureResult,
            "Evaluate the supplied number using the declared criteria.",
            {"number": 7},
        )
    if name in {"decision:positive", "decision:negative"}:
        return (
            "positive",
            bool,
            "Is the number positive?",
            {"number": 7 if name.endswith("positive") else -7},
        )
    if name == "media:image":
        return (
            "image",
            _ImageResult,
            "Inspect the attached image.",
            {"image": Image.from_bytes(png_fixture(), media_type="image/png")},
        )
    raise ValueError("Unknown decision fixture")


def planned_body(name):
    """Synthetic preview with no encoded image, URL or credentials."""
    _, result_type, guidance, inputs = case(name)
    schema = compile_decision_schema(result_type, guidance)
    request = schema.request(inputs, media_inputs=["image"] if name == "media:image" else [])
    return {
        "state": request.state,
        "questions": {key: asdict(value) for key, value in request.questions.items()},
    }


async def typed_decision(client, name):
    """Run an actual Agent/DecideStrategy call, capturing only this response."""
    method, _, _, inputs = case(name)
    # Agent construction may announce automatic tracing. This synchronous scope
    # keeps the library silent and cannot swallow concurrent async task output.
    with hooks_scope(_NoHooks(), compose=False), use_span(INVALID_SPAN):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            agent = _DecisionFixture(decision_model=client)
    captured = []

    async def one_shot(ctx, nxt):
        if captured:
            raise RuntimeError("Decision check permits only one call")
        captured.append(None)
        ctx = await nxt(ctx)
        captured[0] = ctx.response
        return ctx

    remove = agent.event_manager.intercept("decision_call", one_shot)
    try:
        with hooks_scope(_NoHooks(), compose=False), use_span(INVALID_SPAN):
            result = await getattr(agent, method)(**inputs)
        response = captured[0]
        if response is None:
            raise RuntimeError("No native decision response captured")
        facts = {"typed_result_validated": True}
        if name == "routing":
            facts["probability_true"] = response.answers["supported"].probability_true
        elif name == "decision:typed":
            facts["answer_correct"] = (
                result.positive is True
                and result.sign == "positive"
                and abs(result.score.value - 2) <= 0.25
            )
            facts["typed_result"] = result.model_dump(mode="json")
        elif name.startswith("decision:"):
            facts["answer_correct"] = result is name.endswith("positive")
            facts["typed_result"] = result
        else:
            facts.update(
                access_reported=result.accessible,
                answer_correct=result.color == "red",
                typed_result=result.model_dump(mode="json"),
            )
        return response, facts
    finally:
        remove()
        agent.event_manager.clear()
