# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile annotated Python return types into backend-neutral decision schemas."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, create_model

from nooa.decisions.client import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    ScoreAnswer,
    ScoreQuestion,
)
from nooa.decisions.types import (
    BooleanDecision,
    ChoiceDecision,
    Criteria,
    Criterion,
    DecisionModelRequiredError,
    Instructions,
    ScoreDecision,
    Threshold,
)


@dataclass(frozen=True, slots=True)
class ChoiceValue:
    """Map a stable wire identifier to its declared Python choice."""

    wire_id: str
    value: Any
    criterion: Criterion


@dataclass(frozen=True, slots=True)
class DecisionOutputSchema:
    """Compiled schema for one named decision output."""

    name: str
    kind: Literal["boolean", "choice", "score"]
    instructions: Criterion
    criteria: Any
    threshold: float | None
    detailed: bool
    requires_decision_model: bool
    choices: tuple[ChoiceValue, ...] = ()

    def question(self) -> DecisionQuestion:
        """Build the normalized backend question for this output."""
        if self.kind == "boolean":
            return BooleanQuestion(instructions=self.instructions, criteria=self.criteria)
        if self.kind == "choice":
            return ChoiceQuestion(
                instructions=self.instructions,
                criteria={choice.wire_id: choice.criterion for choice in self.choices},
            )
        return ScoreQuestion(instructions=self.instructions, criteria=list(self.criteria))

    def fallback_question(self) -> dict[str, Any]:
        """Describe this output for a chat model in terms of Python result values."""
        question: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.kind == "boolean":
            if self.criteria is not None:
                question["criteria"] = self.criteria
        elif self.kind == "choice":
            question["options"] = [
                {
                    "value": TypeAdapter(Any).dump_python(choice.value, mode="json"),
                    "criterion": choice.criterion,
                }
                for choice in self.choices
            ]
        else:
            question["levels"] = [
                {"level": level, "criterion": criterion}
                for level, criterion in enumerate(self.criteria)
            ]
            question["answer"] = (
                f"The expected level: a number from 0 to {self.max_score}, "
                "weighted by how likely each level is."
            )
        return question

    @property
    def max_score(self) -> int:
        """Return the highest score level index."""
        return len(self.criteria) - 1


@dataclass(frozen=True, slots=True)
class DecisionSchema:
    """Compiled request and reconstruction rules for a decision method."""

    outputs: tuple[DecisionOutputSchema, ...]
    result_type: Any
    composite: bool
    shared_guidance: str | None

    def request(
        self,
        parameters: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
        events: Sequence[Any] | None = None,
    ) -> DecisionRequest:
        """Build a decision request from arguments and explicitly selected state."""
        state = render_decision_state(
            parameters,
            self.shared_guidance,
            context=context,
            events=events,
        )
        return DecisionRequest(
            state=state,
            questions={output.name: output.question() for output in self.outputs},
        )

    def fallback_guidance(self) -> str:
        """Render the compiled questions for the Predict fallback prompt."""
        questions = {output.name: output.fallback_question() for output in self.outputs}
        target = (
            "Answer each question in the result field with the same name."
            if self.composite
            else "Answer this question as the result."
        )
        return (
            "## Decision questions\n"
            f"{target} Use only the listed options or levels.\n\n"
            f"{json.dumps(questions, indent=2, ensure_ascii=False)}"
        )

    def fallback_result_type(self) -> Any:
        """Return the result type with score outputs bounded to their levels."""
        scores = {
            output.name: output.max_score for output in self.outputs if output.kind == "score"
        }
        if not scores:
            return self.result_type
        if not self.composite:
            return Annotated[self.result_type, Field(ge=0, le=scores["result"])]
        fields: dict[str, Any] = {}
        for name, upper in scores.items():
            field = self.result_type.model_fields[name]
            default = ... if field.is_required() else field.default
            fields[name] = (Annotated[float, Field(ge=0, le=upper)], default)
        return create_model(self.result_type.__name__, __base__=self.result_type, **fields)

    def restore_fallback_result(self, result: Any) -> Any:
        """Convert a bounded fallback result back to the declared result type."""
        if self.composite and type(result) is not self.result_type:
            return self.result_type.model_validate(result.model_dump())
        return result

    def reconstruct(self, response: DecisionResponse) -> Any:
        """Validate a normalized response and create the declared return value."""
        missing = {output.name for output in self.outputs} - set(response.answers)
        extra = set(response.answers) - {output.name for output in self.outputs}
        if missing or extra:
            raise ValueError(f"answer names differ from schema (missing={missing}, extra={extra})")
        values = {
            output.name: _reconstruct_output(output, response.answers[output.name])
            for output in self.outputs
        }
        if self.composite:
            return self.result_type.model_validate(values)
        return values[self.outputs[0].name]

    def require_llm_fallback_support(self) -> None:
        """Reject result shapes whose evidence cannot be produced by chat generation."""
        required = [output.name for output in self.outputs if output.requires_decision_model]
        if not required:
            return
        names = ", ".join(repr(name) for name in required)
        raise DecisionModelRequiredError(
            "DecideStrategy requires a configured decision_model for "
            f"output(s) {names}: detailed Decision results and Threshold metadata "
            "require probability evidence that the LLM fallback does not provide."
        )

    def fallback_answers(self, result: Any) -> dict[str, Any]:
        """Serialize an LLM fallback result without inventing probability evidence."""
        serialized = TypeAdapter(self.result_type).dump_python(result, mode="json")
        if self.composite:
            if not isinstance(serialized, dict):
                raise TypeError("Composite decision fallback did not serialize to an object")
            return {name: {"value": serialized[name]} for name in serialized}
        return {self.outputs[0].name: {"value": serialized}}


def compile_decision_schema(return_type: Any, docstring: str | None) -> DecisionSchema:
    """Compile a resolved method return annotation into a decision schema."""
    if return_type is None:
        raise TypeError("DecideStrategy requires a return type annotation")
    guidance = inspect.cleandoc(docstring or "").strip() or None
    if (
        inspect.isclass(return_type)
        and issubclass(return_type, BaseModel)
        and not issubclass(return_type, (BooleanDecision, ChoiceDecision, ScoreDecision))
    ):
        outputs: list[DecisionOutputSchema] = []
        for name, field in return_type.model_fields.items():
            annotation = field.annotation
            if field.metadata:
                annotation = Annotated[annotation, *field.metadata]
            output = _compile_output(name, annotation, None, require_instructions=True)
            outputs.append(output)
        if not outputs:
            raise TypeError("A composite decision result must contain at least one field")
        return DecisionSchema(tuple(outputs), return_type, True, guidance)

    output = _compile_output("result", return_type, guidance, require_instructions=False)
    explicit_instructions = _find_metadata(return_type, Instructions) is not None
    return DecisionSchema(
        (output,),
        return_type,
        False,
        guidance if explicit_instructions else None,
    )


def _compile_output(
    name: str,
    annotation: Any,
    fallback_instructions: str | None,
    *,
    require_instructions: bool,
) -> DecisionOutputSchema:
    """Compile one primitive or detailed decision output annotation."""
    base, metadata = _unwrap_annotated(annotation)
    base, nullable = _strip_none(base)
    base, nested_metadata = _unwrap_annotated(base)
    metadata.extend(nested_metadata)
    instructions_meta = next((item for item in metadata if isinstance(item, Instructions)), None)
    criteria_meta = next((item for item in metadata if isinstance(item, Criteria)), None)
    threshold_meta = next((item for item in metadata if isinstance(item, Threshold)), None)

    if instructions_meta is None and require_instructions:
        raise TypeError(f"Composite decision field {name!r} requires Instructions(...)")
    instructions: Criterion = (
        instructions_meta.value if instructions_meta is not None else fallback_instructions
    )
    if instructions is None:
        raise TypeError(f"Decision output {name!r} requires instructions or a method docstring")
    threshold = threshold_meta.value if threshold_meta is not None else None

    detailed = False
    kind: Literal["boolean", "choice", "score"]
    choice_type: Any = None

    origin = get_origin(base)
    if base is BooleanDecision:
        detailed = True
        kind = "boolean"
    elif base is ScoreDecision:
        detailed = True
        kind = "score"
    elif inspect.isclass(base) and issubclass(base, ChoiceDecision):
        detailed = True
        kind = "choice"
        args = get_args(base) or base.__pydantic_generic_metadata__.get("args", ())
        if not args:
            raise TypeError("ChoiceDecision must be parameterized with an Enum type")
        choice_type = args[0]
    else:
        origin = get_origin(base)
        if base is bool:
            kind = "boolean"
        elif base is float:
            kind = "score"
        elif inspect.isclass(base) and issubclass(base, Enum):
            kind = "choice"
            choice_type = base
        elif origin is Literal:
            kind = "choice"
            choice_type = base
        else:
            raise TypeError(
                f"Unsupported decision output {name!r}: expected bool, float, Enum, "
                "Literal, or a detailed Decision type"
            )

    if kind == "boolean":
        criteria = _boolean_criteria(criteria_meta, name)
        return DecisionOutputSchema(
            name,
            kind,
            instructions,
            criteria,
            threshold if threshold is not None else 0.5,
            detailed,
            detailed or threshold_meta is not None,
        )
    if kind == "score":
        if threshold is not None:
            raise TypeError(f"Threshold is not supported for score output {name!r}")
        if criteria_meta is None or isinstance(criteria_meta.value, dict):
            raise TypeError(f"Score output {name!r} requires positional Criteria(...)")
        criteria = tuple(criteria_meta.value)
        if not 2 <= len(criteria) <= 10:
            raise ValueError(f"Score output {name!r} requires between 2 and 10 criteria")
        return DecisionOutputSchema(name, kind, instructions, criteria, None, detailed, detailed)

    choices = _choice_values(choice_type, criteria_meta, name)
    if threshold is not None and not detailed and not nullable:
        raise TypeError(
            f"Thresholded primitive choice output {name!r} must include None in its return type"
        )
    return DecisionOutputSchema(
        name,
        kind,
        instructions,
        None,
        threshold,
        detailed,
        detailed or threshold_meta is not None,
        choices,
    )


def _boolean_criteria(criteria: Criteria | None, name: str) -> dict[str, Criterion] | None:
    """Normalize optional boolean criteria to the decision wire keys."""
    if criteria is None:
        return None
    if not isinstance(criteria.value, dict):
        raise TypeError(
            f"Boolean output {name!r} requires Criteria(by_value={{True: ..., False: ...}})"
        )
    if set(criteria.value) != {True, False}:
        raise ValueError(f"Boolean output {name!r} criteria must cover True and False exactly")
    return {"true": criteria.value[True], "false": criteria.value[False]}


def _choice_values(
    choice_type: Any, criteria: Criteria | None, name: str
) -> tuple[ChoiceValue, ...]:
    """Compile enum or literal choices with stable identifiers and descriptions."""
    origin = get_origin(choice_type)
    if origin is Literal:
        values = list(get_args(choice_type))
        if criteria is None:
            descriptions: list[Criterion] = list(values)
        elif isinstance(criteria.value, dict):
            if set(criteria.value) != set(values):
                raise ValueError(f"Literal output {name!r} criteria must cover every value exactly")
            descriptions = [criteria.value[value] for value in values]
        else:
            descriptions = list(criteria.value)
        if len(descriptions) != len(values):
            raise ValueError(f"Literal output {name!r} criteria count must match its values")
        return tuple(
            ChoiceValue(wire_id=f"choice_{index}", value=value, criterion=descriptions[index])
            for index, value in enumerate(values)
        )

    if not inspect.isclass(choice_type) or not issubclass(choice_type, Enum):
        raise TypeError(f"Choice output {name!r} must use an Enum or Literal type")
    members = list(choice_type)
    if criteria is None:
        inferred = _enum_docstring_criteria(choice_type)
        if inferred is None:
            raise TypeError(
                f"Enum output {name!r} requires Criteria(...) or complete Attributes documentation"
            )
        descriptions = [inferred[member.name] for member in members]
    elif isinstance(criteria.value, dict):
        if set(criteria.value) != set(members):
            raise ValueError(f"Enum output {name!r} criteria must cover every non-alias member")
        descriptions = [criteria.value[member] for member in members]
    else:
        descriptions = list(criteria.value)
        if len(descriptions) != len(members):
            raise ValueError(f"Enum output {name!r} criteria count must match its members")
    return tuple(
        ChoiceValue(member.name, member, descriptions[index])
        for index, member in enumerate(members)
    )


def _enum_docstring_criteria(enum_type: type[Enum]) -> dict[str, str] | None:
    """Read complete enum-member descriptions from an ``Attributes`` docstring."""
    doc = inspect.cleandoc(enum_type.__doc__ or "")
    match = re.search(r"(?ms)^Attributes:\s*\n(?P<body>.*?)(?:\n\S|\Z)", doc)
    if not match:
        return None
    result: dict[str, str] = {}
    for line in match.group("body").splitlines():
        item = re.match(r"\s*([A-Za-z_]\w*):\s*(.+)", line)
        if item:
            result[item.group(1)] = item.group(2).strip()
    names = {member.name for member in enum_type}
    return result if set(result) >= names else None


def _unwrap_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    """Return the base annotation and all nested ``Annotated`` metadata."""
    metadata: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        metadata.extend(args[1:])
    return annotation, metadata


def _find_metadata(annotation: Any, metadata_type: type[Any]) -> Any | None:
    """Find the first metadata item of the requested type around an annotation."""
    base, metadata = _unwrap_annotated(annotation)
    base, _ = _strip_none(base)
    _, nested_metadata = _unwrap_annotated(base)
    metadata.extend(nested_metadata)
    return next((item for item in metadata if isinstance(item, metadata_type)), None)


def _strip_none(annotation: Any) -> tuple[Any, bool]:
    """Remove ``None`` from a nullable union and report whether it was present."""
    origin = get_origin(annotation)
    if origin not in {Union, types.UnionType}:
        return annotation, False
    args = get_args(annotation)
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1 and len(non_none) != len(args):
        return non_none[0], True
    return annotation, False


def render_decision_state(
    parameters: Mapping[str, Any],
    guidance: str | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    events: Sequence[Any] | None = None,
) -> dict[str, JsonValue]:
    """Render arguments, guidance, and opted-in state as deterministic JSON."""
    inputs: dict[str, JsonValue] = {}
    for name, value in parameters.items():
        try:
            inputs[name] = _json_value(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Decision input parameter {name!r} is not JSON-compatible: {exc}"
            ) from exc
    envelope: dict[str, JsonValue] = {"inputs": inputs}
    if guidance:
        envelope["shared_guidance"] = guidance
    if context:
        resolved_context: dict[str, JsonValue] = {}
        for name, value in context.items():
            try:
                resolved_context[name] = _json_value(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Decision context block {name!r} is not JSON-compatible: {exc}"
                ) from exc
        envelope["context"] = resolved_context
    if events is not None:
        resolved_events: list[JsonValue] = []
        for index, event in enumerate(events):
            try:
                resolved_events.append(_json_value(event))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Decision event at index {index} is not JSON-compatible: {exc}"
                ) from exc
        envelope["events"] = resolved_events
    return envelope


def _json_value(value: Any) -> JsonValue:
    """Convert supported Python values recursively into JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not supported")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings")
        return cast(JsonValue, {key: _json_value(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported value of type {type(value).__name__}")


def _validate_distribution(
    probabilities: Mapping[Any, float], expected: set[Any], name: str
) -> None:
    """Validate that a probability distribution exactly covers its outcomes."""
    if set(probabilities) != expected:
        raise ValueError(f"{name} distribution does not cover the declared outcomes exactly")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError(f"{name} distribution contains an invalid probability")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(f"{name} distribution must sum to one")


def _validate_probability(value: float, name: str) -> None:
    """Validate one finite probability value."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _reconstruct_output(output: DecisionOutputSchema, answer: Any) -> Any:
    """Reconstruct one declared Python output from a normalized answer."""
    if output.kind == "boolean":
        if not isinstance(answer, BooleanAnswer):
            raise TypeError(f"Answer for {output.name!r} is not boolean")
        probability = answer.probability_true
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Boolean probability for {output.name!r} must be in [0, 1]")
        threshold = output.threshold if output.threshold is not None else 0.5
        decision = BooleanDecision(
            value=probability >= threshold,
            probability_true=probability,
            threshold=threshold,
        )
        return decision if output.detailed else decision.value

    if output.kind == "choice":
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"Answer for {output.name!r} is not a choice")
        by_id = {choice.wire_id: choice.value for choice in output.choices}
        _validate_distribution(answer.probabilities, set(by_id), output.name)
        if answer.selected not in by_id:
            raise ValueError(f"Selected choice {answer.selected!r} is not declared")
        _validate_probability(answer.confidence, f"Choice confidence for {output.name!r}")
        probabilities = {by_id[key]: value for key, value in answer.probabilities.items()}
        selected = by_id[answer.selected]
        selected_probability = answer.probabilities[answer.selected]
        value = (
            selected
            if output.threshold is None or selected_probability >= output.threshold
            else None
        )
        if not output.detailed:
            return value
        decision = ChoiceDecision(
            value=value,
            selected=selected,
            probabilities=probabilities,
            confidence=answer.confidence,
            threshold=output.threshold,
        )
        return decision

    if not isinstance(answer, ScoreAnswer):
        raise TypeError(f"Answer for {output.name!r} is not a score")
    expected = set(range(len(output.criteria)))
    _validate_distribution(answer.probabilities, expected, output.name)
    if set(answer.legend) != expected:
        raise ValueError(f"Score legend for {output.name!r} is incomplete")
    if any(answer.legend[index] != output.criteria[index] for index in expected):
        raise ValueError(f"Score legend for {output.name!r} differs from the declared criteria")
    _validate_probability(answer.confidence, f"Score confidence for {output.name!r}")
    expected_score = sum(index * probability for index, probability in answer.probabilities.items())
    if not math.isfinite(answer.score) or not math.isclose(
        answer.score, expected_score, rel_tol=1e-4, abs_tol=1e-4
    ):
        raise ValueError(f"Score for {output.name!r} does not match its distribution")
    decision = ScoreDecision(
        value=answer.score,
        probabilities=answer.probabilities,
        legend=answer.legend,
        confidence=answer.confidence,
    )
    return decision if output.detailed else decision.value


__all__ = [
    "DecisionOutputSchema",
    "DecisionSchema",
    "compile_decision_schema",
    "render_decision_state",
]
