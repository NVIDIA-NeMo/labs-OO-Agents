# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from nooa.decisions import ChoiceDecision, Criteria, Instructions, Threshold
from nooa.decisions.client import (
    BooleanAnswer,
    ChoiceAnswer,
    DecisionResponse,
    ScoreAnswer,
)
from nooa.decisions.schema import compile_decision_schema, render_decision_state


class Department(Enum):
    BILLING = 1
    TECHNICAL = 2


class DocumentedDepartment(Enum):
    """A department documented only through its docstring.

    Attributes:
        BILLING: Payments and refunds.
        TECHNICAL: Bugs and outages.
    """

    BILLING = "billing"
    TECHNICAL = "technical"


def test_compile_and_reconstruct_composite_decisions() -> None:
    class Triage(BaseModel):
        urgent: Annotated[
            bool,
            Instructions("Is it urgent?"),
            Criteria(by_value={True: "urgent", False: "not urgent"}),
            Threshold(0.8),
        ]
        department: Annotated[
            Department,
            Instructions("Who owns it?"),
            Criteria(by_value={Department.BILLING: "payments", Department.TECHNICAL: "bugs"}),
        ]
        frustration: Annotated[
            float,
            Instructions("How frustrated?"),
            Criteria("calm", "frustrated", "angry"),
        ]

    schema = compile_decision_schema(Triage, "Use only the supplied message.")
    request = schema.request({"message": "Payouts have failed."})

    assert request.questions["urgent"].type == "noul"
    assert request.questions["department"].criteria == {
        "BILLING": "payments",
        "TECHNICAL": "bugs",
    }
    assert request.questions["frustration"].criteria == ["calm", "frustrated", "angry"]
    assert request.state == {
        "inputs": {"message": "Payouts have failed."},
        "shared_guidance": "Use only the supplied message.",
    }

    result = schema.reconstruct(
        DecisionResponse(
            model="test",
            answers={
                "urgent": BooleanAnswer(0.8),
                "department": ChoiceAnswer(
                    selected="BILLING",
                    probabilities={"BILLING": 0.75, "TECHNICAL": 0.25},
                    confidence=0.5,
                ),
                "frustration": ScoreAnswer(
                    score=1.25,
                    probabilities={0: 0.0, 1: 0.75, 2: 0.25},
                    legend={0: "calm", 1: "frustrated", 2: "angry"},
                    confidence=0.5,
                ),
            },
        )
    )

    assert result == Triage(
        urgent=True,
        department=Department.BILLING,
        frustration=1.25,
    )


def test_literal_uses_collision_safe_wire_ids() -> None:
    result_type = Annotated[
        Literal[1, "1"],
        Criteria("integer one", "string one"),
    ]
    schema = compile_decision_schema(result_type, "Choose one.")
    assert schema.outputs[0].question().criteria == {
        "choice_0": "integer one",
        "choice_1": "string one",
    }
    result = schema.reconstruct(
        DecisionResponse(
            model="test",
            answers={
                "result": ChoiceAnswer(
                    selected="choice_1",
                    probabilities={"choice_0": 0.1, "choice_1": 0.9},
                    confidence=0.8,
                )
            },
        )
    )
    assert result == "1"


def test_enum_docstring_alone_does_not_supply_criteria() -> None:
    with pytest.raises(
        TypeError, match=r"requires Criteria\(by_value=\{DocumentedDepartment\.MEMBER"
    ):
        compile_decision_schema(DocumentedDepartment, "Choose the owner.")


def test_detailed_and_composite_enum_outputs_require_criteria() -> None:
    class Triage(BaseModel):
        department: Annotated[DocumentedDepartment, Instructions("Which team owns it?")]

    for result_type in (ChoiceDecision[DocumentedDepartment], Triage):
        with pytest.raises(TypeError, match="requires Criteria"):
            compile_decision_schema(result_type, "Choose the owner.")


def test_mapped_criteria_describe_enum_members() -> None:
    schema = compile_decision_schema(
        Annotated[
            DocumentedDepartment,
            Criteria(
                by_value={
                    DocumentedDepartment.BILLING: "Payments and refunds.",
                    DocumentedDepartment.TECHNICAL: "Bugs and outages.",
                }
            ),
        ],
        "Choose the owner.",
    )

    assert schema.outputs[0].question().criteria == {
        "BILLING": "Payments and refunds.",
        "TECHNICAL": "Bugs and outages.",
    }


def test_thresholded_primitive_choice_must_be_nullable() -> None:
    with pytest.raises(TypeError, match="must include None"):
        compile_decision_schema(
            Annotated[Department, Criteria("payments", "bugs"), Threshold(0.8)],
            "Choose a department.",
        )


def test_detailed_choice_preserves_rejected_selection() -> None:
    result_type = Annotated[
        ChoiceDecision[Department],
        Criteria("payments", "bugs"),
        Threshold(0.8),
    ]
    schema = compile_decision_schema(result_type, "Choose a department.")
    result = schema.reconstruct(
        DecisionResponse(
            model="test",
            answers={
                "result": ChoiceAnswer(
                    selected="BILLING",
                    probabilities={"BILLING": 0.7, "TECHNICAL": 0.3},
                    confidence=0.4,
                )
            },
        )
    )
    assert result.value is None
    assert result.selected is Department.BILLING


def test_nullable_nested_annotated_alias_is_supported() -> None:
    alias = Annotated[
        Department,
        Instructions("Pick the owner."),
        Criteria("payments", "bugs"),
    ]
    schema = compile_decision_schema(Annotated[alias | None, Threshold(0.8)], "Choose one.")
    assert schema.outputs[0].choices[0].value is Department.BILLING
    assert schema.shared_guidance == "Choose one."


def test_bare_float_is_rejected() -> None:
    with pytest.raises(TypeError, match="requires positional Criteria"):
        compile_decision_schema(float, "Score it.")


def test_criteria_rejects_positional_entries_with_by_value() -> None:
    with pytest.raises(TypeError, match="positional entries or by_value"):
        Criteria("ordered", by_value={True: "mapped"})


def test_state_rejects_unsupported_parameter_by_name() -> None:
    with pytest.raises(TypeError, match="parameter 'stream'.*unsupported value"):
        render_decision_state({"stream": object()})


def test_state_rejects_non_finite_float_nested_in_pydantic_model() -> None:
    class Input(BaseModel):
        value: float

    with pytest.raises(TypeError, match="parameter 'payload'.*non-finite"):
        render_decision_state({"payload": Input(value=float("inf"))})


def test_state_includes_json_compatible_context_and_events() -> None:
    state = render_decision_state(
        {"message": "hello"},
        context={"policy": {"limit": 500}},
        events=[{"type": "Message", "data": {"content": "Earlier message"}}],
    )

    assert state == {
        "inputs": {"message": "hello"},
        "context": {"policy": {"limit": 500}},
        "events": [{"type": "Message", "data": {"content": "Earlier message"}}],
    }


def test_state_rejects_unsupported_context_by_name() -> None:
    with pytest.raises(TypeError, match="Decision context block 'policy'.*not JSON-compatible"):
        render_decision_state({}, context={"policy": object()})


def test_state_rejects_unsupported_event_by_index() -> None:
    with pytest.raises(TypeError, match="Decision event at index 0.*not JSON-compatible"):
        render_decision_state({}, events=[object()])


def test_primitive_choice_rejects_invalid_confidence() -> None:
    schema = compile_decision_schema(
        Annotated[Department, Criteria("payments", "bugs")],
        "Choose a department.",
    )
    with pytest.raises(ValueError, match="Choice confidence.*must be in"):
        schema.reconstruct(
            DecisionResponse(
                model="test",
                answers={
                    "result": ChoiceAnswer(
                        selected="BILLING",
                        probabilities={"BILLING": 0.7, "TECHNICAL": 0.3},
                        confidence=1.1,
                    )
                },
            )
        )
