# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact progressive disclosure of forms, independent controls and agent results."""

import json
from pathlib import Path

import pytest

from nooa.agentdoc import doc, spec
from nooa.forms import (
    BooleanQuestion,
    ChoiceQuestion,
    FormChoice,
    FormResponse,
    IntegerQuestion,
    MultiChoiceQuestion,
    NeedInputForm,
    NumberQuestion,
    TextQuestion,
)
from nooa.interactive import InteractiveAgent, NeedInput

FIXTURES = Path(__file__).parent / "fixtures" / "input_forms.json"
CONTROLS = [
    NeedInputForm,
    TextQuestion,
    IntegerQuestion,
    NumberQuestion,
    BooleanQuestion,
    ChoiceQuestion,
    MultiChoiceQuestion,
    FormChoice,
    FormResponse,
]


def snapshot(name):
    return json.loads(FIXTURES.read_text())[name]


@pytest.mark.parametrize("concise,name", [(False, "form"), (True, "form_concise")])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_direct_form_is_compact_until_individual_types_are_requested(concise, name, depth):
    assert doc(NeedInputForm, concise=concise, inline_depth=depth) == snapshot(name)


def test_agent_return_annotation_discovers_form_without_expanding_graph():
    assert doc(InteractiveAgent.handle, concise=True, inline_depth=2) == snapshot("handle")
    assert "class NeedInputForm" in snapshot("handle")
    assert "class TextQuestion" not in snapshot("handle")


def test_explicit_combined_docs_show_complete_deduplicated_constructors():
    assert doc(*CONTROLS, inline_depth=2) == snapshot("controls")
    assert doc(CONTROLS, inline_depth=2) == snapshot("controls")
    for cls in CONTROLS:
        assert snapshot("controls").count(f"class {cls.__name__}(") == 1
    assert "validate_answer" not in snapshot("controls")
    assert "def validate_response" in snapshot("controls")


@pytest.mark.parametrize("cls", CONTROLS)
def test_individual_docs_expose_exact_constructor_contract(cls):
    sections = snapshot("controls").split("\nclass ")
    sections = [sections[0]] + ["class " + section for section in sections[1:]]
    expected = next(section for section in sections if section.startswith(f"class {cls.__name__}("))
    assert doc(cls, inline_depth=0) == expected
    if cls in (ChoiceQuestion, MultiChoiceQuestion):
        choice = next(section for section in sections if section.startswith("class FormChoice("))
        expected += "\n## Referenced Types\n" + choice
    assert doc(cls) == expected


def test_existing_unrelated_docs_unchanged():
    assert doc(NeedInput, concise=True) == (
        "class NeedInput(BaseModel):\n"
        '    """Conversational question, answered by the next user message; never a dialog."""\n\n'
        "    kind: Literal[question] = 'question'\n"
        "    question: str\n"
        "    reason: str | None = None"
    )


def test_field_and_method_expand_metadata_are_local_not_type_suppression():
    class Details:
        value: int

    class API:
        config: Details

        @spec(expand=False)
        def check(self, value: Details) -> Details:
            """Check explicitly on demand."""

    expected = "class Details:\n    value: int"
    assert doc(Details) == expected
    assert doc(API).endswith("## Referenced Types\n" + expected)
    spec(API, "config", expand=False)
    assert doc(API) == (
        "class API:\n    config: Details\n\n"
        "    def check(self, value: Details) -> Details:\n"
        '        """Check explicitly on demand."""'
    )
    assert doc(API, Details).endswith(expected)
