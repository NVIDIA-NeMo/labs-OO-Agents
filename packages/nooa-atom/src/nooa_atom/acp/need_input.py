# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map explicit descriptors to verified form schemas, without hidden conversion.

Pool mappings here are experimentally tested text, oneOf const/title and anyOf
picker-or-text schemas, not a claim about every Pool type. Standard ACP maps
text and single enum; picker-or-text falls back explicitly to a text property.
"""

from typing import Any

from acp.schema import ElicitationSchema, ElicitationStringPropertySchema

from nooa.interactive import FormResponse, NeedInput, NeedInputForm


def need_input_schema(need: NeedInputForm | NeedInput) -> ElicitationSchema | None:
    """Standard ACP: strings and strict single enums; choice-or-other is text."""
    if not isinstance(need, NeedInputForm):
        return None
    properties = {}
    for question in need.questions:
        description = question.help
        if question.kind == "pick_one_or_text":
            choices = "; ".join(f"{c.title}: {c.value}" for c in question.choices)
            description = "\n".join(
                filter(None, [description, f"Enter text (suggestions: {choices})."])
            )
        elif question.kind == "pick_one":
            # ACP's enum has no verified distinct label field. Display labels honestly
            # in help, while keeping the actual submitted values in enum.
            choices = "; ".join(f"{c.title}: {c.value}" for c in question.choices)
            description = "\n".join(filter(None, [description, f"Choices: {choices}"]))
        properties[question.id] = ElicitationStringPropertySchema(
            type="string",
            title=question.label,
            description=description,
            enum=[c.value for c in question.choices] if question.kind == "pick_one" else None,
        )
    return ElicitationSchema(
        title=need.heading,
        properties=properties,
        required=[q.id for q in need.questions if q.required],
    )


def pool_form_schema(need: NeedInputForm | NeedInput) -> dict[str, Any] | None:
    """Verified Pool 1.0.16 schemas; optional pickers are rejected by descriptors."""
    if not isinstance(need, NeedInputForm):
        return None
    properties = {}
    for question in need.questions:
        prop: dict[str, Any] = {"title": question.label}
        if question.help:
            prop["description"] = question.help
        if question.kind == "text":
            prop["type"] = "string"
        else:
            picker = {"oneOf": [{"const": c.value, "title": c.title} for c in question.choices]}
            if question.kind == "pick_one":
                prop.update(type="string", **picker)
            else:
                prop["anyOf"] = [picker, {"type": "string"}]
        properties[question.id] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [q.id for q in need.questions if q.required],
    }


def form_content(need: NeedInputForm, content: dict[str, Any] | None) -> dict[str, str]:
    """Validate only response envelope shape; request validation is done by the owner."""
    response = FormResponse(action="accept", content=content)
    assert response.content is not None
    return response.content


pool_content = form_content


def answer_from_content(need: NeedInputForm, content: dict[str, Any] | None) -> dict[str, str]:
    response = need.validate_response(FormResponse(action="accept", content=content))
    assert response.content is not None
    return response.content


pool_answer = answer_from_content

__all__ = [
    "answer_from_content",
    "form_content",
    "need_input_schema",
    "pool_answer",
    "pool_content",
    "pool_form_schema",
]
