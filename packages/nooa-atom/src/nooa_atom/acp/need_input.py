# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a ``NeedInputForm`` to an elicitation schema and read the answer back.

Each question becomes one property:

- ``TextQuestion``: ``{"type": "string"}``.
- ``IntegerQuestion``: ``{"type": "integer"}`` with ``minimum``/``maximum``.
- ``BooleanQuestion``: ``{"type": "boolean"}``.
- ``PickOneQuestion(allow_other=False)``: a single-select string,
  ``oneOf`` of ``{"const": value, "title": title}``.
- ``PickOneQuestion(allow_other=True)``: standard ACP has no select-or-text
  property, so it is a plain string whose description lists the choices;
  Pool shows ``anyOf`` of that single-select and a plain string.
"""

from typing import Any

from acp.schema import (
    ElicitationBooleanPropertySchema,
    ElicitationIntegerPropertySchema,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    EnumOption,
)

from nooa.interactive import FormResponse, FormValue, NeedInput, NeedInputForm


def need_input_schema(need: NeedInputForm | NeedInput) -> ElicitationSchema | None:
    """The ACP elicitation schema of a form; ``None`` for a plain question."""
    if not isinstance(need, NeedInputForm):
        return None
    properties: dict[str, Any] = {}
    for question in need.questions:
        description = question.help
        if question.kind == "integer":
            properties[question.id] = ElicitationIntegerPropertySchema(
                type="integer",
                title=question.label,
                description=description,
                minimum=question.minimum,
                maximum=question.maximum,
            )
            continue
        if question.kind == "boolean":
            properties[question.id] = ElicitationBooleanPropertySchema(
                type="boolean", title=question.label, description=description
            )
            continue
        one_of = None
        if question.kind == "pick_one" and question.allow_other:
            choices = ", ".join(c.value for c in question.choices)
            description = "\n".join(filter(None, [description, f"For example: {choices}."]))
        elif question.kind == "pick_one":
            one_of = [EnumOption(const=c.value, title=c.title) for c in question.choices]
        properties[question.id] = ElicitationStringPropertySchema(
            type="string", title=question.label, description=description, one_of=one_of
        )
    return ElicitationSchema(
        title=need.heading,
        properties=properties,
        required=[q.id for q in need.questions if q.required],
    )


def pool_form_schema(need: NeedInputForm | NeedInput) -> dict[str, Any] | None:
    """The ``_poolside/elicitation`` schema of a form; ``None`` for a plain question."""
    if not isinstance(need, NeedInputForm):
        return None
    properties = {}
    for question in need.questions:
        prop: dict[str, Any] = {"title": question.label}
        if question.help:
            prop["description"] = question.help
        if question.kind == "text":
            prop["type"] = "string"
        elif question.kind == "integer":
            prop["type"] = "integer"
            if question.minimum is not None:
                prop["minimum"] = question.minimum
            if question.maximum is not None:
                prop["maximum"] = question.maximum
        elif question.kind == "boolean":
            prop["type"] = "boolean"
        else:
            picker = {"oneOf": [{"const": c.value, "title": c.title} for c in question.choices]}
            if not question.allow_other:
                prop.update(type="string", **picker)
            else:
                prop["anyOf"] = [picker, {"type": "string"}]
        properties[question.id] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [q.id for q in need.questions if q.required],
    }


def answer_from_content(
    need: NeedInputForm, content: dict[str, Any] | None
) -> dict[str, FormValue]:
    """An accepted form's content, checked (and typed) against the form."""
    response = need.validate_response(FormResponse(action="accept", content=content))
    assert response.content is not None
    return response.content


__all__ = ["answer_from_content", "need_input_schema", "pool_form_schema"]
