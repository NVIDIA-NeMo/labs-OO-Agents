# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a ``NeedInputForm`` to a standard ACP form elicitation schema (``requestedSchema``)."""

from typing import Any

from acp.schema import (
    ElicitationBooleanPropertySchema,
    ElicitationIntegerPropertySchema,
    ElicitationMultiSelectPropertySchema,
    ElicitationNumberPropertySchema,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    EnumOption,
    TitledMultiSelectItems,
)

from nooa.interactive import NeedInput, NeedInputForm


def need_input_schema(need: NeedInputForm | NeedInput) -> ElicitationSchema | None:
    """The ACP elicitation schema of a form; ``None`` for a plain question."""
    if not isinstance(need, NeedInputForm):
        return None
    return ElicitationSchema(
        type="object",
        title=need.heading,
        properties={question.id: _property(question) for question in need.questions},
        required=[q.id for q in need.questions if q.required],
    )


def _property(question: Any) -> Any:
    common = {"title": question.label, "description": question.help}
    kind = question.kind
    if kind == "number":
        schema = (
            ElicitationIntegerPropertySchema if question.whole else ElicitationNumberPropertySchema
        )
        return schema(
            type="integer" if question.whole else "number",
            minimum=question.minimum,
            maximum=question.maximum,
            default=question.default,
            **common,
        )
    if kind == "boolean":
        return ElicitationBooleanPropertySchema(type="boolean", default=question.default, **common)
    if kind == "pick_many":
        return ElicitationMultiSelectPropertySchema(
            type="array",
            items=TitledMultiSelectItems(any_of=_options(question)),
            min_items=question.min_items,
            max_items=question.max_items,
            default=question.default,
            **common,
        )
    if kind == "pick_one":
        if question.allow_other:
            examples = ", ".join(c.value for c in question.choices)
            common["description"] = "\n".join(
                filter(None, [question.help, f"For example: {examples}."])
            )
            return ElicitationStringPropertySchema(
                type="string", default=question.default, **common
            )
        return ElicitationStringPropertySchema(
            type="string", one_of=_options(question), default=question.default, **common
        )
    return ElicitationStringPropertySchema(
        type="string",
        min_length=question.min_length,
        max_length=question.max_length,
        pattern=question.pattern,
        format=question.format,
        default=question.default,
        **common,
    )


def _options(question: Any) -> list[EnumOption]:
    return [EnumOption(const=c.value, title=c.title) for c in question.choices]


__all__ = ["need_input_schema"]
