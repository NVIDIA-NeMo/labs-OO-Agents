# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``NeedInput`` question as an ACP elicitation form, and the answer back.

ACP forms take flat primitive properties only (string, number, integer,
boolean, and multi-select arrays of strings), so a pydantic
``model_json_schema()`` cannot be passed through. ``need_input_schema``
builds the form directly and returns ``None`` for anything it cannot
flatten; the host then asks in free text.
"""

import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from acp.schema import (
    ElicitationBooleanPropertySchema,
    ElicitationIntegerPropertySchema,
    ElicitationMultiSelectPropertySchema,
    ElicitationNumberPropertySchema,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    StringMultiSelectItems,
)
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from nooa.interactive import NeedInput

_ANSWER = "answer"


def need_input_schema(need: NeedInput) -> ElicitationSchema | None:
    """The form for a question, or ``None`` when its ``answer_type`` is not flat.

    - ``options``: one required string property ``answer`` with an ``enum``,
      titled with the question.
    - neither ``options`` nor ``answer_type``: one required string ``answer``.
    - ``answer_type``: one property per field. ``str``, ``int``, ``float``
      and ``bool`` are string, integer, number and boolean; a ``Literal``
      of strings is a string with an ``enum``; ``list[Literal[...]]`` of
      strings is a multi-select array; ``X | None`` is ``X`` and not
      required. ``Annotated`` wrappers are looked through; the length,
      ``ge``/``le`` and pattern constraints they (or the field) carry are
      kept where the form has an equivalent, others are left out. Any other
      field type gives ``None``.
    """
    if need.options is not None:
        return ElicitationSchema(
            properties={
                _ANSWER: ElicitationStringPropertySchema(
                    type="string", title=need.question, enum=list(need.options)
                )
            },
            required=[_ANSWER],
        )
    if need.answer_type is None:
        return ElicitationSchema(
            properties={
                _ANSWER: ElicitationStringPropertySchema(type="string", title=need.question)
            },
            required=[_ANSWER],
        )
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, field in need.answer_type.model_fields.items():
        annotation, optional, metadata = _unwrap_optional(field.annotation)
        prop = _property(annotation, field, name, [*field.metadata, *metadata])
        if prop is None:
            return None
        properties[name] = prop
        if field.is_required() and not optional:
            required.append(name)
    return ElicitationSchema(title=need.question, properties=properties, required=required)


def answer_from_content(need: NeedInput, content: dict[str, Any] | None) -> Any:
    """The item to submit for an accepted form.

    The chosen option or the text for ``options`` and free-text questions;
    an ``answer_type`` instance for typed ones (the raw content when it does
    not validate, so the agent still sees what the person entered).
    """
    content = dict(content or {})
    if need.answer_type is None:
        return str(content.get(_ANSWER, ""))
    try:
        return need.answer_type.model_validate(content)
    except ValidationError:
        return content


def _strip_annotated(annotation: Any, metadata: list[Any]) -> Any:
    """``annotation`` without ``Annotated``; its metadata is appended to ``metadata``."""
    while get_origin(annotation) is Annotated:
        annotation, *extras = get_args(annotation)
        for extra in extras:
            metadata.extend(extra.metadata if isinstance(extra, FieldInfo) else [extra])
    return annotation


def _unwrap_optional(annotation: Any) -> tuple[Any, bool, list[Any]]:
    """``(type, optional, metadata)``: ``X | None`` and ``Annotated`` unwrapped.

    Pydantic strips a field's outer ``Annotated`` itself, but not one inside
    ``| None``.
    """
    metadata: list[Any] = []
    annotation = _strip_annotated(annotation, metadata)
    optional = False
    if get_origin(annotation) in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1 and len(members) < len(get_args(annotation)):
            annotation, optional = members[0], True
    return _strip_annotated(annotation, metadata), optional, metadata


def _constraints(metadata: list[Any]) -> dict[str, Any]:
    """The constraints in pydantic/annotated-types metadata, by attribute name."""
    found: dict[str, Any] = {}
    for item in metadata:
        for attribute in ("min_length", "max_length", "ge", "le", "pattern"):
            value = getattr(item, attribute, None)
            if value is not None:
                found[attribute] = getattr(
                    value, "pattern", value
                )  # a compiled re.Pattern as its source
    return found


def _string_literals(annotation: Any) -> list[str] | None:
    if get_origin(annotation) is not Literal:
        return None
    values = get_args(annotation)
    if not values or not all(isinstance(value, str) for value in values):
        return None
    return list(values)


def _property(annotation: Any, field: FieldInfo, name: str, metadata: list[Any]) -> Any:
    title = field.title or name.replace("_", " ").title()
    common: dict[str, Any] = {"title": title, "description": field.description}
    default = None if field.is_required() else field.default
    limits = _constraints(metadata)
    if annotation is bool:
        return ElicitationBooleanPropertySchema(
            type="boolean", default=default if isinstance(default, bool) else None, **common
        )
    if annotation is int:
        return ElicitationIntegerPropertySchema(
            type="integer",
            default=default if isinstance(default, int) and not isinstance(default, bool) else None,
            minimum=limits.get("ge") if isinstance(limits.get("ge"), int) else None,
            maximum=limits.get("le") if isinstance(limits.get("le"), int) else None,
            **common,
        )
    if annotation is float:
        return ElicitationNumberPropertySchema(
            type="number",
            default=default if isinstance(default, (int, float)) else None,
            minimum=limits.get("ge") if isinstance(limits.get("ge"), (int, float)) else None,
            maximum=limits.get("le") if isinstance(limits.get("le"), (int, float)) else None,
            **common,
        )
    if annotation is str:
        return ElicitationStringPropertySchema(
            type="string",
            default=default if isinstance(default, str) else None,
            min_length=limits.get("min_length"),
            max_length=limits.get("max_length"),
            pattern=limits.get("pattern") if isinstance(limits.get("pattern"), str) else None,
            **common,
        )
    if (choices := _string_literals(annotation)) is not None:
        return ElicitationStringPropertySchema(
            type="string",
            enum=choices,
            default=default if isinstance(default, str) else None,
            **common,
        )
    if get_origin(annotation) is list:
        (item,) = get_args(annotation) or (None,)
        if (choices := _string_literals(_strip_annotated(item, []))) is not None:
            return ElicitationMultiSelectPropertySchema(
                type="array",
                items=StringMultiSelectItems(type="string", enum=choices),
                default=(
                    list(default)
                    if isinstance(default, (list, tuple)) and all(v in choices for v in default)
                    else None
                ),
                min_items=limits.get("min_length"),
                max_items=limits.get("max_length"),
                **common,
            )
    return None


__all__ = ["answer_from_content", "need_input_schema"]
