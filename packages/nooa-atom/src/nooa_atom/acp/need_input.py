# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted standard forms and protocol-only choice/Other companions."""

from typing import Any

from acp.schema import ElicitationSchema

from nooa.forms import FormResponse, NeedInputForm, _validate_form_response
from nooa.interactive import NeedInput


def _other_ids(need: NeedInputForm) -> dict[str, str]:
    # Reserve the complete authored namespace. Never infer companion IDs from
    # suffixes or reinterpret a stored answer as a selection by coincidence.
    used = {q.id for q in need.questions}
    result = {}
    for index, q in enumerate(need.questions):
        if q.kind not in ("choice", "multi_choice"):
            continue
        key = f"__acp_other_{index}"
        while key in used:
            key += "_"
        used.add(key)
        result[q.id] = key
    return result


def _enum(q, *, multiple=False) -> dict[str, Any]:
    if q.choices[0].title is None:
        return {"enum": [c.value for c in q.choices]}
    return {
        "anyOf" if multiple else "oneOf": [
            {
                "const": c.value,
                "title": c.title,
                **({"description": c.description} if c.description is not None else {}),
            }
            for c in q.choices
        ]
    }


def _property(q) -> dict[str, Any]:
    prop = {"title": q.label}
    if q.help is not None:
        prop["description"] = q.help
    if q.kind in ("choice", "multi_choice"):
        if q.kind == "choice":
            prop.update(type="string", **_enum(q))
        else:
            prop.update(type="array", items={"type": "string", **_enum(q, multiple=True)})
        listed = {c.value for c in q.choices}
        if q.default is not None:
            if q.kind == "choice" and q.default in listed:
                prop["default"] = q.default
            elif q.kind == "multi_choice":
                selected = [v for v in q.default if v in listed]
                if selected or not q.min_items:
                    prop["default"] = selected
    else:
        prop["type"] = "string" if q.kind == "text" else q.kind
        for name, alias in (
            ("min_length", "minLength"),
            ("max_length", "maxLength"),
            ("pattern", "pattern"),
            ("format", "format"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("default", "default"),
        ):
            value = getattr(q, name, None)
            if value is not None:
                prop[alias] = value
    if q.kind == "multi_choice":
        for name, alias in (("min_items", "minItems"), ("max_items", "maxItems")):
            value = getattr(q, name)
            if value is not None:
                prop[alias] = value
    return prop


def need_input_schema(need: NeedInputForm | NeedInput) -> ElicitationSchema | None:
    """Paired optional choice/Other fields, collapsed only at the boundary.

    Restricted ACP cannot express required-either or aggregate array bounds
    across companions. Logical constraints are checked by the owner. Listed
    arrays retain standard bounds; Other adds at most one item.
    """
    if not isinstance(need, NeedInputForm):
        return None
    properties = {}
    companions = _other_ids(need)
    for q in need.questions:
        if q.id in companions and q.default is not None:
            listed = {c.value for c in q.choices}
            values = q.default if q.kind == "multi_choice" else [q.default]
            if "" in values:
                raise ValueError(
                    f"{q.id}: empty written Other default cannot be represented by standard ACP"
                )
            if q.kind == "multi_choice":
                if any(v not in listed for v in values[:-1]):
                    raise ValueError(f"{q.id}: standard default's written Other must be last")
                selected = [v for v in values if v in listed]
                if selected and q.min_items is not None and len(selected) < q.min_items:
                    raise ValueError(
                        f"{q.id}: listed default is below the standard picker minItems"
                    )
        properties[q.id] = _property(q)
        if q.id in companions:
            other = {"type": "string", "title": f"{q.label} — Other"}
            listed = {c.value for c in q.choices}
            if q.kind == "choice" and q.default is not None and q.default not in listed:
                other["default"] = q.default
            elif q.kind == "multi_choice" and q.default is not None:
                custom = [v for v in q.default if v not in listed]
                if len(custom) > 1:
                    raise ValueError("standard multiselect has only one Other input")
                if custom:
                    other["default"] = custom[0]
            properties[companions[q.id]] = other
    return ElicitationSchema.model_validate(
        {
            "type": "object",
            "title": need.heading,
            "properties": properties,
            "required": [q.id for q in need.questions if q.required and q.id not in companions],
        }
    )


def _fold_standard_form_response(need: NeedInputForm, envelope: dict[str, Any]) -> FormResponse:
    """Check wire values BEFORE folding. Written Other overrides single selection.

    Empty Other means unused. Multiselect appends one nonempty Other unless
    already selected. No notes or generated keys reach the request owner.
    """
    if envelope.get("action") != "accept":
        return FormResponse(action=envelope.get("action"))
    content = envelope.get("content")
    if content is None:
        content = {}
    wire = FormResponse(action="accept", content=content).content
    assert wire is not None
    companions = _other_ids(need)
    unknown = wire.keys() - ({q.id for q in need.questions} | set(companions.values()))
    if unknown:
        raise ValueError(f"Unknown answer ids: {sorted(unknown)}")
    answers = {}
    for q in need.questions:
        if q.id not in companions:
            if q.id in wire:
                answers[q.id] = wire[q.id]
            continue
        selected = wire.get(q.id)
        if q.id in wire:
            q.validate_answer(selected)
            values = selected if q.kind == "multi_choice" else [selected]
            if any(v not in {c.value for c in q.choices} for v in values):
                raise ValueError(f"{q.id}: invalid enum value on selection field")
        other = wire.get(companions[q.id], "")
        if type(other) is not str:
            raise ValueError("Other must be a string")
        if q.kind == "choice":
            if other != "":
                answers[q.id] = other
            elif q.id in wire:
                answers[q.id] = selected
        elif other != "" or q.id in wire:
            values = list(selected) if q.id in wire else []
            if other != "" and other not in values:
                values.append(other)
            answers[q.id] = values
    return FormResponse(action="accept", content=answers)


def standard_form_response(need: NeedInputForm, envelope: dict[str, Any]) -> FormResponse:
    """Fold the standard companions, then check the native answer synchronously."""
    return need.validate_response(_fold_standard_form_response(need, envelope))


async def _standard_form_response_async(
    need: NeedInputForm, envelope: dict[str, Any]
) -> FormResponse:
    return await _validate_form_response(need, _fold_standard_form_response(need, envelope))


def pool_form_schema(need: NeedInputForm | NeedInput) -> dict[str, Any] | None:
    """Pool 1.0.16: equivalent required choice-or-text and text controls only.

    Unsupported fields/options leave the WHOLE form as text. Optional choices
    are never made required merely to render a Pool picker.
    """
    if not isinstance(need, NeedInputForm):
        return None
    properties = {}
    for q in need.questions:
        if q.kind == "text":
            if any(
                getattr(q, key) is not None
                for key in ("default", "min_length", "max_length", "pattern", "format")
            ):
                return None
            prop = _property(q)
        elif q.kind == "choice" and q.required and q.default is None:
            if any(c.description is not None for c in q.choices):
                return None
            picker = {"oneOf": [{"const": c.value, "title": c.title or c.value} for c in q.choices]}
            prop = {"title": q.label, "anyOf": [picker, {"type": "string"}]}
            if q.help is not None:
                prop["description"] = q.help
        else:
            return None
        properties[q.id] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [q.id for q in need.questions if q.required],
    }


__all__ = ["need_input_schema", "pool_form_schema", "standard_form_response"]
