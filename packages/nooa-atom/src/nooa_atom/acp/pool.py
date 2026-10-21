# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool's form extension (``_poolside/elicitation``), the only Pool-specific form code.

Pool declines integer, boolean and optional non-string properties, so every
property is sent as a string, with constraints as hints in its description.
Delete this module (and its branch in ``AtomACPAgent._ask``) once Pool supports standard ACP.
"""

from typing import Any

from nooa.interactive import NeedInput
from nooa_atom.session.forms import NeedInputForm, form_default_text, form_hints

POOL_FORM_METHOD = "poolside/elicitation"
"""Pool's form request; ``ext_method`` sends it as ``_poolside/elicitation``."""

_YES_NO = ({"const": "true", "title": "Yes"}, {"const": "false", "title": "No"})


def pool_form_params(session_id: str, need: NeedInputForm) -> dict[str, Any]:
    """The ``_poolside/elicitation`` request parameters for ``need``."""
    schema = pool_form_schema(need)
    assert schema is not None
    params: dict[str, Any] = {
        "sessionId": session_id,
        "mode": "form",
        "message": f"{need.heading}\n\n{need.reason}" if need.reason else need.heading,
        "requestedSchema": schema,
    }
    if len(schema["properties"]) > 1:
        params["_meta"] = {"poolside/field_order": list(schema["properties"])}
    return params


def pool_form_schema(need: NeedInputForm | NeedInput) -> dict[str, Any] | None:
    """The ``requestedSchema`` Pool can show for ``need``; ``None`` for a plain question."""
    if not isinstance(need, NeedInputForm):
        return None
    return {
        "type": "object",
        "properties": {question.id: _property(question) for question in need.questions},
        "required": [q.id for q in need.questions if q.required],
    }


def _property(question: Any) -> dict[str, Any]:
    prop: dict[str, Any] = {"title": question.label}
    hints: list[str] = []
    kind = question.kind
    choices = getattr(question, "choices", [])
    picker = {"oneOf": [{"const": c.value, "title": c.title} for c in choices]}
    if kind == "pick_one" and question.required and not question.allow_other:
        prop.update(type="string", **picker)
    elif kind == "boolean" and question.required:
        prop.update(type="string", oneOf=[*_YES_NO])
    else:
        prop["type"] = "string"
        hints.append(_downgrade_hint(question))
    if getattr(question, "default", None) is not None:
        hints.append(f"Default: {form_default_text(question)}.")
    description = "\n".join(filter(None, [question.help, *hints]))
    if description:
        prop["description"] = description
    return prop


def _downgrade_hint(question: Any) -> str:
    """What a plain text box must contain, as one sentence (defaults excluded)."""
    kind = question.kind
    titles = ", ".join(c.title for c in getattr(question, "choices", []))
    if kind == "pick_one":
        lead = "For example" if question.allow_other else "One of"
        return f"{lead}: {titles}."
    if kind == "boolean":
        return "Yes or no."
    hints = form_hints(question, choices=False, default=False)
    if kind == "pick_many":
        return "; ".join([f"Comma-separated; choose from: {titles}", *hints]) + "."
    text = "; ".join(hints)
    return f"{text[:1].upper()}{text[1:]}." if text else ""


__all__ = ["POOL_FORM_METHOD", "pool_form_params", "pool_form_schema"]
