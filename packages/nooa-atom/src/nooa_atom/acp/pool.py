# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool's form extension: the only Pool-specific elicitation code.

Pool (1.0.16) asks forms with ``_poolside/elicitation`` instead of standard
ACP ``elicitation/create``, and shows only part of the standard schema. What
was checked against Pool:

- it shows one question per page, renders ``type: string`` as a text box
  (multi-line descriptions are fine) and a required ``type: string`` +
  ``oneOf`` of ``{const, title}`` as a radio picker, and honours ``required``
  and ``_meta["poolside/field_order"]``;
- a picker-or-text (``anyOf: [{oneOf: ...}, {type: string}]``) shows as a bare
  text box without the choices, so it is not used;
- it declines the whole form itself (``{"action": "decline"}``) when a
  property is ``type: integer`` or ``type: boolean``, or when an optional
  property is not a plain string.

``number``, ``array``, ``minLength``/``maxLength``, ``pattern``, ``format`` and
``default`` are untested. So every property sent is ``type: string`` (a
``oneOf`` picker only when required) with only ``title``, ``description``,
``oneOf`` and the top-level ``required``, and every question is shown; other
question types and constraints become a string whose description says what
to enter:

- text constraints, formats and defaults -> hints in the description;
- integer / number -> a string with the range ("A whole number from 1 to 10.");
- boolean -> a strict Yes/No picker when required, a "yes or no" string when optional;
- a picker that allows other answers, or an optional picker -> a string
  listing the choices;
- pick many -> a comma-separated string listing the choices;
- a default -> "Default: x." in the description.

Pool's answers reach the agent as the strings the person typed.

Delete this module (and its branch in ``AtomACPAgent._ask``) once Pool
supports standard ACP elicitation.
"""

from typing import Any

from nooa.interactive import NeedInput, NeedInputForm, form_hints

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
        hints.append(f"Default: {_default(question)}.")
    description = "\n".join(filter(None, [question.help, *hints]))
    if description:
        prop["description"] = description
    return prop


def _downgrade_hint(question: Any) -> str:
    """What a plain text box must contain, as one sentence (defaults excluded)."""
    hints = [h for h in form_hints(question) if not h.startswith("default: ")]
    kind = question.kind
    titles = ", ".join(c.title for c in getattr(question, "choices", []))
    if kind == "pick_one":
        lead = "For example" if question.allow_other else "One of"
        return f"{lead}: {titles}."
    if kind == "pick_many":
        rest = [h for h in hints if not h.startswith("choose any of: ")]
        return "; ".join([f"Comma-separated; choose from: {titles}", *rest]) + "."
    if kind == "boolean":
        return "Yes or no."
    text = "; ".join(hints)
    return f"{text[:1].upper()}{text[1:]}." if text else ""


def _default(question: Any) -> str:
    default = question.default
    titles = {c.value: c.title for c in getattr(question, "choices", [])}
    if isinstance(default, bool):
        return "yes" if default else "no"
    if isinstance(default, list):
        return ", ".join(titles.get(v, v) for v in default)
    if isinstance(default, float) and default.is_integer():
        return str(int(default))
    return titles.get(default, str(default))


__all__ = ["POOL_FORM_METHOD", "pool_form_params", "pool_form_schema"]
