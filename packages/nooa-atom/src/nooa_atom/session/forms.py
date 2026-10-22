# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit forms: Atom's ``NeedInputForm`` turn result and its ``FormResponse`` answer.

Core ``nooa.interactive`` knows only the conversational ``NeedInput``. Atom
adds ``NeedInputForm`` as an extra return type of its turn methods; the
form's question types, its answer and the readable-text helpers used by the
session store and the ACP adapters live here.

``NeedInputForm`` sets ``_turn_kind = "need_input"``, so the core turn
loop settles it as an input request; the Atom session tells it apart by
type and records it as ``need_input_form``.
"""

import re
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from nooa import hidden
from nooa.interactive import NeedInput
from nooa.runtime.turn_loop import ResultKind


def _non_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class FormChoice(BaseModel):
    """One option of a picker: the stored ``value`` and the ``title`` shown."""

    model_config = {"extra": "forbid", "strict": True}
    value: str = Field(description="What the answer contains when this option is chosen")
    title: str = Field(description="What the person sees")
    _check_value = field_validator("value")(_non_blank)
    _check_title = field_validator("title")(_non_blank)


class _FormQuestionBase(BaseModel):
    """Fields every form question has (documented once on ``NeedInputForm``)."""

    model_config = {"extra": "forbid", "strict": True}
    id: Annotated[str, hidden] = Field(description="Key of this answer in FormResponse.content")
    label: Annotated[str, hidden] = Field(description="The question shown to the person")
    help: Annotated[str | None, hidden] = Field(default=None, description="Optional hint")
    required: Annotated[bool, hidden] = Field(
        default=True, description="False lets the person skip it"
    )
    _check_id = field_validator("id")(_non_blank)
    _check_label = field_validator("label")(_non_blank)


TextFormat = Literal["email", "uri", "date", "date-time"]


class TextQuestion(_FormQuestionBase):
    """Free text; the answer is a ``str``."""

    kind: Annotated[Literal["text"], hidden] = "text"
    default: str | None = Field(default=None, description="Prefilled answer")
    min_length: int | None = Field(default=None, ge=0, description="Fewest characters")
    max_length: int | None = Field(default=None, ge=1, description="Most characters")
    pattern: str | None = Field(default=None, description="Regex the whole answer must match")
    format: TextFormat | None = Field(
        default=None, description="email, uri, date (YYYY-MM-DD) or date-time (ISO 8601)"
    )

    @model_validator(mode="before")
    @classmethod
    def _check_constraints(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _ordered(data.get("min_length"), data.get("max_length"), "min_length", "max_length")
            if isinstance(data.get("pattern"), str):
                try:
                    re.compile(data["pattern"])
                except re.error as exc:
                    raise ValueError(f"pattern is not a valid regex: {exc}") from None
        return data


class PickOneQuestion(_FormQuestionBase):
    """Pick one of ``choices``; the answer is the chosen ``value`` (a ``str``)."""

    kind: Annotated[Literal["pick_one"], hidden] = "pick_one"
    default: str | None = Field(default=None, description="Preselected value")
    choices: list[FormChoice] = Field(min_length=1, description="The options, in order")
    allow_other: bool = Field(
        default=True, description="True also accepts a typed answer that is not listed"
    )

    _unique = field_validator("choices")(classmethod(lambda cls, value: _unique_choices(value)))


class PickManyQuestion(_FormQuestionBase):
    """Pick any of ``choices``; the answer is a ``list[str]`` of chosen values."""

    kind: Annotated[Literal["pick_many"], hidden] = "pick_many"
    default: list[str] | None = Field(default=None, description="Preselected values")
    choices: list[FormChoice] = Field(min_length=1, description="The options, in order")
    min_items: int | None = Field(default=None, ge=0, description="Fewest options to pick")
    max_items: int | None = Field(default=None, ge=1, description="Most options to pick")

    _unique = field_validator("choices")(classmethod(lambda cls, value: _unique_choices(value)))

    @model_validator(mode="before")
    @classmethod
    def _check_counts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _ordered(data.get("min_items"), data.get("max_items"), "min_items", "max_items")
            low, choices = data.get("min_items"), data.get("choices")
            if isinstance(low, int) and isinstance(choices, list) and low > len(choices):
                raise ValueError("min_items must not exceed the number of choices")
        return data


class NumberQuestion(_FormQuestionBase):
    """A number; ``whole=True`` asks for a whole number (an ``int``)."""

    kind: Annotated[Literal["number"], hidden] = "number"
    whole: bool = Field(default=False, description="True for a whole number")
    default: int | float | None = Field(default=None, description="Prefilled answer")
    minimum: int | float | None = Field(default=None, description="Smallest allowed value")
    maximum: int | float | None = Field(default=None, description="Largest allowed value")

    @model_validator(mode="before")
    @classmethod
    def _check_range(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _ordered(data.get("minimum"), data.get("maximum"), "minimum", "maximum")
        return data


class BooleanQuestion(_FormQuestionBase):
    """Yes or no; the answer is a ``bool``."""

    kind: Annotated[Literal["boolean"], hidden] = "boolean"
    default: bool | None = Field(default=None, description="Preselected answer")


FormQuestion = Annotated[
    TextQuestion | PickOneQuestion | PickManyQuestion | NumberQuestion | BooleanQuestion,
    Field(discriminator="kind"),
]
FormValue = str | int | float | bool | list[str] | None
"""One answer: ``str``, ``list[str]``, ``int``, ``float`` or ``bool``."""


def _ordered(low: Any, high: Any, low_name: str, high_name: str) -> None:
    numbers = (int, float)
    if (
        isinstance(low, numbers)
        and isinstance(high, numbers)
        and not isinstance(low, bool)
        and not isinstance(high, bool)
        and low > high
    ):
        raise ValueError(f"{low_name} must not exceed {high_name}")


def _unique_choices(choices: list[FormChoice]) -> list[FormChoice]:
    if len({choice.value for choice in choices}) != len(choices):
        raise ValueError("choice values must be unique")
    return choices


class FormResponse(BaseModel):
    """The person's answer to a form: ``accept`` with ``content``, ``decline`` or ``cancel``."""

    model_config = {"extra": "forbid", "strict": True}
    action: Literal["accept", "decline", "cancel"] = Field(
        description="accept (answered), decline or cancel"
    )
    content: dict[str, FormValue] | None = Field(
        default=None,
        description="On accept: question id -> answer; skipped questions may be missing",
    )

    @model_validator(mode="after")
    def _content_for_action(self) -> "FormResponse":
        if self.action == "accept" and self.content is None:
            raise ValueError("accept requires content")
        if self.action != "accept" and self.content is not None:
            raise ValueError("decline/cancel must not carry content")
        return self

    def as_text(self) -> str:
        """The response as a person reads it: ``id: answer`` lines, or the action."""
        if self.action != "accept":
            return "Declined the form." if self.action == "decline" else "Cancelled the form."
        lines = [
            f"{key}: {_answer_text(value)}"
            for key, value in (self.content or {}).items()
            if not _empty(value)
        ]
        return "\n".join(lines) or "Submitted the form."


class NeedInputForm(BaseModel):
    """Ask the person to fill in a form; the answer is a ``FormResponse``.

    Every question has ``id`` (its key in ``FormResponse.content``), ``label``
    (the question shown), optional ``help`` and ``required`` (default True).
    Most need nothing else, or maybe ``default``::

        NeedInputForm(heading="Release", questions=[
            TextQuestion(id="branch", label="Which branch should I use?", default="main"),
            BooleanQuestion(id="dry_run", label="Dry run first?", default=True),
        ])

    Add constraints only when the task needs them, and say the rule in words
    in ``help`` (a regex means nothing to a person)::

        TextQuestion(id="slug", label="Project slug", min_length=3, max_length=24,
                     pattern=r"^[a-z0-9-]+$",
                     help="3-24 characters: lowercase letters, digits or hyphens.")

    The client checks answers; you get exactly what the person entered, so
    handle unexpected values. A client that can't show forms sends a plain ``str``.
    """

    model_config = {"extra": "forbid", "strict": True}
    # The core turn loop settles this as an input request (see module doc).
    _turn_kind: ClassVar[ResultKind] = "need_input"
    kind: Literal["form"] = "form"
    heading: str = Field(description="What the form is for")
    reason: str | None = Field(default=None, description="Optional: why you are asking")
    questions: list[FormQuestion] = Field(min_length=1, description="The fields, in order")
    _check_heading = field_validator("heading")(_non_blank)

    @field_validator("questions")
    @classmethod
    def _unique_ids(cls, questions: list[FormQuestion]) -> list[FormQuestion]:
        if len({question.id for question in questions}) != len(questions):
            raise ValueError("question ids must be unique")
        return questions

    def as_text(self) -> str:
        """The form as a person reads it: heading, reason, one line per question."""
        lines = []
        for question in self.questions:
            line = f"- {question.label}" + (f" ({question.help})" if question.help else "")
            hints = form_hints(question)
            if hints:
                line += ": " + "; ".join(hints)
            lines.append(line)
        return "\n\n".join(part for part in (self.heading, self.reason, "\n".join(lines)) if part)


def _empty(value: Any) -> bool:
    """No answer: missing, blank text, or nothing picked."""
    return value is None or value == [] or (isinstance(value, str) and not value.strip())


_FORMAT_TEXT = {
    "email": "an email address",
    "uri": "a URI such as https://example.com",
    "date": "a date (YYYY-MM-DD)",
    "date-time": "a date and time (YYYY-MM-DDThh:mm)",
}


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _range_text(question: Any) -> str:
    noun = "a whole number" if question.whole else "a number"
    low, high = question.minimum, question.maximum
    if low is not None and high is not None:
        return f"{noun} from {_number_text(low)} to {_number_text(high)}"
    if low is not None:
        return f"{noun} of at least {_number_text(low)}"
    if high is not None:
        return f"{noun} of at most {_number_text(high)}"
    return noun


def _bounds_text(low: int | None, high: int | None, noun: str) -> str:
    last = high if high is not None else low
    if last == 1:
        noun = noun.removesuffix("s")
    if low is not None and high is not None:
        return f"{low} {noun}" if low == high else f"{low} to {high} {noun}"
    if low is not None:
        return f"at least {low} {noun}"
    if high is not None:
        return f"at most {high} {noun}"
    return ""


def _length_text(question: Any) -> str:
    return _bounds_text(question.min_length, question.max_length, "characters")


def _count_text(question: Any) -> str:
    return _bounds_text(question.min_items, question.max_items, "options")


def _answer_text(value: FormValue) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(value)
    if isinstance(value, float):
        return _number_text(value)
    return str(value)


@hidden
def form_hints(question: Any, *, choices: bool = True, default: bool = True) -> list[str]:
    """Readable hints for one question; ``choices``/``default`` include those parts."""
    kind = question.kind
    hints: list[str] = []
    if kind == "pick_one":
        if choices:
            hints.append(", ".join(choice.title for choice in question.choices))
    elif kind == "pick_many":
        if choices:
            hints.append("choose any of: " + ", ".join(c.title for c in question.choices))
        if _count_text(question):
            hints.append(_count_text(question))
    elif kind == "number":
        hints.append(_range_text(question))
    elif kind == "boolean":
        hints.append("yes or no")
    else:
        if _length_text(question):
            hints.append(_length_text(question))
        if question.format:
            hints.append(_FORMAT_TEXT[question.format])
        if question.pattern:
            hints.append(f"matching {question.pattern}")
    if default and getattr(question, "default", None) is not None:
        hints.append(f"default: {form_default_text(question)}")
    return hints


@hidden
def form_default_text(question: Any) -> str:
    """A question's default as a person reads it (choice titles, yes/no)."""
    default = question.default
    titles = {choice.value: choice.title for choice in getattr(question, "choices", [])}
    if isinstance(default, list):
        return ", ".join(titles.get(value, value) for value in default)
    if isinstance(default, str):
        return titles.get(default, default)
    return _answer_text(default)


@hidden
def recorded_form_text(record: dict[str, Any]) -> str | None:
    """A recorded form request as readable text; ``None`` if it no longer validates."""
    data = {k: v for k, v in record.items() if k not in ("request_id", "question_id")}
    try:
        return NeedInputForm.model_validate(data).as_text()
    except ValueError:
        return None


@hidden
def recorded_item_text(item_json: str, item_type: str = "") -> str | None:
    """A recorded ``FormResponse`` item as readable text; ``None`` for anything else."""
    if not item_type.endswith(":FormResponse"):
        return None
    try:
        return FormResponse.model_validate_json(item_json).as_text()
    except ValueError:
        return None


InputRequest = Annotated[NeedInput | NeedInputForm, Field(discriminator="kind")]
"""Structured request outputs require kind=question/form; constructors supply it."""
