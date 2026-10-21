# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive agent base: queue-driven turn loop, persistent vars, summarization.

``InteractiveAgent`` is the base class for agents driven by an outer
dispatcher — a terminal UI, a harness, or any host that feeds input
queues and re-enters ``handle()`` once per notification. It provides:

* ``self.user_messages`` via ``QueueManager``; hosts declare whatever further
  queues they need (``self.queue_manager.queue("name")``),
* ``self.v`` — snapshot-backed persistent variables that survive turns
  and sessions,
* ``message()`` — send a Markdown message to the user,
* the turn protocol: ``handle()`` returns ``Done``, ``NeedInput``, ``NeedInputForm`` or
  ``Waiting``; ``handle_batch()`` runs unattended turns and returns ``Done`` or ``Waiting``;
  ``handle_work()`` is a child session's worker turn, whose ``Done`` carries a ``TaskResult``,
* token-budget history summarization (``install_summarizer`` /
  ``apply_model_limits``).

``nooa_cli.coding.CodingAgent`` adds the shared coding tools used by
interactive hosts such as the TUI and ACP.
"""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from nooa import hidden, strategy
from nooa.agentdoc import doc  # noqa: F401  # visible in the REPL, as on base
from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role
from nooa.storage.markers import nosnapshot
from nooa.storage.snapshot_vars import SnapshotVars

with hidden:
    from collections.abc import Callable

    from nooa import Agent, no_trace
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig, PredictConfig  # noqa: F401
    from nooa.events import _json_safe
    from nooa.runtime.channels import Channel, ChannelReader, QueueManager
    from nooa.runtime.producers_skill import ProducersSkill
    from nooa.runtime.turn_loop import TurnLoop
    from nooa.strategies import CodeActStrategy
    from nooa.strategy_validation import InvariantError

# Standard library — all visible in REPL
import asyncio  # noqa: F401
import datetime  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401

from nooa.runtime import producers  # noqa: F401
from nooa.runtime.producers import after, cron, monitor, run_job, tail  # noqa: F401

# Optional third-party libraries — visible in REPL (use np, pd, px, go directly)
try:
    import numpy as np  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import pandas as pd  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import plotly.express as px  # noqa: F401  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # noqa: F401  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import scipy  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import sklearn  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

with hidden:
    from nooa.unifiedllm import UnifiedLLM


def _non_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


def _blank_to_none(value: str | None) -> str | None:
    return None if value is None else (value.strip() or None)


class Done(BaseModel):
    """Turn result: the work for this turn is finished.

    ``message`` is the preferred way to answer the user on a turn that
    handled a user message: put the reply here and the host shows it.
    ``self.message()`` remains for text you want to show before the turn
    ends. ``explanation`` stays a short status line. ``evidence`` is
    optional: short factual lines a reviewer can check, used when you
    verified something. ``result`` is set only when a delegated objective
    or benchmark task completes; it is then a ``TaskResult`` (a worker
    turn, ``handle_work``, requires one; other turns may leave it unset).
    """

    explanation: str = Field(description="Status line saying what was finished; not the reply")
    message: str | None = Field(default=None, description="The reply to show the user")
    evidence: list[str] = Field(
        default_factory=list,
        description='Short checkable facts, e.g. "pytest tests/x.py: 24 passed"',
    )
    result: Any | None = Field(
        default=None, description="TaskResult when a delegated or bench task completes"
    )

    _check_explanation = field_validator("explanation")(_non_blank)
    _check_message = field_validator("message")(_blank_to_none)

    @field_serializer("result", when_used="json")
    def _serialize_result(self, value: Any) -> Any:
        """Keep JSON dumps (the event store) from raising on an arbitrary object."""
        return _json_safe(value)


class NeedInput(BaseModel):
    """Conversational question, answered by the person's next message; never a dialog.

    The answer arrives as a plain ``str``. Put any suggested answers in the
    question text; use NeedInputForm when you want pickers or several fields.
    """

    model_config = {"extra": "forbid"}
    kind: Literal["question"] = "question"
    question: str
    reason: str | None = None

    _check_question = field_validator("question")(_non_blank)


class FormChoice(BaseModel):
    """One option of a picker: the stored ``value`` and the ``title`` shown."""

    model_config = {"extra": "forbid", "strict": True}
    value: str = Field(description="What the answer contains when this option is chosen")
    title: str = Field(description="What the person sees")
    _check_value = field_validator("value")(_non_blank)
    _check_title = field_validator("title")(_non_blank)


class _FormQuestionBase(BaseModel):
    """Fields shared by every form question."""

    model_config = {"extra": "forbid", "strict": True}
    id: str = Field(description="Key of this answer in FormResponse.content")
    label: str = Field(description="The question shown to the person")
    help: str | None = Field(default=None, description="Optional hint shown with the question")
    required: bool = Field(default=True, description="False lets the person leave it empty")
    _check_id = field_validator("id")(_non_blank)
    _check_label = field_validator("label")(_non_blank)

    @model_validator(mode="after")
    def _check_default(self) -> Any:
        default = getattr(self, "default", None)
        if default is not None:
            problem = _problem(self, default)
            if problem:
                raise ValueError(f"default {default!r} is invalid: {problem}")
        return self


TextFormat = Literal["email", "uri", "date", "date-time"]


class TextQuestion(_FormQuestionBase):
    """Free text; the answer is a ``str`` ("" when optional and left empty)."""

    kind: Literal["text"] = Field(default="text", description="Set for you")
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

    kind: Literal["pick_one"] = Field(default="pick_one", description="Set for you")
    default: str | None = Field(default=None, description="Preselected value")
    choices: list[FormChoice] = Field(min_length=1, description="The options, in order")
    allow_other: bool = Field(
        default=True, description="True also accepts a typed answer that is not listed"
    )

    _unique = field_validator("choices")(classmethod(lambda cls, value: _unique_choices(value)))


class PickManyQuestion(_FormQuestionBase):
    """Pick any of ``choices``; the answer is a ``list[str]`` of chosen values."""

    kind: Literal["pick_many"] = Field(default="pick_many", description="Set for you")
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
    """A number; the answer is a ``float``."""

    kind: Literal["number"] = Field(default="number", description="Set for you")
    default: float | None = Field(default=None, description="Prefilled answer")
    minimum: float | None = Field(default=None, description="Smallest allowed value")
    maximum: float | None = Field(default=None, description="Largest allowed value")

    @model_validator(mode="before")
    @classmethod
    def _check_range(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _ordered(data.get("minimum"), data.get("maximum"), "minimum", "maximum")
        return data


class IntegerQuestion(_FormQuestionBase):
    """A whole number; the answer is an ``int``."""

    kind: Literal["integer"] = Field(default="integer", description="Set for you")
    default: int | None = Field(default=None, description="Prefilled answer")
    minimum: int | None = Field(default=None, description="Smallest allowed value")
    maximum: int | None = Field(default=None, description="Largest allowed value")

    @model_validator(mode="before")
    @classmethod
    def _check_range(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _ordered(data.get("minimum"), data.get("maximum"), "minimum", "maximum")
        return data


class BooleanQuestion(_FormQuestionBase):
    """Yes or no; the answer is a ``bool``."""

    kind: Literal["boolean"] = Field(default="boolean", description="Set for you")
    default: bool | None = Field(default=None, description="Preselected answer")


FormQuestion = Annotated[
    TextQuestion
    | PickOneQuestion
    | PickManyQuestion
    | NumberQuestion
    | IntegerQuestion
    | BooleanQuestion,
    Field(discriminator="kind"),
]
FormValue = str | int | float | bool | list[str] | None
"""One answer: ``str``, ``list[str]``, ``float``, ``int``, ``bool``; ``None`` if optional and empty."""


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
        description='On accept: question id -> answer (left-empty optional: "" for text, None otherwise)',
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

    Most questions need only ``id``, ``label`` and maybe ``default``::

        NeedInputForm(heading="Release", questions=[
            TextQuestion(id="branch", label="Which branch should I use?", default="main"),
            BooleanQuestion(id="dry_run", label="Dry run first?", default=True),
        ])

    Add constraints only when the task needs them, and say the rule in words
    in ``help`` (a regex means nothing to a person)::

        TextQuestion(id="slug", label="Project slug", min_length=3, max_length=24,
                     pattern=r"^[a-z0-9-]+$",
                     help="3-24 characters: lowercase letters, digits or hyphens.")

    Answers are checked against each question's constraints. When the client
    can't show the form, or the answers fail the checks, you get a plain
    ``str`` instead (what the person entered): read it as unchecked text.
    """

    model_config = {"extra": "forbid", "strict": True}
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
            hints = _hints(question)
            if hints:
                line += ": " + "; ".join(hints)
            lines.append(line)
        return "\n\n".join(part for part in (self.heading, self.reason, "\n".join(lines)) if part)

    @hidden
    def validate_response(self, response: FormResponse) -> FormResponse:
        """Check accepted answers; raise one ``ValueError`` naming every failing field.

        Types are strict (no conversion). Every question id is in the result:
        an empty optional text answer is "", any other empty optional answer None.
        """
        response = FormResponse.model_validate(response)
        if response.action != "accept":
            return response
        content = response.content
        assert content is not None
        errors = []
        unknown = content.keys() - {question.id for question in self.questions}
        if unknown:
            errors.append(f"unknown answer ids {sorted(unknown)}")
        answers: dict[str, FormValue] = {}
        for question in self.questions:
            value = content.get(question.id)
            if _empty(value):
                if question.required:
                    errors.append(f"{question.id}: an answer is required")
                answers[question.id] = "" if question.kind == "text" else None
                continue
            problem = _problem(question, value)
            if problem:
                errors.append(f"{question.id}: {problem}")
            else:
                answers[question.id] = float(value) if question.kind == "number" else value
        if errors:
            raise ValueError("; ".join(errors))
        return FormResponse(action="accept", content=answers)


def _empty(value: Any) -> bool:
    """No answer: missing, blank text, or nothing picked."""
    return value is None or value == [] or (isinstance(value, str) and not value.strip())


_FORMAT_TEXT = {
    "email": "an email address",
    "uri": "a URI such as https://example.com",
    "date": "a date (YYYY-MM-DD)",
    "date-time": "a date and time (YYYY-MM-DDThh:mm)",
}


def _problem(question: Any, value: Any) -> str | None:
    """Why ``value`` does not answer ``question`` (a person-readable phrase), or None."""
    kind = question.kind
    if kind == "boolean":
        return None if isinstance(value, bool) else "answer yes or no"
    if kind in ("integer", "number"):
        whole = kind == "integer"
        if isinstance(value, bool) or not isinstance(value, int if whole else (int, float)):
            return f"enter {_range_text(question)}"
        if not math.isfinite(value) or not _in_range(value, question.minimum, question.maximum):
            return f"enter {_range_text(question)}"
        return None
    if kind == "pick_many":
        values = {choice.value for choice in question.choices}
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return "choose from the list"
        if len(set(value)) != len(value):
            return "choose each option at most once"
        if not set(value) <= values:
            return "choose only listed options"
        if not _in_range(len(value), question.min_items, question.max_items):
            return f"choose {_count_text(question)}"
        return None
    if not isinstance(value, str):
        return "enter text"
    if kind == "pick_one":
        if not question.allow_other and value not in {c.value for c in question.choices}:
            return "choose a listed option"
        return None
    if not _in_range(len(value), question.min_length, question.max_length):
        return f"enter {_length_text(question)}"
    if question.format and not _has_format(value, question.format):
        return f"enter {_FORMAT_TEXT[question.format]}"
    if question.pattern and re.fullmatch(question.pattern, value) is None:
        return f"enter text matching {question.pattern}"
    return None


def _in_range(value: float, low: float | None, high: float | None) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


def _has_format(value: str, kind: str) -> bool:
    if kind == "email":
        return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is not None
    if kind == "uri":
        return re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*:[^\s]+", value) is not None
    try:
        if kind == "date":
            datetime.date.fromisoformat(value)
            return re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None
        datetime.datetime.fromisoformat(value)
        return re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*", value) is not None
    except ValueError:
        return False


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _range_text(question: Any) -> str:
    noun = "a whole number" if question.kind == "integer" else "a number"
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
def form_hints(question: Any) -> list[str]:
    """Readable constraint hints for one question (shared by text renderers)."""
    return _hints(question)


def _hints(question: Any) -> list[str]:
    kind = question.kind
    hints: list[str] = []
    if kind == "pick_one":
        hints.append(", ".join(choice.title for choice in question.choices))
    elif kind == "pick_many":
        hints.append("choose any of: " + ", ".join(choice.title for choice in question.choices))
        if _count_text(question):
            hints.append(_count_text(question))
    elif kind in ("integer", "number"):
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
    default = getattr(question, "default", None)
    if default is not None:
        hints.append(f"default: {_default_text(question, default)}")
    return hints


def _default_text(question: Any, default: Any) -> str:
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
    if isinstance(data.get("questions"), list):
        data["questions"] = [migrate_form_question(q) for q in data["questions"]]
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


def migrate_form_question(value: Any) -> Any:
    """Map a stored legacy question record onto the current descriptors.

    ``pick_one_or_text`` became ``pick_one`` with ``allow_other=True``; legacy
    ``pick_one`` records (no ``allow_other`` key) were strict pickers.
    Only for persisted data: authored dicts get the ``allow_other`` default.
    """
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "pick_one_or_text":
        return {**value, "kind": "pick_one", "allow_other": True}
    if kind == "pick_one" and "allow_other" not in value:
        return {**value, "allow_other": False}
    return value


InputRequest = Annotated[NeedInput | NeedInputForm, Field(discriminator="kind")]
"""Structured request outputs require kind=question/form; constructors supply it."""


class Waiting(BaseModel):
    """Turn result: waiting on a background job or queue, not on a person.

    ``on`` lists what is being waited on by name: a queue channel
    (``"delegates"``, ``"jobs"``), a spawned job's label, a subagent's name.
    ``explanation`` says why, for the host. ``message`` is an optional line
    the host shows the user. The host keeps the request open and runs the
    next turn when one of them delivers.
    """

    explanation: str = Field(description="Why the turn is waiting")
    message: str | None = Field(default=None, description="Line to show the user while waiting")
    on: list[str] = Field(
        min_length=1, description="Names of the channels, jobs or subagents being waited on"
    )

    _check_explanation = field_validator("explanation")(_non_blank)
    _check_message = field_validator("message")(_blank_to_none)

    @field_validator("on")
    @classmethod
    def _names_not_blank(cls, value: list[str]) -> list[str]:
        return [_non_blank(name) for name in value]


class TaskResult(BaseModel):
    """Structured result of a delegated objective or a benchmark task.

    A worker turn (``handle_work``) ends with a ``Done`` whose ``result`` is
    one. The field descriptions are what the model reads when it builds one.
    """

    solution_description: str = Field(
        description="What you did and why it solves the problem. Describe root cause and fix."
    )
    evidence: str = Field(
        description=(
            "Concrete evidence that the task is done: what tests passed, "
            "what output was produced, what behavior changed. Not a guess -- "
            "cite the actual results you observed."
        )
    )
    how_to_verify: str = Field(
        title="How to Verify",
        description=(
            "How a verifier can confirm correctness: concrete checks or steps and their "
            "expected results. Include commands when appropriate; a shell command is not required."
        ),
    )
    report: str = Field(
        default="",
        description="Concise human-readable report that a parent or runner shows inline.",
    )
    value: str = Field(
        default="",
        description=(
            "The exact output the task asks for, if any, with nothing added: no label, "
            "explanation or formatting. It is delivered verbatim, whitespace included. "
            "Leave empty when the task only asks for actions."
        ),
    )


ONE_SHOT_QUESTION: Annotated[str, hidden] = (
    "Your parent can't answer a one-shot worker: decide, state your assumptions in "
    "the TaskResult report, and return Done."
)


@hidden
def require_task_result(agent: Any, result: Any, call: Any) -> None:
    """Reject a worker ``Done`` whose ``result`` is not a ``TaskResult``, and a one-shot question.

    The postcondition of every worker turn (``handle_work``). A worker's
    parent reads ``done.result``; ``self.message()`` text reaches it only as
    progress notes. A dict that validates as a ``TaskResult`` is accepted
    (the parent receives it as one). A one-shot worker may not ask: its
    session port says so (``agent.session.may_ask``).
    """
    if isinstance(result, (NeedInput, NeedInputForm)):
        port = getattr(agent, "session", None)
        if port is not None and not getattr(port, "may_ask", True):
            raise InvariantError(ONE_SHOT_QUESTION)
        return
    if not isinstance(result, Done) or isinstance(result.result, TaskResult):
        return
    if isinstance(result.result, dict):
        try:
            TaskResult.model_validate(result.result)
            return
        except ValidationError:
            pass
    raise InvariantError(
        "A worker turn must end with Done(explanation=..., result=TaskResult(...)): "
        "set solution_description, evidence and how_to_verify, and put the full report "
        "in report. Your parent reads only the result; self.message() is no substitute."
    )


class AgentMessage(Metadata):
    """A Markdown message sent by the agent to the user via ``self.message()``.

    Interactive hosts share this event so durable sessions can be replayed
    without knowing which frontend produced the response.
    """

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""
    is_result_text: bool = False
    """The text of the turn's ``Done``/``Waiting`` message, not a progress message.

    A session hands it to a parent with the turn's result, so a relay of
    progress messages skips it.
    """


class SummarizationConfig(BaseModel):
    """Configuration for history summarization.

    ``max_tokens=None`` uses ``threshold_fraction`` of the usable input window
    (model window minus the effective reply reserve), resolved for each completed
    request. Set an explicit integer to pin a threshold across model switches.
    """

    policy: Literal["token_budget", "none"] = "token_budget"
    max_tokens: int | None = None
    threshold_fraction: float = Field(default=0.75, gt=0, lt=1)
    preserve_recent: int = 10
    target_chars: int = 4000


# Default model used when an agent class is defined without an explicit LLM
# (overridden at instantiation).
DEFAULT_MODEL = "claude-opus-4-8"

with hidden:
    try:
        from nooa.unifiedllm import get_llm_client

        _DEFAULT_LLM = get_llm_client(DEFAULT_MODEL)
    except Exception:
        from nooa.unifiedllm import FakeLLMClient

        _DEFAULT_LLM = FakeLLMClient()


# Summarizer trigger as a fraction of the LLM's context window. This is the
# ONLY budget managed here — event-pile truncation is enforced at the
# runtime level (see ActorRuntime._build_messages) and adapts to whichever
# LLM is actually resolved for each call (including per-call overrides).
_SUMMARIZER_BUDGET_PCT = 0.75


def _summarizer_budget(
    llm: "UnifiedLLM",
    fallback_reserve: int = 0,
    *,
    threshold_fraction: float = _SUMMARIZER_BUDGET_PCT,
) -> int:
    """Resolve the summarizer trigger from the LLM's usable input window.

    Falls back to 100K when the LLM doesn't expose ``context_window`` so
    we still have a functional threshold.
    """
    from nooa.agents.summarization import context_budget

    return context_budget(llm, threshold_fraction, fallback_reserve=fallback_reserve)


def apply_model_limits(agent: Agent) -> None:
    """Sync automatic summarizers against the selected model's usable window.

    Call after a model switch so the summarizer threshold moves with the
    new context window. Runtime-level event truncation picks up the new
    window automatically on the next ``_build_messages`` call.
    """
    for summarizer in getattr(agent, "_summarizers", []):
        if not getattr(summarizer, "_automatic_context_budget", False):
            continue
        summarizer_max = _summarizer_budget(
            agent.llm,
            agent._truncation.response_reserve_tokens,
            threshold_fraction=summarizer._automatic_context_budget_percent,
        )
        current = summarizer.config
        summarizer.config = current.model_copy(update={"max_tokens": summarizer_max})


def install_summarizer(config: SummarizationConfig, agent: Agent) -> None:
    """Install a summarizer on the agent based on configuration.

    Args:
        config: Summarization configuration. ``config.max_tokens=None`` follows
            ``threshold_fraction`` (75% by default) of each request's usable input window.
        agent: Agent to install summarizer on (inherits LLM, attaches to history)
    """
    if config.policy == "none":
        return

    from nooa.config.summarizer_config import TokenBudgetConfig

    summarizer_max = (
        config.max_tokens
        if config.max_tokens is not None
        else _summarizer_budget(
            agent.llm,
            agent._truncation.response_reserve_tokens,
            threshold_fraction=config.threshold_fraction,
        )
    )

    summarizer = TokenBudgetSummarizer.install(
        agent,
        config=TokenBudgetConfig(
            max_tokens=summarizer_max,
            preserve_recent=config.preserve_recent,
            target_chars=config.target_chars,
        ),
    )
    summarizer._automatic_context_budget = config.max_tokens is None
    summarizer._automatic_context_budget_percent = config.threshold_fraction


class AgentVars:
    """Attribute-access proxy for an agent's persistent ``vars`` dict.

    Mirrors ``TodoVars``: write ``self.v.spec = "..."`` instead of
    ``self.vars["spec"] = "..."``. Reads and writes go straight
    through to ``self.vars`` so snapshot serialization is unaffected.

    Use for variables that need to survive across turns and across
    sessions but aren't tied to a specific todo. (For per-todo state,
    use ``self.todo.<id>.v`` — same shape, narrower scope.)

    Values are snapshot-backed: assigning something that can't be
    snapshot-serialized (a live client, socket, callable, ...) logs a
    warning and is **not stored** — it won't survive ``/exit`` + resume.
    Store serializable data (dict/str/number/Pydantic model) instead.
    """

    def __init__(self, agent: Any):
        object.__setattr__(self, "_agent", agent)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._agent.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r} on agent") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self._agent.vars[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(
            f"self.v does not support item assignment; write self.v.{key} = ... instead."
        )

    def __delattr__(self, key: str) -> None:
        try:
            del self._agent.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r} on agent") from None

    def __contains__(self, key: str) -> bool:
        return key in self._agent.vars

    def __repr__(self) -> str:
        return repr(self._agent.vars)


@hidden
class InteractiveAgent(Agent, llm=_DEFAULT_LLM):
    """Base class for agents driven by an outer dispatcher (TUI, harness, ...).

    Subclass this and implement ``handle()`` to build a custom interactive
    agent. ``message()`` is provided for free.

    **Input queues.** Every ``InteractiveAgent`` has a ``self.user_messages``
    queue (``InputQueue``) that the host feeds when the human types.
    Subclasses may declare additional queues as instance attributes for
    other producers (long-running job output, monitor streams, etc.).

    Each queue is two objects: an ``InputQueue`` (full producer +
    dispatcher API, hidden from the LLM under ``_<name>_in``) and an
    ``OutputQueue`` reader facade exposed under the public name. The
    LLM can ``await self.user_messages.get()`` to dequeue mid-turn;
    everything else (put, snapshot, qsize, etc.) is dispatcher-only.

    ``handle()`` runs *per turn* — the outer dispatcher calls it with
    the next notification (a ``dict[str, list]``). Use ``self.v`` for
    state that should survive across turns.
    """

    _render_message: Annotated[Callable[[str], None] | None, hidden, nosnapshot]
    # QueueManager owns the channel registry. Hidden from the LLM by
    # default — the LLM should access individual channels (e.g.
    # ``self.user_messages``) directly, not through a string-keyed
    # registry lookup.
    queue_manager: Annotated[QueueManager, hidden, nosnapshot]
    # Drives handle() from the channels once a host calls turns.start().
    # Hidden: the model must not cancel or await its own turns.
    turns: Annotated[TurnLoop, hidden, nosnapshot]
    # Producer-side Channel (full put / pop_last / snapshot / etc.)
    # — hidden, since the LLM has no business calling those.
    _user_messages_in: Annotated[Channel, hidden, nosnapshot]
    # Read-only facade (just .get() / .status() / .name) is what the
    # LLM sees as ``self.user_messages``.
    user_messages: Annotated[ChannelReader, nosnapshot]
    # Persistent variables for the LLM — survives across turns AND
    # across sessions (snapshot-backed). Accessed via the ``self.v``
    # proxy for dot-attribute reads/writes (``self.v.spec = "..."``).
    vars: SnapshotVars

    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm or _DEFAULT_LLM, **kwargs)
        self._render_message = None
        self.vars = SnapshotVars()
        self.queue_manager = QueueManager(event_manager=self.event_manager)
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.user_messages = self._user_messages_in.reader
        self.event_manager.on_close(self.queue_manager.shutdown)
        self.turns = TurnLoop(self)
        self.producers = ProducersSkill()
        # Surface pending-queue counts (and a short preview of each item)
        # to the LLM every turn — the agent reads queue depth straight
        # from the ``queues`` context block. Composed via
        # ``QueueManager.status()`` so adding new channels Just Works.
        from nooa import Context

        self.context["queues"] = Context(expr="self.queue_manager.status()")

    @hidden
    @no_trace
    async def aclose(self) -> None:
        """Stop owned turns/jobs before the caller releases shared resources.

        Self-close from an executing turn is rejected; its external owner must
        close after settlement. This avoids joining one's own task indirectly.
        """
        if self.turns.in_turn:
            raise RuntimeError("an executing turn cannot close its own agent")
        # Register for this close epoch; the manager shields and joins cleanup.
        # Last registered runs first, before other components release resources.
        self.event_manager.on_close(self._close_runtime)
        await super().aclose()

    @hidden
    @no_trace
    async def _close_runtime(self) -> None:
        try:
            await self.turns.stop()
        finally:
            await self.queue_manager.shutdown()

    @property
    def v(self) -> AgentVars:
        """Attribute-access proxy for ``self.vars`` — the agent's
        persistent variable dict.

        Use for state that should survive across turns and sessions
        but isn't tied to a specific todo. Snapshot-backed via
        ``self.vars``.

        Usage::

            self.v.spec = "implement queue mode"
            self.v.cursor = 0
            print(self.v.spec)
            del self.v.cursor

        Compare:
        - REPL locals → cleared between turns.
        - ``self.v.k = v`` → snapshot-backed, survives turns + sessions.
        - ``self.todo.<t>.v.k = v`` → same as ``self.v`` but scoped to
          one todo.
        """
        return AgentVars(self)

    def message(self, text: str, *, echo: bool = False) -> None:
        """Send a Markdown message to the user.

        In a child session the text goes to the parent instead, as a
        ``ChildMessage`` on its ``delegates`` channel: a progress note, not
        the child's result.

        Each call renders as an independent block — so every call must be a
        complete, self-contained Markdown document.  In particular, never split
        a table across calls: the header row and all data rows must be in the
        same ``message()`` call, otherwise the table will not render correctly.

        Args:
            text: Markdown content to send.
            echo: If True, also ``print()`` the text so the LLM can see it
                in the execution output. By default the LLM does NOT see the
                content of ``message()`` calls — only the user does. Use
                ``echo=True`` when you need to reference the sent content in
                subsequent cells.
        """
        self._send_message(text)
        if echo:
            print(text)

    @hidden
    def _send_message(self, text: str, *, is_result_text: bool = False) -> None:
        """Record and render one message; ``is_result_text`` marks a turn result's message."""
        event = AgentMessage(content=str(text), is_result_text=is_result_text)
        tag = self.event_manager.add(event)
        if self._render_message is not None:
            try:
                self._render_message(text, event_id=str(event.id), tags={str(tag)})
            except TypeError:
                self._render_message(text)

    @hidden
    @strategy(CodeActStrategy())
    async def handle(
        self,
        notification: dict[str, list],
    ) -> Done | InputRequest | Waiting:
        """Handle one interactive turn.

        Called once per inbound notification (or batch). Unpack
        ``notification``, do the work, then end the turn with one typed
        result. Use ``self.v.<name> = value`` for state that must survive
        across turns (snapshot-backed via ``self.vars``).

        ## Turn anatomy

        Notification is always ``dict[str, list]`` — channel name →
        list of items that arrived since last turn::

            msgs = notification.get("user_messages", [])
            lines = notification.get("ci", [])

        ## Work ethic

        Do ALL the work before returning. Use as many ``execute_python``
        calls as needed — explore, implement, test, iterate. A turn that
        returns after one or two cells when the task clearly needs more
        is a bug. End the turn only when the request is complete, when you
        need an answer from the person, or when you are waiting on a
        background job.

        ## Returning

        A turn has exactly one terminal result: the value you return.
        Everything else is intermediate. End the turn with one
        ``return_result(...)`` of one of these:

        - ``Done(message=..., explanation=...)`` — the request is complete.
          ``message`` is the reply the host shows the user; ``explanation``
          is a short status line. Add ``evidence=[...]`` for checks you ran::

              return_result(Done(message="Added the flag and its test.", explanation="added flag"))

          Use ``self.message()`` only for text to show before the turn ends.

        - ``NeedInput(question=...)`` — you cannot continue without an
          answer. The host shows ``question``, so do not also send it with
          ``message()``. The next message answers it as plain text; put any
          suggested answers in the question. ``reason`` optionally says why::

              return_result(NeedInput(question="Which branch should I push to: main or dev?"))

        - ``NeedInputForm(heading=..., questions=[...])`` — a form of
          ``TextQuestion``, ``PickOneQuestion``, ``PickManyQuestion``,
          ``NumberQuestion``, ``IntegerQuestion`` and ``BooleanQuestion`` fields
          (``doc()`` shows their constraints). The answer is a ``FormResponse``:
          ``accept`` with ``content`` mapping each question id to its checked
          answer, or ``decline``/``cancel``; plain text when no form could be used::

              return_result(NeedInputForm(heading="Release", questions=[
                  PickOneQuestion(id="branch", label="Branch?", choices=[
                      FormChoice(value="main", title="Main"), FormChoice(value="dev", title="Dev")]),
                  PickManyQuestion(id="regions", label="Regions?", choices=[
                      FormChoice(value="us", title="US"), FormChoice(value="eu", title="EU")]),
                  IntegerQuestion(id="replicas", label="Replicas?", minimum=1, maximum=10),
                  BooleanQuestion(id="dry_run", label="Dry run?", default=True),
                  TextQuestion(id="contact", label="Contact?", format="email", required=False),
              ]))

        - ``Waiting(message=..., explanation=..., on=[...])`` — waiting on a
          background job or queue, not on a person. ``message`` is shown to
          the user; ``on`` names what you wait for: a channel, a job label,
          a subagent::

              return_result(Waiting(message="Tests are running; I will report when they finish.", explanation="tests running", on=["jobs:ci-42"]))

        ## Available queues

        The dispatcher delivers the next item via ``notification``.
        You can also dequeue extra items mid-turn (without ending it)
        by awaiting the queue directly::

            extra = await self.user_messages.get()  # blocks until next message

        Recognized ``queue_name`` values:

        - ``"user_messages"`` — text from the human; ``item`` is a str.

        Hosts declare the rest. ``CodingAgent`` adds ``"slash_commands"``
        (``SlashCommandResult``) and ``"system_messages"`` (host-owned
        prompts such as keep-going continuations); a harness might add
        ``"job_outputs"``. The
        ``<queue_status>`` context block lists the pending count per
        queue each turn. After any result, the dispatcher races every
        declared queue/event and re-enters with the first arrival.
        """
        ...

    @hidden
    @strategy(CodeActStrategy())
    async def handle_batch(
        self,
        notification: dict[str, list],
    ) -> Done | Waiting:
        """Handle one unattended turn.

        No person is watching this turn and nobody can answer a question,
        so ``NeedInput`` is not allowed here and is rejected as a type
        error. Unpack ``notification`` (channel name → list of items) and
        do all the work.

        A turn has exactly one terminal result: the value you return.
        Everything else is intermediate. End with one ``return_result(...)``:

        - ``Done(explanation=...)`` — the work is finished, or cannot go
          further. If something blocks you, say what in ``explanation``.
          Set ``result`` when the task asks for a structured result.
        - ``Waiting(message=..., explanation=..., on=[...])`` — waiting on a
          background job or queue you started. ``on`` names it.
        """
        ...

    @hidden
    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[require_task_result])))
    async def handle_work(
        self,
        notification: dict[str, list],
    ) -> Done | InputRequest | Waiting:
        """Work on a task for your parent agent and report back to it.

        You are a child session. Your parent is an agent, not a person, and it
        reads only your result. The task is in ``notification["user_messages"]``;
        its first line says whether you can ask your parent questions (later
        messages there are follow-ups). Your task is the brief; your
        ``TaskResult`` is your report; a Todo in your list is your own working
        copy. Do all the work, verify it, then end with exactly one
        ``return_result(...)``:

        - ``Done`` with a ``TaskResult``, when the task is finished or blocked
          (then say what blocked you and what you tried)::

              return_result(Done(
                  explanation="fixed the parser bug",
                  result=TaskResult(
                      solution_description="What was done, including modified paths",
                      evidence="Commands run and what they showed",
                      how_to_verify="How someone else can check it",
                      report="The full report for your parent",
                  ),
              ))

          ``TaskResult`` comes from ``nooa.interactive``, like ``Done``; a dict
          with the same fields is accepted.
        - ``NeedInput(question=...)`` (or a ``NeedInputForm``) only if you can
          ask (a kept child) and cannot go on without your parent's answer; it
          arrives as your next message. A one-shot worker's question is rejected.
        - ``Waiting(explanation=..., on=[...])`` only while a job you started
          is still running.

        ``self.message()`` sends your parent a progress note (a
        ``ChildMessage``); it does not replace the result.
        """
        ...
