# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small transport-independent descriptors for ACP's restricted form subset."""

import asyncio
import json
import math
import subprocess
import sys
import threading
from collections import OrderedDict
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from nooa.agentdoc import hidden, spec

FormValue = str | int | float | bool | list[str]


# Portable payload/concurrency limits; Unix adds a virtual-memory worker cap.
_PATTERN_LIMIT = 4096
_VALUE_LIMIT = 65536
_PATTERN_SLOTS = threading.BoundedSemaphore(4)
_PATTERN_CACHE: OrderedDict[tuple[str, str], bool] = OrderedDict()
_PATTERN_LOCK = threading.Lock()
_PATTERN_WORKER = """
import json, re, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
except (ImportError, AttributeError, OSError, ValueError):
    pass
p, v = json.load(sys.stdin)
print(int(re.search(p, v) is not None))
"""


def _pattern_input(pattern: str, value: str) -> bytes:
    if len(pattern) > _PATTERN_LIMIT or len(value) > _VALUE_LIMIT:
        raise ValueError("pattern/input exceeds size budget (4096/65536 characters)")
    return json.dumps([pattern, value]).encode()


def _cached_pattern(pattern: str, value: str) -> bool | None:
    with _PATTERN_LOCK:
        result = _PATTERN_CACHE.get((pattern, value))
        if result is not None:
            _PATTERN_CACHE.move_to_end((pattern, value))
        return result


def _pattern_result(pattern: str, value: str, code: int, output: bytes) -> bool:
    if code:
        raise ValueError("invalid pattern or worker resource budget exceeded")
    result = output.strip() == b"1"
    with _PATTERN_LOCK:
        _PATTERN_CACHE[pattern, value] = result
        _PATTERN_CACHE.move_to_end((pattern, value))
        if len(_PATTERN_CACHE) > 128:
            _PATTERN_CACHE.popitem(last=False)
    return result


def _pattern_matches(pattern: str, value: str) -> bool:
    data = _pattern_input(pattern, value)
    cached = _cached_pattern(pattern, value)
    if cached is not None:
        return cached
    if not _PATTERN_SLOTS.acquire(blocking=False):
        raise ValueError("pattern worker concurrency budget exhausted; retry later")
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _PATTERN_WORKER],
            input=data,
            capture_output=True,
            timeout=1,
        )
        return _pattern_result(pattern, value, result.returncode, result.stdout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("pattern evaluation exceeded its time budget") from exc
    except OSError as exc:
        raise ValueError("pattern worker unavailable") from exc
    finally:
        _PATTERN_SLOTS.release()


async def _pattern_matches_async(pattern: str, value: str) -> bool:
    data = _pattern_input(pattern, value)
    cached = _cached_pattern(pattern, value)
    if cached is not None:
        return cached
    # Fail fast, not an unlimited task queue or an event-loop-bound semaphore.
    if not _PATTERN_SLOTS.acquire(blocking=False):
        raise ValueError("pattern worker concurrency budget exhausted; retry later")
    process = None
    starting = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _PATTERN_WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    )
    try:
        process = await asyncio.shield(starting)
        try:
            output, _ = await asyncio.wait_for(process.communicate(data), timeout=1)
        except TimeoutError as exc:
            raise ValueError("pattern evaluation exceeded its time budget") from exc
        return _pattern_result(pattern, value, process.returncode, output)
    except (OSError, NotImplementedError) as exc:
        raise ValueError("pattern worker unavailable") from exc
    finally:
        # Shielded cleanup owns the slot even if the caller is cancelled again.
        await asyncio.shield(asyncio.create_task(_reap_pattern_worker(starting, process)))


async def _reap_pattern_worker(starting, process) -> None:
    try:
        if process is None:
            try:
                process = await starting
            except (OSError, NotImplementedError):
                return
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
    finally:
        _PATTERN_SLOTS.release()


class FormChoice(BaseModel):
    """A listed value; omit titles for an untitled enum."""

    model_config = {"extra": "forbid", "strict": True}
    value: str = Field(min_length=1)
    title: str | None = Field(
        default=None,
        min_length=1,
        description="All choices need titles if any title/description is supplied",
    )
    description: str | None = None


class Question(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    help: str | None = None
    required: bool = Field(
        default=True, description="Answer key must be present; does not imply nonblank"
    )

    @model_validator(mode="after")
    def _valid_descriptor(self):
        if not self.id.strip() or not self.label.strip():
            raise ValueError("id and label must not be blank")
        for lower, upper in (
            ("min_length", "max_length"),
            ("minimum", "maximum"),
            ("min_items", "max_items"),
        ):
            low, high = getattr(self, lower, None), getattr(self, upper, None)
            if low is not None and high is not None and low > high:
                raise ValueError(f"{lower} must not exceed {upper}")
        if getattr(self, "pattern", None) is not None and self.default is None:
            _pattern_matches(self.pattern, "")
        if getattr(self, "default", None) is not None:
            self.validate_answer(self.default)
        return self

    @hidden
    def validate_answer(self, value: FormValue) -> None:
        """Check native values, without coercion or default insertion."""
        self._validate_answer(value)

    def _validate_answer(self, value: FormValue, *, check_pattern: bool = True) -> None:
        if self.kind in ("text", "choice"):
            if type(value) is not str:
                raise ValueError(f"{self.id}: expected a string")
            if self.kind == "text":
                if self.min_length is not None and len(value) < self.min_length:
                    raise ValueError(f"{self.id}: below minLength")
                if self.max_length is not None and len(value) > self.max_length:
                    raise ValueError(f"{self.id}: above maxLength")
                if (
                    check_pattern
                    and self.pattern is not None
                    and not _pattern_matches(self.pattern, value)
                ):
                    raise ValueError(f"{self.id}: does not match pattern")
                # Formats, including unknown formats, are annotations.
        elif self.kind in ("integer", "number"):
            allowed = (int,) if self.kind == "integer" else (int, float)
            if type(value) not in allowed or (type(value) is float and not math.isfinite(value)):
                raise ValueError(f"{self.id}: expected a finite {self.kind}")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{self.id}: below minimum")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"{self.id}: above maximum")
        elif self.kind == "boolean":
            if type(value) is not bool:
                raise ValueError(f"{self.id}: expected a boolean")
        elif self.kind == "multi_choice":
            if type(value) is not list or any(type(v) is not str for v in value):
                raise ValueError(f"{self.id}: expected a string list")
            listed = {c.value for c in self.choices}
            if sum(v not in listed for v in value) > 1:
                raise ValueError(f"{self.id}: at most one written Other is allowed")
            if self.min_items is not None and len(value) < self.min_items:
                raise ValueError(f"{self.id}: below minItems")
            if self.max_items is not None and len(value) > self.max_items:
                raise ValueError(f"{self.id}: above maxItems")


class TextQuestion(Question):
    """Native text; length bounds count characters. Formats are annotations."""

    kind: Literal["text"] = "text"
    default: str | None = Field(
        default=None, description="UI suggestion only; never fills a missing answer"
    )
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    pattern: str | None = Field(
        default=None,
        description="Python regex search; 1s/check, pattern/input limits 4096/65536 characters",
    )
    format: str | None = None


class IntegerQuestion(Question):
    """Native integer (not bool); inclusive bounds."""

    kind: Literal["integer"] = "integer"
    minimum: int | None = None
    maximum: int | None = None
    default: int | None = Field(
        default=None, description="UI suggestion only; never fills a missing answer"
    )


class NumberQuestion(Question):
    """Native finite int/float (not bool); inclusive bounds."""

    kind: Literal["number"] = "number"
    minimum: int | float | None = None
    maximum: int | float | None = None
    default: int | float | None = Field(
        default=None, description="UI suggestion only; never fills a missing answer"
    )


class BooleanQuestion(Question):
    """Native bool, without parsing or numeric coercion."""

    kind: Literal["boolean"] = "boolean"
    default: bool | None = Field(
        default=None, description="UI suggestion only; never fills a missing answer"
    )


class ChoiceQuestion(Question):
    """Choose a listed value or write another value."""

    kind: Literal["choice"] = "choice"
    choices: list[FormChoice] = Field(min_length=1)
    default: str | None = Field(
        default=None, description="UI suggestion only; never fills a missing answer"
    )

    @model_validator(mode="after")
    def _choices(self):
        if len({c.value for c in self.choices}) != len(self.choices):
            raise ValueError("choice values must be unique")
        if any(c.title is not None or c.description is not None for c in self.choices):
            if any(c.title is None for c in self.choices):
                raise ValueError("all choices need titles when any title/description is supplied")
        return self


class MultiChoiceQuestion(ChoiceQuestion):
    """One native list: listed values plus an optional written Other value."""

    kind: Literal["multi_choice"] = "multi_choice"
    default: list[str] | None = Field(
        default=None, description="UI suggestion only; never fills a missing answer"
    )
    min_items: int | None = Field(
        default=None,
        ge=0,
        description="Inclusive list minimum; standard ACP also bounds the listed picker when supplied",
    )
    max_items: int | None = Field(default=None, ge=0, description="Inclusive final list maximum")


FormQuestion = Annotated[
    TextQuestion
    | IntegerQuestion
    | NumberQuestion
    | BooleanQuestion
    | ChoiceQuestion
    | MultiChoiceQuestion,
    Field(discriminator="kind"),
]


class FormResponse(BaseModel):
    """One native answer per authored id. Optional omissions stay omitted."""

    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    action: Literal["accept", "decline", "cancel"]
    content: dict[str, FormValue] | None = Field(
        default=None, description="Required dict for accept; omit for decline/cancel"
    )

    @model_validator(mode="after")
    def _content_for_action(self):
        if self.action == "accept" and self.content is None:
            raise ValueError("accept requires content")
        if self.action != "accept" and self.content is not None:
            raise ValueError("decline/cancel must not carry content")
        return self


class NeedInputForm(BaseModel):
    """Ordered native questions. Ask doc(TextQuestion), etc. for control details.

    Choices allow Other. Never request secrets, even in plain-text fallback.
    """

    model_config = {"extra": "forbid", "strict": True}
    kind: Literal["form"] = "form"
    heading: str = Field(min_length=1)
    reason: str | None = None
    questions: list[FormQuestion] = Field(
        min_length=1,
        max_length=64,
        description="Authored order; unique ids; ask doc(Type) for each control",
    )

    @model_validator(mode="after")
    def _unique_ids(self):
        if not self.heading.strip():
            raise ValueError("heading must not be blank")
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("question ids must be unique")
        return self

    @spec(expand=False)
    def validate_response(self, response: FormResponse) -> FormResponse:
        """Check ids, required presence and native constraints; no coercion/default insertion."""
        return self._validate_response(response)

    def _validate_response(
        self, response: FormResponse, *, check_pattern: bool = True
    ) -> FormResponse:
        response = FormResponse.model_validate(response)
        if response.action != "accept":
            return response
        content = response.content
        assert content is not None
        unknown = content.keys() - {q.id for q in self.questions}
        if unknown:
            raise ValueError(f"Unknown answer ids: {sorted(unknown)}")
        for q in self.questions:
            if q.id not in content:
                if q.required:
                    raise ValueError(f"{q.id}: an answer is required")
            else:
                q._validate_answer(content[q.id], check_pattern=check_pattern)
        return response


# Keep the primary contract compact, without suppressing individual control docs.
spec(NeedInputForm, "questions", expand=False)


async def _validate_form_response(need: NeedInputForm, response: FormResponse) -> FormResponse:
    response = need._validate_response(response, check_pattern=False)
    if response.action == "accept":
        for q in need.questions:
            if q.kind == "text" and q.pattern is not None and q.id in response.content:
                if not await _pattern_matches_async(q.pattern, response.content[q.id]):
                    raise ValueError(f"{q.id}: does not match pattern")
    return response


def question_lines(questions: list[dict]) -> list[str]:
    """Shared display/replay; tolerate old presentation records, not answers."""
    lines = []
    for q in questions:
        if not isinstance(q, dict) or not isinstance(q.get("label"), str):
            continue
        line = f"{len(lines) + 1}. {q['label']}"
        if isinstance(q.get("help"), str) and q["help"]:
            line += f" — {q['help']}"
        if isinstance(q.get("choices"), list) and q["choices"]:
            labels = [
                (
                    f"{c['title']} ({c['value']})"
                    if isinstance(c.get("title"), str) and c["title"]
                    else c["value"]
                )
                + (
                    f" — {c['description']}"
                    if isinstance(c.get("description"), str) and c["description"]
                    else ""
                )
                for c in q["choices"]
                if isinstance(c, dict) and isinstance(c.get("value"), str)
            ]
            if labels:
                line += " Choices: " + ", ".join(labels) + ", or write Other"
            if q.get("kind") == "multi_choice":
                line += " (select multiple)"
        elif q.get("kind") in ("integer", "number", "boolean"):
            line += f" ({q['kind']})"
        if not q.get("required", True):
            line += " (optional)"
        hints = [
            f"{k}={q[k]}"
            for k in (
                "minimum",
                "maximum",
                "min_length",
                "max_length",
                "pattern",
                "format",
                "min_items",
                "max_items",
                "default",
            )
            if q.get(k) is not None
        ]
        if hints:
            line += " [" + ", ".join(hints) + "]"
        lines.append(line)
    return lines
