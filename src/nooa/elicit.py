# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ask the person at the client to fill in a small form.

``ask(message, **fields)`` sends a form to whichever *asker* the host bound
in the current context (``bind_asker``) and returns an :class:`Elicited`.
The host decides how a form is shown: the Atom ACP server shows it as an ACP
elicitation for root sessions only. Without an asker ``ask`` raises
:class:`ElicitationUnavailable` at once, so callers can fall back to text.

Fields are flat primitives (the ACP/MCP form subset): :class:`Text`,
:class:`Number`, :class:`Integer`, :class:`Confirm`, :class:`Choice` and
:class:`MultiChoice`. There is no secret field: never use a form to collect
an API key or password; ask for the name of an environment variable instead.

This is a host-side API; it is deliberately not part of any agent class.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "Asker",
    "Choice",
    "Confirm",
    "ElicitationRequest",
    "ElicitationUnavailable",
    "Elicited",
    "Integer",
    "MultiChoice",
    "Number",
    "Text",
    "ask",
    "bind_asker",
    "current_asker",
]

Action = Literal["accept", "decline", "cancel"]


class ElicitationUnavailable(RuntimeError):
    """No form can be shown here: no asker is bound, or the client has no forms."""


class Elicited(BaseModel):
    """The person's answer: ``action`` and, when accepted, the coerced ``values``."""

    action: Action
    values: dict[str, Any] = Field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.action == "accept"


_TRUE = {"true", "yes", "y", "1", "on"}
_FALSE = {"false", "no", "n", "0", "off"}


@dataclass(frozen=True, kw_only=True)
class _Field:
    """Common field options. ``required`` fields must be answered when accepted."""

    title: str | None = None
    description: str | None = None
    default: Any = None
    required: bool = True

    _type: str = field(default="string", init=False, repr=False)

    def _base(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self._type}
        if self.title is not None:
            schema["title"] = self.title
        if self.description is not None:
            schema["description"] = self.description
        return schema

    def to_schema(self) -> dict[str, Any]:
        """This field as a flat JSON-schema property (the ACP/MCP form subset)."""
        schema = self._base()
        if self.default is not None:
            schema["default"] = self._dump(self.default)
        return schema

    def to_string_schema(self) -> dict[str, Any]:
        """This field as a string property, for clients that accept only strings."""
        schema = self.to_schema()
        schema["type"] = "string"
        if "default" in schema:
            schema["default"] = self._as_text(schema["default"])
        return schema

    def _dump(self, value: Any) -> Any:
        return value

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)

    def coerce(self, value: Any) -> Any:
        """Validate ``value`` (possibly a string from a string-only client) for this field."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Text(_Field):
    """A line of text."""

    pattern: str | None = None
    min_length: int | None = None
    max_length: int | None = None

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.pattern is not None:
            schema["pattern"] = self.pattern
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        return schema

    def coerce(self, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise ValueError("expected text")
        text = str(value)
        if self.min_length is not None and len(text) < self.min_length:
            raise ValueError(f"must be at least {self.min_length} characters")
        if self.max_length is not None and len(text) > self.max_length:
            raise ValueError(f"must be at most {self.max_length} characters")
        if self.pattern is not None and re.search(self.pattern, text) is None:
            raise ValueError(f"must match {self.pattern}")
        return text


@dataclass(frozen=True, kw_only=True)
class Number(_Field):
    """A number, optionally between ``minimum`` and ``maximum`` (inclusive)."""

    minimum: float | None = None
    maximum: float | None = None

    _type: str = field(default="number", init=False, repr=False)

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema

    def to_string_schema(self) -> dict[str, Any]:
        schema = super().to_string_schema()
        schema.pop("minimum", None)
        schema.pop("maximum", None)
        return schema

    def _parse(self, value: Any) -> float | int:
        if isinstance(value, bool):
            raise ValueError("expected a number")
        if isinstance(value, int | float):
            number: float | int = value
        elif isinstance(value, str):
            text = value.strip().replace(",", "").replace("_", "")
            try:
                number = int(text)
            except ValueError:
                try:
                    number = float(text)
                except ValueError:
                    raise ValueError("expected a number") from None
        else:
            raise ValueError("expected a number")
        if isinstance(number, float) and not math.isfinite(number):
            raise ValueError("expected a finite number")
        return number

    def _check_range(self, number: float | int) -> None:
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"must be at least {self.minimum}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"must be at most {self.maximum}")

    def coerce(self, value: Any) -> float:
        number = float(self._parse(value))
        self._check_range(number)
        return number


@dataclass(frozen=True, kw_only=True)
class Integer(Number):
    """A whole number, optionally between ``minimum`` and ``maximum`` (inclusive)."""

    minimum: int | None = None  # type: ignore[assignment]
    maximum: int | None = None  # type: ignore[assignment]

    _type: str = field(default="integer", init=False, repr=False)

    def coerce(self, value: Any) -> int:
        number = self._parse(value)
        if isinstance(number, float):
            if not number.is_integer():
                raise ValueError("expected a whole number")
            number = int(number)
        self._check_range(number)
        return number


@dataclass(frozen=True, kw_only=True)
class Confirm(_Field):
    """Yes or no (a boolean)."""

    _type: str = field(default="boolean", init=False, repr=False)

    def to_string_schema(self) -> dict[str, Any]:
        schema = super().to_string_schema()
        schema["oneOf"] = [{"const": "true", "title": "Yes"}, {"const": "false", "title": "No"}]
        return schema

    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        raise ValueError("expected yes or no")


def _options(options: Sequence[str] | Mapping[str, str]) -> tuple[tuple[str, str | None], ...]:
    if isinstance(options, Mapping):
        pairs = tuple((str(value), str(label)) for value, label in options.items())
    elif isinstance(options, str):
        raise TypeError("options must be a list of strings or a mapping of value to label")
    else:
        pairs = tuple((str(value), None) for value in options)
    if not pairs:
        raise ValueError("a choice needs at least one option")
    if len({value for value, _ in pairs}) != len(pairs):
        raise ValueError("choice options must be unique")
    return pairs


@dataclass(frozen=True, kw_only=True)
class Choice(_Field):
    """One of ``options``: a list of strings, or a mapping of value to label."""

    options: Sequence[str] | Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _options(self.options))

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(value for value, _ in self.options)  # type: ignore[misc]

    def _labelled(self) -> bool:
        return any(label is not None for _, label in self.options)  # type: ignore[misc]

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self._labelled():
            schema["oneOf"] = [
                {"const": value, "title": label if label is not None else value}
                for value, label in self.options  # type: ignore[misc]
            ]
        else:
            schema["enum"] = list(self.values)
        return schema

    def to_string_schema(self) -> dict[str, Any]:
        # String-only clients (Pool) show a picker for ``oneOf`` but a bare text
        # box for ``enum``, so always send titled options to them.
        schema = super().to_string_schema()
        schema.pop("enum", None)
        schema["oneOf"] = [
            {"const": value, "title": label if label is not None else value}
            for value, label in self.options  # type: ignore[misc]
        ]
        return schema

    def coerce(self, value: Any) -> str:
        text = value if isinstance(value, str) else None
        if text not in self.values:
            raise ValueError("must be one of " + ", ".join(self.values))
        return text


@dataclass(frozen=True, kw_only=True)
class MultiChoice(Choice):
    """Any number of ``options`` (a list of strings in the answer)."""

    min_items: int | None = None
    max_items: int | None = None

    _type: str = field(default="array", init=False, repr=False)

    def to_schema(self) -> dict[str, Any]:
        schema = self._base()
        if self._labelled():
            items: dict[str, Any] = {
                "anyOf": [
                    {"const": value, "title": label if label is not None else value}
                    for value, label in self.options  # type: ignore[misc]
                ]
            }
        else:
            items = {"type": "string", "enum": list(self.values)}
        schema["items"] = items
        if self.default is not None:
            schema["default"] = list(self.default)
        if self.min_items is not None:
            schema["minItems"] = self.min_items
        if self.max_items is not None:
            schema["maxItems"] = self.max_items
        return schema

    def to_string_schema(self) -> dict[str, Any]:
        schema = _Field.to_schema(self)
        schema["type"] = "string"
        if self.default is not None:
            schema["default"] = self._as_text(list(self.default))
        hint = "Comma-separated, from: " + ", ".join(self.values)
        schema["description"] = f"{self.description} ({hint})" if self.description else hint
        return schema

    def coerce(self, value: Any) -> list[str]:  # type: ignore[override]
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
            items = list(value)
        else:
            raise ValueError("expected a list of options")
        unknown = [item for item in items if item not in self.values]
        if unknown:
            raise ValueError("unknown options: " + ", ".join(unknown))
        items = list(dict.fromkeys(items))
        if self.min_items is not None and len(items) < self.min_items:
            raise ValueError(f"choose at least {self.min_items}")
        if self.max_items is not None and len(items) > self.max_items:
            raise ValueError(f"choose at most {self.max_items}")
        return items


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ElicitationRequest:
    """A form to show: a message and named fields, in order."""

    message: str
    fields: Mapping[str, _Field]

    def __post_init__(self) -> None:
        for name, spec in self.fields.items():
            if not _NAME.fullmatch(name):
                raise ValueError(f"field name {name!r} is not an identifier")
            if not isinstance(spec, _Field):
                raise TypeError(f"field {name!r} must be Text, Number, Integer, Confirm, ...")

    def _required(self) -> list[str]:
        return [name for name, spec in self.fields.items() if spec.required]

    def to_schema(self) -> dict[str, Any]:
        """The requested schema: a flat object of primitive properties."""
        return {
            "type": "object",
            "properties": {name: spec.to_schema() for name, spec in self.fields.items()},
            "required": self._required(),
        }

    def to_string_schema(self) -> dict[str, Any]:
        """The same form with every property a string (for string-only clients)."""
        return {
            "type": "object",
            "properties": {name: spec.to_string_schema() for name, spec in self.fields.items()},
            "required": self._required(),
        }

    def coerce(self, values: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate an accepted answer; optional fields left blank take their default.

        Raises ``ValueError`` naming the field when a value does not fit.
        """
        values = dict(values or {})
        result: dict[str, Any] = {}
        for name, spec in self.fields.items():
            value = values.get(name)
            blank = value is None or (
                isinstance(value, str) and not value.strip() and not isinstance(spec, Text)
            )
            if blank:
                if spec.default is not None:
                    result[name] = spec.coerce(spec.default)
                elif spec.required:
                    raise ValueError(f"{name}: an answer is required")
                else:
                    result[name] = None
                continue
            try:
                result[name] = spec.coerce(value)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from None
        return result


Asker = Callable[[ElicitationRequest], Awaitable[Elicited]]
"""Shows a form and returns the answer; raises ``ElicitationUnavailable`` if it cannot."""

_asker: ContextVar[Asker | None] = ContextVar("nooa_elicitation_asker", default=None)


def current_asker() -> Asker | None:
    """The asker bound in this context, or ``None``."""
    return _asker.get()


@contextlib.contextmanager
def bind_asker(asker: Asker | None) -> Iterator[None]:
    """Bind ``asker`` (``None`` unbinds) for code run in this context."""
    token = _asker.set(asker)
    try:
        yield
    finally:
        _asker.reset(token)


async def ask(message: str, *, timeout: float | None = None, **fields: _Field) -> Elicited:
    """Show a form with ``fields`` and return the person's answer.

    Raises ``ElicitationUnavailable`` when no form can be shown here. Declining
    or dismissing returns ``action="decline"`` or ``"cancel"`` with no values;
    a ``timeout`` (seconds) without an answer returns ``"cancel"``.
    """
    asker = current_asker()
    if asker is None:
        raise ElicitationUnavailable("No client form is available here.")
    request = ElicitationRequest(message, fields)
    with _trace(request) as record:
        try:
            if timeout is None:
                answer = await asker(request)
            else:
                async with asyncio.timeout(timeout):
                    answer = await asker(request)
        except TimeoutError:
            answer = Elicited(action="cancel")
        if not isinstance(answer, Elicited):
            answer = Elicited.model_validate(answer)
        if answer.accepted:
            answer = Elicited(action="accept", values=request.coerce(answer.values))
        else:
            answer = Elicited(action=answer.action)
        record(answer.action)
        return answer


def _tracer() -> Any:
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover
        return None
    return trace.get_tracer("nooa.elicit")


@contextlib.contextmanager
def _trace(request: ElicitationRequest) -> Iterator[Callable[[str], None]]:
    """One ``elicitation`` span under the current span (never the values)."""
    tracer = _tracer()
    if tracer is None:  # pragma: no cover - opentelemetry is a core dependency
        yield lambda _action: None
        return
    with tracer.start_as_current_span("elicitation") as span:
        span.set_attribute("elicitation.message", request.message[:2000])
        span.set_attribute("elicitation.fields", list(request.fields))

        def record(action: str) -> None:
            span.set_attribute("elicitation.action", action)

        try:
            yield record
        except ElicitationUnavailable:
            span.set_attribute("elicitation.action", "unavailable")
            raise
