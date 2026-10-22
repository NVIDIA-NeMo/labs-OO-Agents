# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.elicit: schemas, coercion, binding and timeouts."""

from __future__ import annotations

import asyncio

import pytest

from nooa import elicit
from nooa.elicit import (
    Choice,
    Confirm,
    ElicitationRequest,
    ElicitationUnavailable,
    Elicited,
    Integer,
    MultiChoice,
    Number,
    Text,
    ask,
    bind_asker,
    current_asker,
)


def _fields():
    return {
        "name": Text(title="Name", default="me", min_length=1, pattern="^[a-z]+$"),
        "count": Integer(minimum=1, maximum=10),
        "ratio": Number(required=False),
        "ok": Confirm(title="OK?"),
        "kind": Choice(options={"a": "Alpha", "b": "Beta"}),
        "plain": Choice(options=["x", "y"]),
        "tags": MultiChoice(options=["p", "q"], required=False),
    }


def test_schema_is_the_flat_form_subset():
    schema = ElicitationRequest("Pick", _fields()).to_schema()
    props = schema["properties"]
    assert schema["type"] == "object"
    assert schema["required"] == ["name", "count", "ok", "kind", "plain"]
    assert props["name"] == {
        "type": "string",
        "title": "Name",
        "default": "me",
        "pattern": "^[a-z]+$",
        "minLength": 1,
    }
    assert props["count"] == {"type": "integer", "minimum": 1, "maximum": 10}
    assert props["ratio"] == {"type": "number"}
    assert props["ok"] == {"type": "boolean", "title": "OK?"}
    assert props["kind"]["oneOf"] == [
        {"const": "a", "title": "Alpha"},
        {"const": "b", "title": "Beta"},
    ]
    assert props["plain"] == {"type": "string", "enum": ["x", "y"]}
    assert props["tags"] == {"type": "array", "items": {"type": "string", "enum": ["p", "q"]}}
    # Every property is a primitive (or a multi-select of strings): no nesting.
    assert all(
        p["type"] in {"string", "number", "integer", "boolean", "array"} for p in props.values()
    )


def test_schema_is_accepted_by_acp():
    from acp.schema import ElicitationSchema

    parsed = ElicitationSchema.model_validate(ElicitationRequest("Pick", _fields()).to_schema())
    assert set(parsed.properties or {}) == set(_fields())


def test_string_schema_has_only_strings():
    schema = ElicitationRequest("Pick", _fields()).to_string_schema()
    assert {p["type"] for p in schema["properties"].values()} == {"string"}
    assert "minimum" not in schema["properties"]["count"]
    assert schema["properties"]["ok"]["oneOf"][0] == {"const": "true", "title": "Yes"}


def test_coerces_string_answers():
    request = ElicitationRequest("Pick", _fields())
    values = request.coerce(
        {
            "name": "bob",
            "count": "3",
            "ratio": "0.5",
            "ok": "false",
            "kind": "b",
            "plain": "x",
            "tags": "q, p",
        }
    )
    assert values == {
        "name": "bob",
        "count": 3,
        "ratio": 0.5,
        "ok": False,
        "kind": "b",
        "plain": "x",
        "tags": ["q", "p"],
    }


def test_blank_optional_takes_default_or_none():
    request = ElicitationRequest(
        "Pick", {"name": Text(default="me"), "n": Integer(required=False), "c": Confirm()}
    )
    assert request.coerce({"c": True}) == {"name": "me", "n": None, "c": True}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (Integer(minimum=1), "0"),
        (Integer(), "1.5"),
        (Integer(), True),
        (Number(), "nan"),
        (Confirm(), "maybe"),
        (Choice(options=["a"]), "b"),
        (MultiChoice(options=["a"]), "a, z"),
        (Text(max_length=2), "abc"),
        (Text(pattern="^a"), "b"),
    ],
)
def test_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match="f:"):
        ElicitationRequest("m", {"f": field}).coerce({"f": value})


def test_required_answer_missing():
    with pytest.raises(ValueError, match="required"):
        ElicitationRequest("m", {"f": Integer()}).coerce({})


def test_bad_field_names_and_types():
    with pytest.raises(ValueError):
        ElicitationRequest("m", {"not a name": Text()})
    with pytest.raises(TypeError):
        ElicitationRequest("m", {"f": "text"})  # type: ignore[dict-item]


async def test_no_asker_raises_at_once():
    assert current_asker() is None
    with pytest.raises(ElicitationUnavailable):
        await ask("Hello?", name=Text())


async def test_ask_coerces_the_askers_answer():
    seen = []

    async def asker(request):
        seen.append(request)
        return Elicited(action="accept", values={"n": "7", "ok": "true"})

    with bind_asker(asker):
        assert current_asker() is asker
        answer = await ask("Count?", n=Integer(), ok=Confirm())
    assert current_asker() is None
    assert answer.accepted and answer.values == {"n": 7, "ok": True}
    assert seen[0].message == "Count?" and list(seen[0].fields) == ["n", "ok"]


@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_decline_and_cancel_have_no_values(action):
    async def asker(_request):
        return Elicited(action=action, values={"n": "1"})

    with bind_asker(asker):
        answer = await ask("Count?", n=Integer())
    assert answer.action == action and not answer.accepted and answer.values == {}


async def test_asker_unavailable_propagates():
    async def asker(_request):
        raise ElicitationUnavailable("no forms")

    with bind_asker(asker), pytest.raises(ElicitationUnavailable):
        await ask("Count?", n=Integer())


async def test_timeout_is_cancel():
    async def asker(_request):
        await asyncio.sleep(10)
        raise AssertionError("not reached")

    with bind_asker(asker):
        answer = await ask("Count?", timeout=0.01, n=Integer())
    assert answer.action == "cancel"


async def test_binding_reaches_tasks_created_inside():
    async def asker(_request):
        return Elicited(action="accept", values={"ok": True})

    async def in_task():
        return await ask("ok?", ok=Confirm())

    with bind_asker(asker):
        task = asyncio.create_task(in_task())
    assert (await task).values == {"ok": True}


async def test_records_a_trace_span(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(elicit, "_tracer", lambda: provider.get_tracer("test"))

    async def asker(_request):
        return Elicited(action="decline")

    with bind_asker(asker):
        await ask("Secretless?", name=Text())
    (span,) = exporter.get_finished_spans()
    assert span.name == "elicitation"
    assert span.attributes["elicitation.message"] == "Secretless?"
    assert tuple(span.attributes["elicitation.fields"]) == ("name",)
    assert span.attributes["elicitation.action"] == "decline"
