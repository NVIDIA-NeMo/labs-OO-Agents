# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import boundaries and neutral types used by the future public shell."""

import ast
import inspect
import traceback
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel

from nooa.unifiedllm import (
    FailureKind,
    LLMFailure,
    LLMRequest,
    Tool,
    create_tool_from_callable,
    tools,
    unifiedllm,
)
from nooa.unifiedllm.adapters import schema
from nooa.unifiedllm.errors import LLMFailure as ErrorFailure


def test_historical_import_identity():
    assert Tool is tools.Tool is unifiedllm.Tool
    assert create_tool_from_callable is tools.create_tool_from_callable
    assert create_tool_from_callable is unifiedllm.create_tool_from_callable
    assert LLMFailure is ErrorFailure
    for name in (
        "_resolve_schema_refs",
        "_strip_schema_noise",
        "_clean_schema",
        "_strict_schema_valid",
    ):
        assert getattr(unifiedllm, name) is getattr(schema, name)


@pytest.mark.parametrize("module", [tools, inspect.getmodule(LLMRequest)])
def test_neutral_modules_do_not_depend_on_monolith(module):
    tree = ast.parse(Path(module.__file__).read_text())
    # Include lazy imports too: they must not sneak the dependency back in.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("nooa.unifiedllm.unifiedllm", "unifiedllm")
        elif isinstance(node, ast.Import):
            assert all(alias.name != "nooa.unifiedllm.unifiedllm" for alias in node.names)


def test_callable_constructor_defaults_and_execution():
    def add(self, x: int, label="hi"):
        """Add a labelled item."""
        return x, label

    tool = create_tool_from_callable(add)
    assert (tool.name, tool.description, tool.callable, tool.parameters_model) == (
        "add",
        "Add a labelled item.",
        add,
        None,
    )
    assert tool.callable(None, 2) == (2, "hi")
    assert tool.get_parameter_schema() == {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "label": {"type": "string"}},
        "required": ["x"],
    }
    assert tool.get_parameter_schema(strict=True) == {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "label": {"type": "string"}},
        "required": ["x", "label"],
        "additionalProperties": False,
    }

    def empty():
        return 3

    empty_tool = create_tool_from_callable(empty)
    assert empty_tool.description == "Call the empty function"
    assert empty_tool.get_parameter_schema() == {"type": "object", "properties": {}, "required": []}


def test_parameter_model_overrides_callable_and_delegates_cleanup(monkeypatch):
    class Params(BaseModel):
        count: int

    tool = Tool("fn", "desc", lambda: None, Params)
    seen = []

    def clean(raw, *, strict):
        seen.append((raw, strict))
        return {"sentinel": True}

    monkeypatch.setattr(schema, "_clean_schema", clean)
    assert tool.get_parameter_schema(strict=True) == {"sentinel": True}
    assert seen == [(Params.model_json_schema(), True)]


def test_request_fields_defaults_and_identity_are_unchanged():
    assert list(LLMRequest.model_fields) == [
        "messages",
        "tools",
        "tool_choice",
        "output_type",
        "reasoning_level",
        "max_output_tokens",
    ]
    assert get_args(LLMRequest.model_fields["tools"].annotation) == (Tool,)
    request = LLMRequest()
    assert request.model_dump() == {
        "messages": [],
        "tools": [],
        "tool_choice": "auto",
        "output_type": None,
        "reasoning_level": None,
        "max_output_tokens": None,
    }
    assert request.tools is not LLMRequest().tools
    history = [{"role": "user", "content": "hello"}]
    passed_tools = [create_tool_from_callable(lambda: None)]
    request = LLMRequest(messages=history, tools=passed_tools)
    assert request.messages is history
    assert request.tools is passed_tools


KINDS = (
    "rate_limit",
    "transient",
    "timeout",
    "context_window",
    "media_limit",
    "media_rejected",
    "output_limit",
    "bad_request",
    "auth",
    "config",
)


def test_exact_failure_taxonomy():
    assert tuple(kind.value for kind in FailureKind) == KINDS


@pytest.mark.parametrize("kind", KINDS)
def test_failure_kind_message_and_absent_facts(kind):
    error = LLMFailure(kind, "refused")
    assert isinstance(error.kind, FailureKind)
    assert error.kind == kind
    assert LLMFailure(FailureKind(kind), "refused").kind is error.kind
    assert error.args == ("refused",)
    assert str(error) == "refused"
    assert repr(error) == "LLMFailure('refused')"
    assert error.__cause__ is None
    for field in (
        "prompt_tokens",
        "context_window",
        "media_kind",
        "media_count",
        "media_count_limit",
        "status_code",
        "retry_after",
    ):
        assert getattr(error, field) is None


@pytest.mark.parametrize("kind", ["", "unknown", "RATE_LIMIT", 1, None, [], {}])
def test_invalid_failure_kind(kind):
    with pytest.raises(ValueError):
        LLMFailure(kind, "refused")


@pytest.mark.parametrize(
    "field, invalid",
    [
        ("prompt_tokens", -1),
        ("prompt_tokens", True),
        ("prompt_tokens", "1"),
        ("prompt_tokens", 1.5),
        ("context_window", 0),
        ("context_window", False),
        ("media_count", -1),
        ("media_count", []),
        ("media_count_limit", 2.0),
        ("media_count_limit", -1),
        ("media_kind", "text"),
        ("media_kind", []),
        ("status_code", 99),
        ("status_code", 600),
        ("status_code", "429"),
        ("retry_after", -0.1),
        ("retry_after", True),
        ("retry_after", "2"),
        ("retry_after", float("nan")),
        ("retry_after", float("inf")),
        ("retry_after", 10**400),
        ("cause", "error"),
        ("message", None),
    ],
)
def test_malformed_failure_metadata(field, invalid):
    kwargs = {field: invalid}
    if field != "message":
        kwargs["message"] = "refused"
    with pytest.raises(ValueError, match=field):
        LLMFailure("bad_request", **kwargs)


@pytest.mark.parametrize("media_kind", ["image", "pdf", "audio", "video"])
def test_normalized_recovery_metadata(media_kind):
    error = LLMFailure(
        "media_limit",
        "too many",
        prompt_tokens=0,
        context_window=1,
        media_kind=media_kind,
        media_count=5,
        media_count_limit=0,
        status_code=429,
        retry_after=2,
    )
    assert (error.prompt_tokens, error.context_window) == (0, 1)
    assert (error.media_kind, error.media_count, error.media_count_limit) == (media_kind, 5, 0)
    assert error.status_code == 429
    assert error.retry_after == 2.0
    assert type(error.retry_after) is float
    assert LLMFailure("timeout", "wait", retry_after=0).retry_after == 0.0


def test_exception_cause_traceback_and_context_work_normally():
    cause = RuntimeError("transport refused")
    error = LLMFailure("transient", "unavailable", cause=cause)
    assert error.__cause__ is cause
    try:
        raise error
    except LLMFailure as caught:
        assert caught is error
        assert caught.__traceback__ is not None
        rendered = "".join(traceback.format_exception(caught))
        assert "RuntimeError: transport refused" in rendered
        assert "direct cause" in rendered
        assert "LLMFailure: unavailable" in rendered

    try:
        raise cause
    except RuntimeError:
        try:
            raise LLMFailure("transient", "failed") from cause
        except LLMFailure as caught:
            assert caught.__cause__ is cause
            assert caught.__context__ is cause
