# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python_cell ``texts``: named text variables bound as data before a cell runs."""

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks import ToolCallEvent
from nooa.events import PythonOutput
from nooa.strategies.cell_texts import (
    PYTHON_BUILTIN_NAMES,
    TEXTS_HINT,
    CellTextsError,
    parse_cell_texts,
)
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from nooa.unifiedllm.unifiedllm import _strict_schema_valid

BACKENDS = ["inprocess", pytest.param("sandbox", marks=pytest.mark.sandbox)]


def module_helper() -> str:
    """A module-level tool function, visible in every cell of agents defined here."""
    return "helper"


def _cell(args: dict[str, Any], index: int) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"cell_{index}", name="python_cell", arguments=json.dumps(args))],
        finish_reason="tool_calls",
    )


def _llm(*cells: dict[str, Any]) -> FakeLLMClient:
    return FakeLLMClient(
        scripted_responses=[_cell(args, index) for index, args in enumerate(cells)],
        strict_exhaustion=True,
    )


def _agent(llm: FakeLLMClient, backend: str = "inprocess", returns: Any = Any) -> Agent:
    class TextsAgent(Agent, llm=llm):
        """Records side effects so tests can prove a rejected cell did not run."""

        effects: list[str] = []

        def namespace_names(self) -> list[str]:
            """Return the names bound in the calling cell's namespace."""
            frame = sys._getframe(1)
            while not frame.f_code.co_filename.startswith("Cell In["):
                frame = frame.f_back
            return sorted(frame.f_globals)

        @strategy(
            CodeActV2(config=CodeActConfig(prefill=None, execution_backend=backend, max_retries=5))
        )
        async def answer(self, topic: str = "t") -> returns:
            """Answer."""
            ...

    agent = TextsAgent()
    agent.effects = []
    return agent


def _outputs(agent: Agent) -> list[PythonOutput]:
    return list(agent.events.query(type="PythonOutput"))


SPECIAL_TEXTS = {
    "quotes": "she said \"hi\" and 'bye'",
    "triple": "a \"\"\"triple\"\"\" and '''single triple''' end",
    "backslashes": "C:\\new\\table \\n not a newline \\\\ \\x41 \\u00e9",
    "trailing_newline": "line one\nline two\n",
    "unicode": "naïve café — ✓ 漢字 🚀 \u200b",
    "big": ("0123456789abcdef\"'\\\n" * 5200)[:100_000],
    "source_like": "x = 1\nraise SystemExit('should never run')\n",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("key", list(SPECIAL_TEXTS))
async def test_binding_is_exact_str(backend, key):
    text = SPECIAL_TEXTS[key]
    agent = _agent(
        _llm(
            {
                "code": "assert type(value) is str\nreturn_result(value)",
                "texts": [{"name": "value", "text": text}],
            }
        ),
        backend,
    )
    try:
        assert await agent.answer() == text
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_text_is_data_never_source():
    """A text that looks like code is a value; none of it executes."""
    agent = _agent(
        _llm(
            {
                "code": "return_result([src, 'x' in self.namespace_names()])",
                "texts": [{"name": "src", "text": "x = 1\nself.effects.append('ran')"}],
            }
        )
    )
    try:
        assert await agent.answer() == ["x = 1\nself.effects.append('ran')", False]
        assert agent.effects == []
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_texts_persist_to_next_cell_and_may_be_rebound(backend):
    agent = _agent(
        _llm(
            {"code": "first = report", "texts": [{"name": "report", "text": "v1"}]},
            {"code": "report_before = report", "texts": [{"name": "report", "text": "v2"}]},
            {"code": "return_result([first, report_before, report])"},
        ),
        backend,
    )
    try:
        assert await agent.answer() == ["v1", "v2", "v2"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_text_may_rebind_ordinary_local_and_parameter():
    agent = _agent(
        _llm(
            {"code": "note = 5"},
            {
                "code": "return_result([note, topic])",
                "texts": [{"name": "note", "text": "n"}, {"name": "topic", "text": "p"}],
            },
        )
    )
    try:
        assert await agent.answer() == ["n", "p"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "args",
    [
        {"texts": [{"name": "report", "text": "only texts"}]},
        {"code": "", "texts": [{"name": "report", "text": "only texts"}]},
        {"code": "  \n", "texts": [{"name": "report", "text": "only texts"}]},
    ],
)
async def test_texts_only_cell_binds_without_code(backend, args):
    agent = _agent(_llm(args, {"code": "return_result(report)"}), backend)
    try:
        assert await agent.answer() == "only texts"
        first = _outputs(agent)[0]
        assert first.error in (None, "")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"code": "return_result(7)"},
        {"code": "return_result(7)", "texts": None},
        {"code": "return_result(7)", "texts": []},
    ],
)
async def test_omitting_texts_is_unchanged(args):
    agent = _agent(_llm(args))
    try:
        assert await agent.answer() == 7
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_empty_code_without_texts_is_still_an_error():
    agent = _agent(_llm({"code": ""}, {"code": "return_result(1)"}))
    try:
        assert await agent.answer() == 1
        assert "empty code" in (_outputs(agent)[0].error or "")
    finally:
        await agent.aclose()


BAD_TEXTS = {
    "bad identifier": ([{"name": "my-report", "text": "x"}], "not a valid variable name"),
    "leading digit": ([{"name": "1st", "text": "x"}], "not a valid variable name"),
    "keyword": ([{"name": "class", "text": "x"}], "not a valid variable name"),
    "duplicate": (
        [{"name": "report", "text": "a"}, {"name": "report", "text": "b"}],
        "duplicate name 'report'",
    ),
    "reserved self": ([{"name": "self", "text": "x"}], "runtime-provided name"),
    "reserved return_result": ([{"name": "return_result", "text": "x"}], "runtime-provided"),
    "dunder": ([{"name": "__name__", "text": "x"}], "dunder name"),
    "not a list": ({"name": "report", "text": "x"}, "must be a list"),
    "missing key": ([{"name": "report"}], "exactly the keys"),
    "non-string text": ([{"name": "report", "text": 3}], "must be strings"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(BAD_TEXTS))
async def test_invalid_texts_reject_the_cell_and_run_nothing(backend, case):
    texts, message = BAD_TEXTS[case]
    good = [{"name": "early", "text": "bound?"}]
    texts = [*good, *texts] if isinstance(texts, list) else texts
    agent = _agent(
        _llm(
            {"code": "self.effects.append('ran')", "texts": texts},
            {
                "code": (
                    "try:\n    early\n    bound = True\nexcept NameError:\n    bound = False\n"
                    "return_result([len(self.effects), bound])"
                )
            },
        ),
        backend,
    )
    try:
        # Sandbox: self.effects lives in the parent, the worker proxies calls.
        assert await agent.answer() == [0, False]
        error = _outputs(agent)[0].error or ""
        assert error.startswith("Texts error: ")
        assert message in error
        assert "\n" not in error
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "self",
        "return_result",
        "print",
        "pprint",
        "doc",
        "show",
        "asyncio",
        "typing",
        "python_cell_state",
        "Image",
        "strategy",
        "len",
        "json",  # a module-level import of the agent's module (skills/tools live here)
        "module_helper",  # a module-level function of the agent's module
        "Agent",
    ],
)
async def test_runtime_helpers_are_reserved(name):
    agent = _agent(
        _llm(
            {"code": "self.effects.append('ran')", "texts": [{"name": name, "text": "x"}]},
            {"code": "return_result(len(self.effects))"},
        )
    )
    try:
        assert await agent.answer() == 0
        assert "runtime-provided name" in (_outputs(agent)[0].error or "")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_reserved_set_covers_every_runtime_injected_name(monkeypatch):
    """Everything in a live cell's namespace that the cell did not define is reserved.

    Captured from the running cell, so a helper added to the runtime later is
    caught here if the reserved set stops covering it.
    """
    seen: list[frozenset[str]] = []
    original = CodeActStrategy._reserved_cell_names

    def recording(self, runtime, builtins, call):
        reserved = original(self, runtime, builtins, call)
        seen.append(reserved)
        return reserved

    monkeypatch.setattr(CodeActStrategy, "_reserved_cell_names", recording)
    agent = _agent(
        _llm(
            {"code": "mine = 1"},
            {
                "code": "return_result(self.namespace_names())",
                "texts": [{"name": "report", "text": "x"}],
            },
        )
    )
    try:
        names = set(await agent.answer())
    finally:
        await agent.aclose()
    (reserved,) = seen
    cell_defined = {"mine", "report", "topic"}  # cell locals, texts and method parameters
    injected = {n for n in names - cell_defined if not (n.startswith("__") and n.endswith("__"))}
    assert injected - reserved == set()
    assert {"self", "return_result", "print", "pprint", "doc", "show", "asyncio", "typing"} <= (
        reserved
    )
    assert PYTHON_BUILTIN_NAMES <= reserved
    assert not ({"topic", "mine", "report"} & reserved)


def test_parse_cell_texts_unit():
    assert parse_cell_texts(None, ()) == {}
    assert parse_cell_texts([], ()) == {}
    assert parse_cell_texts([{"name": "a", "text": "1"}, {"name": "b", "text": ""}], ()) == {
        "a": "1",
        "b": "",
    }
    with pytest.raises(CellTextsError, match="runtime-provided"):
        parse_cell_texts([{"name": "a", "text": "1"}], {"a"})
    # Soft keywords are valid variable names.
    assert parse_cell_texts([{"name": "match", "text": "m"}], ()) == {"match": "m"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,hinted",
    [
        ('x = """never closed\nmore', True),
        ("x = 'never closed", True),
        ('cmd = "a\\" b', True),  # an escaped closing quote leaves the literal open
        ("x = (1,", False),
        ("def f(:\n    pass", False),
        ("1 +* 2", False),
        ("x = 'a' \\ 'b'", True),  # unexpected character after line continuation character
    ],
)
async def test_syntax_error_hint_points_at_texts(code, hinted):
    agent = _agent(_llm({"code": code}, {"code": "return_result(1)"}))
    try:
        assert await agent.answer() == 1
        error = _outputs(agent)[0].error or ""
        assert "Error" in error
        assert (TEXTS_HINT in error) is hinted
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_execute_python_has_no_texts_hint_or_argument():
    """The legacy two-tool CodeAct is unchanged."""
    strategy_ = CodeActStrategy()
    tool = strategy_._build_execute_python_tool()
    assert set(tool.get_parameter_schema()["properties"]) == {"code"}
    assert strategy_._supports_cell_texts() is False


def test_schema_contains_texts_array():
    tool = CodeActV2()._build_execute_python_tool()
    schema = tool.get_parameter_schema()
    assert schema["required"] == ["code"]
    texts = schema["properties"]["texts"]
    array = next(option for option in texts["anyOf"] if option.get("type") == "array")
    assert array["items"]["properties"] == {"name": {"type": "string"}, "text": {"type": "string"}}
    assert set(array["items"]["required"]) == {"name", "text"}
    strict = tool.get_parameter_schema(strict=True)
    assert _strict_schema_valid(strict)
    assert "`texts`" in tool.description
    assert '[{"name": "report", "text": "..."}]' in tool.description


@pytest.mark.asyncio
async def test_history_keeps_texts_in_arguments_and_output_does_not_echo_them():
    text = "UNIQUE-TEXT-MARKER " * 3
    agent = _agent(
        _llm(
            {"code": "print(len(report))", "texts": [{"name": "report", "text": text}]},
            {"code": "return_result(1)"},
        )
    )
    try:
        await agent.answer()
        call = next(
            e for e in agent.events.query(type="ToolCallEvent") if e.tool_call_id == "cell_0"
        )
        assert isinstance(call, ToolCallEvent)
        assert call.arguments["texts"] == [{"name": "report", "text": text}]
        output = _outputs(agent)[0]
        assert output.stdout.strip() == str(len(text))
        assert "UNIQUE-TEXT-MARKER" not in (output.stdout + output.stderr + (output.error or ""))
    finally:
        await agent.aclose()


def test_code_execution_span_records_texts():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nooa.tracing._hooks_impl import OpenInferenceHooks

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    hooks = OpenInferenceHooks(tracer=provider.get_tracer("test"))
    agent = MagicMock()
    type(agent).__name__ = "TestAgent"
    texts = {"report": "a\nb", "big": "x" * 20_000}
    ctx = hooks.before_code_execution(
        agent=agent, code="print(report)", execution_id="e1", cell_texts=texts
    )
    hooks.after_code_execution(
        agent=agent,
        code="print(report)",
        result=None,
        exception=None,
        context=ctx,
        execution_id="e1",
    )
    (span,) = exporter.get_finished_spans()
    attrs = span.attributes
    assert list(attrs["code.texts.names"]) == ["report", "big"]
    assert list(attrs["code.texts.lengths"]) == [3, 20_000]
    payload = json.loads(attrs["input.value"])
    assert payload["code"] == "print(report)"
    assert payload["texts"][0] == {"name": "report", "text": "a\nb"}
    assert len(payload["texts"][1]["text"]) == 10_000
    assert "tool.cell_texts" not in attrs
