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

        def record(self, effect: str) -> None:
            """Record a side effect (a method call, so it reaches the parent from the sandbox)."""
            self.effects.append(effect)

        def recorded(self) -> list[str]:
            """Return the recorded side effects."""
            return list(self.effects)

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


def _code_span(cell_texts):
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
    ctx = hooks.before_code_execution(
        agent=agent, code="print(report)", execution_id="e1", cell_texts=cell_texts
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
    return span.attributes


def test_code_execution_span_records_text_names_and_lengths_only():
    attrs = _code_span({"report": "SECRET-BODY\nb", "big": "x" * 20_000})
    assert list(attrs["code.texts.names"]) == ["report", "big"]
    assert list(attrs["code.texts.lengths"]) == [13, 20_000]
    assert attrs["code.texts.count"] == 2
    assert attrs["code.texts.truncated"] is False
    assert json.loads(attrs["input.value"]) == {"code": "print(report)"}
    assert not any("SECRET-BODY" in str(value) for value in attrs.values())
    assert "tool.cell_texts" not in attrs


def test_code_execution_span_text_metadata_is_bounded():
    texts = {f"n{i}_" + "a" * 300: "v" for i in range(80)}
    attrs = _code_span(texts)
    assert attrs["code.texts.count"] == 80
    assert attrs["code.texts.truncated"] is True
    assert len(attrs["code.texts.names"]) == 50
    assert len(attrs["code.texts.lengths"]) == 50
    assert max(len(name) for name in attrs["code.texts.names"]) == 100
    assert json.loads(attrs["input.value"]) == {"code": "print(report)"}


def test_code_execution_span_without_texts_has_no_text_attributes():
    attrs = _code_span(None)
    assert not any(key.startswith("code.texts") for key in attrs)


# --- Binding contract: same on both backends ------------------------------------
#
# A cell rejected before it runs leaves every binding unchanged; once execution
# starts, texts persist like ordinary assignments, even if the code then raises.

PRE_EXECUTION_FAILURES = {
    "syntax error": "x = (1,",
    "unterminated string": "x = 'never closed",
    "restricted code": "import subprocess",
    "compile-time error": "for i in range(2):\n    pass\nbreak",
}

_PROBE = (
    "try:\n    seen = report\nexcept NameError:\n    seen = None\n"
    "return_result([seen, len(self.effects)])"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", list(PRE_EXECUTION_FAILURES))
async def test_pre_execution_failure_binds_no_fresh_name(backend, kind):
    code = f"self.effects.append('ran')\n{PRE_EXECUTION_FAILURES[kind]}"
    agent = _agent(
        _llm({"code": code, "texts": [{"name": "report", "text": "new"}]}, {"code": _PROBE}),
        backend,
    )
    try:
        assert await agent.answer() == [None, 0]
        assert _outputs(agent)[0].error
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", list(PRE_EXECUTION_FAILURES))
async def test_pre_execution_failure_keeps_existing_value(backend, kind):
    code = f"self.effects.append('ran')\n{PRE_EXECUTION_FAILURES[kind]}"
    agent = _agent(
        _llm(
            {"code": "report = 'old'"},
            {"code": code, "texts": [{"name": "report", "text": "new"}]},
            {"code": _PROBE},
        ),
        backend,
    )
    try:
        assert await agent.answer() == ["old", 0]
        assert _outputs(agent)[1].error
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_runtime_error_after_start_keeps_texts(backend):
    agent = _agent(
        _llm(
            {"code": "report = 'old'"},
            {
                "code": "used = report + '!'\nraise ValueError('boom')",
                "texts": [{"name": "report", "text": "new"}, {"name": "fresh", "text": "f"}],
            },
            {"code": "return_result([report, fresh, used])"},
        ),
        backend,
    )
    try:
        assert await agent.answer() == ["new", "f", "new!"]
        assert "ValueError" in (_outputs(agent)[1].error or "")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_texts_read_back_in_next_cell_on_success(backend):
    agent = _agent(
        _llm(
            {"code": "print(len(report))", "texts": [{"name": "report", "text": "abc"}]},
            {"code": "return_result(report)"},
        ),
        backend,
    )
    try:
        assert await agent.answer() == "abc"
    finally:
        await agent.aclose()


_HELPERS = """\
def tag(fn):
    fn.tag = report
    return fn

@tag
def f():
    return report

def g(x=report):
    return x
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_texts_visible_to_same_cell_helpers_decorators_and_defaults(backend):
    agent = _agent(
        _llm(
            {
                "code": _HELPERS + "same = [f(), f.tag, g()]",
                "texts": [{"name": "report", "text": "R"}],
            },
            {"code": "return_result([same, f(), f.tag, g()])"},
        ),
        backend,
    )
    try:
        assert await agent.answer() == [["R", "R", "R"], "R", "R", "R"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_texts_from_earlier_cell_visible_to_helpers(backend):
    agent = _agent(
        _llm(
            {"code": "", "texts": [{"name": "report", "text": "R"}]},
            {"code": _HELPERS + "same = [f(), f.tag, g()]"},
            {"code": "return_result([same, f(), f.tag, g()])"},
        ),
        backend,
    )
    try:
        assert await agent.answer() == [["R", "R", "R"], "R", "R", "R"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.sandbox
async def test_reserved_set_covers_every_sandbox_worker_name(monkeypatch):
    """The sandbox worker builds its namespace on its own; capture the real one.

    Every name in a live worker cell's globals that the cell did not define must
    be reserved, so a text can never shadow something the worker injects.
    (``globals()`` is a restricted call, so the cell reads its frame's globals.)
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
                "code": "import sys\nreturn_result(sorted(sys._getframe(0).f_globals))",
                "texts": [{"name": "report", "text": "x"}],
            },
        ),
        "sandbox",
    )
    try:
        names = set(await agent.answer())
    finally:
        await agent.aclose()
    (reserved,) = seen
    cell_defined = {"mine", "report", "topic", "sys"}  # cell locals, texts, parameters
    injected = {n for n in names - cell_defined if not (n.startswith("__") and n.endswith("__"))}
    assert {"self", "return_result", "mine", "report"} <= names  # the live worker namespace
    assert injected - reserved == set()


# --- Rejected cells install no helpers ------------------------------------------
#
# In-process, helper defs are pre-compiled before the runtime runs the cell (so
# ``@strategy`` sees ``_generated_source``). That pre-compilation is gated on the
# runtime's own pre-execution checks: a cell the runtime would reject (restricted
# code, compile error) runs no decorator or default and installs or replaces no
# helper, so a rejected text cannot escape through a helper's globals.

HELPER_STAGE_REJECTIONS = {
    "restricted code": "import subprocess",
    "compile-time error": "for i in range(2):\n    pass\nbreak",
}

_SIDE_EFFECT_HELPERS = """\
def deco(fn):
    self.record('deco')
    return fn

@deco
def f():
    return report

def k(x=self.record('default')):
    return report
"""

_HELPER_PROBE = (
    "out = []\n"
    "for probe in (lambda: report, lambda: f(), lambda: k()):\n"
    "    try:\n        out.append(probe())\n"
    "    except NameError:\n        out.append(None)\n"
    "return_result([out, self.recorded()])"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", list(HELPER_STAGE_REJECTIONS))
async def test_rejected_cell_installs_no_helper_capturing_its_texts(backend, kind):
    code = _SIDE_EFFECT_HELPERS + HELPER_STAGE_REJECTIONS[kind]
    agent = _agent(
        _llm({"code": code, "texts": [{"name": "report", "text": "new"}]}, {"code": _HELPER_PROBE}),
        backend,
    )
    try:
        # No text, no helper, and neither the decorator nor the default ran.
        assert await agent.answer() == [[None, None, None], []]
        assert _outputs(agent)[0].error
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", list(HELPER_STAGE_REJECTIONS))
async def test_rejected_cell_keeps_existing_text_and_helper(backend, kind):
    agent = _agent(
        _llm(
            {"code": "report = 'old'\ndef f():\n    return report\n\ndef k():\n    return report"},
            {
                "code": _SIDE_EFFECT_HELPERS + HELPER_STAGE_REJECTIONS[kind],
                "texts": [{"name": "report", "text": "new"}],
            },
            {"code": _HELPER_PROBE},
        ),
        backend,
    )
    try:
        assert await agent.answer() == [["old", "old", "old"], []]
        assert _outputs(agent)[1].error
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_accepted_cell_still_binds_texts_and_helpers(backend):
    """Control: the same cells without the rejected line run and persist."""
    agent = _agent(
        _llm(
            {"code": "report = 'old'"},
            {"code": _SIDE_EFFECT_HELPERS, "texts": [{"name": "report", "text": "new"}]},
            {"code": _HELPER_PROBE},
        ),
        backend,
    )
    try:
        values, effects = await agent.answer()
        assert values == ["new", "new", "new"]
        # Decorators and defaults may run more than once (pre-compile and rebind).
        assert {"deco", "default"} == set(effects)
    finally:
        await agent.aclose()


# Known stage difference (kept): in-process, helper pre-compilation evaluates
# decorators and defaults before the cell runs, so an undefined name there (a
# typo such as ``@stratgy``) rejects the cell before execution starts. The
# sandbox has no pre-compile stage: the name fails at runtime, after execution
# started, so the texts persist like assignments.
_TYPO_HELPER = "self.record('ran')\n@stratgy\ndef f():\n    return report\n"


@pytest.mark.asyncio
async def test_undefined_decorator_rejects_before_execution_in_process():
    agent = _agent(
        _llm({"code": _TYPO_HELPER, "texts": [{"name": "report", "text": "new"}]}, {"code": _PROBE})
    )
    try:
        assert await agent.answer() == [None, 0]
        assert "Failed to define helper" in (_outputs(agent)[0].error or "")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.sandbox
async def test_undefined_decorator_fails_at_runtime_in_sandbox():
    agent = _agent(
        _llm(
            {"code": _TYPO_HELPER, "texts": [{"name": "report", "text": "new"}]}, {"code": _PROBE}
        ),
        "sandbox",
    )
    try:
        assert await agent.answer() == ["new", 1]  # execution started in the worker
        assert "NameError" in (_outputs(agent)[0].error or "")
    finally:
        await agent.aclose()


# --- Runtime API: the non-wrapper path ------------------------------------------
#
# CodeAct and PurePython always run cells with ``wrap_in_function=True``; the
# ``wrap_in_function=False`` path of ``ActorRuntime.execute_code`` is only
# reachable through the runtime API, so it is tested there.


class _PlainAgent(Agent, llm=FakeLLMClient(scripted_responses=[])):
    """Agent for runtime-API tests (no LLM calls)."""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,bound,error",
    [
        ("x = report + '!'", True, None),
        ("x = report\nraise ValueError('boom')", True, "ValueError"),
        ("x = (1,", False, "SyntaxError"),
        ("import subprocess", False, "subprocess"),
    ],
)
async def test_non_wrapper_execute_code_binds_texts_once_started(code, bound, error):
    agent = _PlainAgent()
    try:
        result = await agent.runtime.execute_code(code, text_bindings={"report": "R"})
        if error is None:
            assert result.error is None
        else:
            assert error in f"{type(result.error).__name__}: {result.error}"
        assert (result.captured_locals.get("report") == "R") is bound
        assert ("report" in result.captured_locals) is bound
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,error",
    [
        ("import subprocess", "subprocess"),
        ("for i in range(2):\n    pass\nbreak", "break"),
        ("x = (1,", "SyntaxError"),
    ],
)
async def test_check_code_matches_execute_code_rejection(code, error):
    """``check_code`` runs the runtime's own pre-execution checks and executes nothing."""
    agent = _PlainAgent()
    try:
        runtime = agent.runtime
        checked = runtime.check_code(code, wrap_in_function=True, text_bindings={"report": "R"})
        assert checked is not None
        executed = await runtime.execute_code(
            code, wrap_in_function=True, text_bindings={"report": "R"}
        )
        assert type(checked) is type(executed.error)
        assert error in f"{type(checked).__name__}: {checked}"
        assert runtime.check_code("x = report", wrap_in_function=True) is None
    finally:
        await agent.aclose()
