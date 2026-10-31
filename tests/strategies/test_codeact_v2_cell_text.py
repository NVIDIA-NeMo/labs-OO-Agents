# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python_cell's optional raw ``text`` argument: schema, verbatim binding, errors, guard."""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput
from nooa.runtime.harness_metrics import HarnessMetrics, get_span_schema
from nooa.strategies.cell_text import CELL_TEXT_DESCRIPTION, parse_cell_text
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.loop_guard import action_fingerprint
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

EDGE_CASES = {
    "quotes.py": "print(\"double\") + 'single' + `back`",
    "backslashes.txt": "C:\\new\\table\\x00 \\\\n \\u00e9 \\",
    "triple.py": 'a = """doc"""\nb = \'\'\'doc\'\'\'\n"""unterminated',
    "tabs.txt": "\tindented\t\ttabs\n\t",
    "crlf.txt": "line1\r\nline2\r\n\rlone cr\n",
    "unicode.txt": "héllo wörld — 日本語 🤖 \u200b zero-width, \U0001f600",
    "nul": "a\x00b",
    "trailing quote": 'ends with "',
    "empty": "",
}
BIG = "x\\\"'\n" * (300 * 1024)  # 1.5 MiB of quote/backslash-heavy text


def _cell(code: str, call_id: str, *, text=None, has_text: bool = True) -> LLMResponse:
    args: dict = {"code": code}
    if has_text and text is not None:
        args["text"] = text
    return LLMResponse(
        parts=(ToolCall(id=call_id, name="python_cell", arguments=json.dumps(args)),),
        finish_reason="tool_calls",
    )


def _agent(responses, strategy_type=CodeActV2):
    llm = FakeLLMClient(scripted_responses=responses)

    class TextAgent(Agent, llm=llm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ran = 0

        def mark(self) -> None:
            """Count a run."""
            self.ran += 1

        @strategy(strategy_type(config=CodeActConfig(prefill=None, max_retries=10)))
        async def answer(self) -> dict:
            """Return a dict."""
            ...

    return TextAgent()


def _outputs(agent) -> list[PythonOutput]:
    return [e for e in agent.event_manager.values() if isinstance(e, PythonOutput)]


def test_python_cell_schema_has_optional_object_text():
    schema = CodeActV2()._build_execute_python_tool().get_parameter_schema()
    assert schema["required"] == ["code"]
    assert schema["properties"]["code"]["type"] == "string"
    assert schema["properties"]["text"] == {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": CELL_TEXT_DESCRIPTION,
    }
    assert "anyOf" not in json.dumps(schema)


def test_codeact_v1_execute_python_schema_is_unchanged():
    tool = CodeActStrategy()._build_execute_python_tool()
    assert tool.name == "execute_python"
    assert tool.get_parameter_schema() == {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }


def test_python_cell_description_mentions_text_once():
    description = CodeActV2()._build_execute_python_tool().description
    assert description.count("`text` argument") == 1


async def test_text_round_trips_verbatim_including_edge_cases_and_a_large_value():
    texts = {**EDGE_CASES, "big": BIG}
    agent = _agent([_cell("return_result(dict(text))", "1", text=texts)])
    result = await agent.answer()
    assert result == texts
    assert len(result["big"]) >= 1024 * 1024


async def test_text_is_the_exact_dict_and_files_written_from_it_match(tmp_path):
    target = tmp_path / "out"
    code = (
        f"import pathlib\nroot = pathlib.Path({str(target)!r})\nroot.mkdir()\n"
        "for name, content in text.items():\n"
        "    (root / name.replace(' ', '_')).write_bytes(content.encode('utf-8'))\n"
        "return_result({'n': len(text)})"
    )
    agent = _agent([_cell(code, "1", text=EDGE_CASES)])
    assert await agent.answer() == {"n": len(EDGE_CASES)}
    for name, content in EDGE_CASES.items():
        assert (target / name.replace(" ", "_")).read_bytes() == content.encode("utf-8")


async def test_a_call_without_text_leaves_the_namespace_alone():
    agent = _agent(
        [
            _cell("text = 'mine'", "1"),
            _cell("assert text == 'mine'\ntext = {'a': 'old'}", "2"),
            _cell("assert text == {'a': 'x'}\ntext['b'] = 'y'", "3", text={"a": "x"}),
            _cell("return_result(dict(text))", "4"),
        ]
    )
    assert await agent.answer() == {"a": "x", "b": "y"}
    assert not any(o.error for o in _outputs(agent))


async def test_null_and_empty_text_bind_nothing():
    agent = _agent(
        [
            _cell("text = 'mine'", "1"),
            LLMResponse(
                parts=(
                    ToolCall(
                        id="2",
                        name="python_cell",
                        arguments=json.dumps({"code": "assert text == 'mine'", "text": None}),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _cell("return_result({'t': text})", "3", text={}),
        ]
    )
    assert await agent.answer() == {"t": "mine"}


async def test_text_overwrites_a_variable_of_that_name():
    agent = _agent(
        [_cell("text = 5", "1"), _cell("return_result(dict(text))", "2", text={"k": "v"})]
    )
    assert await agent.answer() == {"k": "v"}


def _flushed_metrics(monkeypatch) -> list[HarnessMetrics]:
    flushed: list[HarnessMetrics] = []
    monkeypatch.setattr(HarnessMetrics, "flush_to_span", lambda self: flushed.append(self))
    return flushed


async def test_a_bare_string_is_accepted_as_text_text_with_a_note(monkeypatch):
    flushed = _flushed_metrics(monkeypatch)
    agent = _agent(
        [
            _cell("print('ran')", "1", text='raw "s"\\'),
            _cell("return_result(dict(text))", "2"),
        ]
    )
    assert await agent.answer() == {"text": 'raw "s"\\'}
    [hm] = flushed
    first = _outputs(agent)[0]
    assert first.stdout.startswith("Note: `text` arrived as a bare string")
    assert first.stdout.count("bare string") == 1
    assert "ran" in first.stdout
    assert hm.cell_text_arg_bare_string_count == 1
    assert hm.cell_text_arg_count == 1


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"a": 1}, "`text['a']` must be a string; got int"),
        ({"": "x"}, "every `text` key must be a non-empty string"),
        (["a"], "`text` must be an object mapping names to strings"),
        ({"a": None}, "`text['a']` must be a string; got NoneType"),
    ],
)
async def test_invalid_text_is_a_clear_error_and_nothing_runs(bad, fragment):
    agent = _agent(
        [
            _cell("self.mark()", "1", text=bad),
            _cell("return_result({'ran': self.ran})", "2"),
        ]
    )
    assert await agent.answer() == {"ran": 0}
    error = _outputs(agent)[0].error
    assert fragment in error and "Nothing ran." in error


async def test_text_without_code_says_code_is_needed():
    agent = _agent(
        [
            LLMResponse(
                parts=(
                    ToolCall(
                        id="1", name="python_cell", arguments=json.dumps({"text": {"a": "b"}})
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _cell("return_result({})", "2"),
        ]
    )
    assert await agent.answer() == {}
    error = _outputs(agent)[0].error
    assert "empty code provided" in error and "put the Python that uses it in `code`" in error


async def test_codeact_v1_ignores_a_text_argument():
    llm_call = LLMResponse(
        parts=(
            ToolCall(
                id="1",
                name="execute_python",
                arguments=json.dumps(
                    {"code": "return_result({'has': 'text' in dir()})", "text": {"a": "b"}}
                ),
            ),
        ),
        finish_reason="tool_calls",
    )
    agent = _agent([llm_call], strategy_type=CodeActStrategy)
    assert await agent.answer() == {"has": False}


async def test_metrics_count_entries_and_chars(monkeypatch):
    flushed = _flushed_metrics(monkeypatch)
    agent = _agent(
        [
            _cell("pass", "1", text={"a": "abc", "b": "de"}),
            _cell("return_result({})", "2", text={"c": "f"}),
        ]
    )
    await agent.answer()
    [hm] = flushed
    assert (hm.cell_text_arg_count, hm.cell_text_arg_entries, hm.cell_text_arg_chars) == (2, 3, 6)
    attrs = hm.to_span_attributes()
    assert attrs["harness.cell_text_arg.count"] == 2
    assert attrs["harness.cell_text_arg.entries"] == 3
    assert attrs["harness.cell_text_arg.chars"] == 6


def test_span_schema_lists_the_cell_text_metrics():
    keys = {entry.key for entry in get_span_schema()}
    assert {
        "harness.cell_text_arg.count",
        "harness.cell_text_arg.entries",
        "harness.cell_text_arg.chars",
        "harness.cell_text_arg.bare_string.count",
    } <= keys
    hm = HarnessMetrics()
    hm.cell_text_arg(2, 10, bare_string=True)
    assert hm.to_span_attributes()["harness.cell_text_arg.bare_string.count"] == 1


def test_loop_guard_distinguishes_calls_that_differ_only_in_text():
    a = action_fingerprint("python_cell", {"code": "x", "text": {"f": "1"}}, "python_cell")
    b = action_fingerprint("python_cell", {"code": "x", "text": {"f": "2"}}, "python_cell")
    c = action_fingerprint("python_cell", {"code": "x  # same", "text": {"f": "1"}}, "python_cell")
    none = action_fingerprint("python_cell", {"code": "x"}, "python_cell")
    assert a != b and a != none
    assert a == c  # code is still normalized


def test_parse_cell_text_keeps_values_exactly():
    parsed = parse_cell_text({"k": BIG})
    assert parsed.value == {"k": BIG} and parsed.chars == len(BIG) and parsed.error is None
