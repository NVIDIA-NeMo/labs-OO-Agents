# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the experimental named-snippet CodeAct V3 strategy."""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.strategies import CodeActV2, CodeActV3
from nooa.strategies.codeact_v3 import _TURN_SNIPPETS, parse_snippets, resolve_snippet
from nooa.unifiedllm import AssistantText, FakeLLMClient, LLMResponse, ToolCall

FENCE = "```"


def _block(name: str, code: str, header: str | None = None) -> str:
    return f"{FENCE}{header or f'python {name}'}\n{code}\n{FENCE}"


def _reply(text: str, *names: str, call_prefix: str = "call") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(
                id=f"{call_prefix}_{i}", name="run_snippet", arguments=json.dumps({"name": name})
            )
            for i, name in enumerate(names)
        ],
        finish_reason="tool_calls",
    )


def _agent(llm: FakeLLMClient, return_type: type = int, **config):
    config.setdefault("prefill", None)

    class TestAgent(Agent, llm=llm):
        @strategy(CodeActV3(config=CodeActConfig(**config)))
        async def answer(self) -> return_type:  # type: ignore[valid-type]
            """Compute the answer."""
            ...

    return TestAgent()


def _outputs(agent) -> list:
    return list(agent.events.query(type="PythonOutput"))


@pytest.fixture
def metrics(monkeypatch) -> HarnessMetrics:
    collected = HarnessMetrics()
    monkeypatch.setattr("nooa.strategies.codeact_v3.get_harness_metrics", lambda: collected)
    return collected


# ── Parser ──────────────────────────────────────────────────────────


def test_parser_reads_named_unnamed_and_ignores_other_languages():
    text = "\n\n".join(
        [
            "Plan first.",
            _block("scan-files", "x = 1\nprint(x)"),
            _block("", "print('example')", header="python"),
            _block("", "ls", header="bash list"),
        ]
    )
    snippets = parse_snippets(text)
    assert [(s.name, s.code, s.variant) for s in snippets] == [
        ("scan-files", "x = 1\nprint(x)", None),
        (None, "print('example')", None),
    ]


@pytest.mark.parametrize(
    "header,variant",
    [
        ('python title="scan-files"', "title"),
        ("python title='scan-files'", "title"),
        ("python [scan-files]", "bracket"),
        ("python:scan-files", "colon"),
        ("{.python #scan-files}", "pandoc"),
        ("{python scan-files, echo=FALSE}", "knitr"),
        ("py scan-files", None),
        ("Python3 scan-files", None),
    ],
)
def test_parser_accepts_common_naming_conventions(header, variant):
    (snippet,) = parse_snippets(_block("", "print(1)", header=header))
    assert snippet.name == "scan-files"
    assert snippet.variant == variant


def test_parser_follows_commonmark_fence_rules():
    text = (
        "````python outer\n"
        "s = '''\n```python inner\nnot a block\n```\n'''\n"
        "````\n"
        "~~~python tilde\nprint('t')\n~~~\n"
        "  ```python indented\n  if True:\n      pass\n  ```\n"
    )
    snippets = {s.name: s for s in parse_snippets(text)}
    assert set(snippets) == {"outer", "tilde", "indented"}
    assert "```python inner" in snippets["outer"].code
    assert snippets["tilde"].code == "print('t')"
    assert snippets["indented"].code == "if True:\n    pass"


def test_parser_marks_unclosed_and_invalid_names():
    unclosed, invalid = parse_snippets("```python bad!name\nx\n```\n```python open\nx = 1\n")[::-1]
    assert unclosed.name == "open" and not unclosed.closed
    assert invalid.name is None and invalid.invalid_name == "bad!name"
    assert "closing" in (resolve_snippet([unclosed], "open")[1] or "")


def test_resolution_errors_explain_how_to_recover():
    snippets = parse_snippets(
        _block("a", "1") + "\n" + _block("a", "2") + "\n" + _block("", "3", header="python")
    )
    assert "appears 2 times" in (resolve_snippet(snippets, "a")[1] or "")
    missing = resolve_snippet(snippets, "b")[1] or ""
    assert "`b` was not found" in missing
    assert "`a`" in missing
    assert "1 unnamed" in missing
    assert "earlier replies cannot be run" in missing
    assert "one string argument `name`" in (resolve_snippet(snippets, None)[1] or "")


# ── Strategy ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_named_snippets_run_in_one_session_and_finish():
    llm = FakeLLMClient(
        scripted_responses=[
            _reply(
                "Two steps.\n\n"
                + _block("setup", "base = 40")
                + "\n\nAn example I will not run:\n\n"
                + _block("", "raise RuntimeError('must not run')", header="python")
                + "\n\n"
                + _block("add", "total = base + 2\nprint(total)"),
                "setup",
                "add",
            ),
            _reply(_block("finish", "return_result(total)"), "finish", call_prefix="end"),
        ]
    )
    agent = _agent(llm)
    try:
        assert await agent.answer() == 42
        assert [tool.name for tool in llm.last_tools or []] == ["run_snippet"]
        calls = [e for e in agent.events.query(type="ToolCallEvent") if e.name == "run_snippet"]
        assert [c.arguments for c in calls] == [
            {"name": "setup"},
            {"name": "add"},
            {"name": "finish"},
        ]
        assert calls[1].metadata["snippet_code"] == "total = base + 2\nprint(total)"
        assert any("42" in o.stdout for o in _outputs(agent))
        assert not any("must not run" in (o.error or "") for o in _outputs(agent))
        # The provider history keeps the snippet text beside the name-only call.
        history = json.dumps(llm.last_messages, default=str)
        assert "python add" in history
        assert '\\"name\\": \\"add\\"' in history or '"name": "add"' in history
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_snippet_from_earlier_reply_is_not_runnable(metrics):
    llm = FakeLLMClient(
        scripted_responses=[
            _reply(_block("calc", "value = 7"), "calc"),
            _reply("Run it again.", "calc", call_prefix="again"),
            _reply(_block("finish", "return_result(value)"), "finish", call_prefix="end"),
        ]
    )
    agent = _agent(llm)
    try:
        assert await agent.answer() == 7
        errors = [o.error for o in _outputs(agent) if o.error]
        assert len(errors) == 1
        assert "`calc` was not found in this reply" in errors[0]
        assert metrics.snippet_errors and "calc" in metrics.snippet_errors[0]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_code_argument_is_rejected_instead_of_executed():
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                content=_block("x", "print('from text')"),
                tool_calls=[
                    ToolCall(
                        id="bad",
                        name="run_snippet",
                        arguments=json.dumps({"name": "x", "code": "print('from arg')"}),
                    )
                ],
                finish_reason="tool_calls",
            ),
            _reply(_block("finish", "return_result(1)"), "finish", call_prefix="end"),
        ]
    )
    agent = _agent(llm)
    try:
        assert await agent.answer() == 1
        outputs = _outputs(agent)
        assert not any("from" in o.stdout for o in outputs)
        assert any("takes only `name`" in (o.error or "") for o in outputs)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_text_only_reply_with_snippet_asks_model_to_run_it(metrics):
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(parts=(AssistantText(text=_block("finish", "return_result(3)")),)),
            _reply(_block("finish", "return_result(3)"), "finish"),
        ]
    )
    agent = _agent(llm)
    try:
        assert await agent.answer() == 3
        feedback = [
            str(m.get("content", ""))
            for m in llm.last_messages
            if "made no tool call" in str(m.get("content", ""))
        ]
        assert len(feedback) == 1
        assert 'run_snippet(name="finish")' in feedback[0]
        assert metrics.snippets_not_run_count == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_non_canonical_header_runs_and_is_measured(metrics):
    llm = FakeLLMClient(
        scripted_responses=[
            _reply(_block("", "return_result(5)", header='python title="finish"'), "finish")
        ]
    )
    agent = _agent(llm)
    try:
        assert await agent.answer() == 5
        assert metrics.snippet_header_variants == ['title: python title="finish"']
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_provider_return_result_gets_snippet_guidance():
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                tool_calls=[ToolCall(id="r", name="return_result", arguments='{"result": 1}')],
                finish_reason="tool_calls",
            ),
            _reply(_block("finish", "return_result(1)"), "finish"),
        ]
    )
    agent = _agent(llm)
    try:
        assert await agent.answer() == 1
        result = next(
            e.result.content
            for e in agent.events.query(type="ToolCallEvent")
            if e.name == "return_result" and e.result
        )
        assert "Available tools: run_snippet" in result
        assert 'run_snippet(name="finish")' in result
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_prefill_is_recorded_as_a_named_run():
    llm = FakeLLMClient(scripted_responses=[_reply(_block("finish", "return_result(2)"), "finish")])

    class TestAgent(Agent, llm=llm):
        @strategy(CodeActV3())
        async def answer(self, value: int) -> int:
            """Return the value."""
            ...

    agent = TestAgent()
    try:
        assert await agent.answer(2) == 2
        prefill = next(
            e for e in agent.events.query(type="ToolCallEvent") if e.metadata.get("prefill")
        )
        assert prefill.name == "run_snippet"
        assert prefill.arguments == {"name": "inspect-inputs"}
    finally:
        await agent.aclose()


def test_loop_guard_fingerprints_snippet_code_not_name():
    strategy_ = CodeActV3()
    call = ToolCall(id="c", name="run_snippet", arguments='{"name": "step"}')
    fingerprints = []
    for code in ("x = 1", "x = 2", "x = 1"):
        token = _TURN_SNIPPETS.set(parse_snippets(_block("step", code)))
        try:
            fingerprints.append(strategy_._loop_guard_action(call)[0])
        finally:
            _TURN_SNIPPETS.reset(token)
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[0] == fingerprints[2]


def test_tool_contract_and_context_describe_snippets():
    strategy_ = CodeActV3()
    (tool,) = strategy_._build_tools(int, "answer")
    assert tool.name == "run_snippet"
    assert "```python scan-files" in tool.description
    assert "python_cell(" not in tool.description
    stub = strategy_._format_execution_context_stub(["x = 1"], [])
    assert "inside every snippet" in stub
    assert strategy_.name == "CODEACT_V3"


def test_v2_contract_is_unchanged():
    strategy_ = CodeActV2()
    (tool,) = strategy_._build_tools(int, "answer")
    assert tool.name == "python_cell"
    assert "inside python_cell()" in strategy_._format_execution_context_stub([], [])
    assert strategy_._prefill_tool_arguments("pprint(x)", "inspect_inputs") == {"code": "pprint(x)"}
