# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The code_execution span records python_cell's raw ``text`` argument."""

import json

from openinference.semconv.trace import SpanAttributes

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.tracing.test_openinference_conformance import framework_spans  # noqa: F401

TEXTS = {"app.py": 'print("hi")\r\n\'\'\' """ \\ 🤖', "big": "y" * 12_000}


def _cell(args: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        parts=(ToolCall(id=call_id, name="python_cell", arguments=json.dumps(args)),),
        finish_reason="tool_calls",
    )


async def test_code_execution_input_contains_the_cell_text(framework_spans):  # noqa: F811
    llm = FakeLLMClient(
        scripted_responses=[
            _cell({"code": "n = len(text)", "text": TEXTS}, "1"),
            _cell({"code": "return_result(n)"}, "2"),
        ]
    )

    class TextAgent(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def run(self) -> int:
            """Count."""
            ...

    assert await TextAgent().run() == 2
    spans = [s for s in framework_spans.get_finished_spans() if s.name == "code_execution"]
    # The global provider may carry processors from earlier tests: one span per execution.
    by_execution = {s.attributes["execution.id"]: s for s in spans}
    with_text, without = list(by_execution.values())
    recorded = json.loads(with_text.attributes[SpanAttributes.INPUT_VALUE])
    assert recorded["code"] == "n = len(text)"
    assert recorded["text"]["app.py"] == TEXTS["app.py"]
    assert recorded["text"]["big"] == "y" * 10_000  # capped like code
    assert with_text.attributes["code.text.count"] == 2
    assert with_text.attributes["code.text.chars"] == sum(len(v) for v in TEXTS.values())
    assert "text" not in json.loads(without.attributes[SpanAttributes.INPUT_VALUE])
    assert "code.text.count" not in without.attributes
