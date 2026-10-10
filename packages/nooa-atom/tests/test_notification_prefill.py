# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notification metadata must reach the executed CodeAct input prefill."""

import pytest
from nooa_atom.agent import AtomAgent
from test_experimental_agent import python_cell

from nooa import strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActV2
from nooa.unifiedllm import FakeLLMClient


class PlainInputAgent(AtomAgent):
    @strategy(CodeActV2(config=CodeActConfig()))
    async def inspect_plain(self, data: dict) -> None:
        """Inspect an ordinary input without notification-specific metadata."""
        ...


@pytest.mark.parametrize("method", ["handle", "handle_batch"])
async def test_notification_prefill_keeps_nested_task_middle(tmp_path, method):
    marker = "TASK_MIDDLE_MUST_BE_VISIBLE"
    text = "a" * 1500 + marker + "z" * 1500
    assert len(text) > 2731
    notification = {
        "user_messages": [text],
        # Exercise both collection length and depth beyond the default caps.
        "context": ["item"] * 15
        + [{"payload": {"items": [{"task": text, "leaf": "DEEP_LEAF"}]}}]
        + ["item"] * 15,
        "system_messages": ["housekeeping"],
    }
    llm = FakeLLMClient(
        [
            python_cell(
                "return_result(Done(explanation='finished', result=TaskResult("
                "solution_description='a', evidence='b', how_to_verify='c')))",
                "done",
            )
        ],
        strict_exhaustion=True,
    )
    agent = AtomAgent(llm=llm, cwd=tmp_path)
    try:
        await getattr(agent, method)(notification)
    finally:
        await agent.aclose()

    # These are the actual generated code and captured PythonOutput presented
    # after prefill execution, not merely the annotation or a pformat unit test.
    rendered = "\n".join(str(m) for m in llm.calls[0].messages)
    assert "pprint(notification, max_length=100, max_string=100000, max_depth=10)" in rendered
    output = next(
        str(m["content"])
        for m in llm.calls[0].messages
        if m["role"] == "user" and "PythonOutput(" in str(m.get("content", ""))
    )
    assert output.count(text) == 2
    assert "DEEP_LEAF" in output and "housekeeping" in output
    assert "str(len=" not in output


async def test_ordinary_input_prefill_still_truncates(tmp_path):
    marker = "PLAIN_MIDDLE_MUST_BE_TRUNCATED"
    text = "a" * 1500 + marker + "z" * 1500
    llm = FakeLLMClient([python_cell("return_result(None)", "done")], strict_exhaustion=True)
    agent = PlainInputAgent(llm=llm, cwd=tmp_path)
    try:
        await agent.inspect_plain({"items": [text]})
    finally:
        await agent.aclose()

    rendered = "\n".join(str(m) for m in llm.calls[0].messages)
    assert "pprint(data, max_length=25, max_string=2000, max_depth=4)" in rendered
    assert marker not in rendered
    assert f"str(len={len(text)}, [:1000]=" in rendered
    assert "[-1000:]=" in rendered
