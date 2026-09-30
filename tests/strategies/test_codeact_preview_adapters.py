# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct execution registers the pformat preview adapters.

Regression test for the adapter-registration gap: register_all() used to run
only on the opaque-return-type path (_render_return_type_doc), so a method
with a JSON-schemable return type (-> str) never activated the numpy/pandas
previews and argument inspection fell back to truncated reprs. Found live —
the first end-to-end run showed `ndarray(repr_len=...)` in the prefill
PythonOutput instead of the structural preview.
"""

import json

import pytest

pd = pytest.importorskip("pandas")

from nooa import Agent, strategy  # noqa: E402
from nooa.strategies.codeact import CodeActStrategy  # noqa: E402
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall  # noqa: E402


@pytest.fixture(autouse=True)
def _register():
    # Reload to re-run the @spec.define_preview decorators even if a sibling
    # agentdoc test cleared the registry (register_all() alone is a no-op once
    # the module is imported).
    import importlib

    import nooa.agentdoc.adapters.pandas as _pandas_adapter

    importlib.reload(_pandas_adapter)


def _return_result(result, call_id: str = "call_return") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _resp(tool_calls: list) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


@pytest.mark.asyncio
async def test_prefill_inspection_uses_structural_preview_for_dataframe():
    """A schemable return type (-> str) must still activate preview adapters."""

    class FrameAgent(Agent, llm=FakeLLMClient()):
        @strategy(CodeActStrategy())
        async def summarize(self, table: pd.DataFrame) -> str:
            """Summarize the table."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp([_return_result("ok")])])
    agent = FrameAgent(llm=fake_llm)

    big = pd.DataFrame({"a": range(500), "b": [str(i) for i in range(500)]})
    result = await agent.summarize(big)
    assert result == "ok"

    outputs = agent.events.query(type="PythonOutput")
    assert outputs, "expected the prefill input-inspection PythonOutput event"
    stdout = getattr(outputs[0], "stdout", "") or ""
    assert "DataFrame(shape=(500, 2)" in stdout
    # the old fallback marker must be gone
    assert "repr_len=" not in stdout
