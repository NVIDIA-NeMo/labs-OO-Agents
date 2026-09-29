# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playground continuation declares the Python tool used in its history."""

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["execute_python", "python_cell"])
async def test_playground_declares_historical_python_tool(monkeypatch, tool_name):
    import litellm

    from nooa.viewer import trace_routes

    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return litellm.ModelResponse(
            model="model", choices=[{"message": {"role": "assistant", "content": "ok"}}]
        )

    monkeypatch.setattr(litellm, "acompletion", completion)
    monkeypatch.setattr(trace_routes, "get_model_config", lambda model: None)
    result = await trace_routes.run_inference(
        trace_routes.InferenceRequest(
            model="openai/model",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call", "name": tool_name, "arguments": {"code": "1 + 1"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call", "content": "2"},
            ],
        )
    )
    assert result["status"] == "success"
    tools = {tool["function"]["name"]: tool["function"] for tool in captured["tools"]}
    assert tools[tool_name]["parameters"]["required"] == ["code"]
    assert "python_cell" not in {
        tool["function"]["name"] for tool in trace_routes.DEFAULT_SANDBOX_TOOLS
    }
