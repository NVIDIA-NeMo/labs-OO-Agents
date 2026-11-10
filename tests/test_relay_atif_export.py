# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF normalization must use sanitized values and explicit provider IDs."""

import json

import pytest

from nooa._relay_atif import _normalize_atif, export_nemo_relay_atif


def _step(identifier, content="answer"):
    return {
        "source": "agent",
        "message": "legacy JSON",
        "extra": {
            "llm_response": {
                "message": {
                    "content": content,
                    "reasoning_content": "redacted reasoning",
                    "tool_calls": [
                        {
                            "id": identifier,
                            "function": {
                                "name": "execute_python",
                                "arguments": '{"code":"print(1)"}',
                            },
                        }
                    ],
                }
            }
        },
    }


def test_unique_provider_id_repairs_concurrent_ownership_and_caps_results():
    first, second = _step("first"), _step("second")
    second["tool_calls"] = [{"tool_call_id": "first", "function_name": "execute_python"}]
    second["observation"] = {
        "results": [
            {
                "source_call_id": "first",
                "extra": {
                    "tool_result": {"text": "x" * 5000},
                    "event_uuid": "event-1",
                    "tool_result_annotation": {"is_error": True},
                },
            }
        ]
    }
    second["extra"]["tool_invocations"] = [{"invocation_id": "first"}]
    second["extra"]["tool_ancestry"] = [{"function_id": "event-1"}]
    result = _normalize_atif({"steps": [first, second]})
    assert result["steps"][0]["message"] == "answer"
    assert first["reasoning_content"] == "redacted reasoning"
    assert "reasoning_content" not in first["extra"]["llm_response"]["message"]
    observation = first["observation"]["results"][0]
    assert observation["source_call_id"] == "first"
    assert len(observation["content"]) < 4200
    assert "tool_result" not in observation["extra"]
    assert observation["extra"]["is_error"] is True
    assert first["extra"]["tool_invocations"] == [{"invocation_id": "first"}]
    assert first["extra"]["tool_ancestry"] == [{"function_id": "event-1"}]
    assert "tool_invocations" not in second["extra"]
    assert "tool_ancestry" not in second["extra"]
    assert "observation" not in second
    assert second["tool_calls"][0]["tool_call_id"] == "second"


def test_ambiguous_ids_stay_put_and_absent_results_are_not_reconstructed():
    first, second = _step("duplicate"), _step("duplicate")
    second["observation"] = {"results": [{"source_call_id": "duplicate", "content": "safe"}]}
    result = _normalize_atif({"steps": [first, second]})
    assert "observation" not in result["steps"][0]
    assert result["steps"][1]["observation"]["results"][0]["content"] == "safe"
    assert _normalize_atif({"steps": []}) == {"steps": []}


@pytest.mark.parametrize("arguments", ["{malformed", "[1]", "null", "123", '{"x":NaN}'])
def test_invalid_provider_arguments_remain_evidence_in_valid_atif_object(arguments):
    step = _step("bad-arguments")
    step["extra"]["llm_response"]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    _normalize_atif({"steps": [step]})
    call = step["tool_calls"][0]
    assert call["arguments"] == {}
    assert call["extra"]["raw_arguments"] == arguments


def test_nontext_content_and_removed_content_do_not_leave_stale_json():
    block = [{"type": "text", "text": "safe"}]
    first, second = _step("first", block), _step("second")
    del second["extra"]["llm_response"]["message"]["content"]
    _normalize_atif({"steps": [first, second]})
    assert json.loads(first["message"]) == block
    assert second["message"] == ""


@pytest.mark.asyncio
async def test_real_relay_export_preserves_late_legacy_redaction():
    relay = pytest.importorskip("nemo_relay")
    from nooa import Agent
    from nooa.events import ExecutionResult
    from nooa.llm_types import LLMResponse, ToolCall
    from nooa.nemo_relay_middleware import (
        nemo_relay_llm_middleware,
        nemo_relay_scope,
        nemo_relay_tool_middleware,
    )
    from nooa.runtime.middleware import ExecutePythonContext, LLMCallContext
    from nooa.unifiedllm import FakeLLMClient

    class AuditAgent(Agent, llm=FakeLLMClient()):
        pass

    exporter = relay.AtifExporter(
        session_id="safe-export", agent_name="audit", agent_version="test"
    )
    exporter.register("nooa-safe-export-test")
    try:
        async with nemo_relay_scope(AuditAgent(), "safe-export") as root:

            def redact(event, fields):
                data = fields.get("data")
                if (
                    event.category == "llm"
                    and isinstance(data, dict)
                    and isinstance(data.get("message"), dict)
                ):
                    data["message"]["content"] = "REDACTED"
                return fields

            async def model(ctx):
                # Registration during execution reproduces the old priority tie.
                relay.scope_local.register_scope_sanitize_end(
                    root, "late-redactor", 2**31 - 1, redact
                )
                ctx.response = LLMResponse(
                    content="SECRET",
                    tool_calls=[
                        ToolCall(
                            id="provider-1", name="execute_python", arguments='{"code":"pass"}'
                        )
                    ],
                )
                return ctx

            await nemo_relay_llm_middleware(LLMCallContext(messages=[]), model)

            async def tool(ctx):
                ctx.result = ExecutionResult(returned_value={"safe": "value"})
                return ctx

            await nemo_relay_tool_middleware(
                ExecutePythonContext(code="pass", params={"tool_call_id": "provider-1"}), tool
            )
        trajectory = await export_nemo_relay_atif(exporter)
        assert trajectory["schema_version"] == "ATIF-v1.7"
        assert "SECRET" not in json.dumps(trajectory)
        model_step = next(step for step in trajectory["steps"] if step.get("llm_call_count"))
        assert model_step["message"] == "REDACTED"
        assert model_step["tool_calls"][0]["tool_call_id"] == "provider-1"
        assert model_step["observation"]["results"][0]["content"] == '{"safe": "value"}'
    finally:
        await relay.subscribers.flush_async()
        exporter.deregister("nooa-safe-export-test")
