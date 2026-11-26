# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF normalization must use sanitized values and explicit provider IDs."""

import json
from typing import Any, cast

import pytest

from nooa._relay_atif import _normalize_atif, export_nemo_relay_atif
from nooa._relay_telemetry import inspection_snapshot, inspection_value


def _step(identifier, content: Any = "answer"):
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


@pytest.mark.parametrize("chat_completion", [False, True])
def test_unique_provider_id_repairs_concurrent_ownership_and_caps_results(chat_completion):
    first, second = _step("first"), _step("second")
    if chat_completion:
        for step in (first, second):
            response = step["extra"]["llm_response"]
            response["choices"] = [{"message": response.pop("message")}]
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
    response = first["extra"]["llm_response"]
    message = response["choices"][0]["message"] if chat_completion else response["message"]
    assert "reasoning_content" not in message
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


def test_optional_null_fields_and_unidentified_calls_are_preserved():
    step = {
        "extra": {"llm_response": None, "tool_ancestry": None},
        "tool_calls": [{"function_name": "first"}, {"function_name": "second"}],
        "observation": None,
    }
    _normalize_atif({"steps": [step]})
    assert [call["function_name"] for call in step["tool_calls"]] == ["first", "second"]
    assert "observation" not in step
    _normalize_atif({"steps": [{"tool_calls": None, "observation": {"results": None}}]})


def test_snapshot_inspection_keeps_small_legacy_shapes_and_caller_objects():
    original = {"message": {"content": "safe", "role": "assistant"}, "usage": {"prompt_tokens": 3}}
    data, metadata = inspection_snapshot(original)
    assert data == original
    assert data is not original and data["message"] is not original["message"]
    assert metadata["nooa.output.preview.incomplete"] is False
    data["message"]["content"] = "changed"
    assert original["message"]["content"] == "safe"


def test_snapshot_inspection_bounds_before_encoding_and_never_runs_value_hooks():
    class Dangerous:
        def __getattribute__(self, name):
            raise AssertionError("No attribute hooks")

        def __repr__(self):
            raise AssertionError("No repr")

        def __str__(self):
            raise AssertionError("No str")

        def __reduce_ex__(self, protocol):
            raise AssertionError("No pickle")

    original = {"large": "x" * 5_000_000, "dangerous": Dangerous()}
    data, metadata = inspection_snapshot(original)
    assert len(json.dumps(data)) <= 50_000
    assert metadata["nooa.output.preview.incomplete"] is True
    assert len(original["large"]) == 5_000_000
    opaque, _ = inspection_snapshot(Dangerous())
    assert "opaque" in opaque


def test_snapshot_inspection_supports_structured_values_and_native_json_limits():
    from collections import deque
    from dataclasses import dataclass, field
    from datetime import datetime

    @dataclass
    class Result:
        values: object
        secret: str = field(default="DO NOT EXPORT", repr=False)

    original = Result(
        {
            "items": deque([1, 2]),
            "time": datetime(2026, 1, 1),
            "large": 2**64,
            "surrogate": "\ud800",
        }
    )
    data = inspection_value(original)
    assert "DO NOT EXPORT" not in json.dumps(data)
    assert data["values"]["items"] == [1, 2]
    assert data["values"]["large"] == str(2**64)
    assert data["values"]["surrogate"] == "\\ud800"
    assert data["values"]["time"]["iso"] == "2026-01-01T00:00:00"


def test_explicit_null_result_is_empty_and_preview_metadata_reports_cuts():
    step = {
        "observation": {
            "results": [
                {"extra": {"tool_result": None}},
                {"extra": {"tool_result": list(range(100_000))}},
            ]
        }
    }
    _normalize_atif({"steps": [step]})
    empty, large = step["observation"]["results"]
    assert empty["content"] == ""
    assert large["extra"]["nooa.output.preview.incomplete"] is True
    assert large["extra"]["nooa.output.preview.paths"]
    data, metadata = inspection_snapshot(10**100_000)
    assert len(json.dumps(data)) < 50_000
    assert metadata["nooa.output.preview.incomplete"] is True


def test_inspection_failure_does_not_replace_the_caller_result(monkeypatch):
    import nooa._relay_telemetry as telemetry

    def failed_capture(*args, **kwargs):
        raise RuntimeError("inspection failed")

    monkeypatch.setattr(telemetry, "capture", failed_capture)
    data, metadata = telemetry.inspection_snapshot(object())
    assert data is not None
    assert metadata["nooa.output.preview.incomplete"] is True
    assert "error" in metadata["nooa.output.preview.reasons"]


def test_export_keeps_original_incomplete_report_when_display_fits():
    data, metadata = inspection_snapshot(10**100_000)
    original_metadata = dict(metadata)
    step = {
        "observation": {
            "results": [
                {
                    "extra": {
                        "tool_result": data,
                        "tool_result_annotation": {"is_error": False, **metadata},
                    }
                }
            ]
        }
    }
    _normalize_atif({"steps": [step]})
    extra = step["observation"]["results"][0]["extra"]
    assert extra["nooa.output.preview.incomplete"] is True
    assert extra["nooa.output.preview.paths"] == original_metadata["nooa.output.preview.paths"]
    assert extra["nooa.output.preview.reasons"] == original_metadata["nooa.output.preview.reasons"]
    assert extra["display_preview"]["nooa.output.preview.incomplete"] is False
    assert extra["tool_result_annotation"]["nooa.output.preview.incomplete"] is True
    assert extra["is_error"] is False


@pytest.mark.asyncio
async def test_real_relay_export_preserves_late_redaction():
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
                    and isinstance(data.get("choices"), list)
                ):
                    data["choices"][0]["message"]["content"] = "REDACTED"
                return fields

            async def model(ctx):
                # Registration during execution reproduces the old priority tie.
                relay.scope_local.register_scope_sanitize_end(
                    root, "late-redactor", 2**31 - 1, redact
                )
                ctx.response = cast(Any, LLMResponse)(
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
