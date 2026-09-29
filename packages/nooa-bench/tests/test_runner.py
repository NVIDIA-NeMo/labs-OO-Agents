# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle tests for the benchmark runner."""

import asyncio
from types import SimpleNamespace

import pytest
from nooa_bench import runner


@pytest.mark.asyncio
async def test_circular_debug_value_does_not_prevent_success_or_verifier_answer(
    monkeypatch, tmp_path
):
    from nooa.events import PythonOutput
    from nooa.runtime.event_manager import EventManager
    from nooa.unifiedllm import FakeLLMClient

    circular = {}
    circular["self"] = circular
    answers = []

    class FinishedAgent:
        def __init__(self, llm):
            self.event_manager = EventManager()
            self.event_manager.add(
                PythonOutput(
                    tool_call_id="c", execution_count=1, execution_status="complete", value=circular
                )
            )

        async def _run_evaluation(self, task_input):
            return {"success": True, "response": "verification-command"}

    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_import_agent_class", lambda _: FinishedAgent)
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: FakeLLMClient())
    monkeypatch.setattr(runner, "_write_answer", lambda result: answers.append(result["response"]))
    (tmp_path / "trajectory.json").write_text("[]")
    (tmp_path / "behavior.json").write_text('{"task_id": "previous-task"}')
    assert await runner._run("task", "model", "bench", None) == 0
    assert answers == ["verification-command"]
    assert (tmp_path / "result.json").exists()
    assert not (tmp_path / "trajectory.json").exists()
    assert not (tmp_path / "behavior.json").exists()


@pytest.mark.parametrize("agent_async", [False, True])
@pytest.mark.parametrize("client_async", [False, True])
@pytest.mark.parametrize(
    "outcome", ["success", "failure", "execution_error", "write_error", "cancelled"]
)
async def test_cleanup_failures_preserve_the_original_outcome(
    monkeypatch, caplog, agent_async, client_async, outcome
):
    calls = []
    original_error = (
        asyncio.CancelledError("benchmark cancelled")
        if outcome == "cancelled"
        else OSError("original benchmark failure")
    )

    def fail_close(label, asynchronous):
        def fail():
            calls.append(label)
            raise RuntimeError(f"{label} cleanup broke")

        async def async_fail():
            fail()

        return async_fail if asynchronous else fail

    client = SimpleNamespace(aclose=fail_close("llm", client_async))

    class FakeAgent:
        def __init__(self, llm):
            self.close = fail_close("agent", agent_async)

        async def _run_evaluation(self, task_input):
            if outcome in {"execution_error", "cancelled"}:
                raise original_error
            return {"success": outcome != "failure", "response": "done"}

    def write_result(*args):
        if outcome == "write_error":
            raise original_error

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, "_import_agent_class", lambda name: FakeAgent)
    monkeypatch.setattr(runner, "_write_result", write_result)
    for name in ("_write_trajectory", "_write_behavior_report", "_write_answer"):
        monkeypatch.setattr(runner, name, lambda *args: None)

    if outcome.endswith("error") or outcome == "cancelled":
        with pytest.raises(type(original_error)) as caught:
            await runner._run("task", "model", "bench", None)
        assert caught.value is original_error
    else:
        assert await runner._run("task", "model", "bench", None) == (outcome == "failure")
    assert calls == ["agent", "llm"]
    assert "Agent cleanup failed" in caplog.text
    assert "Model client cleanup failed" in caplog.text


@pytest.mark.asyncio
async def test_run_closes_agent_and_llm_when_result_writing_fails(monkeypatch):
    calls: list[str] = []

    class FakeLLM:
        async def aclose(self):
            calls.append("llm")

    class FakeAgent:
        def __init__(self, llm):
            self.llm = llm
            self.event_manager = SimpleNamespace(items=lambda: [])

        async def _run_evaluation(self, task_input):
            return {"success": True, "response": "done"}

        async def close(self):
            calls.append("agent")

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(runner, "_import_agent_class", lambda name: FakeAgent)
    monkeypatch.setattr(
        runner, "_write_result", lambda *args: (_ for _ in ()).throw(OSError("disk"))
    )

    with pytest.raises(OSError, match="disk"):
        await runner._run("task", "model", "bench", None)

    assert calls == ["agent", "llm"]


@pytest.mark.asyncio
async def test_run_closes_llm_when_agent_construction_fails(monkeypatch):
    calls: list[str] = []

    class FakeLLM:
        async def aclose(self):
            calls.append("llm")

    class BrokenAgent:
        def __init__(self, llm):
            raise RuntimeError("constructor failed")

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(runner, "_import_agent_class", lambda name: BrokenAgent)

    with pytest.raises(RuntimeError, match="constructor failed"):
        await runner._run("task", "model", "bench", None)

    assert calls == ["llm"]


@pytest.mark.parametrize("agent_type", ["bench", "rlm"])
async def test_runner_executes_delegation_and_preserves_provider_turns(
    monkeypatch, tmp_path, agent_type
):
    import json

    from nooa.unifiedllm import (
        AssistantReasoning,
        CacheBoundary,
        FakeLLMClient,
        LLMResponse,
        ToolCall,
    )

    class RecordingLLM(FakeLLMClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.requests = []
            self.close_count = 0

        async def acall(self, messages, **kwargs):
            self.requests.append(list(messages))
            return await super().acall(messages, **kwargs)

        async def aclose(self):
            self.close_count += 1
            await super().aclose()

    def response(code, call_id, *, native=False):
        return LLMResponse(
            parts=(
                AssistantReasoning(
                    text="Inspect and verify.",
                    native={"signature": "fixture-provider-state"} if native else None,
                ),
                ToolCall(id=call_id, name="python_cell", arguments=json.dumps({"code": code})),
            ),
            replay_scope="fixture-issuer" if native else None,
            finish_reason="tool_calls",
        )

    first = response(
        "task = self.todo.add('Verify workspace')\n"
        "report = await self.delegate(task, supplied_context={'api_key': 'private-sentinel'})",
        "parent-start",
        native=True,
    )
    llm = RecordingLLM(
        scripted_responses=[
            first,
            response(
                f"assert str(self.shell.cwd) == {str(tmp_path)!r}\n"
                "checked = await self.shell.run('printf verified')\n"
                "assert checked.returncode == 0 and checked.stdout == 'verified'\n"
                "task = self.todo.list_todos()[0]\n"
                "self.todo.comment(task, 'Observed verification')\n"
                "self.todo.set_var(task, 'checked', True)\n"
                "return_result(TaskResult(solution_description='Verified workspace', "
                "evidence=checked.stdout, how_to_verify='true'))",
                "worker-check",
            ),
            response(
                "assert task.v.checked\n"
                "assert task.comments[0].body == 'Observed verification'\n"
                "return_result(report)",
                "parent-finish",
            ),
        ]
    )
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: llm)
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(runner, "ANSWER_FILE", tmp_path / "answer.txt")
    assert (
        await runner._run(
            "Verify the workspace",
            "fixture-model",
            agent_type,
            api_base=None,
            working_dir=str(tmp_path),
        )
        == 0
    )
    assert llm.call_count == 3
    assert llm.close_count == 1  # Worker cleanup must not close the shared client.
    assert (tmp_path / "answer.txt").read_text() == "true"
    result = json.loads((tmp_path / "logs/result.json").read_text())
    assert result["success"] is True
    metrics = json.loads((tmp_path / "logs/behavior.json").read_text())
    # Only controller model cells are in this trajectory; prefill is excluded.
    # Worker cells live in their separate event managers.
    assert metrics["signals"]["python_cells"] == 2
    assert metrics["signals"]["delegations"] == 1
    assert metrics["rates"]["completion_rate"] == 1.0

    # The parent's history retains the immutable provider turn across delegation.
    messages = llm.requests[-1]
    index = next(
        i for i, m in enumerate(messages) if isinstance(m, LLMResponse) and m.id == first.id
    )
    assert messages[index].parts == first.parts
    assert messages[index].replay_scope == first.replay_scope
    paired = next(i for i, m in enumerate(messages) if m.get("tool_call_id") == "parent-start")
    boundary = next(i for i, m in enumerate(messages) if isinstance(m, CacheBoundary))
    assert index < paired < boundary
    assert not any(
        isinstance(message, LLMResponse)
        and any(isinstance(part, ToolCall) and part.id == "worker-check" for part in message.parts)
        for message in messages
    )
    assert not any(
        isinstance(message, LLMResponse) and message.id == first.id for message in llm.requests[1]
    )
    assert "private-sentinel" in str(llm.requests[1])  # Ordinary input, no custom redaction.
    assert "fixture-provider-state" not in (tmp_path / "logs/trajectory.json").read_text()
