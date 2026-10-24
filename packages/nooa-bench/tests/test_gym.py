# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa_bench.gym: the thin NeMo Gym harness pieces, without Gym."""

from __future__ import annotations

import asyncio
import json
import socket
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nooa_bench.gym import (
    GymEpisode,
    ModelCallRecord,
    OutputTokenLimitExceeded,
    PolicyCallBudgetExceeded,
    ShellAgent,
    ToolCallRecord,
    ToolLoopStrategy,
    _classify,
    _episode_items,
    _RecordingClient,
    resource_tools,
    run_gym_episode,
)

from nooa import Agent, strategy
from nooa.errors import IterationLimitExceeded, LoopDetectedError, RetryLimitExceeded
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from nooa.unifiedllm.cache_policy import CacheBoundary

TRAIN = {"prompt_token_ids": [1, 2], "generation_token_ids": [3], "generation_log_probs": [-0.5]}


# --- ToolLoopStrategy -------------------------------------------------------------------------


class Counter(Agent):
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.calls: list[str] = []

    async def shell(self, command: str) -> str:
        """Run a command."""
        self.calls.append(command)
        if command == "boom":
            raise RuntimeError("exit 7")
        return f"ran {command}"

    @strategy(ToolLoopStrategy(("shell",), max_iterations=3))
    async def solve(self, task: str) -> str:
        """Do the task."""
        ...


def _tool_turn(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"call_{uuid4().hex}", name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
    )


async def test_tool_loop_executes_tools_then_returns_text():
    llm = FakeLLMClient(
        [
            _tool_turn("shell", command="ls"),
            _tool_turn("shell", command="boom"),
            LLMResponse(content="done"),
        ]
    )
    agent = Counter(llm=llm)
    assert await agent.solve("task") == "done"
    assert agent.calls == ["ls", "boom"]
    tool_events = list(agent.events.query(type="ToolCallEvent"))
    assert [e.result.content for e in tool_events] == ["ran ls", "exit 7"]
    assert [str(e.result.result_status) for e in tool_events] == ["complete", "error"]


async def test_tool_loop_stops_at_iteration_limit():
    llm = FakeLLMClient([_tool_turn("shell", command="ls")] * 4)
    with pytest.raises(IterationLimitExceeded):
        await Counter(llm=llm).solve("task")


async def test_unknown_tool_is_reported_to_the_model_not_raised():
    llm = FakeLLMClient([_tool_turn("rm_rf"), LLMResponse(content="ok")])
    agent = Counter(llm=llm)
    assert await agent.solve("task") == "ok"
    event = agent.events.query(type="ToolCallEvent")[0]
    assert "unknown tool" in event.result.content


# --- stop reasons -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (PolicyCallBudgetExceeded("x"), "policy_budget_exceeded"),
        (OutputTokenLimitExceeded("x"), "max_output_tokens_exceeded"),
        (IterationLimitExceeded("x"), "max_iterations_exceeded"),
        (LoopDetectedError("x"), "loop_detected"),
        (RetryLimitExceeded("x"), "retry_limit_exceeded"),
        (ValueError("x"), "error"),
    ],
)
def test_classify_direct(error, reason):
    assert _classify(error)[0] == reason


def test_classify_walks_cause_chain():
    try:
        try:
            raise IterationLimitExceeded("inner")
        except IterationLimitExceeded as inner:
            raise RuntimeError("turn failed") from inner
    except RuntimeError as outer:
        assert _classify(outer) == ("max_iterations_exceeded", "inner")


# --- episode projection -----------------------------------------------------------------------


def _record(request_input, output, status=200):
    return ModelCallRecord(
        invocation_id="inv",
        request={"input": request_input},
        response={"id": "resp", "output": output} if status == 200 else {"error": {}},
        status_code=status,
    )


def test_episode_items_keep_typed_items_and_full_conversation():
    history = [
        {"role": "user", "content": "task"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}", **TRAIN},
        {"type": "function_call_output", "call_id": "c1", "output": "ran"},
        {"role": "user", "content": "<context/>"},
    ]
    final = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
            **TRAIN,
        }
    ]
    output, conversation = _episode_items(
        [_record([{"role": "user", "content": "task"}], [], 500), _record(history, final)]
    )
    assert [i.get("type") for i in output] == ["function_call", "function_call_output", "message"]
    assert conversation == [*history, *final]
    assert output[0]["generation_token_ids"] == [3]


def test_episode_items_without_a_finished_response():
    assert _episode_items([_record([{"role": "user", "content": "t"}], [], 400)]) == ([], [])


def test_episode_json_round_trip():
    episode = GymEpisode(
        output_items=[{"type": "message"}],
        conversation=[],
        model_calls=[_record([], [])],
        tool_calls=[
            ToolCallRecord("inv", "c1", "shell", {"command": "ls"}, "ran", 1.0, 2.0, "completed")
        ],
        return_value={"value": 1},
        stop_reason="completed",
        resource_cookies={"sid": "1"},
    )
    restored = GymEpisode.from_json(episode.to_json())
    assert restored == episode


# --- resource tools ---------------------------------------------------------------------------


def test_resource_tools_build_schema_and_validate(monkeypatch):
    seen = {}

    async def fake_post(self, url, json=None, headers=None):
        seen["url"], seen["json"], seen["headers"] = url, json, headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, json={"ok": True}, request=request, headers={"set-cookie": "rs=abc"}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    cookies: dict[str, str] = {}
    [tool] = resource_tools(
        [
            {
                "type": "function",
                "name": "increment",
                "description": "Add to the counter.",
                "parameters": {
                    "type": "object",
                    "properties": {"by": {"type": "integer"}},
                    "required": ["by"],
                },
            }
        ],
        base_url="http://resources/ng-rollout/r/",
        headers={"x-grant": "1"},
        cookies=cookies,
    )
    assert tool.get_parameter_schema()["properties"]["by"]["type"] == "integer"
    assert asyncio.run(tool.callable(by="three")) == {
        "error": "Invalid arguments for increment: 'three' is not of type 'integer'"
    }
    assert asyncio.run(tool.callable(by=3)) == {"ok": True}
    assert seen["url"] == "http://resources/ng-rollout/r/increment" and seen["json"] == {"by": 3}
    assert seen["headers"] == {"x-grant": "1"}


def test_resource_tools_reject_non_function_tools():
    with pytest.raises(ValueError):
        resource_tools([{"type": "web_search"}], base_url="http://r")


# --- run_gym_episode against a Gym-shaped Responses server ------------------------------------


def _responses_app(log: list[dict]) -> FastAPI:
    app = FastAPI()
    state = {"n": 0}

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        body = await request.json()
        log.append({"headers": dict(request.headers), "body": body})
        state["n"] += 1
        if state["n"] == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "shell",
                    "status": "completed",
                    "arguments": json.dumps({"command": "echo hi"}),
                    **TRAIN,
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "DONE", "annotations": []}],
                    **TRAIN,
                }
            ]
        payload = {
            "id": f"resp_{state['n']}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "model": "gym-policy",
            "output": output,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        response = JSONResponse(payload)
        response.set_cookie("gym_model_session", "s1")
        return response

    return app


async def _serve(app: FastAPI):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, sock.getsockname()[1]


async def test_run_gym_episode_records_wire_and_tools():
    log: list[dict] = []
    server, task, port = await _serve(_responses_app(log))
    commands: list[str] = []

    async def run_command(command: str) -> str:
        commands.append(command)
        return "hi\n[exit 0]"

    try:
        episode = await run_gym_episode(
            lambda llm: ShellAgent(llm=llm, run_command=run_command),
            lambda agent: agent.solve("say hi"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            max_policy_calls=5,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "completed" and episode.return_value == "DONE"
    assert commands == ["echo hi"]
    assert [r.status_code for r in episode.model_calls] == [200, 200]
    assert [i["type"] for i in episode.output_items] == [
        "function_call",
        "function_call_output",
        "message",
    ]
    assert episode.output_items[0]["generation_token_ids"] == [
        3
    ]  # replayed verbatim with training fields
    assert [t.name for t in episode.tool_calls] == ["shell"]
    assert "x-session-id" in log[0]["headers"]
    assert "gym_model_session=s1" in log[1]["headers"].get("cookie", "")
    # Gym's request model forbids extra keys; the OpenAI explicit cache option must never go out.
    assert all("prompt_cache_options" not in entry["body"] for entry in log)
    # ... and chat-backed Gym model servers cannot downconvert `truncation`.
    assert all(entry["body"].get("truncation") is None for entry in log)


def test_recording_client_keeps_openai_explicit_cache_off():
    client = _RecordingClient(
        "gym-policy", max_policy_calls=None, records=[], api_base="http://127.0.0.1:1/v1"
    )
    assert client.cache_breakpoint is None
    history = [
        {"role": "system", "content": "stable"},
        CacheBoundary(),
        {"role": "user", "content": "task"},
    ]
    _messages, _instructions, openai_explicit = client._prepare_cache_boundary(
        history, responses=True
    )
    assert openai_explicit is False
    # Still an opt-in for providers that support it.
    explicit = _RecordingClient(
        "gym-policy",
        max_policy_calls=None,
        records=[],
        api_base="http://127.0.0.1:1/v1",
        cache_breakpoint="openai",
    )
    assert explicit.cache_breakpoint == "openai"


async def test_run_gym_episode_policy_budget_is_a_gradable_stop():
    log: list[dict] = []
    server, task, port = await _serve(_responses_app(log))
    try:
        episode = await run_gym_episode(
            lambda llm: ShellAgent(llm=llm, run_command=lambda c: asyncio.sleep(0, result="ok")),
            lambda agent: agent.solve("say hi"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            max_policy_calls=1,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "policy_budget_exceeded"
    assert len(log) == 1


def _reasoning_app(log: list[dict]) -> FastAPI:
    """First turn: a reasoning item plus a shell call; second turn: the final message."""
    app = FastAPI()
    state = {"n": 0}

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        log.append({"body": await request.json()})
        state["n"] += 1
        if state["n"] == 1:
            output = [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "summary": [
                        {"type": "summary_text", "text": "Let me start by running the command."}
                    ],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "shell",
                    "status": "completed",
                    "arguments": json.dumps({"command": "echo hi"}),
                },
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "DONE", "annotations": []}],
                }
            ]
        return JSONResponse(
            {
                "id": f"resp_{state['n']}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gym-policy",
                "output": output,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )

    return app


async def _second_request_types(replay_reasoning: bool) -> tuple[list[str], GymEpisode]:
    log: list[dict] = []
    server, task, port = await _serve(_reasoning_app(log))

    async def run_command(command: str) -> str:
        return "hi\n[exit 0]"

    try:
        episode = await run_gym_episode(
            lambda llm: ShellAgent(llm=llm, run_command=run_command),
            lambda agent: agent.solve("say hi"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            max_policy_calls=5,
            replay_reasoning=replay_reasoning,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "completed", episode.error
    return [item.get("type") or item.get("role") for item in log[1]["body"]["input"]], episode


async def test_prior_reasoning_is_not_replayed_by_default():
    # The Gym route looks like OpenAI, so ResponsesClient would otherwise replay the reasoning
    # summary every turn; the thick nooa_agent never does, and chat-backed Gym servers would
    # render it back to the policy as <think> text.
    types, episode = await _second_request_types(replay_reasoning=False)
    assert "reasoning" not in types, types
    assert "function_call" in types and "function_call_output" in types
    # The capture still has the reasoning the policy produced.
    assert any(item["type"] == "reasoning" for item in episode.model_calls[0].response["output"])


async def test_prior_reasoning_is_replayed_when_asked():
    types, _ = await _second_request_types(replay_reasoning=True)
    assert "reasoning" in types, types


def _final_message_app(log: list[dict]) -> FastAPI:
    """Every turn: record the request and answer with a final text message."""
    app = FastAPI()

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        log.append({"body": await request.json()})
        return JSONResponse(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gym-policy",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Red", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )

    return app


class _ImageAgent:
    """Sends one user turn carrying a captured image, the shape Atom's show() produces."""

    def __init__(self, llm):
        self.llm = llm

    async def solve(self, task: str):
        return await self.llm.acall(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ]
        )


async def _image_request_content(media_inputs: tuple[str, ...]) -> list[dict]:
    log: list[dict] = []
    server, task, port = await _serve(_final_message_app(log))
    try:
        episode = await run_gym_episode(
            _ImageAgent,
            lambda agent: agent.solve("What colour is this?"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            max_policy_calls=2,
            media_inputs=media_inputs,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "completed", episode.error
    (user_turn,) = [item for item in log[0]["body"]["input"] if item.get("role") == "user"]
    return user_turn["content"]


async def test_images_reach_the_policy_with_detail_when_enabled():
    # Gym's Responses schema requires `detail` on input_image; NOOA sets it only when the image
    # carried one, so the harness fills in the Responses default.
    content = await _image_request_content(media_inputs=("image",))
    images = [block for block in content if block["type"] == "input_image"]
    assert images == [
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"}
    ]


async def test_images_are_replaced_by_a_note_when_media_is_off():
    content = await _image_request_content(media_inputs=())
    assert all(block["type"] == "input_text" for block in content), content
    assert len(content) == 2 and "image" in content[1]["text"].lower(), content


# --- stopping at the time budget --------------------------------------------------------------


def _stalling_app(log: list[dict], release: asyncio.Event) -> FastAPI:
    """First turn: a shell call; the second request stalls until ``release`` is set."""
    app = FastAPI()

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        log.append({"body": await request.json()})
        if len(log) > 1:
            await asyncio.wait_for(release.wait(), 30)
        return JSONResponse(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gym-policy",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "shell",
                        "status": "completed",
                        "arguments": json.dumps({"command": "echo hi"}),
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )

    return app


async def _wait_for(predicate, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def test_an_interrupted_episode_keeps_its_records_when_asked():
    log: list[dict] = []
    render: list = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = asyncio.create_task(
            run_gym_episode(
                lambda llm: ShellAgent(
                    llm=llm, run_command=lambda c: asyncio.sleep(0, result="hi\n[exit 0]")
                ),
                lambda agent: agent.solve("say hi"),
                model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
                interrupt=lambda: "timeout_exceeded",
                snapshot=render.append,
            )
        )
        await _wait_for(lambda: len(log) == 2)
        # What a deadline checkpoint would write right now, before anything is cancelled.
        checkpoint = render[0]("timeout_exceeded")
        run.cancel()
        episode = await run
    finally:
        release.set()
        server.should_exit = True
        await task
    assert episode.stop_reason == "timeout_exceeded" and episode.return_value is None
    # The snapshot taken before the cancellation equals the finished episode.
    assert checkpoint.stop_reason == "timeout_exceeded"
    assert checkpoint.to_json() == episode.to_json()
    assert [r.status_code for r in episode.model_calls] == [
        200,
        None,
    ]  # the stalled call stays pending
    assert [t.name for t in episode.tool_calls] == ["shell"]
    # The summary comes from the last finished call; the stalled call's input is not replayed.
    assert [i["type"] for i in episode.output_items] == ["function_call"]
    GymEpisode.from_json(episode.to_json())  # the checkpoint is a complete, serialisable episode


async def test_an_interrupted_episode_propagates_the_cancellation_by_default():
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = asyncio.create_task(
            run_gym_episode(
                lambda llm: ShellAgent(
                    llm=llm, run_command=lambda c: asyncio.sleep(0, result="hi\n[exit 0]")
                ),
                lambda agent: agent.solve("say hi"),
                model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            )
        )
        await _wait_for(lambda: len(log) == 2)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    finally:
        release.set()
        server.should_exit = True
        await task


async def test_worker_checkpoints_its_episode_on_sigterm(tmp_path):
    """The sandbox supervisor sends SIGTERM at the agent's time budget and waits a short grace
    period; the worker must leave result.json behind so Gym grades the sandbox state."""
    import os
    import signal
    import sys

    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    inputs, result = tmp_path / "input.json", tmp_path / "result.json"
    inputs.write_text(
        json.dumps(
            {
                "task": "say hi",
                "agent": "nooa_bench.gym:ShellAgent",
                "model_url": f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
                "workdir": str(tmp_path),
            }
        )
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "nooa_bench.gym_worker",
        str(inputs),
        str(result),
        env=dict(os.environ),
    )
    try:
        await _wait_for(lambda: len(log) == 2, timeout=60)
        process.send_signal(signal.SIGTERM)
        async with asyncio.timeout(10):  # well inside the supervisor's TERM grace
            assert await process.wait() == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        release.set()
        server.should_exit = True
        await task
    episode = GymEpisode.from_json(result.read_text())
    assert episode.stop_reason == "timeout_exceeded"
    assert episode.runtime["stop_signal"] == "SIGTERM"
    assert [r.status_code for r in episode.model_calls] == [200, None]
    assert episode.runtime["python"] == sys.executable


async def test_an_interrupt_that_declines_lets_the_cancellation_propagate():
    """`interrupt` returning None is the worker's answer to any cancellation it did not ask for."""
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = asyncio.create_task(
            run_gym_episode(
                lambda llm: ShellAgent(
                    llm=llm, run_command=lambda c: asyncio.sleep(0, result="hi\n[exit 0]")
                ),
                lambda agent: agent.solve("say hi"),
                model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
                interrupt=lambda: None,
            )
        )
        await _wait_for(lambda: len(log) == 2)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    finally:
        release.set()
        server.should_exit = True
        await task


class SelfCancellingAgent(ShellAgent):
    """Probe: an internal cancellation of the worker's task, raised from inside a tool call
    (so the probe has made one model call before it misbehaves)."""

    async def shell(self, command: str) -> str:
        asyncio.current_task().cancel()
        await asyncio.sleep(0)
        return "unreachable"


async def _worker(tmp_path, port: int, agent: str, *, env_extra: dict | None = None):
    import os
    import sys

    inputs, result = tmp_path / "input.json", tmp_path / "result.json"
    inputs.write_text(
        json.dumps(
            {
                "task": "say hi",
                "agent": agent,
                "model_url": f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
                "workdir": str(tmp_path),
            }
        )
    )
    env = dict(os.environ)
    # The probe agents live in this test module; make it importable by the worker process.
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(__import__("pathlib").Path(__file__).parent), env.get("PYTHONPATH")])
    )
    env.update(env_extra or {})
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "nooa_bench.gym_worker", str(inputs), str(result), env=env
    )
    return process, result


async def test_worker_leaves_no_result_on_sigint(tmp_path):
    import signal

    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    process, result = await _worker(tmp_path, port, "nooa_bench.gym:ShellAgent")
    try:
        await _wait_for(lambda: len(log) == 2, timeout=60)
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(10):
            code = await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        release.set()
        server.should_exit = True
        await task
    assert code != 0 and not result.exists()


async def test_worker_leaves_no_result_on_an_internal_cancellation(tmp_path):
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    process, result = await _worker(tmp_path, port, "test_gym:SelfCancellingAgent")
    try:
        async with asyncio.timeout(60):
            code = await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        release.set()
        server.should_exit = True
        await task
    assert len(log) == 1  # the probe ran: one model call, then it cancelled itself
    assert code != 0 and not result.exists()
