# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual worker/Responses serialization into HTTPX MockTransport; no sockets/provider."""

import asyncio
import json
import os
import socket
import traceback

import httpx
import pytest
from nooa_atom.benchmark_policy import POLICY
from nooa_bench.gym import GymEpisode, _RecordingClient
from nooa_bench.gym_worker import main

DONE = "return Done(explanation='done', result=TaskResult(solution_description='s', evidence='e', how_to_verify='v', report='accepted'))"


class IgnoringHarness:
    benchmark_policy_required = True

    def __init__(self, **kwargs):
        pass  # A stale artifact that accepts and ignores policy kwargs.

    async def solve(self, task):
        raise AssertionError("missing policy must be caught before invoke")


@pytest.fixture
async def wire(monkeypatch):
    # Disable only the optional local-viewer discovery probe. Runtime event
    # subscriptions, middleware, and _ToolHooks remain live and are asserted.
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)
    # A prior test may already have installed optional LiteLLM viewer callbacks.
    # Stub only that outbound journal sink, not runtime hooks/middleware or HTTP
    # policy serialization. This fixture exercises the local artifact, not OTLP.
    monkeypatch.setattr("nooa.tracing._litellm_journal._post_json", lambda *args, **kwargs: None)
    scripts, calls, clients, transports = [], [], [], []
    blocked = []

    def deny(*args, **kwargs):
        blocked.append((str(args), "".join(traceback.format_stack(limit=28))))
        raise AssertionError("real network forbidden by offline fixture")

    async def deny_async(*args, **kwargs):
        deny()

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_async)

    async def respond(request):
        body = json.loads(request.content)
        calls.append((body, request.headers["x-session-id"]))
        assert scripts, "unexpected model request"
        script = scripts.pop(0)
        if isinstance(script, asyncio.Event):
            await script.wait()
            raise AssertionError("cancelled mock request should not finish")
        offered = {tool["name"] for tool in body.get("tools", [])}
        tool = "python_cell" if "python_cell" in offered else "execute_python"
        if isinstance(script, dict):
            output = [
                {
                    "type": "message",
                    "id": f"msg_{len(calls)}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": json.dumps(script), "annotations": []}
                    ],
                }
            ]
        else:
            output = [
                {
                    "type": "function_call",
                    "id": f"fc_{len(calls)}",
                    "call_id": f"call_{len(calls)}",
                    "name": tool,
                    "arguments": json.dumps({"code": script}),
                    "status": "completed",
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(calls)}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gym-policy",
                "output": output,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = _RecordingClient.__init__

    def init(client, *args, **kwargs):
        original(client, *args, **kwargs)
        clients.append(client)
        http = client._http.httpx_async
        transports.extend([http._transport, *[t for t in http._mounts.values() if t is not None]])
        mock = httpx.MockTransport(respond)
        http._transport = mock
        http._mounts = dict.fromkeys(http._mounts, mock)

    monkeypatch.setattr(_RecordingClient, "__init__", init)
    yield scripts, calls, clients
    for client in clients:
        await client.aclose()
    for transport in transports:
        await transport.aclose()
    assert not blocked, "\n".join(stack for _, stack in blocked)


def payload(tmp_path, *, agent="nooa_atom.gym:AtomHarness", **updates):
    data = {
        "agent": agent,
        "task": "ordinary task quoting DoneReflectionPlan _reflect_done_plan name_session",
        "workdir": str(tmp_path),
        "model_url": "http://policy.invalid/v1",
        "agent_kwargs": {
            "benchmark_policy": POLICY,
            "experimental_done_reflection": False,
            "benchmark_allow_session_titles": False,
            "experimental_reflection_min_turns": 1,
            "experimental_reflection_min_seconds": 0,
            "sessions_dir": str(tmp_path / "sessions"),
            "session_name": "fixed_host_task",
        },
    }
    data.update(updates)
    inputs, output = tmp_path / "input.json", tmp_path / "output.json"
    inputs.write_text(json.dumps(data))
    return data, inputs, output


def read(output):
    episode = GymEpisode.from_json(output.read_text())
    receipt_path = output.parent / episode.runtime["benchmark_policy_path"]
    assert receipt_path.name == output.name + ".atom-policy.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["run_id"] == episode.runtime["benchmark_policy_run_id"]
    return episode, receipt


async def test_worker_actual_constructor_to_session_and_wire_map(tmp_path, wire):
    scripts, calls, _ = wire
    scripts.append(DONE)
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "completed" and receipt["verdict"] == "pass"
    assert episode.return_value["report"] == "accepted"
    assert len(calls) == len(episode.model_calls) == 1
    [session] = receipt["sessions"]
    assert session["runtime_namespace"] == "session" and session["runtime_reflection"] is False
    assert session["ready_context_session_id"] == session["session_id"]
    assert session["initial"]["name"] == "fixed_host_task"
    assert session["initial"]["title"] is None
    mapping = {row["invocation_id"]: row for row in receipt["invocations"]}
    for (body, header), record in zip(calls, episode.model_calls, strict=True):
        assert record.request == body and record.invocation_id == header
        assert mapping[header]["session_id"] == session["session_id"] != header
        assert mapping[header]["http_dispatches"] == 1
        assert "DoneReflectionPlan" in json.dumps(body)  # quoted data is not reflection
    assert receipt["observed_attempt_counts"] == {
        "reflection": 0,
        "title_generation": 0,
        "rename": 0,
        "title_mutation": 0,
    }


async def test_worker_child_keeps_outer_wire_and_tool_correlation_without_titling(tmp_path, wire):
    scripts, calls, _ = wire
    scripts.extend(
        [
            "await self.jobs.launch(Subagent('not a display title').run('work'), name='kid')\n"
            + DONE,
            DONE,
        ]
    )
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "completed" and len(receipt["sessions"]) == 2
    mapping = {row["invocation_id"]: row["session_id"] for row in receipt["invocations"]}
    assert {mapping[header] for _, header in calls} == {
        row["session_id"] for row in receipt["sessions"]
    }
    assert len(calls) == len(episode.model_calls) == 2
    assert episode.terminal_response_id == episode.model_calls[0].response["id"]
    assert episode.terminal_response_id != episode.model_calls[1].response["id"]
    assert all(
        row["initial"]["title"] is row["final"]["title"] is None for row in receipt["sessions"]
    )
    assert all(record.invocation_id in mapping for record in episode.tool_calls)


@pytest.mark.parametrize(
    "method,kind",
    [
        ("_reflect_done_plan({})", "reflection"),
        ("name_session('quoted _reflect_done_plan')", "title_generation"),
        ("rename_session('new')", "rename"),
    ],
)
async def test_worker_latched_attempt_overrides_model_done_and_never_dispatches_predict(
    tmp_path, wire, method, kind
):
    scripts, calls, _ = wire
    scripts.append("try:\n await self." + method + "\nexcept RuntimeError:\n pass\n" + DONE)
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "error" and episode.return_value is None
    assert receipt["verdict"] == "fail" and receipt["observed_attempt_counts"][kind] == 1
    assert len(calls) == len(episode.model_calls) == 1
    assert all(not (body.get("text") or {}).get("format") for body, _ in calls)


@pytest.mark.parametrize(
    "missing",
    [
        "benchmark_policy",
        "experimental_done_reflection",
        "benchmark_allow_session_titles",
    ],
)
async def test_worker_missing_constructor_configuration_writes_failed_receipt_without_requests(
    tmp_path, wire, missing
):
    data, inputs, output = payload(tmp_path)
    del data["agent_kwargs"][missing]
    inputs.write_text(json.dumps(data))
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "error" and receipt["verdict"] == "fail"
    assert receipt["phase"] == "final" and not receipt["sessions"] and not wire[1]


async def test_worker_stale_ignoring_harness_gets_durable_failure_before_invoke(tmp_path, wire):
    _, inputs, output = payload(tmp_path, agent=f"{__name__}:IgnoringHarness")
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "error" and receipt["verdict"] == "fail"
    assert receipt["violations"][0]["code"] == "policy_guard_missing"
    assert not wire[1]


async def test_worker_cancellation_retains_policy_sidecar_but_no_success_output(tmp_path, wire):
    scripts, calls, _ = wire
    scripts.append(asyncio.Event())
    _, inputs, output = payload(tmp_path)
    task = asyncio.create_task(main(inputs, output))
    try:
        async with asyncio.timeout(30):
            while not calls:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert not output.exists()  # preserves worker's ordinary-cancellation contract
    receipt = json.loads(output.with_name(output.name + ".atom-policy.json").read_text())
    assert receipt["phase"] == "final" and receipt["outcome"] == "cancelled"
    assert all(row["closed"] for row in receipt["sessions"])


async def test_worker_ordinary_predict_with_quoted_markers_is_not_forbidden(tmp_path, wire):
    scripts, calls, _ = wire
    scripts.extend(
        [
            "@strategy(PredictStrategy())\nasync def ordinary_echo(text: str) -> str:\n \"Return ordinary input data, not maintenance or a title.\"\n ...\nanswer=await ordinary_echo('DoneReflectionPlan _reflect_done_plan name_session')\nassert answer == 'ok'\n"
            + DONE,
            {"value": "ok"},
        ]
    )
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "completed" and receipt["verdict"] == "pass"
    assert len(calls) == len(episode.model_calls) == 2
    predict = calls[1][0]
    assert predict["text"]["format"]["name"] == "OrdinaryEchoResponse"
    assert "DoneReflectionPlan" in json.dumps(predict["input"])
    assert all(count == 0 for count in receipt["observed_attempt_counts"].values())


@pytest.mark.parametrize("violation", [False, True])
async def test_worker_sigterm_checkpoint_is_incomplete_until_final_coverage_and_never_masks_violation(
    tmp_path, wire, monkeypatch, violation
):
    import signal

    scripts, calls, clients = wire
    callbacks = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, fn: callbacks.__setitem__(sig, fn))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: callbacks.pop(sig, None))
    if violation:
        scripts.append(
            "try:\n await self.rename_session('denied')\nexcept RuntimeError:\n pass\nawait asyncio.sleep(1000)"
        )
    else:
        scripts.append(asyncio.Event())
    _, inputs, output = payload(tmp_path)
    task = asyncio.create_task(main(inputs, output))
    try:
        async with asyncio.timeout(30):
            while not calls or (
                violation and clients[0]._required_episode_policy.counts["rename"] == 0
            ):
                await asyncio.sleep(0.01)
        callbacks[signal.SIGTERM]()
        checkpoint, interim = read(output)
        assert checkpoint.stop_reason == "error" and interim["phase"] == "pending"
        await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    episode, receipt = read(output)
    assert receipt["phase"] == "final" and all(row["closed"] for row in receipt["sessions"])
    assert episode.runtime["stop_signal"] == "SIGTERM"
    assert episode.stop_reason == ("error" if violation else "timeout_exceeded")
    assert receipt["verdict"] == ("fail" if violation else "pass")
    assert not callbacks


async def test_worker_missing_final_verdict_interface_cannot_be_replaced_by_done(
    tmp_path, wire, monkeypatch
):
    from nooa_atom import gym

    from nooa.agentdoc import hidden

    class MissingFinalVerdict(gym.AtomHarness):
        @hidden
        def benchmark_policy_receipt(self):
            return None

    # f497's harness is a real Agent: method mutation is correctly forbidden.
    # Simulate an incompatible artifact by defining its override normally.
    monkeypatch.setattr(gym, "AtomHarness", MissingFinalVerdict)
    wire[0].append(DONE)
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "error" and episode.return_value is None
    assert receipt["verdict"] == "fail"
    assert "harness_verdict_mismatch" in {row["code"] for row in receipt["violations"]}


async def test_worker_receipt_write_failure_leaves_no_success_output(tmp_path, wire, monkeypatch):
    from nooa_atom.benchmark_policy import BenchmarkPolicyGuard, BenchmarkPolicyViolation

    original = BenchmarkPolicyGuard._write

    def broken(guard):
        if guard.finalized:
            raise OSError("receipt writer failed")
        return original(guard)

    monkeypatch.setattr(BenchmarkPolicyGuard, "_write", broken)
    wire[0].append(DONE)
    _, inputs, output = payload(tmp_path)
    with pytest.raises(BenchmarkPolicyViolation, match="verdict missing"):
        await main(inputs, output)
    assert not output.exists()
    receipt = json.loads(output.with_name(output.name + ".atom-policy.json").read_text())
    assert receipt["phase"] == "pending"


async def test_worker_cannot_redirect_receipt_with_agent_kwargs(tmp_path, wire):
    data, inputs, output = payload(tmp_path)
    data["agent_kwargs"]["benchmark_policy_path"] = str(tmp_path / "redirected.json")
    inputs.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="worker-owned"):
        await main(inputs, output)
    assert not wire[1] and not output.exists() and not (tmp_path / "redirected.json").exists()


async def test_worker_shell_is_completion_only_and_does_not_add_model_calls(tmp_path, wire):
    scripts, calls, _ = wire
    # Output, then a pause past the former 1-second preview window, then more output.
    command = "printf READY; sleep 1.3; printf END"
    scripts.append(
        f"work=self.jobs.launch(Shell().run({command!r}), name='live')\n"
        "seen=[part async for part in work]\n"
        "result=await work\n"
        "assert seen == [] and result.stdout == 'READYEND', (seen, result)\n" + DONE
    )
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "completed" and receipt["verdict"] == "pass"
    assert len(calls) == len(episode.model_calls) == 1
    assert receipt["sessions"][0]["runtime_namespace"] == "session"
    assert all(count == 0 for count in receipt["observed_attempt_counts"].values())


@pytest.mark.parametrize("cancelled", [False, True], ids=["exception", "cancel"])
@pytest.mark.parametrize("child", [False, True], ids=["root", "child"])
async def test_worker_cleanup_failure_cannot_return_an_accepted_task_result(
    tmp_path, wire, monkeypatch, cancelled, child
):
    from nooa_atom.session.session import Session

    original = Session._close_step
    seen = []

    async def fail_agent_close(session, what, step):
        if what == "closing the agent" and bool(session.parent_id) is child:
            seen.append(session)

            async def fail():
                raise (asyncio.CancelledError if cancelled else OSError)("SECRET_WORKER_CLEANUP")

            step = fail
        return await original(session, what, step)

    monkeypatch.setattr(Session, "_close_step", fail_agent_close)
    scripts, calls, _ = wire
    if child:
        scripts.append(
            "await self.jobs.launch(Subagent('child').run('task'), name='child')\n" + DONE
        )
    scripts.append(DONE)
    _, inputs, output = payload(tmp_path)
    try:
        await main(inputs, output)
        episode, receipt = read(output)
        assert episode.stop_reason == "error" and episode.return_value is None
        assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
        row = next(row for row in receipt["sessions"] if row["session_id"] == seen[0].id)
        assert row["cleanup"]["error_count"] == 1
        assert all(row["closed"] for row in receipt["sessions"])
        assert "SECRET_WORKER_CLEANUP" not in json.dumps(receipt) + (episode.error or "")
        assert len(calls) == 1 + int(child)
    finally:
        for session in seen:
            await session._agent.aclose()


class AlternateEntryHarness:
    def __init__(self, llm, **kwargs):
        pass

    async def solve(self, task):
        raise AssertionError("worker ignored explicit entry method")

    async def work(self, task):
        return {"task": task}


async def test_worker_entry_and_startup_environment_names_only(tmp_path, wire, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("OTLP_", "OTEL_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OTLP_ENDPOINT", "DO_NOT_RECORD_THIS_VALUE")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "DO_NOT_RECORD_SECRET_HEADERS")
    _, inputs, output = payload(
        tmp_path, agent=f"{__name__}:AlternateEntryHarness", agent_kwargs={}, turn_method="work"
    )
    await main(inputs, output)
    episode = GymEpisode.from_json(output.read_text())
    assert episode.stop_reason == "completed" and episode.return_value["task"]
    assert episode.runtime["trace_env"] == "OTEL_EXPORTER_OTLP_HEADERS,OTLP_ENDPOINT"
    assert "DO_NOT_RECORD" not in output.read_text()
    assert "otlp_endpoint" not in episode.runtime and not wire[1]


@pytest.mark.parametrize("updates", [{"tool_binding": "unsupported:bind"}, {"tools": [{}]}])
async def test_atom_worker_rejects_unbound_gym_tools_before_inference(tmp_path, wire, updates):
    _, inputs, output = payload(tmp_path, **updates)
    with pytest.raises(ValueError, match="requires no Gym tools"):
        await main(inputs, output)
    assert not wire[1] and not output.exists()


async def test_worker_requires_actual_task_result(tmp_path, wire, monkeypatch):
    from nooa_atom.session.session import Session

    from nooa.interactive import Done

    original = Session.prompt

    async def wrong_result(session, *args, **kwargs):
        await original(session, *args, **kwargs)
        return Done.model_construct(explanation="done", result="not a TaskResult")

    monkeypatch.setattr(Session, "prompt", wrong_result)
    wire[0].append(DONE)
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "error" and episode.return_value is None
    assert "without a TaskResult" in episode.error
    assert receipt["verdict"] == "pass"  # result-contract failure, not a policy violation


@pytest.mark.parametrize("child,violate", [(False, False), (True, False), (True, True)])
async def test_worker_native_file_exports_and_policy_correlation(
    tmp_path, wire, monkeypatch, child, violate
):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    import nooa.atif.install as atif_install
    import nooa.tracing as tracing
    from nooa.agent import Agent
    from nooa.runtime.hooks import get_hooks, set_hooks

    prior_hooks = get_hooks()
    provider = TracerProvider(shutdown_on_exit=False)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_hooks", None)
    # Optional LiteLLM callbacks are not the native hooks/exports under test.
    monkeypatch.setattr(tracing, "_instrument_litellm", lambda provider: None)
    monkeypatch.setattr(Agent, "__init__", Agent.__init__)
    monkeypatch.setattr(atif_install, "_AGENT_INIT_PATCHED", False)
    monkeypatch.setattr(atif_install, "_ENABLE_ATIF_CONFIG", {})
    scripts, calls, _ = wire
    reply = (
        "try:\n await self.rename_session('forbidden')\nexcept RuntimeError:\n pass\n" + DONE
        if violate
        else DONE
    )
    scripts.extend(
        ["await self.jobs.launch(Subagent('child').run('work'), name='kid')\n" + DONE, reply]
        if child
        else [reply]
    )
    artifacts = tmp_path / "artifacts"
    _, inputs, output = payload(
        tmp_path,
        artifacts_dir=str(artifacts),
        atif=True,
        trace_jsonl=True,
        otlp_endpoint=None,
        tool_binding=None,
        turn_method="solve",
    )
    try:
        await main(inputs, output)
        tracing.flush_traces(timeout_millis=1000)
        episode, receipt = read(output)
        assert episode.stop_reason == ("error" if violate else "completed")
        assert receipt["verdict"] == ("fail" if violate else "pass")
        if violate:
            assert (
                episode.return_value is None and receipt["observed_attempt_counts"]["rename"] == 1
            )
        assert len(receipt["sessions"]) == (2 if child else 1)
        assert len(episode.model_calls) == len(calls) == (2 if child else 1)
        assert episode.terminal_response_id == episode.model_calls[0].response["id"]
        assert "otlp_endpoint" not in episode.runtime
        atif_files = list((artifacts / "atif").rglob("*.json"))
        agent_files = [path for path in atif_files if path.parent.name == "AtomAgent"]
        wrapper_files = [path for path in atif_files if path.parent.name == "AtomHarness"]
        assert len(agent_files) == (2 if child else 1)
        assert len(wrapper_files) == 1  # actual f497 Agent-based harness, not a native child
        assert all(json.loads(path.read_text())["steps"] for path in agent_files)
        traces = list((artifacts / "traces").rglob("*.jsonl"))
        assert traces and all(path.stat().st_size for path in traces)
        for path in traces:
            for line in path.read_text().splitlines():
                assert json.loads(line)
        assert all(
            record.invocation_id in {row["invocation_id"] for row in receipt["invocations"]}
            for record in episode.model_calls
        )
    finally:
        provider.shutdown()
        set_hooks(prior_hooks)


@pytest.mark.parametrize("state", ["running", "waiting", "queued"])
async def test_worker_rejects_real_unfinished_children_and_closes_them(
    tmp_path, wire, monkeypatch, state
):
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.session import Session

    from nooa.runtime.channels import QueueManager

    published = []
    registries = []
    original_publish = SessionRegistry._publish
    original_prompt = Session.prompt
    original_wait = QueueManager.wait_ready
    hold_queue = asyncio.Event()
    observed = []

    def publish(registry, session, reservation):
        published.append(session)
        registries.append(registry)
        return original_publish(registry, session, reservation)

    async def wait_ready(manager, *args, **kwargs):
        if state == "queued" and any(
            manager is child._agent.queue_manager for child in published[1:]
        ):
            await hold_queue.wait()
        return await original_wait(manager, *args, **kwargs)

    async def prompt(session, *args, **kwargs):
        outcome = await original_prompt(session, *args, **kwargs)
        if published and session is published[0]:
            # Admit real child work at the host boundary after a valid root Done.
            # Generated root Done already refuses its own live jobs, so asking the
            # model to violate that invariant would test a repair loop, not this seam.
            child = await registries[0].create(
                session.options.inherit(name="pending-child", kept=True, turn_method="handle_work"),
                parent_id=session.id,
                initial_items=[("user_messages", "work")],
            )
            async with asyncio.timeout(10):
                while True:
                    ready = (
                        len(wire[1]) == 2
                        if state == "running"
                        else child.last_outcome_kind == "waiting"
                        if state == "waiting"
                        else child.has_queued_input()
                    )
                    if ready:
                        observed.append((state, child.id))
                        break
                    await asyncio.sleep(0)
        return outcome

    monkeypatch.setattr(SessionRegistry, "_publish", publish)
    monkeypatch.setattr(QueueManager, "wait_ready", wait_ready)
    monkeypatch.setattr(Session, "prompt", prompt)
    scripts = wire[0]
    scripts.append(DONE)
    if state == "running":
        scripts.append(asyncio.Event())
    elif state == "waiting":
        scripts.append(
            "self.jobs.launch(asyncio.sleep(3600), name='pending')\n"
            "return Waiting(explanation='work pending', on=['pending'])"
        )
    _, inputs, output = payload(tmp_path)
    await main(inputs, output)
    episode, receipt = read(output)
    assert observed and len(published) == len(receipt["sessions"]) == 2
    assert episode.stop_reason == "error" and episode.return_value is None
    assert "unfinished work" in episode.error
    assert all(session._closed for session in published)
    assert all(
        job.status != "running" for session in published for job in session._agent.jobs.list()
    )
    assert all(row["cleanup"]["error_count"] == 0 for row in receipt["sessions"])


@pytest.mark.parametrize("obsolete", ["legacy", "call", "session", None])
async def test_worker_rejects_removed_namespace_selector_before_model_call(
    tmp_path, wire, obsolete
):
    data, inputs, output = payload(tmp_path)
    data["agent_kwargs"]["experimental_namespace"] = obsolete
    inputs.write_text(json.dumps(data))
    await main(inputs, output)
    episode, receipt = read(output)
    assert episode.stop_reason == "error" and receipt["verdict"] == "fail"
    assert receipt["phase"] == "final" and not receipt["sessions"] and not wire[1]
