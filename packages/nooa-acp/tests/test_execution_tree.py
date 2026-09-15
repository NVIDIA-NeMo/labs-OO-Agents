# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution trees reflect native nested calls, including errors and concurrency."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from itertools import count
from types import SimpleNamespace
from typing import Any, cast

from acp.interfaces import Client
from acp.schema import ToolCallProgress, ToolCallStart
from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_acp.event_bridge import ACPEventBridge
from nooa_acp.execution_tree import EXECUTION_META_KEY
from nooa_cli.coding import CodingAgent
from nooa_cli.coding.experimental_agent import ExperimentalCodingAgent

from nooa import Agent, no_trace
from nooa.runtime.hooks import get_hooks, hooks_scope
from nooa.slash_dispatch import SlashCommandResult
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class TreeChild(Agent):
    async def work(self, value: int) -> int:
        await asyncio.sleep(0)
        return self.double(value)

    def double(self, value: int) -> int:
        return value * 2

    async def generated(self, value: int) -> int:
        """Double the value."""
        ...

    async def fail(self) -> None:
        raise ValueError("deliberate child failure")

    async def wait(self) -> None:
        await asyncio.Event().wait()

    @no_trace
    def untraced(self) -> int:
        return self.double(3)

    def stream(self) -> Iterator[int]:
        yield self.double(1)
        yield self.double(2)


class TreeCodingAgent(CodingAgent):
    child: TreeChild

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = TreeChild(
            llm=FakeLLMClient.with_tool_call(
                "execute_python", {"code": "return_result(self.double(value))"}
            )
        )


class RecordingClient:
    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.wait_started = asyncio.Event()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)
        if isinstance(update, ToolCallStart) and update.title == "TreeChild.wait":
            self.wait_started.set()


class ExistingHooks:
    """Record the protocol callback names without changing their results."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def record(**kwargs: Any) -> dict[str, bool]:
            self.calls.append(name)
            return {"observed": True}

        return record


def metadata(update: ToolCallStart | ToolCallProgress) -> dict[str, Any]:
    assert update.field_meta is not None
    node = update.field_meta[EXECUTION_META_KEY]
    assert node["version"] == 1
    assert node["spanId"] == update.tool_call_id
    return node


async def test_tree_tracks_real_nested_methods_python_file_and_terminal(tmp_path):
    code = (
        "assert await asyncio.gather(self.child.work(2), self.child.work(4)) == [4, 8]\n"
        "assert await self.child.generated(5) == 10\n"
        "assert self.child.untraced() == 6\n"
        "try:\n"
        "    await self.child.fail()\n"
        "except ValueError:\n"
        "    pass\n"
        "await self.shell.write_file('tree.txt', 'tree checked')\n"
        "result = await self.shell.run('cat tree.txt')\n"
        "assert result.returncode == 0\n"
        "self.message('Nested execution verified.')\n"
        "return_result(RespondReason.DONE, explanation='verified tree')"
    )
    agent = TreeCodingAgent(
        cwd=tmp_path, llm=FakeLLMClient.with_tool_call("execute_python", {"code": code})
    )
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "tree", execution_tree=True)
    dispatcher = InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)
    existing = ExistingHooks()
    try:
        with hooks_scope(existing):
            assert await dispatcher.submit("exercise tree") is not None
            assert get_hooks() is existing
        await bridge.flush()
        starts = [update for update in client.updates if isinstance(update, ToolCallStart)]
        updates = [update for update in client.updates if isinstance(update, ToolCallProgress)]
        nodes = {update.tool_call_id: metadata(update) for update in starts}
        assert len({node["runId"] for node in nodes.values()}) == 1
        assert all(
            node["parentSpanId"] is None or node["parentSpanId"] in nodes for node in nodes.values()
        )
        assert all("endedAtMs" not in node for node in nodes.values() if node["nodeType"] != "file")
        assert {node["nodeType"] for node in nodes.values()} == {
            "method",
            "python",
            "file",
            "terminal",
        }
        assert not any(node["name"] == "TreeChild.untraced" for node in nodes.values())
        workers = [node for node in nodes.values() if node["name"] == "TreeChild.work"]
        assert len(workers) == 2
        assert workers[0]["parentSpanId"] == workers[1]["parentSpanId"]
        assert nodes[workers[0]["parentSpanId"]]["nodeType"] == "python"
        for worker in workers:
            assert any(
                node["name"] == "TreeChild.double" and node["parentSpanId"] == worker["spanId"]
                for node in nodes.values()
            )
        generated = next(node for node in nodes.values() if node["name"] == "TreeChild.generated")
        inner_python = next(
            node
            for node in nodes.values()
            if node["nodeType"] == "python" and node["parentSpanId"] == generated["spanId"]
        )
        assert any(
            node["name"] == "TreeChild.double" and node["parentSpanId"] == inner_python["spanId"]
            for node in nodes.values()
        )
        for update in updates:
            node = metadata(update)
            original = nodes[update.tool_call_id]
            assert all(node[key] == value for key, value in original.items())
            if update.status in {"completed", "failed"}:
                assert node["endedAtMs"] >= node["startedAtMs"]
        failure = next(node for node in nodes.values() if node["name"] == "TreeChild.fail")
        assert any(
            update.tool_call_id == failure["spanId"] and update.status == "failed"
            for update in updates
        )
        assert "before_agent_call" in existing.calls and "after_agent_call" in existing.calls
        assert bridge.execution_tree is not None and not bridge.execution_tree._nodes
    finally:
        await bridge.close()
        await dispatcher.close()


async def test_tree_cancellation_finishes_method_nodes_and_next_turn_has_new_run(tmp_path):
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="same-provider-id",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ],
                finish_reason="tool_calls",
            )
            for code in (
                "await self.child.wait()",
                "self.message('Ready again.')\nreturn_result(RespondReason.DONE, explanation='ready')",
            )
        ]
    )
    agent = TreeCodingAgent(cwd=tmp_path, llm=llm)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "cancel", execution_tree=True)
    dispatcher = InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)
    try:
        task = asyncio.create_task(dispatcher.submit("wait"))
        await asyncio.wait_for(client.wait_started.wait(), timeout=10)
        assert await dispatcher.cancel()
        assert await task is None
        await bridge.fail_open_tools("Cancelled by user.")
        await bridge.flush()
        starts = [update for update in client.updates if isinstance(update, ToolCallStart)]
        first_run = metadata(starts[0])["runId"]
        ends = {
            update.tool_call_id: update
            for update in client.updates
            if isinstance(update, ToolCallProgress) and update.status in {"failed", "completed"}
        }
        assert all(update.tool_call_id in ends for update in starts)
        waiting = next(update for update in starts if update.title == "TreeChild.wait")
        assert ends[waiting.tool_call_id].status == "failed"
        assert metadata(ends[waiting.tool_call_id])["endedAtMs"] >= metadata(waiting)["startedAtMs"]
        before = len(client.updates)
        assert await dispatcher.submit("try again") is not None
        await bridge.flush()
        second_starts = [
            update for update in client.updates[before:] if isinstance(update, ToolCallStart)
        ]
        assert second_starts and all(
            metadata(update)["runId"] != first_run for update in second_starts
        )
        assert bridge.execution_tree is not None and not bridge.execution_tree._nodes
    finally:
        await bridge.close()
        await dispatcher.close()


async def test_concurrent_sessions_do_not_mix_shared_child_activity(tmp_path):
    code = (
        "assert await self.child.work(3) == 6\n"
        "return_result(RespondReason.DONE, explanation='concurrent child complete')"
    )
    agents = [
        TreeCodingAgent(
            cwd=tmp_path, llm=FakeLLMClient.with_tool_call("execute_python", {"code": code})
        )
        for _ in range(2)
    ]
    agents[1].child = agents[0].child
    clients = [RecordingClient(), RecordingClient()]
    bridges = [
        ACPEventBridge(agent, cast(Client, client), f"session-{index}", execution_tree=True)
        for index, (agent, client) in enumerate(zip(agents, clients, strict=True))
    ]
    dispatchers = [
        InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)
        for agent, bridge in zip(agents, bridges, strict=True)
    ]
    try:
        await asyncio.gather(*(dispatcher.submit("run") for dispatcher in dispatchers))
        await asyncio.gather(*(bridge.flush() for bridge in bridges))
        run_ids: list[str] = []
        all_span_ids: list[set[str]] = []
        for client in clients:
            starts = [update for update in client.updates if isinstance(update, ToolCallStart)]
            nodes = [metadata(update) for update in starts]
            runs = {node["runId"] for node in nodes}
            assert len(runs) == 1
            run_ids.append(runs.pop())
            span_ids = {node["spanId"] for node in nodes}
            all_span_ids.append(span_ids)
            assert all(
                node["parentSpanId"] is None or node["parentSpanId"] in span_ids for node in nodes
            )
            assert sum(node["name"] == "TreeChild.work" for node in nodes) == 1
        assert run_ids[0] != run_ids[1]
        assert all_span_ids[0].isdisjoint(all_span_ids[1])
    finally:
        await asyncio.gather(*(bridge.close() for bridge in bridges))
        await asyncio.gather(*(dispatcher.close() for dispatcher in dispatchers))


async def test_generator_resumes_in_another_thread_without_reparenting_consumer(tmp_path):
    agent = TreeCodingAgent(cwd=tmp_path, llm=FakeLLMClient())
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "generator", execution_tree=True)
    tree = bridge.execution_tree
    assert tree is not None
    stream = agent.child.stream()
    try:
        with tree.turn():
            assert next(stream) == 2
        with tree.turn():
            assert await asyncio.to_thread(next, stream) == 4
            assert agent.child.double(9) == 18
            stream.close()
        await bridge.flush()
        starts = [update for update in client.updates if isinstance(update, ToolCallStart)]
        nodes = [metadata(update) for update in starts]
        generator = next(node for node in nodes if node["name"] == "TreeChild.stream")
        children = [node for node in nodes if node["parentSpanId"] == generator["spanId"]]
        assert len(children) == 2
        assert all(node["runId"] == generator["runId"] for node in children)
        consumer = next(
            node
            for node in nodes
            if node["name"] == "TreeChild.double" and node["parentSpanId"] is None
        )
        assert consumer["runId"] != generator["runId"]
        assert not tree._nodes
    finally:
        stream.close()
        await bridge.close()
        await agent.close()


async def test_synchronous_siblings_keep_order_within_one_millisecond(tmp_path, monkeypatch):
    # Simulate two real method calls finishing within one millisecond. A
    # persisted client may return cards out of order and sort by their starts.
    ticks = count(1_800_000_000_000_000_000, 10_000)
    monkeypatch.setattr(
        "nooa_acp.execution_tree.time", SimpleNamespace(time_ns=lambda: next(ticks))
    )
    agent = TreeCodingAgent(cwd=tmp_path, llm=FakeLLMClient())
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "ordering", execution_tree=True)
    tree = bridge.execution_tree
    assert tree is not None
    try:
        with tree.turn():
            assert agent.child.double(1) == 2
            assert agent.child.double(2) == 4
        await bridge.flush()
        starts = [
            metadata(update) for update in client.updates if isinstance(update, ToolCallStart)
        ]
        assert len(starts) == 2
        assert int(starts[0]["startedAtMs"]) == int(starts[1]["startedAtMs"])
        assert starts[0]["startedAtMs"] < starts[1]["startedAtMs"]
        persisted = starts[::-1]
        reloaded = sorted(persisted, key=lambda node: node["startedAtMs"])
        assert [node["spanId"] for node in reloaded] == [node["spanId"] for node in starts]
        completions = [
            metadata(update) for update in client.updates if isinstance(update, ToolCallProgress)
        ]
        assert starts[0]["startedAtMs"] < completions[0]["endedAtMs"] < starts[1]["startedAtMs"]
    finally:
        await bridge.close()
        await agent.close()


class TreeExperimentalAgent(ExperimentalCodingAgent):
    child: TreeChild

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = TreeChild(llm=FakeLLMClient())


def _cell_response(code: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(
                id="reused-provider-id", name="python_cell", arguments=json.dumps({"code": code})
            )
        ],
        finish_reason="tool_calls",
    )


def _completed_cell() -> LLMResponse:
    return _cell_response(
        "assert await self.child.work(4) == 8\n"
        "return_result(RespondResult(kind='DONE', explanation='done'))"
    )


async def test_reused_runner_captures_each_prompt_hooks_and_run(tmp_path):
    agent = TreeExperimentalAgent(
        cwd=tmp_path, llm=FakeLLMClient(scripted_responses=[_completed_cell(), _completed_cell()])
    )
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "reuse", execution_tree=True)
    dispatcher = InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)
    hooks = [ExistingHooks(), ExistingHooks()]
    runs = []
    checkpoint_results = []

    async def checkpoint(current, result):
        checkpoint_results.append(result)

    dispatcher.runtime.set_dispatch_hooks(on_after_handle=checkpoint)
    try:
        for index, observer in enumerate(hooks):
            before = len(client.updates)
            previous_calls = len(hooks[0].calls)
            with hooks_scope(observer):
                assert await dispatcher.submit(f"request {index}") is not None
                assert get_hooks() is observer
            await bridge.flush()
            if index == 0:
                persistent_task = dispatcher.runtime.task
            else:
                assert dispatcher.runtime.task is persistent_task
                assert len(hooks[0].calls) == previous_calls
            assert "before_agent_call" in observer.calls
            starts = [u for u in client.updates[before:] if isinstance(u, ToolCallStart)]
            nodes = {u.tool_call_id: metadata(u) for u in starts}
            assert nodes
            request_runs = {node["runId"] for node in nodes.values()}
            assert len(request_runs) == 1
            runs.append(request_runs.pop())
            assert any(
                node["name"] == "TreeExperimentalAgent.handle" and node["parentSpanId"] is None
                for node in nodes.values()
            )
            assert all(
                node["parentSpanId"] is None or node["parentSpanId"] in nodes
                for node in nodes.values()
            )
            assert not bridge.execution_tree._nodes
        assert runs[0] != runs[1]
        assert len(checkpoint_results) == 2
    finally:
        await bridge.close()
        await dispatcher.close()


async def test_slash_invocation_and_continuation_share_one_fresh_run(tmp_path):
    agent = TreeExperimentalAgent(
        cwd=tmp_path, llm=FakeLLMClient(scripted_responses=[_completed_cell(), _completed_cell()])
    )
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "slash", execution_tree=True)
    dispatcher = InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)

    class Commands:
        async def invoke(self, name, raw_args):
            await agent.child.work(2)
            return SlashCommandResult(
                name, raw_args, text="Continue with this result", output_to_agent=name == "continue"
            )

        def get(self, name):
            return SimpleNamespace(_method=self.invoke)

    try:
        await dispatcher.submit("initial prompt")
        await bridge.flush()
        runs = {metadata(u)["runId"] for u in client.updates if isinstance(u, ToolCallStart)}
        persistent_task = dispatcher.runtime.task
        for name in ("show", "continue"):
            before = len(client.updates)
            result = await dispatcher.invoke_slash(Commands(), name, "two words")
            assert result is not None
            assert (result[1] is not None) == (name == "continue")
            await bridge.flush()
            starts = [u for u in client.updates[before:] if isinstance(u, ToolCallStart)]
            nodes = {u.tool_call_id: metadata(u) for u in starts}
            current_runs = {node["runId"] for node in nodes.values()}
            assert len(current_runs) == 1 and runs.isdisjoint(current_runs)
            runs.update(current_runs)
            assert any(node["name"] == "TreeChild.work" for node in nodes.values())
            assert any(
                node["name"] == "TreeExperimentalAgent.handle" for node in nodes.values()
            ) == (name == "continue")
            assert all(
                node["parentSpanId"] is None or node["parentSpanId"] in nodes
                for node in nodes.values()
            )
            assert dispatcher.runtime.task is persistent_task
            assert not bridge.execution_tree._nodes
    finally:
        await bridge.close()
        await dispatcher.close()


async def test_default_agent_cancellation_and_next_prompt_have_separate_runs(tmp_path):
    agent = TreeExperimentalAgent(
        cwd=tmp_path,
        llm=FakeLLMClient(
            scripted_responses=[_cell_response("await self.child.wait()"), _completed_cell()]
        ),
    )
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "cancel-default", execution_tree=True)
    dispatcher = InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)
    try:
        task = asyncio.create_task(dispatcher.submit("wait"))
        await asyncio.wait_for(client.wait_started.wait(), 5)
        assert await dispatcher.cancel()
        assert await task is None
        await bridge.fail_open_tools("Cancelled by user.")
        await bridge.flush()
        first_starts = [u for u in client.updates if isinstance(u, ToolCallStart)]
        first_runs = {metadata(u)["runId"] for u in first_starts}
        endings = {
            u.tool_call_id
            for u in client.updates
            if isinstance(u, ToolCallProgress) and u.status in {"failed", "completed"}
        }
        assert all(u.tool_call_id in endings for u in first_starts)
        before = len(client.updates)
        assert await dispatcher.submit("recover") is not None
        await bridge.flush()
        next_runs = {
            metadata(u)["runId"] for u in client.updates[before:] if isinstance(u, ToolCallStart)
        }
        assert len(next_runs) == 1 and next_runs.isdisjoint(first_runs)
        assert not bridge.execution_tree._nodes
    finally:
        await bridge.close()
        await dispatcher.close()


async def test_wait_continuation_keeps_foreground_run(tmp_path):
    first = _cell_response(
        "self.queue_manager.get_channel('system_messages').put('ready')\n"
        "return_result(RespondResult(kind='WAIT', explanation='waiting for result'))"
    )
    agent = TreeExperimentalAgent(
        cwd=tmp_path, llm=FakeLLMClient(scripted_responses=[first, _completed_cell()])
    )
    client = RecordingClient()
    bridge = ACPEventBridge(agent, cast(Client, client), "wait", execution_tree=True)
    dispatcher = InteractiveSessionDispatcher(agent, execution_tree=bridge.execution_tree)
    try:
        assert await dispatcher.submit("start") is not None
        await bridge.flush()
        nodes = [metadata(u) for u in client.updates if isinstance(u, ToolCallStart)]
        assert sum(node["name"] == "TreeExperimentalAgent.handle" for node in nodes) == 2
        assert len({node["runId"] for node in nodes}) == 1
        assert not bridge.execution_tree._nodes
    finally:
        await bridge.close()
        await dispatcher.close()
