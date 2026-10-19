# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed CodeActV2 summary fork cache matrix, separate from the four-cell test.

NOOA_RUN_AGENT_SUMMARY_CACHE_LIVE=1 NVIDIA_INFERENCE_API_KEY=... uv run pytest \
    -m integration -s tests/integration/test_agent_summary_cache_live.py

Exactly three capped POSTs/model: parent, background fork, safe-boundary parent.
No live calls run by default. Requests stay in memory; never log bodies, headers,
keys, native state or hashes. Diagnostics contain only normalized usage/counts.
"""

from __future__ import annotations

from nooa import Agent, Context, hidden, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact_v2 import CodeActV2

with hidden:
    import ast
    import asyncio
    import json
    import os
    import uuid

    import httpx
    import pytest

    from nooa.agents import TokenBudgetSummarizer
    from nooa.agents.summarization import _in_summary_fork
    from nooa.config.summarizer_config import TokenBudgetConfig
    from nooa.context_blocks.events import UserEvent
    from nooa.events import Summary
    from nooa.llm_types import LLMResponse
    from nooa.unifiedllm import CacheBoundary
    from nooa.unifiedllm.unifiedllm import _ClientHttp
    from tests.integration.test_agent_cache_live import (
        CASES,
        FIXED_PREFIX,
        _make_client,
        _mock_reply,
        _no_auto_trace_export,  # noqa: F401 -- imported autouse safety fixture
        _strings,
    )


class SummaryCacheAgent(Agent):
    """Return only the requested constant; ignore inert reference padding."""

    @strategy(CodeActV2(config=CodeActConfig(prefill=None, max_iterations=1, max_retries=1)))
    async def answer(self) -> int:
        """Make exactly one python_cell call containing only return_result(3).

        Do not inspect self, use other tools, or do any other work.
        """
        ...


@hidden
def _validate_parent(response):
    assert response.finish_reason == "tool_calls", "parent response incomplete"
    assert len(response.tool_calls) == 1, "expected one safe parent cell"
    tool = response.tool_calls[0]
    assert tool.name == "python_cell", "unexpected parent tool"
    try:
        code = json.loads(tool.arguments)["code"]
        valid = ast.dump(ast.parse(code)) == ast.dump(ast.parse("return_result(3)"))
    except (ValueError, TypeError, KeyError, SyntaxError):
        valid = False
    assert valid, "unsafe parent cell rejected before execution"


@hidden
def _require_fork_cache(parent, fork):
    assert parent is not None and fork is not None, "missing provider usage"
    # Normalized input already includes cache reads/writes; do not add them again.
    assert parent.input_tokens >= 8192, "fixed prefix was too small or evicted"
    assert fork.cached_input_tokens > parent.input_tokens / 2, (
        "summary fork cache read must exceed half the parent input tokens"
    )


@hidden
def _marker_paths(value, path=()):
    """Locate projected checkpoints without recording or printing their content."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"cache_control", "prompt_cache_breakpoint"}:
                yield path + (key,)
            else:
                yield from _marker_paths(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _marker_paths(child, path + (index,))


@hidden
def _assert_projected_fork(case, parent, fork):
    key = "input" if case.responses else "messages"
    settings_equal = {k: v for k, v in parent.items() if k != key} == {
        k: v for k, v in fork.items() if k != key
    }
    assert settings_equal, "fork changed tools/cache key/model/settings on wire"
    summary = fork[key][-1]
    if case.name == "opus55":
        # Anthropic coalesces adjacent users. Only the final summary text block
        # is new; retain all historical blocks in that same user container.
        assert summary["content"][-1]["type"] == "text"
        text = summary["content"][-1]["text"]
        prefix = fork[key][:-1] + [{**summary, "content": summary["content"][:-1]}]
        assert not list(_marker_paths(summary["content"][-1])), "summary was cache marked"
    else:
        content = summary["content"]
        text = content[0]["text"] if case.responses else content
        prefix = fork[key][:-1]
        assert not list(_marker_paths(summary)), "summary was cache marked"
    assert "Background memory compaction:" in text, "summary instruction missing on wire"
    equal = prefix == parent[key]
    assert equal, "projection changed the exact sent parent prefix/checkpoints"
    parent_markers = set(_marker_paths(parent))
    assert parent_markers <= set(_marker_paths(fork)), "historical checkpoints were dropped"
    if case.responses:
        assert len(parent_markers) >= 3, "historical Responses checkpoints missing"
    elif case.name == "opus55":
        assert parent_markers, "Anthropic cache endpoint missing"


@hidden
async def _run_summary_agent(case, client, monkeypatch, *, mock_usage=None):
    """Use the installed runtime fork, shared client and documented llm_call hook."""
    reference = f"Summary cache isolation nonce: {uuid.uuid4().hex}\n{FIXED_PREFIX}"
    agent = SummaryCacheAgent(llm=client)
    agent.context["cache_reference"] = Context(reference, prefix=True)
    agent._summary_phase = 0
    agent.context["live"] = Context(expr="self._summary_phase")
    old = [
        agent.event_manager.add(UserEvent(content=f"Old decision {i}: retain number {i}."))
        for i in range(4)
    ]
    started, release = asyncio.Event(), asyncio.Event()
    records, bodies = [], []
    parent_request = None
    real_send = httpx.AsyncClient.send

    async def capture(http_client, request, *args, **kwargs):
        if request.method == "POST":
            assert len(bodies) < 3, "unexpected retry or extra model POST"
            assert request.url.host == "inference-api.nvidia.com", "unexpected provider host"
            body = json.loads(request.content)
            cap = "max_output_tokens" if case.responses else "max_tokens"
            assert body[cap] == 4096, "missing bounded output"
            phase = 1 if len(bodies) == 2 else 0
            live = f'<live expr="self._summary_phase">\n{phase}\n</live>'
            strings = list(_strings(body))
            assert any(reference in s for s in strings), "fixed reference changed on wire"
            assert any(live in s for s in strings), "fork lost captured live context"
            bodies.append(body)
        return await real_send(http_client, request, *args, **kwargs)

    async def observe(ctx, nxt):
        nonlocal parent_request
        is_fork = _in_summary_fork.get()
        assert ctx.client is client, "fork did not use original shared client"
        assert not ctx.filtered_history
        boundaries = [i for i, m in enumerate(ctx.messages) if isinstance(m, CacheBoundary)]
        assert len(boundaries) == 1, "expected one runtime cache boundary"
        assert any(reference in s for s in _strings(ctx.messages)), "fixed nonce prefix changed"
        if is_fork:
            assert parent_request is not None, "fork ran without a parent request"
            messages, params, runtime = parent_request
            equal = ctx.messages[:-1] == messages
            assert equal, "fork did not reuse the actual sent parent request"
            expected_params = {**params, "output_model": None}
            equal = ctx.params == expected_params
            assert equal, "fork changed parent tools/key/settings"
            assert ctx.runtime is runtime, "fork changed policy runtime"
            assert boundaries[0] < len(ctx.messages) - 1
            assert "Background memory compaction:" in ctx.messages[-1]["content"]
            started.set()
            await release.wait()
        if not is_fork and parent_request is None:
            # Keep params live so the blocking fork sees dispatch's cache key.
            parent_request = (list(ctx.messages), ctx.params, ctx.runtime)
        before = [(tag, event.id) for tag, event in agent.event_manager.items()]
        result = await nxt(ctx)
        if is_fork:
            assert [(tag, event.id) for tag, event in agent.event_manager.items()] == before, (
                "fork executed tools or wrote parent events"
            )
        response = result.response
        assert isinstance(response, LLMResponse)
        if mock_usage:
            usage = response.usage
            assert usage is not None, "missing normalized mock usage"
            assert usage.input_tokens == 14_000
            assert usage.output_tokens == 40
            assert usage.total_tokens == 14_040
            expected_cached = 12_000 if is_fork or len(records) > 1 else 0
            if is_fork and mock_usage == "cache":
                expected_cached = 0
            assert usage.cached_input_tokens == expected_cached
            assert usage.cache_write_input_tokens == 0
        records.append((is_fork, response.usage))
        print(
            json.dumps(
                {
                    "model": case.model,
                    "summary_fork": is_fork,
                    "call": len(records),
                    "usage": response.usage.model_dump() if response.usage else None,
                }
            ),
            flush=True,
        )
        if not is_fork:
            _validate_parent(response)  # Validate before CodeActV2 executes anything.
        return result

    monkeypatch.setattr(httpx.AsyncClient, "send", capture)
    agent.event_manager.intercept("llm_call", observe)
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1, target_chars=600)
    )
    parent_call = asyncio.create_task(agent.answer())
    started_wait = asyncio.create_task(started.wait())
    try:
        await asyncio.wait_for(
            asyncio.wait({parent_call, started_wait}, return_when=asyncio.FIRST_COMPLETED), 200
        )
        task = summarizer._pending_task
        assert task is not None, "installed threshold summarizer did not fork"
        assert started.is_set()
        assert not task.done()
        assert not parent_call.done(), "parent advanced before summary completed"
        source = dict(summarizer._pending_source)
        assert set(old) <= source.keys(), "fork did not select old events"
        before = [(tag, event.id) for tag, event in agent.event_manager.items()]
        recent = {tag: identity for tag, identity in before if tag not in source}
        assert not any(isinstance(e, Summary) for e in agent.event_manager.values())
        agent._summary_phase = 1  # Fork must keep the already-sent phase 0.
        release.set()
        assert await asyncio.wait_for(parent_call, 200) == 3, "fork changed parent result"
        _assert_projected_fork(case, *bodies[:2])
        assert [fork for fork, _ in records] == [True, False]
        _require_fork_cache(records[1][1], records[0][1])
        text = summarizer._pending_summary
        assert isinstance(text, str) and text.strip(), "fork produced no usable pending summary"
        assert all(tag in agent.event_manager.keys() for tag in old), "summary applied too early"
        summarizer.config = summarizer.config.model_copy(update={"max_tokens": 1_000_000_000})
        # A new user task after compaction keeps summary instructions in memory
        # from being mistaken for the current task. Do not alter tool_choice.
        reminder = UserEvent(
            content="Continue the original task: make exactly one python_cell call "
            "containing only return_result(3). Do not answer in plain text."
        )
        reminder_tag = agent.event_manager.add(reminder)
        recent[reminder_tag] = reminder.id
        assert await asyncio.wait_for(agent.answer(), 200) == 3
        summaries = [e for e in agent.event_manager.values() if isinstance(e, Summary)]
        assert len(summaries) == 1, "summary did not apply at real BeforeTurn boundary"
        assert summaries[0].summary_text == text
        assert set(old) <= set(summaries[0].children_tags)
        assert not set(old) & set(agent.event_manager.keys()), "old events did not collapse"
        assert all(agent.event_manager[tag].id == identity for tag, identity in recent.items())
        assert summarizer._pending_task is summarizer._pending_summary is None
        assert len(bodies) == len(records) == 3, "expected three calls per model"
        assert [fork for fork, _ in records] == [True, False, False]
        continuation = list(_strings(bodies[2]))
        assert any(text in s for s in continuation), "summary text missing on wire"
        # Old facts (and even quoted event markers) may legitimately appear in
        # the summary. Only separately rendered original events are forbidden.
        outside_summary = [s.replace(text, "") for s in continuation]
        for tag, identity in before:
            if tag in old:
                assert not any(f'tag="{tag}"' in s or identity in s for s in outside_summary), (
                    "collapsed event identity still separately rendered"
                )
        assert any(reminder.content in s for s in continuation), "latest task missing on wire"
        assert any("return_result(3)" in s.replace(reminder.content, "") for s in continuation), (
            "recent parent work missing"
        )
        return reference
    finally:
        release.set()
        await agent.aclose()
        parent_call.cancel()
        started_wait.cancel()
        await asyncio.gather(parent_call, started_wait, return_exceptions=True)
        bodies.clear()  # Never retain native requests as artifacts.


@pytest.mark.integration
@pytest.mark.timeout(700)
@pytest.mark.skipif(
    os.getenv("NOOA_RUN_AGENT_SUMMARY_CACHE_LIVE") != "1",
    reason="set NOOA_RUN_AGENT_SUMMARY_CACHE_LIVE=1 to spend inference tokens",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_agent_summary_cache_live(case, monkeypatch):
    key = os.environ.get("NVIDIA_INFERENCE_API_KEY")
    assert key, "NVIDIA_INFERENCE_API_KEY required for opted-in live tests"
    async with _make_client(case, key) as client:
        await _run_summary_agent(case, client, monkeypatch)


@hidden
def _summary_reply(case, index, broken):
    reply = _mock_reply(case, index)
    code = json.dumps({"code": "return_result(3)"})
    # Regression: legitimate summary prose must not trigger event-collapse checks.
    summary = "Summary: Old decision 0 and Old decision 1 retain numbers 0, 1, 2 and 3."
    fork = index == 1
    if case.responses:
        tool = reply["output"][-1]
        tool["arguments"] = code
        if fork and broken != "tool":
            reply["output"][-1] = {
                "type": "message",
                "id": "summary",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": summary, "annotations": []}],
            }
        if fork and broken == "cache":
            reply["usage"]["input_tokens_details"]["cached_tokens"] = 0
    elif case.name == "opus55":
        reply["content"][-1]["input"] = json.loads(code)
        if fork and broken != "tool":
            reply["content"][-1] = {"type": "text", "text": summary}
            reply["stop_reason"] = "end_turn"
        if fork and broken == "cache":
            reply["usage"].update(input_tokens=14_000, cache_read_input_tokens=0)
    else:
        choice = reply["choices"][0]
        choice["message"]["tool_calls"][0]["function"]["arguments"] = code
        if fork and broken != "tool":
            choice["message"].pop("tool_calls")
            choice["message"]["content"] = summary
            choice["finish_reason"] = "stop"
        if fork and broken == "cache":
            reply["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    return reply


@hidden
async def _mock_summary_run(case, monkeypatch, broken=None):
    count = 0

    def no_network(*args, **kwargs):
        raise AssertionError("offline summary test escaped mock HTTP transport")

    def respond(request):
        nonlocal count
        index = count
        count += 1
        assert count <= 3, "unexpected extra mock call"
        return httpx.Response(200, json=_summary_reply(case, index, broken))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    async with _make_client(case, "offline-dummy-key") as client:
        result = await _run_summary_agent(case, client, monkeypatch, mock_usage=broken or "valid")
    assert count == 3
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_summary_cache_matrix_mock_http(case, monkeypatch):
    await _mock_summary_run(case, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("broken", ["cache", "tool"])
async def test_summary_fork_negative_controls(case, monkeypatch, broken):
    message = "exceed half" if broken == "cache" else "no usable pending summary"
    with pytest.raises(AssertionError, match=message):
        await _mock_summary_run(case, monkeypatch, broken)


@pytest.mark.asyncio
async def test_summary_reference_unique_per_run(monkeypatch):
    with monkeypatch.context() as scoped:
        first = await _mock_summary_run(CASES[0], scoped)
    with monkeypatch.context() as scoped:
        second = await _mock_summary_run(CASES[0], scoped)
    assert first != second, "separate runs reused a warmed prefix"


@pytest.mark.parametrize("code", ["return_result(3); import os", "print(self._llm.config)"])
def test_summary_parent_rejects_unsafe_code(code):
    from nooa.llm_types import ToolCall

    response = LLMResponse(
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="unsafe", name="python_cell", arguments=json.dumps({"code": code}))
        ],
    )
    with pytest.raises(AssertionError, match="unsafe parent cell"):
        _validate_parent(response)
