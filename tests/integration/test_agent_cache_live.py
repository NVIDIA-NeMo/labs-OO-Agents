# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CodeActV2 cache matrix, opt-in; local HTTP/negative controls stay offline.

NOOA_RUN_AGENT_CACHE_LIVE=1 NVIDIA_INFERENCE_API_KEY=... uv run pytest \
    -m integration -s tests/integration/test_agent_cache_live.py

Live tests use a pytest-scoped, forwarding HTTP observer: only model/tools/static
system/instructions and history are compared in memory, ignoring cache markers
and removing the prior live suffix. No request body/header logging, raw payload
artifacts, or native hashes. Prefix diagnostics print only stability and POST
count; existing output is allowlisted route configuration and normalized usage. SQLite archives are temporary,
may contain provider-native state, and must not be published as test artifacts.
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
    from dataclasses import dataclass
    from types import SimpleNamespace

    import httpx
    import litellm
    import pytest

    from nooa.events import PythonOutput
    from nooa.llm_types import LLMResponse, LLMUsage
    from nooa.storage import SQLiteStorageManager
    from nooa.unifiedllm import CacheBoundary, CompletionClient, ResponsesClient
    from nooa.unifiedllm.http_config import HttpConfig
    from nooa.unifiedllm.retry_config import RetryConfig
    from nooa.unifiedllm.unifiedllm import _ClientHttp, _extract_usage


@hidden
@dataclass(frozen=True)
class CacheCase:
    name: str
    model: str
    api_base: str
    responses: bool = False


with hidden:
    CASES = (
        CacheCase(
            "gpt61-sol",
            "openai/azure/openai/gpt-6.1-sol",
            "https://inference-api.nvidia.com/v1",
            True,
        ),
        CacheCase(
            "opus55",
            "anthropic/azure/anthropic/claude-opus-5-5",
            "https://inference-api.nvidia.com",
        ),
        CacheCase("glm53", "openai/nvidia/zai-org/glm-5.3", "https://inference-api.nvidia.com/v1"),
        CacheCase(
            "kimi-k3", "openai/nvidia/moonshotai/kimi-k3", "https://inference-api.nvidia.com/v1"
        ),
    )
    # Approximately 12K tokens, not a tokenizer benchmark. Each run prepends a
    # nonce BEFORE this padding to avoid reusing a previous run's warmed prefix.
    # The resulting reference is immutable across all turns of that run.
    FIXED_PREFIX = (
        "The following reference padding is inert. Ignore it; execute the task.\n"
        + " amber" * 12_288
    )
    CODES = (
        "x = 1; self._live_state = x; print(x)",
        "x += 1; self._live_state = x; print(x)",
        "x += 1; self._live_state = x; print(x)",
        "return_result(x)",
    )


class CacheAgent(Agent):
    """Execute the requested Python cells; do not inspect inert reference padding."""

    def __init__(self, *, llm, reference):
        super().__init__(llm=llm)
        self._live_state = 0
        self.context["cache_reference"] = Context(reference, prefix=True)
        self.context["live_state"] = Context(expr="self._live_state")

    @strategy(CodeActV2(config=CodeActConfig(prefill=None, max_iterations=8, max_retries=1)))
    async def count(self) -> int:
        """Make exactly FOUR sequential python_cell tool calls, one per response.

        First cell: x = 1; self._live_state = x; print(x)
        Second cell: x += 1; self._live_state = x; print(x)
        Third cell: x += 1; self._live_state = x; print(x)
        Fourth cell: return_result(x)

        x survives between cells. Do not combine cells, loop, add extra work,
        inspect the environment, or return early. Return the integer 3.
        """
        ...


@hidden
def _make_client(case, api_key):
    config = {
        "model": case.model,
        "api_base": case.api_base,
        "api_key": api_key,
        "http_config": HttpConfig(read_timeout=180),
        "timeout": 180,
        "num_retries": 0,
        "retry_config": RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        "cache_breakpoint": "auto",
    }
    if case.responses:
        return ResponsesClient(
            **config,
            max_output_tokens=4096,
            reasoning={"effort": "low"},
            include=["reasoning.encrypted_content"],
            store=False,
        )
    if case.name == "opus55":
        return CompletionClient(
            **config,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
        )
    # GLM and Kimi deliberately receive no unverified reasoning-effort extras.
    return CompletionClient(**config, max_tokens=4096)


@hidden
def _require_cache_hits(usages):
    assert len(usages) == 4, "expected exactly four real responses"
    # input_tokens is the normalized client total (including Anthropic cache
    # reads/writes via LiteLLM); never add cache counters to this baseline.
    assert all(usage is not None for usage in usages), "missing provider usage"
    assert usages[0].input_tokens >= 8192, "fixed prefix was too small or evicted"
    assert sum(usage.cached_input_tokens > 0 for usage in usages[1:]) >= 2, (
        "provider must report positive cache reads on at least two later calls"
    )
    assert usages[-1].cached_input_tokens > usages[0].input_tokens / 2, (
        "latest cache read must exceed half the initial input tokens"
    )


@hidden
def _validate_cell(arguments, phase):
    # Reject arbitrary generated code before execution; this process holds the
    # live API key. AST equality permits whitespace/comments, not extra work.
    try:
        code = json.loads(arguments)["code"]
        actual = ast.dump(ast.parse(code), include_attributes=False)
        expected = ast.dump(ast.parse(CODES[phase]), include_attributes=False)
    except (ValueError, TypeError, KeyError, SyntaxError, IndexError):
        raise AssertionError("invalid Python cell arguments") from None
    matches = actual == expected
    assert matches, "model cell deviated from the required safe arithmetic step"


@hidden
def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


@hidden
async def _run_agent(case, client, database):
    reference = f"Cache isolation nonce: {uuid.uuid4().hex}\n{FIXED_PREFIX}"
    agent = CacheAgent(llm=client, reference=reference)
    usages, states, replay_counts = [], [], []

    async def observe(ctx, nxt):
        assert not ctx.filtered_history, "unexpected history filtering"
        assert sum(isinstance(m, CacheBoundary) for m in ctx.messages) == 1
        public_strings = list(_strings([dict(m) for m in ctx.messages]))
        fixed = f"<cache_reference>\n{reference}\n</cache_reference>"
        assert any(fixed in text for text in public_strings), "per-agent fixed reference changed"
        live = f'<live_state expr="self._live_state">\n{agent._live_state}\n</live_state>'
        assert any(live in text for text in public_strings), "live Context was not refreshed"
        states.append(agent._live_state)
        turns = [(i, m) for i, m in enumerate(ctx.messages) if isinstance(m, LLMResponse)]
        assert len(turns) == len(usages), "prior native assistant turns missing"
        # Roundtrip only historical responses: the current completion has not
        # been enriched/stored by the runtime yet. Never flatten native replay.
        with SQLiteStorageManager(database) as storage:
            for _, turn in turns:
                if storage.event_backend.get(turn.id) is None:
                    storage.event_backend.store(turn.id, turn)
        with SQLiteStorageManager(database) as storage:
            for i, turn in turns:
                restored = storage.event_backend.get(turn.id)
                assert isinstance(restored, LLMResponse)
                assert restored is not turn and restored.raw_response is None
                parts_equal = restored.parts == turn.parts
                scope_equal = restored.replay_scope == turn.replay_scope
                has_scope = bool(restored.replay_scope)
                assert parts_equal, "SQLite changed ordered/native parts"
                assert scope_equal, "SQLite changed replay scope"
                assert has_scope, "provider returned no native replay scope"
                assert restored.usage == turn.usage, "SQLite changed usage"
                ctx.messages[i] = restored
        replay_counts.append(len(turns))
        ctx = await nxt(ctx)
        response = ctx.response
        assert isinstance(response, LLMResponse)
        usages.append(response.usage)
        print(
            json.dumps(
                {
                    "model": case.model,
                    "api_base": case.api_base,
                    "client": "ResponsesClient" if case.responses else "CompletionClient",
                    "cache_breakpoint": "auto",
                    "max_output_tokens": 4096,
                    "call": len(usages),
                    "usage": response.usage.model_dump() if response.usage else None,
                }
            ),
            flush=True,
        )
        assert response.finish_reason == "tool_calls", "invalid or truncated model response"
        assert len(response.tool_calls) == 1, "expected one cell per model response"
        assert response.tool_calls[0].name == "python_cell", "unexpected tool"
        _validate_cell(response.tool_calls[0].arguments, len(usages) - 1)
        return ctx

    agent.event_manager.intercept("llm_call", observe)
    try:
        result = await asyncio.wait_for(agent.count(), timeout=8 * 180)
        assert result == 3, "agent did not return the computed integer 3"
        assert agent._live_state == 3, "cells did not update private live context"
        assert len(usages) == 4, "expected exactly four real responses"
        assert states == [0, 1, 2, 3], "expected exactly four sequential live phases"
        assert replay_counts == list(range(len(usages)))
        outputs = [e for e in agent.event_manager.all_events() if isinstance(e, PythonOutput)]
        assert not any(e.stderr for e in outputs), "Python cell failed"
        assert len(outputs) >= 3, "missing executed Python cells"
        printed = [e.stdout.strip() for e in outputs if e.stdout.strip()]
        assert all(str(n) in printed for n in (1, 2, 3)), "missing increment printouts"
        _require_cache_hits(usages)
        return usages
    finally:
        try:
            await agent.aclose()
        finally:
            # pytest retains tmp_path across successful runs. Remove native
            # archives and sidecars even on assertion/provider failures.
            for suffix in ("", "-wal", "-shm", "-journal"):
                database.with_name(database.name + suffix).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _no_auto_trace_export(monkeypatch):
    # These tests must not export prompts/native state to an ambient dev viewer.
    import nooa.agent as agent_module

    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", True)


@pytest.mark.integration
@pytest.mark.timeout(1500)
@pytest.mark.skipif(
    os.getenv("NOOA_RUN_AGENT_CACHE_LIVE") != "1",
    reason="set NOOA_RUN_AGENT_CACHE_LIVE=1 to spend inference tokens",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_agent_cache_live(case, tmp_path, _live_http_observer):
    # Opt-in without credentials is an error, never a false-positive skip.
    key = os.environ.get("NVIDIA_INFERENCE_API_KEY")
    assert key, "NVIDIA_INFERENCE_API_KEY is required for opted-in live tests"
    async with _make_client(case, key) as client:
        await _run_agent(case, client, tmp_path / "native-turns.db")
    _live_http_observer.require_complete()


@hidden
def _mock_reply(case, index):
    arguments = json.dumps({"code": CODES[index]})
    cached = 0 if index == 0 else 12_000
    if case.responses:
        return {
            "id": f"resp_{index}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": case.model.removeprefix("openai/"),
            "output": [
                {
                    "id": f"rs_{index}",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": f"synthetic-encrypted-{index}",
                },
                {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": f"call_{index}",
                    "name": "python_cell",
                    "arguments": arguments,
                    "status": "completed",
                },
            ],
            "parallel_tool_calls": False,
            "store": False,
            "tools": [],
            "usage": {
                "input_tokens": 14_000,
                "output_tokens": 40,
                "input_tokens_details": {"cached_tokens": cached},
            },
        }
    if case.name == "opus55":
        return {
            "id": f"msg_{index}",
            "type": "message",
            "role": "assistant",
            "model": case.model.removeprefix("anthropic/"),
            "content": [
                {
                    "type": "thinking",
                    "thinking": "synthetic thought",
                    "signature": f"synthetic-signature-{index}",
                },
                {
                    "type": "tool_use",
                    "id": f"call_{index}",
                    "name": "python_cell",
                    "input": json.loads(arguments),
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 14_000 - cached,
                "output_tokens": 40,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
            },
        }
    return {
        "id": f"chat_{index}",
        "object": "chat.completion",
        "created": 0,
        "model": case.model.removeprefix("openai/"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "synthetic thought",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "python_cell",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 14_000,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


@hidden
def _strip_cache_metadata(value):
    if isinstance(value, dict):
        return {
            key: _strip_cache_metadata(child)
            for key, child in value.items()
            if key not in {"cache_control", "prompt_cache_breakpoint"}
        }
    if isinstance(value, list):
        return [_strip_cache_metadata(child) for child in value]
    return value


@hidden
def _without_live_suffix(case, body, phase):
    """Copy stable history, retaining Anthropic's coalesced historical blocks."""
    key = "input" if case.responses else "messages"
    try:
        items = _strip_cache_metadata(body[key])
        terminal = items[-1]
        is_user = terminal.get("role") == "user"
        assert is_user, "live suffix was not a user message"
        content = terminal["content"]
        if case.responses:
            is_text = len(content) == 1 and content[0]["type"] == "input_text"
            assert is_text, "unexpected live suffix shape"
            text = content[0]["text"]
        elif case.name == "opus55":
            is_text = content[-1]["type"] == "text"
            assert is_text, "unexpected live suffix shape"
            text = content[-1]["text"]
        else:
            is_text = isinstance(content, str)
            assert is_text, "unexpected live suffix shape"
            text = content
        is_envelope = text.startswith("<context>\n") and text.endswith("\n</context>")
        assert is_envelope, "live suffix was not a context envelope"
        correct_phase = f'<live_state expr="self._live_state">\n{phase}\n</live_state>' in text
        assert correct_phase, "live suffix had an unexpected phase"
        if case.name == "opus55":
            content.pop()
            if not content:
                items.pop()
        else:
            items.pop()
        return items
    except (KeyError, IndexError, TypeError, AttributeError):
        raise AssertionError("malformed wire history/live suffix") from None


@hidden
def _assert_history_prefix_stable(case, previous, later, phase):
    previous, later = map(_strip_cache_metadata, (previous, later))
    for key in ("model", "instructions", "system", "tools"):
        equal = previous.get(key) == later.get(key)
        assert equal, f"stable {key} changed on wire"
    key = "input" if case.responses else "messages"
    items = _without_live_suffix(case, previous, phase)
    assert len(later[key]) > len(items), "wire history did not grow"
    equal = later[key][: len(items)] == items
    assert equal, "historical content/shape changed on wire"


@hidden
@dataclass
class _WirePrefixObserver:
    """Keep only allowlisted prefix/history in memory; never log request data."""

    case: CacheCase
    previous: dict | None = None
    call_count: int = 0
    prefix_stable: bool = True

    def observe(self, request):
        if request.method != "POST":
            return
        self.call_count += 1
        try:
            assert self.call_count <= 4, "unexpected retry or extra model POST"
            try:
                body = json.loads(request.content)
                key = "input" if self.case.responses else "messages"
                # Do not retain headers, URLs, generation parameters, or the
                # whole payload. Cache-marker differences are not prefix drift.
                current = _strip_cache_metadata(
                    {
                        k: body[k]
                        for k in ("model", "tools", "system", "instructions", key)
                        if k in body
                    }
                )
                del body
            except (ValueError, TypeError, KeyError, httpx.RequestNotRead):
                raise AssertionError("unreadable model POST") from None
            has_history = key in current
            assert has_history, "model POST is missing history"
            # Validate the current suffix too: the final POST has no later
            # continuation that could otherwise validate its live phase.
            _without_live_suffix(self.case, current, self.call_count - 1)
            if self.previous is not None:
                _assert_history_prefix_stable(
                    self.case, self.previous, current, self.call_count - 2
                )
            self.previous = current
        except AssertionError:
            self.prefix_stable = False
            raise

    def require_complete(self):
        assert self.call_count == 4, "expected exactly four outgoing model POSTs"
        assert self.prefix_stable, "wire prefix was unstable"


@hidden
def _install_wire_observer(case, monkeypatch):
    observer = _WirePrefixObserver(case)
    original_send = httpx.AsyncClient.send

    async def send(client, request, *args, **kwargs):
        observer.observe(request)
        # Forward the original request unchanged, including streaming options.
        return await original_send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return observer


@pytest.fixture
def _live_http_observer(case, monkeypatch):
    observer = _install_wire_observer(case, monkeypatch)
    try:
        yield observer
    finally:
        print(
            json.dumps(
                {"prefix_stable": observer.prefix_stable, "call_count": observer.call_count}
            ),
            flush=True,
        )
        # Release even the in-memory native history after failure or success.
        observer.previous = None
        # pytest's monkeypatch fixture restores AsyncClient.send at teardown.


@hidden
async def _check_mock_http_run(case, tmp_path, monkeypatch):
    """Same real client/SDK/runtime/middleware as live, with hermetic HTTP replies."""
    # Other modules set this process-global flag during collection. Keep the
    # offline wire contract independent without changing live configuration.
    monkeypatch.setattr(litellm, "drop_params", False)
    bodies = []
    observer = _install_wire_observer(case, monkeypatch)

    def no_network(*args, **kwargs):
        raise AssertionError("offline test escaped mock HTTP transport")

    def respond(request):
        body = json.loads(request.content)
        index = len(bodies)
        assert index < 4, "unexpected retry or extra model call"
        assert request.url.host == "inference-api.nvidia.com"
        bodies.append(body)
        return httpx.Response(200, json=_mock_reply(case, index))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    async with _make_client(case, "offline-dummy-key") as client:
        assert client.model == case.model
        assert client.cache_breakpoint == "auto"
        assert client.retry_config.max_retries == client.retry_config.rate_limit_extra_retries == 0
        assert client.config["num_retries"] == 0
        assert client._http_config.read_timeout == client.config["timeout"] == 180
        assert client.config["api_base"] == case.api_base
        assert isinstance(client, ResponsesClient) == case.responses
        usages = await _run_agent(case, client, tmp_path / "native-turns.db")
    observer.require_complete()
    assert [u.input_tokens for u in usages] == [14_000] * 4
    assert [u.cached_input_tokens for u in usages] == [0, 12_000, 12_000, 12_000]
    assert [u.cache_write_input_tokens for u in usages] == [0] * 4
    assert [u.total_tokens for u in usages] == [14_040] * 4
    assert not (tmp_path / "native-turns.db").exists(), "native archive was retained"
    assert len(bodies) == 4
    # The middleware checks the exact reference on every LLM call. Check it on
    # the serialized wire too, and return it for the cross-instance regression.
    reference = next(
        text.split("<cache_reference>\n", 1)[1].split("\n</cache_reference>", 1)[0]
        for text in _strings(bodies[0])
        if "<cache_reference>\n" in text
    )
    assert reference.startswith("Cache isolation nonce: ")
    assert reference.split("\n", 1)[1] == FIXED_PREFIX
    for body in bodies:
        assert any(
            f"<cache_reference>\n{reference}\n</cache_reference>" in text for text in _strings(body)
        ), "per-agent reference changed on wire"
    for phase, (previous, later) in enumerate(zip(bodies, bodies[1:], strict=False)):
        _assert_history_prefix_stable(case, previous, later, phase)
    cap = "max_output_tokens" if case.responses else "max_tokens"
    for index, body in enumerate(bodies):
        assert body[cap] == 4096
        assert body["model"] == case.model.split("/", 1)[1]
        if case.responses:
            assert body["reasoning"] == {"effort": "low"}
            assert body["store"] is False
            assert body["include"] == ["reasoning.encrypted_content"]
            assert body["prompt_cache_options"] == {"mode": "explicit"}
            if index:
                assert any(
                    item.get("encrypted_content") == "synthetic-encrypted-0"
                    for item in body["input"]
                ), "native reasoning missing on wire"
        elif case.name == "opus55":
            assert body["thinking"] == {"type": "adaptive"}
            assert body["output_config"] == {"effort": "low"}
            assert "cache_control" in json.dumps(body)
            if index:
                assert "synthetic-signature-0" in json.dumps(body["messages"])
        else:
            assert "reasoning_effort" not in body and "thinking" not in body
            if index:
                assert any(
                    m.get("reasoning_content") == "synthetic thought"
                    for m in body["messages"]
                    if m.get("role") == "assistant"
                )
    return reference


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_agent_cache_matrix_mock_http(case, tmp_path, monkeypatch):
    await _check_mock_http_run(case, tmp_path, monkeypatch)


@pytest.mark.asyncio
async def test_agent_cache_mock_http_restores_prior_drop_params(tmp_path, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    with monkeypatch.context() as scoped:
        # The helper checks output_config={"effort": "low"} on all four Opus
        # HTTP bodies; drop_params=True would silently remove that parameter.
        await _check_mock_http_run(
            next(case for case in CASES if case.name == "opus55"), tmp_path, scoped
        )
        assert litellm.drop_params is False
    assert litellm.drop_params is True


@pytest.mark.asyncio
async def test_agent_cache_reference_unique_per_run_and_fixed_across_calls(tmp_path, monkeypatch):
    with monkeypatch.context() as scoped:
        first = await _check_mock_http_run(CASES[0], tmp_path, scoped)
    with monkeypatch.context() as scoped:
        second = await _check_mock_http_run(CASES[0], tmp_path, scoped)
    assert first != second, "separate agent instances reused a warmed reference"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_wire_prefix_comparison_preserves_content_not_cache_metadata(case):
    live = '<context>\n<live_state expr="self._live_state">\n0\n</live_state>\n</context>'
    task = {"type": "text" if case.name == "opus55" else "input_text", "text": "fixed task"}
    if case.name == "opus55":
        # The historical task and live block share one user container.
        previous_items = [{"role": "user", "content": [task, {"type": "text", "text": live}]}]
        stable_items = [{"role": "user", "content": [task]}]
    elif case.responses:
        stable_items = [{"role": "user", "content": [task]}]
        previous_items = stable_items + [
            {"role": "user", "content": [{"type": "input_text", "text": live}]}
        ]
    else:
        stable_items = [{"role": "user", "content": "fixed task"}]
        previous_items = stable_items + [{"role": "user", "content": live}]
    key = "input" if case.responses else "messages"
    previous = {key: previous_items}
    later = {key: stable_items + [{"role": "assistant", "content": "next turn"}]}
    # Marker movement must not count as content drift.
    later[key][0]["cache_control"] = {"type": "ephemeral"}
    _assert_history_prefix_stable(case, previous, later, 0)
    # Copy before mutating: comparison must not modify the captured bodies.
    changed = json.loads(json.dumps(later))
    if isinstance(changed[key][0]["content"], list):
        changed[key][0]["content"][0]["text"] += " changed"
    else:
        changed[key][0]["content"] += " changed"
    with pytest.raises(AssertionError, match="historical content/shape changed"):
        _assert_history_prefix_stable(case, previous, changed, 0)


@pytest.mark.parametrize("cached", [(0, 0, 0, 0), (0, 9000, 0, 0), (0, 9000, 9000, 7000)])
def test_agent_cache_misses_fail(cached):
    usages = [LLMUsage(input_tokens=14_000, cached_input_tokens=n) for n in cached]
    with pytest.raises(AssertionError):
        _require_cache_hits(usages)


def test_agent_cache_hits_pass():
    _require_cache_hits(
        [LLMUsage(input_tokens=14_000, cached_input_tokens=n) for n in (0, 9000, 9000, 9000)]
    )


@pytest.mark.parametrize(
    "code",
    [
        "x = 1; self._live_state = x; print(x); import os",
        "print(self._llm.config)",
        "return_result(3)",
    ],
)
def test_unexpected_cell_fails_before_execution(code):
    with pytest.raises(AssertionError):
        _validate_cell(json.dumps({"code": code}), 0)


def test_cell_validation_allows_formatting_only():
    _validate_cell(json.dumps({"code": "x=1\nself._live_state=x\nprint(x) # fine"}), 0)


@hidden
def _observer_body(case, phase):
    key = "input" if case.responses else "messages"
    live = f'<context>\n<live_state expr="self._live_state">\n{phase}\n</live_state>\n</context>'
    block_type = "text" if case.name == "opus55" else "input_text"
    if case.name == "opus55":
        items = [{"role": "user", "content": [{"type": block_type, "text": "fixed task"}]}]
    elif case.responses:
        items = [{"role": "user", "content": [{"type": block_type, "text": "fixed task"}]}]
    else:
        items = [{"role": "user", "content": "fixed task"}]
    for index in range(phase):
        items.append({"role": "assistant", "content": f"synthetic native turn {index}"})
        if case.name == "opus55":
            items.append(
                {"role": "user", "content": [{"type": "tool_result", "content": "result"}]}
            )
        else:
            items.append({"role": "tool", "content": "result"})
    if case.name == "opus55":
        items[-1]["content"].append({"type": "text", "text": live})
    else:
        content = [{"type": block_type, "text": live}] if case.responses else live
        items.append({"role": "user", "content": content})
    return {
        "model": case.model,
        "tools": [{"name": "python_cell"}],
        "system": "fixed system",
        "instructions": "fixed instructions",
        key: items,
    }


@hidden
def _observer_request(body, method="POST"):
    return httpx.Request(method, "https://example.invalid/model", json=body)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("drift", ["model", "tools", "system", "instructions", "history"])
def test_live_observer_rejects_prefix_drift(case, drift):
    observer = _WirePrefixObserver(case)
    observer.observe(_observer_request(_observer_body(case, 0)))
    body = _observer_body(case, 1)
    if drift == "history":
        key = "input" if case.responses else "messages"
        body[key][0]["content"] = "changed historical shape"
    else:
        body[drift] = "changed static prefix"
    with pytest.raises(AssertionError, match="changed on wire"):
        observer.observe(_observer_request(body))
    assert not observer.prefix_stable
    assert observer.call_count == 2


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_live_observer_rejects_extra_attempts(case):
    observer = _WirePrefixObserver(case)
    observer.observe(_observer_request({}, method="GET"))
    for phase in range(4):
        body = _observer_body(case, phase)
        key = "input" if case.responses else "messages"
        body[key][0]["cache_control"] = {"type": "ephemeral", "synthetic": phase}
        body["prompt_cache_breakpoint"] = phase
        observer.observe(_observer_request(body))
    observer.require_complete()
    with pytest.raises(AssertionError, match="unexpected retry or extra model POST"):
        observer.observe(_observer_request(_observer_body(case, 3)))
    assert observer.call_count == 5
    assert not observer.prefix_stable


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_live_observer_rejects_missing_attempts(count):
    observer = _WirePrefixObserver(CASES[0])
    for phase in range(count):
        observer.observe(_observer_request(_observer_body(CASES[0], phase)))
    with pytest.raises(AssertionError, match="exactly four outgoing"):
        observer.require_complete()


@pytest.mark.asyncio
async def test_live_observer_forwards_unchanged_and_restores(monkeypatch):
    original_send = httpx.AsyncClient.send
    request = _observer_request(_observer_body(CASES[0], 0))
    content = request.content
    headers = dict(request.headers)
    seen = []
    response = httpx.Response(200, json={"ok": True})

    async def fake_send(client, outgoing, *args, **kwargs):
        assert outgoing is request
        assert outgoing.content == content and dict(outgoing.headers) == headers
        seen.append((args, kwargs))
        return response

    with monkeypatch.context() as scoped:
        scoped.setattr(httpx.AsyncClient, "send", fake_send)
        observer = _install_wire_observer(CASES[0], scoped)
        async with httpx.AsyncClient() as client:
            result = await client.send(request, stream=True, follow_redirects=False)
        assert result is response
        assert seen == [((), {"stream": True, "follow_redirects": False})]
        assert observer.call_count == 1
    assert httpx.AsyncClient.send is original_send


@pytest.mark.parametrize("count", [3, 5, 8])
def test_agent_cache_hits_reject_extra_or_missing_responses(count):
    with pytest.raises(AssertionError, match="exactly four real responses"):
        _require_cache_hits([LLMUsage(input_tokens=14_000, cached_input_tokens=12_000)] * count)


@pytest.mark.parametrize(
    "raw_usage",
    [
        {"prompt_tokens": 14_000, "completion_tokens": 40},
        {"input_tokens": 14_000, "output_tokens": 40},
        {"input_tokens": 14_000, "output_tokens": 40, "input_tokens_details": None},
    ],
)
def test_usage_omitted_cache_fields_normalize_to_zero(raw_usage):
    usage = _extract_usage(SimpleNamespace(usage=raw_usage))
    assert usage is not None
    assert usage.input_tokens == 14_000
    assert usage.total_tokens == 14_040
    assert usage.cached_input_tokens == usage.cache_write_input_tokens == 0
    with pytest.raises(AssertionError, match="positive cache reads"):
        _require_cache_hits([usage] * 4)


def test_missing_usage_is_not_a_cache_hit():
    assert _extract_usage(SimpleNamespace()) is None
    with pytest.raises(AssertionError, match="missing provider usage"):
        _require_cache_hits([None] * 4)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_live_observer_rejects_final_live_suffix_drift(case):
    observer = _WirePrefixObserver(case)
    for phase in range(3):
        observer.observe(_observer_request(_observer_body(case, phase)))
    # Well-shaped context, but stale phase in the final outgoing POST.
    body = _observer_body(case, 3)
    key = "input" if case.responses else "messages"
    content = body[key][-1]["content"]
    if isinstance(content, list):
        content[-1]["text"] = content[-1]["text"].replace("\n3\n", "\n2\n")
    else:
        body[key][-1]["content"] = content.replace("\n3\n", "\n2\n")
    with pytest.raises(AssertionError, match="live suffix had an unexpected phase"):
        observer.observe(_observer_request(body))
    assert observer.call_count == 4
    assert not observer.prefix_stable
