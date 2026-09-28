# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hub reasoning/tool replay over the Responses API: two capped calls per model.

Responses-API sibling of test_open_model_tool_reasoning_live.py. That test only
exercises CompletionClient (Chat's reasoning_content); this one exercises
ResponsesClient (a "reasoning" output item, either OpenAI's summary field or
DeepSeek's content/reasoning_text field) for the same four Hub-routed open-weight
families, to check whether native reasoning replay actually works over Responses
for models other than DeepSeek — not just whether the provider gate admits them.

NOOA_RUN_OPEN_MODEL_REPLAY=1 uv run --env-file ../.env pytest -m integration -s
    tests/integration/test_open_model_responses_reasoning_live.py

Tests the raw reasoning item on the next HTTP request after SQLite close/reopen,
not merely its visibility somewhere in answer text. No opaque state is printed.
"""

import json
import os

import httpx
import pytest

from nooa.llm_types import LLMResponse
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import ResponsesClient, Tool
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.retry_config import RetryConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NOOA_RUN_OPEN_MODEL_REPLAY") != "1",
        reason="set NOOA_RUN_OPEN_MODEL_REPLAY=1 to spend inference tokens",
    ),
]

MODELS = {
    "deepseek": "openai/nvidia/deepseek-ai/deepseek-v4-pro",
    "kimi": "openai/nvidia/moonshotai/kimi-k3",
    "glm": "openai/nvidia/zai-org/glm-5.3",
    "qwen": "openai/nvidia/qwen/qwen3-5-397b-a17b",
}


def lookup(key: str) -> int:
    """Look up the offset needed to finish the comparison."""
    return 0


def _field(item, key, default=None):
    """Read a key from either a dict (replayed request body) or an SDK object (raw response)."""
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _reasoning_item(output: list):
    return next((item for item in output if _field(item, "type") == "reasoning"), None)


def _reasoning_wire_text(item) -> str:
    """Read whichever field this route used: OpenAI's summary or DeepSeek's content."""
    content = _field(item, "content")
    if isinstance(content, list) and content:
        return "\n".join(_field(block, "text", "") for block in content)
    summary = _field(item, "summary")
    if isinstance(summary, list):
        return "\n".join(_field(block, "text", "") for block in summary)
    if isinstance(summary, str):
        return summary
    return ""


@pytest.mark.asyncio
@pytest.mark.parametrize("family", MODELS)
async def test_open_model_responses_reasoning_after_sqlite_resume(family, tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    sent = []
    original_send = httpx.AsyncClient.send

    async def capture(client, request, *args, **kwargs):
        if request.method == "POST" and request.url.host == "inference-api.nvidia.com":
            sent.append(json.loads(request.content))
        return await original_send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", capture)
    # A scheduling-puzzle prompt (same shape used by the reasoning-level connect
    # probe) forces genuine reasoning even at a low/default effort; a trivial
    # arithmetic ask let some routes skip thinking entirely with zero reasoning
    # tokens, which produced no reasoning output item to test replay against.
    puzzle = (
        "Eight jobs—A, B, C, D, E, F, G and H—must run one at a time.\n"
        "Each job runs exactly once.\n\n"
        "Rules:\n"
        "- A runs exactly three positions after B.\n"
        "- C runs immediately before E.\n"
        "- F runs immediately after E.\n"
        "- G runs immediately before D.\n"
        "- E runs after A.\n"
        "- H runs last.\n\n"
        "Find the order that satisfies every rule."
    )
    messages = [
        {
            "role": "system",
            "content": "Solve the puzzle. Call the lookup tool with key='offset' before "
            "answering; do not answer until the tool returns.",
        },
        {"role": "user", "content": puzzle},
        {"role": "user", "content": "Live context: phase=before lookup."},
    ]
    options = {
        "model": MODELS[family],
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_INFERENCE_API_KEY"],
        "max_tokens": 1536,
        "http_config": HttpConfig(read_timeout=120),
        "num_retries": 0,
        "retry_config": RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    }
    tools = [Tool(name="lookup", description="Look up an offset", callable=lookup)]
    # Force a real reasoning effort: some routes emit zero reasoning tokens (and
    # therefore no "reasoning" output item at all) on a plain call with no
    # explicit effort, even for a nontrivial prompt.
    reasoning_kw = {"reasoning": {"effort": "medium"}}
    async with ResponsesClient(**options) as client:
        seed = await client.acall(messages, tools=tools, **reasoning_kw)
    print(
        json.dumps(
            {
                "family": family,
                "phase": "seed",
                "usage": seed.usage.model_dump() if seed.usage else None,
                "finish_reason": seed.finish_reason,
                "reasoning_chars": len(seed.reasoning or ""),
                "replay_scope_set": seed.replay_scope is not None,
            }
        ),
        flush=True,
    )
    assert seed.tool_calls and seed.finish_reason != "length"
    reasoning_tokens = seed.usage.reasoning_tokens if seed.usage else 0
    seed_output = _reasoning_item(seed.raw_response.output)
    if seed_output is None:
        # Not a NOOA-side gap: some routes spend reasoning tokens (billed, real
        # thinking) but never expose a "reasoning" output item on the wire at
        # all, so there is nothing for any client-side code to capture or
        # replay. Report it distinctly from a route that exposes reasoning and
        # then loses it (the real, actionable case below).
        pytest.skip(
            f"{family}: gateway spent {reasoning_tokens} reasoning tokens but exposed no "
            "reasoning output item on the wire -- nothing to capture or replay"
        )
    raw = _reasoning_wire_text(seed_output)
    assert raw, "route returned no readable reasoning text"
    if seed.replay_scope is None:
        dumped = seed_output.model_dump() if hasattr(seed_output, "model_dump") else seed_output
        present_fields = sorted(k for k, v in dumped.items() if v is not None)
        pytest.fail(
            f"{family}: reasoning output item was exposed on the wire but capture_parts "
            "fell back to portable-only text (no native replay) -- this route's reasoning "
            f"item has fields {present_fields}, not handled as native state"
        )
    database = tmp_path / "session.db"
    with SQLiteStorageManager(database) as storage:
        storage.event_backend.store("1", seed)
    with SQLiteStorageManager(database) as storage:
        restored = next(storage.event_backend.all_events())
    assert isinstance(restored, LLMResponse)
    assert restored.reasoning == seed.reasoning
    assert restored.raw_response is None
    history = [
        *messages,
        restored,
        *[
            {"role": "tool", "tool_call_id": call.id, "content": "0"}
            for call in restored.tool_calls
        ],
        {"role": "user", "content": "Live context: lookup complete. Answer concisely."},
    ]
    async with ResponsesClient(**options) as client:
        result = await client.acall(history, tools=tools, **reasoning_kw)
    assert len(sent) == 2, "probe must not retry"
    replay_item = _reasoning_item(sent[1]["input"])
    assert replay_item is not None, "native reasoning item dropped from the replayed request"
    assert _reasoning_wire_text(replay_item) == raw, "native reasoning text changed on wire"
    assert result.finish_reason == "stop"
    assert result.content
    print(
        json.dumps(
            {
                "family": family,
                "phase": "resumed",
                "usage": result.usage.model_dump() if result.usage else None,
                "wire_reasoning_equal": True,
                "sqlite_reasoning_equal": True,
            }
        ),
        flush=True,
    )
