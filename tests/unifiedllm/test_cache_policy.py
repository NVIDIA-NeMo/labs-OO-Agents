# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable-prefix policy defaults, ownership, and migration contract."""

import json
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.cache_policy import apply_cache_policy


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("nested", [False, True])
def test_legacy_cache_setting_fails_with_migration_help(client_type, nested):
    config = {"cache_control_injection_points": []}
    if nested:
        config = {"extra_body": config}
    with pytest.raises(ValueError, match="removed.*cache_breakpoint=.*nooa_cache_boundary"):
        client_type("openai/gpt-5.6", **config)
    with client_type("openai/gpt-5.6") as client:
        with pytest.raises(ValueError, match="removed.*cache_breakpoint"):
            client.call([{"role": "user", "content": "hi"}], **config)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
async def test_legacy_cache_setting_fails_before_async_dispatch(client_type):
    async with client_type("openai/gpt-5.6") as client:
        with pytest.raises(ValueError, match="removed.*cache_breakpoint"):
            await client.acall([], cache_control_injection_points=[])


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
def test_direct_anthropic_default_marks_only_leading_instructions(client_type):
    original = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "changing"},
        {"role": "system", "content": "also changing"},
    ]
    with client_type("anthropic/claude-sonnet-4-5") as client:
        wire, _, _ = client._prepare_cache_boundary(
            original, responses=client_type is ResponsesClient
        )
    assert wire[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert wire[1] is original[1]
    assert wire[2] is original[2]
    assert original[0]["content"] == "stable"


def test_boundary_copies_only_the_marker_target_containers():
    original = [
        {"role": "system", "content": "stable"},
        {
            "role": "tool",
            "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        },
        {"role": "metadata", "nooa_cache_boundary": True},
        {"role": "user", "content": "live"},
    ]
    before = json.dumps(original)
    wire, _, _ = apply_cache_policy(original, "anthropic", responses=False)
    assert json.dumps(original) == before
    assert wire[0] is original[0]
    assert wire[1] is not original[1]
    assert wire[1]["content"][0] is original[1]["content"][0]
    assert wire[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert wire[2] is original[3]


@pytest.mark.parametrize("mapping", [None, "anthropic", "openai"])
def test_no_stable_prefix_never_marks_dynamic_content(mapping):
    messages = [
        {"role": "metadata", "nooa_cache_boundary": True},
        {"role": "system", "content": "live"},
    ]
    wire, _, _ = apply_cache_policy(messages, mapping, responses=True)
    assert wire == [{"role": "system", "content": "live"}]


@pytest.mark.parametrize("mapping", [None, "anthropic", "openai"])
@pytest.mark.parametrize("invalid", [False, "true", 1, None])
def test_invalid_boundary_is_not_silently_ignored(mapping, invalid):
    with pytest.raises(ValueError, match="must be true"):
        apply_cache_policy(
            [{"role": "metadata", "nooa_cache_boundary": invalid}], mapping, responses=True
        )


@pytest.mark.parametrize(
    "message",
    [
        {"nooa_cache_boundary": True},
        {"role": "user", "content": "must not disappear", "nooa_cache_boundary": True},
        {"role": "metadata", "content": "must not disappear", "nooa_cache_boundary": True},
    ],
)
def test_boundary_must_be_a_separate_metadata_element(message):
    with pytest.raises(ValueError, match="Use a separate.*metadata"):
        apply_cache_policy([message], None, responses=True)


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": "live"},
        {"role": "tool", "tool_call_id": "c", "content": "live"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c", "function": {"name": "run", "arguments": "{}"}},
            ],
        },
    ],
)
def test_projection_does_not_silently_drop_misplaced_boundaries(client_type, message):
    with client_type("openai/gpt-5.6", api_key="test") as client:
        with pytest.raises(ValueError, match="Use a separate.*metadata"):
            client.call([{**message, "nooa_cache_boundary": True}])


@pytest.mark.asyncio
async def test_auto_mapping_uses_effective_model_and_none_disables_markers():
    original = [{"role": "system", "content": "stable"}, {"role": "user", "content": "hi"}]
    async with CompletionClient("openai/gpt-5.6") as client:
        with patch("litellm.acompletion", new_callable=AsyncMock) as request:
            request.return_value = litellm.ModelResponse(
                choices=[{"message": {"role": "assistant", "content": "ok"}}]
            )
            await client.acall(original, model="anthropic/claude-sonnet-4-5")
            sent = request.await_args.kwargs["messages"]
            assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    with CompletionClient("anthropic/claude-sonnet-4-5", cache_breakpoint=None) as client:
        wire, _, _ = client._prepare_cache_boundary(original, responses=False)
        assert wire == original
