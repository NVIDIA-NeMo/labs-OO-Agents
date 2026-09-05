# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import litellm

from nooa.unifiedllm import CompletionClient


def _response(prompt_tokens: int) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model="test-model",
        choices=[
            litellm.Choices(
                message=litellm.Message(role="assistant", content="done"),
                index=0,
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "total_tokens": prompt_tokens + 1,
        },
    )


def test_output_budget_uses_previous_provider_prompt_usage(tmp_path) -> None:
    usage_log = tmp_path / "llm_usage.jsonl"
    client = CompletionClient(
        model="test-model",
        context_window=1_000_000,
        max_tokens=384_000,
        output_token_margin=64_000,
        usage_log_path=str(usage_log),
    )
    try:
        with patch(
            "litellm.completion", side_effect=[_response(700_000), _response(710_000)]
        ) as completion:
            client.call([{"role": "user", "content": "first"}])
            client.call([{"role": "user", "content": "second"}])

        assert completion.call_args_list[0].kwargs["max_tokens"] == 384_000
        assert completion.call_args_list[1].kwargs["max_tokens"] == 236_000
        records = [json.loads(line) for line in usage_log.read_text().splitlines()]
        assert [record["prompt_tokens"] for record in records] == [700_000, 710_000]
        assert [record["requested_max_tokens"] for record in records] == [384_000, 236_000]
    finally:
        client.close()


def test_budget_controls_are_not_forwarded_to_provider(tmp_path) -> None:
    client = CompletionClient(
        model="test-model",
        context_window=1_000_000,
        max_tokens=384_000,
        output_token_margin=64_000,
        reasoning_output_floor=128_000,
        usage_log_path=str(tmp_path / "usage.jsonl"),
    )
    try:
        with patch("litellm.completion", return_value=_response(10)) as completion:
            client.call([{"role": "user", "content": "hello"}])

        kwargs = completion.call_args.kwargs
        assert "output_token_margin" not in kwargs
        assert "reasoning_output_floor" not in kwargs
        assert "usage_log_path" not in kwargs
    finally:
        client.close()


def test_usage_log_records_endpoint_and_reasoning_effort(tmp_path) -> None:
    usage_log = tmp_path / "usage.jsonl"
    client = CompletionClient(
        model="test-model",
        api_base="https://api.example.test/v1",
        max_tokens=384_000,
        reasoning_effort="max",
        usage_log_path=str(usage_log),
    )
    try:
        with patch("litellm.completion", return_value=_response(10)):
            client.call([{"role": "user", "content": "hello"}])

        record = json.loads(usage_log.read_text())
        assert record["endpoint"] == "https://api.example.test/v1"
        assert record["reasoning_effort"] == "max"
    finally:
        client.close()
