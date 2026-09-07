"""Probe hosted tool-call reasoning replay without printing sensitive content."""

from __future__ import annotations

import asyncio
import json
import os

from nooa.unifiedllm import CompletionClient, RetryConfig, create_tool_from_callable


def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def probe() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    client = CompletionClient(
        model="openai/deepseek-v4-flash",
        api_base=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key,
        max_tokens=2048,
        reasoning_effort="max",
        allowed_openai_params=["reasoning_effort"],
        retry_config=RetryConfig(max_retries=1),
    )
    tool = create_tool_from_callable(add_numbers)
    try:
        first = await client.acall(
            [{"role": "user", "content": "Use add_numbers to add 2 and 3."}],
            tools=[tool],
        )
        if len(first.tool_calls) != 1:
            raise AssertionError(f"expected one tool call, got {len(first.tool_calls)}")
        assistant = first.assistant_message
        if not isinstance(assistant, dict) or not assistant.get("reasoning_content"):
            raise AssertionError("hosted response did not expose reasoning_content")
        call = first.tool_calls[0]
        second = await client.acall(
            [
                {"role": "user", "content": "Use add_numbers to add 2 and 3."},
                assistant,
                {"role": "tool", "tool_call_id": call.id, "content": "5"},
            ],
            tools=[tool],
        )
        if second.finish_reason not in {"stop", "tool_calls"}:
            raise AssertionError(f"unexpected finish reason: {second.finish_reason}")
        summary_client = CompletionClient(
            model="openai/deepseek-v4-flash",
            api_base=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=api_key,
            max_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
            retry_config=RetryConfig(max_retries=1),
        )
        try:
            summary = await summary_client.acall(
                [{"role": "user", "content": "Reply with exactly: ready"}]
            )
            if summary.reasoning:
                raise AssertionError("thinking-disabled response contained reasoning_content")
        finally:
            await summary_client.aclose()
        print(
            json.dumps(
                {
                    "first_finish_reason": first.finish_reason,
                    "reasoning_replayed": True,
                    "second_finish_reason": second.finish_reason,
                    "second_request_accepted": True,
                    "summary_thinking_disabled": True,
                },
                sort_keys=True,
            )
        )
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(probe())
