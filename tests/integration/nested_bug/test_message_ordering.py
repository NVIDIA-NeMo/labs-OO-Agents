# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test tool_call_id ordering constraints across providers.

This script sends hand-built message streams directly to LLMs to understand
which providers enforce strict tool_call/tool_response ordering.
"""

import asyncio
import os

import litellm
from dotenv import load_dotenv

load_dotenv()

# Drop unsupported params
litellm.drop_params = True


# Model configs to test
MODELS = {
    "gpt-5": {
        "model": "openai/azure/openai/gpt-5.1",
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_INFERENCE_API_KEY"),
    },
    "claude-haiku": {
        "model": "openai/aws/anthropic/claude-haiku-4-5-v1",
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_INFERENCE_API_KEY"),
    },
    "gemini-flash": {
        "model": "openai/gcp/google/gemini-2.5-flash-lite",
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_INFERENCE_API_KEY"),
    },
    "gpt-oss-120b": {
        "model": "openai/nvidia/openai/gpt-oss-120b",
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_INFERENCE_API_KEY"),
    },
}

# Tools definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_value",
            "description": "Get a value",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]


def stream1_valid():
    """Stream 1: Valid - tool call followed immediately by tool response."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Get the value for key 'foo'"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {"name": "get_value", "arguments": '{"key": "foo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_A", "content": "bar"},
    ]


def stream2_nested_bug():
    """Stream 2: Simulates nested agent bug - tool response A comes after B's conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Get the value for key 'foo'"},
        # Outer agent calls execute_python (tool call A)
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {"name": "get_value", "arguments": '{"key": "foo"}'},
                }
            ],
        },
        # Inner agent starts (new user message)
        {"role": "user", "content": "Inner agent task: process the value"},
        # Inner agent makes tool call B
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_B",
                    "type": "function",
                    "function": {"name": "get_value", "arguments": '{"key": "inner"}'},
                }
            ],
        },
        # Inner agent gets response B
        {"role": "tool", "tool_call_id": "call_B", "content": "inner_result"},
        # NOW outer agent's tool response A comes (out of order!)
        {"role": "tool", "tool_call_id": "call_A", "content": "outer_result"},
        # Continue conversation
        {"role": "user", "content": "What did you get?"},
    ]


def stream3_duplicate_response():
    """Stream 3: Tool response A appears twice - once correctly, once after B."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Get the value for key 'foo'"},
        # Tool call A
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {"name": "get_value", "arguments": '{"key": "foo"}'},
                }
            ],
        },
        # Tool response A (correct placement)
        {"role": "tool", "tool_call_id": "call_A", "content": "result_A"},
        {"role": "user", "content": "Now get another value"},
        # Tool call B
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_B",
                    "type": "function",
                    "function": {"name": "get_value", "arguments": '{"key": "bar"}'},
                }
            ],
        },
        # Tool response B
        {"role": "tool", "tool_call_id": "call_B", "content": "result_B"},
        # Duplicate tool response A (appears again after B!)
        {"role": "tool", "tool_call_id": "call_A", "content": "result_A_again"},
        {"role": "user", "content": "What did you get?"},
    ]


def stream4_missing_response():
    """Stream 4: Tool call with NO response (missing tool response)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Get the value for key 'foo'"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {"name": "get_value", "arguments": '{"key": "foo"}'},
                }
            ],
        },
        # NO tool response - just continue with user message
        {"role": "user", "content": "What happened?"},
    ]


async def _check_stream(model_name: str, stream_name: str, messages: list):
    """Check a message stream against a model (helper, not a pytest test)."""
    config = MODELS[model_name]

    try:
        response = await litellm.acompletion(
            model=config["model"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            messages=messages,
            tools=TOOLS,
            max_tokens=100,
            temperature=1.0,
        )
        content = response.choices[0].message.content or "(no content)"
        return f"✓ PASS - {content[:50]}..."
    except Exception as e:
        error_msg = str(e)
        # Extract the key error message
        if "tool_call" in error_msg.lower() or "tool_use" in error_msg.lower():
            # Find the relevant part
            if "must be followed by" in error_msg:
                return "✗ FAIL - tool_call ordering violation"
            elif "tool_result" in error_msg:
                return "✗ FAIL - tool_use/tool_result mismatch"
        return f"✗ FAIL - {error_msg[:80]}..."


async def main():
    streams = {
        "Stream 1 (valid)": stream1_valid(),
        "Stream 2 (nested bug)": stream2_nested_bug(),
        "Stream 3 (duplicate response)": stream3_duplicate_response(),
        "Stream 4 (missing response)": stream4_missing_response(),
    }

    print("=" * 70)
    print("Testing tool_call_id ordering constraints across providers")
    print("=" * 70)

    for stream_name, messages in streams.items():
        print(f"\n{stream_name}:")
        print("-" * 50)

        for model_name in MODELS:
            result = await _check_stream(model_name, stream_name, messages)
            print(f"  {model_name:15} {result}")


if __name__ == "__main__":
    asyncio.run(main())
