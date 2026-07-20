# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared LLM configuration for examples."""

import os

from dotenv import load_dotenv

from nooa.unifiedllm import CompletionClient

load_dotenv(override=True)

# Shared LLM client for all examples
# Short name 'qwen' refers to nvidia_nim/qwen/qwen3-next-80b-a3b-instruct
qwen = CompletionClient(
    model="nvidia_nim/qwen/qwen3-next-80b-a3b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    api_base="https://integrate.api.nvidia.com/v1",
)

# Note: Use CompletionClient for tool calling (Chat Completions API format)
# ResponsesClient uses Responses API which has different tool call format
gpt_oss_120b = CompletionClient(
    model="openai/nvidia/openai/gpt-oss-120b",
    api_base="https://inference-api.nvidia.com/v1/",
    api_key=os.getenv("NVIDIA_INFERENCE_API_KEY"),
)

# Nemotron-3-Nano-30B reasoning model with extended thinking
nemotron3_nano_30b = CompletionClient(
    model="openai/nvidia/nvidia/Nemotron-3-Nano-30B-A3B",
    api_base="https://inference-api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_INFERENCE_API_KEY"),
    max_tokens=65384,
    max_thinking_tokens=4096,
)
