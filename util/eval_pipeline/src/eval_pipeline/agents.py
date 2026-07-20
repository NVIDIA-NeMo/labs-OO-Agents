# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent utilities for the evaluation pipeline.

This module provides:
- AgentWrapper: Adapts agents to the pipeline's run() interface
- agent_from_spec: Reconstructs an agent from an AgentSpec in a subprocess
"""

import os
from typing import Any


class AgentWrapper:
    """Wraps any agent to conform to pipeline's run() interface."""

    def __init__(self, agent_instance, method_name: str):
        self.agent = agent_instance
        self.method_name = method_name
        self.method = getattr(agent_instance, method_name)

    async def run(self, input: tuple) -> Any:
        """Run agent method with args/kwargs unpacking."""
        args, kwargs = input
        return await self.method(*args, **kwargs)


def agent_from_spec(spec) -> AgentWrapper:
    """Reconstruct an agent from an AgentSpec in a subprocess.

    Imports the agent class by module path (or file path for --agent flag),
    creates a fresh LLM client from the config dict, and wraps the result.
    """
    import importlib.util

    if spec.agent_file:
        # File-based agent (--agent flag): load from file path
        import sys
        from pathlib import Path

        path = Path(spec.agent_file)
        module_name = f"_agent_subprocess_{path.stem}"
        file_spec = importlib.util.spec_from_file_location(module_name, path)
        if file_spec is None or file_spec.loader is None:
            raise ImportError(f"Cannot load agent from file: {spec.agent_file}")
        mod = importlib.util.module_from_spec(file_spec)
        sys.modules[module_name] = mod
        file_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(spec.agent_module)
    cls = getattr(mod, spec.agent_class)

    # Build client from config dict (empty config → no LLM needed)
    client = None
    if spec.client_config and spec.client_config.get("model"):
        from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig

        cc = dict(spec.client_config)
        # Resolve API key from env var name at construction time
        api_key_env = cc.pop("api_key_env", None)
        if api_key_env:
            cc.setdefault("api_key", os.getenv(api_key_env, ""))

        # Extract retry config if present
        retry_kwargs = cc.pop("retry_config", None)
        retry_config = RetryConfig(**retry_kwargs) if retry_kwargs else None

        # Dispatch based on client_type
        client_type = cc.pop("client_type", "completion")
        if client_type == "responses":
            client = ResponsesClient(retry_config=retry_config, **cc)
        else:
            client = CompletionClient(retry_config=retry_config, **cc)

    agent_instance = cls(llm=client)
    return AgentWrapper(agent_instance, spec.method)
