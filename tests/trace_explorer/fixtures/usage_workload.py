# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record a normal NOOA workflow using LiteLLM's local mock, never a provider."""

import asyncio
import sys
from unittest.mock import patch

from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.strategies import PredictStrategy
from nooa.tracing import enable_tracing, exporters, flush_traces, set_session
from nooa.unifiedllm import CompletionClient, RetryConfig


class Answer(BaseModel):
    """Typed synthetic response for the normal Predict validation path."""

    value: str


class Worker(Agent):
    """Produce a short synthetic answer."""

    @strategy(PredictStrategy())
    async def answer(self, question: str) -> Answer:
        """Answer the question concisely."""
        ...


class Workflow(Agent):
    """Exercise normal orchestration, child agents and two model identifiers."""

    async def run(self) -> list[Answer]:
        """Run independent workers through the ordinary mocked Completion transport."""
        clients = [
            CompletionClient(
                model,
                mock_response='{"value":"synthetic answer"}',
                retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
            )
            for model in ("openai/gpt-4o-mini", "openai/gpt-4o")
        ]
        try:
            return await asyncio.gather(
                *[Worker(llm=client).answer("synthetic question") for client in clients]
            )
        finally:
            for client in clients:
                await client.aclose()


async def main() -> None:
    """Write OTLP or portable-journal artifacts through production file exporters."""
    name = sys.argv[2] if len(sys.argv) > 2 else "jsonl"
    factory = {"jsonl": exporters.jsonl, "journal_file": exporters.journal_file}[name]
    enable_tracing(exporters=[factory(sys.argv[1])])
    set_session("usage-workload")
    client = CompletionClient("openai/gpt-4o-mini", mock_response="unused")
    try:
        results = await Workflow(llm=client).run()
        assert [result.value for result in results] == ["synthetic answer", "synthetic answer"]
        flush_traces()
    finally:
        await client.aclose()


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("Network disabled in fixture")):
        asyncio.run(main())
