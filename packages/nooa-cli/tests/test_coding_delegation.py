# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delegated workers preserve controller settings and accept empty objectives."""

import asyncio

import pytest
from nooa_cli.coding.agent import CodingAgent
from nooa_cli.coding.experimental_agent import ExperimentalCodingAgent

from nooa.interactive import SummarizationConfig
from nooa.unifiedllm import FakeLLMClient


@pytest.mark.parametrize("agent_type", [CodingAgent, ExperimentalCodingAgent])
@pytest.mark.parametrize("policy", ["none", "token_budget"])
async def test_delegate_preserves_installed_summarization(
    agent_type, policy, tmp_path, monkeypatch
):
    config = SummarizationConfig(
        policy=policy, max_tokens=12345, preserve_recent=3, target_chars=6789
    )
    parent = agent_type(llm=FakeLLMClient(), cwd=tmp_path, summarization=config)
    inspected = []

    class InspectingWorker(parent._worker_type):
        async def investigate(self, objective: str, supplied_context=None) -> str:
            summaries = getattr(self, "_summarizers", [])
            if policy == "none":
                assert not summaries
            else:
                assert len(summaries) == 1
                assert summaries[0].config.max_tokens == 12345
                assert summaries[0].config.preserve_recent == 3
                assert summaries[0].config.target_chars == 6789
            assert self.llm is parent.llm
            assert self.shell is not parent.shell
            inspected.append(objective)
            return "worker report"

    monkeypatch.setattr(parent, "_worker_type", InspectingWorker)
    try:
        assert await parent.delegate("inspect") == "worker report"
        assert inspected == ["inspect"]
    finally:
        await parent.close()


@pytest.mark.parametrize("objective", ["", "   ", "\n"])
async def test_spawn_empty_objective_finishes_with_fallback_label(objective, tmp_path, monkeypatch):
    parent = CodingAgent(
        llm=FakeLLMClient(), cwd=tmp_path, summarization=SummarizationConfig(policy="none")
    )

    class EmptyTaskWorker(parent._worker_type):
        async def investigate(self, actual_objective: str, supplied_context=None) -> str:
            assert actual_objective == objective
            return "empty task report"

    monkeypatch.setattr(parent, "_worker_type", EmptyTaskWorker)
    try:
        handle = parent.spawn(objective)
        assert handle.label == "Delegated task"
        await asyncio.wait_for(handle._task, 2)
        assert handle.state == "done"
    finally:
        await parent.close()
