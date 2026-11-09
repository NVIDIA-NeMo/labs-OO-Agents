# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in live model/CodeAct/Predict checks with real Relay and ATIF exports.

NOOA_RUN_RELAY_LIVE=1 uv run --extra nemo-relay pytest -m integration -s \
    tests/integration/test_nemo_relay_live.py

Uses existing NVIDIA internal gateway credentials (or OPENAI_API_KEY). Optional
NOOA_RELAY_LIVE_MODEL and NOOA_RELAY_LIVE_BASE_URL select another Responses
model/endpoint. No configuration is saved. Artifacts go to pytest's temp directory.
"""

from __future__ import annotations

from pydantic import BaseModel

from nooa import Agent, hidden, strategy
from nooa.strategies import PredictStrategy

with hidden:
    import asyncio
    import json
    import os

    import pytest
    from dotenv import dotenv_values

    from nooa.nemo_relay_middleware import nemo_relay_scope
    from nooa.unifiedllm import CompletionClient, ResponsesClient


class RelayAnswer(BaseModel):
    value: int
    label: str


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_relay_nested_concurrent_atif(tmp_path):
    if os.getenv("NOOA_RUN_RELAY_LIVE") != "1":
        pytest.skip("Set NOOA_RUN_RELAY_LIVE=1 for paid model calls")
    relay = pytest.importorskip("nemo_relay")
    env = {**dotenv_values(".env"), **os.environ}
    key = env.get("NVIDIA_INFERENCE_API_KEY") or env.get("NVIDIA_INTERNAL_API_KEY")
    if key:
        model = env.get("NOOA_RELAY_LIVE_MODEL", "openai/azure/openai/gpt-6.1-sol")
        base = env.get("NOOA_RELAY_LIVE_BASE_URL", "https://inference-api.nvidia.com/v1")
    else:
        key = env.get("OPENAI_API_KEY")
        model = env.get("NOOA_RELAY_LIVE_MODEL", "gpt-5-mini")
        base = env.get("NOOA_RELAY_LIVE_BASE_URL", "https://api.openai.com/v1")
    if not key:
        pytest.fail("Live Relay test requires existing model credentials")
    client = ResponsesClient(
        model, api_base=base, api_key=key, max_tokens=4096, reasoning={"effort": "low"}
    )
    default_client = CompletionClient("unused-default-model")

    class LiveRelayAgent(Agent, llm=default_client):
        @strategy(PredictStrategy(), llm=client)
        async def check(self, number: int) -> RelayAnswer:
            """Return the number as value, with label 'checked'."""
            ...

        @strategy(llm=client)
        async def double(self, number: int) -> int:
            """Double the number using Python.

            First call execute_python to assign doubled = number * 2 and print it.
            In a later call, execute Python that calls return_result(doubled).
            Keep these as separate calls so the variable persists between calls.
            """
            ...

        async def workflow(self, number: int) -> int:
            checked = await self.check(number)
            assert checked.value == number and checked.label == "checked"
            return await self.double(checked.value)

    exporter = relay.AtifExporter(
        session_id="nooa-relay-live", agent_name="LiveRelayAgent", agent_version="test"
    )
    events = []
    exporter.register("nooa-live-atif")
    relay.subscribers.register("nooa-live-events", events.append)
    try:
        agent = LiveRelayAgent()
        async with nemo_relay_scope(agent, "nooa-live") as root:
            results = await asyncio.gather(agent.workflow(21), agent.workflow(7))
            assert results == [42, 14]
            assert relay.scope.get_handle().uuid == root.uuid
        await relay.subscribers.flush_async()
        trajectory = json.loads(exporter.export_json())
        output = tmp_path / "relay-live-atif.json"
        output.write_text(json.dumps(trajectory, indent=2))
        print(f"Live ATIF: {output}")

        assert trajectory["schema_version"] == "ATIF-v1.7"
        steps = trajectory["steps"]
        assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
        model_steps = [s for s in steps if s.get("metrics")]
        assert len(model_steps) >= 4
        assert all(s["model_name"] == model for s in model_steps)
        assert all(s["metrics"]["prompt_tokens"] > 0 for s in model_steps)
        assert all(isinstance(s["message"], str) for s in model_steps)
        assert not any('"assistant_message"' in s["message"] for s in model_steps)
        observations = []
        prefill_calls = []
        for step in steps:
            calls = {c["tool_call_id"] for c in step.get("tool_calls", [])}
            if step.get("metrics"):
                response = step["extra"]["llm_response"]
                issued = {
                    c["id"] for c in response.get("assistant_message", {}).get("tool_calls", [])
                }
                # Relay may attach host-generated prefills to a preceding model
                # step. They are not provider-issued calls; do not count them as
                # evidence that the model chose to execute this code.
                prefills = {call_id for call_id in calls if call_id.startswith("prefill_")}
                assert calls - prefills == issued
                prefill_calls.extend(prefills)
            for observation in step.get("observation", {}).get("results", []):
                assert observation["source_call_id"] in calls
                assert isinstance(observation["content"], str)
                observations.append(observation)
        assert len(observations) >= 4
        print(
            f"ATIF host-generated prefill calls (ownership is upstream-defined): {len(prefill_calls)}"
        )
        assert trajectory["final_metrics"]["total_prompt_tokens"] == sum(
            s["metrics"].get("prompt_tokens", 0) for s in model_steps
        )
        starts = [e for e in events if getattr(e, "scope_category", None) == "start"]
        ends = [e for e in events if getattr(e, "scope_category", None) == "end"]
        assert {str(e.uuid) for e in starts} == {str(e.uuid) for e in ends}
        assert not any(e.name == "unused-default-model" for e in events)
        assert key not in output.read_text()
    finally:
        await relay.subscribers.flush_async()
        exporter.deregister("nooa-live-atif")
        relay.subscribers.deregister("nooa-live-events")
        await client.aclose()
        await default_client.aclose()
