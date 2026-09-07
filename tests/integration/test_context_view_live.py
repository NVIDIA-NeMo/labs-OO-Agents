# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live ContextView end-to-end test on NVIDIA Nemotron Super v3."""

from __future__ import annotations

import os
from typing import Literal

import pytest
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

from nooa import Agent, Block, Context, DefaultSkillView, Skill, resolve_context_view, strategy
from nooa.context_blocks.events import UserEvent
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import get_llm_client

load_dotenv(find_dotenv(usecwd=True), override=True)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.getenv("NVIDIA_INFERENCE_API_KEY") or os.getenv("NVIDIA_INTERNAL_API_KEY")),
        reason="NVIDIA internal API key not set",
    ),
]


class Decision(BaseModel):
    can_fulfill: bool
    missing: list[str]
    policy: Literal["SKILL_DENY_MISSING"]


class InventoryPolicyView:
    async def assemble(self, skill, call):
        yield Block(
            key="inventory_policy",
            content=(
                "Policy SKILL_DENY_MISSING: can_fulfill is false when any requested "
                "item has quantity zero; include each such item in missing."
            ),
        )


class InventoryView:
    async def assemble(self, agent, call):
        yield Block(key="instructions", content="Apply the supplied inventory policy exactly.")
        for skill in agent.active_skills():
            view = resolve_context_view(skill, default=DefaultSkillView())
            async for item in view.assemble(skill, call):
                yield item
        yield UserEvent(content=call.format_parameters_as_code())
        yield Block(key="request", content="Return the inventory decision.", role="user")


class InventoryPolicy(Skill, context_view=InventoryPolicyView()):
    pass


class InventoryAgent(Agent, context_view=InventoryView()):
    policy = InventoryPolicy()

    @strategy(PredictStrategy())
    async def decide(self, inventory: dict[str, int]) -> Decision:
        """Decide whether the requested inventory can be fulfilled."""
        ...


class DefaultInventoryPolicy(Skill):
    context_block = ("inventory_policy", "self.inventory_policy_text()")


class DefaultInventoryAgent(Agent):
    policy = DefaultInventoryPolicy()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context["inventory_state"] = Context(expr="self.inventory_state()")

    def inventory_policy_text(self) -> str:
        return (
            "Policy SKILL_DENY_MISSING: can_fulfill is false when any requested "
            "item has quantity zero; include each such item in missing."
        )

    def inventory_state(self) -> dict[str, int]:
        return {"apple": 3, "orange": 0}

    @strategy(PredictStrategy())
    async def decide(self, inventory: dict[str, int]) -> Decision:
        """Apply the inventory policy and return the decision."""
        ...


@pytest.mark.asyncio
async def test_context_view_end_to_end_on_nemotron_super_v3(monkeypatch):
    monkeypatch.setenv("OTLP_ENDPOINT", "http://127.0.0.1:1/v1/traces")
    key = os.getenv("NVIDIA_INFERENCE_API_KEY") or os.environ["NVIDIA_INTERNAL_API_KEY"]
    llm = get_llm_client(
        "openai/nvidia/nvidia/nemotron-3-super-v3",
        api_base="https://inference-api.nvidia.com/v1",
        api_key=key,
        max_tokens=512,
        temperature=0,
    )
    agent = InventoryAgent(llm=llm, context={"ignored_manager_block": "ignored"})
    inventory = {"apple": 3, "orange": 0}

    assembled = await agent.runtime._prepare_context(InventoryAgent.decide, call_args=(inventory,))
    assert [getattr(item, "key", "event") for item in assembled] == [
        "instructions",
        "inventory_policy",
        "event",
        "request",
    ]

    result = await agent.decide(inventory)
    assert result == Decision(
        can_fulfill=False,
        missing=["orange"],
        policy="SKILL_DENY_MISSING",
    )


@pytest.mark.asyncio
async def test_default_views_end_to_end_on_nemotron_super_v3(monkeypatch):
    monkeypatch.setenv("OTLP_ENDPOINT", "http://127.0.0.1:1/v1/traces")
    key = os.getenv("NVIDIA_INFERENCE_API_KEY") or os.environ["NVIDIA_INTERNAL_API_KEY"]
    llm = get_llm_client(
        "openai/nvidia/nvidia/nemotron-3-super-v3",
        api_base="https://inference-api.nvidia.com/v1",
        api_key=key,
        max_tokens=512,
        temperature=0,
    )
    agent = DefaultInventoryAgent(llm=llm)
    inventory = {"apple": 3, "orange": 0}

    assembled = await agent.runtime._prepare_context(
        DefaultInventoryAgent.decide, call_args=(inventory,)
    )
    keys = [getattr(item, "key", "event") for item in assembled]
    assert keys.index("inventory_policy") < keys.index("inventory_state")
    assert next(
        item for item in assembled if getattr(item, "key", None) == "inventory_state"
    ).content == ("{'apple': 3, 'orange': 0}")

    result = await agent.decide(inventory)
    assert result == Decision(
        can_fulfill=False,
        missing=["orange"],
        policy="SKILL_DENY_MISSING",
    )
