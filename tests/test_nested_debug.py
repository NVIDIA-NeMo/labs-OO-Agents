# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal test to debug nested generation return value issue."""

import asyncio

from nooa.agent import Agent
from nooa.decorators import strategy
from nooa.runtime.actor import ActorRuntime
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class DebugAgent(Agent, llm=_TEST_LLM):
    def __init__(self):
        super().__init__()
        self.trace = []

    @strategy(PurePythonStrategy())
    async def outer(self) -> str:
        """Outer method - calls inner."""
        ...

    @strategy(PurePythonStrategy())
    async def inner(self) -> str:
        """Inner method - returns value."""
        ...


async def main():
    print("=" * 80)
    print("DEBUGGING NESTED GENERATION RETURN VALUES")
    print("=" * 80)

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # Outer method generation
            _resp("Calling inner"),
            # Inner method generation
            _resp("Returning value"),
        ]
    )

    agent_instance = DebugAgent()
    agent_instance._llm = fake_llm
    _runtime = ActorRuntime(agent=agent_instance)  # noqa: F841

    print("\n1. Calling outer method...")
    result = await agent_instance.outer()

    print("\n2. Results:")
    print(f"   - Outer returned: {repr(result)}")
    print(f"   - LLM call count: {fake_llm.call_count}")
    print(f"   - Trace: {agent_instance.trace}")

    print("\n3. Analysis:")
    if result is None:
        print("   ❌ ISSUE: Outer returned None instead of 'inner_value'")
        print("   - Inner was called? ", "inner:returning" in agent_instance.trace)
        print(
            "   - Outer received result? ",
            any("outer:got_result=" in t for t in agent_instance.trace),
        )
        if any("outer:got_result=" in t for t in agent_instance.trace):
            for t in agent_instance.trace:
                if "outer:got_result=" in t:
                    print(f"   - Outer saw: {t}")
    else:
        print(f"   ✓ SUCCESS: Outer correctly returned '{result}'")

    print("=" * 80)
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nFinal result: {repr(result)}")
