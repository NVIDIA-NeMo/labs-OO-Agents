# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct Event Sequence Example - Debug tool for understanding event flow.

This example demonstrates the exact sequence of events that CodeActStrategy
creates in the history when processing a tool call.

Expected sequence:
1. Task - The initial task/prompt
2. ToolCallEvent - The LLM's request to call execute_python
3. ToolResultEvent - The result of code execution
4. LLMOutput - The final structured output

Run with:
    uv run python examples/advanced/codeact_event_sequence.py
"""

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig
from nooa.util.quickstart import autorun, llm


class SimpleAgent(Agent, llm=llm):
    """A simple agent for testing CodeActStrategy event sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 42

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def get_value(self) -> int:
        """Get the value stored in self.value.

        Use self.value to access the stored value.
        Return it as the result.
        """
        ...


def print_event_sequence(agent: Agent):
    """Print the event sequence from the agent's event manager."""
    events = list(agent.event_manager.values())
    W = 70

    print("\n" + "─" * W)
    print(f"  EVENT SEQUENCE  ({len(events)} events)")
    print("─" * W)

    for i, event in enumerate(events):
        etype = event.event_type
        tool_call_id = getattr(event, "tool_call_id", "") or ""
        tag = "  [prefill]" if tool_call_id.startswith("prefill_") else ""

        print(f"\n[{i}] {etype.upper()}{tag}")

        if etype == "task":
            print(f"    {getattr(event, 'prompt', '')}")

        elif etype == "tool_call":
            code = (getattr(event, "arguments", {}) or {}).get("code", "")
            for line in code.splitlines():
                print(f"    {line}")

        elif etype == "python_output":
            stdout = str(getattr(event, "stdout", "") or "").rstrip()
            error = getattr(event, "error", None)
            value = getattr(event, "value", None)
            if stdout:
                for line in stdout.splitlines():
                    print(f"    {line}")
            if error:
                print(f"    error: {error}")
            if value is not None:
                print(f"    → {value}")

        else:
            content = str(getattr(event, "content", "") or "")
            if content:
                print(f"    {content}")

    print("\n" + "─" * W)
    flow = " → ".join(
        e.event_type
        + (" [prefill]" if (getattr(e, "tool_call_id", "") or "").startswith("prefill_") else "")
        for e in events
    )
    print(f"  {flow}")
    print("─" * W + "\n")


@autorun
async def main():
    agent = SimpleAgent()
    print("Calling agent.get_value()  (self.value = 42)...")
    print("-" * 70)

    try:
        result = await agent.get_value()
        print(f"\n✅ Result: {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Print the event sequence
    print_event_sequence(agent)

    print("Done!")
