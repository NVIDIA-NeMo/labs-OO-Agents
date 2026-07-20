# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quickstart 12: Long-term memory — an opt-in, brain-inspired memory add-on.

Demonstrates the additive memory subsystem end to end WITHOUT needing an LLM
(uses the deterministic, offline hashing embedder + a FakeLLMClient):

  * install memory onto any agent (zero core changes)
  * conscious tools: remember() / recall()
  * dedup-on-write (encoding the same fact twice reinforces, doesn't duplicate)
  * spontaneous association (recall injected from recent task context)
  * reflection (offline consolidation: merge duplicates, form edges, prune)
  * forgetting (old, low-value memories decay and are pruned)

    uv run python examples/quickstart/12_memory.py
"""

import time

from nooa_memory import (
    Memory,
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    MemoryType,
)
from nooa_memory.config import ForgetPolicy, ReflectionPolicy

from nooa import Agent, hidden
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient

DAY = 24 * 3600.0


class AssistantAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    """A plain agent that has opted into the memory tools via the mixin."""


@hidden  # harness glue, not an agent tool — keep it out of the execution context
def main() -> None:
    agent = AssistantAgent()

    # Opt in. Everything stays in one in-memory SQLite DB for this demo; point
    # `path=...` at a file to persist across sessions.
    mgr = MemoryManager.install(
        agent,
        config=MemoryConfig(
            enabled=True,
            path=":memory:",
            reflection=ReflectionPolicy(merge_threshold=0.95),
            forget=ForgetPolicy(prune_min_age_hours=24.0),
        ),
    )

    print("=== 1. Conscious encoding (remember) ===")
    agent.remember("To deploy the service, run `make ship` from CI.", type="info")
    agent.remember("Roll back a bad release with `make undeploy`.", type="skill")
    agent.remember("The on-call rotation is tracked in PagerDuty.", type="info")
    print("memories stored:", mgr.store.count())

    print("\n=== 2. Deliberate recall ===")
    for m in agent.recall("how do I deploy?", k=2):
        print(f"  [{m.type}] {m.content}")

    print("\n=== 3. Dedup-on-write (reinforce, not duplicate) ===")
    id1 = agent.remember("To deploy the service, run `make ship` from CI.", type="info")
    print(
        "count still:", mgr.store.count(), "| reinforced:", mgr.store.get(id1).reinforcement_count
    )

    print("\n=== 4. Spontaneous association (auto-injected from task context) ===")
    agent.event_manager.add(Task(prompt="I need to deploy the service to production"))
    print(mgr.recall_for_context())

    print("\n=== 5. Reflection (offline consolidation) ===")
    # Two duplicates that bypass write-time dedup (e.g. arriving from different
    # sources / episodic writes). Reflection folds them into one canonical record.
    dup = "Run database migrations with `make migrate` before deploying."
    mgr.remember(dup, type=MemoryType.SKILL, dedup=False)
    mgr.remember(dup, type=MemoryType.SKILL, dedup=False)
    before = mgr.store.count()
    report = mgr.reflect()
    print(f"  before={before} after={mgr.store.count()} report={report.model_dump()}")

    print("\n=== 6. Forgetting (decay + prune) ===")
    old_text = "A throwaway note nobody will ever need again."
    old = mgr.store.add(
        Memory(
            content=old_text,
            type=MemoryType.INFO,
            importance=1.0,
            created_at=time.time() - 200 * DAY,
            last_accessed_at=time.time() - 200 * DAY,
        ),
        mgr.embedder.embed(old_text),
    )
    print("  added stale memory; active count:", mgr.store.count())
    pruned = mgr.forgetting.prune()
    print("  pruned:", old.id in pruned, "| active count now:", mgr.store.count())

    mgr.uninstall()
    print("\nDone. (memory uninstalled; agent is back to its original behaviour)")


if __name__ == "__main__":
    main()
