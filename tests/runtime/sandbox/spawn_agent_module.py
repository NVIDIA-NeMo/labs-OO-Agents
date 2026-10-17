# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent used by the spawn-backend test.

The class must be importable by ``module:qualname`` in the spawned child — that
is the whole point of the blueprint — so it lives at module level in a module
the child can import, never inside a test function.
"""

from __future__ import annotations

from nooa import Agent
from nooa.unifiedllm.fake import FakeLLMClient

COUNTER_VALUE = 41


class SpawnProbeAgent(Agent, llm=FakeLLMClient()):
    """Minimal agent whose tools are exercised across a spawned worker boundary."""

    prefix: str = "value"

    def label(self, name: str) -> str:
        """Return a greeting; deterministic, so it can run parent-side."""
        return f"{self.prefix}:{name}"

    def bump(self) -> int:
        """Mutate parent-side state; the cell must observe the live value."""
        self.prefix = "changed"
        return COUNTER_VALUE
