# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent class for the Windows AppContainer acceptance tests.

It lives in its own module because the spawned worker resolves the agent by
``module:qualname``. A class defined in a test function would be reported as
``<locals>``, and one defined in the test module's ``__main__`` equivalent is
not importable from the container's own ``__main__``.
"""

from __future__ import annotations

from nooa import Agent
from nooa.unifiedllm.fake import FakeLLMClient


class WinProbeAgent(Agent, llm=FakeLLMClient()):  # type: ignore[misc]
    label: str = "probe"

    def answer(self) -> int:
        return 42

    def echo(self, value: str) -> str:
        return f"{self.label}:{value}"
