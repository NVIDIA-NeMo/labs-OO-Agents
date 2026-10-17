# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent used by ``m3_smoke.py``.

It lives in its own module because the spawned worker rebuilds the namespace
from ``module:qualname``. A class defined in the running ``__main__`` cannot be
imported by the container: the child's ``__main__`` is ``win_worker``, not this
script. This is a property of the spawn contract, not of the Windows backend.
"""

from __future__ import annotations

from nooa import Agent
from nooa.unifiedllm.fake import FakeLLMClient


class ProbeAgent(Agent, llm=FakeLLMClient()):  # type: ignore[misc]
    label: str = "probe"

    def answer(self) -> int:
        return 42

    def echo(self, value: str) -> str:
        return f"{self.label}:{value}"
