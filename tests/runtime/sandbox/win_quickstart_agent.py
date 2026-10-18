# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent written the way the examples are: on top of ``nooa.util.quickstart``.

``win_agent_module`` builds its own ``FakeLLMClient`` and so never runs nooa's
config discovery. This one does, at import time, inside the container — which
is the path real agents take and the toy agent skipped.
"""

from __future__ import annotations

from nooa import Agent
from nooa.util.quickstart import llm


class QuickstartAgent(Agent, llm=llm):
    def answer(self) -> int:
        """Return a fixed value; deterministic, so it runs parent-side."""
        return 42
