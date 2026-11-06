# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark agent and Harbor runner for the NOOA framework.

Benchmark-agnostic coding agents for SWE-bench and Terminal-Bench, the
``nemo-harbor`` container runner, and trajectory/token analysis utilities.
"""

from __future__ import annotations

# The built-in --agent-type values, as dotted import paths: "module:ClassName".
# Other packages add agents through the ``nooa_bench.agents`` entry-point
# group (nooa-atom adds ``atom``); ``nooa_bench.agents.load_agent`` finds both.
AGENT_CLASSES: dict[str, str] = {
    # Unified SWE-bench + Terminal-Bench baseline.
    "bench": "nooa_bench.bench_agent:BenchAgent",
    "rlm": "nooa_bench.rlm_bench_agent:RLMBenchAgent",
}

__all__ = ["AGENT_CLASSES"]
