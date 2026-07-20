# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark agent and Harbor runner for the NOOA framework.

Minimal reproducibility package for the NOOA tech report: the benchmark-agnostic
``BenchAgent`` (SWE-bench Verified, Terminal-Bench 2.0), the ``nemo-harbor``
container runner, and the trace analyzer used to extract the per-task token
statistics reported in the paper.
"""

from __future__ import annotations

# Maps --agent-type CLI values to dotted import paths: "module:ClassName"
AGENT_CLASSES: dict[str, str] = {
    # Unified SWE-bench + Terminal-Bench agent (the tech report's BenchAgent)
    "bench": "nooa_bench.bench_agent:BenchAgent",
}

__all__ = ["AGENT_CLASSES"]
