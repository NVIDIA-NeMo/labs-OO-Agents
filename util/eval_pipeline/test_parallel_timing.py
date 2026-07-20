#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test to verify actual parallel execution timing in the pipeline."""

import asyncio

# Add workspace to path
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval_pipeline.models import Task
from eval_pipeline.pipeline import (
    PipelineConfig,
    Sample,
    run_evaluation,
)
from eval_pipeline.scoring import ExactMatchScorer, ScorerConfig

# Track timing
sample_times: dict[str, dict] = {}


class TimingAgent:
    """Agent that records timing info."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay

    async def run(self, input: str) -> str:
        sample_id = input
        sample_times[sample_id] = {
            "start": time.perf_counter(),
            "start_time": datetime.now().isoformat(),
        }
        await asyncio.sleep(self.delay)
        sample_times[sample_id]["end"] = time.perf_counter()
        sample_times[sample_id]["end_time"] = datetime.now().isoformat()
        return "done"


class MockWriter:
    """Mock writer that does nothing."""

    def start(self, **kwargs):
        return Path("/tmp/mock")

    def append_result(self, result):
        pass

    def finalize(self, **kwargs):
        pass


async def run_test(n_samples: int, max_concurrent: int, delay: float = 0.5):
    """Run pipeline test."""
    sample_times.clear()

    samples = [
        Sample(
            task=Task(id=f"sample_{i}", input=f"sample_{i}", expected="done"),
            method="run",
            agent_class="TimingAgent",
            scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
            agent_factory=lambda d=delay: TimingAgent(d),
            display_name=f"sample_{i}",
        )
        for i in range(n_samples)
    ]

    # Create temporary directory for traces
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(trace_dir=Path(tmpdir))
        writer = MockWriter()

        print(f"\nRunning {n_samples} samples (max_concurrent={max_concurrent})...")
        start = time.perf_counter()
        results = await run_evaluation(samples, config, writer, max_concurrent=max_concurrent)
        total_time = time.perf_counter() - start

    return results, total_time


def analyze_times(n_samples: int, total_time: float, delay: float):
    """Analyze timing results."""
    if not sample_times:
        print("No timing data!")
        return

    base = min(t["start"] for t in sample_times.values())

    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Expected sequential: {n_samples * delay:.3f}s")
    print(f"Speedup: {(n_samples * delay) / total_time:.2f}x")
    print("\nTimeline:")
    for sample_id in sorted(sample_times.keys()):
        t = sample_times[sample_id]
        start_rel = t["start"] - base
        end_rel = t["end"] - base
        print(f"  {sample_id}: {start_rel:.3f}s -> {end_rel:.3f}s")

    # Check for overlap
    starts = [(t["start"] - base, sample_id) for sample_id, t in sample_times.items()]
    starts.sort()
    max_concurrent = 0
    for start, _sample_id in starts:
        active = sum(
            1 for sid, t in sample_times.items() if t["start"] - base <= start < t["end"] - base
        )
        max_concurrent = max(max_concurrent, active)

    print(f"\nMax concurrent samples: {max_concurrent}")


async def main():
    print("=" * 60)
    print("PIPELINE PARALLEL EXECUTION TEST")
    print("=" * 60)

    n = 6
    delay = 0.5

    # Test sequential
    print("\n--- SEQUENTIAL (max_concurrent=1) ---")
    results, total_time = await run_test(n, max_concurrent=1, delay=delay)
    analyze_times(n, total_time, delay)

    # Test parallel
    print("\n--- PARALLEL (max_concurrent=6) ---")
    results, total_time = await run_test(n, max_concurrent=6, delay=delay)
    analyze_times(n, total_time, delay)


if __name__ == "__main__":
    asyncio.run(main())
