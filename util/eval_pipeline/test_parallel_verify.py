#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick test to verify parallel execution is actually working."""

import asyncio
import time
from dataclasses import dataclass


@dataclass
class FakeResult:
    passed: bool = True
    display_name: str = ""


async def process_sample(sample_id: int, delay: float = 0.5) -> tuple[int, float, float]:
    """Simulate processing a sample.

    Returns:
        (sample_id, start_time, end_time)
    """
    start = time.perf_counter()
    await asyncio.sleep(delay)
    end = time.perf_counter()
    return sample_id, start, end


async def run_sequential(n: int, delay: float = 0.5):
    """Run samples sequentially."""
    results = []
    for i in range(n):
        result = await process_sample(i, delay)
        results.append(result)
    return results


async def run_parallel(n: int, max_concurrent: int = 10, delay: float = 0.5):
    """Run samples in parallel with semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(sample_id: int):
        async with semaphore:
            return await process_sample(sample_id, delay)

    tasks = [process_with_semaphore(i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    return results


def analyze_results(results: list[tuple[int, float, float]], label: str):
    """Analyze timing results."""
    if not results:
        return

    base_time = results[0][1]
    total_duration = max(r[2] for r in results) - base_time

    print(f"\n{label}:")
    print(f"  Total samples: {len(results)}")
    print(f"  Total duration: {total_duration:.3f}s")
    print(f"  Expected sequential: {len(results) * 0.5:.3f}s")
    print(f"  Parallelism ratio: {(len(results) * 0.5) / total_duration:.2f}x")
    print("\n  Sample timeline (relative to start):")
    for sample_id, start, end in sorted(results):
        start_rel = start - base_time
        end_rel = end - base_time
        print(f"    Sample {sample_id}: {start_rel:.3f}s -> {end_rel:.3f}s")


async def main():
    n_samples = 6
    delay = 0.5

    print("=" * 60)
    print("PARALLEL EXECUTION VERIFICATION TEST")
    print("=" * 60)
    print(f"\nTesting with {n_samples} samples, each takes {delay}s")

    # Sequential test
    print("\n--- Running SEQUENTIAL ---")
    start = time.perf_counter()
    seq_results = await run_sequential(n_samples, delay)
    seq_time = time.perf_counter() - start
    analyze_results(seq_results, "Sequential Results")

    # Parallel test
    print("\n--- Running PARALLEL (max_concurrent=6) ---")
    start = time.perf_counter()
    par_results = await run_parallel(n_samples, max_concurrent=6, delay=delay)
    par_time = time.perf_counter() - start
    analyze_results(list(par_results), "Parallel Results")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Sequential: {seq_time:.3f}s")
    print(f"  Parallel:   {par_time:.3f}s")
    print(f"  Speedup:    {seq_time / par_time:.2f}x")
    print("=" * 60)

    if par_time < seq_time * 0.5:
        print("\n✓ Parallel execution IS working correctly!")
    else:
        print("\n✗ Parallel execution may NOT be working!")


if __name__ == "__main__":
    asyncio.run(main())
