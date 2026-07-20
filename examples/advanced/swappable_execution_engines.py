# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example: Swappable Execution Engines

This example demonstrates the ExecutionEngine protocol pattern. The protocol
(defined in eval_pipeline/protocol.py) allows different execution strategies
to be swapped without changing higher-level code.

Current implementation:
- ConcurrencyEngine: Async I/O with semaphore (for LLM APIs)

Future implementations can be added by implementing the ExecutionEngine protocol:
- MultiprocessEngine: Process pool (for CPU-bound tasks like local models)
- RayEngine: Distributed execution across cluster
- NemoRunEngine: HPC cluster submission (Slurm, cloud)

Usage:
    uv run python examples/advanced/swappable_execution_engines.py --tasks 20 --concurrent 5
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass

from eval_pipeline.concurrency import ConcurrencyConfig, ConcurrencyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExampleTask:
    task_id: str
    operation: str


@dataclass(frozen=True)
class ExampleResult:
    task_id: str
    result: str
    success: bool


class TestAgent:
    """Simple agent for demonstrating execution engines."""

    async def run(self, task: ExampleTask) -> ExampleResult:
        """Execute the agent."""
        await asyncio.sleep(0.1)
        return ExampleResult(
            task_id=task.task_id,
            result=f"Completed {task.operation}",
            success=True,
        )


async def run_with_engine(
    num_tasks: int = 10,
    max_concurrent: int = 5,
):
    """
    Run evaluation with the ConcurrencyEngine.

    The ExecutionEngine protocol in eval_pipeline/protocol.py defines the interface
    for swappable engines. When new engines are implemented (MultiprocessEngine,
    RayEngine, NemoRunEngine), they can expose the same run_tasks interface without changing
    higher-level code.

    Args:
        num_tasks: Number of tasks to run
        max_concurrent: Maximum concurrent executions
    """
    logger.info("Creating AsyncIO execution engine (semaphore-based)")
    engine = ConcurrencyEngine()

    tasks = {
        f"task_{i:03d}": ExampleTask(task_id=f"task_{i:03d}", operation="compute")
        for i in range(num_tasks)
    }
    task_ids = list(tasks)

    logger.info(f"Created {num_tasks} tasks")

    agent = TestAgent()

    logger.info(f"Running {num_tasks} tasks with asyncio engine...")
    logger.info(f"Max concurrent: {max_concurrent}")

    start_time = time.time()

    results = await engine.run_tasks(
        task_ids=task_ids,
        task_fn=lambda task_id: agent.run(tasks[task_id]),
        config=ConcurrencyConfig(max_concurrent=max_concurrent, timeout_seconds=30),
    )

    duration = time.time() - start_time

    # Analyze results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    logger.info("\nExecution complete!")
    logger.info("  Engine: asyncio (ConcurrencyEngine)")
    logger.info(f"  Duration: {duration:.2f}s")
    logger.info(f"  Tasks: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Throughput: {len(results) / duration:.1f} tasks/sec")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Demonstrate swappable execution engines")

    parser.add_argument(
        "--tasks",
        "-t",
        type=int,
        default=10,
        help="Number of tasks to run",
    )

    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=5,
        help="Maximum concurrent tasks",
    )

    args = parser.parse_args()

    await run_with_engine(
        num_tasks=args.tasks,
        max_concurrent=args.concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
