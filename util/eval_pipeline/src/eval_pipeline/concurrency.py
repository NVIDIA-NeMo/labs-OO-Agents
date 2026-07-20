# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure concurrency engine with no domain knowledge.

This module provides generic async task execution with:
- Semaphore-based parallelism
- Timeout support
- Progress tracking
- Resume/checkpoint support

Layer 0: Knows ONLY about asyncio, task IDs, and generic callables.
Does NOT know about: agents, evaluation, scoring, benchmarks, etc.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from eval_pipeline.protocol import TaskState

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ConcurrencyConfig:
    """Pure concurrency configuration - no domain knowledge."""

    max_concurrent: int = 10
    timeout_seconds: float | None = None


class ConcurrencyEngine:
    """Pure async execution engine with no domain knowledge.

    This is a reusable concurrency primitive that knows nothing about
    evaluation, agents, or any domain concepts. It just:
    - Executes async functions in parallel
    - Enforces concurrency limits with semaphores
    - Handles timeouts
    - Supports resume from checkpoint
    - Calls progress callbacks

    The engine treats everything as opaque:
    - Task IDs are just strings
    - Task functions are just callables
    - Results are just returned values
    - It doesn't interpret any of this
    """

    async def run_tasks(
        self,
        task_ids: list[str],
        task_fn: Callable[[str], Awaitable[R]],
        config: ConcurrencyConfig,
        checkpoint_state: list[TaskState] | None = None,
        on_task_complete: Callable[[str, R | Exception], None] | None = None,
    ) -> list[R]:
        """Run tasks in parallel with concurrency control."""
        completed, results_dict, pending_ids = _split_checkpoint(task_ids, checkpoint_state)

        if pending_ids:
            semaphore = asyncio.Semaphore(config.max_concurrent)

            async def run_with_controls(task_id: str) -> R:
                try:
                    async with semaphore:
                        if config.timeout_seconds:
                            result = await asyncio.wait_for(
                                task_fn(task_id),
                                timeout=config.timeout_seconds,
                            )
                        else:
                            result = await task_fn(task_id)
                    if on_task_complete:
                        on_task_complete(task_id, result)
                    return result
                except BaseException as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    if on_task_complete:
                        on_task_complete(task_id, e)
                    raise

            tasks = [run_with_controls(tid) for tid in pending_ids]
            pending_results = await asyncio.gather(*tasks, return_exceptions=True)

            for task_id, result in zip(pending_ids, pending_results, strict=True):
                results_dict[task_id] = result

        return [results_dict[tid] for tid in task_ids]


class SubprocessEngine:
    """Subprocess-based execution engine using persistent JSON workers.

    Spawns a pool of long-lived ``subprocess_worker`` processes.
    Each worker reads JSON lines from stdin and writes JSON lines to stdout.
    Python startup + import cost is paid once per worker, not once per task.

    No fork, no pickle, no inherited state.
    """

    async def run_tasks(
        self,
        task_ids: list[str],
        task_data: dict[str, str],
        config: ConcurrencyConfig,
        checkpoint_state: list[TaskState] | None = None,
        on_task_complete: Callable[[str, R | Exception], None] | None = None,
    ) -> list[R]:
        """Run tasks via a pool of persistent subprocess workers.

        Args:
            task_ids: List of task identifiers
            task_data: Dict mapping task_id -> JSON string (SubprocessTaskInput)
            config: Concurrency config (max_concurrent = worker pool size)
            checkpoint_state: Optional previous state for resume
            on_task_complete: Optional callback after each task completes
        """
        _, results_dict, pending_ids = _split_checkpoint(task_ids, checkpoint_state)

        if pending_ids:
            # Workers are one-shot when memory limiting is active: they call
            # os._exit(0) after each task and must never be reused.
            import json as _json

            _first = _json.loads(task_data[pending_ids[0]])
            _one_shot = bool(_first.get("memory_limit_mb"))
            pool = _WorkerPool(max_workers=config.max_concurrent, one_shot=_one_shot)
            try:
                pending_results = await pool.map(
                    [(tid, task_data[tid]) for tid in pending_ids],
                    on_task_complete=on_task_complete,
                )
                for task_id, result in zip(pending_ids, pending_results, strict=True):
                    results_dict[task_id] = result
            finally:
                pool.shutdown()

        return [results_dict[tid] for tid in task_ids]


def _split_checkpoint(
    task_ids: list[str],
    checkpoint_state: list[TaskState] | None,
) -> tuple[dict, dict, list[str]]:
    """Split task_ids into completed (from checkpoint) and pending."""
    completed: dict[str, TaskState] = {}
    if checkpoint_state:
        completed = {
            s.task_id: s
            for s in checkpoint_state
            if s.completed and s.result is not None and not s.error
        }
    results_dict: dict[str, Any] = {}
    pending_ids: list[str] = []
    for tid in task_ids:
        if tid in completed:
            results_dict[tid] = completed[tid].result
        else:
            pending_ids.append(tid)
    return completed, results_dict, pending_ids


class _WorkerPool:
    """Pool of persistent subprocess workers communicating via JSON lines."""

    def __init__(self, max_workers: int = 4, one_shot: bool = False):
        self._max_workers = max_workers
        # one_shot=True: workers exit after every task (e.g. --memory-limit).
        # Never put a completed worker back in the free queue in this mode —
        # os._exit(0) and waitpid(WNOHANG) have a race on macOS where poll()
        # returns None for a brief window after the process has actually exited,
        # causing the dead worker to be reused for the next task.
        self._one_shot = one_shot
        self._workers: list = []
        self._free: asyncio.Queue = asyncio.Queue()
        try:
            for _ in range(max_workers):
                self._spawn_worker()
        except Exception:
            # Clean up already-started workers on partial failure
            self.shutdown()
            raise

    def _spawn_worker(self) -> None:
        import subprocess
        import sys

        proc = subprocess.Popen(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent's stderr
        )
        self._workers.append(proc)
        self._free.put_nowait(proc)

    async def map(
        self,
        items: list[tuple[str, str]],
        on_task_complete: Callable | None = None,
    ) -> list[Any]:
        """Run tasks across the worker pool. Returns results in order."""
        loop = asyncio.get_running_loop()

        async def run_one(task_id: str, json_input: str) -> Any:
            worker = await self._free.get()
            try:
                result = await loop.run_in_executor(None, _send_to_worker, worker, json_input)
            except Exception as e:
                self._replace_dead_worker(worker)
                result = e
            else:
                if not self._one_shot and worker.poll() is None:
                    self._free.put_nowait(worker)
                else:
                    self._replace_dead_worker(worker)

            if on_task_complete:
                try:
                    on_task_complete(task_id, result)
                except Exception:
                    pass  # Don't let callback errors replace the result
            return result

        tasks = [run_one(tid, data) for tid, data in items]
        return list(await asyncio.gather(*tasks, return_exceptions=True))

    def _replace_dead_worker(self, dead: Any) -> None:
        """Kill a dead/broken worker and spawn a replacement."""
        try:
            dead.kill()
        except Exception:
            pass
        if dead in self._workers:
            self._workers.remove(dead)
        try:
            self._spawn_worker()
        except Exception:
            import logging

            logging.getLogger(__name__).warning("Failed to spawn replacement worker", exc_info=True)

    def shutdown(self) -> None:
        """Close stdin on all workers and wait for them to exit."""
        for proc in self._workers:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def _send_to_worker(worker, json_input: str) -> Any:
    """Send one task to a persistent worker and read one result.

    Runs in a thread (via run_in_executor). Raises on failure.
    """
    worker.stdin.write(json_input.encode())
    worker.stdin.write(b"\n")
    worker.stdin.flush()

    result_line = worker.stdout.readline()
    if not result_line:
        exit_code = worker.poll()
        if exit_code is not None and exit_code < 0:
            import signal as _signal

            try:
                sig_name = _signal.Signals(-exit_code).name
            except (ValueError, AttributeError):
                sig_name = str(-exit_code)
            raise RuntimeError(
                f"Worker killed by {sig_name} (exit code {exit_code}). "
                f"Likely out-of-memory (OOM kill)."
            )
        raise RuntimeError(
            f"Worker closed stdout unexpectedly (exit code {exit_code})"
            if exit_code is not None
            else "Worker closed stdout unexpectedly"
        )

    from eval_pipeline.eval_types import EvalTestResult

    return EvalTestResult.model_validate_json(result_line)
