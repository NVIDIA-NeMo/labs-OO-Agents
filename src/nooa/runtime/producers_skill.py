# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Producers Skill — spawn helpers for background jobs.

Attached to the agent as ``self.producers``. Methods return coroutines
or async generators suitable for ``self.queue_manager.spawn(...)``.
"""

from nooa.skill import Skill

from . import producers as _mod


class ProducersSkill(Skill):
    """Background job producers for QueueManager.spawn().

    Each method returns a coroutine or async generator. Pass the result
    to ``self.queue_manager.spawn(self.producers.<method>(...), channel="name", buffer=N)``.
    """

    def monitor(self, cmd: str):
        """Stream stdout lines from a shell command.

        Yields each line as it appears. stderr merged into stdout.
        """
        return _mod.monitor(cmd)

    def after(self, delay: float):
        """One-shot timer: returns None after *delay* seconds."""
        return _mod.after(delay)

    def cron(self, interval_seconds: float):
        """Periodic tick counter at fixed intervals. Yields incrementing int."""
        return _mod.cron(interval_seconds)

    def tail(self, path: str, *, poll_interval: float = 0.5):
        """Tail a file, yielding new lines as they appear."""
        return _mod.tail(path, poll_interval=poll_interval)

    def run_job(self, coro, job_id: str):
        """Wrap a coroutine with a job_id. Returns {"job_id": ..., "result": ...}."""
        return _mod.run_job(coro, job_id)
