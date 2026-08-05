# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI-local command execution queue.

This is intentionally separate from ``agent.queue_manager.spawn()``. Agent jobs
are background agent work; command records are user-submitted TUI commands
(``/...`` and ``!...``) with UI lifecycle/status.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from .output import CommandStatus, Output

CommandKind = Literal["slash", "bang"]
CommandState = Literal["queued", "running", "done", "failed", "cancelled"]


@dataclass
class CommandRecord:
    """One user-submitted TUI command invocation."""

    id: int
    kind: CommandKind
    text: str
    state: CommandState = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str = ""


class CommandRunner:
    """Serialize TUI command execution and render lifecycle status."""

    def __init__(
        self,
        render: Callable[[Output], Awaitable[None]],
        *,
        set_dynamic_status: Callable[[str], None] | None = None,
        set_dynamic_queue: Callable[[list[str]], None] | None = None,
        history_size: int = 50,
    ):
        self._render = render
        self._set_dynamic_status = set_dynamic_status or (lambda _text: None)
        self._set_dynamic_queue = set_dynamic_queue or (lambda _commands: None)
        self._status_generation = 0
        self._spinner_frames = "·•"
        self._spinner_frame = self._spinner_frames[0]
        self._spinner_task: asyncio.Task[None] | None = None
        self._history_size = history_size
        self._ids = itertools.count(1)
        self._records: list[CommandRecord] = []
        self._queue: asyncio.Queue[
            tuple[CommandRecord, Callable[[], Awaitable[Any]], asyncio.Future[None]]
        ] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._active_record: CommandRecord | None = None
        self._active_work: asyncio.Task[Any] | None = None
        self._cancel_requested: set[int] = set()

    @property
    def records(self) -> list[CommandRecord]:
        return list(self._records)

    async def run(
        self,
        *,
        kind: CommandKind,
        text: str,
        work: Callable[[], Awaitable[Any]],
    ) -> None:
        """Enqueue one command and wait for it to finish.

        Waiting here does not block prompt_toolkit's loop; callers await a Future
        while the runner worker performs the command. Multiple command callback
        tasks can therefore submit concurrently while execution stays ordered.
        """
        record = CommandRecord(id=next(self._ids), kind=kind, text=text)
        self._records.append(record)
        if len(self._records) > self._history_size:
            del self._records[: len(self._records) - self._history_size]
        self._update_dynamic_status()

        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        await self._queue.put((record, work, done))
        self._ensure_worker()
        await done

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_loop(), name="tui-command-runner")

    def cancel_active(self, *, kind: CommandKind | None = None) -> bool:
        """Request cancellation of the active command without dropping its queue.

        Returns ``True`` only when a matching running command owned an active
        task. Cancellation is acknowledged by ``_run_loop`` before the next
        queued command starts. Work delegated to an ordinary Python thread
        cannot be force-killed; its owning coroutine is still cancelled and
        must prevent any late result from being attached.
        """
        record = self._active_record
        task = self._active_work
        if record is None or task is None or task.done():
            return False
        if kind is not None and record.kind != kind:
            return False
        self._cancel_requested.add(record.id)
        task.cancel()
        return True

    async def _run_loop(self) -> None:
        while True:
            try:
                record, work, done = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                record.state = "running"
                record.started_at = datetime.now()
                self._update_dynamic_status()
                self._active_record = record
                self._active_work = asyncio.create_task(
                    work(), name=f"tui-{record.kind}-command-{record.id}"
                )
                post_done = await self._active_work
                if record.id in self._cancel_requested:
                    raise asyncio.CancelledError
            except asyncio.CancelledError as exc:
                record.state = "cancelled"
                record.finished_at = datetime.now()
                requested = record.id in self._cancel_requested
                self._update_dynamic_status()
                await self._render_status(record)
                if not done.done():
                    if requested:
                        done.set_result(None)
                    else:
                        done.set_exception(exc)
                if not requested:
                    raise
            except Exception as exc:  # noqa: BLE001 - surface command failures to scrollback
                record.state = "failed"
                record.finished_at = datetime.now()
                record.error = f"{type(exc).__name__}: {exc}"
                if not done.done():
                    done.set_result(None)
                self._update_dynamic_status()
                await self._render_status(record)
            else:
                record.state = "done"
                record.finished_at = datetime.now()
                self._update_dynamic_status()
                await self._render_status(record)
                try:
                    if callable(post_done):
                        await post_done()
                except Exception as exc:  # noqa: BLE001 - surface deferred render failures
                    record.state = "failed"
                    record.error = f"post-completion render failed: {type(exc).__name__}: {exc}"
                    await self._render_status(record)
                finally:
                    if not done.done():
                        done.set_result(None)
            finally:
                if self._active_record is record:
                    self._active_record = None
                    self._active_work = None
                self._cancel_requested.discard(record.id)
                self._queue.task_done()

    def _set_status(self, text: str) -> None:
        self._status_generation += 1
        self._set_dynamic_status(text)

    def _running_status_text(self) -> str:
        running = [r for r in self._records if r.state == "running"]
        if not running:
            return ""
        current = running[0]
        return f"{self._spinner_frame} {current.text}"

    def _ensure_spinner_task(self) -> None:
        if self._spinner_task is not None and not self._spinner_task.done():
            return

        async def _animate() -> None:
            i = 1
            while any(r.state == "running" for r in self._records):
                await asyncio.sleep(0.5)
                if not any(r.state == "running" for r in self._records):
                    break
                self._spinner_frame = self._spinner_frames[i % len(self._spinner_frames)]
                self._set_status(self._running_status_text())
                i += 1

        self._spinner_task = asyncio.create_task(_animate(), name="tui-command-spinner")

    def _update_dynamic_status(self) -> None:
        running = [r for r in self._records if r.state == "running"]
        queued = [r for r in self._records if r.state == "queued"]
        self._set_dynamic_queue([r.text for r in queued])
        if running:
            self._set_status(self._running_status_text())
            self._ensure_spinner_task()
        elif queued:
            self._set_status(f"○ {len(queued)} queued")
        else:
            self._set_status("")

    async def _render_status(self, record: CommandRecord) -> None:
        await self._render(
            CommandStatus(
                id=record.id,
                kind=record.kind,
                state=record.state,
                text=record.text,
                error=record.error,
            )
        )
