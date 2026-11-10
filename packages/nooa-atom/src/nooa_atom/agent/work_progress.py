# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional batching of managed Shell output into progress previews (``Throttle``).

A job launched with ``progress=Throttle(...)`` gets ``ShellProgress`` items
while its command runs; without one it is completion-only. There is no EOF or
terminal flush: the pending tail stays in the ordinary ``CommandResult``, which
supersedes any preview not yet delivered. Capture limits and files are
independent of previews and of their delivery bounds.
"""

from __future__ import annotations

import asyncio
import codecs
from dataclasses import dataclass
from typing import Any, Literal

from nooa_atom.agent.work_manager import MAX_BYTES_LIMIT, Throttle, check_policy

__all__ = ["MAX_BYTES_LIMIT", "ShellProgress", "Throttle"]

_MARK = "\n[... progress omitted ...]\n"


def _safe_end(data: bytes | bytearray, end: int) -> int:
    """``end`` moved back to a UTF-8 code point boundary (forward if that leaves nothing)."""
    end = min(end, len(data))
    cut = end
    while 0 < cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut -= 1
    if cut > 0 or end == len(data):
        return cut
    cut = end  # a code point longer than the batch: keep it whole
    while cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut += 1
    return cut


def _safe_start(data: bytes, start: int) -> int:
    while start < len(data) and data[start] & 0xC0 == 0x80:
        start += 1
    return start


def _keep(text: str, limit: int) -> tuple[str, int]:
    """``text`` cut to its head and tail of ``limit`` UTF-8 bytes, and the bytes left out."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, 0
    head_end = _safe_end(data, limit // 4)
    tail_start = _safe_start(data, len(data) - (limit - head_end))
    head = data[:head_end].decode("utf-8", errors="replace")
    tail = data[tail_start:].decode("utf-8", errors="replace")
    mark = _MARK.encode("utf-8")
    earlier = data[head_end:tail_start].count(mark) * len(mark)  # not output: not counted
    return head + _MARK + tail, tail_start - head_end - earlier


@dataclass(frozen=True)
class ShellProgress:
    """Output of a running command since the last preview; the CommandResult has it all.

    ``omitted`` counts bytes left out (head and tail kept) when unread batches merged.
    """

    stdout: str = ""
    stderr: str = ""
    omitted: int = 0
    batches: int = 1

    def _merge(self, later: ShellProgress, limit: int) -> tuple[ShellProgress, int]:
        """One preview of ``self`` then ``later``, each stream bounded; and batches cut."""
        out, cut_out = (
            _keep(self.stdout + later.stdout, limit) if later.stdout else (self.stdout, 0)
        )
        err, cut_err = (
            _keep(self.stderr + later.stderr, limit) if later.stderr else (self.stderr, 0)
        )
        omitted = cut_out + cut_err
        merged = ShellProgress(
            out, err, self.omitted + later.omitted + omitted, self.batches + later.batches
        )
        return merged, 1 if omitted else 0


class _Stream:
    def __init__(self, owner: _ShellProgress, name: Literal["stdout", "stderr"]):
        self.owner, self.name = owner, name
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = bytearray()
        self.lines = 0  # complete lines in pending
        self.timer: Any = None
        self.ended = False
        self.skipped = False  # bytes went by with no policy: resynchronise on UTF-8

    def feed(self, data: bytes, *, final: bool = False) -> None:
        owner = self.owner
        if owner.closed or self.ended:
            return
        policy = owner.policy
        if policy is None:
            self.skipped = self.skipped or bool(data)
            self.ended = final
            return
        if self.skipped:
            self.skipped = False
            self.decoder.reset()
            data = data[_safe_start(data, 0) :]
        decoded = self.decoder.decode(data, final=final).encode("utf-8")
        self.pending.extend(decoded)
        self.lines += decoded.count(b"\n")
        self.ended = final
        self._drain(policy)

    def _drain(self, policy: Any) -> None:
        emitted = False
        while self.pending:
            if len(self.pending) >= policy.max_bytes:
                end = _safe_end(self.pending, policy.max_bytes)
            elif policy.items is not None and self.lines >= policy.items:
                end, seen = 0, 0
                while seen < policy.items:
                    end = self.pending.index(b"\n", end) + 1
                    seen += 1
            else:
                break
            if not emitted:
                self._cancel_timer()  # the rest is newer than the oldest fragment was
                emitted = True
            self._emit(end)
            if self.owner.closed:
                return
        if self.pending and self.timer is None and policy.seconds is not None:
            # Further fragments never postpone the oldest pending fragment.
            self.timer = self.owner.loop.call_later(policy.seconds, self._flush)

    def _cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def _emit(self, size: int) -> None:
        chunk = bytes(self.pending[:size])
        del self.pending[:size]
        self.lines -= chunk.count(b"\n")
        text = chunk.decode("utf-8", errors="replace")
        if text and self.owner.live():
            self.owner.sink.emit(ShellProgress(**{self.name: text}))

    def flush_all(self) -> None:
        """Deliver what is pending (under the policy in force), in batches it allows."""
        self._cancel_timer()
        policy = self.owner.policy
        limit = policy.max_bytes if policy is not None else len(self.pending)
        while self.pending and not self.owner.closed:
            self._emit(_safe_end(self.pending, limit) or len(self.pending))

    def _flush(self) -> None:
        self.timer = None
        if not self.owner.closed:
            self.flush_all()

    def close(self) -> None:
        self._cancel_timer()
        self.pending.clear()
        self.lines = 0
        self.decoder.reset()


class _ShellProgress:
    """One command's previews: per-stream buffers and timers under the job's policy."""

    def __init__(self, sink: Any, policy: Throttle | None = None, *, loop: Any = None):
        self.sink = sink
        self.loop = loop if loop is not None else asyncio.get_running_loop()
        self.policy = check_policy(policy)
        self.closed = False
        self.live = lambda: not self.closed
        self.stdout = _Stream(self, "stdout")
        self.stderr = _Stream(self, "stderr")
        sink.on_close(self.close)
        on_policy = getattr(sink, "on_policy", None)
        if on_policy is not None:
            on_policy(self.set_policy, self.buffered)

    def set_policy(self, policy: Throttle | None) -> None:
        """Flush what is pending under the old policy once, then batch under ``policy``."""
        policy = check_policy(policy)
        if self.closed or policy == self.policy:
            return
        for stream in (self.stdout, self.stderr):
            stream.flush_all()
        self.policy = policy  # a new timer starts with the next fragment

    def buffered(self) -> int:
        """Bytes pending for the next preview."""
        return len(self.stdout.pending) + len(self.stderr.pending)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stdout.close()
        self.stderr.close()
        self.live = lambda: False  # detach command/process references from lingering pumps
        sink, self.sink = self.sink, None
        sink.close()
