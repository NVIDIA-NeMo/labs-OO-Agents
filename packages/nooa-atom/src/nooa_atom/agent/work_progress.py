# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private fixed batching for managed Shell output, not a stream-return API.

No EOF/terminal flush: the pending tail stays in the ordinary CommandResult.
An incremental preview can duplicate a prefix of that final summary. Capture
limits/files remain independent of the preview and its delivery/drop bounds.
"""

from __future__ import annotations

import asyncio
import codecs
from dataclasses import dataclass
from typing import Any, Literal

_SECONDS = 1.0
_BYTES = 8192


@dataclass(frozen=True)
class ShellProgress:
    """An incremental preview, not the final command output. Sequence is per stream."""

    stream: Literal["stdout", "stderr"]
    text: str
    sequence: int


class _Stream:
    def __init__(self, owner: _ShellProgress, name: Literal["stdout", "stderr"]):
        self.owner, self.name = owner, name
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = bytearray()
        self.timer: Any = None
        self.sequence = 0
        self.ended = False

    def feed(self, data: bytes, *, final: bool = False) -> None:
        if self.owner.closed or self.ended:
            return
        self.pending.extend(self.decoder.decode(data, final=final).encode("utf-8"))
        self.ended = final
        size = self.owner.bytes
        if len(self.pending) >= size:
            self._cancel_timer()
            while len(self.pending) >= size:
                end = size
                while end < len(self.pending) and self.pending[end] & 0xC0 == 0x80:
                    end -= 1  # never split a UTF-8 code point
                self._emit(end)
        if self.pending and self.timer is None:
            # Further fragments never postpone the oldest pending fragment.
            self.timer = self.owner.loop.call_later(self.owner.seconds, self._flush)

    def _cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def _emit(self, size: int) -> None:
        text = bytes(self.pending[:size]).decode("utf-8")
        del self.pending[:size]
        if text and self.owner.live():
            self.sequence += 1
            self.owner.sink.emit(ShellProgress(self.name, text, self.sequence))

    def _flush(self) -> None:
        self.timer = None
        if not self.owner.closed:
            self._emit(len(self.pending))

    def close(self) -> None:
        self._cancel_timer()
        self.pending.clear()
        self.decoder.reset()


class _ShellProgress:
    def __init__(self, sink, *, loop=None):
        self.sink = sink
        self.loop = loop if loop is not None else asyncio.get_running_loop()
        self._policy = (_SECONDS, _BYTES)  # immutable fixed copy for this command
        self.closed = False
        self.live = lambda: not self.closed
        self.stdout = _Stream(self, "stdout")
        self.stderr = _Stream(self, "stderr")
        sink.on_close(self.close)

    @property
    def seconds(self) -> float:
        return self._policy[0]

    @property
    def bytes(self) -> int:
        return self._policy[1]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stdout.close()
        self.stderr.close()
        self.live = lambda: False  # detach command/process references from lingering pumps
        sink, self.sink = self.sink, None
        sink.close()
