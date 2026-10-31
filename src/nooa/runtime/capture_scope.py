# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner/lifetime leases on the existing ContextVarStream capture plumbing.

Experimental callers opt in. Descendant writes use the originating OPEN scope;
late writes are dropped, never redirected to a later cell or a closed file.
Managed jobs install their own
scope and publish its bounded output through the job's existing delivery.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from nooa.agentdoc import FileBackedTruncatingStringIO
from nooa.runtime.media_capture import _media_buffer_var, _MediaBuffer
from nooa.runtime.stream_wrappers import BlockedStdinWrapper, ContextVarStream, _block_stdin_var


class _TextLease:
    def __init__(self, scope: CaptureScope, buffer: Any):
        self.scope, self.buffer = scope, buffer

    def write(self, data: str) -> int:
        with self.scope.lock:
            if self.scope.accepts_write():
                return self.buffer.write(data)
            self.scope.dropped_chars += len(data)
            return len(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        with self.scope.lock:
            if self.scope.accepts_write():
                self.buffer.flush()

    def getvalue(self):
        return self.buffer.getvalue()


class CaptureScope:
    """Bounded existing buffers with shared originating-scope lifetime."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.active = False
        self.dropped_chars = 0
        self._tokens = []
        self._finished = None

    def accepts_write(self) -> bool:
        # Output capture is inherited, not a completion capability. Awaited
        # gather/to_thread descendants must preserve ordinary capture semantics.
        # WorkManager replaces the scope for managed jobs; every inherited copy
        # shares this liveness bit, so no late write can reach a finished sink.
        return self.active

    def __enter__(self):
        from nooa.runtime.actor import _make_capture_buffer, _stderr_buffer_var, _stdout_buffer_var

        self.active = True
        capture = self.config.capture
        files = {
            "file_backed": capture.file_backed,
            "file_dir": capture.file_dir,
            "file_notice": capture.file_notice,
        }
        self.stdout = _TextLease(
            self, _make_capture_buffer(limit=capture.max_stdout, tail_chars=capture.tail, **files)
        )
        self.stderr = _TextLease(
            self, _make_capture_buffer(limit=capture.max_stderr, tail_chars=capture.tail, **files)
        )
        self.media = _MediaBuffer(
            max_attachments=self.config.media_capture.max_attachments_per_execution,
            accepts_write=self.accepts_write,
            write_lock=self.lock,
        )
        for variable, value in (
            (_stdout_buffer_var, self.stdout),
            (_stderr_buffer_var, self.stderr),
            (_media_buffer_var, self.media),
            (_block_stdin_var, True),
        ):
            self._tokens.append((variable, variable.set(value)))
        if not isinstance(sys.stdout, ContextVarStream):
            sys.stdout = ContextVarStream(sys.stdout, _stdout_buffer_var, "stdout")
        if not isinstance(sys.stderr, ContextVarStream):
            sys.stderr = ContextVarStream(sys.stderr, _stderr_buffer_var, "stderr")
        if not isinstance(sys.stdin, BlockedStdinWrapper):
            sys.stdin = BlockedStdinWrapper(sys.stdin)
        return self

    def finish(self):
        """Freeze once before publishing completion. Does not reset task tokens."""
        from nooa.events import ExecutionResult

        with self.lock:
            if self._finished is None:
                self.active = False
                self._finished = ExecutionResult(
                    stdout=self.stdout.getvalue(),
                    stderr=self.stderr.getvalue(),
                    images=list(self.media.blocks),
                )
            return self._finished

    def __exit__(self, *exc):
        self.finish()
        for variable, token in reversed(self._tokens):
            variable.reset(token)
        self._tokens.clear()
        for lease in (self.stdout, self.stderr):
            buffer = lease.buffer
            if isinstance(buffer, FileBackedTruncatingStringIO) and not buffer.was_truncated:
                buffer.cleanup()  # whole in the result: no file to keep
            else:
                buffer.close()  # a cut text names its file: its directory's owner removes it
