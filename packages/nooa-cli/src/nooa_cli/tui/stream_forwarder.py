# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward stray ``sys.stdout`` / ``sys.stderr`` writes into the TUI scrollback.

Libraries (aiohttp, litellm, warnings, third-party logs, etc.) sometimes
write directly to ``sys.stdout`` / ``sys.stderr``. Under
prompt_toolkit those raw writes interleave with the TUI's cursor
commands and corrupt the display.

``_StrayStreamForwarder`` replaces the real stream with a line-buffered
wrapper that routes each complete line through ``emit_block`` — the
same pipeline the TUI uses for code previews and agent messages — so
the output lands in the transcript above the prompt instead of over
top of it. Styled with a small prefix (``·`` dim for stdout, ``!`` red
for stderr) so the user can tell it apart from first-class output.

Deduplication: repeated identical lines are collapsed into a single
emission with a repeat count (e.g. ``· rate_limit (×5)``), reducing
noise during LLM retry storms while still surfacing what happened.

Installation order matters: wrap sys.stdout BEFORE the framework
installs its own ``ContextVarStream`` wrapper. That way the framework's
wrapper's ``_original`` becomes our forwarder, and agent-cell stdout
capture still works — the framework only forwards when no contextvar
buffer is set, i.e. exactly the stray-write case we care about.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import IO, Any

from .terminal_safety import (
    fallback_transcript_columns,
    normalize_transcript_block,
    sanitize_live_text,
)


class _StrayStreamForwarder:
    """Thread-safe line-buffered forwarder for stray stdout/stderr writes.

    ``write()`` accumulates partial lines in an internal buffer and flushes
    each complete line through ``emit_block`` as an ANSI-styled chunk.
    Non-write attribute access (``encoding``, ``fileno``, etc.) falls
    through to the wrapped stream so the object passes duck-type checks
    elsewhere in the process (e.g. ``sys.stdout.isatty()``).

    Deduplication: consecutive identical lines are collapsed. When a new
    different line arrives (or ``flush()`` is called), the collapsed count
    is emitted.
    """

    def __init__(
        self,
        original: IO[str],
        emit_block: Callable[[str], None],
        *,
        prefix: str,
        ansi_color: str,
        on_stray: Callable[[str, str], None] | None = None,
    ) -> None:
        self._original = original
        self._emit_block = emit_block
        self._prefix = prefix
        self._ansi_color = ansi_color
        self._on_stray: Callable[[str, str], None] | None = on_stray
        self._pending = ""
        self._lock = threading.RLock()
        # Dedup state
        self._last_stripped: str = ""
        self._last_line: str = ""
        self._repeat_count: int = 0

    # ── IO protocol ────────────────────────────────────────────────────

    def write(self, data: str) -> int:
        """Accept a write; emit each newline-terminated line via ``emit_block``.

        Returns the number of characters accepted (following ``io.TextIOBase``).
        """
        if not data:
            return 0
        with self._lock:
            buf = self._pending + data
            lines = buf.split("\n")
            self._pending = lines[-1]
            complete = lines[:-1]
            for line in complete:
                self._emit_line(line)
        return len(data)

    def writelines(self, lines: list[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        """Flush any buffered partial line.

        Libraries that write ``"foo"`` then ``flush()`` without a trailing
        newline still want ``foo`` visible; flushing treats the pending
        buffer as a complete line.
        """
        with self._lock:
            pending = self._pending
            self._pending = ""
            if pending:
                self._emit_line(pending)
            self._flush_dedup()

    # ── internals ──────────────────────────────────────────────────────

    # Patterns suppressed entirely — pure noise for TUI users.
    _NOISE_PATTERNS = (
        "Give Feedback / Get Help:",
        "LiteLLM.Info:",
        "Provider List: https://docs.litellm.ai/docs/providers",
        "LiteLLM completion()",
        "litellm.litellm_core_utils",
    )

    def _emit_line(self, line: str) -> None:
        """Wrap a single line in the configured ANSI style and emit it."""
        # Even the fallback below can write to the real terminal.  Normalize
        # before dedup/styling so subprocess cursor controls, BEL, CR, OSC,
        # and malformed CSI can never escape by making ``emit_block`` fail.
        line = sanitize_live_text(line).replace("\n", "\\n")
        # Suppress purely empty/whitespace lines.
        stripped = line.strip()
        if not stripped:
            return

        # Suppress known noise patterns.
        if any(pat in stripped for pat in self._NOISE_PATTERNS):
            self._notify_stray(stripped, "suppressed")
            return

        # Deduplication: collapse consecutive identical lines.
        if stripped == self._last_stripped:
            self._repeat_count += 1
            self._notify_stray(stripped, "repeated")
            return

        # Different line — flush any pending dedup, then emit.
        self._flush_dedup()
        self._last_stripped = stripped
        self._last_line = line
        self._repeat_count = 1

        styled = f"\x1b[{self._ansi_color}m{self._prefix}{line}\x1b[0m\n"
        self._emit_styled(styled)

        self._notify_stray(stripped, "emitted")

    def _emit_styled(self, styled: str) -> None:
        """Use the transcript pipeline, with one normalized terminal fallback."""
        try:
            self._emit_block(styled)
        except Exception:
            try:
                self._original.write(
                    normalize_transcript_block(
                        styled,
                        columns=fallback_transcript_columns(),
                    )
                )
                self._original.flush()
            except Exception:
                pass

    def _flush_dedup(self) -> None:
        """Emit a repeat summary if the last line was seen more than once."""
        if self._repeat_count > 1:
            summary = f"{self._last_line} (×{self._repeat_count - 1} more)"
            styled = f"\x1b[{self._ansi_color}m{self._prefix}{summary}\x1b[0m\n"
            self._emit_styled(styled)
        self._last_stripped = ""
        self._last_line = ""
        self._repeat_count = 0

    def _notify_stray(self, content: str, disposition: str) -> None:
        """Notify the on_stray callback if registered."""
        if self._on_stray is not None:
            try:
                self._on_stray(content, disposition)
            except Exception:
                pass

    # ── duck-type compatibility ────────────────────────────────────────

    def isatty(self) -> bool:
        """Report non-tty so libraries don't emit ANSI escape sequences
        (which would otherwise show up as literal characters in the line
        the forwarder emits)."""
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def install_stray_stream_capture(
    emit_block: Callable[[str], None],
    *,
    on_stray: Callable[[str, str], None] | None = None,
) -> Callable[[], None]:
    """Install forwarders on ``sys.stdout`` and ``sys.stderr``.

    Returns a zero-arg ``uninstall`` callable that restores whatever
    streams were in place *before* this call. Call it in the TUI's
    shutdown path so post-exit prints go back to the real terminal.

    Args:
        emit_block: Callable that renders a styled string block in the TUI.
        on_stray: Optional callback ``(content, disposition)`` invoked for
            every intercepted line. ``disposition`` is one of ``"emitted"``,
            ``"repeated"``, or ``"suppressed"``. Use this to emit hidden
            runtime events for later inspection via ``/events``.

    Must be invoked on the main thread before any agent code runs —
    otherwise the framework's ``ContextVarStream`` will be layered
    underneath our wrapper and agent-cell stdout capture will break.
    """
    import sys

    prior_stdout = sys.stdout
    prior_stderr = sys.stderr

    sys.stdout = _StrayStreamForwarder(  # type: ignore[assignment]
        prior_stdout,
        emit_block,
        prefix="· ",
        ansi_color="2",  # dim
        on_stray=on_stray,
    )
    sys.stderr = _StrayStreamForwarder(  # type: ignore[assignment]
        prior_stderr,
        emit_block,
        prefix="! ",
        ansi_color="31",  # red
        on_stray=on_stray,
    )

    def uninstall() -> None:
        # Flush any partial lines before restoring original streams.
        # Always restore even if flush raises, to avoid stale forwarder objects.
        try:
            if hasattr(sys.stdout, "flush"):
                sys.stdout.flush()
        except Exception:
            pass
        try:
            if hasattr(sys.stderr, "flush"):
                sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = prior_stdout
        sys.stderr = prior_stderr

    return uninstall
