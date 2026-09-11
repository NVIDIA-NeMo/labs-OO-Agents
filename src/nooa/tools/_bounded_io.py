# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory file primitives backing ``ShellTools`` file operations.

A coding agent points its file tools at whatever the repository contains,
including generated, minified, or otherwise untrusted files. Reading such a file
whole — to hand back five lines of it, or to check that a search string occurs
exactly once — lets the file's size, rather than the request's size, decide how
much memory the agent's process needs.

The primitives here bound that cost:

- :func:`iter_lines_keepends` walks a file in fixed-size chunks, holding at most
  one chunk plus one line.
- :func:`read_line_range` reads only as far as the requested range.
- :func:`read_specific_lines` keeps only the lines actually asked for, so cost
  scales with the number of matches rather than the size of the file.
- :func:`read_whole_file_checked` refuses an unbounded read above a byte budget
  instead of discovering the size after allocating it.
- :func:`atomic_replace_text` swaps contents through a sibling temp file, so a
  failed write leaves the original intact.

**Line boundaries follow** ``str.splitlines()``, not file iteration. Iterating a
file object splits on ``\\n`` alone, while ``str.splitlines()`` also breaks on
``\\v``, ``\\f``, ``\\x1c``-``\\x1e``, ``\\x85``, ``\\u2028`` and ``\\u2029``.
The previous implementation sliced ``read_text().splitlines(keepends=True)``, so
adopting file iteration would silently renumber every :class:`Match` in a file
containing any of those characters. These helpers reproduce the
``splitlines()`` boundary set exactly.

Text decoding also matches the previous code path: files are opened with the
locale default encoding and universal-newline translation, the same as
``Path.read_text()``.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

#: Default ceiling for a single unbounded file read or replacement.
#:
#: Source files are overwhelmingly under a megabyte; the headroom here is for
#: checked-in generated output (bundles, lockfiles, fixtures) that an agent may
#: legitimately need to edit. It is deliberately a bound rather than a guess at
#: the largest useful file — callers that genuinely need more pass
#: ``ShellTools(max_file_bytes=...)`` and make that choice explicit.
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024

#: Characters read per chunk while walking a file. Bounds peak memory together
#: with the longest line encountered.
CHUNK_CHARS = 64 * 1024


class FileTooLargeError(ValueError):
    """A file operation exceeded its byte budget.

    Subclasses :class:`ValueError` on purpose: ``ShellTools`` already raises
    ``ValueError`` for unusable file arguments, and ``_harvest_matches`` guards
    its reads with ``except (OSError, ValueError)``. Inheriting keeps both
    call sites behaving as they did without a new exception type to catch.
    """

    def __init__(self, path: Path | str, size: int, limit: int, hint: str, summary: str) -> None:
        self.path = str(path)
        self.size = size
        self.limit = limit
        super().__init__(f"{self.path} {summary}. {hint}")


def iter_lines_keepends(
    path: Path,
    *,
    max_line_chars: int,
    chunk_chars: int = CHUNK_CHARS,
) -> Iterator[str]:
    """Yield lines exactly as ``read_text().splitlines(keepends=True)`` would.

    Peak memory is one chunk plus the longest line, instead of the whole file.

    Args:
        path: File to read.
        max_line_chars: Ceiling on a single line. A file with no line break in
            this many characters would otherwise defeat chunking, so it raises
            :class:`FileTooLargeError` rather than growing without limit.
        chunk_chars: Characters per read.

    Raises:
        FileTooLargeError: A single line exceeded ``max_line_chars``.
        OSError: The file could not be opened or read.
    """
    carry = ""
    with open(path) as handle:
        while True:
            chunk = handle.read(chunk_chars)
            if not chunk:
                break
            parts = (carry + chunk).splitlines(keepends=True)
            # The trailing element may be an incomplete line, so it is held back
            # until the next chunk confirms where it ends.
            carry = parts.pop() if parts else ""
            if len(carry) > max_line_chars:
                raise FileTooLargeError(
                    path,
                    len(carry),
                    max_line_chars,
                    "A single line cannot be read in bounded chunks. Raise the budget "
                    "with ShellTools(max_file_bytes=...) if this file is safe to load "
                    "whole.",
                    f"has a line over {max_line_chars:,} characters",
                )
            yield from parts
    if carry:
        yield carry


def read_line_range(
    path: Path,
    start: int,
    end: int,
    *,
    max_line_chars: int,
) -> tuple[str, int]:
    """Read lines ``[start, end]`` (1-indexed, inclusive) without reading past ``end``.

    Mirrors the slice it replaces —
    ``read_text().splitlines(keepends=True)[start - 1 : end]`` — including its
    clamping: ``end`` is reported unchanged when the file has at least that many
    lines, and lowered to the real line count only when the file ends first.
    That equivalence is what makes reading no further than ``end`` sound: having
    read ``end`` lines successfully already proves ``min(total, end) == end``.

    Returns:
        ``(text, clamped_end)`` — the joined region and the line number to
        report as :attr:`Match.end`.
    """
    if end < 1:
        # A non-positive end selects nothing. The old code reached Python's
        # negative-index slicing here and returned a reversed region for
        # end < 0, which no caller can have meant; an empty region is reported
        # instead.
        return "", end

    pieces: list[str] = []
    seen = 0
    for seen, line in enumerate(  # noqa: B007 - final value is the line count
        iter_lines_keepends(path, max_line_chars=max_line_chars), 1
    ):
        if seen >= start:
            pieces.append(line)
        if seen >= end:
            break
    return "".join(pieces), min(end, seen)


def read_specific_lines(
    path: Path,
    wanted: set[int],
    *,
    max_line_chars: int,
) -> dict[int, str]:
    """Return ``{line_number: text}`` for ``wanted``, reading no further than needed.

    Used for search-anchor harvesting, where a handful of line numbers are
    needed out of a possibly enormous file. Memory scales with ``len(wanted)``
    rather than the file's size, and reading stops after the highest requested
    line.

    Line numbers past the end of the file are simply absent from the result,
    which lets the caller keep its fail-closed "no trustworthy anchors" path.
    """
    if not wanted:
        return {}
    highest = max(wanted)
    if highest < 1:
        return {}

    found: dict[int, str] = {}
    for seen, line in enumerate(iter_lines_keepends(path, max_line_chars=max_line_chars), 1):
        if seen in wanted:
            found[seen] = line
        if seen >= highest:
            break
    return found


def read_whole_file_checked(path: Path, max_bytes: int, *, hint: str) -> str:
    """Read an entire file, refusing above ``max_bytes``.

    The size is taken from ``stat()`` before opening, so an oversized file is
    rejected without ever being allocated.

    Raises:
        FileTooLargeError: The file is larger than ``max_bytes``.
        OSError: The file could not be stat'd or read.
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise FileTooLargeError(
            path, size, max_bytes, hint, f"is {size:,} bytes, over the {max_bytes:,}-byte limit"
        )
    return path.read_text()


def atomic_replace_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically, preserving its permission bits.

    ``Path.write_text()`` truncates in place, so an interrupted or failing write
    leaves the file half-written. This writes a sibling temp file, flushes it to
    disk, copies the original's mode, and swaps it in with ``os.replace()``,
    which is atomic on POSIX and Windows alike. On any failure the original is
    left exactly as it was and the temp file is removed.

    Copying the mode across matters for a coding agent: replacing an executable
    script must not silently drop its executable bit.

    The temp file is created in ``path``'s own directory so the final swap
    cannot cross a filesystem boundary, and encoding and newline handling match
    ``Path.write_text()``.
    """
    mode: int | None = None
    with contextlib.suppress(OSError):
        mode = path.stat().st_mode

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        if mode is not None:
            os.chmod(tmp_name, mode & 0o7777)
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
