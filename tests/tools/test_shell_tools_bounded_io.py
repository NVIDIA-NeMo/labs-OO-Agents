# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory guarantees for ShellTools file operations (#90).

Strategy:
- **Differential oracle.** Ranged reads must return exactly what the previous
  ``read_text().splitlines(keepends=True)[start - 1 : end]`` slice returned,
  including for files containing the line-break characters ``str.splitlines()``
  recognises but file iteration does not. ``_legacy_read`` below is that old
  implementation, kept verbatim as the reference.
- **Proof of absence.** To show a ranged read never takes a whole-file path,
  ``Path.read_text`` is replaced with something that raises. A test that passes
  with the unbounded path sabotaged cannot be reaching it.
- **Measured bound.** ``tracemalloc`` caps peak allocation while reading the
  *end* of a large file, which forces the whole file to be streamed and so
  actually exercises the bound.
- **Durability.** Oversized and failing writes must leave the original bytes,
  permissions, and inode contents untouched.
"""

from __future__ import annotations

import os
import shutil
import stat
import tracemalloc
from pathlib import Path

import pytest

from nooa.tools._bounded_io import (
    DEFAULT_MAX_FILE_BYTES,
    FileTooLargeError,
    atomic_replace_text,
    iter_lines_keepends,
    read_specific_lines,
)
from nooa.tools.shell_tools import Match, ShellTools

# Characters str.splitlines() breaks on that iterating a file object does not.
# A naive streaming rewrite renumbers every Match in a file containing these.
EXOTIC_BREAKS = "\v\f\x1c\x1d\x1e\x85\u2028\u2029"


def _legacy_read(path: Path, start: int, end: int) -> tuple[str, int, int]:
    """The pre-#90 implementation, kept as the differential reference."""
    content = path.read_text()
    all_lines = content.splitlines(keepends=True)
    total = len(all_lines)
    start = max(1, start)
    end = min(total, end)
    return "".join(all_lines[start - 1 : end]), start, end


@pytest.fixture
def sh(tmp_path):
    return ShellTools(cwd=str(tmp_path))


def _write_lines(path: Path, count: int, width: int = 64) -> None:
    path.write_text("".join(f"{i:0{width}d}\n" for i in range(1, count + 1)))


# --------------------------------------------------------------------------
# Ranged reads consume only the requested region
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranged_read_never_takes_whole_file_path(sh, tmp_path, monkeypatch):
    _write_lines(tmp_path / "big.txt", 5_000)

    def _explode(*args, **kwargs):
        raise AssertionError("ranged read fell back to an unbounded read_text()")

    monkeypatch.setattr(Path, "read_text", _explode)

    view = await sh.read("big.txt", (3, 5))
    assert view.start == 3
    assert view.end == 5
    assert view.text.splitlines() == [f"{i:064d}" for i in (3, 4, 5)]


@pytest.mark.asyncio
async def test_ranged_read_stops_at_end_of_range(sh, tmp_path, monkeypatch):
    """Reading lines 1-2 must not touch the rest of the file."""
    _write_lines(tmp_path / "big.txt", 10_000)

    reads: list[int] = []
    real_open = open

    class _CountingHandle:
        """Proxy recording characters read. TextIOWrapper takes no attributes."""

        def __init__(self, inner):
            self._inner = inner

        def read(self, size: int = -1) -> str:
            chunk = self._inner.read(size)
            reads.append(len(chunk))
            return chunk

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    def counting_open(*args, **kwargs):
        return _CountingHandle(real_open(*args, **kwargs))

    monkeypatch.setattr("nooa.tools._bounded_io.open", counting_open, raising=False)
    view = await sh.read("big.txt", (1, 2))

    assert view.text.splitlines() == [f"{i:064d}" for i in (1, 2)]
    # 650 KB of file; a range of two lines must not have pulled all of it.
    assert sum(reads) < (tmp_path / "big.txt").stat().st_size


@pytest.mark.parametrize(
    "start,end",
    [
        (1, 1),
        (1, 3),
        (2, 4),
        (3, 3),
        (1, 999),  # end past EOF -> clamps to the real line count
        (900, 999),  # entirely past EOF -> empty region
        (5, 2),  # inverted -> empty region
        (0, 2),  # start clamps up to 1
        (-4, 2),  # negative start clamps up to 1
    ],
)
@pytest.mark.asyncio
async def test_ranged_read_matches_legacy_slice(sh, tmp_path, start, end):
    path = tmp_path / "f.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n")

    want_text, want_start, want_end = _legacy_read(path, start, end)
    got = await sh.read("f.txt", (start, end))

    assert got.text == want_text
    assert got.start == want_start
    assert got.end == want_end


@pytest.mark.parametrize("breaker", list(EXOTIC_BREAKS))
@pytest.mark.asyncio
async def test_exotic_line_breaks_keep_legacy_numbering(sh, tmp_path, breaker):
    """str.splitlines() boundaries are preserved, not \\n-only file iteration."""
    path = tmp_path / "f.txt"
    path.write_text(f"one{breaker}two\nthree{breaker}four\nfive\n")

    for start, end in ((1, 1), (1, 3), (2, 4), (3, 6)):
        want_text, want_start, want_end = _legacy_read(path, start, end)
        got = await sh.read("f.txt", (start, end))
        assert (got.text, got.start, got.end) == (want_text, want_start, want_end), (
            f"diverged for {breaker!r} at ({start}, {end})"
        )


def test_iter_lines_matches_splitlines_exactly(tmp_path):
    path = tmp_path / "f.txt"
    body = f"a{EXOTIC_BREAKS}b\nc\r\nd\re\nlast-no-newline"
    path.write_text(body)

    assert list(iter_lines_keepends(path, max_line_chars=DEFAULT_MAX_FILE_BYTES)) == (
        path.read_text().splitlines(keepends=True)
    )


def test_iter_lines_handles_chunk_boundaries(tmp_path):
    """A tiny chunk size must not change where lines break."""
    path = tmp_path / "f.txt"
    path.write_text("aaa\nbb\nc\n" * 200 + "tail")

    for chunk in (1, 2, 3, 7, 64):
        assert list(
            iter_lines_keepends(path, max_line_chars=DEFAULT_MAX_FILE_BYTES, chunk_chars=chunk)
        ) == path.read_text().splitlines(keepends=True), f"chunk_chars={chunk}"


@pytest.mark.asyncio
async def test_ranged_read_memory_stays_bounded(sh, tmp_path):
    """Reading the tail of a large file streams it without holding it."""
    path = tmp_path / "big.txt"
    line_count = 200_000
    _write_lines(path, line_count, width=100)
    size = path.stat().st_size
    assert size > 15 * 1024 * 1024, "fixture must be large enough for the bound to mean something"

    tracemalloc.start()
    try:
        view = await sh.read("big.txt", (line_count - 1, line_count))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert view.text.splitlines() == [f"{i:0100d}" for i in (line_count - 1, line_count)]
    # Whole-file read would peak above the file size; streaming holds a chunk.
    assert peak < 4 * 1024 * 1024, f"peak {peak:,} bytes for a {size:,}-byte file"


# --------------------------------------------------------------------------
# Whole-file reads are budgeted
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whole_file_read_over_budget_is_actionable(tmp_path):
    sh = ShellTools(cwd=str(tmp_path), max_file_bytes=1_000)
    (tmp_path / "big.txt").write_text("x" * 5_000)

    with pytest.raises(FileTooLargeError) as excinfo:
        await sh.read("big.txt")

    message = str(excinfo.value)
    assert "5,000" in message and "1,000" in message
    # Recovery guidance, not just a refusal.
    assert "lines=" in message
    assert "max_file_bytes" in message


@pytest.mark.asyncio
async def test_whole_file_budget_boundary_is_deterministic(tmp_path):
    sh = ShellTools(cwd=str(tmp_path), max_file_bytes=100)
    exact = tmp_path / "exact.txt"
    over = tmp_path / "over.txt"
    exact.write_text("y" * 100)
    over.write_text("y" * 101)

    assert len((await sh.read("exact.txt")).text) == 100
    with pytest.raises(FileTooLargeError):
        await sh.read("over.txt")


@pytest.mark.asyncio
async def test_ranged_read_is_not_capped_by_the_whole_file_budget(tmp_path):
    """The budget bounds unbounded reads; a range is bounded by construction."""
    sh = ShellTools(cwd=str(tmp_path), max_file_bytes=200)
    _write_lines(tmp_path / "big.txt", 2_000)

    view = await sh.read("big.txt", (10, 12))
    assert view.text.splitlines() == [f"{i:064d}" for i in (10, 11, 12)]


@pytest.mark.asyncio
async def test_single_enormous_line_is_refused(tmp_path):
    sh = ShellTools(cwd=str(tmp_path), max_file_bytes=1_000)
    (tmp_path / "oneline.txt").write_text("z" * 50_000)

    with pytest.raises(FileTooLargeError, match="line"):
        await sh.read("oneline.txt", (1, 1))


# --------------------------------------------------------------------------
# Replacements: budgeted, atomic, durable
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_by_path_over_budget_leaves_file_untouched(tmp_path):
    sh = ShellTools(cwd=str(tmp_path), max_file_bytes=100)
    path = tmp_path / "big.py"
    original = "needle\n" + "pad\n" * 200
    path.write_text(original)

    with pytest.raises(FileTooLargeError):
        await sh.replace("big.py", "needle", "thread")

    assert path.read_text() == original


@pytest.mark.asyncio
async def test_replace_by_match_over_budget_leaves_file_untouched(tmp_path):
    sh = ShellTools(cwd=str(tmp_path), max_file_bytes=100)
    path = tmp_path / "big.py"
    original = "a\n" * 200
    path.write_text(original)

    with pytest.raises(FileTooLargeError):
        await sh.replace(Match("big.py", 1, 1, "a\n"), "b\n")

    assert path.read_text() == original


@pytest.mark.asyncio
async def test_replace_budget_boundary_is_deterministic(tmp_path):
    body = "needle\n" + "pad\n" * 10
    exact = tmp_path / "exact.py"
    exact.write_text(body)
    # Take the budget from the file on disk, so newline translation can't skew it.
    on_disk = exact.stat().st_size
    sh_ok = ShellTools(cwd=str(tmp_path), max_file_bytes=on_disk)
    sh_no = ShellTools(cwd=str(tmp_path), max_file_bytes=on_disk - 1)

    await sh_ok.replace("exact.py", "needle", "thread")
    assert exact.read_text().startswith("thread\n")

    over = tmp_path / "over.py"
    over.write_text(body)
    with pytest.raises(FileTooLargeError):
        await sh_no.replace("over.py", "needle", "thread")
    assert over.read_text() == body


@pytest.mark.asyncio
async def test_replace_preserves_executable_bit(sh, tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\necho old\n")
    script.chmod(0o755)
    before = stat.S_IMODE(script.stat().st_mode)

    await sh.replace("run.sh", "echo old", "echo new")

    assert "echo new" in script.read_text()
    assert stat.S_IMODE(script.stat().st_mode) == before, "atomic swap dropped the mode"


@pytest.mark.asyncio
async def test_failed_replace_leaves_original_intact(sh, tmp_path, monkeypatch):
    path = tmp_path / "f.py"
    original = "keep = 1\nneedle = 2\n"
    path.write_text(original)

    def _fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("nooa.tools._bounded_io.os.replace", _fail)

    with pytest.raises(OSError, match="disk full"):
        await sh.replace("f.py", "needle = 2", "needle = 99")

    assert path.read_text() == original
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "f.py"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


@pytest.mark.asyncio
async def test_failed_write_file_leaves_previous_contents(sh, tmp_path, monkeypatch):
    path = tmp_path / "f.txt"
    path.write_text("original\n")

    def _fail(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("nooa.tools._bounded_io.os.replace", _fail)

    with pytest.raises(OSError):
        await sh.write_file("f.txt", "replacement\n")

    assert path.read_text() == "original\n"


def test_atomic_replace_writes_through_a_symlink(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("before\n")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/filesystem")

    atomic_replace_text(link.resolve(), "after\n")

    assert target.read_text() == "after\n"
    assert link.is_symlink(), "symlink was replaced instead of written through"


@pytest.mark.asyncio
async def test_replace_still_rejects_paths_outside_cwd(sh, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret\n")
    try:
        with pytest.raises(ValueError, match="escapes ShellTools cwd"):
            await sh.replace("../outside.txt", "secret", "leaked")
        assert outside.read_text() == "secret\n"
    finally:
        outside.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Search-anchor harvesting
# --------------------------------------------------------------------------


def test_read_specific_lines_only_keeps_requested(tmp_path):
    path = tmp_path / "f.txt"
    _write_lines(path, 500)

    found = read_specific_lines(path, {2, 400}, max_line_chars=DEFAULT_MAX_FILE_BYTES)

    assert set(found) == {2, 400}
    assert found[400].strip() == f"{400:064d}"


def test_read_specific_lines_omits_out_of_range(tmp_path):
    path = tmp_path / "f.txt"
    _write_lines(path, 5)

    found = read_specific_lines(path, {1, 99}, max_line_chars=DEFAULT_MAX_FILE_BYTES)

    assert 1 in found
    assert 99 not in found, "a line past EOF must be absent so callers can fail closed"


@pytest.mark.skipif(
    shutil.which("rg") is None or shutil.which("grep") is None,
    reason="needs rg and grep on PATH",
)
@pytest.mark.asyncio
async def test_grep_anchor_harvest_does_not_read_whole_files(sh, tmp_path, monkeypatch):
    """A grep hit must not pull its whole file in just to quote one line."""
    path = tmp_path / "big.py"
    filler = "".join(f"pad_{i}\n" for i in range(20_000))
    path.write_text(filler + "needle_here = 1\n")

    result = await sh.run("grep -rn 'needle_here' .")
    assert result.matches, "expected structured anchors for a pure search"
    assert "needle_here" in result.matches[0].text

    # Same query with the unbounded path sabotaged.
    def _explode(*args, **kwargs):
        raise AssertionError("anchor harvesting fell back to read_text()")

    monkeypatch.setattr(Path, "read_text", _explode)
    again = await sh.run("grep -rn 'needle_here' .")
    assert again.matches
    assert "needle_here" in again.matches[0].text


@pytest.mark.asyncio
async def test_write_file_then_ranged_read_round_trips(sh, tmp_path):
    await sh.write_file("f.txt", "line1\nline2\nline3\n")
    assert (tmp_path / "f.txt").read_text() == "line1\nline2\nline3\n"

    window = await sh.read("f.txt", (2, 2))
    assert window.text == "line2\n"
    assert (window.start, window.end) == (2, 2)


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(sh, tmp_path):
    await sh.write_file("nested/deeper/f.txt", "hello\n")
    assert (tmp_path / "nested" / "deeper" / "f.txt").read_text() == "hello\n"


def test_default_budget_is_documented_and_sane():
    assert DEFAULT_MAX_FILE_BYTES == 8 * 1024 * 1024
    assert ShellTools(cwd=os.getcwd()).max_file_bytes == DEFAULT_MAX_FILE_BYTES
