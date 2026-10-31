# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FileBackedTruncatingStringIO."""

import json
import os
import tempfile
import threading
import time

import pytest

from nooa.agentdoc._truncating_stream import CaptureStats, FileBackedTruncatingStringIO


@pytest.fixture
def buf():
    """Yield a FileBackedTruncatingStringIO and clean up on teardown."""
    b = FileBackedTruncatingStringIO(limit=100)
    yield b
    b.cleanup()


@pytest.fixture
def small_buf():
    """Yield a small-limit buffer for truncation tests."""
    b = FileBackedTruncatingStringIO(limit=10)
    yield b
    b.cleanup()


class TestFileBackedBasicBehavior:
    """Verify it behaves identically to TruncatingStringIO for in-memory operations."""

    def test_small_write_not_truncated(self, buf):
        """Short output returns verbatim without truncation."""
        buf.write("hello")
        assert buf.getvalue() == "hello"
        assert not buf.was_truncated

    def test_write_exactly_at_limit(self, small_buf):
        """Output at exact limit is not truncated."""
        small_buf.write("x" * 10)
        assert not small_buf.was_truncated
        assert small_buf.getvalue() == "x" * 10

    def test_write_over_limit_truncates(self, small_buf):
        """Output exceeding limit triggers truncation with prose notice."""
        small_buf.write("x" * 100)
        assert small_buf.was_truncated
        assert small_buf.chars_written == 100
        value = small_buf.getvalue()
        assert "Output too large" in value
        assert "100 chars" in value

    def test_multiple_writes_accumulate(self, buf):
        """Successive writes concatenate in both memory and file."""
        buf.write("abc")
        buf.write("def")
        assert buf.getvalue() == "abcdef"
        assert buf.chars_written == 6

    def test_empty_write(self, buf):
        """Empty string write is a no-op."""
        buf.write("")
        assert buf.getvalue() == ""
        assert not buf.was_truncated

    def test_custom_tail_chars(self):
        """Custom tail_chars parameter controls the tail window size."""
        buf = FileBackedTruncatingStringIO(limit=20, tail_chars=15)
        try:
            buf.write("A" * 5)  # head_limit = 5
            buf.write("B" * 100)  # tail keeps last 15
            assert buf.was_truncated
            value = buf.getvalue()
            assert value.count("B") == 15
        finally:
            buf.cleanup()

    def test_unicode_round_trip(self, buf):
        """Unicode characters survive the write-read round trip in both memory and file."""
        text = "Hello 世界! 🚀 café"
        buf.write(text)
        assert buf.getvalue() == text
        with open(buf.file_path) as f:
            assert f.read() == text


class TestFileBackedFileOutput:
    """Verify file-backed behavior: full output written to temp file."""

    def test_file_created(self, buf):
        """Temp file exists on disk after construction."""
        buf.write("hello")
        assert buf.file_path is not None
        assert os.path.exists(buf.file_path)

    def test_file_contains_full_output_when_truncated(self, small_buf):
        """Temp file contains the entire untruncated output even when in-memory is truncated."""
        content = "A" * 5 + "B" * 90 + "C" * 5
        small_buf.write(content)
        assert small_buf.was_truncated
        with open(small_buf.file_path) as f:
            assert f.read() == content

    def test_file_contains_full_output_when_not_truncated(self, buf):
        """Temp file contains the output even when truncation was not triggered."""
        buf.write("hello world")
        with open(buf.file_path) as f:
            assert f.read() == "hello world"

    def test_file_contains_multiple_writes(self, small_buf):
        """Multiple writes are concatenated in the temp file."""
        small_buf.write("hello ")
        small_buf.write("world ")
        small_buf.write("this is a long string")
        with open(small_buf.file_path) as f:
            assert f.read() == "hello world this is a long string"

    def test_custom_dir(self):
        """Temp file is placed in the specified directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = FileBackedTruncatingStringIO(limit=100, dir=tmpdir)
            try:
                buf.write("test")
                assert buf.file_path.startswith(tmpdir)
            finally:
                buf.cleanup()

    def test_custom_prefix_suffix(self):
        """Temp file name honours the prefix and suffix arguments."""
        buf = FileBackedTruncatingStringIO(limit=100, prefix="myprefix_", suffix=".log")
        try:
            buf.write("test")
            basename = os.path.basename(buf.file_path)
            assert basename.startswith("myprefix_")
            assert basename.endswith(".log")
        finally:
            buf.cleanup()

    def test_large_output(self):
        """File correctness verified with 50k chars of output."""
        buf = FileBackedTruncatingStringIO(limit=1000)
        try:
            content = "x" * 50_000
            buf.write(content)
            assert buf.was_truncated
            with open(buf.file_path) as f:
                assert f.read() == content
        finally:
            buf.cleanup()


class TestFileBackedTruncationNotice:
    """Verify truncation notice includes file path."""

    def test_truncation_notice_includes_file_path(self, small_buf):
        """Truncation notice contains the temp file path."""
        small_buf.write("x" * 100)
        value = small_buf.getvalue()
        assert small_buf.file_path in value
        assert "full untruncated output" in value

    def test_truncation_notice_has_standard_format(self, small_buf):
        """Truncation notice uses the standard <truncated-output> wrapper format."""
        small_buf.write("x" * 100)
        value = small_buf.getvalue()
        assert value.startswith("<truncated-output>")
        assert "Output too large" in value
        assert "100 chars" in value
        assert "not shown" in value

    def test_not_truncated_no_file_path_in_output(self, buf):
        """When output fits in the budget, getvalue() returns plain content (no notice)."""
        buf.write("hello")
        value = buf.getvalue()
        assert value == "hello"
        assert "saved to" not in value

    def test_head_and_tail_preserved(self):
        """Both head and tail content appear in the truncated notice alongside the file path."""
        buf = FileBackedTruncatingStringIO(limit=50)
        try:
            buf.write("HEADDATA" + "x" * 200 + "TAILCONTENT")
            value = buf.getvalue()
            assert "HEADDATA" in value
            assert "TAILCONTENT" in value
            assert buf.file_path in value
        finally:
            buf.cleanup()

    def test_getvalue_idempotent(self, small_buf):
        """getvalue() returns the same result on repeated calls."""
        small_buf.write("x" * 100)
        v1 = small_buf.getvalue()
        v2 = small_buf.getvalue()
        assert v1 == v2


class TestFileBackedErrorHandling:
    """Verify graceful fallback when file I/O fails."""

    def test_fallback_on_invalid_dir(self):
        """Invalid dir falls back to in-memory-only; no file path in notice."""
        buf = FileBackedTruncatingStringIO(limit=10, dir="/nonexistent/path/xyz")
        try:
            buf.write("x" * 100)
            assert buf.was_truncated
            value = buf.getvalue()
            assert "Output too large" in value
            assert "full untruncated output" not in value
        finally:
            buf.cleanup()

    def test_fallback_still_counts_chars(self):
        """Char counting works even when file creation failed."""
        buf = FileBackedTruncatingStringIO(limit=10, dir="/nonexistent/path/xyz")
        try:
            buf.write("x" * 50)
            assert buf.chars_written == 50
        finally:
            buf.cleanup()

    def test_file_path_none_on_failure(self):
        """file_path is None when temp file creation failed."""
        buf = FileBackedTruncatingStringIO(limit=10, dir="/nonexistent/path/xyz")
        try:
            assert buf.file_path is None
        finally:
            buf.cleanup()


class TestFileBackedCleanup:
    """Verify close() and cleanup behavior."""

    def test_close_closes_file_but_keeps_on_disk(self, buf):
        """close() releases the file handle but leaves the file on disk."""
        buf.write("test")
        path = buf.file_path
        buf.close()
        assert os.path.exists(path)
        os.unlink(path)

    def test_cleanup_removes_file(self, buf):
        """cleanup() closes the handle and deletes the temp file."""
        buf.write("test")
        path = buf.file_path
        buf.cleanup()
        assert not os.path.exists(path)

    def test_cleanup_idempotent(self, buf):
        """Calling cleanup() twice does not raise."""
        buf.write("test")
        buf.cleanup()
        buf.cleanup()

    def test_close_idempotent(self, buf):
        """Calling close() twice does not raise."""
        buf.write("test")
        buf.close()
        buf.close()

    def test_getvalue_after_close(self, small_buf):
        """getvalue() works after close() — reads from in-memory buffers."""
        small_buf.write("x" * 100)
        small_buf.close()
        value = small_buf.getvalue()
        assert "Output too large" in value
        assert small_buf.file_path in value

    def test_cleanup_then_getvalue(self, small_buf):
        """After cleanup the file is gone; in-memory truncation notice still renders."""
        small_buf.write("x" * 100)
        small_buf.cleanup()
        value = small_buf.getvalue()
        assert "Output too large" in value


class TestFileBackedExport:
    """Verify the class is properly exported."""

    def test_importable_from_agentdoc(self):
        """FileBackedTruncatingStringIO is importable from the agentdoc public API."""
        from nooa.agentdoc import FileBackedTruncatingStringIO as FB

        assert FB is FileBackedTruncatingStringIO


class TestExplicitPathAndCap:
    def test_writes_to_the_given_path(self, tmp_path):
        target = tmp_path / "out.log"
        buf = FileBackedTruncatingStringIO(limit=4, tail_chars=4, path=target)
        buf.write("abcdefgh")
        buf.close()
        assert buf.file_path == str(target)
        assert target.read_text() == "abcdefgh"
        assert buf.getvalue() == f"[... 4 chars omitted; full output: {target}]\nefgh"

    def test_max_file_bytes_caps_the_file_exactly(self, tmp_path):
        target = tmp_path / "out.log"
        buf = FileBackedTruncatingStringIO(limit=0, tail_chars=0, path=target, max_file_bytes=5)
        buf.write_bytes(b"abc")
        buf.write_bytes(b"defgh")
        assert buf.file_bytes_dropped == 3
        buf.close()
        buf.close()  # idempotent
        assert target.read_bytes() == b"abcde"  # byte-exact: no marker appended

    def test_write_bytes_goes_to_the_file_and_the_view(self, tmp_path):
        target = tmp_path / "out.log"
        buf = FileBackedTruncatingStringIO(limit=10, path=target)
        buf.write_bytes(b"\xffraw")
        buf.write("text")
        buf.close()
        assert target.read_bytes() == b"\xffrawtext"
        assert buf.getvalue() == "\ufffdrawtext"

    def test_unwritable_path_falls_back_to_memory(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=4, tail_chars=4, path=tmp_path / "no" / "x")
        buf.write("abcdefgh")
        assert buf.getvalue() == "[... 4 chars omitted]\nefgh"


class TestFileBackedBytes:
    """write_bytes: byte-exact file, incrementally decoded view."""

    def test_bytes_go_to_file_unchanged_and_to_the_view(self, tmp_path):
        data = b"ok \xff\xfe raw\n" + "é🚀".encode()
        with FileBackedTruncatingStringIO(limit=100, path=tmp_path / "o.log") as buf:
            assert buf.write_bytes(data) == len(data)
        assert (tmp_path / "o.log").read_bytes() == data
        assert buf.getvalue() == data.decode("utf-8", errors="replace")
        assert buf.bytes_written == len(data)

    def test_multibyte_split_across_chunks_decodes(self, tmp_path):
        data = "aé世🚀z".encode()
        buf = FileBackedTruncatingStringIO(limit=100, path=tmp_path / "o.log")
        for i in range(len(data)):  # one byte at a time: every character is split
            buf.write_bytes(data[i : i + 1])
        assert buf.getvalue() == "aé世🚀z"
        buf.close()
        assert (tmp_path / "o.log").read_bytes() == data

    def test_close_flushes_an_incomplete_trailing_character(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=100, path=tmp_path / "o.log")
        buf.write_bytes(b"x" + "é".encode()[:1])
        assert buf.getvalue() == "x"
        buf.close()
        assert buf.getvalue() == "x\ufffd"
        buf.close()  # idempotent: no second replacement character
        assert buf.getvalue() == "x\ufffd"

    def test_bytes_over_the_limit_cut_the_view_and_name_the_file(self, tmp_path):
        data = b"".join(b"line %06d\n" % i for i in range(2_000))
        buf = FileBackedTruncatingStringIO(limit=1_000, path=tmp_path / "o.log")
        for i in range(0, len(data), 333):
            buf.write_bytes(data[i : i + 333])
        buf.close()
        value = buf.getvalue()
        assert value.startswith("<truncated-output>")
        assert str(tmp_path / "o.log") in value
        assert "line 000000" in value and "line 001999" in value
        assert (tmp_path / "o.log").read_bytes() == data


class TestFileBackedCap:
    """max_file_bytes: the file holds exactly the first N bytes."""

    def test_cap_keeps_first_bytes_and_counts_the_rest(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=10, path=tmp_path / "o", max_file_bytes=25)
        buf.write_bytes(b"a" * 20)
        buf.write("b" * 20)
        buf.close()
        assert (tmp_path / "o").read_bytes() == b"a" * 20 + b"b" * 5
        assert buf.file_bytes == 25
        assert buf.file_bytes_dropped == 15
        value = buf.getvalue()
        assert "The file is incomplete: it holds the first 25 bytes; 15 more bytes" in value

    def test_negative_cap_is_refused(self):
        with pytest.raises(ValueError):
            FileBackedTruncatingStringIO(max_file_bytes=-1)


class TestFileBackedNotice:
    """notice=: the caller phrases how to read the file."""

    def test_template_gets_the_stats_fields_and_quoted_path(self, tmp_path):
        path = tmp_path / "o.log"
        buf = FileBackedTruncatingStringIO(
            limit=10,
            path=path,
            notice="Read it: open({quoted_path}) ({dropped_chars:,} chars cut of {chars_written:,})",
        )
        buf.write("x" * 1_100)
        value = buf.getvalue()
        assert f"Read it: open({json.dumps(str(path))}) (1,090 chars cut of 1,100)\n" in value
        assert "full untruncated output" not in value
        buf.cleanup()

    def test_callable_gets_the_stats(self, tmp_path):
        seen = []
        buf = FileBackedTruncatingStringIO(
            limit=10, path=tmp_path / "o", notice=lambda s: seen.append(s) or f"at {s.path}"
        )
        buf.write("x" * 100)
        assert f"at {tmp_path / 'o'}\n" in buf.getvalue()
        assert seen[0].truncated and seen[0].dropped_chars == 90
        buf.cleanup()

    def test_default_notice_stays_generic(self, small_buf):
        small_buf.write("x" * 100)
        assert f"is in: {small_buf.path}\nRead that file" in small_buf.getvalue()

    def test_no_notice_without_truncation(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=100, path=tmp_path / "o", notice="X {path}")
        buf.write("short")
        assert buf.getvalue() == "short"
        buf.cleanup()


class TestFileBackedStatsAndLifecycle:
    def test_stats(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=10, path=tmp_path / "o")
        buf.write("é" * 30)
        stats = buf.stats
        assert stats == CaptureStats(
            path=str(tmp_path / "o"),
            chars_written=30,
            bytes_written=60,
            file_bytes=60,
            file_bytes_dropped=0,
            truncated=True,
            dropped_chars=20,
        )
        buf.cleanup()

    def test_path_is_none_after_cleanup_but_file_path_stays(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=10, path=tmp_path / "o")
        buf.write("x" * 100)
        buf.cleanup()
        assert buf.path is None
        assert buf.file_path == str(tmp_path / "o")
        assert not (tmp_path / "o").exists()
        assert "not recoverable" in buf.getvalue()  # the notice no longer names a gone file

    def test_with_closes_and_keeps_the_file(self, tmp_path):
        with FileBackedTruncatingStringIO(limit=10, path=tmp_path / "o") as buf:
            buf.write("hello")
        assert (tmp_path / "o").read_text() == "hello"
        buf.write("more")  # after close: view only, the file is not reopened
        assert (tmp_path / "o").read_text() == "hello"
        assert buf.getvalue() == "hellomore"

    def test_writes_after_cleanup_do_not_recreate_the_file(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=10, path=tmp_path / "o")
        buf.cleanup()
        buf.write_bytes(b"late")
        assert not (tmp_path / "o").exists()


class TestFileBackedConcurrencyAndCost:
    def test_threads_write_while_another_reads(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=1_000, path=tmp_path / "o")
        chunk = b"0123456789" * 10

        def writer():
            for _ in range(500):
                buf.write_bytes(chunk)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for th in threads:
            th.start()
        while any(th.is_alive() for th in threads):
            buf.getvalue()
        for th in threads:
            th.join()
        buf.close()
        assert buf.bytes_written == buf.file_bytes == 4 * 500 * len(chunk)
        assert (tmp_path / "o").read_bytes() == chunk * 2_000

    def test_many_small_writes_at_a_1_mib_limit_stay_linear(self, tmp_path):
        buf = FileBackedTruncatingStringIO(limit=1 << 20, path=tmp_path / "o")
        start = time.perf_counter()
        for _ in range(200_000):  # 2 MB in 10-byte writes, half of it into the head
            buf.write_bytes(b"abcdefghi\n")
        value = buf.getvalue()
        elapsed = time.perf_counter() - start
        buf.cleanup()
        assert value.startswith("<truncated-output>")
        assert elapsed < 10  # quadratic head copies took minutes here
