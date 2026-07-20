# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ContextVarStream and BlockedStdinWrapper."""

import contextvars
import io
from unittest.mock import MagicMock

import pytest

from nooa.runtime.stream_wrappers import (
    BlockedStdinWrapper,
    ContextVarStream,
    _block_stdin_var,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_stream(name="stdout"):
    """Return a (ContextVarStream, buf_var, original) triple."""
    buf_var: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
        f"test_buf_{name}", default=None
    )
    original = io.StringIO()
    stream = ContextVarStream(original, buf_var, name)
    return stream, buf_var, original


# ===========================================================================
# ContextVarStream
# ===========================================================================


class TestContextVarStreamWrite:
    """Tests for ContextVarStream.write() routing."""

    def test_write_to_buffer_when_set(self):
        stream, buf_var, original = make_stream()
        buf = io.StringIO()
        token = buf_var.set(buf)
        try:
            stream.write("hello")
            assert buf.getvalue() == "hello"
            assert original.getvalue() == ""
        finally:
            buf_var.reset(token)

    def test_write_to_original_when_no_buffer(self):
        stream, buf_var, original = make_stream()
        stream.write("world")
        assert original.getvalue() == "world"

    def test_write_returns_length_from_buffer(self):
        stream, buf_var, original = make_stream()
        buf = io.StringIO()
        token = buf_var.set(buf)
        try:
            n = stream.write("abc")
            assert n == 3
        finally:
            buf_var.reset(token)

    def test_write_returns_length_from_original(self):
        stream, buf_var, original = make_stream()
        n = stream.write("xyz")
        assert n == 3


class TestContextVarStreamWritelines:
    """Tests for ContextVarStream.writelines() routing."""

    def test_writelines_to_buffer_when_set(self):
        stream, buf_var, original = make_stream()
        buf = io.StringIO()
        token = buf_var.set(buf)
        try:
            stream.writelines(["a", "b", "c"])
            assert buf.getvalue() == "abc"
            assert original.getvalue() == ""
        finally:
            buf_var.reset(token)

    def test_writelines_to_original_when_no_buffer(self):
        stream, buf_var, original = make_stream()
        stream.writelines(["x", "y", "z"])
        assert original.getvalue() == "xyz"


class TestContextVarStreamFlush:
    """Tests for ContextVarStream.flush() behavior with and without a buffer."""

    def test_flush_with_buffer_flushes_both(self):
        """flush() should flush the contextvar buffer AND the original stream."""
        buf_var: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
            "test_flush_buf", default=None
        )
        mock_original = MagicMock()
        mock_buf = MagicMock()
        stream = ContextVarStream(mock_original, buf_var, "stdout")
        token = buf_var.set(mock_buf)
        try:
            stream.flush()
            mock_buf.flush.assert_called_once()
            mock_original.flush.assert_called_once()
        finally:
            buf_var.reset(token)

    def test_flush_without_buffer_flushes_original(self):
        """flush() with no buffer set should flush only the original stream."""
        buf_var: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
            "test_flush_no_buf", default=None
        )
        mock_original = MagicMock()
        stream = ContextVarStream(mock_original, buf_var, "stdout")
        stream.flush()
        mock_original.flush.assert_called_once()


class TestContextVarStreamFileno:
    """Tests for ContextVarStream.fileno() delegation to the original stream."""

    def test_fileno_delegates_to_original(self):
        buf_var: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
            "test_fileno_buf", default=None
        )
        # Use a real file for fileno support
        import tempfile

        with tempfile.TemporaryFile(mode="w") as f:
            stream = ContextVarStream(f, buf_var, "test")
            assert stream.fileno() == f.fileno()


class TestContextVarStreamIsatty:
    """Tests for ContextVarStream.isatty() delegation to the original stream."""

    def test_isatty_delegates_to_original(self):
        stream, buf_var, original = make_stream()
        assert stream.isatty() == original.isatty()


class TestContextVarStreamReadableWritableSeekable:
    """Tests for ContextVarStream readable/writable/seekable capability flags."""

    def test_readable_always_false(self):
        stream, _, _ = make_stream()
        assert stream.readable() is False

    def test_writable_always_true(self):
        stream, _, _ = make_stream()
        assert stream.writable() is True

    def test_seekable_always_false(self):
        stream, _, _ = make_stream()
        assert stream.seekable() is False


class TestContextVarStreamClose:
    """Tests for ContextVarStream.close() no-op behavior."""

    def test_close_is_noop_does_not_close_original(self):
        stream, buf_var, original = make_stream()
        stream.close()
        assert not original.closed

    def test_close_can_be_called_multiple_times(self):
        stream, _, _ = make_stream()
        stream.close()
        stream.close()  # should not raise


class TestContextVarStreamClosed:
    """Tests for ContextVarStream.closed property delegation to the original stream."""

    def test_closed_delegates_to_original_open(self):
        stream, _, original = make_stream()
        assert stream.closed is False

    def test_closed_delegates_to_original_closed(self):
        buf_var: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
            "test_closed_buf", default=None
        )
        original = io.StringIO()
        stream = ContextVarStream(original, buf_var, "test")
        original.close()
        assert stream.closed is True


class TestContextVarStreamRepr:
    """Tests for ContextVarStream.__repr__() output format."""

    def test_repr_contains_stream_name(self):
        stream, _, _ = make_stream("mystdout")
        r = repr(stream)
        assert "mystdout" in r

    def test_repr_contains_contextvarstream(self):
        stream, _, _ = make_stream("err")
        assert "ContextVarStream" in repr(stream)


class TestContextVarStreamGetattr:
    """Tests for ContextVarStream.__getattr__() pass-through to the original stream."""

    def test_getattr_passes_through_to_original(self):
        stream, buf_var, original = make_stream()
        # StringIO has a 'name' attribute via getattr fallback on some versions;
        # use a custom attribute we set directly.
        original.custom_attr = "test_value"
        assert stream.custom_attr == "test_value"

    def test_getattr_raises_on_missing_attr(self):
        stream, _, _ = make_stream()
        with pytest.raises(AttributeError):
            _ = stream.nonexistent_attribute_xyz


# ===========================================================================
# BlockedStdinWrapper
# ===========================================================================


def make_stdin_wrapper(content="line1\nline2\n"):
    """Return a (BlockedStdinWrapper, original StringIO) pair."""
    original = io.StringIO(content)
    wrapper = BlockedStdinWrapper(original)
    return wrapper, original


# ---------------------------------------------------------------------------
# Parametrized "raises when blocked" tests
# ---------------------------------------------------------------------------


def _do_read(wrapper):
    wrapper.read()


def _do_readline(wrapper):
    wrapper.readline()


def _do_readlines(wrapper):
    wrapper.readlines()


def _do_iter(wrapper):
    iter(wrapper)


def _do_next(wrapper):
    next(wrapper)


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(_do_read, id="read"),
        pytest.param(_do_readline, id="readline"),
        pytest.param(_do_readlines, id="readlines"),
        pytest.param(_do_iter, id="iter"),
        pytest.param(_do_next, id="next"),
    ],
)
def test_blocked_stdin_raises_when_blocked(action):
    """Each read method raises RuntimeError when stdin is blocked."""
    wrapper, _ = make_stdin_wrapper()
    token = _block_stdin_var.set(True)
    try:
        with pytest.raises(RuntimeError, match="stdin is forbidden"):
            action(wrapper)
    finally:
        _block_stdin_var.reset(token)


# ---------------------------------------------------------------------------
# Per-method pass-through tests (kept in their own classes)
# ---------------------------------------------------------------------------


class TestBlockedStdinRead:
    """Tests for BlockedStdinWrapper.read() pass-through when not blocked."""

    def test_read_passes_through_when_not_blocked(self):
        wrapper, original = make_stdin_wrapper("hello")
        result = wrapper.read()
        assert result == "hello"

    def test_read_size_when_not_blocked(self):
        wrapper, _ = make_stdin_wrapper("abcde")
        assert wrapper.read(3) == "abc"


class TestBlockedStdinReadline:
    """Tests for BlockedStdinWrapper.readline() pass-through when not blocked."""

    def test_readline_passes_through_when_not_blocked(self):
        wrapper, _ = make_stdin_wrapper("line1\nline2\n")
        assert wrapper.readline() == "line1\n"


class TestBlockedStdinReadlines:
    """Tests for BlockedStdinWrapper.readlines() pass-through when not blocked."""

    def test_readlines_passes_through_when_not_blocked(self):
        wrapper, _ = make_stdin_wrapper("a\nb\n")
        assert wrapper.readlines() == ["a\n", "b\n"]


class TestBlockedStdinIter:
    """Tests for BlockedStdinWrapper.__iter__() pass-through when not blocked."""

    def test_iter_passes_through_when_not_blocked(self):
        wrapper, _ = make_stdin_wrapper("x\ny\n")
        lines = list(iter(wrapper))
        assert lines == ["x\n", "y\n"]


class TestBlockedStdinNext:
    """Tests for BlockedStdinWrapper.__next__() pass-through when not blocked."""

    def test_next_passes_through_when_not_blocked(self):
        wrapper, _ = make_stdin_wrapper("line1\nline2\n")
        assert next(wrapper) == "line1\n"


class TestBlockedStdinFileno:
    """Tests for BlockedStdinWrapper.fileno() delegation to the original stream."""

    def test_fileno_delegates_to_original(self):
        import tempfile

        with tempfile.TemporaryFile(mode="r") as f:
            wrapper = BlockedStdinWrapper(f)
            assert wrapper.fileno() == f.fileno()


class TestBlockedStdinIsatty:
    """Tests for BlockedStdinWrapper.isatty() delegation to the original stream."""

    def test_isatty_delegates_to_original(self):
        wrapper, original = make_stdin_wrapper()
        assert wrapper.isatty() == original.isatty()


class TestBlockedStdinReadable:
    """Tests for BlockedStdinWrapper.readable() delegation to the original stream."""

    def test_readable_delegates_to_original(self):
        wrapper, original = make_stdin_wrapper()
        assert wrapper.readable() == original.readable()


class TestBlockedStdinWritable:
    """Tests for BlockedStdinWrapper.writable() always returning False."""

    def test_writable_always_false(self):
        wrapper, _ = make_stdin_wrapper()
        assert wrapper.writable() is False


class TestBlockedStdinSeekable:
    """Tests for BlockedStdinWrapper.seekable() always returning False."""

    def test_seekable_always_false(self):
        wrapper, _ = make_stdin_wrapper()
        assert wrapper.seekable() is False


class TestBlockedStdinClose:
    """Tests for BlockedStdinWrapper.close() no-op behavior."""

    def test_close_is_noop(self):
        wrapper, original = make_stdin_wrapper()
        wrapper.close()
        assert not original.closed

    def test_close_can_be_called_multiple_times(self):
        wrapper, _ = make_stdin_wrapper()
        wrapper.close()
        wrapper.close()  # should not raise


class TestBlockedStdinClosed:
    """Tests for BlockedStdinWrapper.closed property delegation to the original stream."""

    def test_closed_false_when_open(self):
        wrapper, _ = make_stdin_wrapper()
        assert wrapper.closed is False

    def test_closed_true_when_original_closed(self):
        original = io.StringIO("data")
        wrapper = BlockedStdinWrapper(original)
        original.close()
        assert wrapper.closed is True


class TestBlockedStdinRepr:
    """Tests for BlockedStdinWrapper.__repr__() output format."""

    def test_repr_contains_blocked_when_blocked(self):
        wrapper, _ = make_stdin_wrapper()
        token = _block_stdin_var.set(True)
        try:
            r = repr(wrapper)
            assert "BLOCKED" in r
        finally:
            _block_stdin_var.reset(token)

    def test_repr_contains_open_when_not_blocked(self):
        wrapper, _ = make_stdin_wrapper()
        r = repr(wrapper)
        assert "open" in r

    def test_repr_contains_blocked_stdin_wrapper(self):
        wrapper, _ = make_stdin_wrapper()
        assert "BlockedStdinWrapper" in repr(wrapper)


class TestBlockedStdinGetattr:
    """Tests for BlockedStdinWrapper.__getattr__() pass-through to the original stream."""

    def test_getattr_passes_through_to_original(self):
        original = io.StringIO("data")
        wrapper = BlockedStdinWrapper(original)
        original.custom_attr = "hello"
        assert wrapper.custom_attr == "hello"

    def test_getattr_raises_on_missing_attr(self):
        wrapper, _ = make_stdin_wrapper()
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attribute_abc
