# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TextStream, SpillTarget and merge: bounded memory, UTF-8 cuts, spill files, deltas."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from nooa.snapshot import Spill, SpillTarget, Text, TextStream, merge, render_output


def _stream(unit: str = "bytes", head: int = 4, tail: int = 4, **kwargs) -> TextStream:
    return TextStream(unit=unit, head=head, tail=tail, **kwargs)  # type: ignore[arg-type]


# --- chars -----------------------------------------------------------------------


def test_chars_complete_until_head_plus_tail() -> None:
    stream = _stream("chars")
    stream.write("abcd")
    stream.write("efgh")

    assert stream.snapshot() == Text("abcdefgh", "", 8, "chars", True, None)


def test_chars_keep_head_and_rolling_tail() -> None:
    stream = _stream("chars")
    for piece in ("abc", "defg", "hij", "klmnop"):
        stream.write(piece)

    text = stream.close()

    assert text == Text("abcd", "mnop", 16, "chars", False, "max_length", omitted=8)
    assert render_output(text) == "abcd\n[... 8 chars omitted; full output not kept]\nmnop"


def test_memory_stays_bounded_with_huge_writes() -> None:
    stream = _stream("chars", head=10, tail=10)
    for _ in range(100):
        stream.write("x" * 100_000)

    assert stream._head_len == 10 and stream._tail_len == 10
    assert sum(len(chunk) for chunk in stream._tail_chars) == 10
    assert stream.snapshot().length == 10_000_000


def test_tail_only_stream() -> None:
    stream = _stream("chars", head=0, tail=3)
    stream.write("abcdef")

    text = stream.snapshot()
    assert (text.head, text.tail, text.omitted) == ("", "def", 3)
    assert render_output(text) == "[... 3 chars omitted; full output not kept]\ndef"


# --- bytes and UTF-8 cut points ------------------------------------------------------


@pytest.mark.parametrize(("char", "size"), [("é", 2), ("€", 3), ("😀", 4)])
def test_bytes_head_and_tail_cut_on_code_points(char: str, size: int) -> None:
    head_limit = size + 1  # one byte into the second code point
    stream = TextStream(unit="bytes", head=head_limit, tail=head_limit)
    stream.write((char * 10).encode())

    text = stream.close()

    assert text.head == char.encode()  # the split code point moves to the omitted part
    assert text.tail == char.encode()
    assert text.length == 10 * size
    assert text.omitted == 10 * size - 2 * size


def test_four_byte_code_point_written_one_byte_at_a_time() -> None:
    stream = TextStream(unit="bytes", head=2, tail=2)
    for byte in "😀😀".encode():
        stream.write(bytes([byte]))

    text = stream.close()
    # Neither 2-byte window can hold a whole 4-byte code point.
    assert (text.head, text.tail, text.omitted) == (b"", b"", 8)


def test_invalid_bytes_are_kept_raw() -> None:
    stream = TextStream(unit="bytes", head=100, tail=100)
    stream.write(b"ok\xff\xfeend")
    stream.write("\ufffd".encode())

    text = stream.close()
    assert text.head == b"ok\xff\xfeend\xef\xbf\xbd"  # the raw bytes
    assert render_output(text) == "ok\ufffd\ufffdend\ufffd"  # decoded only to print


def test_bytes_stream_render_output_uses_bytes() -> None:
    stream = TextStream(unit="bytes", head=2, tail=2)
    stream.write(b"x" * 1_234_567)

    assert render_output(stream.close()) == (
        "xx\n[... 1,234,563 bytes omitted; full output not kept]\nxx"
    )


def test_write_after_close_is_refused() -> None:
    stream = _stream()
    stream.close()
    with pytest.raises(ValueError):
        stream.write(b"x")


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        TextStream(unit="lines", head=1, tail=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TextStream(unit="chars", head=-1, tail=1)


# --- spilling ----------------------------------------------------------------------


def test_spill_file_holds_everything(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    stream = TextStream(unit="bytes", head=3, tail=3, spill=SpillTarget(path=str(path)))
    stream.write(b"0123456789")
    running = stream.snapshot()
    stream.write(b"abcdef")

    text = stream.close()

    assert path.read_bytes() == b"0123456789abcdef"
    assert running.full == Spill(str(path), 10, 10)
    assert text.full == Spill(str(path), 16, 16)
    assert text.reason == "max_length"
    assert render_output(text) == f"012\n[... 10 bytes omitted; full output: {path}]\ndef"


def test_spill_max_bytes_cuts_the_file(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    stream = TextStream(
        unit="bytes", head=2, tail=2, spill=SpillTarget(path=str(path), max_bytes=5)
    )
    stream.write(b"0123456789" * 1000)

    text = stream.close()

    assert path.read_bytes() == b"01234"
    assert text.full == Spill(str(path), 5, 10_000)
    assert text.full.cut  # the file is cut: ``full`` says so, not the reason
    assert text.reason == "max_length"
    assert render_output(text) == (
        f"01\n[... 9,996 bytes omitted; {path} holds the first 5 of 10,000 bytes]\n89"
    )


def test_chars_stream_spills_utf8(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    stream = TextStream(unit="chars", head=1, tail=1, spill=SpillTarget(path=str(path)))
    stream.write("é€😀")

    text = stream.close()

    assert path.read_text(encoding="utf-8") == "é€😀"
    assert text.full == Spill(str(path), 9, 9)
    assert text.length == 3 and text.omitted == 1


def test_uncut_stream_deletes_its_spill_file(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    stream = TextStream(unit="bytes", head=10, tail=10, spill=SpillTarget(path=str(path)))
    stream.write(b"small")

    text = stream.close()

    assert not path.exists()
    assert text.full is None and text.complete


def test_always_keeps_the_spill_file(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    stream = TextStream(
        unit="bytes", head=10, tail=10, spill=SpillTarget(path=str(path), always=True)
    )
    stream.write(b"small")

    text = stream.close()

    assert path.read_bytes() == b"small"
    assert text.full == Spill(str(path), 5, 5)
    assert render_output(text) == "small"


def test_temp_spill_file_by_default() -> None:
    stream = TextStream(unit="bytes", head=1, tail=1, spill=SpillTarget())
    stream.write(b"abcdef")

    text = stream.close()

    assert text.full is not None
    try:
        assert Path(text.full.path).read_bytes() == b"abcdef"
    finally:
        os.unlink(text.full.path)


def test_spill_open_failure_falls_back_to_memory(tmp_path: Path) -> None:
    missing = tmp_path / "no" / "such" / "dir" / "out.log"
    stream = TextStream(unit="bytes", head=2, tail=2, spill=SpillTarget(path=str(missing)))
    stream.write(b"abcdefgh")

    text = stream.close()

    assert text.full is None  # no file: ``full`` says so, not the reason
    assert text.reason == "max_length"
    assert render_output(text) == "ab\n[... 4 bytes omitted; full output not kept]\ngh"


def test_spill_write_failure_falls_back_to_memory(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    stream = TextStream(unit="bytes", head=2, tail=2, spill=SpillTarget(path=str(path)))

    class Broken:
        def write(self, data: bytes) -> int:
            raise OSError("disk full")

        def close(self) -> None:
            pass

    stream._file = Broken()  # type: ignore[assignment]
    stream.write(b"abcdefgh")

    text = stream.close()
    assert text.full is None and text.reason == "max_length"
    assert not path.exists()  # the partial file is removed
    assert (text.head, text.tail, text.length) == (b"ab", b"gh", 8)


def test_chars_lone_surrogate_without_spill_does_not_raise() -> None:
    stream = _stream("chars", head=10, tail=10)
    stream.write("a\ud800b")

    assert stream.close() == Text("a\ud800b", "", 3, "chars", True, None)


def test_chars_lone_surrogate_spills_backslash_escaped(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    spill = SpillTarget(path=str(path), always=True)
    stream = TextStream(unit="chars", head=10, tail=10, spill=spill)
    stream.write("a\ud800b")

    text = stream.close()
    assert text.head == "a\ud800b" and text.full is not None
    assert path.read_bytes() == b"a\\ud800b" and text.full.kept == len(b"a\\ud800b")


def test_bytes_stream_accepts_str_with_lone_surrogate() -> None:
    stream = _stream("bytes", head=20, tail=20)
    stream.write("a\ud800b")

    assert stream.close().head == b"a\\ud800b"


def test_running_snapshot_and_since_match_the_file_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    spill = SpillTarget(path=str(path), always=True)
    stream = TextStream(unit="bytes", head=10, tail=10, spill=spill)
    stream.write(b"y" * 100)

    text = stream.snapshot()
    assert text.full is not None and text.full.kept == 100 == os.path.getsize(path)
    stream.write(b"z" * 7)
    delta, _ = stream.since(100)
    assert delta.length == 7 and os.path.getsize(path) == 107
    stream.close()


def test_spill_flush_failure_falls_back_to_memory(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    stream = TextStream(unit="bytes", head=2, tail=2, spill=SpillTarget(path=str(path)))
    stream.write(b"abcdefgh")
    real = stream._file
    assert real is not None

    class Broken:
        def write(self, data: bytes) -> int:
            return real.write(data)

        def flush(self) -> None:
            raise OSError("disk full")

        def close(self) -> None:
            real.close()

    stream._file = Broken()  # type: ignore[assignment]

    text = stream.snapshot()
    assert text.full is None and text.reason == "max_length"
    assert not path.exists()  # handled like a write failure
    assert (text.head, text.tail, text.length) == (b"ab", b"gh", 8)


def test_text_reads_do_not_rebind_the_writer_buffers() -> None:
    stream = _stream("chars", head=4, tail=4)
    for piece in ("ab", "cd", "ef", "gh", "ij"):
        stream.write(piece)
    head, tail = stream._head_chars, stream._tail_chars
    pieces = (list(head), list(tail))

    stream.snapshot()
    stream.since(3)

    assert stream._head_chars is head and stream._tail_chars is tail
    assert (list(head), list(tail)) == pieces


# --- since / merge -----------------------------------------------------------------


@pytest.mark.parametrize("data", [b"abc", bytearray(b"abc"), memoryview(b"abc")])
def test_chars_stream_rejects_bytes(data: object) -> None:
    stream = TextStream(unit="chars", head=4, tail=4)
    stream.write("ok")

    with pytest.raises(TypeError, match="bytes stream"):
        stream.write(data)  # type: ignore[arg-type]

    # Nothing of the rejected chunk was counted or kept.
    assert stream.close() == Text("ok", "", 2, "chars", True, None)


def test_bytes_stream_still_accepts_str() -> None:
    stream = TextStream(unit="bytes", head=8, tail=8)
    stream.write("é")
    stream.write(b"!")

    assert stream.close() == Text("é!".encode(), b"", 3, "bytes", True, None)


def test_since_returns_deltas_with_correct_lengths() -> None:
    stream = TextStream(unit="chars", head=4, tail=4)
    stream.write("abc")
    first, mark = stream.since()
    stream.write("defgh")
    second, mark = stream.since(mark)
    stream.write("x" * 100 + "yz")
    third, mark = stream.since(mark)

    assert first == Text("abc", "", 3, "chars", True, None)
    assert second == Text("defgh", "", 5, "chars", True, None)
    assert third.length == 102 and third.tail == "xxyz" and third.head == ""
    assert third.omitted == 98
    assert mark == 110
    nothing, same = stream.since(mark)
    assert nothing == Text("", "", 0, "chars", True, None) and same == mark


def test_since_stops_before_an_incomplete_code_point() -> None:
    stream = TextStream(unit="bytes", head=16, tail=16)
    data = "a😀".encode()
    stream.write(data[:3])  # "a" and half of the emoji
    first, mark = stream.since()
    stream.write(data[3:])
    second, mark = stream.since(mark)

    assert (first.head, first.length, mark) == (b"a", 1, 5)
    assert (second.head, second.length) == ("😀".encode(), 4)


def test_merge_keeps_length_and_bounds() -> None:
    a = Text("abcdef", "", 6, "chars", True, None)
    b = Text("gh", "yz", 40, "chars", False, "max_length", omitted=36)

    merged = merge(a, b, head=3, tail=3)

    assert merged.length == 46
    assert (merged.head, merged.tail) == ("abc", "yz")  # b's own gap stops the tail
    assert merged.omitted == 46 - 3 - 2
    assert not merged.complete


def test_merge_of_small_complete_deltas_is_complete() -> None:
    a = Text("ab", "", 2, "chars", True, None)
    b = Text("cd", "", 2, "chars", True, None)

    assert merge(a, b, head=2, tail=2) == Text("abcd", "", 4, "chars", True, None)


def test_merge_cuts_complete_deltas_that_are_too_long() -> None:
    a = Text("abcde", "", 5, "chars", True, None)
    b = Text("fghij", "", 5, "chars", True, None)

    merged = merge(a, b, head=2, tail=3)

    assert (merged.head, merged.tail, merged.length, merged.omitted) == ("ab", "hij", 10, 5)
    assert render_output(merged) == "ab\n[... 5 chars omitted; full output not kept]\nhij"


def test_merge_of_since_deltas_matches_the_stream_length() -> None:
    stream = TextStream(unit="bytes", head=8, tail=8)
    merged: Text | None = None
    mark = 0
    for chunk in (b"abc", "é€".encode() * 20, b"xyz" * 50, "😀".encode()):
        stream.write(chunk)
        delta, mark = stream.since(mark)
        merged = delta if merged is None else merge(merged, delta, head=8, tail=8)

    assert merged is not None
    assert merged.length == stream.snapshot().length
    assert len(merged.head) <= 8 and len(merged.tail) <= 8
    assert merged.tail.endswith("😀".encode())
    assert merged.omitted == merged.length - len(merged.head) - len(merged.tail)


def test_merge_bytes_cuts_on_code_points() -> None:
    a = Text("😀😀".encode(), b"", 8, "bytes", True, None)
    b = Text("😀😀".encode(), b"", 8, "bytes", True, None)

    merged = merge(a, b, head=5, tail=7)

    assert (merged.head, merged.tail) == ("😀".encode(), "😀".encode())
    assert merged.omitted == 8


def test_merge_rejects_mixed_units() -> None:
    with pytest.raises(ValueError):
        merge(Text("", "", 0, "chars", True, None), Text(b"", b"", 0, "bytes", True, None), 1, 1)


def test_context_manager_closes() -> None:
    with TextStream(unit="chars", head=1, tail=1) as stream:
        stream.write("abc")
    with pytest.raises(ValueError):
        stream.write("d")


# --- review findings: short tails, split U+FFFD, merge units ------------------------


@pytest.mark.parametrize("tail", [0, 1, 2])
def test_since_mark_stops_before_an_incomplete_sequence_with_a_short_tail(tail: int) -> None:
    stream = _stream("bytes", head=1, tail=tail)
    stream.write(b"a\xf0\x9f\x98")

    _delta, mark = stream.since()
    assert mark == 1
    stream.write(b"\x80b")
    delta, mark = stream.since(mark)
    assert mark == 6 and delta.length == 5


def test_merge_keeps_raw_bytes_exactly() -> None:
    def closed(data: bytes) -> Text:
        stream = _stream("bytes", head=10, tail=10)
        stream.write(data)
        return stream.close()

    invalid, plain = closed(b"\xff"), closed(b"x")

    merged = merge(invalid, plain, head=1, tail=1)
    assert merged == Text(b"\xffx", b"", 2, "bytes", True, None)
    cut = merge(closed(b"\xffab"), closed(b"cd\xfe"), head=1, tail=1)
    assert (cut.head, cut.tail, cut.omitted) == (b"\xff", b"\xfe", 4)
    dropped = merge(closed(b"\xffabc"), closed(b"de"), head=0, tail=2)
    assert (dropped.tail, dropped.omitted) == (b"de", 4)


def test_merge_cuts_a_genuine_replacement_character_whole() -> None:
    # Head: a genuine U+FFFD (3 bytes); tail: an invalid byte (1 byte).
    stream = TextStream(unit="bytes", head=3, tail=1)
    stream.write(b"\xef\xbf\xbdX\xff")
    a = stream.snapshot()
    assert (a.head, a.tail, a.omitted) == (b"\xef\xbf\xbd", b"\xff", 1)
    empty = TextStream(unit="bytes", head=3, tail=1).snapshot()

    merged = merge(a, empty, head=1, tail=0)

    # The genuine U+FFFD is 3 bytes: it does not fit a 1-byte head.
    assert (merged.head, merged.tail, merged.omitted) == (b"", b"", 5)
    kept_genuine = merge(a, empty, head=3, tail=0)
    assert (kept_genuine.head, kept_genuine.omitted) == (b"\xef\xbf\xbd", 2)
    kept_invalid = merge(a, empty, head=0, tail=1)
    assert (kept_invalid.tail, kept_invalid.omitted) == (b"\xff", 4)


def test_capture_of_bytes_keeps_the_raw_bytes() -> None:
    from nooa.snapshot import Limits, capture

    node = capture(b"\xff" * 50 + b"\xef\xbf\xbd" * 50, Limits(max_string=20, tail_share=0.5))
    assert isinstance(node, Text) and not node.complete
    assert isinstance(node.head, bytes) and node.head == b"\xff" * len(node.head)
    assert isinstance(node.tail, bytes) and node.tail.endswith(b"\xef\xbf\xbd")
    assert node.length == len(node.head) + len(node.tail) + node.omitted


_FRAGMENTS = {
    "valid": [b"a", b"z", "é".encode(), "€".encode(), "😀".encode()],
    "genuine": [b"\xef\xbf\xbd"],
    "single": [b"\xff", b"\x80", b"\xfe"],
    "multi": [b"\xe2\x82", b"\xf0\x9f\x98", b"\xed\xa0", b"\xe2"],
}


def _random_bytes(rng: random.Random, kinds: list[str]) -> bytes:
    pool = [fragment for kind in kinds for fragment in _FRAGMENTS[kind]]
    return b"".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))


def _check(text: Text, data: bytes, head: int, tail: int) -> None:
    """``text`` describes ``data`` exactly: raw bytes within ``head``/``tail``."""
    assert isinstance(text.head, bytes) and isinstance(text.tail, bytes)
    assert text.length == len(data) == len(text.head) + len(text.tail) + text.omitted
    if text.complete:
        assert (text.head, text.tail, text.omitted) == (data, b"", 0)
        return
    assert len(text.head) <= head and len(text.tail) <= tail
    assert data.startswith(text.head) and data.endswith(text.tail)


@pytest.mark.parametrize(
    "kinds",
    [["valid", "genuine"], ["valid", "single"], ["valid", "multi"], list(_FRAGMENTS)],
)
def test_bytes_streams_and_merges_are_exact_for_random_splits(kinds: list[str]) -> None:
    rng = random.Random(f"units-{'-'.join(kinds)}")
    for _ in range(400):
        head, tail = rng.randint(0, 8), rng.randint(0, 8)
        stream = TextStream(unit="bytes", head=head, tail=tail)
        merge_head, merge_tail = rng.randint(0, 8), rng.randint(0, 8)
        whole = b"".join(_random_bytes(rng, kinds) for _ in range(rng.randint(1, 5)))
        # Random split points: multibyte characters are split across chunks.
        cuts = sorted(rng.sample(range(len(whole) + 1), min(len(whole) + 1, rng.randint(1, 6))))
        chunks = [whole[i:j] for i, j in zip([0, *cuts], [*cuts, len(whole)], strict=True)]
        data = b""
        mark = 0
        merged: Text | None = None
        merged_from = 0
        for chunk in chunks:
            data += chunk
            stream.write(chunk)
            delta, new_mark = stream.since(mark)
            _check(delta, data[mark:new_mark], head, tail)
            if merged is None:
                merged, merged_from = delta, mark
            else:
                merged = merge(merged, delta, merge_head, merge_tail)
                _check(merged, data[merged_from:new_mark], merge_head, merge_tail)
            mark = new_mark
        _check(stream.snapshot(), data, head, tail)
        final = stream.close()
        rest, _ = stream.since(mark)
        _check(rest, data[mark:], head, tail)
        _check(final, data, head, tail)
