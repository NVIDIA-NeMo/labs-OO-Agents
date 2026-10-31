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


def test_chars_stream_decodes_bytes_incrementally() -> None:
    stream = _stream("chars", head=10, tail=10)
    data = "é😀".encode()
    for byte in data:
        stream.write(bytes([byte]))

    assert stream.close() == Text("é😀", "", 2, "chars", True, None)


# --- bytes and UTF-8 cut points ------------------------------------------------------


@pytest.mark.parametrize(("char", "size"), [("é", 2), ("€", 3), ("😀", 4)])
def test_bytes_head_and_tail_cut_on_code_points(char: str, size: int) -> None:
    head_limit = size + 1  # one byte into the second code point
    stream = TextStream(unit="bytes", head=head_limit, tail=head_limit)
    stream.write((char * 10).encode())

    text = stream.close()

    assert text.head == char  # the split code point moves to the omitted part
    assert text.tail == char
    assert text.length == 10 * size
    assert text.omitted == 10 * size - 2 * size
    assert text.replaced == 0


def test_four_byte_code_point_written_one_byte_at_a_time() -> None:
    stream = TextStream(unit="bytes", head=2, tail=2)
    for byte in "😀😀".encode():
        stream.write(bytes([byte]))

    text = stream.close()
    # Neither 2-byte window can hold a whole 4-byte code point.
    assert (text.head, text.tail, text.omitted, text.replaced) == ("", "", 8, 0)


def test_invalid_bytes_count_replacements() -> None:
    stream = TextStream(unit="bytes", head=100, tail=100)
    stream.write(b"ok\xff\xfeend")
    stream.write("\ufffd".encode())  # a genuine replacement character is not counted

    text = stream.close()
    assert text.head == "ok\ufffd\ufffdend\ufffd"
    assert text.replaced == 2


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
    assert text.full.cut
    assert text.reason == "file_limit"
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

    assert text.full is None
    assert text.reason == "not_kept"
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
    assert text.full is None and text.reason == "not_kept"
    assert not path.exists()  # the partial file is removed
    assert (text.head, text.tail, text.length) == ("ab", "gh", 8)


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

    assert stream.close().head == "a\\ud800b"


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
    assert text.full is None and text.reason == "not_kept"
    assert not path.exists()  # handled like a write failure
    assert (text.head, text.tail, text.length) == ("ab", "gh", 8)


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


def test_since_replaced_counts_only_the_window_for_chars() -> None:
    stream = _stream("chars", head=100, tail=100)
    stream.write(b"\xff")
    first, mark = stream.since(0)
    stream.write("ok")
    second, mark2 = stream.since(mark)
    stream.write(b"\xfe\xfd")
    third, _ = stream.since(mark2)

    assert (first.replaced, second.replaced, third.replaced) == (1, 0, 2)
    assert merge(first, second, 100, 100).replaced == 1
    assert merge(merge(first, second, 100, 100), third, 100, 100).replaced == 3
    assert stream.snapshot().replaced == 3


def test_since_replaced_for_an_unremembered_mark_never_over_reports() -> None:
    stream = _stream("chars", head=100, tail=100)
    stream.write("a")
    stream.write(b"\xff")
    stream.write("b")

    assert stream.since(0)[0].replaced == 1  # the whole stream: the cumulative count
    # Mark 1 was never returned; the nearest remembered mark after it is 3,
    # so the window's count is a lower bound (the U+FFFD at 1 isn't counted).
    assert stream.since(1)[0].replaced == 0
    stream.write(b"\xff")
    assert stream.since(1)[0].replaced == 1  # counted since mark 3
    assert stream.since(3)[0].replaced == 1  # remembered: exact


def test_since_remembers_only_recent_marks() -> None:
    stream = _stream("chars", head=10, tail=10)
    marks = []
    for _ in range(40):
        stream.write(b"\xff")
        marks.append(stream.since(marks[-1] if marks else 0)[1])

    assert len(stream._replaced_at) <= 16
    stream.write(b"\xff")
    assert stream.since(marks[-1])[0].replaced == 1


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

    assert (first.head, first.length, mark) == ("a😀"[:1], 1, 5)
    assert (second.head, second.length, second.replaced) == ("😀", 4, 0)


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
    assert len(merged.head.encode()) <= 8 and len(merged.tail.encode()) <= 8
    assert merged.tail.endswith("😀")
    assert merged.omitted == merged.length - len(merged.head.encode()) - len(merged.tail.encode())


def test_merge_bytes_cuts_on_code_points() -> None:
    a = Text("😀😀", "", 8, "bytes", True, None)
    b = Text("😀😀", "", 8, "bytes", True, None)

    merged = merge(a, b, head=5, tail=7)

    assert (merged.head, merged.tail) == ("😀", "😀")
    assert merged.omitted == 8


def test_merge_rejects_mixed_units() -> None:
    with pytest.raises(ValueError):
        merge(Text("", "", 0, "chars", True, None), Text("", "", 0, "bytes", True, None), 1, 1)


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


def test_genuine_replacement_split_across_writes_is_not_counted() -> None:
    stream = _stream("chars", head=10, tail=10)
    stream.write(b"\xef")
    stream.write(b"\xbf\xbd")
    stream.write(b"\xff")

    text = stream.close()
    assert text.head == "\ufffd\ufffd" and text.replaced == 1


def test_str_write_ends_an_incomplete_byte_sequence() -> None:
    stream = _stream("chars", head=10, tail=10)
    stream.write(b"\xe2\x82")
    stream.write("x")
    stream.write(b"\xac")

    text = stream.close()
    assert text.head == "\ufffdx\ufffd" and text.replaced == 2


def test_str_write_resets_the_genuine_replacement_carry() -> None:
    stream = _stream("chars", head=10, tail=10)
    stream.write(b"\xef\xbf")
    stream.write("x")
    stream.write(b"\xbd")

    text = stream.close()
    assert text.head == "\ufffdx\ufffd" and text.replaced == 2


def test_close_after_a_final_str_keeps_the_order() -> None:
    stream = _stream("chars", head=10, tail=10)
    stream.write(b"a\xe2\x82")
    stream.write("x")

    text = stream.close()
    assert text.head == "a\ufffdx" and text.replaced == 1


def test_merge_counts_replaced_bytes_as_original_units() -> None:
    def closed(data: bytes) -> Text:
        stream = _stream("bytes", head=10, tail=10)
        stream.write(data)
        return stream.close()

    invalid, plain = closed(b"\xff"), closed(b"x")

    merged = merge(invalid, plain, head=1, tail=1)
    assert merged == Text(
        "\ufffdx", "", 2, "bytes", True, None, replaced=1, head_units=2, tail_units=0
    )
    cut = merge(closed(b"\xffab"), closed(b"cd\xfe"), head=1, tail=1)
    assert (cut.head, cut.tail, cut.omitted) == ("\ufffd", "\ufffd", 4)
    assert cut.replaced == 2
    dropped = merge(closed(b"\xffabc"), closed(b"de"), head=0, tail=2)
    assert (dropped.tail, dropped.omitted, dropped.replaced) == ("de", 4, 0)


# --- merge: recorded original units (CodeRabbit 4432013584) -------------------------


def test_merge_uses_recorded_units_for_genuine_and_decoded_replacements() -> None:
    # Head: a genuine U+FFFD (3 bytes); tail: a decoding replacement of 1 byte.
    stream = TextStream(unit="bytes", head=3, tail=1)
    stream.write(b"\xef\xbf\xbdX\xff")
    a = stream.snapshot()
    assert (a.head, a.tail, a.omitted, a.replaced) == ("\ufffd", "\ufffd", 1, 1)
    assert a.units() == (3, 1)
    empty = TextStream(unit="bytes", head=3, tail=1).snapshot()

    merged = merge(a, empty, head=1, tail=0)

    # The genuine U+FFFD stands for 3 bytes: it does not fit a 1-byte head.
    assert (merged.head, merged.tail, merged.omitted, merged.replaced) == ("", "", 5, 0)
    assert merged.units() == (0, 0)
    kept_genuine = merge(a, empty, head=3, tail=0)
    assert (kept_genuine.head, kept_genuine.omitted) == ("\ufffd", 2)
    assert kept_genuine.replaced == 0  # the kept U+FFFD is the genuine one
    kept_replacement = merge(a, empty, head=0, tail=1)
    assert (kept_replacement.tail, kept_replacement.omitted) == ("\ufffd", 4)
    assert kept_replacement.replaced == 1


def test_capture_of_bytes_records_units() -> None:
    from nooa.snapshot import Limits, capture

    node = capture(b"\xff" * 50 + b"\xef\xbf\xbd" * 50, Limits(max_string=20, tail_share=0.5))
    assert isinstance(node, Text) and not node.complete
    head_units, tail_units = node.units()
    assert (node.head_units, node.tail_units) == (head_units, tail_units)
    assert head_units == len(node.head)  # each \xff is one byte
    assert tail_units == 3 * len(node.tail)  # each genuine U+FFFD is three
    assert node.length == head_units + tail_units + node.omitted


_FRAGMENTS = {
    "valid": [b"a", b"z", "é".encode(), "€".encode(), "😀".encode()],
    "genuine": [b"\xef\xbf\xbd"],
    "single": [b"\xff", b"\x80", b"\xfe"],
    "multi": [b"\xe2\x82", b"\xf0\x9f\x98", b"\xed\xa0", b"\xe2"],
}


def _random_bytes(rng: random.Random, kinds: list[str]) -> bytes:
    pool = [fragment for kind in kinds for fragment in _FRAGMENTS[kind]]
    return b"".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))


def _decode(data: bytes) -> tuple[str, int]:
    text = data.decode("utf-8", errors="replace")
    return text, text.count("\ufffd") - data.count(b"\xef\xbf\xbd")


def _check(text: Text, data: bytes, head: int, tail: int, *, exact: bool) -> None:
    """``text`` describes ``data`` within ``head``/``tail`` original bytes."""
    head_units, tail_units = text.units()
    assert (text.head_units, text.tail_units) == (head_units, tail_units)
    assert text.length == len(data)
    assert text.length == head_units + tail_units + text.omitted
    if text.complete:
        assert (text.head + text.tail, text.omitted) == (_decode(data)[0], 0)
        assert text.replaced == _decode(data)[1]
        return
    assert head_units <= head and tail_units <= tail
    # The fewest original bytes the kept strings can stand for: within the counts.
    fewest_head = min(k for k in range(len(data) + 1) if _decode(data[:k])[0] == text.head)
    fewest_tail = min(
        k for k in range(len(data) + 1) if _decode(data[len(data) - k :])[0] == text.tail
    )
    assert fewest_head <= head_units and fewest_tail <= tail_units
    assert fewest_head + fewest_tail + text.omitted <= len(data)
    kept_head = _decode(data[:head_units]) if exact else _decode(data[:fewest_head])
    kept_tail = _decode(data[len(data) - (tail_units if exact else fewest_tail) :])
    if exact:
        assert kept_head[0] == text.head and kept_tail[0] == text.tail
        assert text.replaced == kept_head[1] + kept_tail[1]
    else:
        # Never counts a replacement it dropped.
        assert text.replaced <= text.head.count("\ufffd") + text.tail.count("\ufffd")


@pytest.mark.parametrize(
    ("kinds", "exact_merge"),
    [
        (["valid", "genuine"], True),
        (["valid", "single"], True),
        (["valid", "multi"], False),
        (["valid", "genuine", "single", "multi"], False),
    ],
)
def test_bytes_units_hold_for_random_streams(kinds: list[str], exact_merge: bool) -> None:
    rng = random.Random(f"units-{'-'.join(kinds)}")
    for _ in range(400):
        head, tail = rng.randint(0, 8), rng.randint(0, 8)
        stream = TextStream(unit="bytes", head=head, tail=tail)
        merge_head, merge_tail = rng.randint(0, 8), rng.randint(0, 8)
        data = b""
        mark = 0
        merged: Text | None = None
        merged_from = 0
        for _chunk in range(rng.randint(1, 5)):
            chunk = _random_bytes(rng, kinds)
            data += chunk
            stream.write(chunk)
            delta, new_mark = stream.since(mark)
            # since(): a window of whole code points, exact.
            _check(delta, data[mark:new_mark], head, tail, exact=True)
            if merged is None:
                merged, merged_from = delta, mark
            else:
                merged = merge(merged, delta, merge_head, merge_tail)
                _check(
                    merged, data[merged_from:new_mark], merge_head, merge_tail, exact=exact_merge
                )
            mark = new_mark
        _check(stream.snapshot(), data, head, tail, exact=True)
