# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TextStream and merge: bounded memory, UTF-8 cuts, overflow files, deltas."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir, OverflowWriter
from nooa.snapshot import Overflow, Text, TextStream, merge, render_output
from nooa.snapshot._codec import encode_text, safe_end


def _stream(unit: str = "bytes", head: int = 4, tail: int = 4, **kwargs) -> TextStream:
    return TextStream(unit=unit, head=head, tail=tail, **kwargs)  # type: ignore[arg-type]


# --- chars -----------------------------------------------------------------------


def test_chars_complete_until_head_plus_tail() -> None:
    stream = _stream("chars")
    stream.write("abcd")
    stream.write("efgh")

    assert stream.snapshot() == Text(head="abcdefgh", tail="", length=8, unit="chars", reason=None)


def test_chars_keep_head_and_rolling_tail() -> None:
    stream = _stream("chars")
    for piece in ("abc", "defg", "hij", "klmnop"):
        stream.write(piece)

    text = stream.close()

    assert text == Text(
        head="abcd",
        tail="mnop",
        length=16,
        unit="chars",
        reason="max_length",
    )
    assert render_output(text) == "abcd\n[... 8 chars omitted; full output not kept]\nmnop"


def test_memory_stays_bounded_with_huge_writes() -> None:
    stream = _stream("chars", head=10, tail=10)
    for _ in range(100):
        stream.write("x" * 100_000)

    window = stream._window
    assert window.head_len == 10 and window.tail_len == 10
    assert sum(len(chunk) for chunk in window._tail) == 10
    assert stream.snapshot().length == 10_000_000


def test_one_large_write_copies_only_what_it_keeps() -> None:
    import tracemalloc

    data = "x" * 20_000_000
    stream = _stream("chars", head=15_000, tail=15_000)
    tracemalloc.start()
    try:
        stream.write(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000  # the head and tail slices, not a 20 MB copy past the head
    text = stream.snapshot()
    assert (len(text.head), len(text.tail), text.length) == (15_000, 15_000, 20_000_000)


@pytest.mark.parametrize("unit", ["chars", "bytes"])
def test_window_matches_a_reference_on_random_writes(unit: str) -> None:
    rng = random.Random(f"window-{unit}")
    for _ in range(2000):
        head, tail = rng.randint(0, 6), rng.randint(0, 6)
        stream = _stream(unit, head=head, tail=tail)
        data = "".join(rng.choice("ab") * rng.randint(0, 9) for _ in range(rng.randint(0, 5)))
        whole = data if unit == "chars" else data.encode()
        cuts = sorted(rng.choices(range(len(whole) + 1), k=rng.randint(0, 4)))
        for i, j in zip([0, *cuts], [*cuts, len(whole)], strict=True):
            stream.write(whole[i:j])
        window = stream._window
        assert window.head() == whole[:head]
        assert window.tail() == (whole[head:][-tail:] if tail else whole[:0])


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


# --- overflow files --------------------------------------------------------------


@pytest.fixture
def folder() -> Iterator[OverflowDir]:
    with OverflowDir() as folder:
        yield folder


def _files(folder: OverflowDir) -> list[str]:
    return os.listdir(folder.path)


def test_overflow_file_holds_everything(folder: OverflowDir) -> None:
    stream = TextStream(unit="bytes", head=3, tail=3, overflow=folder)
    stream.write(b"0123456789")
    running = stream.snapshot()
    stream.write(b"abcdef")

    text = stream.close()

    assert text.full is not None
    path = Path(text.full.path)
    assert path.parent == Path(folder.path)
    assert path.read_bytes() == b"0123456789abcdef"
    assert running.full == Overflow(str(path), 10, 10)
    assert text.full == Overflow(str(path), 16, 16)
    assert text.reason == "max_length"
    assert render_output(text) == f"012\n[... 10 bytes omitted; full output: {path}]\ndef"


def test_overflow_bytes_cuts_the_file(folder: OverflowDir) -> None:
    stream = TextStream(unit="bytes", head=2, tail=2, overflow=folder, overflow_bytes=5)
    stream.write(b"0123456789" * 1000)

    text = stream.close()

    assert text.full is not None
    path = text.full.path
    assert Path(path).read_bytes() == b"01234"
    assert text.full == Overflow(path, 5, 10_000)
    assert text.full.cut  # the file is cut: ``full`` says so, not the reason
    assert text.reason == "max_length"
    assert render_output(text) == (
        f"01\n[... 9,996 bytes omitted; {path} holds the first 5 of 10,000 bytes]\n89"
    )


def test_overflow_bytes_never_exceeds_the_folders_file_bytes() -> None:
    with OverflowDir(file_bytes=3) as folder:
        stream = TextStream(unit="bytes", head=1, tail=1, overflow=folder, overflow_bytes=100)
        stream.write(b"0123456789")
        full = stream.close().full
        assert full is not None and full.kept == 3


def test_chars_stream_writes_utf8_to_its_file(folder: OverflowDir) -> None:
    stream = TextStream(unit="chars", head=1, tail=1, overflow=folder)
    stream.write("é€😀")

    text = stream.close()

    assert text.full is not None
    assert Path(text.full.path).read_text(encoding="utf-8") == "é€😀"
    assert text.full == Overflow(text.full.path, 9, 9)
    assert text.length == 3 and text.omitted == 1


def test_chars_overflow_cut_lands_on_a_code_point(folder: OverflowDir) -> None:
    # overflow_bytes=3 keeps b"\xc3\xa9", never a split code point such as b"\xc3\xa9\xc3".
    stream = _stream("chars", 1, 1, overflow=folder, overflow_bytes=3)
    stream.write("ééé")
    stream.write("x")  # fits the byte left, but the file is a prefix: not written

    text = stream.close()

    assert text.full is not None
    path = Path(text.full.path)
    assert path.read_bytes() == "é".encode()
    assert path.read_text(encoding="utf-8") == "é"
    assert text.full == Overflow(str(path), 2, 7)
    assert text.full.cut


@pytest.mark.parametrize("max_bytes", range(10))
def test_chars_overflow_file_is_valid_utf8_for_any_max_bytes(
    folder: OverflowDir, max_bytes: int
) -> None:
    stream = _stream("chars", 1, 1, overflow=folder, overflow_bytes=max_bytes)
    for part in ("a", "é€", "😀", "b"):
        stream.write(part)

    text = stream.close()

    assert text.full is not None
    written = Path(text.full.path).read_bytes()
    assert "aé€😀b".encode().startswith(written)
    written.decode("utf-8")
    assert text.full.kept == len(written) <= max_bytes


def test_chars_overflow_file_keeps_a_surrogate_escape_as_its_byte(folder: OverflowDir) -> None:
    # The documented exception to "UTF-8": a surrogateescape character is written as its
    # original byte, so the file is not valid UTF-8.
    stream = _stream("chars", 0, 0, overflow=folder)
    stream.write("a" + chr(0xDCFF) + "b")

    full = stream.close().full

    assert full is not None
    assert Path(full.path).read_bytes() == b"a\xffb"
    with pytest.raises(UnicodeDecodeError):
        Path(full.path).read_bytes().decode("utf-8")


def test_a_streams_file_is_private(folder: OverflowDir) -> None:
    stream = _stream("bytes", 1, 1, overflow=folder)
    stream.write(b"abc")

    full = stream.close().full
    assert full is not None
    assert os.stat(full.path).st_mode & 0o777 == 0o600
    assert os.stat(folder.path).st_mode & 0o777 == 0o700


def test_uncut_stream_makes_no_file(folder: OverflowDir) -> None:
    # The file opens only once the output is cut: a complete output never touches disk.
    stream = TextStream(unit="bytes", head=10, tail=10, overflow=folder)
    stream.write(b"small")
    assert _files(folder) == []
    stream.write(b"0" * 15)  # 20 bytes: still head + tail, still complete

    text = stream.close()

    assert text.full is None and text.complete
    assert _files(folder) == []


def test_the_file_opens_at_the_first_cut_with_everything_written_before() -> None:
    with OverflowDir(max_files=1) as folder:
        stream = TextStream(unit="bytes", head=2, tail=2, overflow=folder)
        stream.write(b"ab")
        stream.write(b"cd")
        assert _files(folder) == [] and stream.snapshot().full is None
        stream.write(b"e")  # cut now
        (name,) = _files(folder)
        assert (Path(folder.path) / name).read_bytes() == b"abcde"
        assert stream.snapshot().full == Overflow(str(Path(folder.path) / name), 5, 5)
        stream.close()


@pytest.mark.parametrize("seed", range(40))
def test_a_lazily_opened_chars_file_is_what_writing_from_the_start_gives(
    folder: OverflowDir, seed: int
) -> None:
    # Each write is encoded on its own (encode_text), and a cut file ends at safe_end
    # within the write it ends in: opening at the first cut must give the same bytes,
    # with surrogate escapes, lone surrogates and multi-byte characters in any order.
    rng = random.Random(seed)
    alphabet = ["a", "é", "😀", "\udcc3", "\udc80", "\udcff", "\ud800", "\udfff"]
    writes = [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 4)))
        for _ in range(rng.randrange(1, 12))
    ]
    head, tail, cap = rng.randrange(0, 6), rng.randrange(0, 6), rng.randrange(0, 40)
    stream = _stream("chars", head, tail, overflow=folder, overflow_bytes=cap)
    expected, room, produced = b"", cap, 0  # writing every write from the start
    for data in writes:
        stream.write(data)
        encoded = encode_text(data)
        produced += len(encoded)
        if room > 0:
            part = encoded[: safe_end(encoded, room)] if len(encoded) > room else encoded
            room = room - len(part) if len(encoded) <= room else 0
            expected += part

    text = stream.close()

    if text.complete:
        assert text.full is None and _files(folder) == []
    else:
        assert text.full is not None
        assert Path(text.full.path).read_bytes() == expected
        assert (text.full.kept, text.full.total) == (len(expected), produced)


def test_overflow_without_a_file_falls_back_to_memory() -> None:
    with OverflowDir(max_files=0) as folder:  # the folder gives no file
        stream = TextStream(unit="bytes", head=2, tail=2, overflow=folder)
        stream.write(b"abcdefgh")

        text = stream.close()

    assert text.full is None  # no file: ``full`` says so, not the reason
    assert text.reason == "max_length"
    assert render_output(text) == "ab\n[... 4 bytes omitted; full output not kept]\ngh"


def test_a_stream_of_a_closed_folder_keeps_memory_only() -> None:
    folder = OverflowDir()
    folder.close()
    stream = TextStream(unit="bytes", head=2, tail=2, overflow=folder)
    stream.write(b"abcdefgh")
    assert stream.close().full is None


def test_overflow_write_failure_falls_back_to_memory(
    folder: OverflowDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: OverflowWriter, data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(OverflowWriter, "write", broken)
    stream = TextStream(unit="bytes", head=2, tail=2, overflow=folder)
    stream.write(b"abcdefgh")

    text = stream.close()
    assert text.full is None and text.reason == "max_length"
    assert _files(folder) == []  # the partial file is removed
    assert (text.head, text.tail, text.length) == (b"ab", b"gh", 8)


def test_chars_lone_surrogate_without_a_file_does_not_raise() -> None:
    stream = _stream("chars", head=10, tail=10)
    stream.write("a\ud800b")

    assert stream.close() == Text(head="a\ud800b", tail="", length=3, unit="chars", reason=None)


def test_chars_lone_surrogate_written_backslash_escaped(folder: OverflowDir) -> None:
    stream = TextStream(unit="chars", head=1, tail=1, overflow=folder)
    stream.write("a\ud800b")

    text = stream.close()
    assert (text.head, text.tail) == ("a", "b") and text.full is not None
    assert Path(text.full.path).read_bytes() == b"a\\ud800b"
    assert text.full.kept == len(b"a\\ud800b")


def test_bytes_stream_accepts_str_with_lone_surrogate() -> None:
    stream = _stream("bytes", head=20, tail=20)
    stream.write("a\ud800b")

    assert stream.close().head == b"a\\ud800b"


def test_running_snapshot_matches_the_file_on_disk(folder: OverflowDir) -> None:
    stream = TextStream(unit="bytes", head=10, tail=10, overflow=folder)
    stream.write(b"y" * 100)

    text = stream.snapshot()
    assert text.full is not None and text.full.kept == 100 == os.path.getsize(text.full.path)
    stream.write(b"z" * 7)
    text = stream.snapshot()
    assert text.full is not None and text.full.kept == 107 == os.path.getsize(text.full.path)
    stream.close()


def test_overflow_close_failure_falls_back_to_memory(
    folder: OverflowDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = TextStream(unit="bytes", head=2, tail=2, overflow=folder)
    stream.write(b"abcdefgh")
    real_close = os.close

    def failing_close(fd: int) -> None:
        real_close(fd)
        raise OSError("disk full")

    monkeypatch.setattr(os, "close", failing_close)
    text = stream.close()
    monkeypatch.undo()
    assert text.full is None and text.reason == "max_length"
    assert _files(folder) == []  # a file that did not close cleanly is not kept
    assert (text.head, text.tail, text.length) == (b"ab", b"gh", 8)


@pytest.mark.parametrize("unit", ["chars", "bytes"])
def test_snapshots_interleaved_with_writes_see_every_write(unit: str) -> None:
    stream = _stream(unit, head=4, tail=4)
    whole = "" if unit == "chars" else b""
    taken: list[tuple[Text, str | bytes]] = []
    for piece in ("ab", "cd", "ef", "gh", "ij", "k", "lmnopq"):
        data = piece if unit == "chars" else piece.encode()
        stream.write(data)
        whole += data  # type: ignore[operator]
        for _ in range(2):  # reading twice changes nothing
            text = stream.snapshot()
            assert text.length == len(whole)
            if len(whole) <= 8:
                assert (text.head, text.tail, text.complete) == (whole, whole[:0], True)
            else:
                assert (text.head, text.tail) == (whole[:4], whole[-4:])
        taken.append((text, whole))

    final = stream.close()
    assert (final.head, final.tail, final.length) == (whole[:4], whole[-4:], len(whole))
    for text, seen in taken:  # earlier snapshots still describe the stream as it was
        assert text.length == len(seen)
        assert text.head + text.tail == (seen if len(seen) <= 8 else seen[:4] + seen[-4:])


# --- write types / merge -----------------------------------------------------------------


@pytest.mark.parametrize("data", [b"abc", bytearray(b"abc"), memoryview(b"abc")])
def test_chars_stream_rejects_bytes(data: object) -> None:
    stream = TextStream(unit="chars", head=4, tail=4)
    stream.write("ok")

    with pytest.raises(TypeError, match="bytes stream"):
        stream.write(data)  # type: ignore[arg-type]

    # Nothing of the rejected chunk was counted or kept.
    assert stream.close() == Text(head="ok", tail="", length=2, unit="chars", reason=None)


@pytest.mark.parametrize("data", [10, [1, 2], 1.5, None, object()])
def test_bytes_stream_rejects_non_bytes(data: object) -> None:
    # write(10) raises TypeError rather than recording ten NUL bytes (bytes(10)).
    stream = _stream("bytes", head=100, tail=10)

    with pytest.raises(TypeError, match="bytes TextStream"):
        stream.write(data)  # type: ignore[arg-type]

    assert stream.close() == Text(head=b"", length=0, unit="bytes")


def test_bytes_stream_accepts_bytes_like_values() -> None:
    stream = _stream("bytes", head=100, tail=10)
    stream.write(b"ab")
    stream.write(bytearray(b"cd"))
    stream.write(memoryview(b"xefx")[1:3])

    assert stream.close().head == b"abcdef"


def test_bytes_stream_still_accepts_str() -> None:
    stream = TextStream(unit="bytes", head=8, tail=8)
    stream.write("é")
    stream.write(b"!")

    assert stream.close() == Text(head="é!".encode(), tail=b"", length=3, unit="bytes", reason=None)


def test_merge_keeps_length_and_bounds() -> None:
    a = Text(head="abcdef", tail="", length=6, unit="chars", reason=None)
    b = Text(
        head="gh",
        tail="yz",
        length=40,
        unit="chars",
        reason="max_length",
    )

    merged = merge(a, b, head=3, tail=3)

    assert merged.length == 46
    assert (merged.head, merged.tail) == ("abc", "yz")  # b's own gap stops the tail
    assert merged.omitted == 46 - 3 - 2
    assert not merged.complete


def test_merge_of_small_complete_deltas_is_complete() -> None:
    a = Text(head="ab", tail="", length=2, unit="chars", reason=None)
    b = Text(head="cd", tail="", length=2, unit="chars", reason=None)

    assert merge(a, b, head=2, tail=2) == Text(
        head="abcd", tail="", length=4, unit="chars", reason=None
    )


def test_merge_cuts_complete_deltas_that_are_too_long() -> None:
    a = Text(head="abcde", tail="", length=5, unit="chars", reason=None)
    b = Text(head="fghij", tail="", length=5, unit="chars", reason=None)

    merged = merge(a, b, head=2, tail=3)

    assert (merged.head, merged.tail, merged.length, merged.omitted) == ("ab", "hij", 10, 5)
    assert render_output(merged) == "ab\n[... 5 chars omitted; full output not kept]\nhij"


def test_merge_of_batches_matches_the_stream_length() -> None:
    stream = TextStream(unit="bytes", head=8, tail=8)
    merged: Text | None = None
    for chunk in (b"abc", "é€".encode() * 20, b"xyz" * 50, "😀".encode()):
        stream.write(chunk)
        batch = TextStream(unit="bytes", head=8, tail=8)
        batch.write(chunk)
        delta = batch.close()
        merged = delta if merged is None else merge(merged, delta, head=8, tail=8)

    assert merged is not None
    assert merged.length == stream.snapshot().length
    assert len(merged.head) <= 8 and len(merged.tail) <= 8
    assert merged.tail.endswith("😀".encode())
    assert merged.omitted == merged.length - len(merged.head) - len(merged.tail)


def test_merge_bytes_cuts_on_code_points() -> None:
    a = Text(head="😀😀".encode(), tail=b"", length=8, unit="bytes", reason=None)
    b = Text(head="😀😀".encode(), tail=b"", length=8, unit="bytes", reason=None)

    merged = merge(a, b, head=5, tail=7)

    assert (merged.head, merged.tail) == ("😀".encode(), "😀".encode())
    assert merged.omitted == 8


def test_merge_rejects_mixed_units() -> None:
    with pytest.raises(ValueError):
        merge(
            Text(head="", tail="", length=0, unit="chars", reason=None),
            Text(head=b"", tail=b"", length=0, unit="bytes", reason=None),
            1,
            1,
        )


def test_context_manager_closes() -> None:
    with TextStream(unit="chars", head=1, tail=1) as stream:
        stream.write("abc")
    with pytest.raises(ValueError):
        stream.write("d")


# --- short tails, split U+FFFD, merge units ---------------------------------------


def test_merge_keeps_raw_bytes_exactly() -> None:
    def closed(data: bytes) -> Text:
        stream = _stream("bytes", head=10, tail=10)
        stream.write(data)
        return stream.close()

    invalid, plain = closed(b"\xff"), closed(b"x")

    merged = merge(invalid, plain, head=1, tail=1)
    assert merged == Text(head=b"\xffx", tail=b"", length=2, unit="bytes", reason=None)
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
        merged: Text | None = None
        for chunk in chunks:
            data += chunk
            stream.write(chunk)
            batch = TextStream(unit="bytes", head=head, tail=tail)
            batch.write(chunk)
            delta = batch.close()
            _check(delta, chunk, head, tail)
            if merged is None:
                merged = delta
            else:
                merged = merge(merged, delta, merge_head, merge_tail)
                _check(merged, data, merge_head, merge_tail)
            _check(stream.snapshot(), data, head, tail)
        _check(stream.close(), data, head, tail)
