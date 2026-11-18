# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cut: one whole string or bytes as a Text, the same as a one-write TextStream."""

from __future__ import annotations

import random

import pytest

from nooa.snapshot import Text, TextStream, cut, render_output, render_python


def _stream_of(data: str | bytes, head: int, tail: int, unit: str) -> Text:
    stream = TextStream(unit=unit, head=head, tail=tail)  # type: ignore[arg-type]
    stream.write(data)
    return stream.close()


def test_complete_when_it_fits() -> None:
    assert cut("hello", head=3, tail=2) == Text(head="hello", length=5, unit="chars")
    assert cut(b"", head=0, tail=0) == Text(head=b"", length=0, unit="bytes")


def test_cut_keeps_both_ends_and_counts_exactly() -> None:
    text = cut("0123456789", head=3, tail=2)
    assert (text.head, text.tail, text.length, text.omitted) == ("012", "89", 10, 5)
    assert text.reason == "max_length" and text.full is None
    assert render_output(text) == "012\n[... 5 chars omitted; full output not kept]\n89"
    assert render_python(cut("abcdef", head=1, tail=1)) == "str(len=6, [:1]='a', [-1:]='f')"


def test_head_only_and_tail_only() -> None:
    assert (cut("abcdef", head=2, tail=0).head, cut("abcdef", head=2, tail=0).tail) == ("ab", "")
    assert (cut("abcdef", head=0, tail=2).head, cut("abcdef", head=0, tail=2).tail) == ("", "ef")
    assert cut("abc", head=0, tail=0).omitted == 3


@pytest.mark.parametrize(("char", "size"), [("é", 2), ("€", 3), ("😀", 4)])
def test_bytes_cut_on_code_points(char: str, size: int) -> None:
    text = cut((char * 10).encode(), head=size + 1, tail=size + 1)
    assert text.head == char.encode() and text.tail == char.encode()
    assert text.length == 10 * size and text.omitted == 8 * size


def test_str_as_bytes_and_bytes_like() -> None:
    assert cut("é", head=8, tail=8, unit="bytes") == Text(head="é".encode(), length=2, unit="bytes")
    assert cut("a\ud800", head=9, tail=0, unit="bytes").head == b"a\\ud800"
    assert cut(bytearray(b"ab"), head=1, tail=0).head == b"a"
    assert cut(memoryview(b"ab"), head=2, tail=0).head == b"ab"


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        cut("x", head=-1, tail=0)
    with pytest.raises(ValueError):
        cut("x", head=1, tail=0, unit="lines")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cut(b"x", head=1, tail=0, unit="chars")
    with pytest.raises(TypeError):
        cut(10, head=1, tail=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("unit", ["chars", "bytes"])
def test_matches_a_one_write_stream_on_random_input(unit: str) -> None:
    rng = random.Random(f"cut-{unit}")
    pool = ["a", "z", "é", "€", "😀", "\n", "'"]
    raw = [b"\xff", b"\xe2\x82", b"\xef\xbf\xbd"]
    for _ in range(3000):
        head, tail = rng.randint(0, 9), rng.randint(0, 9)
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))
        data: str | bytes = text
        if unit == "bytes":
            data = text.encode() + b"".join(rng.choice(raw) for _ in range(rng.randint(0, 2)))
        assert cut(data, head=head, tail=tail) == _stream_of(data, head, tail, unit)
