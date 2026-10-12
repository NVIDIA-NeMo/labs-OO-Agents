# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actionable, bounded diagnostics without changing exact replacement semantics."""

import re

import pytest

from nooa.tools.shell_tools import ShellTools, _ambiguous_replace_message


@pytest.fixture
async def shell(tmp_path):
    tool = ShellTools(cwd=str(tmp_path))
    yield tool
    await tool.close()


async def ambiguous_error(shell, tmp_path, content, old, new="replacement", filename="example.py"):
    path = tmp_path / filename
    path.write_text(content)
    before = path.read_bytes()
    with pytest.raises(ValueError) as error:
        await shell.replace(filename, old, new)
    assert path.read_bytes() == before
    return str(error.value)


def headers(message):
    return re.findall(r"(?m)^Match \d+: (.*)$", message)


def sections(message):
    return re.split(r"(?m)^Match \d+: ", message)[1:]


def source_rows(message):
    return [
        (mark, int(number), text)
        for mark, number, text in re.findall(r"(?m)^([> ]) (\d+)\| (.*)$", message)
    ]


def assert_bounded(message):
    assert len(message) <= 12_000
    for line in message.splitlines():
        if re.match(r"^[> ] \d+\|", line):
            assert len(line) <= 160
    assert all(char.isprintable() or char == "\n" for char in message)


async def test_ambiguous_replace_shows_distinct_context_and_recovery(shell, tmp_path):
    content = "def first():\n    return value\n\ndef second():\n    return value\n"
    message = await ambiguous_error(shell, tmp_path, content, "return value")

    assert message.startswith("old text matched 2 times in example.py.")
    assert "No file was changed." in message
    assert "line:column, 1-based, inclusive; Python characters, not tab width" in message
    assert headers(message) == ["lines 2-2 (2:5-2:16)", "lines 5-5 (5:5-5:16)"]
    assert "> 2|     return value\\n" in message  # Preserve usable space indentation.
    assert "def first():" in message
    assert "def second():" in message
    assert "add surrounding context" in message
    assert "read()" in message
    assert "Showing 2 of 2 matches; 0 more matches not shown." in message

    # Retrying with surrounding context selects one occurrence, as before.
    await shell.replace(
        "example.py", "def second():\n    return value", "def second():\n    return other"
    )
    assert (tmp_path / "example.py").read_text() == content.replace(
        "def second():\n    return value", "def second():\n    return other"
    )


@pytest.mark.parametrize(
    ("content", "old", "positions"),
    [
        ("x = 1; x = 1", "x = 1", ["1:1-1:5", "1:8-1:12"]),
        ("a\nb\na\nb\n", "a\nb\n", ["1:1-2:2", "3:1-4:2"]),
        ("λ 🙂 value\nβ value", "value", ["1:5-1:9", "2:3-2:7"]),
        ("aaaaa", "aa", ["1:1-1:2", "1:3-1:4"]),
        ("value\nvalue", "value", ["1:1-1:5", "2:1-2:5"]),
        ("\tvalue\tvalue", "value", ["1:2-1:6", "1:8-1:12"]),
        ("a\n\na\n\n", "\n", ["1:2-1:2", "2:1-2:1", "3:2-3:2", "4:1-4:1"]),
        ("xxa\nb!xxa\nb", "a\nb", ["1:3-2:1", "2:5-3:1"]),
    ],
)
async def test_ambiguous_replace_reports_nonoverlapping_character_positions(
    shell, tmp_path, content, old, positions
):
    message = await ambiguous_error(shell, tmp_path, content, old)
    assert f"matched {len(positions)} times" in message
    assert [header.split("(")[1].removesuffix(")") for header in headers(message)] == positions


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"],
)
async def test_ambiguous_replace_line_numbers_agree_with_read(shell, tmp_path, separator):
    # read_text() normalizes CR and CRLF, as both read() and replace() always did.
    content = f"before{separator}λ hit{separator}β hit{separator}after{separator}"
    message = await ambiguous_error(shell, tmp_path, content, "hit")
    assert headers(message) == ["lines 2-2 (2:3-2:5)", "lines 3-3 (3:3-3:5)"]
    assert [number for _, number, _ in source_rows(sections(message)[0])] == [1, 2, 3, 4]
    second_line = await shell.read("example.py", (2, 2))
    assert second_line.text.splitlines() == ["λ hit"]
    assert_bounded(message)


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"],
)
@pytest.mark.parametrize("newline_only", [False, True])
def test_raw_separator_endpoints_follow_splitlines(separator, newline_only):
    # Raw CRLF must be tested directly: the filesystem API normalizes it to LF.
    content = f"a{separator}a{separator}"
    old = separator if newline_only else "a" + separator
    message = _ambiguous_replace_message("example.py", content, old, content.count(old))
    first_col = 2 if newline_only else 1
    last_col = 1 + len(separator)
    assert headers(message) == [
        f"lines 1-1 (1:{first_col}-1:{last_col})",
        f"lines 2-2 (2:{first_col}-2:{last_col})",
    ]
    assert [number for _, number, _ in source_rows(sections(message)[0])] == [1, 2]
    assert len(content.splitlines(keepends=True)) == 2
    assert_bounded(message)


@pytest.mark.parametrize("old", ["\r", "\n"])
def test_raw_partial_crlf_matches_stay_on_preceding_line(old):
    message = _ambiguous_replace_message("example.py", "a\r\na\r\n", old, 2)
    col = 2 if old == "\r" else 3
    assert headers(message) == [f"lines 1-1 (1:{col}-1:{col})", f"lines 2-2 (2:{col}-2:{col})"]


@pytest.mark.parametrize("matched_lines", [1, 2, 3, 4, 5, 20_000])
async def test_five_lines_before_and_after_entire_match(shell, tmp_path, matched_lines):
    before = [f"before {n}\n" for n in range(1, 7)]
    after = [f"after {n}\n" for n in range(1, 7)]
    old = "".join(f"match {n}\n" for n in range(1, matched_lines + 1))
    block = "".join(before) + old + "".join(after)
    message = await ambiguous_error(shell, tmp_path, block * 2, old)
    assert_bounded(message)
    for index, section in enumerate(sections(message)):
        first = index * (12 + matched_lines) + 7
        last = first + matched_lines - 1
        assert headers(message)[index] == (
            f"lines {first}-{last} ({first}:1-{last}:{len(f'match {matched_lines}') + 1})"
        )
        rows = source_rows(section)
        assert [number for mark, number, _ in rows if mark != ">"] == (
            list(range(first - 5, first)) + list(range(last + 1, last + 6))
        )
        expected = (
            list(range(first, last + 1))
            if matched_lines <= 4
            else [first, first + 1, last - 1, last]
        )
        assert [number for mark, number, _ in rows if mark == ">"] == expected
        assert "before 1" not in section
        assert "after 6" not in section
        if matched_lines > 4:
            assert f"... {matched_lines - 4} matched lines omitted ..." in section
        else:
            assert "matched lines omitted" not in section


@pytest.mark.parametrize("terminal_newline", [False, True])
async def test_borders_have_no_phantom_context_lines(shell, tmp_path, terminal_newline):
    content = "hit\n\nhit" + ("\n" if terminal_newline else "")
    message = await ambiguous_error(shell, tmp_path, content, "hit")
    for section in sections(message):
        assert [number for _, number, _ in source_rows(section)] == [1, 2, 3]
    assert source_rows(sections(message)[0])[1] == (" ", 2, "\\n")


async def test_ambiguous_replace_bounds_many_matches_and_escapes_context(shell, tmp_path):
    content = "\t\x1b[31mneedle\x00 " * 1_000
    message = await ambiguous_error(shell, tmp_path, content, "needle", "NEW" * 100_000)
    assert "matched 1000 times" in message
    assert len(headers(message)) == 5
    assert "Showing 5 of 1000 matches; 995 more matches not shown." in message
    assert "\\t" in message
    assert "\\x1b" in message
    assert "\\x00" in message
    assert "NEW" not in message
    assert "clipped" in message
    assert all("needle" in section for section in sections(message))
    assert_bounded(message)


async def test_ambiguous_replace_bounds_large_multiline_old_text(shell, tmp_path):
    old = "match line\n" * 20_000
    message = await ambiguous_error(shell, tmp_path, old + old, old)
    assert "matched 2 times" in message
    assert headers(message) == [
        "lines 1-20000 (1:1-20000:11)",
        "lines 20001-40000 (20001:1-40000:11)",
    ]
    assert message.count("19996 matched lines omitted") == 2
    assert [number for mark, number, _ in source_rows(message) if mark == ">"] == [
        1,
        2,
        19999,
        20000,
        20001,
        20002,
        39999,
        40000,
    ]
    assert_bounded(message)


async def test_ambiguous_replace_bounds_heavily_escaped_excerpt(shell, tmp_path):
    content = "\U000e0001" * 100 + "hit" + "\U000e0001" * 100 + "hit"
    message = await ambiguous_error(shell, tmp_path, content, "hit")
    assert headers(message) == ["lines 1-1 (1:101-1:103)", "lines 1-1 (1:204-1:206)"]
    assert message.count("hit") == 2
    for _, _, text in source_rows(message):
        assert "hit" in text
        # Every backslash introduces one complete escape; there are no cuts.
        assert re.fullmatch(r"(?:[^\\]|\\U[0-9a-f]{8})*", text)
    assert_bounded(message)


@pytest.mark.parametrize("old", ["NEEDLE", "NEEDLE" + "z" * 200_000])
async def test_long_lines_crop_around_late_match_not_start_of_line(shell, tmp_path, old):
    block = "x" * 100_000 + old + "y" * 100_000
    message = await ambiguous_error(shell, tmp_path, block * 2, old, "NEW" * 100_000)
    assert headers(message) == [
        f"lines 1-1 (1:100001-1:{100000 + len(old)})",
        f"lines 1-1 (1:{len(block) + 100001}-1:{len(block) + 100000 + len(old)})",
    ]
    assert all("NEEDLE" in section for section in sections(message))
    assert all(
        "[columns " in text and "; clipped] ... " in text for _, _, text in source_rows(message)
    )
    assert "NEW" not in message
    assert_bounded(message)


async def test_five_maximal_contexts_have_bounded_total_output(shell, tmp_path):
    context = "\U000e0001\t\x00\x1b\\" * 10_000 + "\n"
    old = ("MATCH" + "m" * 10_000 + "\n") * 4
    block = context * 5 + old + context * 5
    message = await ambiguous_error(shell, tmp_path, block * 6, old, "NEW" * 100_000)
    assert len(headers(message)) == 5
    assert len(source_rows(message)) == 70
    assert sum(mark == ">" for mark, _, _ in source_rows(message)) == 20
    assert "Showing 5 of 6 matches; 1 more matches not shown." in message
    assert "NEW" not in message
    for _, _, text in source_rows(message):
        assert re.fullmatch(r"(?:[^\\]|\\(?:U[0-9a-f]{8}|x[0-9a-f]{2}|[\\tn]))*", text)
    assert_bounded(message)


async def test_short_deeply_indented_lines_are_not_clipped(shell, tmp_path):
    old = "hit"
    line = " " * 60 + old + "\n"
    message = await ambiguous_error(shell, tmp_path, line * 2, old)
    assert headers(message) == ["lines 1-1 (1:61-1:63)", "lines 2-2 (2:61-2:63)"]
    assert "clipped" not in message
    assert all(text == " " * 60 + "hit\\n" for _, _, text in source_rows(message))
    assert_bounded(message)


async def test_global_budget_omits_context_not_headers_or_matched_lines(shell, tmp_path):
    # Five-digit line labels and full-width rows exercise the actual global
    # fallback, without monkeypatching bounds or generating huge old/new values.
    context = "c" * 149 + "\n"
    old = ("m" * 149 + "\n") * 5
    content = context * 10_000 + (old + context * 10) * 6
    message = await ambiguous_error(shell, tmp_path, content, old, filename="p" * 240)
    assert len(headers(message)) == 5
    assert sum(mark == ">" for mark, _, _ in source_rows(message)) == 20
    assert message.count("1 matched lines omitted") == 5
    omitted = re.search(
        r"(\d+) context lines omitted \(12000-character diagnostic limit\)", message
    )
    assert omitted is not None
    assert int(omitted[1]) > 0
    assert len(source_rows(message)) + int(omitted[1]) == 70
    assert "Showing 5 of 6 matches; 1 more matches not shown." in message
    assert_bounded(message)


def test_path_is_also_bounded_and_escaped():
    message = _ambiguous_replace_message("\x1b" + "p" * 100_000, "hit hit", "hit", 2)
    assert "\\x1b" in message
    assert "clipped" in message.splitlines()[0]
    assert len(headers(message)) == 2
    assert_bounded(message)


async def test_empty_old_text_explains_ambiguity_without_listing_boundaries(shell, tmp_path):
    message = await ambiguous_error(shell, tmp_path, "x" * 100_000, "")
    assert "matched 100001 times" in message
    assert "No file was changed." in message
    assert "empty" in message
    assert "non-empty old text" in message
    assert not headers(message)
    assert len(message) < 500


@pytest.mark.parametrize(
    ("content", "old", "new", "expected"),
    [("", "", "inserted", "inserted"), ("aaa", "aa", "b", "ba"), ("abc", "b", "", "ac")],
)
async def test_unique_replacement_semantics_unchanged(shell, tmp_path, content, old, new, expected):
    path = tmp_path / "example.py"
    path.write_text(content)
    await shell.replace("example.py", old, new)
    assert path.read_text() == expected


async def test_not_found_diagnostic_and_file_unchanged(shell, tmp_path):
    message = await ambiguous_error(shell, tmp_path, "original", "missing")
    assert "old text not found" in message
    assert "check whitespace and indentation" in message
