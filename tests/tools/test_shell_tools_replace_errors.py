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
        for mark, number, text in re.findall(r"(?m)^([> ]) (\d+)\|(?: (.*))?$", message)
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
    assert "  2| [prefix columns 1-4]     \n" in message  # Preserve usable indentation.
    assert "[suffix columns" not in message
    assert message.count("> [1 matching line omitted; identical to the supplied old text]") == 2
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
    assert [number for _, number, _ in source_rows(sections(message)[0])] == (
        [1, 2, 3, 4] if separator in ["\n", "\r\n", "\r"] else [1, 2, 2, 3, 4]
    )
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
    assert [number for _, number, _ in source_rows(sections(message)[0])] == (
        [1, 2] if newline_only else [2]
    )
    assert len(content.splitlines(keepends=True)) == 2
    assert_bounded(message)


@pytest.mark.parametrize("old", ["\r", "\n"])
def test_raw_partial_crlf_matches_stay_on_preceding_line(old):
    message = _ambiguous_replace_message("example.py", "a\r\na\r\n", old, 2)
    col = 2 if old == "\r" else 3
    assert headers(message) == [f"lines 1-1 (1:{col}-1:{col})", f"lines 2-2 (2:{col}-2:{col})"]
    for line, section in enumerate(sections(message), 1):
        boundaries = [
            (number, text) for _, number, text in source_rows(section) if "columns" in text
        ]
        assert boundaries == [(line, "[prefix columns 1-1] a")]
        assert "\\r" not in section and "\\n" not in section
        assert "> [1 matching line omitted; identical to the supplied old text]" in section
    assert_bounded(message)


@pytest.mark.parametrize("matched_lines", [1, 2, 3, 4, 5, 28, 20_000])
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
        assert [text for _, _, text in rows] == (
            [f"before {n}" for n in range(2, 7)] + [f"after {n}" for n in range(1, 6)]
        )
        assert not any(mark == ">" for mark, _, _ in rows)
        assert not any("match " in text for _, _, text in rows)
        assert "before 1" not in section
        assert "after 6" not in section
        unit = "line" if matched_lines == 1 else "lines"
        assert (
            f"> [{matched_lines} matching {unit} omitted; identical to the supplied old text]"
            in section
        )


@pytest.mark.parametrize("terminal_newline", [False, True])
async def test_borders_have_no_phantom_context_lines(shell, tmp_path, terminal_newline):
    content = "hit\n\nhit" + ("\n" if terminal_newline else "")
    message = await ambiguous_error(shell, tmp_path, content, "hit")
    first, second = sections(message)
    assert [number for _, number, _ in source_rows(first)] == [2, 3]
    assert [number for _, number, _ in source_rows(second)] == [1, 2]
    assert "[suffix columns" not in message
    assert source_rows(first)[0] == (" ", 2, "")
    assert "  2|\n" in message  # Blank rows have no trailing display space.


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
    assert (
        message.count("> [20000 matching lines omitted; identical to the supplied old text]") == 2
    )
    assert [number for _, number, _ in source_rows(sections(message)[0])] == list(
        range(20001, 20006)
    )
    assert [number for _, number, _ in source_rows(sections(message)[1])] == list(
        range(19996, 20001)
    )
    assert not any(mark == ">" for mark, _, _ in source_rows(message))
    assert_bounded(message)


async def test_ambiguous_replace_bounds_heavily_escaped_excerpt(shell, tmp_path):
    content = "\U000e0001" * 100 + "hit" + "\U000e0001" * 100 + "hit"
    message = await ambiguous_error(shell, tmp_path, content, "hit")
    assert headers(message) == ["lines 1-1 (1:101-1:103)", "lines 1-1 (1:204-1:206)"]
    assert "hit" not in message
    assert message.count("matching line omitted") == 2
    for _, _, text in source_rows(message):
        assert "prefix columns" in text or "suffix columns" in text
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
    assert "NEEDLE" not in message
    assert all("[columns " in text and "; clipped]" in text for _, _, text in source_rows(message))
    for section, start in zip(sections(message), [100_000, len(block) + 100_000], strict=True):
        prefix, suffix = source_rows(section)
        prefix_columns = re.search(r"\[columns (\d+)-(\d+); clipped\]", prefix[2])
        suffix_columns = re.search(r"\[columns (\d+)-(\d+); clipped\]", suffix[2])
        assert prefix_columns is not None and suffix_columns is not None
        assert int(prefix_columns[2]) == start
        assert int(suffix_columns[1]) == start + len(old) + 1
        assert int(suffix_columns[2]) > int(suffix_columns[1])
        assert "x" in prefix[2] and "y" in suffix[2]
    assert "NEW" not in message
    assert_bounded(message)


async def test_five_maximal_contexts_have_bounded_total_output(shell, tmp_path):
    context = "\U000e0001\t\x00\x1b\\" * 10_000 + "\n"
    old = ("MATCH" + "m" * 10_000 + "\n") * 4
    block = context * 5 + old + context * 5
    message = await ambiguous_error(shell, tmp_path, block * 6, old, "NEW" * 100_000)
    assert len(headers(message)) == 5
    assert len(source_rows(message)) == 50
    assert not any(mark == ">" for mark, _, _ in source_rows(message))
    assert "Showing 5 of 6 matches; 1 more matches not shown." in message
    assert "NEW" not in message
    assert "MATCH" not in message
    assert message.count("> [4 matching lines omitted; identical to the supplied old text]") == 5
    for _, _, text in source_rows(message):
        assert re.fullmatch(r"(?:[^\\]|\\(?:U[0-9a-f]{8}|x[0-9a-f]{2}|[\\tn]))*", text)
    assert_bounded(message)


async def test_short_deeply_indented_lines_are_not_clipped(shell, tmp_path):
    old = "hit"
    line = " " * 60 + old + "\n"
    message = await ambiguous_error(shell, tmp_path, line * 2, old)
    assert headers(message) == ["lines 1-1 (1:61-1:63)", "lines 2-2 (2:61-2:63)"]
    assert "clipped" not in message
    for section in sections(message):
        texts = [text for _, _, text in source_rows(section)]
        assert sorted(texts) == sorted(
            [
                "[prefix columns 1-60] " + " " * 60,
                " " * 60 + "hit",  # The other occurrence is legitimate context.
            ]
        )
    assert_bounded(message)


async def test_global_budget_keeps_all_context_with_omission_placeholders(shell, tmp_path):
    # Five-digit line labels and full-width rows now fit without a fallback:
    # placeholders replace the old twenty marked source rows.
    context = "c" * 149 + "\n"
    old = ("m" * 149 + "\n") * 5
    content = context * 10_000 + (old + context * 10) * 6
    message = await ambiguous_error(shell, tmp_path, content, old, filename="p" * 240)
    assert len(headers(message)) == 5
    assert not any(mark == ">" for mark, _, _ in source_rows(message))
    assert message.count("> [5 matching lines omitted; identical to the supplied old text]") == 5
    assert "context lines omitted" not in message
    assert len(source_rows(message)) == 50
    assert not any("m" in text for _, _, text in source_rows(message))
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


async def test_same_line_occurrences_keep_distinguishing_unmatched_boundaries(shell, tmp_path):
    message = await ambiguous_error(shell, tmp_path, "left HIT middle HIT right", "HIT")
    assert headers(message) == ["lines 1-1 (1:6-1:8)", "lines 1-1 (1:17-1:19)"]
    first, second = sections(message)
    assert source_rows(first) == [
        (" ", 1, "[prefix columns 1-5] left "),
        (" ", 1, "[suffix columns 9-25]  middle HIT right"),
    ]
    assert source_rows(second) == [
        (" ", 1, "[prefix columns 1-16] left HIT middle "),
        (" ", 1, "[suffix columns 20-25]  right"),
    ]
    # The other occurrence is legitimate context; only the current span is omitted.
    assert first.count("HIT") == second.count("HIT") == 1
    assert message.count("> [1 matching line omitted; identical to the supplied old text]") == 2
    assert_bounded(message)


@pytest.mark.parametrize("terminal_newline", [False, True])
async def test_multiline_partial_boundaries_omit_all_current_text(
    shell, tmp_path, terminal_newline
):
    old = "SECRET first\nSECRET middle\nSECRET last"
    block = "λ prefix " + old + " suffix 🙂"
    content = block + "\n" + "gap\n" * 6 + block + ("\n" if terminal_newline else "")
    message = await ambiguous_error(shell, tmp_path, content, old, "NEW" * 100_000)
    assert headers(message) == ["lines 1-3 (1:10-3:11)", "lines 10-12 (10:10-12:11)"]
    for index, section in enumerate(sections(message)):
        assert "SECRET" not in section
        assert "NEW" not in section
        assert "> [3 matching lines omitted; identical to the supplied old text]" in section
        boundaries = [
            (number, text) for _, number, text in source_rows(section) if "columns" in text
        ]
        assert boundaries == [
            (1 + index * 9, "[prefix columns 1-9] λ prefix "),
            (
                3 + index * 9,
                "[suffix columns 12-20]  suffix 🙂",
            ),
        ]
    assert_bounded(message)


def test_clipped_boundary_columns_identify_exact_escaped_source():
    # Both boundaries are clipped at late original columns, with whole escapes.
    prefix = "p" * 100_000 + "\U000e0001\t\x00\x1b\\" * 100
    suffix = "\U000e0001\t\x00\x1b\\" * 100 + "s" * 100_000
    old = "HIDDEN" + "z" * 200_000
    block = prefix + old + suffix + "\n"
    content = block + "gap\n" * 6 + block
    message = _ambiguous_replace_message("example.py", content, old, 2)
    assert "HIDDEN" not in message
    assert headers(message) == [
        f"lines {line}-{line} ({line}:{len(prefix) + 1}-{line}:{len(prefix) + len(old)})"
        for line in [1, 8]
    ]
    for section in sections(message):
        boundaries = [text for _, _, text in source_rows(section) if "columns" in text]
        assert len(boundaries) == 2
        for kind, text in zip(["prefix", "suffix"], boundaries, strict=True):
            window = re.fullmatch(
                r"\[(prefix|suffix) columns (\d+)-(\d+)\] "
                r"\[columns (\d+)-(\d+); clipped\] (.*)",
                text,
            )
            assert window is not None and window[1] == kind
            start, end = int(window[4]), int(window[5])
            expected = "".join(
                char if char.isprintable() and char != "\\" else repr(char)[1:-1]
                for char in block[start - 1 : end]
            )
            assert window[6] == ("... " if start > int(window[2]) else "") + expected + (
                " ..." if end < int(window[3]) else ""
            )
            if kind == "prefix":
                assert end == len(prefix)
            else:
                assert start == len(prefix) + len(old) + 1
    assert_bounded(message)


def test_maximal_boundaries_context_and_path_stay_within_global_budget():
    # Ten full context lines and both boundary rows per occurrence: at most 60
    # source rows total, even with long line/column labels and escaped path data.
    context = "\U000e0001\t\x00\x1b\\" * 100 + "\n"
    old = "SECRET first\nSECRET last"
    block = context * 5 + "p" * 100_000 + old + "s" * 100_000 + "\n" + context * 5
    content = "padding\n" * 10_000 + block * 6
    message = _ambiguous_replace_message("\x1b" + "p" * 100_000, content, old, 6)
    assert len(headers(message)) == 5
    assert len(source_rows(message)) == 60
    assert "SECRET" not in message
    assert message.count("> [2 matching lines omitted; identical to the supplied old text]") == 5
    assert "Showing 5 of 6 matches; 1 more matches not shown." in message
    assert "context lines omitted" not in message
    assert_bounded(message)


@pytest.mark.parametrize("terminator", ["\n", "\r\n", "\r"])
def test_raw_ordinary_terminators_are_display_only(terminator):
    content = terminator.join([" before  ", "", "hit", " after  ", "hit", ""])
    message = _ambiguous_replace_message("example.py", content, "hit", 2)
    assert headers(message) == ["lines 3-3 (3:1-3:3)", "lines 5-5 (5:1-5:3)"]
    assert source_rows(sections(message)[0]) == [
        (" ", 1, " before  "),
        (" ", 2, ""),
        (" ", 4, " after  "),
        (" ", 5, "hit"),
    ]
    assert "  2|\n" in message
    assert "[suffix columns" not in message
    assert "\\n" not in message and "\\r" not in message
    assert_bounded(message)


@pytest.mark.parametrize("terminator", ["\n", "\r\n", "\r"])
def test_boundary_spaces_literals_and_controls_survive_terminator_trimming(terminator):
    prefix = " \\n\t\x1b\x00 "
    suffix = " \\r\t\x1b\x00  "
    line = prefix + "hit" + suffix
    message = _ambiguous_replace_message("example.py", (line + terminator) * 2, "hit", 2)
    escaped_prefix = r" \\n\t\x1b\x00 "
    escaped_suffix = r" \\r\t\x1b\x00  "
    for section in sections(message):
        boundaries = [text for _, _, text in source_rows(section) if "columns" in text]
        assert boundaries == [
            f"[prefix columns 1-{len(prefix)}] " + escaped_prefix,
            f"[suffix columns {len(prefix) + 4}-{len(line)}] " + escaped_suffix,
        ]
        context = [text for _, _, text in source_rows(section) if "columns" not in text]
        assert context == [escaped_prefix + "hit" + escaped_suffix]
    assert_bounded(message)


@pytest.mark.parametrize(
    "separator", ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_other_line_separators_still_escape_in_context_and_boundaries(separator):
    message = _ambiguous_replace_message("example.py", ("hit" + separator) * 2, "hit", 2)
    escaped = repr(separator)[1:-1]
    assert source_rows(sections(message)[0]) == [
        (" ", 1, "[suffix columns 4-4] " + escaped),
        (" ", 2, "hit" + escaped),
    ]
    assert_bounded(message)


@pytest.mark.parametrize("old", ["\r", "\n", "\r\n"])
def test_terminator_only_prefix_and_suffix_are_omitted(old):
    message = _ambiguous_replace_message("example.py", "\r\n\r\n", old, 2)
    assert source_rows(sections(message)[0]) == [(" ", 2, "")]
    assert source_rows(sections(message)[1]) == [(" ", 1, "")]
    assert "[prefix columns" not in message and "[suffix columns" not in message
    first_col = 2 if old == "\n" else 1
    last_col = 1 if old == "\r" else 2
    assert headers(message) == [
        f"lines {line}-{line} ({line}:{first_col}-{line}:{last_col})" for line in [1, 2]
    ]
    assert_bounded(message)


@pytest.mark.parametrize("old", ["a", "a\r"])
def test_partial_crlf_suffix_is_omitted_after_visible_match(old):
    message = _ambiguous_replace_message("example.py", "a\r\na\r\n", old, 2)
    assert headers(message) == [
        f"lines {line}-{line} ({line}:1-{line}:{len(old)})" for line in [1, 2]
    ]
    assert source_rows(sections(message)[0]) == [(" ", 2, "a")]
    assert source_rows(sections(message)[1]) == [(" ", 1, "a")]
    assert_bounded(message)


@pytest.mark.parametrize("prefix", ["p" * 100_000, "\x1b" * 100_000])
def test_long_prefix_focus_is_clamped_before_partial_crlf(prefix):
    content = (prefix + "\r\n") * 2
    message = _ambiguous_replace_message("example.py", content, "\n", 2)
    assert headers(message) == [
        f"lines {line}-{line} ({line}:{len(prefix) + 2}-{line}:{len(prefix) + 2})"
        for line in [1, 2]
    ]
    for section in sections(message):
        boundary = [text for _, _, text in source_rows(section) if "prefix columns" in text]
        assert len(boundary) == 1
        window = re.fullmatch(
            r"\[prefix columns 1-(\d+)\] \[columns (\d+)-(\d+); clipped\] \.\.\. (.*)",
            boundary[0],
        )
        assert window is not None
        assert int(window[1]) == int(window[3]) == len(prefix)
        assert 1 < int(window[2]) <= int(window[3])
        expected = "".join(repr(char)[1:-1] for char in prefix[int(window[2]) - 1 :])
        assert window[4] == expected
    assert "\\r" not in message and "\\n" not in message
    assert_bounded(message)


@pytest.mark.parametrize("terminator", ["\n", "\r\n", "\r"])
def test_ordinary_terminators_in_paths_are_still_escaped(terminator):
    path = "example" + terminator + ".py" + terminator
    message = _ambiguous_replace_message(path, "hit hit", "hit", 2)
    escaped = "".join(repr(char)[1:-1] for char in terminator)
    assert message.splitlines()[0] == (
        f"old text matched 2 times in example{escaped}.py{escaped}. No file was changed."
    )
    assert_bounded(message)


# --- zero-match diagnostics -------------------------------------------------


async def not_found_error(
    shell, tmp_path, content, old, new="replacement text", filename="example.py"
):
    path = tmp_path / filename
    path.write_text(content)
    before = path.read_bytes()
    with pytest.raises(ValueError) as error:
        await shell.replace(filename, old, new)
    assert path.read_bytes() == before  # Never write on zero matches.
    return str(error.value)


async def test_not_found_reports_resolved_path_and_first_line(shell, tmp_path):
    message = await not_found_error(shell, tmp_path, "alpha\nbeta\n", "gamma\ndelta")

    assert f"(resolved path: {tmp_path / 'example.py'})" in message
    assert "No file was changed." in message
    assert "First line of old text: 'gamma' (no indentation)" in message
    assert "does not occur as a whole line" in message
    assert "re-read it with read()" in message
    assert "same cell" in message


async def test_not_found_reports_tab_versus_space_indentation(shell, tmp_path):
    content = "def f():\n\tif x:\n\t\treturn 1\n"
    message = await not_found_error(shell, tmp_path, content, "    if x:\n        return 1")

    assert "First line of old text: '    if x:' (4 space(s))" in message
    assert "line 2 has it with 1 tab(s): '\\tif x:'" in message
    assert "The file uses tabs; old text uses none." in message


async def test_not_found_reports_different_space_indentation(shell, tmp_path):
    content = "class A:\n    def f(self):\n        return 1\n"
    message = await not_found_error(shell, tmp_path, content, "def f(self):\n    return 1")

    assert "line 2 has it with 4 space(s): '    def f(self):'" in message


async def test_not_found_when_first_line_matches_points_to_later_line(shell, tmp_path):
    content = "if x:\n    return 1\n"
    message = await not_found_error(shell, tmp_path, content, "if x:\n    return 2")

    assert "occurs unchanged at line(s) 1; the difference is in a later line" in message


async def test_not_found_flags_possibly_already_applied_edit(shell, tmp_path):
    content = "value = compute_new()\n"
    message = await not_found_error(
        shell, tmp_path, content, "value = compute_old()", "value = compute_new()"
    )

    assert "already appears in the file (starting on line 1)" in message
    assert "may already have been applied" in message


async def test_not_found_ignores_short_replacement_coincidences(shell, tmp_path):
    message = await not_found_error(shell, tmp_path, "x = 1\n", "y = 2", "x")

    assert "already appears" not in message


async def test_not_found_is_bounded_and_escaped(shell, tmp_path):
    first = "\x1b" + "y" * 5_000
    content = "".join(f"{' ' * n}{first}\n" for n in range(1, 50))
    message = await not_found_error(shell, tmp_path, content, first + "\nmissing")

    assert "\x1b" not in message
    assert message.count(" has it with ") <= 3
    assert len(message) < 2_000
    assert all(len(line) <= 400 for line in message.splitlines())


# --- replace() call-form errors ----------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"old": "a", "new_text": "b"}, {"old_text": "a"}, {"new_text": "b"}],
)
async def test_misnamed_keywords_name_the_exact_path_call(shell, tmp_path, kwargs):
    path = tmp_path / "example.py"
    path.write_text("a\n")
    with pytest.raises(TypeError) as error:
        await shell.replace("example.py", **kwargs)
    assert path.read_text() == "a\n"
    message = str(error.value)
    assert "no file was changed" in message
    for name in kwargs:
        assert f"{name}=" in message
    assert "replace(path, old_text, new_text)" in message
    assert "parameters are target, old_or_new, new" in message


async def test_misnamed_keyword_with_match_names_the_match_call(shell, tmp_path):
    path = tmp_path / "example.py"
    path.write_text("a\n")
    match = await shell.read("example.py")
    with pytest.raises(TypeError) as error:
        await shell.replace(match, new_text="b\n")
    assert path.read_text() == "a\n"
    assert "replace(match, new_text)  #" in str(error.value)


async def test_match_with_new_keyword_points_to_path_form(shell, tmp_path):
    path = tmp_path / "example.py"
    path.write_text("a\n")
    match = await shell.read("example.py")
    with pytest.raises(ValueError) as error:
        await shell.replace(match, "a", new="b")
    assert path.read_text() == "a\n"
    assert "replace(match, new_text)" in str(error.value)
    assert "replace(match.resolved_path, old_text, new_text)" in str(error.value)


async def test_path_without_new_names_both_forms(shell, tmp_path):
    (tmp_path / "example.py").write_text("a\n")
    with pytest.raises(ValueError) as error:
        await shell.replace("example.py", "a")
    assert "replace(path, old_text, new_text)" in str(error.value)
    assert "replace(match, new_text)" in str(error.value)


def test_replace_docstring_matches_real_parameters():
    import inspect

    doc = inspect.getdoc(ShellTools.replace)
    params = list(inspect.signature(ShellTools.replace).parameters)
    assert params[:4] == ["self", "target", "old_or_new", "new"]
    for name in ("target", "old_or_new", "new"):
        assert f"{name}:" in doc
    # The old placeholders must not appear as if they were parameter names.
    assert "file_path" not in doc
    assert "(target, old_or_new, new)" in doc
    assert "replace(path, old_text, new_text)" in doc
    assert "replace(match, new_text)" in doc


async def test_replace_docstring_examples_execute(shell, tmp_path):
    (tmp_path / "pkg").mkdir()
    target = tmp_path / "pkg" / "mod.py"
    target.write_text(
        "".join(f"line {n}\n" for n in range(1, 13)).replace("line 1\n", "return 1\n", 1)
    )
    await shell.replace("pkg/mod.py", "return 1", "return 2")
    m = await shell.read("pkg/mod.py", (10, 12))
    await shell.replace(m, "new lines 10-12\n")
    assert target.read_text().startswith("return 2\n")
    assert target.read_text().endswith("line 9\nnew lines 10-12\n")
