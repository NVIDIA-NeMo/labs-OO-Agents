# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actionable, bounded diagnostics without changing exact replacement semantics."""

import pytest

from nooa.tools.shell_tools import ShellTools


@pytest.fixture
async def shell(tmp_path):
    tool = ShellTools(cwd=str(tmp_path))
    yield tool
    await tool.close()


async def ambiguous_error(shell, tmp_path, content, old, new="replacement"):
    path = tmp_path / "example.py"
    path.write_text(content)
    before = path.read_bytes()
    with pytest.raises(ValueError) as error:
        await shell.replace("example.py", old, new)
    assert path.read_bytes() == before
    return str(error.value)


async def test_ambiguous_replace_shows_distinct_context_and_recovery(shell, tmp_path):
    content = "def first():\n    return value\n\ndef second():\n    return value\n"
    message = await ambiguous_error(shell, tmp_path, content, "return value")

    assert message.startswith("old text matched 2 times in example.py.")
    assert "No file was changed." in message
    assert "line:column, 1-based" in message
    assert "2:5" in message
    assert "5:5" in message
    assert "def first():" in message
    assert "def second():" in message
    assert "add surrounding context" in message
    assert "read()" in message

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
        ("x = 1; x = 1", "x = 1", ["1:1", "1:8"]),
        ("a\nb\na\nb\n", "a\nb\n", ["1:1", "3:1"]),
        ("λ 🙂 value\nβ value", "value", ["1:5", "2:3"]),
        ("aaaaa", "aa", ["1:1", "1:3"]),
        ("value\nvalue", "value", ["1:1", "2:1"]),
    ],
)
async def test_ambiguous_replace_reports_nonoverlapping_character_positions(
    shell, tmp_path, content, old, positions
):
    message = await ambiguous_error(shell, tmp_path, content, old)
    assert "matched 2 times" in message
    starts = [line.split()[0] for line in message.splitlines() if line.startswith("  ")]
    assert starts == positions


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"],
)
async def test_ambiguous_replace_line_numbers_agree_with_read(shell, tmp_path, separator):
    message = await ambiguous_error(shell, tmp_path, f"λ hit{separator}β hit", "hit")
    assert "1:3" in message
    assert "2:3" in message
    second_line = await shell.read("example.py", (2, 2))
    assert second_line.text == "β hit"


async def test_ambiguous_replace_bounds_many_matches_and_escapes_context(shell, tmp_path):
    content = "\t\x1b[31mneedle\x00 " * 1_000
    message = await ambiguous_error(shell, tmp_path, content, "needle", "NEW" * 10_000)
    assert "matched 1000 times" in message
    assert len([line for line in message.splitlines() if line.startswith("  ")]) == 5
    assert "995 more matches" in message
    assert "\\t" in message
    assert "\\x1b" in message
    assert "\x1b" not in message
    assert "\x00" not in message
    assert "NEW" not in message
    assert len(message) < 1_500


async def test_ambiguous_replace_bounds_large_multiline_old_text(shell, tmp_path):
    old = "match line\n" * 20_000
    message = await ambiguous_error(shell, tmp_path, old + old, old)
    assert "matched 2 times" in message
    assert "1:1" in message
    assert "20001:1" in message
    assert len(message) < 1_000


async def test_ambiguous_replace_bounds_heavily_escaped_excerpt(shell, tmp_path):
    content = "\U000e0001" * 100 + "hit" + "\U000e0001" * 100 + "hit"
    message = await ambiguous_error(shell, tmp_path, content, "hit")
    assert len(message) < 1_000
    assert "1:101" in message
    assert "1:204" in message
    # Escaped preceding context must not crowd out the actual match or leave
    # a partial escape sequence in the excerpt.
    assert message.count("hit") == 2
    import ast

    for line in message.splitlines():
        if line.startswith("  "):
            excerpt = line.split("  ", 2)[-1].removeprefix("... ").removesuffix(" ...")
            assert "hit" in ast.literal_eval(excerpt)


async def test_empty_old_text_explains_ambiguity_without_listing_boundaries(shell, tmp_path):
    message = await ambiguous_error(shell, tmp_path, "x" * 100_000, "")
    assert "matched 100001 times" in message
    assert "No file was changed." in message
    assert "empty" in message
    assert "non-empty old text" in message
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
