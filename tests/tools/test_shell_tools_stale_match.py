# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stale Match edits fail closed at their original range, with bounded diagnostics."""

import shutil

import pytest

from nooa.tools.shell_tools import Match, ShellTools, StaleMatchError


@pytest.fixture
async def shell(tmp_path):
    tool = ShellTools(cwd=str(tmp_path))
    yield tool
    await tool.close()


@pytest.mark.parametrize(
    "current",
    [
        b"first\nmodified\nthird\n",
        b"inserted\nfirst\ncaptured\nthird\n",  # captured text still exists: no relocation
        b"first\nthird\n",  # deletion
        b"first\n",  # shortening past the captured range
        b"",  # missing region
        b"first\ncaptured",  # only EOF newline differs
    ],
)
async def test_stale_region_never_writes(shell, tmp_path, current):
    path = tmp_path / "file.txt"
    path.write_bytes(b"first\ncaptured\nthird\n")
    anchor = await shell.read("file.txt", (2, 2))
    path.write_bytes(current)
    with pytest.raises(StaleMatchError) as caught:
        await shell.replace(anchor, "replacement")
    assert path.read_bytes() == current
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.code == "STALE_MATCH"
    assert error.path == "file.txt"
    assert error.resolved_path == str(path.resolve())
    assert (error.start, error.end) == (2, 2)
    for text in [
        "file.txt",
        str(path.resolve()),
        "captured lines 2-2",
        "No file was changed",
        "Re-read or re-search",
        "recompute the Match",
        "never relocated",
    ]:
        assert text in str(error)


async def test_missing_file_never_created(shell, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("captured\n")
    anchor = await shell.read("file.txt")
    path.unlink()
    with pytest.raises(StaleMatchError, match="file no longer exists") as caught:
        await shell.replace(anchor, "replacement")
    assert not path.exists()
    assert caught.value.diff == ""
    assert not caught.value.diff_complete
    assert "unavailable" in str(caught.value)
    assert "Diff preview truncated" not in str(caught.value)


@pytest.mark.parametrize(
    "current", ["new first\ncaptured\nthird\n", "first\ncaptured\nnew last\nextra\n"]
)
async def test_unrelated_region_changes_allowed(shell, tmp_path, current):
    path = tmp_path / "file.txt"
    path.write_text("first\ncaptured\nthird\n")
    anchor = await shell.read("file.txt", (2, 2))
    path.write_text(current)
    result = await shell.replace(anchor, "replacement")
    assert path.read_text() == current.replace("captured\n", "replacement\n")
    assert result.new_text == "replacement\n"


@pytest.mark.parametrize("producer", ["whole", "window", "slice", "search"])
@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r", b""])
async def test_valid_producers_and_line_endings(shell, tmp_path, producer, ending):
    if producer == "search" and (shutil.which("rg") is None or shutil.which("grep") is None):
        pytest.skip("needs rg and grep on PATH")
    path = tmp_path / "file.txt"
    # Search engines count LF lines; keep bare CR coverage to single-line files.
    original = b"captured" + ending
    path.write_bytes(original)
    if producer == "whole":
        anchor = await shell.read("file.txt")
    elif producer == "window":
        anchor = await shell.read("file.txt", (1, 1))
    elif producer == "slice":
        anchor = (await shell.read("file.txt"))[1:1]
    else:
        result = await shell.run("grep -n captured file.txt")
        assert result.matches
        anchor = result.matches[0]
    assert anchor.text == "captured" + ("\n" if ending else "")
    result = await shell.replace(anchor, "replacement")
    expected = b"replacement" + (b"\n" if ending else b"")
    assert path.read_bytes() == expected  # existing write normalization is unchanged
    assert result.new_text.encode() == expected


@pytest.mark.parametrize(
    "start,end", [(0, 1), (-1, 1), (2, 1), (1, 5), (5, 5), (True, 1), (1.0, 1)]
)
async def test_invalid_or_out_of_bounds_range_never_writes(shell, tmp_path, start, end):
    path = tmp_path / "file.txt"
    path.write_bytes(b"captured\n")
    anchor = Match("file.txt", start, end, "captured\n", resolved_path=path)
    with pytest.raises(StaleMatchError):
        await shell.replace(anchor, "replacement")
    assert path.read_bytes() == b"captured\n"


async def test_invalid_read_and_slice_ranges_never_write(shell, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("captured\n")
    whole = await shell.read("file.txt")
    for anchor in [await shell.read("file.txt", (5, 8)), whole[5:8], whole[1:0]]:
        with pytest.raises(StaleMatchError):
            await shell.replace(anchor, "replacement")
        assert path.read_bytes() == b"captured\n"


async def test_empty_file_anchor_remains_valid_but_detects_added_content(shell, tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"")
    anchor = await shell.read("file.txt")
    assert (anchor.start, anchor.end, anchor.text) == (1, 0, "")
    await shell.replace(anchor, "replacement")
    assert path.read_bytes() == b"replacement"
    with pytest.raises(StaleMatchError):
        await shell.replace(anchor, "overwrite")
    assert path.read_bytes() == b"replacement"


async def test_small_diff_is_complete(shell, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("old\n")
    anchor = await shell.read("file.txt")
    path.write_text("new\n")
    with pytest.raises(StaleMatchError) as caught:
        await shell.replace(anchor, "replacement")
    error = caught.value
    assert error.diff == (
        "--- stored Match.text (when read)\n"
        "+++ current file (same saved line range)\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    assert error.diff_complete
    assert str(error).endswith(error.diff)
    assert "Diff preview truncated" not in str(error)
    assert (
        "stored Match.text (when read) versus current file (same saved line range) diff (complete):"
    ) in str(error)
    assert error.args == (str(error),)  # no raw full regions in exception rendering


@pytest.mark.parametrize(
    "expected,current,detail",
    [
        ("old\n" * 30, "new\n" * 30, "capped at 20 lines and 2 KiB"),
        ("α" * 2000 + "\n", "β" * 2000 + "\n", "capped at 20 lines and 2 KiB"),
        ("x" * 20_000, "y" * 20_000, "exceed 16 KiB"),
        ("α" * 5000, "β" * 5000, "exceed 16 KiB"),
        ("x\n" * 250, "y\n" * 250, "exceed 400 lines"),
    ],
    ids=["line-cap", "byte-cap", "huge-line", "utf8-input-cap", "line-input-cap"],
)
async def test_large_diff_bounded_or_omitted(
    shell, tmp_path, monkeypatch, expected, current, detail
):
    path = tmp_path / "file.txt"
    path.write_text(expected)
    anchor = await shell.read("file.txt")
    path.write_text(current)
    if "exceed" in detail:

        def forbidden_diff(*args, **kwargs):
            pytest.fail("oversized regions must not enter diff computation")

        monkeypatch.setattr("nooa.tools.shell_tools.difflib.unified_diff", forbidden_diff)
    with pytest.raises(StaleMatchError) as caught:
        await shell.replace(anchor, "replacement")
    error = caught.value
    assert not error.diff_complete
    assert len(error.diff.splitlines()) <= 20
    assert len(error.diff.encode("utf-8")) <= 2048
    assert detail in str(error)
    assert "incomplete" in str(error)
    assert expected not in str(error)
    assert current not in str(error)
    assert path.read_bytes() == current.encode()
    footer = "Diff preview truncated (capped at 20 lines and 2 KiB); not a complete diff."
    if "exceed" in detail:
        assert error.diff == ""
        assert "Diff preview truncated" not in str(error)
        assert "--- stored Match.text" not in str(error)
    else:
        assert error.diff
        assert "Diff preview (incomplete; capped at 20 lines and 2 KiB):\n" in str(error)
        separator = "" if error.diff.endswith("\n") else "\n"
        assert str(error).endswith(error.diff + separator + footer)
        assert footer not in error.diff
        if expected.startswith("α"):
            # The byte cap cuts a UTF-8 data line before its newline.
            assert not error.diff.endswith("\n")
            assert len(error.diff.encode("utf-8")) >= 2047
        else:
            assert len(error.diff.splitlines()) == 20


async def test_no_newline_diff_explains_eof(shell, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("old")
    anchor = await shell.read("file.txt")
    path.write_text("new")
    with pytest.raises(StaleMatchError) as caught:
        await shell.replace(anchor, "replacement")
    assert caught.value.diff.count("No newline at end of file") == 2
    assert caught.value.diff_complete


@pytest.mark.parametrize("producer", ["window", "slice", "search"])
async def test_stale_producer_match_never_writes(shell, tmp_path, producer):
    if producer == "search" and (shutil.which("rg") is None or shutil.which("grep") is None):
        pytest.skip("needs rg and grep on PATH")
    path = tmp_path / "file.txt"
    path.write_text("first\ncaptured\nthird\n")
    if producer == "window":
        anchor = await shell.read("file.txt", (2, 3))
    elif producer == "slice":
        anchor = (await shell.read("file.txt"))[2:3]
    else:
        result = await shell.run("grep -n captured file.txt")
        assert result.matches
        anchor = result.matches[0]
    path.write_bytes(b"first\nexternally changed\n")
    before = path.read_bytes()
    with pytest.raises(StaleMatchError):
        await shell.replace(anchor, "replacement")
    assert path.read_bytes() == before


async def test_whole_file_match_detects_appended_unterminated_line(shell, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("captured")
    anchor = await shell.read("file.txt")
    path.write_text("captured more text")
    with pytest.raises(StaleMatchError):
        await shell.replace(anchor, "replacement")
    assert path.read_bytes() == b"captured more text"
