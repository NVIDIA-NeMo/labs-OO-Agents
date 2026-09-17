# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core behavior of the DEFAULT ShellTools (run / read / write_file / replace).

This keeps the *default* ShellTools — the one agents actually use — covered for
its primary file/run surface. Search-anchor behavior is covered separately in
test_shell_tools_modern.py.
"""

import pytest

from nooa.tools.shell_tools import Match, ShellResult, ShellTools


def test_shell_tools_directs_agents_to_its_file_and_command_methods():
    doc = ShellTools.__doc__
    assert doc is not None
    assert "Always use these four methods rather than Python builtins" in doc
    assert "For shell commands and file operations" in doc


def test_run_stream_documents_standalone_usage():
    doc = ShellTools.run_stream.__doc__
    assert doc is not None
    assert "pyp" not in doc
    assert "async for event in self.shell.run_stream(" in doc
    assert "event.returncode" in doc
    assert "event.timed_out" in doc


def test_run_and_run_stream_have_matching_arguments():
    import inspect

    def arguments(method):
        return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()]

    assert arguments(ShellTools.run_stream) == arguments(ShellTools.run)


@pytest.mark.parametrize("payload", ["", "hello", "one\ntwo\n", "'\" $HOME $(echo nope) `pwd` λ\n"])
async def test_run_stream_accepts_stdin_verbatim_and_keeps_exit_status(tmp_path, payload):
    shell = ShellTools(cwd=str(tmp_path))
    try:
        buffered = await shell.run(
            "cat; printf 'problem\\n' >&2; (exit 7)", stdin=payload, timeout=5.0
        )
        events = [
            event
            async for event in shell.run_stream(
                "cat; printf 'problem\\n' >&2; (exit 7)", stdin=payload, timeout=5.0
            )
        ]
        assert "".join(event.text for event in events if event.kind == "stdout") == payload
        assert "".join(event.text for event in events if event.kind == "stderr") == "problem\n"
        assert events[-1].kind == "done"
        assert events[-1].returncode == 7
        # Buffered run has always stripped trailing newlines; streaming does not.
        assert buffered.stdout == payload.rstrip("\n")
        assert buffered.stderr == "problem"
        assert buffered.returncode == events[-1].returncode
        assert not events[-1].timed_out
        assert sum(event.kind == "done" for event in events) == 1
        assert (await shell.run("printf ready")).stdout == "ready"
    finally:
        await shell.close()


@pytest.fixture
def sh(tmp_path):
    return ShellTools(cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_run_persists_state(sh, tmp_path):
    r = await sh.run("echo hello")
    assert r.success
    assert "hello" in r.stdout
    # cd persists across calls in the same session.
    (tmp_path / "sub").mkdir()
    await sh.run("cd sub")
    r2 = await sh.run("pwd")
    assert r2.stdout.strip().endswith("sub")


@pytest.mark.asyncio
async def test_run_reports_failure(sh):
    r = await sh.run("false")
    assert not r.success
    assert r.returncode != 0
    assert r.timed_out is False


def test_match_requires_resolved_path():
    with pytest.raises(TypeError, match="resolved_path"):
        Match("example.py", 1, 1, "value\n")  # type: ignore[call-arg]


def test_shell_result_timeout_flag_preserves_positional_matches_argument():
    match = Match("example.py", 1, 1, "value\n", resolved_path="/tmp/example.py")
    result = ShellResult("value", "", 0, [match], timed_out=True)

    assert result.matches == [match]
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_write_file_then_read(sh, tmp_path):
    await sh.write_file("f.txt", "line1\nline2\nline3\n")
    assert (tmp_path / "f.txt").read_text() == "line1\nline2\nline3\n"
    # read with a numbered gutter (default) -> Match; inspect via .numbered/.text.
    view = await sh.read("f.txt")
    assert "line2" in view.numbered
    # read a line window -> Match for just that line.
    window = await sh.read("f.txt", (2, 2))
    assert "line2" in window.text
    assert "line1" not in window.text


@pytest.mark.asyncio
async def test_replace_path_unique(sh, tmp_path):
    await sh.write_file("f.py", "x = 1\ny = 2\nz = 3\n")
    await sh.replace("f.py", "y = 2", "y = 22")
    assert (tmp_path / "f.py").read_text() == "x = 1\ny = 22\nz = 3\n"


@pytest.mark.asyncio
async def test_replace_path_ambiguous_errors(sh, tmp_path):
    await sh.write_file("f.py", "a = 1\na = 1\n")
    with pytest.raises(ValueError, match="matched 2 times"):
        # Two matches -> must error rather than guess.
        await sh.replace("f.py", "a = 1", "a = 2")


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [None, (2, 2)])
@pytest.mark.parametrize("replacement", ["return a + b", ""])
@pytest.mark.parametrize("keyword", [False, True])
async def test_replace_match_rejects_old_new_without_modifying_file(
    sh, tmp_path, lines, replacement, keyword
):
    """Both whole-file and sliced matches reject the path-form argument pattern."""
    original = "def calc(a, b):\n    return a * b\n"
    path = tmp_path / "calc.py"
    path.write_text(original)
    match = await sh.read("calc.py", lines)
    with pytest.raises(ValueError) as error:
        if keyword:
            await sh.replace(match, "return a * b", new=replacement)
        else:
            await sh.replace(match, "return a * b", replacement)
    assert path.read_bytes() == original.encode()
    assert "replace(match, new_text)" in str(error.value)
    assert "replace(path, old, new)" in str(error.value)
    assert "no file was changed" in str(error.value)


@pytest.mark.asyncio
async def test_replace_match_argument_guard_runs_before_file_access(sh, tmp_path):
    """An invalid call is rejected even when its old Match points to a missing file."""
    match = Match("missing.py", 1, 1, "old", resolved_path=tmp_path / "missing.py")
    with pytest.raises(ValueError, match="ambiguous"):
        await sh.replace(match, "old", "new")
    assert not (tmp_path / "missing.py").exists()


@pytest.mark.asyncio
async def test_write_file_is_overwrite(sh, tmp_path):
    await sh.write_file("f.txt", "old")
    await sh.write_file("f.txt", "new")
    assert (tmp_path / "f.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_file_operations_allow_paths_outside_cwd(sh, tmp_path):
    sibling = tmp_path.parent / f"{tmp_path.name}-sibling"
    sibling.mkdir()
    relative = f"../{sibling.name}/relative.txt"
    absolute = sibling / "absolute.txt"

    await sh.write_file(relative, "one\ntwo\n")
    assert (await sh.read(relative)).text == "one\ntwo\n"

    await sh.replace(relative, "one", "changed")
    await sh.replace(
        Match(relative, 2, 2, "two\n", resolved_path=sibling / "relative.txt"), "replaced"
    )
    # The region reaches EOF in a file that ended with a newline, so the
    # replacement is re-terminated rather than stripping the final byte.
    assert (sibling / "relative.txt").read_text() == "changed\nreplaced\n"

    await sh.write_file(str(absolute), "absolute")
    assert (await sh.read(str(absolute))).text == "absolute"


@pytest.mark.asyncio
async def test_match_from_read_stays_bound_after_cwd_change(sh, tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("before\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "original.txt").write_text("wrong file\n")

    match = await sh.read("original.txt")
    sliced = match[1:1]
    await sh.run("cd other")
    await sh.replace(sliced, "after")

    # Whole-file region at EOF in a newline-terminated file keeps its final
    # newline instead of silently stripping it.
    assert original.read_text() == "after\n"
    assert (other / "original.txt").read_text() == "wrong file\n"


@pytest.mark.asyncio
async def test_close_terminates_underlying_bash_session(sh):
    """Verify close() terminates BashSession and the shell lazily restarts."""
    r = await sh.run("echo started")
    assert r.success
    assert sh._session._process is not None

    await sh.close()

    assert sh._session._process is None
    assert not sh._session._started

    # The shell remains reusable after close(); a fresh session starts lazily.
    r2 = await sh.run("echo restarted")
    assert r2.success
    assert "restarted" in r2.stdout
    await sh.close()


def test_match_rejects_a_relative_resolved_path(tmp_path, monkeypatch):
    """The anchor must be absolute, or it silently binds to the process cwd.

    Path.resolve() on a relative path resolves against os.getcwd(), which is
    not the shell cwd — so a caller passing a relative path would produce a
    Match pointing at a different file, with no error. Every caller passes an
    absolute path today; this keeps that a rule rather than a convention.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        Match("f.txt", 1, 1, "hello\n", resolved_path="f.txt")


def test_match_keeps_an_absolute_resolved_path(tmp_path):
    """The supported form is unaffected."""
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    match = Match("f.txt", 1, 1, "hello\n", resolved_path=target)
    assert match.resolved_path == str(target.resolve())


@pytest.mark.asyncio
async def test_replace_match_at_eof_keeps_trailing_newline(sh, tmp_path):
    """A replacement reaching end-of-file must not strip the file's final newline."""
    await sh.write_file("f.py", "a = 1\nb = 2\n")
    match = await sh.read("f.py", (2, 2))
    await sh.replace(match, "b = 20")  # no trailing newline in the replacement
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 20\n"


@pytest.mark.asyncio
async def test_replace_match_at_eof_preserves_missing_newline(sh, tmp_path):
    """A file that genuinely lacks a final newline must not gain one."""
    await sh.write_file("f.py", "a = 1\nb = 2")  # no trailing newline
    match = await sh.read("f.py", (2, 2))
    await sh.replace(match, "b = 20")
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 20"
