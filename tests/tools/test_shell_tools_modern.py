# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ShellTools transparent grep->Match.

Strategy:
- Gate truth-table: is_pure_search_command() accepts pure searches, rejects
  anything that mangles output or drops anchors (the -rnP cluster bug is the
  regression test).
- Differential oracle: for a pure search, the Match anchors MUST equal the
  (path, line) set the agent's own grep printed; on any divergence -> .matches
  is None (fail-closed).
- Property/fuzz: random corpus x random safe literal pattern, invariant holds.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import string
import subprocess
from pathlib import Path

import pytest

from nooa.tools.shell_tools import (
    ShellTools,
    is_pure_search_command,
)

pytestmark = pytest.mark.skipif(
    shutil.which("rg") is None or shutil.which("grep") is None,
    reason="needs rg and grep on PATH",
)


# --------------------------------------------------------------------------
# Gate truth table
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("grep -rn 'foo' .", True),
        ("grep -rni 'foo' .", True),
        ("rg -n 'def foo' src/", True),
        ("egrep -rn 'a|b' .", True),
        # mangling pipes
        ("grep -rn 'foo' . | head -20", True),
        ("rg -n 'foo' . | sed 's/x/y/'", False),
        ("grep -rn 'foo' . | awk '{print $1}'", False),
        # anchor-dropping flags
        ("grep -rno 'foo' .", False),
        ("grep -rnA2 'foo' .", False),
        ("grep -c 'foo' .", False),
        ("rg -l 'foo' .", False),
        # the regression: -P bundled in a cluster (PCRE semantics differ)
        ("grep -rnP '(?<=x)foo' .", False),
        ("grep --pcre2 -n 'foo' .", False),
        # not a search
        ("cat foo.py", False),
        ("ls -la", False),
    ],
)
def test_gate(cmd, expected):
    assert is_pure_search_command(cmd) is expected


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text('def foo():\n    return "bar"\n# foo again\nfooo = 1\n')
    (tmp_path / "b.txt").write_text("no match here\nFOO upper\n  foo indented\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("x = 1\nfoo = 2\n")
    return tmp_path


def _grep_anchor_lines(repo: Path, cmd: str) -> set[tuple[str, int]]:
    """Ground truth: (relpath, line) the agent's own grep -n reports."""
    res = subprocess.run(cmd, cwd=repo, shell=True, capture_output=True, text=True)
    out = set()
    for ln in res.stdout.splitlines():
        parts = ln.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            p = parts[0]
            if p.startswith("./"):
                p = p[2:]
            out.add((p, int(parts[1])))
    return out


# --------------------------------------------------------------------------
# Read-only sed ranges
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd,start,end,expected",
    [
        ("sed -n '2,3p' a.py", 2, 3, '    return "bar"\n# foo again\n'),
        ('sed -n "2p" a.py', 2, 2, '    return "bar"\n'),
        ("sed -n -e '3,99p' a.py", 3, 4, "# foo again\nfooo = 1\n"),
    ],
)
@pytest.mark.asyncio
async def test_sed_numeric_range_attaches_editable_match(
    repo: Path, cmd: str, start: int, end: int, expected: str
):
    sh = ShellTools(cwd=str(repo))
    result = await sh.run(cmd)

    assert result.matches is not None and len(result.matches) == 1
    match = result.matches[0]
    assert (match.path, match.start, match.end, match.text) == ("a.py", start, end, expected)
    assert match.text.strip() == result.stdout


@pytest.mark.asyncio
async def test_sed_range_after_cd_prefix_attaches_match(repo: Path):
    sh = ShellTools(cwd=str(repo))
    result = await sh.run("cd sub && sed -n '1,2p' c.py")

    assert result.matches is not None and len(result.matches) == 1
    match = result.matches[0]
    assert (match.path, match.start, match.end, match.text) == ("c.py", 1, 2, "x = 1\nfoo = 2\n")


@pytest.mark.asyncio
async def test_sed_range_match_is_editable(repo: Path):
    sh = ShellTools(cwd=str(repo))
    result = await sh.run("sed -n '2,3p' a.py")
    assert result.matches

    await sh.replace(result.matches[0], "    return 42\n")

    assert (repo / "a.py").read_text() == "def foo():\n    return 42\nfooo = 1\n"


@pytest.mark.asyncio
async def test_sed_range_past_eof_attaches_empty_match_list(repo: Path):
    sh = ShellTools(cwd=str(repo))
    result = await sh.run("sed -n '99,100p' a.py")

    assert result.stdout == ""
    assert result.matches == []


@pytest.mark.parametrize(
    "cmd",
    [
        "sed -n 's/foo/bar/p' a.py",  # transformation
        "sed -n '/foo/p' a.py",  # regex address
        "sed -n '1p;2p' a.py",  # multiple expressions
        "sed -n -e '1p' -e '2p' a.py",  # multiple -e expressions
        "sed -n '1,2p' a.py b.txt",  # multiple files
        "sed -n '1,2p'",  # stdin
        "sed -n '1,2p' a.py | cat",  # pipe
        "sed -i -n '1,2p' a.py",  # mutation
        "sed -n '3,2p' a.py",  # reversed range
    ],
)
@pytest.mark.asyncio
async def test_nontrivial_sed_does_not_attach_matches(repo: Path, cmd: str):
    sh = ShellTools(cwd=str(repo))
    result = await sh.run(cmd)

    assert result.matches is None


# --------------------------------------------------------------------------
# Differential oracle on real searches
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "grep -rn 'foo' .",
        "grep -rni 'foo' .",
        "rg -n 'foo' .",
        "grep -rn 'def ' .",
        "grep -rn 'nomatch_zzz' .",
    ],
)
@pytest.mark.asyncio
async def test_matches_equal_grep(repo: Path, cmd: str):
    sh = ShellTools(cwd=str(repo))
    r = await sh.run(cmd)
    truth = _grep_anchor_lines(repo, cmd)
    assert r.matches is not None, "pure search should attach matches"
    got = {(m.path, m.start) for m in r.matches}
    assert got == truth


@pytest.mark.asyncio
async def test_match_anchor_is_editable(repo: Path):
    sh = ShellTools(cwd=str(repo))
    r = await sh.run("grep -rn 'fooo = 1' .")
    assert r.matches, "should find the assignment"
    m = next(x for x in r.matches if x.path == "a.py")
    await sh.replace(m, "fooo = 999\n")
    assert "fooo = 999" in (repo / "a.py").read_text()


@pytest.mark.asyncio
async def test_repeated_noncontiguous_search_path_keeps_resolved_path(repo: Path, monkeypatch):
    """A cached path must not inherit the preceding result's resolved path."""

    class FakeSession:
        async def run_with_timeout_flag(self, command, timeout):
            del command, timeout
            records = [
                {
                    "type": "match",
                    "data": {"path": {"text": path}, "line_number": 1},
                }
                for path in ("a.py", "b.txt", "a.py")
            ]
            return "\n".join(json.dumps(record) for record in records), "", 0, False

    async def fake_get_session():
        return FakeSession()

    sh = ShellTools(cwd=str(repo))
    monkeypatch.setattr(sh, "_get_session", fake_get_session)

    matches = await sh._harvest_matches(
        "grep -rn 'match' .",
        "a.py:1:match\nb.txt:1:match\na.py:1:match\n",
    )

    assert matches is not None
    assert [match.path for match in matches] == ["a.py", "b.txt", "a.py"]
    assert [Path(match.resolved_path) for match in matches] == [
        repo / "a.py",
        repo / "b.txt",
        repo / "a.py",
    ]


@pytest.mark.asyncio
async def test_search_match_stays_bound_after_cwd_change(repo: Path):
    sh = ShellTools(cwd=str(repo))
    other = repo / "other"
    other.mkdir()
    (other / "a.py").write_text("fooo = 1\n")

    result = await sh.run("grep -n 'fooo = 1' a.py")
    assert result.matches
    match = result.matches[0]
    await sh.run("cd other")
    await sh.replace(match, "fooo = 999\n")

    assert "fooo = 999" in (repo / "a.py").read_text()
    assert (other / "a.py").read_text() == "fooo = 1\n"


# --------------------------------------------------------------------------
# Fail-closed: non-search / mangled commands attach nothing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "grep -rno 'foo' .",  # -o: no full line anchor
        "grep -rnA1 'foo' .",  # context lines
        "cat a.py",  # not a search
        "grep -rnP 'fo+' .",  # PCRE
    ],
)
@pytest.mark.asyncio
async def test_fail_closed(repo: Path, cmd: str):
    sh = ShellTools(cwd=str(repo))
    r = await sh.run(cmd)
    assert r.matches is None


@pytest.mark.asyncio
async def test_grep_without_n_is_unverifiable(repo: Path):
    """grep without -n prints no line numbers -> can't verify -> attach nothing."""
    sh = ShellTools(cwd=str(repo))
    r = await sh.run("grep -r 'foo' .")
    assert r.matches is None


@pytest.mark.asyncio
async def test_truncated_head_attaches_displayed_subset(repo: Path):
    """A safe ``| head -N`` truncates the *display* to a prefix; the shown lines
    are still real matches, so attach exactly those (subset, not the full set).
    """
    sh = ShellTools(cwd=str(repo))
    full = await sh.run("grep -rn 'foo' .")
    assert full.matches is not None and len(full.matches) > 1

    r = await sh.run("grep -rn 'foo' . | head -1")
    assert r.matches is not None, "truncated head should still attach displayed matches"
    shown = {(m.path, m.start) for m in r.matches}
    displayed = set()
    for line in r.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            p = parts[0][2:] if parts[0].startswith("./") else parts[0]
            displayed.add((p, int(parts[1])))
    assert shown == displayed
    assert shown <= {(m.path, m.start) for m in full.matches}


@pytest.mark.asyncio
async def test_truncated_head_match_is_editable(repo: Path):
    """A Match from a head-truncated grep edits at the correct anchor."""
    sh = ShellTools(cwd=str(repo))
    r = await sh.run("grep -rn 'fooo = 1' . | head -5")
    assert r.matches, "should attach the assignment despite the head pipe"
    m = next(x for x in r.matches if x.path == "a.py")
    await sh.replace(m, "fooo = 777\n")
    assert "fooo = 777" in (repo / "a.py").read_text()


@pytest.mark.parametrize(
    "cmd",
    [
        "grep -n 'foo' a.py",  # single explicit file: grep omits the path
        "grep -rn 'foo' a.py",  # -r with one file still prints "line:" only
    ],
)
@pytest.mark.asyncio
async def test_single_file_grep_attaches_matches(repo: Path, cmd: str):
    """A grep targeting one explicit file omits the filename ("line:content"),
    but the path is known a priori, so matches must still attach and be correct.
    """
    sh = ShellTools(cwd=str(repo))
    r = await sh.run(cmd)
    truth = _grep_anchor_lines(repo, cmd.replace(" a.py", " --with-filename a.py"))
    assert r.matches is not None, "single-file pure search should attach matches"
    got = {(m.path, m.start) for m in r.matches}
    assert got == truth
    # every reported line belongs to the one file we searched
    assert {m.path for m in r.matches} == {"a.py"}


@pytest.mark.asyncio
async def test_single_file_grep_match_is_editable(repo: Path):
    """A Match from a single-file grep can be passed to replace() to edit in place."""
    sh = ShellTools(cwd=str(repo))
    r = await sh.run("grep -n 'fooo = 1' a.py")
    assert r.matches, "single-file grep should find the assignment"
    await sh.replace(r.matches[0], "fooo = 999\n")
    assert "fooo = 999" in (repo / "a.py").read_text()


# --------------------------------------------------------------------------
# Property / fuzz
# --------------------------------------------------------------------------
def _random_corpus(tmp: Path, seed: int) -> Path:
    rng = random.Random(seed)
    root = tmp / f"c{seed}"
    root.mkdir()
    alphabet = string.ascii_letters + "  \t:#.()[]" + "λ"
    for i in range(rng.randint(1, 4)):
        nlines = rng.randint(0, 6)
        lines = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30))) for _ in range(nlines)
        ]
        (root / f"f{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return root


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.asyncio
async def test_property_oracle(tmp_path: Path, seed: int):
    root = _random_corpus(tmp_path, seed)
    rng = random.Random(seed * 7919)
    pat = rng.choice(["a", "b", "abc", "x", ":", "la", "f"])
    cmd = f"grep -rn -F {pat!r} ."
    sh = ShellTools(cwd=str(root))
    r = await sh.run(cmd)
    truth = _grep_anchor_lines(root, cmd)
    # invariant: either matches is None (gate/verify declined) OR it equals truth
    if r.matches is not None:
        assert {(m.path, m.start) for m in r.matches} == truth


# --------------------------------------------------------------------------
# Data-driven: real grep/rg patterns scraped from the SWE-bench run.
# These are the ACTUAL search inputs the model issued (139 distinct patterns;
# a representative slice spanning plain literals, regex dots/stars, and
# alternations). For each, against a synthetic corpus, the differential oracle
# must hold: either matches is None (gate/verify declined) or it equals what
# grep -n itself reported.
# --------------------------------------------------------------------------
REAL_SCRAPED_PATTERNS = [
    "Cycle",
    "FILE_UPLOAD_PERMISSION",
    "FILE_UPLOAD_PERMISSIONS",
    "GenericForeignKey",
    "InlineModelAdmin",
    "Max",
    "RenameContentType",
    "TIME_ZONE",
    "URLValidator",
    "UUIDField",
    "ValueError",
    "W0611",
    "_chain",
    "_eval_evalf",
    "_imp_",
    "COUNT.*DISTINCT",
    "Count.*Case",
    "Count.*Case.*distinct",
    "\\.evalf\\(",
    "connection.timezone",
    "def test.*hist",
    "def test.*model_to_dict",
    "def test.*union",
    "empty.*fields",
    "hist.*range.*density",
    "hist_kwargs.*=.*dict",
    "inspect.isclass",
    "integrate.*dim",
    "integrate.*dim=",
    "method.*integrate",
    "^def method|^    def method",
    "^import|^from.*import",
    "combined_queries|combinator",
    "cotm|cothm",
    "def hstack|def vstack",
    "def set_xlim|def set_ylim",
    "def set_ylim|def set_xlim",
    "empty.*name|name.*empty",
    "hstack|vstack",
    "prepend|append",
    "set_xlim|set_ylim",
    "set_ylim|set_xlim",
    "set_ylim|set_xlim|invert",
    "test.*delete|test_delete",
    "timezone_name|timezone",
]


@pytest.fixture
def code_repo(tmp_path: Path) -> Path:
    """A corpus seeded so the real patterns actually hit something."""
    (tmp_path / "models.py").write_text(
        "class URLValidator:\n"
        "    pass\n"
        "def _chain(self):\n"
        "    return self\n"
        "COUNT = 1  # COUNT DISTINCT\n"
        "FILE_UPLOAD_PERMISSIONS = 0o644\n"
        "import os\n"
        "from django.db import connection\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_union(self):\n"
        "    pass\n"
        "def test_delete(self):\n"
        "    pass\n"
        "def set_xlim(self):\n"
        "    pass\n"
        "def set_ylim(self):\n"
        "    pass\n"
    )
    return tmp_path


@pytest.mark.parametrize("pattern", REAL_SCRAPED_PATTERNS)
@pytest.mark.asyncio
async def test_scraped_patterns_oracle(code_repo: Path, pattern: str):
    """Differential oracle on real model-issued grep patterns."""
    cmd = f"grep -rn {pattern!r} ."
    if not is_pure_search_command(cmd):
        pytest.skip("gate declined (not a pure search)")
    sh = ShellTools(cwd=str(code_repo))
    r = await sh.run(cmd)
    truth = _grep_anchor_lines(code_repo, cmd)
    # Invariant: matches is None OR equals grep's own reported anchors.
    if r.matches is not None:
        assert {(m.path, m.start) for m in r.matches} == truth
    else:
        # If grep found anchors but we attached none, that is allowed only when
        # the gate/verify declined — but these are all pure searches with -n, so
        # for non-empty results we expect matches to be populated and correct.
        assert truth == set() or r.matches is not None


@pytest.mark.parametrize("pattern", [p for p in REAL_SCRAPED_PATTERNS if "|" in p])
@pytest.mark.asyncio
async def test_scraped_alternation_patterns(code_repo: Path, pattern: str):
    """Alternation patterns (the | is in the quoted pattern, NOT a shell pipe)."""
    cmd = f"grep -rn {pattern!r} ."
    # the gate must NOT mistake the in-pattern | for a shell pipe
    assert is_pure_search_command(cmd), "quoted | must not trip the pipe gate"
    sh = ShellTools(cwd=str(code_repo))
    r = await sh.run(cmd)
    truth = _grep_anchor_lines(code_repo, cmd)
    if r.matches is not None:
        assert {(m.path, m.start) for m in r.matches} == truth


# --- Tests for is_pure_search_command with real-world patterns ---


class TestIsPureSearchCommand:
    """Test is_pure_search_command handles real agent grep patterns."""

    def test_bare_grep(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -rn "pattern" src/') is True

    def test_grep_single_file(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -n "foo" bar.py') is True

    def test_cd_prefix_stripped(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert (
            is_pure_search_command('cd /testbed && grep -n "sympy_integers" sympy/printing/str.py')
            is True
        )

    def test_cd_prefix_with_rn(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert (
            is_pure_search_command('cd /testbed && grep -rn "max_iter" sklearn/decomposition/')
            is True
        )

    def test_pipe_head_allowed(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -rn "pattern" src/ | head -50') is True

    def test_pipe_tail_allowed(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -rn "pattern" src/ | tail -5') is True

    def test_cd_and_pipe_head(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert (
            is_pure_search_command('cd /testbed && grep -rn "max_iter" sklearn/ | head -50') is True
        )

    def test_find_xargs_grep_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert (
            is_pure_search_command('find /testbed -name "*.py" | xargs grep -l "Foo" | head -30')
            is False
        )

    def test_pipe_sort_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -rn "pattern" src/ | sort') is False

    def test_pipe_awk_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -rn "pattern" src/ | awk "{print $1}"') is False

    def test_only_matching_flag_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -orn "pattern" src/') is False

    def test_count_flag_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -crn "pattern" src/') is False

    def test_context_flag_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('grep -A2 "pattern" src/') is False

    def test_echo_pipe_grep_rejected(self):
        from nooa.tools.shell_tools import is_pure_search_command

        assert is_pure_search_command('echo "hello" | grep "h"') is False


@pytest.mark.asyncio
async def test_sed_range_in_a_symlinked_cwd_still_attaches_a_match(repo: Path, tmp_path: Path):
    """A symlinked cwd must not make a read-only sed raise out of run().

    ``self.cwd`` comes from bash's *logical* pwd, so after ``cd`` through a
    symlink it reads ``/tmp/x`` while the file resolves to ``/private/tmp/x``.
    Relating the two without a guard raises ValueError from inside run().
    """
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)

    sh = ShellTools(cwd=str(repo))
    await sh.run(f"cd {link}")
    result = await sh.run("sed -n '2,3p' a.py")

    assert result.matches is not None
    assert result.matches[0].text == '    return "bar"\n# foo again\n'
    # Relative to the resolved cwd. Relating an unresolved logical cwd to the
    # resolved file either raises or degrades this to an absolute path.
    assert result.matches[0].path == "a.py"
    await sh.close()


@pytest.mark.asyncio
async def test_sed_range_cwd_is_captured_with_the_command(repo: Path):
    """A concurrent ``cd`` must not be misattributed to the sed command.

    Ordering between two gathered run() calls is genuinely undefined -- sed may
    legitimately run before or after ``cd sub``. What must hold either way is
    that the harvest uses the cwd *that sed ran in*, so the anchor describes the
    file sed actually read.

    run() gets that by reading the cwd under the same session lock as the
    command. A separate ``pwd`` round-trip can instead observe the other task's
    ``cd``, harvest against the wrong directory, and lose the anchor entirely.
    """
    sh = ShellTools(cwd=str(repo))
    (repo / "sub" / "a.py").write_text("DECOY\nDECOY2\n")
    # Start the session first: otherwise lazy startup yields and `cd sub` wins
    # the lock every time, so sed and the cwd never disagree and the race the
    # test is about never happens.
    await sh.run("true")

    sed_result, _ = await asyncio.gather(
        sh.run("sed -n '1,1p' a.py"),
        sh.run("cd sub"),
    )

    # Whichever a.py sed reached, an anchor must be attached and must agree
    # with what sed printed.
    assert sed_result.stdout.strip() in ("def foo():", "DECOY")
    assert sed_result.matches is not None and len(sed_result.matches) == 1
    assert sed_result.matches[0].text.strip() == sed_result.stdout.strip()
    await sh.close()


@pytest.mark.asyncio
async def test_sed_range_match_stays_bound_to_its_file_after_a_cd(repo: Path):
    """A sed Match must keep pointing at the file sed read, even after a cd."""
    (repo / "sub" / "a.py").write_text("DECOY\nDECOY2\n")

    sh = ShellTools(cwd=str(repo))
    result = await sh.run("sed -n '1,1p' a.py")
    assert result.matches
    await sh.run("cd sub")
    await sh.replace(result.matches[0], "PATCHED\n")

    assert (repo / "a.py").read_text().startswith("PATCHED\n")
    assert (repo / "sub" / "a.py").read_text() == "DECOY\nDECOY2\n"
    await sh.close()


@pytest.mark.asyncio
async def test_sed_range_anchor_excludes_blank_lines_it_never_showed(repo: Path):
    """The anchor must cover what was displayed, not the whole requested range.

    Command output is stripped, so blank lines at either edge of the range never
    reach the screen. Anchoring the full range would let replace() delete lines
    the agent never saw.
    """
    target = repo / "x.py"
    target.write_text("import os\n\n\ndef go():\n    pass\n\n\n")

    sh = ShellTools(cwd=str(repo))
    result = await sh.run("sed -n '2,7p' x.py")

    assert result.matches is not None and len(result.matches) == 1
    match = result.matches[0]
    # Requested 2-7; lines 2, 3, 6 and 7 are blank and were never shown.
    assert (match.start, match.end) == (4, 5)
    assert match.text == "def go():\n    pass\n"

    await sh.replace(match, "def go():\n    return 1\n")
    assert target.read_text() == "import os\n\n\ndef go():\n    return 1\n\n\n"
    await sh.close()


@pytest.mark.asyncio
async def test_sed_range_of_only_blank_lines_anchors_nothing(repo: Path):
    """A range that printed nothing visible has nothing to anchor."""
    (repo / "y.py").write_text("a = 1\n\n\n\nb = 2\n")

    sh = ShellTools(cwd=str(repo))
    result = await sh.run("sed -n '2,4p' y.py")

    assert result.stdout.strip() == ""
    assert result.matches == []
    await sh.close()
