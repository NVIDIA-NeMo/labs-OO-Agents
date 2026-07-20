# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Back-compat tests: verify grep/find produce equivalent results with and without ripgrep."""

import shutil

import pytest

from nooa.tools.shell_tools_legacy import ShellToolsLegacy as ShellTools

has_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")


@pytest.fixture
async def shell(tmp_path):
    """Create a ShellTools instance in a temp directory."""
    s = ShellTools(cwd=tmp_path)
    yield s
    await s.close()


@pytest.fixture
def sample_tree(tmp_path):
    """Create a sample directory tree for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text('def hello():\n    print("hello world")\n')
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    pass\n")
    (tmp_path / "src" / "sub").mkdir()
    (tmp_path / "src" / "sub" / "deep.py").write_text("x = 1\nprint(x)\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_it():\n    pass\n")
    (tmp_path / "README.md").write_text("# Hello\n")
    (tmp_path / "data.txt").write_text("price is $10.00\nno match here\n")
    # Create __pycache__ that should be ignored
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_text("cached")
    return tmp_path


async def _grep_with_rg(shell, *args, **kwargs):
    """Run grep forcing ripgrep path."""
    shell._rg_available = True
    try:
        return await shell.grep(*args, **kwargs)
    finally:
        shell._rg_available = None


async def _grep_without_rg(shell, *args, **kwargs):
    """Run grep forcing fallback path."""
    shell._rg_available = False
    try:
        return await shell.grep(*args, **kwargs)
    finally:
        shell._rg_available = None


async def _find_with_rg(shell, *args, **kwargs):
    """Run find forcing ripgrep path."""
    shell._rg_available = True
    try:
        return await shell.find(*args, **kwargs)
    finally:
        shell._rg_available = None


async def _find_without_rg(shell, *args, **kwargs):
    """Run find forcing fallback path."""
    shell._rg_available = False
    try:
        return await shell.find(*args, **kwargs)
    finally:
        shell._rg_available = None


def _normalize_matches(matches):
    """Normalize match lines for comparison (sort, strip leading ./)."""
    normalized = []
    for m in matches:
        # Remove leading ./ that find might add
        if m.startswith("./"):
            m = m[2:]
        normalized.append(m)
    return sorted(normalized)


# ==========================================================================
# grep back-compat tests
# ==========================================================================
@has_rg
class TestGrepBackCompat:
    """Verify grep produces equivalent results with and without ripgrep."""

    async def test_basic_pattern(self, shell, sample_tree):
        rg_result = await _grep_with_rg(shell, "def", str(sample_tree / "src"))
        fallback_result = await _grep_without_rg(shell, "def", str(sample_tree / "src"))

        assert rg_result.total_matches == fallback_result.total_matches
        assert _normalize_matches(rg_result.matches) == _normalize_matches(fallback_result.matches)

    async def test_no_matches(self, shell, sample_tree):
        rg_result = await _grep_with_rg(shell, "zzz_nonexistent_zzz", str(sample_tree))
        fallback_result = await _grep_without_rg(shell, "zzz_nonexistent_zzz", str(sample_tree))

        assert rg_result.total_matches == 0
        assert fallback_result.total_matches == 0

    async def test_literal_special_chars(self, shell, sample_tree):
        rg_result = await _grep_with_rg(shell, "$10.00", str(sample_tree), literal=True)
        fallback_result = await _grep_without_rg(shell, "$10.00", str(sample_tree), literal=True)

        assert rg_result.total_matches == fallback_result.total_matches
        assert rg_result.total_matches > 0

    async def test_include_glob(self, shell, sample_tree):
        rg_result = await _grep_with_rg(shell, ".", str(sample_tree), include="*.md")
        fallback_result = await _grep_without_rg(shell, ".", str(sample_tree), include="*.md")

        assert rg_result.total_matches == fallback_result.total_matches
        assert rg_result.total_matches > 0

    async def test_context_lines(self, shell, sample_tree):
        rg_result = await _grep_with_rg(shell, "hello", str(sample_tree / "src"), context=1)
        fallback_result = await _grep_without_rg(
            shell, "hello", str(sample_tree / "src"), context=1
        )

        # Both should find matches and include context
        assert rg_result.total_matches == fallback_result.total_matches
        assert rg_result.total_matches > 0

    async def test_max_matches(self, shell, sample_tree):
        rg_result = await _grep_with_rg(shell, ".", str(sample_tree / "src"), max_matches=2)
        fallback_result = await _grep_without_rg(
            shell, ".", str(sample_tree / "src"), max_matches=2
        )

        # -m is per-file for both rg and grep, so total may exceed max_matches
        # but both should produce the same total
        assert rg_result.total_matches == fallback_result.total_matches
        assert rg_result.truncated == fallback_result.truncated


# ==========================================================================
# find back-compat tests
# ==========================================================================
@has_rg
class TestFindBackCompat:
    """Verify find produces equivalent results with and without ripgrep."""

    async def test_find_py_files(self, shell, sample_tree):
        # Need git init for rg to work properly with --files
        await shell.run(f"cd {sample_tree} && git init -q && git add .")
        await shell.run(f"cd {sample_tree}")

        rg_result = await _find_with_rg(shell, "*.py", ".")
        fallback_result = await _find_without_rg(shell, "*.py", ".")

        rg_files = _normalize_matches(rg_result.matches)
        fallback_files = _normalize_matches(fallback_result.matches)

        # Both should find the same .py files
        assert set(rg_files) == set(fallback_files)
        assert rg_result.total_matches > 0

    async def test_find_no_matches(self, shell, sample_tree):
        await shell.run(f"cd {sample_tree} && git init -q && git add .")
        await shell.run(f"cd {sample_tree}")

        rg_result = await _find_with_rg(shell, "*.xyz", ".")
        fallback_result = await _find_without_rg(shell, "*.xyz", ".")

        assert rg_result.total_matches == 0
        assert fallback_result.total_matches == 0

    async def test_find_specific_pattern(self, shell, sample_tree):
        await shell.run(f"cd {sample_tree} && git init -q && git add .")
        await shell.run(f"cd {sample_tree}")

        rg_result = await _find_with_rg(shell, "test_*.py", ".")
        fallback_result = await _find_without_rg(shell, "test_*.py", ".")

        rg_files = _normalize_matches(rg_result.matches)
        fallback_files = _normalize_matches(fallback_result.matches)

        assert set(rg_files) == set(fallback_files)
        assert any("test_main.py" in f for f in rg_files)

    async def test_find_ignores_pycache(self, shell, sample_tree):
        """Fallback find (non-git path) should skip __pycache__ via _IGNORE_DIRS pruning."""
        # rg --files respects .gitignore (covered by test_find_respects_gitignore in
        # test_shell_tools.py). Here we verify the non-git fallback path skips __pycache__.
        fallback_result = await _find_without_rg(shell, "*.pyc", str(sample_tree))

        for m in fallback_result.matches:
            assert "__pycache__" not in m

    async def test_find_directories(self, shell, sample_tree):
        """type='d' should work the same regardless of rg availability (always uses find)."""
        rg_result = await _find_with_rg(shell, "sub", str(sample_tree), entry_type="d")
        fallback_result = await _find_without_rg(shell, "sub", str(sample_tree), entry_type="d")

        assert rg_result.total_matches == fallback_result.total_matches
        assert rg_result.total_matches > 0


# ==========================================================================
# Fallback-only tests (verify fallback works in isolation)
# ==========================================================================
class TestFallbackOnly:
    """Test that the fallback grep/find paths work correctly on their own."""

    async def test_fallback_grep_basic(self, shell, sample_tree):
        """Fallback grep finds expected matches (non-git path)."""
        result = await _grep_without_rg(shell, "hello", str(sample_tree / "src"))
        assert result.total_matches > 0
        assert any("hello" in m for m in result.matches)

    async def test_fallback_grep_returns_file_line_format(self, shell, sample_tree):
        """Fallback grep output has file:line:text format (non-git path)."""
        result = await _grep_without_rg(shell, "def hello", str(sample_tree / "src"))
        assert result.total_matches > 0
        # Each match should contain filename:linenum:content
        for m in result.matches:
            parts = m.split(":")
            assert len(parts) >= 3  # file:num:text

    async def test_fallback_find_basic(self, shell, sample_tree):
        """Fallback find locates files by glob (non-git path)."""
        shell._rg_available = False
        result = await shell.find("*.py", str(sample_tree))
        shell._rg_available = None
        assert result.total_matches > 0
        assert any("main.py" in m for m in result.matches)

    async def test_has_rg_caches(self, shell):
        """_has_rg caches its result after first call."""
        # Force a check
        shell._rg_available = None
        result1 = await shell._has_rg()
        # The cached value should persist
        assert shell._rg_available is not None
        result2 = await shell._has_rg()
        assert result1 == result2


# ==========================================================================
# Binary file and hidden directory exclusion tests
# ==========================================================================
class TestFallbackExclusions:
    """Verify fallback grep skips binary files and noisy directories."""

    async def test_fallback_grep_skips_binary_files(self, shell, sample_tree):
        """Fallback grep should not match content inside binary files."""
        # Create a binary file containing a searchable string
        binary_path = sample_tree / "data.bin"
        binary_path.write_bytes(b"\x00\x01\x02FINDME\x03\x04\x05")
        # Also create a text file with the same string
        text_path = sample_tree / "findme.txt"
        text_path.write_text("FINDME in text\n")

        result = await _grep_without_rg(shell, "FINDME", str(sample_tree))

        # Should find the text file but NOT the binary file
        assert result.total_matches > 0
        assert any("findme.txt" in m for m in result.matches)
        assert not any("data.bin" in m for m in result.matches)

    async def test_fallback_grep_skips_git_dir(self, shell, sample_tree):
        """Fallback grep (non-git path) should not search inside .git directories."""
        # Create a .git directory with searchable content
        git_dir = sample_tree / ".git" / "objects"
        git_dir.mkdir(parents=True)
        (git_dir / "pack-info").write_text("SECRETGIT content\n")
        # Also create a normal file with different content
        (sample_tree / "normal.txt").write_text("SECRETGIT visible\n")

        result = await _grep_without_rg(shell, "SECRETGIT", str(sample_tree))

        # Should find normal.txt but not .git contents
        assert result.total_matches > 0
        assert any("normal.txt" in m for m in result.matches)
        assert not any(".git" in m for m in result.matches)

    async def test_fallback_grep_skips_pycache(self, shell, sample_tree):
        """Fallback grep (non-git path) should not search inside __pycache__ directories."""
        # __pycache__ already exists in sample_tree fixture
        (sample_tree / "__pycache__" / "module.cpython-313.pyc").write_text("CACHEDSTR\n")
        (sample_tree / "visible.txt").write_text("CACHEDSTR here too\n")

        result = await _grep_without_rg(shell, "CACHEDSTR", str(sample_tree))

        assert result.total_matches > 0
        assert any("visible.txt" in m for m in result.matches)
        assert not any("__pycache__" in m for m in result.matches)

    @has_rg
    async def test_binary_skipping_matches_rg(self, shell, sample_tree):
        """Both rg and fallback should skip binary files consistently."""
        binary_path = sample_tree / "binary.dat"
        binary_path.write_bytes(b"\x00\x01MATCHME\x02\x03")
        (sample_tree / "text.txt").write_text("MATCHME in text\n")

        rg_result = await _grep_with_rg(shell, "MATCHME", str(sample_tree))
        fallback_result = await _grep_without_rg(shell, "MATCHME", str(sample_tree))

        # Both should find text.txt, neither should find binary.dat
        assert any("text.txt" in m for m in rg_result.matches)
        assert any("text.txt" in m for m in fallback_result.matches)
        assert not any("binary.dat" in m for m in rg_result.matches)
        assert not any("binary.dat" in m for m in fallback_result.matches)

    @has_rg
    async def test_hidden_dir_skipping_matches_rg(self, shell, sample_tree):
        """Both rg and fallback (non-git path) should skip .git directory consistently."""
        git_dir = sample_tree / ".git" / "refs"
        git_dir.mkdir(parents=True)
        (git_dir / "heads").write_text("HIDDENREF\n")
        (sample_tree / "visible.txt").write_text("HIDDENREF\n")

        rg_result = await _grep_with_rg(shell, "HIDDENREF", str(sample_tree))
        fallback_result = await _grep_without_rg(shell, "HIDDENREF", str(sample_tree))

        # Both should find visible.txt, neither should find .git contents
        assert any("visible.txt" in m for m in rg_result.matches)
        assert any("visible.txt" in m for m in fallback_result.matches)
        assert not any(".git" in m for m in rg_result.matches)
        assert not any(".git" in m for m in fallback_result.matches)
